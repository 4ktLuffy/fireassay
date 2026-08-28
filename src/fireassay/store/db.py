"""SQLite-backed store for fireassay: questions, suites, configs, runs,
results and scores.

Runs are append-only: once a run is sealed (``finish_run`` has been called,
so its status is ``'complete'`` or ``'failed'``) any further write touching
that run_id is rejected. This is enforced twice — once in Python (raising
``RunSealedError``, which gives a clear, catchable error) and again with
SQL triggers directly on the ``result``/``score`` tables (which protect the
invariant even against a caller that goes around the ``Store`` class, e.g.
a stray script opening the database file directly). Both status values,
``'complete'`` and ``'failed'``, are treated as terminal: a run that failed
partway through must not be quietly topped up later either, since a
leaderboard consumer may already have looked at its (partial) numbers.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fireassay.hashing import config_hash as _config_hash
from fireassay.hashing import suite_hash as _suite_hash
from fireassay.models import (
    Config,
    EvidenceSpan,
    Question,
    RetrievedChunk,
    Run,
    Score,
    Suite,
    SystemOutput,
)
from fireassay.score.invariants import InvariantViolation

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


@dataclass(frozen=True)
class ControlCheckRow:
    """One persisted `control_check` row, read back from the store.

    A plain dataclass (not `controls.base.ControlOutcome`) deliberately:
    `store/db.py` must not import `fireassay.controls` — every control
    module imports `Store`, so the reverse import would be a cycle. Callers
    that want a `ControlOutcome` (e.g. a future renderer) build one from
    these fields themselves.
    """

    kind: str
    status: str
    observed: dict[str, float]
    expected: dict[str, object]
    twin_ok: bool
    cause_assertions: dict[str, bool]
    detail: str
    checked_at: str


@dataclass(frozen=True)
class MutationRunRow:
    id: str
    suite_id: str
    suite_hash: str
    base_config_id: str
    detector: str
    killed: int
    total: int
    equivalent: int
    score: float
    created_at: str


@dataclass(frozen=True)
class MutantRow:
    id: str
    mutation_run_id: str
    operator: str
    params: dict[str, object]
    mutant_run_id: str
    killed: bool
    equivalent: bool
    equivalent_reason: str | None
    detail: str


class AdmissibilityAlreadySetError(Exception):
    """Raised by `set_admissibility` when a run already has a recorded
    admissibility verdict that **conflicts** with the one just computed.

    A byte-identical re-assessment (same `admissible`, same canonical
    `admissibility_json`) is *not* an error — `set_admissibility` is
    idempotent for that case, since admissibility is a derived verdict and
    recomputing it (e.g. a second `controls run` against a matrix whose
    run was reused) is legitimate. This error is specifically for a
    *different* verdict landing on a run that already has one: mirrors the
    append-only philosophy that governs `result`/`score` — an admissibility
    verdict must not silently change underneath a reader who already
    looked at it. Recompute under a fresh run if the controls or expected
    bands genuinely changed.
    """


class RunSealedError(Exception):
    """Raised when a write is attempted against a run that has already been
    finished (status 'complete' or 'failed').

    Runs are append-only once finished — that is what makes a leaderboard
    trustworthy. Without this, a number could be reported, and quietly
    revised afterwards with no trace that it happened.
    """


class SuiteExistsError(Exception):
    """Raised by ``freeze_suite`` when (name, version) already exists with a
    different ``suite_hash``.

    A suite version is a promise about exact question-set content. Silently
    overwriting it would retroactively break every run that already cites
    that (name, version) as its comparison basis, without anyone knowing.
    """


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    """SQLite persistence layer. One `Store` wraps one sqlite3 connection.

    `PRAGMA foreign_keys = ON` and `journal_mode = WAL` are set on
    construction, per M1-SPEC.md §3 — WAL so concurrent readers (e.g. a CLI
    `show` command) do not block a long-running writer, and foreign keys ON
    because SQLite defaults them off and this schema relies on them for
    referential integrity.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._configure_connection(self._conn)

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- schema -----------------------------------------------------------

    def migrate(self) -> None:
        """Apply un-applied migration files from migrations/, in filename
        order, tracking what has been applied in a `schema_migrations`
        table. Idempotent: calling this on an already-migrated database is
        a no-op.
        """
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "filename TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        self._conn.commit()
        applied = {
            row["filename"] for row in self._conn.execute("SELECT filename FROM schema_migrations")
        }
        for migration_path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            if migration_path.name in applied:
                continue
            sql = migration_path.read_text(encoding="utf-8")
            self._conn.executescript(sql)
            self._conn.execute(
                "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, ?)",
                (migration_path.name, _now()),
            )
            self._conn.commit()

    # -- writes -------------------------------------------------------------

    def put_questions(self, questions: Iterable[Question]) -> int:
        """Insert questions, keyed by their content-addressed id.

        Idempotent: inserting a question whose id already exists is a
        silent no-op. Because `Question.id` is a pure hash of its content
        (see hashing.question_id), an existing id necessarily means
        byte-identical content, so re-importing an overlapping question set
        is always safe to repeat.

        Returns the number of *newly inserted* questions.
        """
        inserted = 0
        for q in questions:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO question "
                "(id, text, qtype, difficulty, reference_answer, provenance, "
                "generator, source_doc_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    q.id,
                    q.text,
                    q.qtype,
                    q.difficulty,
                    q.reference_answer,
                    q.provenance,
                    q.generator,
                    q.source_doc_id,
                    _now(),
                ),
            )
            if cur.rowcount:
                inserted += 1
                for span in q.evidence_spans:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO evidence_span "
                        "(question_id, doc_id, chunk_id, page, char_start, char_end, quote) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            q.id,
                            span.doc_id,
                            span.chunk_id,
                            span.page,
                            span.char_start,
                            span.char_end,
                            span.quote,
                        ),
                    )
        self._conn.commit()
        return inserted

    def freeze_suite(self, name: str, version: str, question_ids: Sequence[str]) -> Suite:
        """Freeze a named, versioned suite over an exact set of question ids.

        `question_ids` is de-duplicated before hashing and storing: a suite
        is a *set* of questions, so passing the same id twice must not
        change `suite_hash` or `question_count`.

        Idempotent on (name, version): re-freezing with a question set that
        hashes the same returns the existing row rather than erroring, so a
        pipeline can call `freeze_suite` unconditionally on every run.
        Re-freezing the same (name, version) with a set that hashes
        *differently* raises `SuiteExistsError` instead of silently
        changing what that version means underneath every run that already
        cites it.
        """
        unique_ids = sorted(set(question_ids))
        h = _suite_hash(unique_ids)
        row = self._conn.execute(
            "SELECT id, name, version, suite_hash, frozen_at, question_count "
            "FROM suite WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if row is not None:
            if row["suite_hash"] != h:
                raise SuiteExistsError(
                    f"suite {name}@{version} already exists with a different "
                    f"suite_hash ({row['suite_hash']} != {h})"
                )
            return Suite(
                id=row["id"],
                name=row["name"],
                version=row["version"],
                suite_hash=row["suite_hash"],
                frozen_at=row["frozen_at"],
                question_count=row["question_count"],
            )
        suite_id = h
        frozen_at = _now()
        question_count = len(unique_ids)
        self._conn.execute(
            "INSERT INTO suite (id, name, version, suite_hash, frozen_at, question_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (suite_id, name, version, h, frozen_at, question_count),
        )
        for qid in unique_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO suite_question (suite_id, question_id) VALUES (?, ?)",
                (suite_id, qid),
            )
        self._conn.commit()
        return Suite(
            id=suite_id,
            name=name,
            version=version,
            suite_hash=h,
            frozen_at=frozen_at,
            question_count=question_count,
        )

    def put_config(self, spec: Mapping[str, object], label: str | None = None) -> Config:
        """Insert a config keyed by its content hash.

        Idempotent on `config_hash`: calling this twice with an
        equivalent spec returns the existing row (the `label` recorded on
        the *first* call wins). Configs are addressed by content, not by
        label, so two different labels are never allowed to fork what is
        semantically one config.
        """
        spec_dict: dict[str, object] = dict(spec)
        h = _config_hash(spec_dict)
        row = self._conn.execute(
            "SELECT id, config_hash, label, spec_json FROM config WHERE config_hash = ?",
            (h,),
        ).fetchone()
        if row is not None:
            return Config(
                id=row["id"],
                config_hash=row["config_hash"],
                label=row["label"],
                spec=json.loads(row["spec_json"]),
            )
        config_id = h
        spec_json = json.dumps(spec_dict, sort_keys=True)
        self._conn.execute(
            "INSERT INTO config (id, config_hash, label, spec_json) VALUES (?, ?, ?, ?)",
            (config_id, h, label, spec_json),
        )
        self._conn.commit()
        return Config(id=config_id, config_hash=h, label=label, spec=spec_dict)

    def get_config(self, config_id: str) -> Config:
        """Look up a config by id (its `config_hash`).

        Added in M2 for `fireassay mutate --config <config_id>`
        (M2-SPEC.md §9), which is handed a config id that was already
        stored by an earlier `fireassay run`/`controls run`, rather than a
        fresh spec to insert."""
        row = self._conn.execute(
            "SELECT id, config_hash, label, spec_json FROM config WHERE id = ?", (config_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such config: {config_id}")
        return Config(
            id=row["id"],
            config_hash=row["config_hash"],
            label=row["label"],
            spec=json.loads(row["spec_json"]),
        )

    def start_run(self, suite: Suite, config: Config, env: Mapping[str, object]) -> Run:
        """Start a new run against a frozen suite and a config.

        Returns immediately with status 'running'. The run is not sealed —
        and so not eligible for comparison via integrity.assert_comparable,
        which requires status == 'complete' — until `finish_run` is called.
        """
        run_id = uuid.uuid4().hex
        started_at = _now()
        env_dict: dict[str, object] = dict(env)
        env_json = json.dumps(env_dict, sort_keys=True)
        self._conn.execute(
            "INSERT INTO run (id, suite_id, suite_hash, config_id, config_hash, "
            "env_json, started_at, finished_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'running')",
            (run_id, suite.id, suite.suite_hash, config.id, config.config_hash, env_json, started_at),
        )
        self._conn.commit()
        return Run(
            id=run_id,
            suite_id=suite.id,
            suite_hash=suite.suite_hash,
            config_id=config.id,
            config_hash=config.config_hash,
            env_json=env_dict,
            started_at=started_at,
            finished_at=None,
            status="running",
            result_count=0,
        )

    def put_result(self, run_id: str, question_id: str, output: SystemOutput, cost_usd: float) -> None:
        """Persist a system output for one question in one run.

        Raises `RunSealedError` if the run has already been finished. The
        SQL trigger on `result` enforces the same rule at the database
        level; this Python-level check exists to fail fast with a clear
        exception rather than surfacing a raw `sqlite3.IntegrityError`.
        """
        self._assert_run_open(run_id)
        retrieved_json = json.dumps([rc.model_dump() for rc in output.retrieved], sort_keys=True)
        latency_json = json.dumps(output.latency_ms, sort_keys=True)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO result "
                "(run_id, question_id, answer, abstained, retrieved_json, "
                "latency_json, tokens_in, tokens_out, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    question_id,
                    output.answer,
                    int(output.abstained),
                    retrieved_json,
                    latency_json,
                    output.tokens_in,
                    output.tokens_out,
                    cost_usd,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "sealed" in str(exc):
                raise RunSealedError(f"run {run_id} is sealed") from exc
            raise
        self._conn.commit()

    def put_scores(self, run_id: str, question_id: str, scores: Sequence[Score]) -> None:
        """Persist scores for one question in one run. See `put_result` for
        the sealing behaviour."""
        self._assert_run_open(run_id)
        try:
            for s in scores:
                self._conn.execute(
                    "INSERT OR REPLACE INTO score "
                    "(run_id, question_id, metric, value, scorer, rationale) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, question_id, s.metric, s.value, s.scorer, s.rationale),
                )
        except sqlite3.IntegrityError as exc:
            if "sealed" in str(exc):
                raise RunSealedError(f"run {run_id} is sealed") from exc
            raise
        self._conn.commit()

    def finish_run(
        self,
        run_id: str,
        status: Literal["complete", "failed"],
        *,
        admissible: bool = True,
        invariant_violations: Sequence[InvariantViolation] = (),
    ) -> None:
        """Seal a run. After this call, any write touching `run_id`
        (`put_result`, `put_scores`) raises `RunSealedError`, both here in
        Python and via the SQL triggers.

        `admissible`/`invariant_violations` record the result of
        `score.invariants.check_invariants`, run by the caller (see
        `runner.run_matrix`) over every score produced by this run before
        sealing it. Recorded here rather than computed inside `Store`
        because `Store` has no scorer/invariant-checking knowledge of its
        own — it only persists the verdict.
        """
        row = self._conn.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        if row["status"] != "running":
            raise RunSealedError(f"run {run_id} is already sealed (status={row['status']})")
        violations_json = json.dumps([v.model_dump() for v in invariant_violations], sort_keys=True)
        self._conn.execute(
            "UPDATE run SET status = ?, finished_at = ?, admissible = ?, invariant_violations_json = ? "
            "WHERE id = ?",
            (status, _now(), int(admissible), violations_json, run_id),
        )
        self._conn.commit()

    def _assert_run_open(self, run_id: str) -> None:
        row = self._conn.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        if row["status"] != "running":
            raise RunSealedError(f"run {run_id} is sealed (status={row['status']})")

    # -- reads --------------------------------------------------------------

    def get_run(self, run_id: str) -> Run:
        row = self._conn.execute(
            "SELECT id, suite_id, suite_hash, config_id, config_hash, env_json, "
            "started_at, finished_at, status, admissible, admissibility_json FROM run WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        return Run(
            id=row["id"],
            suite_id=row["suite_id"],
            suite_hash=row["suite_hash"],
            config_id=row["config_id"],
            config_hash=row["config_hash"],
            env_json=json.loads(row["env_json"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            result_count=self.run_result_count(row["id"]),
            admissible=bool(row["admissible"]),
            admissibility_json=json.loads(row["admissibility_json"]),
        )

    def run_retrieved(self, run_id: str) -> dict[str, tuple[RetrievedChunk, ...]]:
        """Return `{question_id: retrieved chunks}` for every result
        persisted in a run, reconstructed from `result.retrieved_json`.

        Added for M2's `controls` module, which needs to verify retrieval
        facts directly from what was actually *persisted* — e.g.
        `no_retrieval`'s `len(retrieved) == 0` cause assertion,
        `null_questions`' overlap check against the suite's judged pool —
        rather than trusting the in-memory `SystemOutput` a control's
        system wrapper happened to produce. Same "prove it was genuinely
        attempted" principle behind the writable-twin requirement
        (M2-SPEC.md §2).
        """
        rows = self._conn.execute(
            "SELECT question_id, retrieved_json FROM result WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {
            r["question_id"]: tuple(RetrievedChunk(**c) for c in json.loads(r["retrieved_json"]))
            for r in rows
        }

    # -- M2: admissibility --------------------------------------------------

    def set_admissibility(
        self, run_id: str, admissible: bool, admissibility_json: Mapping[str, object]
    ) -> None:
        """Record `admissibility.assess`'s verdict onto `run_id`.

        **Idempotent, not one-shot:** admissibility is a *derived* verdict
        (controls + invariant violations -> admissible or not), not run
        data — recomputing it is a legitimate, expected operation, e.g.
        re-running `controls run` a second time against a matrix where
        `run_matrix` reused an already-complete run. Calling this again
        with a verdict that is byte-identical to what is already stored
        (same `admissible`, same canonical `admissibility_json`) is a
        silent no-op. Calling it with a *different* verdict for a run that
        already has one raises `AdmissibilityAlreadySetError` — a run's
        recorded admissibility must not silently change underneath a
        reader who already looked at it; that case genuinely needs a fresh
        run (or investigation into why the same suite/config/controls
        produced two different answers) rather than a second, quietly
        overwritten write.

        This is a deliberate, narrow exception to `result`/`score`'s
        append-only-while-running rule: `run.admissible` was already a
        second write to the `run` row at `finish_run` time in M1 (recording
        the invariant-check verdict, which is only known once the run has
        finished); this is a *third*, later write recording a verdict that
        is only knowable once controls have been run against this run's
        suite/config — which can genuinely happen well after the run
        itself sealed. It does not touch `result`/`score` at all, so the
        run's own measured numbers remain exactly as append-only as M1 made
        them.
        """
        row = self._conn.execute(
            "SELECT admissible, admissibility_json FROM run WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        new_json = json.dumps(dict(admissibility_json), sort_keys=True)
        if row["admissibility_json"] not in ("{}", None):
            if bool(row["admissible"]) == admissible and row["admissibility_json"] == new_json:
                return  # identical re-assessment: idempotent no-op
            raise AdmissibilityAlreadySetError(
                f"run {run_id} already has a CONFLICTING admissibility verdict recorded: "
                f"stored admissible={bool(row['admissible'])!r}, "
                f"admissibility_json={row['admissibility_json']!r}; "
                f"new admissible={admissible!r}, admissibility_json={new_json!r}"
            )
        self._conn.execute(
            "UPDATE run SET admissible = ?, admissibility_json = ? WHERE id = ?",
            (int(admissible), new_json, run_id),
        )
        self._conn.commit()

    # -- M2: control_check ---------------------------------------------------

    def put_control_check(
        self,
        run_id: str,
        kind: str,
        status: str,
        observed: Mapping[str, float],
        expected: Mapping[str, object],
        twin_ok: bool,
        cause_assertions: Mapping[str, bool],
        detail: str,
    ) -> str:
        """Persist one control's `ControlOutcome` against `run_id`.

        Takes plain fields rather than a `controls.base.ControlOutcome`
        object so this module never has to import `fireassay.controls`
        (which imports `Store`) — see `ControlCheckRow`'s docstring.
        `control_check` rows are append-only (immutable once inserted,
        enforced by SQL trigger in migration 0002): this method only ever
        INSERTs, never UPDATEs.
        """
        check_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO control_check (id, run_id, kind, status, observed_json, expected_json, "
            "twin_ok, cause_json, detail, checked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                check_id,
                run_id,
                kind,
                status,
                json.dumps(dict(observed), sort_keys=True),
                json.dumps(dict(expected), sort_keys=True),
                int(twin_ok),
                json.dumps(dict(cause_assertions), sort_keys=True),
                detail,
                _now(),
            ),
        )
        self._conn.commit()
        return check_id

    def get_control_checks(self, run_id: str) -> list[ControlCheckRow]:
        """Return every `control_check` row recorded against `run_id`, in
        the order they were checked."""
        rows = self._conn.execute(
            "SELECT kind, status, observed_json, expected_json, twin_ok, cause_json, detail, checked_at "
            "FROM control_check WHERE run_id = ? ORDER BY checked_at",
            (run_id,),
        ).fetchall()
        return [
            ControlCheckRow(
                kind=r["kind"],
                status=r["status"],
                observed=json.loads(r["observed_json"]),
                expected=json.loads(r["expected_json"]),
                twin_ok=bool(r["twin_ok"]),
                cause_assertions=json.loads(r["cause_json"]),
                detail=r["detail"],
                checked_at=r["checked_at"],
            )
            for r in rows
        ]

    # -- M2: mutation ---------------------------------------------------------

    def put_mutation_run(
        self,
        suite: Suite,
        base_config: Config,
        detector: str,
        killed: int,
        total: int,
        equivalent: int,
        score: float,
    ) -> str:
        """Persist the summary row for one `mutate` invocation. `mutant`
        rows (one per operator) are persisted separately via `put_mutant`,
        referencing this id."""
        mutation_run_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO mutation_run (id, suite_id, suite_hash, base_config_id, detector, "
            "killed, total, equivalent, score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mutation_run_id,
                suite.id,
                suite.suite_hash,
                base_config.id,
                detector,
                killed,
                total,
                equivalent,
                score,
                _now(),
            ),
        )
        self._conn.commit()
        return mutation_run_id

    def put_mutant(
        self,
        mutation_run_id: str,
        operator: str,
        params: Mapping[str, object],
        mutant_run_id: str,
        killed: bool,
        equivalent: bool,
        equivalent_reason: str | None,
        detail: str,
    ) -> str:
        """Persist one mutant's result. `equivalent_reason` MUST be set
        whenever `equivalent` is True (M2-SPEC.md §6: equivalence must be
        computed and justified, never assumed) — enforced by the caller
        (`mutation.run.run_mutation`), not here; this method persists
        whatever it is given."""
        mutant_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO mutant (id, mutation_run_id, operator, params_json, mutant_run_id, "
            "killed, equivalent, equivalent_reason, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mutant_id,
                mutation_run_id,
                operator,
                json.dumps(dict(params), sort_keys=True),
                mutant_run_id,
                int(killed),
                int(equivalent),
                equivalent_reason,
                detail,
            ),
        )
        self._conn.commit()
        return mutant_id

    def get_mutation_run(self, mutation_run_id: str) -> MutationRunRow:
        row = self._conn.execute(
            "SELECT id, suite_id, suite_hash, base_config_id, detector, killed, total, "
            "equivalent, score, created_at FROM mutation_run WHERE id = ?",
            (mutation_run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no such mutation_run: {mutation_run_id}")
        return MutationRunRow(
            id=row["id"],
            suite_id=row["suite_id"],
            suite_hash=row["suite_hash"],
            base_config_id=row["base_config_id"],
            detector=row["detector"],
            killed=row["killed"],
            total=row["total"],
            equivalent=row["equivalent"],
            score=row["score"],
            created_at=row["created_at"],
        )

    def get_mutants(self, mutation_run_id: str) -> list[MutantRow]:
        rows = self._conn.execute(
            "SELECT id, mutation_run_id, operator, params_json, mutant_run_id, killed, "
            "equivalent, equivalent_reason, detail FROM mutant WHERE mutation_run_id = ?",
            (mutation_run_id,),
        ).fetchall()
        return [
            MutantRow(
                id=r["id"],
                mutation_run_id=r["mutation_run_id"],
                operator=r["operator"],
                params=json.loads(r["params_json"]),
                mutant_run_id=r["mutant_run_id"],
                killed=bool(r["killed"]),
                equivalent=bool(r["equivalent"]),
                equivalent_reason=r["equivalent_reason"],
                detail=r["detail"],
            )
            for r in rows
        ]

    def get_run_invariant_violations(self, run_id: str) -> list[InvariantViolation]:
        """Return the invariant violations recorded for a run by
        `finish_run` (empty if the run is still running, or was sealed with
        none)."""
        row = self._conn.execute(
            "SELECT invariant_violations_json FROM run WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        return [InvariantViolation(**v) for v in json.loads(row["invariant_violations_json"])]

    def get_suite(self, name: str, version: str) -> Suite:
        row = self._conn.execute(
            "SELECT id, name, version, suite_hash, frozen_at, question_count "
            "FROM suite WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if row is None:
            raise KeyError(f"no such suite: {name}@{version}")
        return Suite(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            suite_hash=row["suite_hash"],
            frozen_at=row["frozen_at"],
            question_count=row["question_count"],
        )

    def iter_questions(self, suite_id: str) -> Iterator[Question]:
        """Yield every question in a suite, ordered by id, with its
        evidence spans populated."""
        rows = self._conn.execute(
            "SELECT q.id, q.text, q.qtype, q.difficulty, q.reference_answer, "
            "q.provenance, q.generator, q.source_doc_id "
            "FROM question q JOIN suite_question sq ON sq.question_id = q.id "
            "WHERE sq.suite_id = ? ORDER BY q.id",
            (suite_id,),
        ).fetchall()
        for row in rows:
            span_rows = self._conn.execute(
                "SELECT doc_id, chunk_id, page, char_start, char_end, quote "
                "FROM evidence_span WHERE question_id = ? "
                "ORDER BY doc_id, chunk_id, char_start, char_end",
                (row["id"],),
            ).fetchall()
            spans = tuple(
                EvidenceSpan(
                    doc_id=sr["doc_id"],
                    chunk_id=sr["chunk_id"],
                    page=sr["page"],
                    char_start=sr["char_start"],
                    char_end=sr["char_end"],
                    quote=sr["quote"],
                )
                for sr in span_rows
            )
            yield Question(
                id=row["id"],
                text=row["text"],
                qtype=row["qtype"],
                difficulty=row["difficulty"],
                reference_answer=row["reference_answer"],
                evidence_spans=spans,
                provenance=row["provenance"],
                generator=row["generator"],
                source_doc_id=row["source_doc_id"],
            )

    def run_scores(self, run_id: str) -> list[tuple[str, str, float]]:
        """Return every (question_id, metric, value) scored for a run,
        collapsing across scorer versions (a run only ever has one scorer
        version per metric name in practice; integrity.assert_comparable is
        what enforces that across runs being compared)."""
        rows = self._conn.execute(
            "SELECT question_id, metric, value FROM score WHERE run_id = ? "
            "ORDER BY question_id, metric",
            (run_id,),
        ).fetchall()
        return [(r["question_id"], r["metric"], r["value"]) for r in rows]

    def run_scorers(self, run_id: str) -> frozenset[str]:
        """Return the set of `scorer` values ("name@version") that produced
        at least one score in this run. Used by integrity.assert_comparable
        to detect SCORER_MISMATCH."""
        rows = self._conn.execute(
            "SELECT DISTINCT scorer FROM score WHERE run_id = ?", (run_id,)
        ).fetchall()
        return frozenset(r["scorer"] for r in rows)

    def run_result_count(self, run_id: str) -> int:
        """Number of results recorded for a run. Used by
        integrity.assert_comparable to detect EMPTY_RUN."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM result WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row["n"])

    def all_question_ids(self) -> list[str]:
        """Return the id of every question currently in the store, sorted.

        Not part of the M1-SPEC.md §3 method list verbatim; added so the
        CLI's `suite freeze` (M1-SPEC.md §10), which takes no
        question-selection flag, has a well-defined "freeze everything
        imported so far" behaviour without reaching into the store's
        private connection.
        """
        rows = self._conn.execute("SELECT id FROM question ORDER BY id").fetchall()
        return [r["id"] for r in rows]

    def runs_for_suite(self, suite_id: str) -> list[Run]:
        """Return every run (any status) started against `suite_id`,
        ordered by `started_at`.

        Not part of the M1-SPEC.md §3 method list verbatim; added so the
        CLI's `compare` command, which is given a suite rather than an
        explicit list of run ids, has a defined set of runs to compare.
        """
        rows = self._conn.execute(
            "SELECT id FROM run WHERE suite_id = ? ORDER BY started_at", (suite_id,)
        ).fetchall()
        return [self.get_run(r["id"]) for r in rows]

    def find_complete_run(self, suite_hash: str, config_hash: str) -> Run | None:
        """Return the most recently started complete run for
        (suite_hash, config_hash), or None if none exists.

        This is not part of the M1-SPEC.md §3 method list verbatim, but is
        required to implement runner.run_matrix's documented behaviour
        ("skip if a complete run exists ... unless rerun=True") without
        reaching into the store's private sqlite3 connection from outside
        this module.
        """
        row = self._conn.execute(
            "SELECT id FROM run WHERE suite_hash = ? AND config_hash = ? "
            "AND status = 'complete' ORDER BY started_at DESC LIMIT 1",
            (suite_hash, config_hash),
        ).fetchone()
        if row is None:
            return None
        return self.get_run(row["id"])
