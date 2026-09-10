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

from fireassay.curate.models import Decision, QueueItem, RubricVerdict
from fireassay.gate import GateReport
from fireassay.generate.models import CandidateFeatures, ResolvedCandidate
from fireassay.hashing import config_hash as _config_hash
from fireassay.hashing import suite_hash as _suite_hash
from fireassay.models import (
    Config,
    EvidenceSpan,
    GateCheckRow,
    Question,
    RetrievedChunk,
    Run,
    Score,
    Suite,
    SystemOutput,
)
from fireassay.score.invariants import InvariantViolation

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: The name of `filter.pipeline`'s final stage ("balance") — duplicated
#: here as a literal rather than imported, for the same reason
#: `FilterResultRow` is its own dataclass rather than `filter.pipeline.
#: StageResult`: `store/db.py` does not import pipeline modules from
#: packages that import `Store`. `get_kept_candidate_ids` uses it to
#: define "kept": a candidate_id with a `kept=1` row at this exact stage.
_FINAL_FILTER_STAGE = "balance"


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
class FilterResultRow:
    """One persisted `filter_result` row (M3).

    A plain dataclass, not `filter.pipeline.StageResult` — mirrors
    `ControlCheckRow`'s reasoning: modules under `generate/`/`filter/`
    import `Store` (`generate.pipeline`, and any future `filter` CLI
    wiring), so `store/db.py` must not import back from either package's
    pipeline modules to build its own return type.
    """

    candidate_id: str
    stage: str
    kept: bool
    reason: str | None
    checked_at: str


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


class AgreementAlreadySetError(Exception):
    """Raised by `set_agreement` when `suite_id` already has a recorded
    `agreement_json` verdict that **conflicts** with the one just
    computed. Mirrors `AdmissibilityAlreadySetError` exactly: a
    byte-identical re-assessment is an idempotent no-op, but a genuinely
    different verdict landing on a suite that already has one must not
    silently overwrite it — a suite's `agreement_json` is surfaced in
    every report built on that suite, permanently (M3-SPEC.md §4), so it
    must not silently change underneath a reader who already looked at it.
    """


class PreStratificationCandidateError(Exception):
    """Raised by `_row_to_candidate` when a `candidate` row has no target
    cell (`target_qtype`/`target_difficulty` both `''`) — a row inserted
    before migration 0004 added stratified generation.

    Migration 0004's own comment already explains why this is correct
    behaviour: `''` is not a valid `QType`/`Difficulty`, and no code path
    ever writes it deliberately. Left unchecked, constructing
    `ResolvedCandidate` from such a row raises pydantic's own
    `literal_error` — a ~40-line traceback that buries the actual
    diagnosis exactly the way an unfiltered Ollama API response once did
    (see `llm/ollama.py`'s `_api_error`). A response body and a raw
    traceback are the same mistake: never let the diagnosis be the thing
    the reader has to decode. This is caught here and re-raised as one
    short, named sentence instead.
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


class GateAlreadyCheckedError(Exception):
    """Raised by `put_gate_report` when a (base_run_id, head_run_id, metric)
    triple already has a recorded `gate_check` row.

    `gate_check.UNIQUE (base_run_id, head_run_id, metric)` (migration
    0006) exists precisely so this cannot silently happen: a gate check
    for a given pair of runs may be recorded exactly once. Re-running a
    gate on the same pair of runs until it happens to come back green is
    p-hacking with extra steps -- "the gate counts how many times you
    asked" (docs/SPEC.md §8) -- so the schema forbids it rather than
    trusting the operator or a retry loop not to do it. A genuinely new
    comparison needs a fresh head run, not a second gate check against
    the same one.
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
            "SELECT id, name, version, suite_hash, frozen_at, question_count, agreement_json "
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
                agreement_json=json.loads(row["agreement_json"]),
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
            "SELECT id, name, version, suite_hash, frozen_at, question_count, agreement_json "
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
            agreement_json=json.loads(row["agreement_json"]),
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

    # -- M3: agreement --------------------------------------------------------

    def set_agreement(self, suite_id: str, agreement_json: Mapping[str, object]) -> None:
        """Record `curate.agreement`'s verdict onto `suite_id`. Idempotent
        for a byte-identical re-assessment; raises `AgreementAlreadySetError`
        for a genuinely different one — see that exception's docstring."""
        row = self._conn.execute("SELECT agreement_json FROM suite WHERE id = ?", (suite_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such suite: {suite_id}")
        new_json = json.dumps(dict(agreement_json), sort_keys=True)
        if row["agreement_json"] not in ("{}", None):
            if row["agreement_json"] == new_json:
                return
            raise AgreementAlreadySetError(
                f"suite {suite_id} already has a CONFLICTING agreement_json recorded: "
                f"stored={row['agreement_json']!r}; new={new_json!r}"
            )
        self._conn.execute("UPDATE suite SET agreement_json = ? WHERE id = ?", (new_json, suite_id))
        self._conn.commit()

    # -- M3: candidate ----------------------------------------------------------

    def put_candidate(self, candidate: ResolvedCandidate) -> None:
        """Persist one span-resolved, feature-measured candidate. Idempotent
        on `id` (`INSERT OR IGNORE`) — `generate.pipeline.generate_candidates`
        mints a fresh uuid4 per raw candidate, so a repeat insert only
        happens if a caller genuinely re-submits the same `ResolvedCandidate`
        object, which is safe to no-op."""
        self._conn.execute(
            "INSERT OR IGNORE INTO candidate (id, batch_id, text, qtype, difficulty, target_qtype, "
            "target_difficulty, reference_answer, quote, source_doc_id, char_start, char_end, "
            "features_json, model_digest, prompt_hash, created_at, chunk_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.id,
                candidate.batch_id,
                candidate.text,
                candidate.qtype,
                candidate.difficulty,
                candidate.target_qtype,
                candidate.target_difficulty,
                candidate.reference_answer,
                candidate.quote,
                candidate.source_doc_id,
                candidate.char_start,
                candidate.char_end,
                json.dumps(candidate.features.model_dump(), sort_keys=True),
                candidate.model_digest,
                candidate.prompt_hash,
                candidate.created_at,
                candidate.chunk_id,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> ResolvedCandidate:
        if row["target_qtype"] == "" or row["target_difficulty"] == "":
            raise PreStratificationCandidateError(
                f"candidate {row['id']} predates stratified generation (migration 0004): it has "
                "no target cell. Regenerate this batch; rows created before stratification "
                "cannot be read back as if they had one."
            )
        return ResolvedCandidate(
            id=row["id"],
            batch_id=row["batch_id"],
            text=row["text"],
            qtype=row["qtype"],
            difficulty=row["difficulty"],
            target_qtype=row["target_qtype"],
            target_difficulty=row["target_difficulty"],
            reference_answer=row["reference_answer"],
            quote=row["quote"],
            source_doc_id=row["source_doc_id"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            features=CandidateFeatures(**json.loads(row["features_json"])),
            model_digest=row["model_digest"],
            prompt_hash=row["prompt_hash"],
            created_at=row["created_at"],
            chunk_id=row["chunk_id"],
        )

    def get_candidate(self, candidate_id: str) -> ResolvedCandidate:
        row = self._conn.execute("SELECT * FROM candidate WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such candidate: {candidate_id}")
        return self._row_to_candidate(row)

    def get_candidates_by_batch(self, batch_id: str) -> list[ResolvedCandidate]:
        rows = self._conn.execute(
            "SELECT * FROM candidate WHERE batch_id = ? ORDER BY created_at, id", (batch_id,)
        ).fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def get_all_candidates(self) -> list[ResolvedCandidate]:
        rows = self._conn.execute("SELECT * FROM candidate ORDER BY created_at, id").fetchall()
        return [self._row_to_candidate(r) for r in rows]

    def candidate_source_chunks(self) -> set[str]:
        """Every `chunk_id` with at least one persisted `candidate` row
        (M3b-SPEC.md Part 1) — real, non-empty chunk ids only. A `''`
        `chunk_id` (a row written before migration 0005 added the column)
        is excluded rather than treated as one shared "no chunk" bucket:
        see migration 0005's own comment for why that is a deliberate,
        bounded trade rather than a bug.

        Read by `generate.pipeline.generate_candidates` to build the
        resume skip set: a chunk already in this set is skipped entirely
        rather than reprocessed, which is what makes a re-run over an
        existing database produce no duplicate candidates and avoid
        replaying an already-cached (but still ~1s/candidate) generation
        call for a chunk that was already generated."""
        rows = self._conn.execute(
            "SELECT DISTINCT chunk_id FROM candidate WHERE chunk_id != ''"
        ).fetchall()
        return {r["chunk_id"] for r in rows}

    # -- M3: filter_result --------------------------------------------------

    def put_filter_result(self, candidate_id: str, stage: str, kept: bool, reason: str | None) -> None:
        """Persist one `(candidate_id, stage)` filter-pipeline outcome.
        Idempotent on `(candidate_id, stage)` (the primary key) — both
        `generate.pipeline` and `filter.pipeline` visit each candidate at
        each stage at most once by construction, so a repeat call is only
        ever a genuine re-run, safe to no-op rather than error."""
        self._conn.execute(
            "INSERT OR IGNORE INTO filter_result (candidate_id, stage, kept, reason, checked_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (candidate_id, stage, int(kept), reason, _now()),
        )
        self._conn.commit()

    def get_all_filter_results(self) -> list[FilterResultRow]:
        rows = self._conn.execute(
            "SELECT candidate_id, stage, kept, reason, checked_at FROM filter_result"
        ).fetchall()
        return [
            FilterResultRow(
                candidate_id=r["candidate_id"],
                stage=r["stage"],
                kept=bool(r["kept"]),
                reason=r["reason"],
                checked_at=r["checked_at"],
            )
            for r in rows
        ]

    def get_kept_candidate_ids(self) -> set[str]:
        """Every candidate_id that survived the full filter pipeline: has a
        `kept=1` `filter_result` row at the final stage (`balance`). A
        candidate the filter pipeline has not yet processed at all — no
        `balance`-stage row either way — is correctly excluded, not
        assumed kept."""
        rows = self._conn.execute(
            "SELECT candidate_id FROM filter_result WHERE stage = ? AND kept = 1",
            (_FINAL_FILTER_STAGE,),
        ).fetchall()
        return {r["candidate_id"] for r in rows}

    # -- M3: queue_item -----------------------------------------------------

    def put_queue_items(self, items: Sequence[QueueItem]) -> None:
        """Persist a batch of queue items. Idempotent on `id` (content
        hash of `(queue_id, curator_id, candidate_id)`, see `curate.queue.
        build_queue`) — re-building an unchanged queue is a no-op."""
        for item in items:
            self._conn.execute(
                "INSERT OR IGNORE INTO queue_item (id, queue_id, curator_id, candidate_id, position, "
                "is_honeypot, honeypot_expected_reason, is_double_review) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    item.queue_id,
                    item.curator_id,
                    item.candidate_id,
                    item.position,
                    int(item.is_honeypot),
                    item.honeypot_expected_reason,
                    int(item.is_double_review),
                ),
            )
        self._conn.commit()

    @staticmethod
    def _row_to_queue_item(row: sqlite3.Row) -> QueueItem:
        return QueueItem(
            id=row["id"],
            queue_id=row["queue_id"],
            curator_id=row["curator_id"],
            candidate_id=row["candidate_id"],
            position=row["position"],
            is_honeypot=bool(row["is_honeypot"]),
            honeypot_expected_reason=row["honeypot_expected_reason"],
            is_double_review=bool(row["is_double_review"]),
        )

    def has_queue_items_for_curator(self, curator_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM queue_item WHERE curator_id = ? LIMIT 1", (curator_id,)
        ).fetchone()
        return row is not None

    def next_undecided_queue_item(self, curator_id: str) -> QueueItem | None:
        """The earliest-position `curator_id` queue item with no recorded
        `decision` yet for `(candidate_id, curator_id)`. Once *any*
        decision exists for a pair, that item is done — `decision` is
        append-only, so a changed mind is a new row for the same pair, not
        a reason to re-serve the item (M3-SPEC.md §5)."""
        row = self._conn.execute(
            "SELECT * FROM queue_item WHERE curator_id = ? AND candidate_id NOT IN "
            "(SELECT candidate_id FROM decision WHERE curator_id = ?) ORDER BY position ASC LIMIT 1",
            (curator_id, curator_id),
        ).fetchone()
        return None if row is None else self._row_to_queue_item(row)

    def get_all_queue_items(self) -> list[QueueItem]:
        rows = self._conn.execute("SELECT * FROM queue_item ORDER BY curator_id, position").fetchall()
        return [self._row_to_queue_item(r) for r in rows]

    # -- M3: decision -----------------------------------------------------------

    def put_decision(self, decision: Decision) -> Decision:
        """Persist `decision`, minting a fresh `id`/`decided_at` regardless
        of whatever `decision.id`/`decision.decided_at` were set to (a
        `Decision` built from a submitted `verdict.json` never sets
        either) — always an INSERT, never an UPDATE: `decision` is
        append-only (M3-SPEC.md §5)."""
        decision_id = uuid.uuid4().hex
        decided_at = _now()
        self._conn.execute(
            "INSERT INTO decision (id, candidate_id, curator_id, decision, reject_reason, rubric_json, "
            "edited_text, edited_answer, notes, duration_ms, decided_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision_id,
                decision.candidate_id,
                decision.curator_id,
                decision.decision,
                decision.reject_reason,
                json.dumps(decision.rubric.model_dump(), sort_keys=True),
                decision.edited_text,
                decision.edited_answer,
                decision.notes,
                decision.duration_ms,
                decided_at,
            ),
        )
        self._conn.commit()
        return decision.model_copy(update={"id": decision_id, "decided_at": decided_at})

    @staticmethod
    def _row_to_decision(row: sqlite3.Row) -> Decision:
        return Decision(
            id=row["id"],
            candidate_id=row["candidate_id"],
            curator_id=row["curator_id"],
            decision=row["decision"],
            reject_reason=row["reject_reason"],
            rubric=RubricVerdict(**json.loads(row["rubric_json"])),
            edited_text=row["edited_text"],
            edited_answer=row["edited_answer"],
            notes=row["notes"],
            duration_ms=row["duration_ms"],
            decided_at=row["decided_at"],
        )

    def get_all_decisions(self) -> list[Decision]:
        rows = self._conn.execute("SELECT * FROM decision ORDER BY decided_at").fetchall()
        return [self._row_to_decision(r) for r in rows]

    # -- M5: gate_check -------------------------------------------------------

    def put_gate_report(self, report: GateReport) -> None:
        """Persist one `gate.GateReport`, one `gate_check` row per metric,
        inside a single transaction.

        Raises `GateAlreadyCheckedError` (translated from the
        `sqlite3.IntegrityError` the `UNIQUE (base_run_id, head_run_id,
        metric)` constraint raises -- see migration 0006's header
        comment) the moment any metric's triple already has a recorded
        row, and rolls back everything this call had already inserted so
        far -- a partially-persisted `GateReport` (some metrics recorded,
        others not) would be exactly as misleading as the repeated check
        this constraint exists to forbid. `gate_check` rows are
        append-only (migration 0006's triggers): this method only ever
        INSERTs, never UPDATEs or DELETEs.
        """
        for result in report.results:
            check_id = uuid.uuid4().hex
            try:
                self._conn.execute(
                    "INSERT INTO gate_check (id, base_run_id, head_run_id, suite_id, metric, "
                    "n_items, base_mean, head_mean, delta, ci_low, ci_high, p_value, p_adjusted, "
                    "mde, threshold, alpha, power, bootstrap_b, seed, verdict, checked_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        check_id,
                        report.base_run_id,
                        report.head_run_id,
                        report.suite_id,
                        result.metric,
                        result.n,
                        result.base_mean,
                        result.head_mean,
                        result.delta,
                        result.ci_low,
                        result.ci_high,
                        result.p_value,
                        result.p_adjusted,
                        result.mde,
                        result.threshold,
                        report.alpha,
                        report.power,
                        report.b,
                        report.seed,
                        result.verdict,
                        _now(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise GateAlreadyCheckedError(
                    f"gate_check already recorded for base_run_id={report.base_run_id!r}, "
                    f"head_run_id={report.head_run_id!r}, metric={result.metric!r} -- a gate "
                    "check cannot be repeated: re-running a gate until it passes is p-hacking"
                ) from exc
        self._conn.commit()

    def get_gate_checks(self, base_run_id: str, head_run_id: str) -> list[GateCheckRow]:
        """Return every `gate_check` row recorded for (base_run_id,
        head_run_id), ordered by metric."""
        rows = self._conn.execute(
            "SELECT id, base_run_id, head_run_id, suite_id, metric, n_items, base_mean, "
            "head_mean, delta, ci_low, ci_high, p_value, p_adjusted, mde, threshold, alpha, "
            "power, bootstrap_b, seed, verdict, checked_at FROM gate_check "
            "WHERE base_run_id = ? AND head_run_id = ? ORDER BY metric",
            (base_run_id, head_run_id),
        ).fetchall()
        return [
            GateCheckRow(
                id=r["id"],
                base_run_id=r["base_run_id"],
                head_run_id=r["head_run_id"],
                suite_id=r["suite_id"],
                metric=r["metric"],
                n_items=r["n_items"],
                base_mean=r["base_mean"],
                head_mean=r["head_mean"],
                delta=r["delta"],
                ci_low=r["ci_low"],
                ci_high=r["ci_high"],
                p_value=r["p_value"],
                p_adjusted=r["p_adjusted"],
                mde=r["mde"],
                threshold=r["threshold"],
                alpha=r["alpha"],
                power=r["power"],
                bootstrap_b=r["bootstrap_b"],
                seed=r["seed"],
                verdict=r["verdict"],
                checked_at=r["checked_at"],
            )
            for r in rows
        ]
