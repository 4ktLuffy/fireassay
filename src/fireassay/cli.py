"""fireassay command-line interface.

`compare` and `diff` exit **2** on `ComparisonRefusedError` — distinct from
Click/Typer's own usage-error exit code so a CI script can tell "the suite
was mistyped" apart from "the tool refused an unsound comparison" (see
M1-SPEC.md §10). All other unhandled errors exit 1, Typer's default.

M2 additions (`controls`, `mutate`, `mutation score` — M2-SPEC.md §9):
`controls run` exits **2** on any control `FAILED`, **3** on any
disallowed `NOT_RUN` (checked in that order: a FAILED anywhere takes
priority, since it is never tolerable regardless of `--allow-not-run`).
`mutate` always exits **0** — a low `gate_mutation_score` is a finding to
report, not a command failure (M2-SPEC.md §9).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from fireassay.admissibility import assess
from fireassay.compare import leaderboard
from fireassay.config import MatrixSpec, expand
from fireassay.controls.base import build_control_context
from fireassay.controls.expected import load_expected_bands
from fireassay.controls.registry import ALL_CONTROLS
from fireassay.curate.models import Decision, RubricVerdict
from fireassay.curate.report import build_report
from fireassay.curate.serve import (
    DEFAULT_DOUBLE_REVIEW_RATE,
    DEFAULT_HONEYPOT_RATE,
    next_item,
    submit_decision,
)
from fireassay.filter.config import load_filter_config
from fireassay.filter.pipeline import run_filter
from fireassay.generate.models import ResolvedCandidate
from fireassay.generate.pipeline import generate_candidates, select_chunks_for_target
from fireassay.integrity import ComparisonRefusedError
from fireassay.llm.cache import ResponseCache
from fireassay.llm.ollama import OllamaClient
from fireassay.models import EvidenceSpan, Question
from fireassay.mutation.detector import ThresholdDetector
from fireassay.mutation.operators import load_mutation_config
from fireassay.mutation.run import run_mutation
from fireassay.report.text import (
    render_control_checks,
    render_control_outcomes,
    render_curate_report,
    render_leaderboard,
    render_mutation_run,
    render_mutation_score,
    render_refusal,
)
from fireassay.runner import run_matrix
from fireassay.score.abstention import AbstentionScorer
from fireassay.score.base import PolicyRule, Scorer, ScoringContext
from fireassay.score.cost import CostScorer
from fireassay.score.latency import LatencyScorer
from fireassay.score.policy import PolicyScorer, load_policy_rules
from fireassay.score.retrieval import RetrievalScorer
from fireassay.store.db import Store, SuiteExistsError
from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, Doc, chunk_corpus, corpus_hash, load_corpus

app = typer.Typer(add_completion=False, no_args_is_help=True)
questions_app = typer.Typer(no_args_is_help=True)
suite_app = typer.Typer(no_args_is_help=True)
controls_app = typer.Typer(no_args_is_help=True)
mutation_app = typer.Typer(no_args_is_help=True)
curate_app = typer.Typer(no_args_is_help=True)
app.add_typer(questions_app, name="questions")
app.add_typer(suite_app, name="suite")
app.add_typer(controls_app, name="controls")
app.add_typer(mutation_app, name="mutation")
app.add_typer(curate_app, name="curate")

#: Packaged default for `controls run --expected`: `controls/expected.yaml`
#: ships inside the `fireassay` package itself (see pyproject.toml's
#: force-include), so a plain `pip install fireassay` has a default band
#: file without the caller needing to locate the source checkout.
_DEFAULT_EXPECTED_PATH = Path(__file__).parent / "controls" / "expected.yaml"


def _load_questions_jsonl(path: Path) -> list[Question]:
    """Parse a questions JSONL file into `Question` models.

    Each line is a JSON object with `text`, `qtype`, `difficulty`,
    `provenance` (required), and optional `reference_answer`,
    `evidence_spans` (list of `EvidenceSpan`-shaped objects), `generator`,
    `source_doc_id`. `id` is never read from the file — it is always
    computed by `Question.model_post_init`, so an id typo in the file
    cannot desynchronise from the content it is supposed to address.
    """
    questions: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            spans = tuple(EvidenceSpan(**span) for span in obj.get("evidence_spans", []))
            questions.append(
                Question(
                    text=obj["text"],
                    qtype=obj["qtype"],
                    difficulty=obj["difficulty"],
                    reference_answer=obj.get("reference_answer"),
                    evidence_spans=spans,
                    provenance=obj["provenance"],
                    generator=obj.get("generator"),
                    source_doc_id=obj.get("source_doc_id"),
                )
            )
    return questions


@app.command()
def init(db: Path = typer.Option(..., "--db", help="path to the SQLite database file")) -> None:
    """Create (or migrate) the fireassay store at DB."""
    store = Store(db)
    store.migrate()
    store.close()
    typer.echo(f"initialized {db}")


@questions_app.command("import")
def questions_import(
    path: Path = typer.Argument(..., help="questions JSONL file"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Import questions from a JSONL file into the store."""
    store = Store(db)
    store.migrate()
    questions = _load_questions_jsonl(path)
    inserted = store.put_questions(questions)
    store.close()
    typer.echo(f"imported {inserted} new question(s) ({len(questions)} read from {path})")


def _candidate_to_question(candidate: ResolvedCandidate, decision: Decision) -> Question:
    """One accepted/edited candidate -> one `Question`.

    `chunk_id` is synthesised (`f"{source_doc_id}#curated"`), not a real
    chunking-strategy chunk id: `generate/`'s span resolution is against
    the **document**, not any one chunk (`generate.spans`'s whole point),
    so a curated candidate has a document-level span but no real chunk_id
    to report. `EvidenceSpan.key()` only actually depends on
    `(doc_id, chunk_id, char_start, char_end)`, so a stable synthetic
    chunk_id is sufficient for content-addressing `Question.id` correctly;
    it is never compared against a retrieval system's own chunk_ids
    (`score.retrieval.RetrievalScorer` matches by character-range overlap,
    not chunk_id equality — M1-SPEC.md correction 1).
    """
    edited = decision.decision == "edit"
    text = decision.edited_text if edited and decision.edited_text else candidate.text
    reference_answer = (
        decision.edited_answer if edited and decision.edited_answer else candidate.reference_answer
    )
    span = EvidenceSpan(
        doc_id=candidate.source_doc_id,
        # Synthetic, not a real chunking-strategy id -- safe because
        # RetrievalScorer resolves relevance by character overlap, never
        # by chunk_id equality (see this function's docstring).
        chunk_id=f"{candidate.source_doc_id}#curated",
        char_start=candidate.char_start,
        char_end=candidate.char_end,
        quote=candidate.quote,
    )
    return Question(
        text=text,
        qtype=candidate.qtype,
        difficulty=candidate.difficulty,
        reference_answer=reference_answer,
        evidence_spans=(span,),
        provenance="synthetic",
        generator=f"generate@{candidate.model_digest[:12]}",
        source_doc_id=candidate.source_doc_id,
    )


def _freeze_curated_questions(store: Store) -> list[str]:
    """Convert every latest accept/edit `Decision` into a `Question`,
    import it, and return the resulting question ids — the `--from-curated`
    source set for `suite freeze` (M3-SPEC.md §6).

    Latest-per-`(candidate_id, curator_id)` (`decision` is append-only, so
    a candidate can have several superseded decisions from the same
    curator). Two curators independently accepting the same candidate with
    identical text/answer collapse to one `Question` for free — `Question`
    is content-addressed, so `Store.put_questions` de-duplicates them; if
    their edits genuinely differ, two distinct questions are correctly
    produced.
    """
    decisions = store.get_all_decisions()
    latest: dict[tuple[str, str], Decision] = {}
    for d in decisions:
        key = (d.candidate_id, d.curator_id)
        existing = latest.get(key)
        if existing is None or d.decided_at > existing.decided_at:
            latest[key] = d

    questions = [
        _candidate_to_question(store.get_candidate(d.candidate_id), d)
        for d in latest.values()
        if d.decision in ("accept", "edit")
    ]
    store.put_questions(questions)
    return [q.id for q in questions]


@suite_app.command("freeze")
def suite_freeze(
    name: str = typer.Option(..., "--name"),
    version: str = typer.Option(..., "--version"),
    db: Path = typer.Option(..., "--db"),
    from_curated: bool = typer.Option(
        False, "--from-curated", help="freeze accepted/edited curated candidates, not every imported question"
    ),
) -> None:
    """Freeze a named, versioned suite: by default, every question
    currently in the store; with `--from-curated`, every accepted/edited
    curation `Decision` instead (M3-SPEC.md §6)."""
    store = Store(db)
    store.migrate()
    question_ids = _freeze_curated_questions(store) if from_curated else store.all_question_ids()
    try:
        suite = store.freeze_suite(name, version, question_ids)
    except SuiteExistsError as exc:
        store.close()
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    store.close()
    typer.echo(
        f"froze {suite.name}@{suite.version}: {suite.question_count} question(s), "
        f"suite_hash={suite.suite_hash}"
    )


@suite_app.command("show")
def suite_show(
    ref: str = typer.Argument(..., help="name@version"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Show details of a frozen suite."""
    name, _, version = ref.partition("@")
    store = Store(db)
    try:
        suite = store.get_suite(name, version)
    except KeyError as exc:
        store.close()
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    store.close()
    table = Table(title=f"{suite.name}@{suite.version}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("id", suite.id)
    table.add_row("suite_hash", suite.suite_hash)
    table.add_row("frozen_at", suite.frozen_at)
    table.add_row("question_count", str(suite.question_count))
    Console().print(table)


def _config_int(config: Mapping[str, object], key: str, default: int) -> int:
    """Narrow one value out of a config dict (`Mapping[str, object]`,
    since a `MatrixSpec` axis value can be a `str` or a `dict[str,
    object]` per M1-SPEC.md §5) to a concrete `int`, at the boundary where
    the CLI actually consumes it. `bool` is rejected explicitly even
    though `isinstance(True, int)` is true in Python — a YAML `true`/
    `false` is never a meaningful `top_k`/`chunk_size`, and silently
    coercing it to `0`/`1` would be a confusing failure mode."""
    value = config.get(key, default)
    if isinstance(value, bool):
        raise TypeError(f"config[{key!r}] must be a number, got a bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"config[{key!r}] must be int-like, got {type(value).__name__}")


def _config_float(config: Mapping[str, object], key: str, default: float) -> float:
    """As `_config_int`, narrowing to `float`."""
    value = config.get(key, default)
    if isinstance(value, bool):
        raise TypeError(f"config[{key!r}] must be a number, got a bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"config[{key!r}] must be float-like, got {type(value).__name__}")


@app.command()
def run(
    matrix: Path = typer.Option(..., "--matrix", help="MatrixSpec YAML file"),
    corpus: Path = typer.Option(..., "--corpus", help="corpus JSONL file"),
    db: Path = typer.Option(..., "--db"),
    price_in: float = typer.Option(0.0, "--price-in", help="USD per million input tokens"),
    price_out: float = typer.Option(0.0, "--price-out", help="USD per million output tokens"),
    policy: Path | None = typer.Option(None, "--policy", help="policy rule pack YAML"),
    chunk_size: int = typer.Option(
        800, "--chunk-size", help="fallback chunk size if a config does not set one"
    ),
    chunk_overlap: int = typer.Option(
        100, "--chunk-overlap", help="fallback chunk overlap if a config does not set one"
    ),
    rerun: bool = typer.Option(False, "--rerun", help="re-run configs that already have a complete run"),
) -> None:
    """Run every config in --matrix's expanded grid against its suite,
    using a pure-Python BM25 retriever over --corpus, and score with all
    five M1 deterministic scorers.

    Per-config knobs (`top_k`, `k1`, `b`, `chunk_size`, `chunk_overlap`) are
    read from each expanded config dict when present, falling back to
    --chunk-size/--chunk-overlap and BM25System's own defaults otherwise —
    so a matrix.yaml that varies `top_k` across configs drives the system
    actually built for each one.
    """
    store = Store(db)
    store.migrate()

    with open(matrix, encoding="utf-8") as f:
        raw_spec = yaml.safe_load(f)
    spec = MatrixSpec(**raw_spec)

    docs = load_corpus(corpus)
    policy_rules = load_policy_rules(policy) if policy is not None else ()

    def _chunks_for(config: Mapping[str, object]) -> list[Chunk]:
        size = _config_int(config, "chunk_size", chunk_size)
        overlap = _config_int(config, "chunk_overlap", chunk_overlap)
        return list(chunk_corpus(docs, size=size, overlap=overlap))

    def system_factory(config: Mapping[str, object]) -> System:
        top_k = _config_int(config, "top_k", 5)
        k1 = _config_float(config, "k1", 1.5)
        b = _config_float(config, "b", 0.75)
        return BM25System(_chunks_for(config), top_k=top_k, k1=k1, b=b)

    def ctx_factory(config: Mapping[str, object]) -> ScoringContext:
        top_k = _config_int(config, "top_k", 5)
        return ScoringContext(
            price_in_per_mtok=price_in,
            price_out_per_mtok=price_out,
            top_k=top_k,
            policy_rules=policy_rules,
        )

    def corpus_hash_factory() -> str:
        # Deliberately does not depend on `config`: chunking is a config-axis
        # concern (already captured in config_hash), not a corpus concern —
        # see system.corpus.corpus_hash's docstring.
        return corpus_hash(docs)

    scorers: list[Scorer] = [
        RetrievalScorer(),
        LatencyScorer(),
        CostScorer(),
        PolicyScorer(),
        AbstentionScorer(),
    ]

    runs = run_matrix(
        store,
        spec,
        system_factory,
        scorers,
        ctx_factory,
        rerun=rerun,
        corpus_hash_factory=corpus_hash_factory,
    )
    store.close()
    for r in runs:
        admissible_note = "" if r.admissible else "  ADMISSIBLE=FALSE (invariant violation)"
        typer.echo(f"{r.id}  config={r.config_hash[:12]}  status={r.status}{admissible_note}")


@app.command()
def compare(
    suite: str = typer.Option(..., "--suite", help="name@version"),
    db: Path = typer.Option(..., "--db"),
    force: bool = typer.Option(False, "--force"),
    by_provenance: bool = typer.Option(False, "--by-provenance"),
    split_by: list[str] = typer.Option([], "--split-by", help="qtype and/or difficulty, repeatable"),
) -> None:
    """Build a leaderboard over every run of --suite. Exits 2 if the runs
    are not soundly comparable (see integrity.assert_comparable), unless
    --force is passed."""
    name, _, version = suite.partition("@")
    store = Store(db)
    suite_row = store.get_suite(name, version)
    runs = store.runs_for_suite(suite_row.id)
    try:
        board = leaderboard(store, runs, force=force, split_by_provenance=by_provenance, split_by=split_by)
    except ComparisonRefusedError as exc:
        store.close()
        render_refusal(exc.reasons)
        raise typer.Exit(code=2) from exc
    store.close()
    render_leaderboard(board)


@app.command()
def diff(
    base: str = typer.Option(..., "--base", help="base run id"),
    head: str = typer.Option(..., "--head", help="head run id"),
    db: Path = typer.Option(..., "--db"),
    force: bool = typer.Option(False, "--force"),
    split_by: list[str] = typer.Option([], "--split-by", help="qtype and/or difficulty, repeatable"),
) -> None:
    """Compare exactly two runs. Exits 2 if they are not soundly
    comparable, unless --force is passed."""
    store = Store(db)
    base_run = store.get_run(base)
    head_run = store.get_run(head)
    try:
        board = leaderboard(store, [base_run, head_run], force=force, split_by=split_by)
    except ComparisonRefusedError as exc:
        store.close()
        render_refusal(exc.reasons)
        raise typer.Exit(code=2) from exc
    store.close()
    render_leaderboard(board)


# -- M2: controls / mutate / mutation ---------------------------------------


def _docs_system_factory(
    chunk_size: int, chunk_overlap: int
) -> Callable[[Mapping[str, object], Sequence[Doc]], System]:
    """Build a `(config_spec, docs) -> System` factory matching
    `controls.base.ControlContext.system_factory` / `mutation.run.
    run_mutation`'s expected shape — a second, docs-parameterised sibling
    of `run`'s inline `system_factory` (which closes over one fixed
    `docs`), needed because `corpus_ablation` (and, in principle, a future
    corpus-mutating operator) must be able to build a system over a
    *different* document set."""

    def factory(config: Mapping[str, object], docs: Sequence[Doc]) -> System:
        size = _config_int(config, "chunk_size", chunk_size)
        overlap = _config_int(config, "chunk_overlap", chunk_overlap)
        chunks = list(chunk_corpus(docs, size=size, overlap=overlap))
        top_k = _config_int(config, "top_k", 5)
        k1 = _config_float(config, "k1", 1.5)
        b = _config_float(config, "b", 0.75)
        return BM25System(chunks, top_k=top_k, k1=k1, b=b)

    return factory


def _scoring_ctx_factory(
    price_in: float, price_out: float, policy_rules: tuple[PolicyRule, ...]
) -> Callable[[Mapping[str, object]], ScoringContext]:
    def factory(config: Mapping[str, object]) -> ScoringContext:
        top_k = _config_int(config, "top_k", 5)
        return ScoringContext(
            price_in_per_mtok=price_in,
            price_out_per_mtok=price_out,
            top_k=top_k,
            policy_rules=policy_rules,
        )

    return factory


@controls_app.command("run")
def controls_run(
    matrix: Path = typer.Option(..., "--matrix", help="MatrixSpec YAML file"),
    corpus: Path = typer.Option(..., "--corpus", help="corpus JSONL file"),
    db: Path = typer.Option(..., "--db"),
    price_in: float = typer.Option(0.0, "--price-in"),
    price_out: float = typer.Option(0.0, "--price-out"),
    policy: Path | None = typer.Option(None, "--policy"),
    chunk_size: int = typer.Option(800, "--chunk-size"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap"),
    expected: Path = typer.Option(_DEFAULT_EXPECTED_PATH, "--expected", help="controls/expected.yaml"),
    allow_not_run: list[str] = typer.Option(
        [], "--allow-not-run", help="control kind(s) whose NOT_RUN is tolerated, repeatable"
    ),
) -> None:
    """Run every config in --matrix's expanded grid (exactly like `run`),
    then run the five deterministic controls (plus the `judge_calibration`
    `NOT_RUN` stub) once against the matrix's first (by config_hash)
    config, and record an admissibility verdict on every subject run.

    Exits **2** if any control is `FAILED`; **3** if any control is
    `NOT_RUN` and not covered by `--allow-not-run` (checked after FAILED,
    so a FAILED control always wins the exit code). A run with any
    `FAILED` control is inadmissible regardless of `--allow-not-run` — that
    flag only ever widens what `NOT_RUN` is tolerated for.
    """
    store = Store(db)
    store.migrate()

    with open(matrix, encoding="utf-8") as f:
        raw_spec = yaml.safe_load(f)
    spec = MatrixSpec(**raw_spec)

    docs = load_corpus(corpus)
    policy_rules = load_policy_rules(policy) if policy is not None else ()
    system_factory = _docs_system_factory(chunk_size, chunk_overlap)
    ctx_factory = _scoring_ctx_factory(price_in, price_out, policy_rules)

    def matrix_system_factory(config: Mapping[str, object]) -> System:
        return system_factory(config, docs)

    def corpus_hash_factory() -> str:
        return corpus_hash(docs)

    scorers: list[Scorer] = [
        RetrievalScorer(),
        LatencyScorer(),
        CostScorer(),
        PolicyScorer(),
        AbstentionScorer(),
    ]

    subject_runs = run_matrix(
        store, spec, matrix_system_factory, scorers, ctx_factory, corpus_hash_factory=corpus_hash_factory
    )

    expanded = expand(spec)
    if not expanded:
        store.close()
        typer.echo("matrix expands to zero configs; nothing to check controls against", err=True)
        raise typer.Exit(code=1)
    base_config = store.put_config(expanded[0])

    name, _, version = spec.suite.partition("@")
    suite = store.get_suite(name, version)
    questions = list(store.iter_questions(suite.id))

    expected_bands = load_expected_bands(expected)
    control_ctx = build_control_context(
        store, suite, base_config, questions, docs, system_factory, ctx_factory, scorers, expected_bands
    )
    outcomes = [control.run(control_ctx) for control in ALL_CONTROLS]

    for run in subject_runs:
        violations = store.get_run_invariant_violations(run.id)
        verdict = assess(run, violations, outcomes, allow_not_run=allow_not_run)
        for outcome in outcomes:
            store.put_control_check(
                run.id,
                outcome.kind,
                outcome.status,
                outcome.observed,
                outcome.expected,
                outcome.twin_ok,
                outcome.cause_assertions,
                outcome.detail,
            )
        store.set_admissibility(run.id, verdict.admissible, verdict.model_dump())

    store.close()
    render_control_outcomes(outcomes)

    any_failed = any(o.status == "FAILED" for o in outcomes)
    disallowed_not_run = any(o.status == "NOT_RUN" and o.kind not in allow_not_run for o in outcomes)
    if any_failed:
        raise typer.Exit(code=2)
    if disallowed_not_run:
        raise typer.Exit(code=3)


@controls_app.command("show")
def controls_show(
    run: str = typer.Argument(..., help="run id"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Show the `control_check` rows and admissibility verdict recorded
    against RUN by an earlier `controls run`."""
    store = Store(db)
    checks = store.get_control_checks(run)
    run_row = store.get_run(run)
    store.close()
    console = Console()
    console.print(
        f"run {run_row.id}  admissible={run_row.admissible}  "
        f"admissibility_json={run_row.admissibility_json}"
    )
    render_control_checks(checks, console)


#: Fallback detector settings, used only when neither the mutants YAML's
#: `detector:` block nor an explicitly-passed `--detector-*` flag sets
#: them. Never used to silently override either -- see `mutate`'s
#: docstring and the precedence comment inline below.
_DEFAULT_DETECTOR_METRIC = "retrieval.recall@5"
_DEFAULT_DETECTOR_MAX_DROP = 0.05


@app.command()
def mutate(
    suite: str = typer.Option(..., "--suite", help="name@version"),
    config: str = typer.Option(..., "--config", help="an existing config id"),
    operators: Path = typer.Option(..., "--operators", help="configs/mutants.yaml"),
    corpus: Path = typer.Option(..., "--corpus", help="corpus JSONL file"),
    db: Path = typer.Option(..., "--db"),
    price_in: float = typer.Option(0.0, "--price-in"),
    price_out: float = typer.Option(0.0, "--price-out"),
    chunk_size: int = typer.Option(800, "--chunk-size"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap"),
    detector_metric: str | None = typer.Option(
        None, "--detector-metric", help="overrides --operators' detector.metric; not a default"
    ),
    detector_max_drop: float | None = typer.Option(
        None, "--detector-max-drop", help="overrides --operators' detector.max_drop; not a default"
    ),
    detector_max_rise: float | None = typer.Option(
        None, "--detector-max-rise", help="overrides --operators' detector.max_rise; not a default"
    ),
) -> None:
    """Mutate CONFIG's system with every operator in --operators, baseline
    it once, and report `gate_mutation_score`. Always exits **0** — a low
    score is a finding to report, not a command failure (M2-SPEC.md §9).

    Detector precedence: `--operators`' `detector:` block (if present)
    configures `ThresholdDetector`; a `--detector-*` flag overrides it
    **only when actually passed** (each defaults to `None`, distinguishable
    from "passed the CLI default", precisely so a flag nobody typed can
    never silently beat a setting the YAML file did specify — that was the
    bug: `--detector-metric`/`--detector-max-drop` used to default to
    fixed values that always won, so `detector:` in the YAML was parsed,
    accepted, and then had no effect). If neither the YAML nor any flag
    sets a field, `_DEFAULT_DETECTOR_METRIC`/`_DEFAULT_DETECTOR_MAX_DROP`
    apply, matching this command's original behaviour when no detector
    configuration is given anywhere.
    """
    store = Store(db)
    store.migrate()

    name, _, version = suite.partition("@")
    suite_row = store.get_suite(name, version)
    questions = list(store.iter_questions(suite_row.id))
    docs = load_corpus(corpus)
    config_row = store.get_config(config)

    system_factory = _docs_system_factory(chunk_size, chunk_overlap)
    ctx_factory = _scoring_ctx_factory(price_in, price_out, ())
    scoring_ctx = ctx_factory(config_row.spec)
    scorers: list[Scorer] = [
        RetrievalScorer(),
        LatencyScorer(),
        CostScorer(),
        PolicyScorer(),
        AbstentionScorer(),
    ]
    mutation_config = load_mutation_config(operators)
    operator_list = mutation_config.operators
    yaml_detector = mutation_config.detector

    resolved_metric = (
        detector_metric
        if detector_metric is not None
        else (yaml_detector.metric if yaml_detector is not None else _DEFAULT_DETECTOR_METRIC)
    )
    resolved_max_drop = (
        detector_max_drop
        if detector_max_drop is not None
        else (yaml_detector.max_drop if yaml_detector is not None else None)
    )
    resolved_max_rise = (
        detector_max_rise
        if detector_max_rise is not None
        else (yaml_detector.max_rise if yaml_detector is not None else None)
    )
    if resolved_max_drop is None and resolved_max_rise is None:
        # Neither the YAML nor a flag set either bound -- fall back to the
        # command's original default rather than letting ThresholdDetector
        # raise for "no bound configured at all".
        resolved_max_drop = _DEFAULT_DETECTOR_MAX_DROP
    detector = ThresholdDetector(
        metric=resolved_metric, max_drop=resolved_max_drop, max_rise=resolved_max_rise
    )

    result = run_mutation(
        store, suite_row, config_row, questions, docs, system_factory, scorers, scoring_ctx,
        operator_list, detector,
    )

    mutation_run_id = store.put_mutation_run(
        suite_row, config_row, result.detector, result.killed, result.total, result.equivalent, result.score
    )
    for m in result.mutants:
        store.put_mutant(
            mutation_run_id, m.operator, m.params, m.mutant_run_id, m.killed, m.equivalent,
            m.equivalent_reason, m.detail,
        )

    store.close()
    typer.echo(f"mutation_run_id={mutation_run_id}")
    render_mutation_score(result)


@mutation_app.command("score")
def mutation_score(
    mutation_run: str = typer.Option(..., "--mutation-run"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Show a previously-computed `mutate` result by its mutation_run id."""
    store = Store(db)
    run_row = store.get_mutation_run(mutation_run)
    mutants = store.get_mutants(mutation_run)
    store.close()
    render_mutation_run(run_row, mutants)


# -- M3: generate / filter / curate -----------------------------------------


@app.command()
def generate(
    corpus: Path = typer.Option(..., "--corpus", help="corpus JSONL file"),
    model: str = typer.Option(..., "--model", help="an Ollama model tag, e.g. qwen2.5:7b"),
    n: int = typer.Option(..., "--n", help="approximate total candidates to generate"),
    cache_dir: Path = typer.Option(..., "--cache-dir", help="LLM response cache directory"),
    db: Path = typer.Option(..., "--db"),
    n_per_chunk: int = typer.Option(1, "--n-per-chunk", help="questions requested per chunk"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url", help="Ollama server base URL"),
    chunk_size: int = typer.Option(800, "--chunk-size"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap"),
    batch_id: str = typer.Option("", "--batch-id", help="defaults to a fresh random id"),
) -> None:
    """Generate candidate questions from --corpus using --model, via
    Ollama (the only command permitted to call an LLM — M3-SPEC.md §1/§2).

    Every rejection (`NOT_A_QUESTION`, `QUOTE_NOT_FOUND`, `QUOTE_AMBIGUOUS`)
    is recorded, not silently dropped — see `generate.pipeline` and
    `curate report`'s funnel.
    """
    store = Store(db)
    store.migrate()

    docs = load_corpus(corpus)
    chunks = list(chunk_corpus(docs, size=chunk_size, overlap=chunk_overlap))
    selected = select_chunks_for_target(chunks, n, n_per_chunk)

    cache = ResponseCache(Path(cache_dir) / "responses.jsonl")
    client = OllamaClient(base_url=base_url, cache=cache)
    model_ref = client.model_ref(model)

    resolved_batch_id = batch_id or uuid.uuid4().hex
    stats = generate_candidates(
        store, client, model_ref, docs, selected, n_per_chunk=n_per_chunk, batch_id=resolved_batch_id
    )
    store.close()
    typer.echo(
        f"batch_id={resolved_batch_id}  generated={stats.generated}  kept={stats.kept}  "
        f"not_a_question={stats.not_a_question}  quote_not_found={stats.quote_not_found}  "
        f"quote_ambiguous={stats.quote_ambiguous}"
    )


@app.command("filter")
def filter_candidates(
    batch: str = typer.Option(..., "--batch", help="a --batch-id from a prior `generate` run"),
    config: Path = typer.Option(..., "--config", help="configs/filter.yaml"),
    corpus: Path = typer.Option(
        ..., "--corpus", help="corpus JSONL file (required: the `generic` stage needs corpus-wide "
        "document frequency, which M3-SPEC.md's CLI signature omits but the stage cannot run without)"
    ),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Run the deterministic filter pipeline (M3-SPEC.md §3) over every
    candidate generated in --batch, persisting a `filter_result` row for
    every candidate at every stage it reaches."""
    store = Store(db)
    store.migrate()

    candidates = store.get_candidates_by_batch(batch)
    if not candidates:
        store.close()
        typer.echo(f"no candidates found for batch {batch!r}", err=True)
        raise typer.Exit(code=1)

    docs = load_corpus(corpus)
    filter_config = load_filter_config(config)
    result = run_filter(candidates, docs, filter_config)
    for stage_result in result.stage_results:
        store.put_filter_result(
            stage_result.candidate_id, stage_result.stage, stage_result.kept, stage_result.reason
        )
    store.close()
    typer.echo(f"batch={batch}  in={len(candidates)}  kept={len(result.kept)}")


@curate_app.command("next")
def curate_next(
    curator: str = typer.Option(..., "--curator"),
    db: Path = typer.Option(..., "--db"),
    honeypot_rate: float = typer.Option(DEFAULT_HONEYPOT_RATE, "--honeypot-rate"),
    double_review_rate: float = typer.Option(DEFAULT_DOUBLE_REVIEW_RATE, "--double-review-rate"),
) -> None:
    """Emit the next undecided queue item for --curator as JSON
    (M3-SPEC.md §6) — the non-interactive core the M3b TUI will drive.
    Lazily builds --curator's queue on first call, over every currently
    filter-kept candidate. Prints `{"done": true}` once the queue is
    exhausted."""
    store = Store(db)
    store.migrate()
    item = next_item(store, curator, honeypot_rate=honeypot_rate, double_review_rate=double_review_rate)
    store.close()
    if item is None:
        typer.echo(json.dumps({"done": True}))
        return
    payload = {
        "queue_item_id": item.queue_item_id,
        "candidate_id": item.candidate_id,
        "curator_id": item.curator_id,
        **item.view,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False))


@curate_app.command("submit")
def curate_submit(
    file: Path = typer.Argument(..., help="a verdict JSON file, shaped like curate.models.Decision"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Record one curator decision from --file (M3-SPEC.md §6). Exits 1 —
    not a silent no-op — if the file is not a valid `Decision` (e.g. a
    reject with no `reject_reason`)."""
    with open(file, encoding="utf-8") as f:
        raw = json.load(f)
    try:
        decision = Decision(
            candidate_id=raw["candidate_id"],
            curator_id=raw["curator_id"],
            decision=raw["decision"],
            reject_reason=raw.get("reject_reason"),
            rubric=RubricVerdict(**raw["rubric"]),
            edited_text=raw.get("edited_text"),
            edited_answer=raw.get("edited_answer"),
            notes=raw.get("notes"),
            duration_ms=raw["duration_ms"],
        )
    except (ValidationError, KeyError) as exc:
        typer.echo(f"invalid verdict in {file}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    store = Store(db)
    store.migrate()
    stored = submit_decision(store, decision)
    store.close()
    typer.echo(f"recorded decision {stored.id}: candidate={stored.candidate_id} curator={stored.curator_id}")


@curate_app.command("report")
def curate_report_cmd(
    db: Path = typer.Option(..., "--db"),
    corpus: Path | None = typer.Option(
        None, "--corpus", help="optional corpus JSONL: enables the content-coverage section"
    ),
    suite: str | None = typer.Option(
        None, "--suite", help="optional name@version: persists the alpha<0.6 flag onto suite.agreement_json"
    ),
) -> None:
    """The funnel, agreement, honeypot accuracy, curator-quality, coverage,
    and difficulty-validation report (M3-SPEC.md §4).

    `--corpus` and `--suite` are not in M3-SPEC.md §6's literal CLI
    signature; both are additive and optional — see the delivery notes for
    why content coverage cannot be computed without corpus access, and why
    persisting `agreement_json` needs a suite to persist it onto.
    """
    store = Store(db)
    store.migrate()
    filter_results = store.get_all_filter_results()
    decisions = store.get_all_decisions()
    queue_items = store.get_all_queue_items()
    candidates = store.get_all_candidates()
    docs = load_corpus(corpus) if corpus is not None else None

    report = build_report(filter_results, decisions, queue_items, candidates, docs)

    if suite is not None:
        name, _, version = suite.partition("@")
        suite_row = store.get_suite(name, version)
        # Status travels with the number, not just the alpha value: a
        # suite built on unmeasurable agreement must carry that fact
        # forward permanently, exactly as a low alpha does (see
        # curate.agreement's module docstring on why collapsing
        # "unmeasurable" into a number is the bug this schema prevents).
        agreement_payload: dict[str, object] = {
            "by_criterion": {
                criterion: {"alpha": result.alpha, "status": result.status, "detail": result.detail}
                for criterion, result in report.agreement_by_criterion.items()
            },
            "low_agreement_criteria": list(report.low_agreement_criteria),
            "unmeasurable_criteria": list(report.unmeasurable_criteria),
        }
        store.set_agreement(suite_row.id, agreement_payload)

    store.close()
    render_curate_report(report)


if __name__ == "__main__":
    app()
