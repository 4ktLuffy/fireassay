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

import csv
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import date
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
from fireassay.evidence import EvidenceError, default_style, load_claims, run_check, run_one_claim
from fireassay.evidence import render_markdown as render_evidence_markdown
from fireassay.filter.config import load_filter_config
from fireassay.filter.pipeline import run_filter
from fireassay.gate import MetricSpec, ThresholdBelowMDEError, evaluate_gate
from fireassay.generate.models import ResolvedCandidate
from fireassay.generate.pipeline import (
    DEFAULT_MAX_GENERATION_FAILURE_RATE,
    GenerationFailureRateExceededError,
    generate_candidates,
    select_chunks_for_target,
)
from fireassay.integrity import ComparisonRefusedError, assert_comparable
from fireassay.items.adapters.store import load_matrix as items_load_matrix
from fireassay.items.adapters.store import load_meta as items_load_meta
from fireassay.items.adapters.tabular import load_meta_jsonl, load_responses_csv, load_responses_jsonl
from fireassay.items.calibration import (
    CalibrationLabel,
    SamplingDesign,
    Stratum,
    append_calibration_jsonl,
    evaluate_detector,
    load_calibration_jsonl,
)
from fireassay.items.core import ItemMeta, ItemResponses
from fireassay.items.core import analyse as run_item_analysis
from fireassay.items.review import (
    ReviewKeyEntry,
    ReviewLabel,
    ReviewSource,
    build_review_batch,
    build_review_key,
    score_review,
)
from fireassay.items.seed import seed_batch
from fireassay.items.validation import (
    DetectorValidation,
    derive_verdict,
    load_validations,
    write_validations,
)
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
    render_detector_evaluation,
    render_gate_report,
    render_items_analysis,
    render_items_score,
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
from fireassay.store.db import GateAlreadyCheckedError, Store, SuiteExistsError
from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, Doc, chunk_corpus, corpus_hash, load_corpus

app = typer.Typer(add_completion=False, no_args_is_help=True)
questions_app = typer.Typer(no_args_is_help=True)
suite_app = typer.Typer(no_args_is_help=True)
controls_app = typer.Typer(no_args_is_help=True)
mutation_app = typer.Typer(no_args_is_help=True)
curate_app = typer.Typer(no_args_is_help=True)
items_app = typer.Typer(no_args_is_help=True)
items_review_app = typer.Typer(no_args_is_help=True)
app.add_typer(questions_app, name="questions")
app.add_typer(suite_app, name="suite")
app.add_typer(controls_app, name="controls")
app.add_typer(mutation_app, name="mutation")
app.add_typer(curate_app, name="curate")
app.add_typer(items_app, name="items")
items_app.add_typer(items_review_app, name="review")

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


# -- M5: gate -----------------------------------------------------------------

#: Exit codes specific to `fireassay gate`, disjoint from the codes every
#: other command already uses (1 usage error, 2 comparison refused, 3
#: disallowed NOT_RUN) so a CI job can tell *why* the gate did not return
#: 0 without parsing output. 2 is shared with compare/diff on purpose: it
#: means the same thing here (integrity.assert_comparable refused).
GATE_EXIT_BLOCKED = 4
GATE_EXIT_BELOW_MDE = 5
GATE_EXIT_ALREADY_CHECKED = 6


def _parse_metric_spec(raw: str) -> MetricSpec:
    """`NAME=THRESHOLD[:lower]` -> `MetricSpec`.

    `NAME` is a metric name exactly as scored (`retrieval.recall@5`);
    `THRESHOLD` is the regression size to block on, in metric units, `> 0`;
    the optional `:lower` suffix says lower is better for this metric
    (`latency.total_ms=50:lower` blocks a 50 ms *rise*). `=` is the
    separator because metric names already contain `@` and `.`.
    """
    name, sep, rest = raw.partition("=")
    if not sep or not name:
        raise typer.BadParameter(f"--metric must be NAME=THRESHOLD[:lower], got {raw!r}")
    threshold_text, _, direction = rest.partition(":")
    try:
        threshold = float(threshold_text)
    except ValueError as exc:
        raise typer.BadParameter(f"--metric {raw!r}: threshold {threshold_text!r} is not a number") from exc
    if direction not in ("", "lower"):
        raise typer.BadParameter(
            f"--metric {raw!r}: the only direction suffix is ':lower', got {direction!r}"
        )
    if threshold <= 0:
        raise typer.BadParameter(f"--metric {raw!r}: threshold must be > 0")
    return MetricSpec(metric=name, threshold=threshold, higher_is_better=(direction != "lower"))


def _aligned_per_item(
    store: Store, base_run_id: str, head_run_id: str, metrics: Sequence[str]
) -> tuple[dict[str, tuple[list[float], list[float]]], dict[str, int]]:
    """Per-metric `(base_values, head_values)` aligned by question id --
    the pairing `gate.paired_bootstrap` depends on -- plus, per metric,
    how many questions were scored on one side only and therefore
    dropped. A question a metric does not apply to on one run (e.g. a
    retrieval metric on a question with no gold span) is outside the
    paired population, never imputed as 0 -- the same rule
    `compare.leaderboard` applies."""
    base_by: dict[str, dict[str, float]] = {}
    for question_id, metric, value in store.run_scores(base_run_id):
        base_by.setdefault(metric, {})[question_id] = value
    head_by: dict[str, dict[str, float]] = {}
    for question_id, metric, value in store.run_scores(head_run_id):
        head_by.setdefault(metric, {})[question_id] = value

    per_item: dict[str, tuple[list[float], list[float]]] = {}
    dropped: dict[str, int] = {}
    for metric in metrics:
        base_values = base_by.get(metric, {})
        head_values = head_by.get(metric, {})
        shared = sorted(set(base_values) & set(head_values))
        per_item[metric] = ([base_values[q] for q in shared], [head_values[q] for q in shared])
        dropped[metric] = len(set(base_values) ^ set(head_values))
    return per_item, dropped


@app.command()
def gate(
    base: str = typer.Option(..., "--base", help="base run id"),
    head: str = typer.Option(..., "--head", help="head run id"),
    db: Path = typer.Option(..., "--db"),
    metric: list[str] = typer.Option(
        ...,
        "--metric",
        help="NAME=THRESHOLD[:lower], repeatable: block if NAME regresses by at least THRESHOLD",
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="family-wise significance level"),
    power: float = typer.Option(0.80, "--power", help="power the MDE refusal is computed at"),
    bootstrap_b: int = typer.Option(10000, "--b", help="paired-bootstrap replicates per metric"),
    seed: int = typer.Option(0, "--seed"),
    force: bool = typer.Option(False, "--force", help="gate an unsound comparison anyway (stamped)"),
    persist: bool = typer.Option(
        True,
        "--persist/--no-persist",
        help="record the verdict as gate_check rows (once per base/head/metric, ever)",
    ),
) -> None:
    """The release gate (M5): decide whether HEAD regressed against BASE
    on every --metric, and say so with a number.

    Refuses before it decides: the two runs must be soundly comparable
    (`integrity.assert_comparable`, exactly as `compare`/`diff`), and every
    --metric threshold must be at least the minimum detectable effect this
    suite can resolve at this sample size and corrected alpha
    (`gate.evaluate_gate`). A verdict is `block` only when the regression
    is both at least THRESHOLD in size and significant after
    Holm-Bonferroni correction across every metric checked.

    Exit codes: **0** every metric passed; **2** comparison refused;
    **4** at least one metric BLOCKED; **5** a threshold is below the MDE
    (no verdict was reached -- refusing is not passing); **6** this
    (base, head, metric) has already been gate-checked and --persist is
    on (re-running a gate on the same pair until it goes green is
    p-hacking, and the schema forbids it); **1** usage error.
    """
    specs = [_parse_metric_spec(m) for m in metric]
    if base == head:
        typer.echo("gate: --base and --head are the same run; nothing to compare")
        raise typer.Exit(code=1)

    store = Store(db)
    store.migrate()
    base_run = store.get_run(base)
    head_run = store.get_run(head)

    scorers_by_run = {r.id: store.run_scorers(r.id) for r in (base_run, head_run)}
    try:
        comparability = assert_comparable([base_run, head_run], scorers_by_run=scorers_by_run, force=force)
    except ComparisonRefusedError as exc:
        store.close()
        render_refusal(exc.reasons)
        raise typer.Exit(code=2) from exc

    per_item, dropped = _aligned_per_item(store, base_run.id, head_run.id, [s.metric for s in specs])
    for spec in specs:
        if not per_item[spec.metric][0]:
            store.close()
            typer.echo(f"gate: no question is scored for {spec.metric!r} in both runs")
            raise typer.Exit(code=1)

    try:
        report = evaluate_gate(
            base_run_id=base_run.id,
            head_run_id=head_run.id,
            suite_id=base_run.suite_id,
            per_item=per_item,
            specs=specs,
            alpha=alpha,
            power=power,
            b=bootstrap_b,
            seed=seed,
        )
    except ThresholdBelowMDEError as exc:
        store.close()
        typer.echo(f"GATE REFUSED: {exc}")
        raise typer.Exit(code=GATE_EXIT_BELOW_MDE) from exc

    if persist:
        try:
            store.put_gate_report(report)
        except GateAlreadyCheckedError as exc:
            store.close()
            typer.echo(
                f"gate: base={base_run.id} head={head_run.id} has already been gate-checked on "
                "at least one of these metrics; a genuinely new comparison needs a fresh head run "
                "(--no-persist recomputes without recording)"
            )
            raise typer.Exit(code=GATE_EXIT_ALREADY_CHECKED) from exc
    store.close()

    render_gate_report(report, comparability, dropped)
    if report.blocked:
        raise typer.Exit(code=GATE_EXIT_BLOCKED)


# -- evidence -----------------------------------------------------------------


@app.command()
def evidence(
    claims: Path = typer.Option(..., "--claims", help="a Python file defining CLAIMS: tuple[Claim, ...]"),
    check: bool = typer.Option(False, "--check", help="recompute every claim; exit 1 on any FAIL/ERROR"),
    markdown: bool = typer.Option(False, "--markdown", help="print EVIDENCE.md for the claims"),
    claim: str | None = typer.Option(None, "--claim", metavar="ID", help="recompute one claim in detail"),
) -> None:
    """Recompute the claims in an EVIDENCE.md from their committed
    artifacts, for this repository or any other (`fireassay.evidence`).

    A claims file is ordinary Python that defines `CLAIMS`, a tuple of
    `fireassay.evidence.Claim`. `--check` recomputes every one and exits
    non-zero on the first mismatch; nothing is ever rewritten to match.
    Exactly one of --check / --markdown / --claim is required.
    """
    modes = sum([check, markdown, claim is not None])
    if modes != 1:
        typer.echo("evidence: pass exactly one of --check, --markdown, --claim ID")
        raise typer.Exit(code=1)
    try:
        loaded = load_claims(claims)
    except EvidenceError as exc:
        typer.echo(f"evidence: {exc}")
        raise typer.Exit(code=1) from exc
    style = default_style(str(claims))
    if markdown:
        typer.echo(render_evidence_markdown(loaded, style))
        return
    code = run_one_claim(loaded, claim) if claim is not None else run_check(loaded)
    if code:
        raise typer.Exit(code=code)


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
        # Same corpus identity `run` records, so a mutant run is comparable
        # with (and gate-checkable against) a `fireassay run` baseline over
        # the same corpus rather than tripping ENV_MISMATCH -- see
        # run_mutation's docstring.
        env_affects_results={"corpus_hash": corpus_hash(docs)},
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
    max_generation_failure_rate: float = typer.Option(
        DEFAULT_MAX_GENERATION_FAILURE_RATE,
        "--max-generation-failure-rate",
        help="abort if more than this fraction of generation attempts exhaust their retries",
    ),
) -> None:
    """Generate candidate questions from --corpus using --model, via
    Ollama (the only command permitted to call an LLM — M3-SPEC.md §1/§2).

    Every rejection (`GENERATION_FAILED`, `NOT_A_QUESTION`,
    `QUOTE_NOT_FOUND`, `QUOTE_AMBIGUOUS`) is recorded, not silently
    dropped — see `generate.pipeline` and `curate report`'s funnel. A few
    `GENERATION_FAILED` discards are normal (some `(chunk, target cell)`
    pairs are genuinely impossible, e.g. `comparative`/`hard` against a
    passage with nothing to compare); exceeding
    `--max-generation-failure-rate` aborts the run instead, on the theory
    that most attempts failing means something is actually wrong rather
    than a few unlucky pairings.

    **Safe to re-run into an existing --db (M3b-SPEC.md Part 1).** Any
    chunk --db already has at least one candidate for is skipped
    entirely, reported as `resumed=N chunks already present` — deleting
    the database first is no longer necessary, and doing so would throw
    away every candidate (and every already-paid-for cache hit) the
    earlier run produced.
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
    try:
        stats = generate_candidates(
            store, client, model_ref, docs, selected, n_per_chunk=n_per_chunk,
            batch_id=resolved_batch_id, max_generation_failure_rate=max_generation_failure_rate,
        )
    except GenerationFailureRateExceededError as exc:
        store.close()
        typer.echo(f"batch_id={resolved_batch_id}  aborted: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    store.close()
    typer.echo(
        f"batch_id={resolved_batch_id}  generated={stats.generated}  kept={stats.kept}  "
        f"generation_failed={stats.generation_failed}  not_a_question={stats.not_a_question}  "
        f"quote_not_found={stats.quote_not_found}  quote_ambiguous={stats.quote_ambiguous}  "
        f"resumed={stats.resumed} chunks already present"
    )


@app.command("filter")
def filter_candidates(
    batch: str = typer.Option(..., "--batch", help="a --batch-id from a prior `generate` run"),
    config: Path = typer.Option(..., "--config", help="configs/filter.yaml"),
    corpus: Path = typer.Option(
        ..., "--corpus", help="corpus JSONL file (required: the `unretrievable` stage builds a BM25 "
        "index over it, which M3-SPEC.md's CLI signature omits but the stage cannot run without)"
    ),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Run the deterministic filter pipeline (M3-SPEC.md §3) over every
    candidate generated in --batch, persisting a `filter_result` row for
    every candidate at every stage it reaches.

    The `unretrievable` stage's composition bias (under-represents
    questions needing semantic rather than lexical matching — see
    `filter.stages.check_unretrievable`'s docstring and the README) is
    not printed here; it is a property of the filter, not of one run of
    it, and belongs in documentation a reader consults once, not in every
    invocation's stdout.
    """
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


@curate_app.command("tui")
def curate_tui(
    curator: str = typer.Option(..., "--curator"),
    db: Path = typer.Option(..., "--db"),
    honeypot_rate: float = typer.Option(DEFAULT_HONEYPOT_RATE, "--honeypot-rate"),
    double_review_rate: float = typer.Option(DEFAULT_DOUBLE_REVIEW_RATE, "--double-review-rate"),
) -> None:
    """Launch the interactive curation TUI (M3b-SPEC.md Part 2).

    Drives the exact same `curate.serve.next_item`/`submit_decision` core
    as `curate next`/`curate submit` — the TUI adds no logic, only
    keystrokes; see `curate/tui_logic.py` and `curate/tui_session.py` for
    the decisions pulled out into plain, tested functions, and
    `curate/tui.py`'s module docstring for what could not be tested in
    this environment.

    `textual` is imported lazily, inside this function, rather than at
    module load time: every other `fireassay` command must keep working
    even before `pip install -e .` has picked up this milestone's one new
    runtime dependency.
    """
    from fireassay.curate.tui import run_curate_tui

    store = Store(db)
    store.migrate()
    try:
        run_curate_tui(store, curator, honeypot_rate=honeypot_rate, double_review_rate=double_review_rate)
    finally:
        store.close()


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


# -- items: item analysis for evaluation sets --------------------------------


def _load_matrix_file(path: Path) -> list[ItemResponses]:
    if path.suffix == ".csv":
        return load_responses_csv(path)
    if path.suffix in (".jsonl", ".ndjson"):
        return load_responses_jsonl(path)
    raise typer.BadParameter(f"--matrix must be .csv or .jsonl, got {path.suffix!r}")


def _load_labels_file(path: Path) -> list[ReviewLabel]:
    """`--labels` accepts either a hand-authored JSONL of `{review_id,
    verdict}`, or the CSV `items review build --worksheet` writes -- a
    filled-in worksheet is itself a valid labels file. CSV needs only
    `review_id`/`verdict` columns; any other column (question,
    reference_answer, evidence_quote) is ignored. A row with an empty or
    whitespace-only verdict is skipped entirely -- unlabelled, not a
    fourth verdict -- so it stays excluded from both `score_review`'s
    numerator and denominator, the same as a `review_id` with no matching
    label at all."""
    if path.suffix == ".csv":
        labels: list[ReviewLabel] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not {"review_id", "verdict"} <= set(reader.fieldnames):
                raise typer.BadParameter(
                    f"--labels: {path}: CSV must have 'review_id' and 'verdict' columns, "
                    f"got {reader.fieldnames!r}"
                )
            for line_no, row in enumerate(reader, start=2):  # header occupies line 1
                review_id = row["review_id"]
                verdict = (row.get("verdict") or "").strip()
                if not verdict:
                    continue
                # Validated explicitly, one literal at a time, rather than
                # passed straight through to `ReviewLabel`: pydantic checks
                # `verdict` against its Literal type at runtime regardless,
                # but mypy cannot narrow a CSV cell's `str` to that Literal
                # on its own, and a bare pydantic ValidationError would not
                # say which worksheet row a typo like "purged"/"ok" is on.
                if verdict == "keep":
                    labels.append(ReviewLabel(review_id=review_id, verdict="keep"))
                elif verdict == "rewrite":
                    labels.append(ReviewLabel(review_id=review_id, verdict="rewrite"))
                elif verdict == "purge":
                    labels.append(ReviewLabel(review_id=review_id, verdict="purge"))
                else:
                    raise typer.BadParameter(
                        f"--labels: {path}: line {line_no}: verdict {verdict!r} for "
                        f"review_id {review_id!r} is not one of 'keep', 'rewrite', 'purge'"
                    )
        return labels
    if path.suffix in (".jsonl", ".ndjson"):
        with open(path, encoding="utf-8") as f:
            return [ReviewLabel(**json.loads(line)) for line in f if line.strip()]
    raise typer.BadParameter(f"--labels must be .csv or .jsonl, got {path.suffix!r}")


def _load_flags_file(path: Path) -> list[str]:
    """`--flags` for `items detector-score`: a plain text file, one
    `item_id` per line -- deliberately not fireassay-specific, so any
    detector, in any language, can emit one. Blank lines and lines
    starting with `#` are ignored."""
    ids: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def _source_to_stratum(source: ReviewSource) -> Stratum:
    """`ReviewKeyEntry.source` has one more value (`"seeded"`) than
    `calibration.Stratum` -- the caller must have already dropped every
    `is_seeded` entry (Change 1's rule) before calling this."""
    if source == "flagged":
        return "flagged"
    if source == "unflagged":
        return "unflagged"
    if source == "calibration":
        return "calibration"
    raise AssertionError(
        f"_source_to_stratum: unexpected source {source!r} -- seeded entries must be "
        "skipped before this is called"
    )


@items_app.command("analyse")
def items_analyse(
    matrix: Path | None = typer.Option(
        None, "--matrix", help="CSV or JSONL response matrix (standalone mode -- no store touched)"
    ),
    meta: Path | None = typer.Option(None, "--meta", help="optional ItemMeta JSONL, keyed on item_id"),
    db: Path | None = typer.Option(None, "--db", help="fireassay store (in-project mode)"),
    suite: str | None = typer.Option(None, "--suite", help="name@version (in-project mode)"),
    metric: str | None = typer.Option(
        None, "--metric", help="in-project mode: score metric to threshold, e.g. retrieval.recall@5"
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", help="in-project mode: item is correct iff metric value > threshold"
    ),
    reliability_floor: float = typer.Option(0.5, "--reliability-floor"),
    n_splits: int = typer.Option(200, "--n-splits"),
    seed: int = typer.Option(0, "--seed"),
    validations: Path | None = typer.Option(
        None,
        "--validations",
        help=(
            "DetectorValidation JSONL (see `items detector-score --write-validation`) -- "
            "prints each detector's measured precision/recall and derived verdict before "
            "the item scorecard"
        ),
    ),
) -> None:
    """Item analysis over a response matrix, via either `--matrix`
    (standalone -- usable with no fireassay store at all) or `--db`/
    `--suite` (in-project, over a suite's own runs).

    The in-project path additionally **requires** `--metric`/`--threshold`
    with no default: fireassay's own runs record only continuous retrieval
    metrics, never a ready-made boolean "correct" -- see
    `items.adapters.store`'s module docstring for why no default is
    offered for either; inventing one here would be exactly the kind of
    unmeasured threshold this tool exists to refuse to produce.
    """
    if matrix is not None and db is not None:
        typer.echo("items analyse: pass either --matrix or --db, not both", err=True)
        raise typer.Exit(code=1)

    item_meta: dict[str, ItemMeta] | None
    if matrix is not None:
        responses = _load_matrix_file(matrix)
        item_meta = load_meta_jsonl(meta) if meta is not None else None
    elif db is not None:
        if suite is None or metric is None or threshold is None:
            typer.echo("items analyse --db requires --suite, --metric, and --threshold", err=True)
            raise typer.Exit(code=1)
        store = Store(db)
        name, _, version = suite.partition("@")
        suite_row = store.get_suite(name, version)
        runs = store.runs_for_suite(suite_row.id)
        responses = items_load_matrix(store, runs, metric=metric, threshold=threshold)
        item_meta = items_load_meta(store, suite_row.id)
        store.close()
    else:
        typer.echo("items analyse: one of --matrix or --db is required", err=True)
        raise typer.Exit(code=1)

    item_stats, panel_stats = run_item_analysis(
        responses, item_meta, reliability_floor=reliability_floor, n_splits=n_splits, seed=seed
    )
    validation_records = load_validations(validations) if validations is not None else None
    render_items_analysis(item_stats, panel_stats, validation_records)


@items_review_app.command("build")
def items_review_build(
    matrix: Path | None = typer.Option(
        None, "--matrix", help="CSV or JSONL response matrix (standalone mode -- no store touched)"
    ),
    meta: Path | None = typer.Option(
        None,
        "--meta",
        help=(
            "ItemMeta JSONL, keyed on item_id -- REQUIRED in standalone mode (unlike "
            "`items analyse`): a review batch with no question or reference answer text is "
            "nothing a human can review"
        ),
    ),
    db: Path | None = typer.Option(None, "--db", help="fireassay store (in-project mode)"),
    suite: str | None = typer.Option(None, "--suite", help="name@version (in-project mode)"),
    metric: str | None = typer.Option(
        None, "--metric", help="in-project mode: see `items analyse --db`'s docstring"
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", help="in-project mode: item is correct iff metric value > threshold"
    ),
    out: Path = typer.Option(..., "--out", help="ReviewItem batch, JSONL -- what a reviewer opens"),
    key: Path = typer.Option(..., "--key", help="ground-truth key, JSON -- reviewer never sees this"),
    worksheet: Path | None = typer.Option(
        None,
        "--worksheet",
        help=(
            "optional CSV a human can fill in directly -- columns review_id,verdict,question,"
            "reference_answer,evidence_quote, verdict left empty"
        ),
    ),
    n_flagged: int = typer.Option(..., "--n-flagged", help="max mislabel_suspect items to include"),
    n_unflagged: int = typer.Option(..., "--n-unflagged", help="max non-flagged items to include"),
    n_calibration: int = typer.Option(
        0, "--n-calibration", help="items sampled uniformly from the whole pool, for calibration"
    ),
    n_seeded_per_kind: int = typer.Option(
        0, "--n-seeded-per-kind", help="known-bad seeded items per corruption kind; 0 disables seeding"
    ),
    exclude: Path | None = typer.Option(
        None,
        "--exclude",
        help=(
            "plain text, one item_id per line (same format as --flags) -- excluded from "
            "every pool, including seeded items whose source item_id matches, for a later "
            "review pass drawing only from items nobody has labelled yet"
        ),
    ),
    reliability_floor: float = typer.Option(0.5, "--reliability-floor"),
    n_splits: int = typer.Option(200, "--n-splits"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Build a blind review batch (M-ITEMS-SPEC.md §3), via either
    `--matrix`/`--meta` (standalone -- usable with no fireassay store at
    all) or `--db`/`--suite`/`--metric`/`--threshold` (in-project, over a
    suite's own runs): flagged (`mislabel_suspect`) items, unflagged items
    sampled from every other classification, calibration items sampled
    uniformly from the whole pool (if `--n-calibration` > 0), and (if
    `--n-seeded-per-kind` > 0) deterministically corrupted known-bad
    items, mixed into one shuffled batch with no `item_id` appearing
    twice (`items.review._plan_batch`'s claim order). `--out` carries
    only question/reference/evidence -- no classification, statistic,
    source, or seeded flag (`items.review`'s leak rule); `--key` is the
    ground-truth mapping and must be kept away from the reviewer.
    `--worksheet`, if given, writes the same batch as a CSV a human can
    fill `verdict` in directly (`review_id,verdict,question,
    reference_answer,evidence_quote`) -- see `items review score`'s
    `--labels` for reading it back. `--exclude`, if given, removes those
    item_ids from every pool, including `seeded` (`_plan_batch`'s
    docstring on why the seeded case is the one that matters) -- pass a
    prior pass's reviewed item_ids to accumulate the calibration set
    (handbook §8) across passes without re-drawing an already-labelled
    item.
    """
    if matrix is not None and db is not None:
        typer.echo("items review build: pass either --matrix or --db, not both", err=True)
        raise typer.Exit(code=1)

    item_meta: dict[str, ItemMeta]
    if matrix is not None:
        if meta is None:
            typer.echo(
                "items review build --matrix requires --meta -- a review batch with no "
                "question or reference answer text is nothing a human can review",
                err=True,
            )
            raise typer.Exit(code=1)
        responses = _load_matrix_file(matrix)
        item_meta = load_meta_jsonl(meta)
    elif db is not None:
        if suite is None or metric is None or threshold is None:
            typer.echo("items review build --db requires --suite, --metric, and --threshold", err=True)
            raise typer.Exit(code=1)
        store = Store(db)
        name, _, version = suite.partition("@")
        suite_row = store.get_suite(name, version)
        runs = store.runs_for_suite(suite_row.id)
        responses = items_load_matrix(store, runs, metric=metric, threshold=threshold)
        item_meta = items_load_meta(store, suite_row.id)
        store.close()
    else:
        typer.echo("items review build: one of --matrix or --db is required", err=True)
        raise typer.Exit(code=1)

    item_stats, _panel_stats = run_item_analysis(
        responses, item_meta, reliability_floor=reliability_floor, n_splits=n_splits, seed=seed
    )
    seeded_items = (
        seed_batch(list(item_meta.values()), n_per_kind=n_seeded_per_kind, seed=seed)
        if n_seeded_per_kind > 0
        else []
    )
    exclude_ids = _load_flags_file(exclude) if exclude is not None else []
    exclude_set = frozenset(exclude_ids)
    batch = build_review_batch(
        item_stats, item_meta, n_flagged=n_flagged, n_unflagged=n_unflagged,
        n_calibration=n_calibration, seeded=seeded_items, seed=seed, exclude=exclude_set,
    )
    key_entries = build_review_key(
        item_stats, item_meta, n_flagged=n_flagged, n_unflagged=n_unflagged,
        n_calibration=n_calibration, seeded=seeded_items, seed=seed, exclude=exclude_set,
    )

    with open(out, "w", encoding="utf-8") as f:
        for review_item in batch:
            f.write(review_item.model_dump_json())
            f.write("\n")
    with open(key, "w", encoding="utf-8") as f:
        json.dump([entry.model_dump() for entry in key_entries], f, indent=2)

    if worksheet is not None:
        with open(worksheet, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["review_id", "verdict", "question", "reference_answer", "evidence_quote"])
            for review_item in batch:
                writer.writerow(
                    [
                        review_item.review_id,
                        "",
                        review_item.question or "",
                        review_item.reference_answer or "",
                        review_item.evidence_quote or "",
                    ]
                )

    flagged_n = sum(1 for e in key_entries if e.source == "flagged")
    seeded_n = sum(1 for e in key_entries if e.source == "seeded")
    unflagged_n = sum(1 for e in key_entries if e.source == "unflagged")
    calibration_n = sum(1 for e in key_entries if e.source == "calibration")
    seeded_dropped = len(seeded_items) - seeded_n
    seeded_text = f"seeded {seeded_n}"
    if seeded_dropped > 0:
        seeded_text += f" ({seeded_dropped} dropped: source item already in the deck)"
    exclude_text = f"\n  excluded {len(exclude_ids)} already-reviewed item(s)" if exclude is not None else ""

    typer.echo(
        f"wrote {len(batch)} review item(s) to {out}; key -> {key}\n"
        f"  flagged {flagged_n}  {seeded_text}\n"
        f"  unflagged {unflagged_n}  calibration {calibration_n}"
        f"{exclude_text}\n"
        "(seeded recall is an UPPER BOUND -- seeded flaws may be easier to spot than natural ones)"
    )


@items_review_app.command("score")
def items_review_score(
    labels: Path = typer.Option(
        ..., "--labels", help="reviewer verdicts -- .jsonl of {review_id, verdict}, or a filled-in .csv"
    ),
    key: Path = typer.Option(..., "--key", help="the ground-truth key from `items review build --key`"),
) -> None:
    """Score a completed blind review against its key (M-ITEMS-SPEC.md
    §3): precision from the flagged items, recall (an upper bound) from
    the seeded items, each with a 95% confidence interval."""
    label_list = _load_labels_file(labels)
    with open(key, encoding="utf-8") as f:
        key_list = [ReviewKeyEntry(**entry) for entry in json.load(f)]
    score = score_review(label_list, key_list)
    render_items_score(score)


@items_app.command("detector-score")
def items_detector_score(
    flags: Path = typer.Option(
        ..., "--flags", help="plain text, one item_id per line -- '#' comments and blank lines ignored"
    ),
    calibration: Path = typer.Option(
        ..., "--calibration", help="CalibrationLabel JSONL (see `items review export-calibration`)"
    ),
    bad_verdicts: str = typer.Option(
        "purge", "--bad-verdicts", help="comma-separated verdicts counted as bad, e.g. 'purge,rewrite'"
    ),
    reviewer: str | None = typer.Option(
        None, "--reviewer", help="restrict to one reviewer's labels; omit to use all reviewers' labels"
    ),
    min_denominator: int = typer.Option(
        10, "--min-denominator", help="below this n, an Estimate's verdict is 'too_few_labels'"
    ),
    pool_size: int | None = typer.Option(
        None, "--pool-size", help="items in the suite the labels were drawn from (with --flagged-size)"
    ),
    flagged_size: int | None = typer.Option(
        None, "--flagged-size", help="size of the flagged stratum in the suite (with --pool-size)"
    ),
    write_validation: Path | None = typer.Option(
        None,
        "--write-validation",
        help=(
            "append a DetectorValidation record built from this measurement to this JSONL "
            "path (with --detector) -- see `items analyse --validations`"
        ),
    ),
    detector: str | None = typer.Option(
        None,
        "--detector",
        help="name of the detector being scored, e.g. 'mislabel_suspect' (with --write-validation)",
    ),
) -> None:
    """Score any detector's flagged-items list against accumulated human
    calibration labels (`items.calibration.evaluate_detector`) -- works for
    any detector that can emit a plain-text list of item_ids, including one
    fireassay never built. `--bad-verdicts` deliberately has no canonical
    default beyond the strict `purge` reading -- pass `purge,rewrite` for
    the lenient one; which matters is the reader's call.

    `--pool-size`/`--flagged-size` enable the stratified estimator for
    recall/fpr/base_rate when the calibration stratum is structurally
    disjoint from the detector's flagged items (see
    `items.calibration`'s module docstring on the disjoint-stratum trap)
    -- both or neither, never one alone.

    `--write-validation`/`--detector` (both or neither) turn this
    measurement into a durable `items.validation.DetectorValidation`
    record -- this is how such a record gets created by running the
    measurement, not by typing numbers in. Its `verdict` is derived from
    `precision`/`base_rate` by `items.validation.write_validations`, never
    taken from anywhere in this command (see that module's docstring)."""
    if (pool_size is None) != (flagged_size is None):
        typer.echo(
            "items detector-score: --pool-size and --flagged-size must be given together, "
            f"or neither (got --pool-size={pool_size!r}, --flagged-size={flagged_size!r})",
            err=True,
        )
        raise typer.Exit(code=1)
    if (write_validation is None) != (detector is None):
        typer.echo(
            "items detector-score: --write-validation and --detector must be given together, "
            f"or neither (got --write-validation={write_validation!r}, --detector={detector!r})",
            err=True,
        )
        raise typer.Exit(code=1)
    design: SamplingDesign | None = None
    if pool_size is not None and flagged_size is not None:
        design = SamplingDesign(pool_size=pool_size, flagged_size=flagged_size)

    flagged_ids = _load_flags_file(flags)
    label_list = load_calibration_jsonl(calibration)
    bad_verdict_set = frozenset(v.strip() for v in bad_verdicts.split(",") if v.strip())
    evaluation = evaluate_detector(
        flagged_ids,
        label_list,
        bad_verdicts=bad_verdict_set,
        min_denominator=min_denominator,
        reviewer=reviewer,
        design=design,
    )
    render_detector_evaluation(evaluation)

    if write_validation is not None and detector is not None:
        base_rate_value = evaluation.base_rate.value
        if base_rate_value is None:
            typer.echo(
                "items detector-score --write-validation: base_rate is unmeasured "
                f"(verdict={evaluation.base_rate.verdict!r}) -- refusing to write a "
                "DetectorValidation record with a fabricated base_rate; see "
                "items.calibration's disjoint-stratum trap",
                err=True,
            )
            raise typer.Exit(code=1)
        record = DetectorValidation(
            detector=detector,
            measured_on=date.today().isoformat(),
            labels=(
                f"{evaluation.n_labels} calibration label(s) from {calibration} "
                f"(stratum_estimator={evaluation.stratum_estimator})"
            ),
            base_rate=base_rate_value,
            precision=evaluation.precision,
            recall=evaluation.recall,
            note=None,
            verdict=derive_verdict(evaluation.precision, base_rate_value),
        )
        n_written = write_validations([record], write_validation)
        typer.echo(
            f"wrote {n_written} DetectorValidation record to {write_validation}: "
            f"detector={record.detector}  verdict={record.verdict}"
        )


@items_review_app.command("export-calibration")
def items_review_export_calibration(
    labels: Path = typer.Option(
        ..., "--labels", help="reviewer verdicts -- .jsonl of {review_id, verdict}, or a filled-in .csv"
    ),
    key: Path = typer.Option(..., "--key", help="the ground-truth key from `items review build --key`"),
    reviewer: str = typer.Option(..., "--reviewer", help="name of the reviewer who produced --labels"),
    batch_id: str = typer.Option(..., "--batch-id", help="identifier for this review batch/pass"),
    out: Path = typer.Option(
        ..., "--out", help="CalibrationLabel JSONL -- appended, never rewritten (accumulates across passes)"
    ),
) -> None:
    """Turn one completed blind-review pass into `CalibrationLabel`s any
    detector can later be scored against (`fireassay items detector-score`)
    -- the export boundary that enforces `items.calibration`'s rule that a
    seeded item must never become a `CalibrationLabel`: every `is_seeded`
    key entry is dropped here, counted but never exported."""
    label_list = _load_labels_file(labels)
    with open(key, encoding="utf-8") as f:
        key_list = [ReviewKeyEntry(**entry) for entry in json.load(f)]
    key_by_id = {entry.review_id: entry for entry in key_list}

    calibration_labels: list[CalibrationLabel] = []
    skipped_seeded = 0
    for label in label_list:
        entry = key_by_id.get(label.review_id)
        if entry is None:
            typer.echo(
                f"items review export-calibration: review_id {label.review_id!r} (from {labels}) "
                f"is absent from --key {key}",
                err=True,
            )
            raise typer.Exit(code=1)
        if entry.is_seeded:
            skipped_seeded += 1
            continue
        calibration_labels.append(
            CalibrationLabel(
                item_id=entry.item_id,
                verdict=label.verdict,
                stratum=_source_to_stratum(entry.source),
                reviewer=reviewer,
                batch_id=batch_id,
            )
        )

    n_written = append_calibration_jsonl(calibration_labels, out)

    per_stratum: dict[str, int] = {}
    for cl in calibration_labels:
        per_stratum[cl.stratum] = per_stratum.get(cl.stratum, 0) + 1

    typer.echo(
        f"exported {n_written} calibration label(s) to {out} (skipped {skipped_seeded} seeded)\n"
        f"  by stratum: {per_stratum}"
    )


if __name__ == "__main__":
    app()
