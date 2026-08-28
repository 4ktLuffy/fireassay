"""fireassay command-line interface (M1 subset).

`compare` and `diff` exit **2** on `ComparisonRefusedError` — distinct from
Click/Typer's own usage-error exit code so a CI script can tell "the suite
was mistyped" apart from "the tool refused an unsound comparison" (see
M1-SPEC.md §10). All other unhandled errors exit 1, Typer's default.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from fireassay.compare import leaderboard
from fireassay.config import MatrixSpec
from fireassay.integrity import ComparisonRefusedError
from fireassay.models import EvidenceSpan, Question
from fireassay.report.text import render_leaderboard, render_refusal
from fireassay.runner import run_matrix
from fireassay.score.abstention import AbstentionScorer
from fireassay.score.base import Scorer, ScoringContext
from fireassay.score.cost import CostScorer
from fireassay.score.latency import LatencyScorer
from fireassay.score.policy import PolicyScorer, load_policy_rules
from fireassay.score.retrieval import RetrievalScorer
from fireassay.store.db import Store, SuiteExistsError
from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, chunk_corpus, corpus_hash, load_corpus

app = typer.Typer(add_completion=False, no_args_is_help=True)
questions_app = typer.Typer(no_args_is_help=True)
suite_app = typer.Typer(no_args_is_help=True)
app.add_typer(questions_app, name="questions")
app.add_typer(suite_app, name="suite")


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


@suite_app.command("freeze")
def suite_freeze(
    name: str = typer.Option(..., "--name"),
    version: str = typer.Option(..., "--version"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    """Freeze every question currently in the store into a named,
    versioned suite."""
    store = Store(db)
    store.migrate()
    question_ids = store.all_question_ids()
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


if __name__ == "__main__":
    app()
