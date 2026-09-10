"""`fireassay gate` end to end, through the CLI, on the fixture suite.

Every exit code the command documents is exercised here against runs the
CLI itself produced (`run` for the baseline, `mutate` for the planted
regressions), not against hand-built score vectors -- `tests/test_gate.py`
already covers `evaluate_gate`'s arithmetic on synthetic data; this file
covers the wiring: run lookup, comparability refusal, per-question
alignment, persistence, and the exit code a CI job actually sees.

The fixture suite has 20 questions, 14 with gold spans, so every
retrieval metric here is paired over n=14. That small n is not
incidental: it is what makes the "threshold below MDE" refusal
reachable with a real regression (`corrupt_query` at 90% drops
`retrieval.recall@5` by ~0.39, and the suite resolves only ~0.35), which
is the M5 thesis in one sentence -- a gate that cannot resolve its own
threshold says so instead of guessing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from fireassay.cli import GATE_EXIT_ALREADY_CHECKED, GATE_EXIT_BELOW_MDE, GATE_EXIT_BLOCKED, app
from fireassay.store.db import Store

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS = str(FIXTURES_DIR / "corpus.jsonl")

runner = CliRunner()


def _invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


@pytest.fixture(scope="module")
def gated_db(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """One database, built once per module through the CLI: the fixture
    suite, a two-config matrix (top_k 5 and 3), and a mutation run over
    the top_k=5 config with two operators -- a subtle regression
    (`corrupt_query` 90%) and a total one (`drop_results` 4 of 5).
    Returns the run ids the tests need."""
    tmp_path = tmp_path_factory.mktemp("gate")
    db = tmp_path / "gate.db"
    assert _invoke("init", "--db", str(db)).exit_code == 0
    questions = str(FIXTURES_DIR / "questions.jsonl")
    assert _invoke("questions", "import", questions, "--db", str(db)).exit_code == 0
    assert _invoke("suite", "freeze", "--name", "kb", "--version", "1.0.0", "--db", str(db)).exit_code == 0

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text('suite: kb@1.0.0\naxes:\n  top_k: ["5", "3"]\n', encoding="utf-8")
    result = _invoke("run", "--matrix", str(matrix), "--corpus", CORPUS, "--db", str(db))
    assert result.exit_code == 0, result.stdout

    store = Store(db)
    suite = store.get_suite("kb", "1.0.0")
    runs = store.runs_for_suite(suite.id)
    by_top_k = {store.get_config(r.config_id).spec["top_k"]: r for r in runs}
    base_run = by_top_k["5"]
    other_run = by_top_k["3"]
    store.close()

    operators = tmp_path / "mutants.yaml"
    operators.write_text(
        "detector:\n  metric: retrieval.recall@5\n  max_drop: 0.05\n"
        "operators:\n  - kind: corrupt_query\n    pct: 0.9\n  - kind: drop_results\n    n: 4\n",
        encoding="utf-8",
    )
    result = _invoke(
        "mutate", "--suite", "kb@1.0.0", "--config", base_run.config_id,
        "--operators", str(operators), "--corpus", CORPUS, "--db", str(db),
    )
    assert result.exit_code == 0, result.stdout
    mutation_run_id = result.stdout.split("mutation_run_id=")[1].split()[0]

    store = Store(db)
    mutants = {m.operator: m.mutant_run_id for m in store.get_mutants(mutation_run_id)}
    store.close()

    return {
        "db": str(db),
        "base": base_run.id,
        "other": other_run.id,
        "subtle": mutants["corrupt_query"],
        "total": mutants["drop_results"],
    }


def _gate(ids: dict[str, str], head: str, *metrics: str, extra: tuple[str, ...] = ()) -> Result:
    args = ["gate", "--db", ids["db"], "--base", ids["base"], "--head", head, "--b", "2000"]
    for m in metrics:
        args += ["--metric", m]
    return runner.invoke(app, args + list(extra))


def test_mutant_runs_are_comparable_with_a_run_baseline(gated_db: dict[str, str]) -> None:
    """The `mutate` fix this milestone needed: a mutant run records the same
    `corpus_hash` a `run` baseline does, so the gate can set them side by
    side instead of refusing with ENV_MISMATCH."""
    store = Store(Path(gated_db["db"]))
    base_env = store.get_run(gated_db["base"]).env_json["affects_results"]
    mutant_env = store.get_run(gated_db["total"]).env_json["affects_results"]
    store.close()
    assert isinstance(base_env, dict) and "corpus_hash" in base_env
    assert mutant_env == base_env


def test_planted_total_regression_is_blocked_and_persisted(gated_db: dict[str, str]) -> None:
    result = _gate(gated_db, gated_db["total"], "retrieval.recall@5=0.5", "retrieval.mrr=0.5")
    assert result.exit_code == GATE_EXIT_BLOCKED, result.stdout
    assert "GATE BLOCKED" in result.stdout

    store = Store(Path(gated_db["db"]))
    rows = store.get_gate_checks(gated_db["base"], gated_db["total"])
    store.close()
    assert [r.metric for r in rows] == ["retrieval.mrr", "retrieval.recall@5"]
    assert all(r.verdict == "block" for r in rows)
    assert all(r.n_items == 14 for r in rows)
    by_metric = {r.metric: r for r in rows}
    assert by_metric["retrieval.recall@5"].base_mean == pytest.approx(1.0)
    assert by_metric["retrieval.recall@5"].head_mean == pytest.approx(0.0)


def test_same_pair_cannot_be_gate_checked_twice(gated_db: dict[str, str]) -> None:
    """Depends on the previous test having persisted -- and that is the
    point: the second attempt on the same (base, head, metric) is refused
    by the schema, exit 6."""
    result = _gate(gated_db, gated_db["total"], "retrieval.recall@5=0.5")
    assert result.exit_code == GATE_EXIT_ALREADY_CHECKED, result.stdout

    # --no-persist recomputes without touching the store, so it still blocks.
    result = _gate(gated_db, gated_db["total"], "retrieval.recall@5=0.5", extra=("--no-persist",))
    assert result.exit_code == GATE_EXIT_BLOCKED, result.stdout
    store = Store(Path(gated_db["db"]))
    assert len(store.get_gate_checks(gated_db["base"], gated_db["total"])) == 2
    store.close()


def test_real_regression_below_mde_is_refused_not_passed(gated_db: dict[str, str]) -> None:
    """`corrupt_query` at 90% drops recall@5 by ~0.39 on this suite. A
    threshold of 0.3 is below what n=14 resolves (~0.35), so the gate
    refuses with exit 5 -- it does not pass, and it does not block."""
    result = _gate(gated_db, gated_db["subtle"], "retrieval.recall@5=0.3", extra=("--no-persist",))
    assert result.exit_code == GATE_EXIT_BELOW_MDE, result.stdout
    assert "GATE REFUSED" in result.stdout
    assert "minimum detectable effect" in result.stdout


def test_benign_config_change_passes(gated_db: dict[str, str]) -> None:
    """top_k=3 against top_k=5 on this fixture: recall@5 moves by at most
    one question (0.93 vs 1.0). Not a threshold-sized regression, not
    significant -- exit 0."""
    result = _gate(gated_db, gated_db["other"], "retrieval.recall@5=0.5", extra=("--no-persist",))
    assert result.exit_code == 0, result.stdout
    assert "GATE PASSED" in result.stdout


def test_unsound_comparison_is_refused_with_exit_2(gated_db: dict[str, str], tmp_path: Path) -> None:
    """A run whose environment differs in an `affects_results` field is
    refused before any statistics run -- the same refusal `compare`/`diff`
    give, same exit code."""
    store = Store(Path(gated_db["db"]))
    suite = store.get_suite("kb", "1.0.0")
    config = store.get_config(store.get_run(gated_db["base"]).config_id)
    foreign = store.start_run(
        suite, config, {"python": "3.12", "fireassay": "0.1.0", "affects_results": {"corpus_hash": "other"}}
    )
    store.finish_run(foreign.id, "complete")
    store.close()

    result = _gate(gated_db, foreign.id, "retrieval.recall@5=0.5", extra=("--no-persist",))
    assert result.exit_code == 2, result.stdout
    assert "ENV_MISMATCH" in result.stdout
    assert "EMPTY_RUN" in result.stdout


def test_metric_spec_parsing_rejects_bad_input(gated_db: dict[str, str]) -> None:
    for bad in ("retrieval.recall@5", "retrieval.recall@5=abc", "retrieval.recall@5=0", "x=0.5:higher"):
        result = _gate(gated_db, gated_db["other"], bad, extra=("--no-persist",))
        assert result.exit_code == 2, bad  # typer BadParameter -> usage error exit 2
        assert "--metric" in result.output


def test_metric_absent_from_both_runs_is_a_usage_error(gated_db: dict[str, str]) -> None:
    result = _gate(gated_db, gated_db["other"], "no.such.metric=0.5", extra=("--no-persist",))
    assert result.exit_code == 1, result.stdout
    assert "no question is scored" in result.stdout
