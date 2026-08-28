from __future__ import annotations

import pytest

from fireassay.compare import leaderboard
from fireassay.integrity import ComparisonRefusedError
from fireassay.models import Question, Run, Score, SystemOutput
from fireassay.store.db import Store


def _question(text: str, qtype: str = "factual", difficulty: str = "easy") -> Question:
    return Question(text=text, qtype=qtype, difficulty=difficulty, provenance="synthetic")


def _seed_run(
    store: Store,
    questions: list[Question],
    scores_by_question: dict[str, list[Score]],
    *,
    suite_name: str = "kb",
    suite_version: str = "1.0.0",
    config_spec: dict[str, object] | None = None,
) -> Run:
    store.put_questions(questions)
    suite = store.freeze_suite(suite_name, suite_version, [q.id for q in questions])
    config = store.put_config(config_spec or {"x": 1})
    env = {"python": "3.12", "fireassay": "0.1.0", "affects_results": {}}
    run = store.start_run(suite, config, env)
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 5.0})
    for q in questions:
        store.put_result(run.id, q.id, output, 0.0)
        scores = scores_by_question.get(q.id, [])
        if scores:
            store.put_scores(run.id, q.id, scores)
    store.finish_run(run.id, "complete")
    return store.get_run(run.id)


def test_means_skip_missing_scores_and_n_differs_per_metric(store: Store) -> None:
    q1 = _question("q1", qtype="factual")
    q2 = _question("q2", qtype="unanswerable")
    scores_by_question = {
        q1.id: [Score(metric="retrieval.recall@5", value=1.0, scorer="retrieval@1.0.0")],
        q2.id: [Score(metric="abstention.correct", value=1.0, scorer="abstention@1.0.0")],
    }
    run = _seed_run(store, [q1, q2], scores_by_question)

    board = leaderboard(store, [run])
    entry = board.entries[0]
    by_metric = {m.metric: m for m in entry.metrics}

    # Each metric only has one question with a recorded score for it, even
    # though the run has two questions total -- the other question's
    # "missing" score for that metric was never imputed as 0.
    assert by_metric["retrieval.recall@5"].n == 1
    assert by_metric["retrieval.recall@5"].mean == pytest.approx(1.0)
    assert by_metric["abstention.correct"].n == 1
    assert by_metric["abstention.correct"].mean == pytest.approx(1.0)


def test_applicable_n_differs_from_n_and_is_surfaced(store: Store) -> None:
    """A metric measured on a strict subset of the population it could
    have applied to must expose both `n` and `applicable_n` -- collapsing
    to a bare mean is exactly the shrinking-subset failure mode this field
    exists to catch."""
    q1 = _question("q1")
    q2 = _question("q2")
    q3 = _question("q3")
    scores_by_question = {
        # Every question gets a latency score (as in a real run).
        q1.id: [Score(metric="latency.total_ms", value=10.0, scorer="latency@1.0.0")],
        q2.id: [Score(metric="latency.total_ms", value=20.0, scorer="latency@1.0.0")],
        q3.id: [Score(metric="latency.total_ms", value=30.0, scorer="latency@1.0.0")],
    }
    # Only q1 additionally gets a retrieval score (e.g. only q1 has gold spans).
    scores_by_question[q1.id].append(Score(metric="retrieval.recall@5", value=1.0, scorer="retrieval@1.0.0"))

    run = _seed_run(store, [q1, q2, q3], scores_by_question)
    board = leaderboard(store, [run])
    by_metric = {m.metric: m for m in board.entries[0].metrics}

    assert by_metric["latency.total_ms"].n == 3
    assert by_metric["latency.total_ms"].applicable_n == 3

    assert by_metric["retrieval.recall@5"].n == 1
    assert by_metric["retrieval.recall@5"].applicable_n == 3
    assert by_metric["retrieval.recall@5"].n < by_metric["retrieval.recall@5"].applicable_n


def test_latency_metrics_get_p50_p95_other_metrics_do_not(store: Store) -> None:
    q1 = _question("q1")
    q2 = _question("q2")
    scores_by_question = {
        q1.id: [
            Score(metric="latency.total_ms", value=100.0, scorer="latency@1.0.0"),
            Score(metric="cost.usd", value=0.5, scorer="cost@1.0.0"),
        ],
        q2.id: [
            Score(metric="latency.total_ms", value=200.0, scorer="latency@1.0.0"),
            Score(metric="cost.usd", value=1.5, scorer="cost@1.0.0"),
        ],
    }
    run = _seed_run(store, [q1, q2], scores_by_question)

    board = leaderboard(store, [run])
    by_metric = {m.metric: m for m in board.entries[0].metrics}

    assert by_metric["latency.total_ms"].n == 2
    assert by_metric["latency.total_ms"].p50 is not None
    assert by_metric["latency.total_ms"].p95 is not None

    assert by_metric["cost.usd"].n == 2
    assert by_metric["cost.usd"].p50 is None
    assert by_metric["cost.usd"].p95 is None


def test_forced_leaderboard_is_flagged(store: Store) -> None:
    q1 = _question("q1 in suite A")
    run_a = _seed_run(
        store,
        [q1],
        {q1.id: [Score(metric="m", value=1.0, scorer="s@1.0.0")]},
        suite_name="kb-a",
        suite_version="1.0.0",
    )
    q2 = _question("q2 in suite B, entirely different content")
    run_b = _seed_run(
        store,
        [q2],
        {q2.id: [Score(metric="m", value=1.0, scorer="s@1.0.0")]},
        suite_name="kb-b",
        suite_version="1.0.0",
    )

    with pytest.raises(ComparisonRefusedError):
        leaderboard(store, [run_a, run_b])

    board = leaderboard(store, [run_a, run_b], force=True)
    assert board.report.forced is True
    assert board.report.comparable is False
    assert any(r.code == "SUITE_MISMATCH" for r in board.report.reasons)
    # Even when forced, entries are still built for both runs.
    assert {e.run_id for e in board.entries} == {run_a.id, run_b.id}


def test_split_by_provenance_produces_separate_entries(store: Store) -> None:
    q_synth = Question(
        text="synthetic question", qtype="factual", difficulty="easy", provenance="synthetic"
    )
    q_real = Question(
        text="real traffic question", qtype="factual", difficulty="easy", provenance="real_traffic"
    )
    scores_by_question = {
        q_synth.id: [Score(metric="m", value=1.0, scorer="s@1.0.0")],
        q_real.id: [Score(metric="m", value=0.0, scorer="s@1.0.0")],
    }
    run = _seed_run(store, [q_synth, q_real], scores_by_question)

    board = leaderboard(store, [run], split_by_provenance=True)
    provenances = {e.slice.get("provenance") for e in board.entries}
    assert provenances == {"synthetic", "real_traffic"}
    for entry in board.entries:
        assert entry.metrics[0].n == 1
        assert entry.metrics[0].applicable_n == 1


def test_split_by_qtype_and_difficulty(store: Store) -> None:
    q_easy_factual = _question("q1", qtype="factual", difficulty="easy")
    q_hard_multi_hop = _question("q2", qtype="multi_hop", difficulty="hard")
    scores_by_question = {
        q_easy_factual.id: [Score(metric="m", value=1.0, scorer="s@1.0.0")],
        q_hard_multi_hop.id: [Score(metric="m", value=0.0, scorer="s@1.0.0")],
    }
    run = _seed_run(store, [q_easy_factual, q_hard_multi_hop], scores_by_question)

    board = leaderboard(store, [run], split_by=["qtype", "difficulty"])
    slices = {(e.slice["qtype"], e.slice["difficulty"]) for e in board.entries}
    assert slices == {("factual", "easy"), ("multi_hop", "hard")}

    by_slice = {(e.slice["qtype"], e.slice["difficulty"]): e for e in board.entries}
    # The hard multi_hop slice's mean is visibly 0.0, not averaged away
    # into an overall mean of 0.5.
    assert by_slice[("multi_hop", "hard")].metrics[0].mean == pytest.approx(0.0)
    assert by_slice[("factual", "easy")].metrics[0].mean == pytest.approx(1.0)


def test_split_by_rejects_unknown_dimension(store: Store) -> None:
    q1 = _question("q1")
    run = _seed_run(store, [q1], {q1.id: [Score(metric="m", value=1.0, scorer="s@1.0.0")]})
    with pytest.raises(ValueError, match="split_by"):
        leaderboard(store, [run], split_by=["not_a_real_dimension"])
