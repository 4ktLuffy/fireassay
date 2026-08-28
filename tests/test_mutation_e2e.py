from __future__ import annotations

import pytest

from fireassay.mutation.detector import ThresholdDetector
from fireassay.mutation.operators import (
    CorruptQueryOperator,
    DropResultsOperator,
    ShuffleTopkOperator,
    SwapRankingOperator,
    TruncateTopkOperator,
)
from fireassay.mutation.run import run_mutation
from fireassay.store.db import Store
from helpers import (
    bm25_system_factory,
    load_fixture_docs,
    scoring_ctx_factory,
    setup_suite,
    standard_scorers,
)


def test_mutation_e2e_baseline_five_mutants_detector_score(store: Store) -> None:
    suite, questions = setup_suite(store)
    docs = load_fixture_docs()
    config = store.put_config({"top_k": 5})
    scoring_ctx = scoring_ctx_factory(config.spec)
    scorers = standard_scorers()
    operators = [
        DropResultsOperator(n=2),
        TruncateTopkOperator(k=1),
        ShuffleTopkOperator(),
        CorruptQueryOperator(pct=0.9),
        SwapRankingOperator(),
    ]
    detector = ThresholdDetector(metric="retrieval.recall@5", max_drop=0.05)

    result = run_mutation(
        store, suite, config, questions, docs, bm25_system_factory, scorers, scoring_ctx, operators, detector
    )

    assert result.total == 5
    assert result.detector == "threshold@1.0.0"
    assert 0.0 <= result.score <= 1.0
    assert result.killed == sum(1 for m in result.mutants if m.killed and not m.equivalent)
    assert result.equivalent + (result.total - result.equivalent) == result.total

    # Every mutant produced a real, distinct persisted run.
    mutant_run_ids = {m.mutant_run_id for m in result.mutants}
    assert len(mutant_run_ids) == 5
    for run_id in mutant_run_ids:
        run = store.get_run(run_id)
        assert run.status == "complete"
        assert run.result_count == len(questions)

    # truncate_topk(k=1) against a top_k=5 baseline should plausibly
    # degrade recall for at least some questions and not be trivially
    # equivalent (k=1 < 5, and the baseline retrieves > 1 chunk for at
    # least one question in this fixture).
    truncate_result = next(m for m in result.mutants if m.operator == "truncate_topk")
    assert truncate_result.equivalent is False

    # Survivors, if any, are listed first by sorted_for_report.
    ordered = result.sorted_for_report()
    equivalent_seen = False
    for m in ordered:
        if m.equivalent:
            equivalent_seen = True
        elif equivalent_seen:
            pytest.fail("a non-equivalent mutant appeared after an equivalent one")


def test_mutation_e2e_equivalent_mutants_are_excluded_and_justified(store: Store) -> None:
    suite, questions = setup_suite(store)
    docs = load_fixture_docs()
    config = store.put_config({"top_k": 5})
    scoring_ctx = scoring_ctx_factory(config.spec)
    scorers = standard_scorers()
    operators = [DropResultsOperator(n=0), CorruptQueryOperator(pct=0.0)]
    detector = ThresholdDetector(metric="retrieval.recall@5", max_drop=0.05)

    result = run_mutation(
        store, suite, config, questions, docs, bm25_system_factory, scorers, scoring_ctx, operators, detector
    )
    assert result.equivalent == 2
    assert result.killed == 0
    assert result.score == 0.0  # denominator is 0 -- must not crash, must not read as perfect
    for m in result.mutants:
        assert m.equivalent is True
        assert m.equivalent_reason
