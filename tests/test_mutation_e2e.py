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


def test_mutation_runs_record_env_affects_results_when_given(store: Store) -> None:
    """Baseline and every mutant carry the caller's `affects_results` (the
    CLI passes the corpus hash), so a mutant run is comparable with a
    `run` baseline over the same corpus. Without the argument, the
    pre-M5 behaviour -- an empty dict -- is preserved."""
    suite, questions = setup_suite(store)
    docs = load_fixture_docs()
    config = store.put_config({"top_k": 5})
    scoring_ctx = scoring_ctx_factory(config.spec)
    detector = ThresholdDetector(metric="retrieval.recall@5", max_drop=0.05)

    run_mutation(
        store, suite, config, questions, docs, bm25_system_factory, standard_scorers(), scoring_ctx,
        [TruncateTopkOperator(k=1)], detector, env_affects_results={"corpus_hash": "abc"},
    )
    first_batch = store.runs_for_suite(suite.id)
    assert len(first_batch) == 2  # baseline + one mutant
    for run in first_batch:
        assert run.env_json["affects_results"] == {"corpus_hash": "abc"}

    run_mutation(
        store, suite, config, questions, docs, bm25_system_factory, standard_scorers(), scoring_ctx,
        [TruncateTopkOperator(k=1)], detector,
    )
    second_batch = store.runs_for_suite(suite.id)[len(first_batch):]
    assert len(second_batch) == 2
    for run in second_batch:
        assert run.env_json["affects_results"] == {}


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


def test_mutation_e2e_order_sensitive_metric_catches_what_recall_does_not(store: Store) -> None:
    """Regression test for the `detector:` block being silently discarded
    (cli.py's `mutate` always built `ThresholdDetector` from its own
    hardcoded `retrieval.recall@5` default, ignoring whatever the mutants
    YAML actually asked for). `swap_ranking` reverses the retrieved list
    without changing *which* chunks are in the top-k window, so it is
    invisible to `retrieval.recall@5` (window membership only) but must be
    caught by `retrieval.mrr` (rank of the first relevant chunk, order-
    sensitive) -- a detector silently defaulted to `recall@5` would report
    this mutant as SURVIVED when the eval setup, configured as asked,
    actually kills it.
    """
    suite, questions = setup_suite(store)
    docs = load_fixture_docs()
    config = store.put_config({"top_k": 5})
    scoring_ctx = scoring_ctx_factory(config.spec)
    scorers = standard_scorers()
    operators = [SwapRankingOperator()]

    recall_detector = ThresholdDetector(metric="retrieval.recall@5", max_drop=0.05)
    recall_result = run_mutation(
        store, suite, config, questions, docs, bm25_system_factory, scorers, scoring_ctx,
        operators, recall_detector,
    )
    swap_under_recall = next(m for m in recall_result.mutants if m.operator == "swap_ranking")
    assert swap_under_recall.equivalent is False
    assert swap_under_recall.killed is False  # recall@5 cannot see a pure re-ordering

    mrr_detector = ThresholdDetector(metric="retrieval.mrr", max_drop=0.05)
    mrr_result = run_mutation(
        store, suite, config, questions, docs, bm25_system_factory, scorers, scoring_ctx,
        operators, mrr_detector,
    )
    swap_under_mrr = next(m for m in mrr_result.mutants if m.operator == "swap_ranking")
    assert swap_under_mrr.equivalent is False
    assert swap_under_mrr.killed is True  # mrr is order-sensitive and must catch it
    assert "retrieval.mrr" in swap_under_mrr.detail
