"""Positive-control tests for score.invariants.check_invariants.

A checker that has never fired is untested: each rule gets a test that
feeds it a deliberately violating input and asserts the violation is
reported -- not just a test that valid input passes silently.
"""

from __future__ import annotations

from fireassay.models import Score
from fireassay.score.invariants import check_invariants


def _score(metric: str, value: float, scorer: str = "s@1.0.0") -> Score:
    return Score(metric=metric, value=value, scorer=scorer)


def test_valid_scores_produce_no_violations() -> None:
    scores_by_question = {
        "q1": [
            _score("retrieval.recall@1", 0.5),
            _score("retrieval.recall@5", 1.0),
            _score("retrieval.ndcg@1", 0.5),
            _score("retrieval.ndcg@5", 0.8),
            _score("retrieval.mrr", 1.0),
            _score("latency.total_ms", 250.0),
            _score("cost.usd", 0.02),
            _score("policy.violations", 0.0),
        ]
    }
    assert check_invariants(scores_by_question) == []


def test_rule_1_range_violation_is_reported() -> None:
    """A bounded metric (not latency.*/cost.*/policy.violations) outside
    [0.0, 1.0] must be flagged."""
    scores_by_question = {"q1": [_score("retrieval.recall@5", 1.5)]}
    violations = check_invariants(scores_by_question)
    assert len(violations) == 1
    assert violations[0].rule == "range_0_1"
    assert violations[0].question_id == "q1"


def test_rule_1_exempts_latency_cost_and_policy_violations() -> None:
    scores_by_question = {
        "q1": [
            _score("latency.total_ms", 5000.0),
            _score("cost.usd", 12.5),
            _score("policy.violations", 3.0),
        ]
    }
    assert check_invariants(scores_by_question) == []


def test_rule_2_recall_monotonicity_violation_is_reported() -> None:
    """recall@k must be non-decreasing in k; recall@5 < recall@1 is
    impossible for a correct implementation and must be flagged."""
    scores_by_question = {
        "q1": [
            _score("retrieval.recall@1", 0.8),
            _score("retrieval.recall@5", 0.2),  # deliberately violates monotonicity
        ]
    }
    violations = check_invariants(scores_by_question)
    assert len(violations) == 1
    assert violations[0].rule == "recall_monotonic"
    assert violations[0].question_id == "q1"


def test_ndcg_is_not_required_to_be_monotonic() -> None:
    """Regression test for the removed (incorrect) ndcg_monotonic rule:
    nDCG@k legitimately falls as k grows once IDCG@k outgrows what was
    actually found (a perfect hit at rank 1 can fall from ndcg@1 == 1.0 to
    ndcg@3 == 0.92 once there are more gold spans than can be covered by
    one hit). This must NOT be flagged."""
    scores_by_question = {
        "q1": [
            _score("retrieval.ndcg@1", 1.0),
            _score("retrieval.ndcg@3", 0.92),  # legitimately lower than ndcg@1
            _score("retrieval.recall@1", 1.0),
            _score("retrieval.recall@3", 1.0),
        ]
    }
    assert check_invariants(scores_by_question) == []


def test_ndcg_recall_consistency_violation_is_reported() -> None:
    """ndcg@k > 0 and recall@k == 0 at the same k is impossible: nDCG can
    only be positive if something relevant was found, which is exactly
    what a positive recall@k means too."""
    scores_by_question = {
        "q1": [
            _score("retrieval.recall@5", 0.0),
            _score("retrieval.ndcg@5", 0.7),  # impossible: nothing relevant found, yet ndcg > 0
        ]
    }
    violations = check_invariants(scores_by_question)
    assert len(violations) == 1
    assert violations[0].rule == "ndcg_recall_consistency"
    assert violations[0].question_id == "q1"


def test_ndcg_recall_consistency_also_flags_the_reverse_disagreement() -> None:
    scores_by_question = {
        "q1": [
            _score("retrieval.recall@5", 0.5),
            _score("retrieval.ndcg@5", 0.0),  # impossible: something relevant found, yet ndcg == 0
        ]
    }
    violations = check_invariants(scores_by_question)
    assert len(violations) == 1
    assert violations[0].rule == "ndcg_recall_consistency"


def test_mrr_recall_consistency_violation_is_reported() -> None:
    """mrr > 0 and recall@k_max == 0 is impossible: mrr > 0 means some
    retrieved chunk was relevant, so recall at the largest evaluated k
    (which covers at least as much of the ranking) cannot be zero."""
    scores_by_question = {
        "q1": [
            _score("retrieval.recall@1", 0.0),
            _score("retrieval.recall@10", 0.0),  # k_max
            _score("retrieval.mrr", 1.0),  # impossible alongside recall@10 == 0.0
        ]
    }
    violations = check_invariants(scores_by_question)
    assert len(violations) == 1
    assert violations[0].rule == "mrr_recall_consistency"
    assert violations[0].question_id == "q1"


def test_mrr_recall_consistency_uses_the_largest_evaluated_k() -> None:
    """mrr disagreeing with recall@1 alone is fine (that is a normal,
    expected shape: something relevant further down the ranking); only
    disagreement with recall@k_max is a violation."""
    scores_by_question = {
        "q1": [
            _score("retrieval.recall@1", 0.0),
            _score("retrieval.recall@5", 1.0),  # k_max: something relevant was found by k=5
            _score("retrieval.mrr", 0.2),  # consistent: first relevant chunk was found, just not at rank 1
        ]
    }
    assert check_invariants(scores_by_question) == []


def test_rule_4_mrr_below_recall_at_1_is_reported() -> None:
    scores_by_question = {
        "q1": [
            _score("retrieval.recall@1", 1.0),
            _score("retrieval.mrr", 0.0),  # impossible: if recall@1 == 1.0, mrr must be 1.0
        ]
    }
    violations = check_invariants(scores_by_question)
    # Two independent rules legitimately catch this same impossible input: mrr_ge_recall1
    # (mrr must be at least recall@1) and mrr_recall_consistency (both must agree on
    # whether anything relevant was retrieved at all). Assert on the rules present rather
    # than a count, so adding a further correct rule does not break this test.
    assert {v.rule for v in violations} == {"mrr_ge_recall1", "mrr_recall_consistency"}
    assert all(v.question_id == "q1" for v in violations)


def test_rule_4_not_checked_when_recall_at_1_absent() -> None:
    # recall@1 is absent, so mrr_ge_recall1 cannot apply. recall@5 is nonzero and mrr is
    # nonzero, so mrr_recall_consistency is satisfied — this input is genuinely clean.
    # A nonzero recall@5 would trip mrr_ge_recall1 if that rule wrongly fell back to k_max.
    scores_by_question = {"q1": [_score("retrieval.mrr", 0.5), _score("retrieval.recall@5", 1.0)]}
    assert check_invariants(scores_by_question) == []


def test_multiple_questions_are_checked_independently() -> None:
    scores_by_question = {
        "good": [_score("retrieval.recall@1", 0.5), _score("retrieval.recall@5", 0.5)],
        "bad": [_score("retrieval.recall@1", 0.9), _score("retrieval.recall@5", 0.1)],
    }
    violations = check_invariants(scores_by_question)
    assert len(violations) == 1
    assert violations[0].question_id == "bad"
