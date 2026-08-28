"""Metric self-consistency invariants.

These must hold for *any* correct retrieval evaluation, regardless of how
good or bad the system under test is. A violation is a bug in the metric
code or the stored result — never a property of the system under test —
which is why a run with a violation is marked inadmissible
(`Run.admissible = False`) rather than merely scored low.

Run automatically at the end of every run by `runner.run_matrix`, over
every score produced by that run.

**History:** an earlier version of this module checked
`ndcg@k` monotonically non-decreasing in `k`, alongside `recall@k`. That
check was wrong and has been removed: unlike recall (whose numerator is
non-decreasing and denominator is fixed as k grows, so it cannot fall),
nDCG@k's *ideal* also grows with k, so a perfect hit at rank 1
(`ndcg@1 == 1.0`) can legitimately fall at `ndcg@3` once there are more
gold spans than can be covered by 1 hit. The checker caught this
correctly on real fixture data — the rule, not the checker, was wrong. It
is replaced by two weaker but actually-true consistency checks between
`ndcg`/`mrr` and `recall`: see `_check_ndcg_recall_consistency` and
`_check_mrr_recall_consistency`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from fireassay.models import Score

# Metrics exempt from the [0.0, 1.0] range check: latency and cost are
# unbounded positive quantities, and policy.violations is a rule-fired
# count (0, 1, 2, ...), not a proportion.
_RANGE_EXEMPT_PREFIXES = ("latency.", "cost.")
_RANGE_EXEMPT_EXACT = {"policy.violations"}

_K_METRIC_RE = re.compile(r"^retrieval\.(recall|ndcg)@(\d+)$")


class InvariantViolation(BaseModel):
    """One failed self-consistency check, scoped to a single question."""

    model_config = ConfigDict(frozen=True)

    rule: str
    question_id: str
    detail: str


def _is_range_exempt(metric: str) -> bool:
    return metric.startswith(_RANGE_EXEMPT_PREFIXES) or metric in _RANGE_EXEMPT_EXACT


def _check_range(question_id: str, scores: Sequence[Score]) -> list[InvariantViolation]:
    """Rule: every metric value lies in [0.0, 1.0], except latency.*,
    cost.* (unbounded) and policy.violations (a count)."""
    violations: list[InvariantViolation] = []
    for s in scores:
        if _is_range_exempt(s.metric):
            continue
        if not (0.0 <= s.value <= 1.0):
            violations.append(
                InvariantViolation(
                    rule="range_0_1",
                    question_id=question_id,
                    detail=f"{s.metric}={s.value!r} is outside [0.0, 1.0]",
                )
            )
    return violations


def _k_values(by_metric: Mapping[str, float], kind: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for metric, value in by_metric.items():
        m = _K_METRIC_RE.match(metric)
        if m and m.group(1) == kind:
            values.append((int(m.group(2)), value))
    values.sort(key=lambda pair: pair[0])
    return values


def _check_recall_monotonic(question_id: str, by_metric: Mapping[str, float]) -> list[InvariantViolation]:
    """`retrieval.recall@k` must be monotonically non-decreasing in k, for
    a fixed question and run: the top-k window only ever grows as k grows,
    so the set of gold spans it covers can only grow too (recall's
    numerator is non-decreasing while its denominator, `len(gold_spans)`,
    is fixed). Unlike nDCG, this one really is always true."""
    violations: list[InvariantViolation] = []
    pairs = _k_values(by_metric, "recall")
    for (k_prev, v_prev), (k_next, v_next) in zip(pairs, pairs[1:], strict=False):
        if v_next < v_prev:
            violations.append(
                InvariantViolation(
                    rule="recall_monotonic",
                    question_id=question_id,
                    detail=(
                        f"retrieval.recall@{k_next}={v_next!r} < retrieval.recall@{k_prev}={v_prev!r}, "
                        "but recall@k must be monotonically non-decreasing in k"
                    ),
                )
            )
    return violations


def _check_mrr_ge_recall1(question_id: str, by_metric: Mapping[str, float]) -> list[InvariantViolation]:
    """`retrieval.mrr >= retrieval.recall@1` when both are present.

    If any gold span is covered within the top 1, recall@1 == (covered
    spans)/(gold spans) > 0 and the top-1 chunk must itself be relevant, so
    mrr == 1.0 >= recall@1 always; if nothing is covered at k=1, recall@1
    == 0.0 and mrr >= 0.0 trivially. A violation here means the two
    computations disagree about whether the top-ranked chunk is relevant.
    """
    violations: list[InvariantViolation] = []
    if "retrieval.mrr" in by_metric and "retrieval.recall@1" in by_metric:
        mrr = by_metric["retrieval.mrr"]
        recall1 = by_metric["retrieval.recall@1"]
        if mrr < recall1:
            violations.append(
                InvariantViolation(
                    rule="mrr_ge_recall1",
                    question_id=question_id,
                    detail=f"retrieval.mrr={mrr!r} < retrieval.recall@1={recall1!r}",
                )
            )
    return violations


def _check_ndcg_recall_consistency(
    question_id: str, by_metric: Mapping[str, float]
) -> list[InvariantViolation]:
    """For each k with both metrics present: `(ndcg@k > 0) <=> (recall@k > 0)`.

    nDCG and recall must agree on the yes/no question "was anything
    relevant found in the top k?", even though they disagree (correctly)
    on *how much* credit that finding deserves as k grows. This is the
    real bug class an unbroken tie-break or a mismatched relevance
    predicate between the two computations produces: a published library
    once shipped a constant scorer that produced `mrr@10 == 1.0` alongside
    `ndcg@10 == 0.0` — the two halves of the same scorer disagreeing about
    whether anything relevant was ever retrieved at all.
    """
    violations: list[InvariantViolation] = []
    recall_pairs = dict(_k_values(by_metric, "recall"))
    ndcg_pairs = dict(_k_values(by_metric, "ndcg"))
    for k in sorted(set(recall_pairs) & set(ndcg_pairs)):
        recall_v = recall_pairs[k]
        ndcg_v = ndcg_pairs[k]
        if (ndcg_v > 0) != (recall_v > 0):
            violations.append(
                InvariantViolation(
                    rule="ndcg_recall_consistency",
                    question_id=question_id,
                    detail=(
                        f"retrieval.ndcg@{k}={ndcg_v!r} and retrieval.recall@{k}={recall_v!r} "
                        "disagree on whether anything relevant was found in the top k "
                        "(ndcg@k > 0 iff recall@k > 0)"
                    ),
                )
            )
    return violations


def _check_mrr_recall_consistency(
    question_id: str, by_metric: Mapping[str, float]
) -> list[InvariantViolation]:
    """`(mrr > 0) <=> (recall@k_max > 0)`, where `k_max` is the largest k
    this question was evaluated at.

    MRR is computed over the system's full retrieved list (see
    `RetrievalScorer.score`), not re-truncated by `eval_ks` — but
    `recall@k_max` is the closest available proxy for "was anything
    relevant found anywhere in what was evaluated", and the two must
    agree on that yes/no question for the same reason `ndcg`/`recall`
    must (see `_check_ndcg_recall_consistency`).
    """
    violations: list[InvariantViolation] = []
    if "retrieval.mrr" not in by_metric:
        return violations
    recall_pairs = _k_values(by_metric, "recall")
    if not recall_pairs:
        return violations
    k_max, recall_at_k_max = max(recall_pairs, key=lambda pair: pair[0])
    mrr = by_metric["retrieval.mrr"]
    if (mrr > 0) != (recall_at_k_max > 0):
        violations.append(
            InvariantViolation(
                rule="mrr_recall_consistency",
                question_id=question_id,
                detail=(
                    f"retrieval.mrr={mrr!r} and retrieval.recall@{k_max}={recall_at_k_max!r} "
                    "(k_max) disagree on whether anything relevant was ever retrieved"
                ),
            )
        )
    return violations


def check_invariants(scores_by_question: Mapping[str, Sequence[Score]]) -> list[InvariantViolation]:
    """Check every self-consistency invariant over every question's scores
    in a run.

    A checker that has never fired is untested: each rule above has a
    corresponding test in tests/test_invariants.py that feeds it a
    deliberately violating input and asserts the violation is reported,
    not just tests that valid input passes.
    """
    violations: list[InvariantViolation] = []
    for question_id, scores in scores_by_question.items():
        by_metric: dict[str, float] = {s.metric: s.value for s in scores}
        violations.extend(_check_range(question_id, scores))
        violations.extend(_check_recall_monotonic(question_id, by_metric))
        violations.extend(_check_mrr_ge_recall1(question_id, by_metric))
        violations.extend(_check_ndcg_recall_consistency(question_id, by_metric))
        violations.extend(_check_mrr_recall_consistency(question_id, by_metric))
    return violations
