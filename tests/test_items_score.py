"""`test_items_score.py` (M-ITEMS-SPEC.md §6): precision/recall arithmetic
on a hand-worked example; CIs widen as n shrinks."""

from __future__ import annotations

import math

import pytest

from fireassay.items.review import ReviewKeyEntry, ReviewLabel, score_review


def _wilson_ci_reference(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """An independently-written reference implementation of the 95%
    Wilson score interval (textbook formula), used to check
    `score_review`'s CI without depending on its own internals -- if the
    two disagree, one of them has a bug."""
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def _flagged_key(n: int) -> list[ReviewKeyEntry]:
    return [
        ReviewKeyEntry(review_id=f"f{i}", item_id=f"item{i}", source="flagged", is_seeded=False)
        for i in range(n)
    ]


def _seeded_key(n: int) -> list[ReviewKeyEntry]:
    return [
        ReviewKeyEntry(review_id=f"s{i}", item_id=f"seed{i}", source="seeded", is_seeded=True)
        for i in range(n)
    ]


def _labels(review_ids: list[str], n_bad: int) -> list[ReviewLabel]:
    """The first `n_bad` review_ids are labelled "purge" (the strict-mode
    bad verdict `score_review`'s default `bad_verdicts` counts), the rest
    "keep"."""
    return [
        ReviewLabel(review_id=rid, verdict="purge" if i < n_bad else "keep")
        for i, rid in enumerate(review_ids)
    ]


def test_precision_recall_hand_worked_example() -> None:
    # 4 flagged items: reviewer calls 3 of them bad -> precision = 3/4
    # 4 seeded items: reviewer catches 2 of them -> recall = 2/4
    key = _flagged_key(4) + _seeded_key(4)
    labels = _labels([f"f{i}" for i in range(4)], n_bad=3) + _labels([f"s{i}" for i in range(4)], n_bad=2)

    score = score_review(labels, key)

    assert score.precision == 0.75
    assert score.precision_n == 4
    assert score.recall == 0.5
    assert score.recall_n == 4

    expected_precision_ci = _wilson_ci_reference(3, 4)
    expected_recall_ci = _wilson_ci_reference(2, 4)
    assert score.precision_ci is not None
    assert score.recall_ci is not None
    assert score.precision_ci[0] == pytest.approx(expected_precision_ci[0], rel=1e-9)
    assert score.precision_ci[1] == pytest.approx(expected_precision_ci[1], rel=1e-9)
    assert score.recall_ci[0] == pytest.approx(expected_recall_ci[0], rel=1e-9)
    assert score.recall_ci[1] == pytest.approx(expected_recall_ci[1], rel=1e-9)


def test_unlabelled_review_items_are_excluded_from_both_sides() -> None:
    key = _flagged_key(3)
    labels = _labels(["f0"], n_bad=1)  # only f0 has a label; f1, f2 unlabelled
    score = score_review(labels, key)
    assert score.precision == 1.0
    assert score.precision_n == 1


def test_precision_and_recall_none_when_nothing_labelled_in_that_subset() -> None:
    key = _flagged_key(2) + _seeded_key(2)
    labels: list[ReviewLabel] = []
    score = score_review(labels, key)
    assert score.precision is None
    assert score.precision_ci is None
    assert score.precision_n == 0
    assert score.recall is None
    assert score.recall_ci is None
    assert score.recall_n == 0


def test_confidence_intervals_widen_as_n_shrinks() -> None:
    """The same 75% proportion, at n=4 vs n=40: the smaller sample must
    produce a wider (or equal) confidence interval."""
    small_key = _flagged_key(4)
    small_labels = _labels([f"f{i}" for i in range(4)], n_bad=3)
    small_score = score_review(small_labels, small_key)

    large_key = _flagged_key(40)
    large_labels = _labels([f"f{i}" for i in range(40)], n_bad=30)
    large_score = score_review(large_labels, large_key)

    assert small_score.precision == large_score.precision == 0.75
    assert small_score.precision_ci is not None
    assert large_score.precision_ci is not None
    small_width = small_score.precision_ci[1] - small_score.precision_ci[0]
    large_width = large_score.precision_ci[1] - large_score.precision_ci[0]
    assert small_width > large_width


def test_confidence_interval_at_n_one_stays_within_bounds() -> None:
    key = _flagged_key(1)
    labels = _labels(["f0"], n_bad=1)
    score = score_review(labels, key)
    assert score.precision == 1.0
    assert score.precision_ci is not None
    assert 0.0 <= score.precision_ci[0] <= score.precision_ci[1] <= 1.0
