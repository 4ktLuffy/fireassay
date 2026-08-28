"""M3-SPEC.md §7: alpha on a hand-computed example; ordinal vs nominal
handled distinctly; missing observations pass through un-imputed.

Also (post-M3 fix): "unmeasurable" must be a distinct, explicit state,
never a NaN or a raised exception silently read as "agreement is fine".
`isinstance(x, float)` does not distinguish a real number from `NaN` --
every assertion about an `"OK"` result below additionally checks
`not math.isnan(...)`.

The complete-disagreement case below is worked by hand from Krippendorff's
own coincidence-matrix formula (Hayes & Krippendorff 2007), for the
smallest case that is still hand-tractable: 2 units, 2 raters, a complete
reversal, nominal metric.

    alice: c1=False(0), c2=True(1)
    bob:   c1=True(1),  c2=False(0)

    coincidence counts: o(0,1) = 2, o(1,0) = 2, o(0,0) = o(1,1) = 0
    n_total = 4; n_0 = 2, n_1 = 2

    Do = (1/n_total) * sum(o_ck * delta_ck) = (1/4) * (2 + 2) = 1.0
    De = (1/(n_total*(n_total-1))) * sum(n_c * n_k * delta_ck)
       = (1/12) * (2*2 + 2*2) = 8/12 = 0.6667
    alpha = 1 - Do/De = 1 - 1.5 = -0.5
"""

from __future__ import annotations

import math
from typing import Literal

import pytest

from fireassay.curate.agreement import krippendorff_alpha, level_of_measurement
from fireassay.curate.models import RubricVerdict


def _rubric(
    self_contained: bool,
    answerable: Literal["yes", "no", "partially"] = "yes",
    ref_correct: Literal["yes", "no", "incomplete"] = "yes",
) -> RubricVerdict:
    return RubricVerdict(
        answerable_from_kb=answerable,
        self_contained=self_contained,
        reference_answer_correct=ref_correct,
        evidence_sufficient=True,
        difficulty_agrees=True,
        qtype_agrees=True,
    )


def test_alpha_perfect_agreement_with_variation_across_units_is_one() -> None:
    """Zero observed disagreement, nonzero expected disagreement (values do
    vary across units) -> alpha = 1.0 exactly, by definition."""
    data = {
        "alice": {"c1": _rubric(True), "c2": _rubric(False), "c3": _rubric(True), "c4": _rubric(False)},
        "bob": {"c1": _rubric(True), "c2": _rubric(False), "c3": _rubric(True), "c4": _rubric(False)},
    }
    result = krippendorff_alpha(data, "self_contained")
    assert result.status == "OK"
    assert result.alpha is not None
    assert not math.isnan(result.alpha)
    assert result.alpha == pytest.approx(1.0)


def test_alpha_hand_computed_complete_disagreement() -> None:
    data = {
        "alice": {"c1": _rubric(False), "c2": _rubric(True)},
        "bob": {"c1": _rubric(True), "c2": _rubric(False)},
    }
    result = krippendorff_alpha(data, "self_contained")
    assert result.status == "OK"
    assert result.alpha is not None
    assert not math.isnan(result.alpha)
    assert result.alpha == pytest.approx(-0.5, abs=1e-6)
    # A genuine, measured disagreement below threshold: low agreement,
    # never "unmeasurable" -- the two must stay distinguishable.
    assert result.alpha < 0.6


def test_level_of_measurement_ordinal_for_the_two_ordinal_criteria() -> None:
    assert level_of_measurement("answerable_from_kb") == "ordinal"
    assert level_of_measurement("reference_answer_correct") == "ordinal"


def test_level_of_measurement_nominal_for_the_boolean_criteria() -> None:
    for field in ("self_contained", "evidence_sufficient", "difficulty_agrees", "qtype_agrees"):
        assert level_of_measurement(field) == "nominal"


def test_level_of_measurement_rejects_unknown_criterion() -> None:
    with pytest.raises(ValueError, match="unknown"):
        level_of_measurement("not_a_real_rubric_field")


def test_missing_observations_are_unmeasurable_not_a_silent_pass() -> None:
    """bob never rated c2 at all. The reliability matrix this produces is
    exactly the shape that makes `krippendorff.alpha` return NaN --
    `isinstance(alpha, float)` cannot tell that apart from a real number,
    which is precisely why `status` exists: a criterion whose agreement is
    undefined must never be reported as though it were merely low."""
    data = {
        "alice": {"c1": _rubric(True), "c2": _rubric(False)},
        "bob": {"c1": _rubric(True)},  # c2 missing entirely
    }
    result = krippendorff_alpha(data, "self_contained")
    assert result.status == "UNMEASURABLE"
    assert result.alpha is None
    assert result.detail  # a human-readable reason is always present


def test_single_valued_domain_is_unmeasurable_and_does_not_raise() -> None:
    """Two curators agreeing on every shared item is a completely ordinary
    outcome -- krippendorff.alpha raises `ValueError: There has to be more
    than one value in the domain` for it, and that must be caught, not
    left to crash `fireassay curate report`."""
    data = {
        "alice": {"c1": _rubric(True), "c2": _rubric(True), "c3": _rubric(True)},
        "bob": {"c1": _rubric(True), "c2": _rubric(True), "c3": _rubric(True)},
    }
    result = krippendorff_alpha(data, "self_contained")  # must not raise
    assert result.status == "UNMEASURABLE"
    assert result.alpha is None
    assert result.detail


def test_alpha_is_insufficient_data_with_fewer_than_two_curators() -> None:
    for data in ({"alice": {"c1": _rubric(True)}}, {}):
        result = krippendorff_alpha(data, "self_contained")
        assert result.status == "INSUFFICIENT_DATA"
        assert result.alpha is None


def test_alpha_is_insufficient_data_with_no_candidates_rated_at_all() -> None:
    result = krippendorff_alpha({"alice": {}, "bob": {}}, "self_contained")
    assert result.status == "INSUFFICIENT_DATA"
    assert result.alpha is None


def test_ordinal_criterion_alpha_is_computed_and_bounded() -> None:
    """A smoke test for the ordinal path specifically (distinct code path
    from nominal -- see `agreement.krippendorff_alpha`'s `_ORDINAL_LEVELS`
    mapping): partial agreement on an ordinal field returns a finite,
    sane alpha rather than raising or silently falling back to nominal."""
    data = {
        "alice": {"c1": _rubric(True, answerable="yes"), "c2": _rubric(True, answerable="no")},
        "bob": {"c1": _rubric(True, answerable="partially"), "c2": _rubric(True, answerable="no")},
    }
    result = krippendorff_alpha(data, "answerable_from_kb")
    assert result.status == "OK"
    assert result.alpha is not None
    assert not math.isnan(result.alpha)
    assert -1.0 <= result.alpha <= 1.0
