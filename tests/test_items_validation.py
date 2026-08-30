"""`test_items_validation.py`: `items.validation.DetectorValidation`'s
`verdict` must be **derived** from `precision`/`base_rate`, never trusted
as written by a caller (see the module docstring, and defect 50 -- the
recorded failure this whole mechanism exists to make impossible to
paper over). Every test here is a regression test for that one property.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fireassay.items.core import Estimate
from fireassay.items.validation import (
    DetectorValidation,
    derive_verdict,
    load_validations,
    write_validations,
)


def _estimate(value: float, ci: tuple[float, float], n: int = 40) -> Estimate:
    return Estimate(value=value, ci=ci, n=n, verdict="measured")


# -- verdict is derived, not asserted ----------------------------------------


def test_verdict_is_derived_caller_cannot_mark_a_failed_detector_validated(tmp_path: Path) -> None:
    """A precision CI spanning the base rate must yield `not_validated`,
    even when the caller constructs the record with `verdict="validated"`
    -- `write_validations` must overrule it before anything reaches disk."""
    precision = _estimate(0.219, (0.10, 0.38))  # CI spans base_rate = 0.174
    record = DetectorValidation(
        detector="mislabel_suspect",
        measured_on="2026-08-29",
        labels="132 uniform-random human labels, run/calibration.jsonl",
        base_rate=0.174,
        precision=precision,
        recall=None,
        note=None,
        verdict="validated",  # the caller's (wrong) assertion
    )
    path = tmp_path / "validations.jsonl"
    write_validations([record], path)

    loaded = load_validations(path)
    assert loaded["mislabel_suspect"].verdict == "not_validated"


def test_derive_verdict_ci_spanning_base_rate_is_not_validated() -> None:
    assert derive_verdict(_estimate(0.219, (0.10, 0.38)), base_rate=0.174) == "not_validated"


def test_derive_verdict_ci_entirely_above_base_rate_is_validated() -> None:
    assert derive_verdict(_estimate(0.65, (0.55, 0.75)), base_rate=0.174) == "validated"


def test_derive_verdict_ci_entirely_below_base_rate_is_not_validated_not_the_reverse() -> None:
    """A detector measured *worse* than chance is not "validated in the
    opposite direction" -- it is still `not_validated`, the same as one
    indistinguishable from chance."""
    assert derive_verdict(_estimate(0.05, (0.02, 0.10)), base_rate=0.174) == "not_validated"


# -- round-trip preserves every field -----------------------------------------


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    precision = _estimate(0.652, (0.55, 0.75), n=46)
    recall = _estimate(0.652, (0.50, 0.79), n=46)
    record = DetectorValidation(
        detector="gpt-5.4 low effort",
        measured_on="2026-08-29",
        labels="132 uniform-random human labels, run/calibration.jsonl",
        base_rate=0.174,
        precision=precision,
        recall=recall,
        note="kept over granite4:7b and DeBERTa-v3-large NLI -- see handbook §9b",
        verdict="validated",
    )
    path = tmp_path / "validations.jsonl"
    write_validations([record], path)

    loaded = load_validations(path)
    round_tripped = loaded["gpt-5.4 low effort"]
    assert round_tripped.detector == record.detector
    assert round_tripped.measured_on == record.measured_on
    assert round_tripped.labels == record.labels
    assert round_tripped.base_rate == pytest.approx(record.base_rate)
    assert round_tripped.precision == record.precision
    assert round_tripped.recall == record.recall
    assert round_tripped.note == record.note
    assert round_tripped.verdict == record.verdict  # already "validated" -- unaffected by re-derivation


# -- no precision at all -> unmeasured ----------------------------------------


def test_no_precision_yields_unmeasured() -> None:
    assert derive_verdict(None, base_rate=0.174) == "unmeasured"


def test_precision_with_no_denominator_yields_unmeasured(tmp_path: Path) -> None:
    """`Estimate.value is None` (`verdict == "no_denominator"`, nothing
    was ever flagged and labelled) is the same "nothing to judge" case as
    `precision is None` -- both must yield `unmeasured`, never a computed
    verdict built on a missing value."""
    precision = Estimate(value=None, ci=None, n=0, verdict="no_denominator")
    record = DetectorValidation(
        detector="untested_detector",
        measured_on="2026-08-29",
        labels="no labels collected yet",
        base_rate=0.174,
        precision=precision,
        recall=None,
        note=None,
        verdict="validated",  # the caller's (wrong) assertion, again
    )
    path = tmp_path / "validations.jsonl"
    write_validations([record], path)

    loaded = load_validations(path)
    assert loaded["untested_detector"].verdict == "unmeasured"
