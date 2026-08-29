"""`test_items_calibration.py`: `items.calibration` is the durable,
detector-agnostic record of human judgment (see its module docstring).

The tests in the "disjoint-stratum trap" section below are the regression
suite for a real shipped bug: `items.review._plan_batch` claims flagged
items first, so a review deck's calibration stratum can never contain one
of them. Computing recall/fpr directly over that stratum therefore reads
`0/n` regardless of the labels -- a detector with real false positives
could (and did, in a real dry run) print `fpr = 0.000`. `evaluate_detector`
must detect this and never report a computed zero for it; supplying a
`SamplingDesign` must recover the correct, population-scaled numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fireassay.items.calibration import (
    CalibrationLabel,
    SamplingDesign,
    evaluate_detector,
    load_calibration_jsonl,
)


def _label(
    item_id: str, verdict: str, stratum: str, reviewer: str = "r1", batch_id: str = "b1"
) -> CalibrationLabel:
    return CalibrationLabel(
        item_id=item_id,
        verdict=verdict,  # type: ignore[arg-type]
        stratum=stratum,  # type: ignore[arg-type]
        reviewer=reviewer,
        batch_id=batch_id,
    )


# -- precision: unaffected by any of the strata-scaling machinery ------------


def test_precision_uses_flagged_stratum_recall_fpr_base_rate_do_not() -> None:
    calibration_labels = [
        _label(f"cal{i}", "purge" if i < 4 else "keep", "calibration") for i in range(20)
    ]
    # detector flags 2 items sampled under the "flagged" stratum, plus one
    # of the genuinely bad calibration items -- this keeps the random
    # stratum NON-disjoint from flagged_ids (cal0 is in both), so both
    # calls stay on the original, unaffected direct path.
    flagged_ids = {"flagA", "flagB", "cal0"}

    without_flagged_stratum = evaluate_detector(flagged_ids, calibration_labels)

    with_flagged_stratum = evaluate_detector(
        flagged_ids,
        [*calibration_labels, _label("flagA", "purge", "flagged"), _label("flagB", "keep", "flagged")],
    )

    # recall/fpr/base_rate are identical -- adding flagged-stratum labels
    # must not move them at all.
    for name in ("recall", "fpr", "base_rate"):
        a = getattr(without_flagged_stratum, name)
        b = getattr(with_flagged_stratum, name)
        assert a.value == b.value
        assert a.n == b.n

    # precision DOES change: flagA/flagB are now labelled and in
    # flagged_ids, so they enter precision's denominator.
    assert without_flagged_stratum.precision.n == 1
    assert with_flagged_stratum.precision.n == 3
    assert without_flagged_stratum.precision.value == pytest.approx(1.0)
    assert with_flagged_stratum.precision.value == pytest.approx(2 / 3)


# -- n == 0 -> None, never 0.0 -------------------------------------------------


def test_zero_denominator_gives_none_never_a_fabricated_zero() -> None:
    evaluation = evaluate_detector(flagged_ids=set(), labels=[])
    for estimate in (evaluation.precision, evaluation.recall, evaluation.fpr, evaluation.base_rate):
        assert estimate.value is None
        assert estimate.ci is None
        assert estimate.n == 0
        assert estimate.verdict == "no_denominator"


# -- 0 < n < min_denominator -> real value/CI, verdict too_few_labels --------


def test_too_few_labels_still_reports_a_real_value_and_ci() -> None:
    labels = [_label(f"cal{i}", "purge" if i < 2 else "keep", "calibration") for i in range(5)]
    # flagged_ids includes one calibration item id so this stays on the
    # (unaffected) direct path -- this test is about the too_few_labels
    # threshold, not about stratification.
    evaluation = evaluate_detector(flagged_ids={"cal0"}, labels=labels)
    assert evaluation.base_rate.n == 5
    assert evaluation.base_rate.value == pytest.approx(2 / 5)
    assert evaluation.base_rate.ci is not None
    assert evaluation.base_rate.verdict == "too_few_labels"


# -- unlabelled flagged items excluded from precision, counted separately ----


def test_unlabelled_flagged_items_excluded_from_precision_and_counted() -> None:
    flagged_ids = {"flag0", "flag1", "flag2"}
    labels = [_label("flag0", "purge", "flagged")]  # flag1, flag2 never labelled

    evaluation = evaluate_detector(flagged_ids, labels)

    assert evaluation.precision.n == 1
    assert evaluation.precision.value == pytest.approx(1.0)
    assert evaluation.n_flagged_total == 3
    assert evaluation.n_flagged_labelled == 1
    assert evaluation.n_flagged_unlabelled == 2


# -- agreeing reviewers both count; disagreeing raises; filter resolves ------


def test_agreeing_reviewers_count_disagreeing_raises_reviewer_filter_resolves() -> None:
    agreeing = [
        _label("x0", "purge", "calibration", reviewer="alice"),
        _label("x0", "purge", "calibration", reviewer="bob"),
    ]
    # flagged_ids={"x0"} keeps this on the direct path -- this test is
    # about reviewer dedup counting, not stratification.
    evaluation = evaluate_detector(flagged_ids={"x0"}, labels=agreeing)
    assert evaluation.base_rate.n == 2  # not deduplicated
    assert evaluation.base_rate.value == pytest.approx(1.0)

    disagreeing = [
        _label("x1", "purge", "calibration", reviewer="alice"),
        _label("x1", "keep", "calibration", reviewer="bob"),
    ]
    with pytest.raises(ValueError) as exc_info:
        evaluate_detector(flagged_ids={"x1"}, labels=disagreeing)
    message = str(exc_info.value)
    assert "x1" in message
    assert "alice" in message
    assert "bob" in message

    resolved = evaluate_detector(flagged_ids={"x1"}, labels=disagreeing, reviewer="alice")
    assert resolved.base_rate.n == 1
    assert resolved.base_rate.value == pytest.approx(1.0)


# -- duplicate (item_id, reviewer) on load raises; different reviewers ok ----


def test_load_rejects_duplicate_item_reviewer_but_allows_different_reviewers(tmp_path: Path) -> None:
    dup_path = tmp_path / "dup.jsonl"
    a = _label("x0", "purge", "calibration", reviewer="alice", batch_id="b1")
    a_again = _label("x0", "keep", "calibration", reviewer="alice", batch_id="b2")
    with open(dup_path, "w", encoding="utf-8") as f:
        f.write(a.model_dump_json())
        f.write("\n")
        f.write(a_again.model_dump_json())
        f.write("\n")
    with pytest.raises(ValueError, match="x0"):
        load_calibration_jsonl(dup_path)

    ok_path = tmp_path / "ok.jsonl"
    b = _label("x0", "keep", "calibration", reviewer="bob", batch_id="b1")
    with open(ok_path, "w", encoding="utf-8") as f:
        f.write(a.model_dump_json())
        f.write("\n")
        f.write(b.model_dump_json())
        f.write("\n")
    loaded = load_calibration_jsonl(ok_path)
    assert len(loaded) == 2
    assert {label.reviewer for label in loaded} == {"alice", "bob"}


# -- a literal U+2028 must not truncate the record (the splitlines() trap) ---


def test_u2028_inside_a_string_does_not_truncate_the_record(tmp_path: Path) -> None:
    path = tmp_path / "u2028.jsonl"
    # Built via chr(), not a source-level escape, so there is no ambiguity
    # about what character ends up in the string this test depends on.
    line_separator_char = chr(0x2028)
    weird_batch_id = "batch" + line_separator_char + "with-line-separator"
    obj = {
        "item_id": "x0",
        "verdict": "purge",
        "stratum": "calibration",
        "reviewer": "alice",
        "batch_id": weird_batch_id,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")

    # Negative control: confirm the trap is real -- str.splitlines() really
    # does split this single JSONL record into two pieces. If this
    # assertion ever fails, the test above is no longer exercising the bug
    # it exists to catch.
    raw = path.read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 2

    loaded = load_calibration_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].batch_id == weird_batch_id


# == The disjoint-stratum trap and its stratified fix =========================


def test_disjoint_random_stratum_without_design_is_not_a_measured_zero() -> None:
    """The regression test for what shipped: `_plan_batch` claims flagged
    items first, so the calibration/random stratum can never contain one
    -- a direct recall/fpr computation over it is identically `0/n`
    regardless of the labels. Without a `SamplingDesign`, this must be
    reported as unmeasurable, never as a measured 0.0."""
    flagged_ids = {f"flag{i}" for i in range(10)}
    flagged_labels = [_label(f"flag{i}", "purge", "flagged") for i in range(10)]
    random_labels = [_label(f"cal{i}", "keep", "calibration") for i in range(20)]
    labels = flagged_labels + random_labels

    evaluation = evaluate_detector(flagged_ids, labels)  # no design given

    assert evaluation.stratum_estimator == "stratified"

    assert evaluation.recall.value is None
    assert evaluation.recall.ci is None
    assert evaluation.recall.verdict == "needs_sampling_design"
    # explicit, not merely implied by `is None` -- this is the exact bug
    assert evaluation.recall.value != 0.0

    assert evaluation.fpr.value is None
    assert evaluation.fpr.ci is None
    assert evaluation.fpr.verdict == "needs_sampling_design"
    assert evaluation.fpr.value != 0.0

    assert evaluation.base_rate.value is None
    assert evaluation.base_rate.verdict == "needs_sampling_design"
    assert evaluation.base_rate.value != 0.0

    # precision is untouched by any of this
    assert evaluation.precision.value == pytest.approx(1.0)


def test_stratified_estimator_reproduces_hand_computed_numbers() -> None:
    """Sanity-check numbers (N_pool=2364, N_flag=32, n_flag_labelled=30,
    TP=17, FP=13; random n=34, bad=1): recall~0.2091, fpr~0.00609,
    base_rate~0.0367. Checked two independent ways: against the formula
    re-derived here (not copied from `calibration.py`'s own code) and
    against the externally hand-computed figures themselves."""
    pool_size, flagged_size = 2364, 32
    tp_obs, fp_obs = 17, 13
    n_rand_labelled, bad_random = 34, 1

    design = SamplingDesign(pool_size=pool_size, flagged_size=flagged_size)
    n_flag_labelled = tp_obs + fp_obs  # 30 of 32 -- a partial census
    census_labels = [_label(f"census{i}", "purge", "flagged") for i in range(tp_obs)] + [
        _label(f"census{i}", "keep", "flagged") for i in range(tp_obs, n_flag_labelled)
    ]
    random_labels = [
        _label(f"rand{i}", "purge" if i < bad_random else "keep", "calibration")
        for i in range(n_rand_labelled)
    ]
    labels = census_labels + random_labels
    flagged_ids = {f"census{i}" for i in range(n_flag_labelled)}

    evaluation = evaluate_detector(flagged_ids, labels, design=design)
    assert evaluation.stratum_estimator == "stratified"

    # Independently re-derived from the same closed-form formula (see the
    # module docstring), not copied from calibration.py's own code path.
    census_scale = flagged_size / n_flag_labelled
    n_rest = pool_size - flagged_size
    rand_scale = n_rest / n_rand_labelled
    tp_pop = tp_obs * census_scale
    fp_pop = fp_obs * census_scale
    fn_pop = bad_random * rand_scale
    tn_pop = (n_rand_labelled - bad_random) * rand_scale
    expected_recall = tp_pop / (tp_pop + fn_pop)
    expected_fpr = fp_pop / (fp_pop + tn_pop)
    expected_base_rate = (tp_pop + fn_pop) / pool_size

    assert evaluation.recall.value == pytest.approx(expected_recall, rel=1e-9)
    assert evaluation.fpr.value == pytest.approx(expected_fpr, rel=1e-9)
    assert evaluation.base_rate.value == pytest.approx(expected_base_rate, rel=1e-9)

    # Anchored separately to the externally hand-computed figures, at
    # their stated precision, so an error that happens to survive the
    # self-consistency check above (e.g. an identical mistake on both
    # sides) still gets caught against an independently-derived target.
    assert evaluation.recall.value == pytest.approx(0.2091, abs=5e-4)
    assert evaluation.fpr.value == pytest.approx(0.00609, abs=5e-5)
    assert evaluation.base_rate.value == pytest.approx(0.0367, abs=5e-4)


def test_non_disjoint_random_stratum_uses_the_direct_path_unaffected() -> None:
    """When the random/calibration stratum genuinely does contain some of
    the detector's flagged items, the original direct computation still
    applies -- with or without a `SamplingDesign`, since the direct path
    does not need one and a supplied design must not be applied on top
    of it."""
    labels = [
        _label("shared0", "purge", "calibration"),
        _label("shared1", "keep", "calibration"),
        _label("only_random0", "purge", "calibration"),
        _label("only_random1", "keep", "calibration"),
    ]
    flagged_ids = {"shared0", "shared1"}  # overlaps the calibration stratum

    without_design = evaluate_detector(flagged_ids, labels)
    with_design = evaluate_detector(
        flagged_ids, labels, design=SamplingDesign(pool_size=1000, flagged_size=50)
    )

    assert without_design.stratum_estimator == "random_stratum_only"
    assert with_design.stratum_estimator == "random_stratum_only"

    # direct computation: base_num=2 (shared0, only_random0 both purge),
    # base_den=4
    assert without_design.base_rate.value == pytest.approx(2 / 4)
    assert without_design.base_rate.n == 4

    # a design must not change anything at all on the non-disjoint path
    assert with_design.base_rate.value == without_design.base_rate.value
    assert with_design.base_rate.n == without_design.base_rate.n
    assert with_design.recall.value == without_design.recall.value
    assert with_design.fpr.value == without_design.fpr.value


def test_sampling_design_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="flagged_size"):
        SamplingDesign(pool_size=10, flagged_size=-1)
    with pytest.raises(ValueError, match="pool_size"):
        SamplingDesign(pool_size=10, flagged_size=10)
    with pytest.raises(ValueError, match="pool_size"):
        SamplingDesign(pool_size=5, flagged_size=10)
    # valid -- must not raise
    SamplingDesign(pool_size=11, flagged_size=10)


def test_stratified_base_rate_differs_from_naive_random_only_on_same_labels() -> None:
    """Proves the two formulas are genuinely different code, not one
    aliased to the other: over the same random-stratum labels, the naive
    (unweighted) proportion and the stratified population estimate must
    disagree, because the naive figure ignores the flagged stratum's
    (much higher) bad rate entirely."""
    design = SamplingDesign(pool_size=2364, flagged_size=32)
    census_labels = [_label(f"c{i}", "purge", "flagged") for i in range(17)] + [
        _label(f"c{i}", "keep", "flagged") for i in range(17, 30)
    ]
    random_labels = [_label(f"r{i}", "purge" if i == 0 else "keep", "calibration") for i in range(34)]
    labels = census_labels + random_labels
    flagged_ids = {f"c{i}" for i in range(30)}

    evaluation = evaluate_detector(flagged_ids, labels, design=design)

    # what a direct/pooled reading of the random stratum alone would give
    naive_random_only_base_rate = 1 / 34
    assert evaluation.base_rate.value is not None
    assert evaluation.base_rate.value != pytest.approx(naive_random_only_base_rate, rel=1e-6)
    # the flagged stratum's much higher bad rate pulls the true estimate up
    assert evaluation.base_rate.value > naive_random_only_base_rate


def test_partial_census_scales_correctly() -> None:
    """The normal case: not every flagged item gets a human label (a
    reviewer may skip one). n_flag_labelled=8 of flagged_size=10 here --
    independently hand-derived numbers, not copied from the N=32 dry-run
    example, to catch an error specific to that particular ratio."""
    pool_size, flagged_size = 100, 10
    tp_obs, fp_obs = 6, 2  # 8 of 10 flagged items labelled
    n_rand_labelled, bad_random = 20, 3

    design = SamplingDesign(pool_size=pool_size, flagged_size=flagged_size)
    n_flag_labelled = tp_obs + fp_obs
    census_labels = [_label(f"c{i}", "purge", "flagged") for i in range(tp_obs)] + [
        _label(f"c{i}", "keep", "flagged") for i in range(tp_obs, n_flag_labelled)
    ]
    random_labels = [
        _label(f"r{i}", "purge" if i < bad_random else "keep", "calibration")
        for i in range(n_rand_labelled)
    ]
    labels = census_labels + random_labels
    flagged_ids = {f"c{i}" for i in range(n_flag_labelled)}

    evaluation = evaluate_detector(flagged_ids, labels, design=design)

    census_scale = flagged_size / n_flag_labelled  # 10/8 = 1.25
    rand_scale = (pool_size - flagged_size) / n_rand_labelled  # 90/20 = 4.5
    tp_pop = tp_obs * census_scale
    fp_pop = fp_obs * census_scale
    fn_pop = bad_random * rand_scale
    tn_pop = (n_rand_labelled - bad_random) * rand_scale

    assert evaluation.recall.value == pytest.approx(tp_pop / (tp_pop + fn_pop), rel=1e-9)
    assert evaluation.fpr.value == pytest.approx(fp_pop / (fp_pop + tn_pop), rel=1e-9)
    assert evaluation.base_rate.value == pytest.approx((tp_pop + fn_pop) / pool_size, rel=1e-9)
    # hand-checked: recall=7.5/21~0.35714  fpr=2.5/79~0.031646  base_rate=0.21
    assert evaluation.recall.value == pytest.approx(0.35714, abs=1e-4)
    assert evaluation.fpr.value == pytest.approx(0.031646, abs=1e-5)
    assert evaluation.base_rate.value == pytest.approx(0.21, abs=1e-9)


def test_complete_census_scale_factor_is_exactly_one() -> None:
    """When every flagged item got labelled (n_flag_labelled ==
    flagged_size), the census scale factor N_flag/n_flag_labelled is
    exactly 1.0, so the stratified estimate's flagged-side inputs
    (TP_pop, FP_pop) coincide exactly with the raw observed counts -- the
    reviewer-representativeness assumption (see `evaluate_detector`'s
    docstring) is moot precisely because nothing was left unlabelled to
    be unrepresentative of."""
    pool_size, flagged_size = 50, 5
    tp_obs, fp_obs = 3, 2  # all 5 flagged items labelled -- a complete census
    n_rand_labelled, bad_random = 10, 2

    assert tp_obs + fp_obs == flagged_size  # sanity: this IS a complete census
    design = SamplingDesign(pool_size=pool_size, flagged_size=flagged_size)

    census_labels = [_label(f"c{i}", "purge", "flagged") for i in range(tp_obs)] + [
        _label(f"c{i}", "keep", "flagged") for i in range(tp_obs, flagged_size)
    ]
    random_labels = [
        _label(f"r{i}", "purge" if i < bad_random else "keep", "calibration")
        for i in range(n_rand_labelled)
    ]
    labels = census_labels + random_labels
    flagged_ids = {f"c{i}" for i in range(flagged_size)}

    evaluation = evaluate_detector(flagged_ids, labels, design=design)

    rand_scale = (pool_size - flagged_size) / n_rand_labelled
    fn_pop = bad_random * rand_scale
    tn_pop = (n_rand_labelled - bad_random) * rand_scale

    # scale factor is exactly 1 -- TP_pop/FP_pop equal the raw observed
    # counts with no scaling applied at all
    expected_recall = tp_obs / (tp_obs + fn_pop)
    expected_fpr = fp_obs / (fp_obs + tn_pop)
    assert evaluation.recall.value == pytest.approx(expected_recall, rel=1e-9)
    assert evaluation.fpr.value == pytest.approx(expected_fpr, rel=1e-9)
