"""`test_gate.py`: covers `fireassay.gate` (M5's release-blocking check) and
the `gate_check` persistence layer in `fireassay.store.db`.

Four tests are cited by name from `gate.py`'s own docstrings, as the tests
that catch specific regressions if this module is ever changed carelessly:
`test_pairing_is_preserved`, `test_gate_refuses_threshold_below_mde`,
`test_refusal_precedes_verdict_even_with_a_real_regression`, and
`test_mde_grows_when_metrics_are_added`. Their names must not change without
updating `gate.py` to match.

The negative-control pair (`test_gate_blocks_an_injected_regression` /
`test_gate_passes_when_head_is_identical_to_base` /
`test_gate_passes_on_noise_with_no_true_effect`) is the integrity claim: the
gate is shown to go red *and* green, deliberately, not just asserted to
"work" against a single fixture.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from fireassay.gate import (
    MetricSpec,
    ThresholdBelowMDEError,
    _metric_seed,
    evaluate_gate,
    holm_bonferroni,
    mde,
    paired_bootstrap,
)
from fireassay.store.db import GateAlreadyCheckedError, Store
from helpers import setup_suite


def _shifted(
    n: int, shift: float, *, spread: float = 1.0, seed: int = 7
) -> tuple[list[float], list[float]]:
    """base drawn once from a fixed RNG; head is base + shift plus independent
    per-item jitter, so delta ≈ shift with a controllable standard error.

    `base`'s own scale (fixed at 10.0) is deliberately irrelevant to the
    resulting delta or its standard error: `head - base = shift + jitter`
    cancels `base` out exactly, so only `spread` (the jitter's std) controls
    how noisy the paired difference is.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=10.0, size=n)
    jitter = rng.normal(loc=0.0, scale=spread, size=n)
    head = base + shift + jitter
    return base.tolist(), head.tolist()


# -- paired_bootstrap ---------------------------------------------------------


def test_pairing_is_preserved() -> None:
    """A paired bootstrap resamples the *difference* vector, so a constant
    per-item shift must produce an exactly-zero standard error regardless of
    how much the base arm itself varies from item to item. If the two arms
    were instead resampled independently, `se` would come out on the order
    of the arms' own spread (~50 here) rather than the (exactly zero) spread
    of their difference -- this is the bug this test exists to catch."""
    rng = np.random.default_rng(11)
    base = rng.normal(loc=0.0, scale=50.0, size=30).tolist()
    head = [x + 1.0 for x in base]

    result = paired_bootstrap(base, head, b=2000, seed=1)

    assert result.delta == pytest.approx(1.0)
    assert result.se == pytest.approx(0.0, abs=1e-12)
    assert result.ci_low == pytest.approx(1.0)
    assert result.ci_high == pytest.approx(1.0)


def test_paired_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([1.0, 2.0, 3.0], [1.0, 2.0], b=10, seed=1)


def test_paired_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        paired_bootstrap([], [], b=10, seed=1)


def test_paired_bootstrap_same_seed_is_deterministic_different_seed_is_not() -> None:
    base, head = _shifted(40, shift=0.5, spread=2.0, seed=3)

    a = paired_bootstrap(base, head, b=2000, seed=99)
    b = paired_bootstrap(base, head, b=2000, seed=99)
    assert a == b

    c = paired_bootstrap(base, head, b=2000, seed=100)
    assert c.delta == pytest.approx(a.delta)  # delta is not resampled
    assert c.se != pytest.approx(a.se)


def test_p_value_can_never_be_zero() -> None:
    """An exactly-constant difference vector puts every bootstrap replicate
    at the observed delta itself, so the null-centred `shifted` array is
    identically zero and `extreme_count` is exactly 0 for any `b` -- this
    pins the `(1 + count) / (b + 1)` rule rather than the naive `count / b`,
    which would report `p_value = 0.0` here."""
    rng = np.random.default_rng(5)
    base = rng.normal(loc=0.0, scale=5.0, size=20).tolist()
    head = [x + 3.0 for x in base]
    b = 2000

    result = paired_bootstrap(base, head, b=b, seed=1)

    assert result.p_value == pytest.approx(1.0 / (b + 1))


def test_p_value_and_ci_are_well_formed_on_an_ordinary_case() -> None:
    base, head = _shifted(50, shift=0.3, spread=1.0, seed=6)
    result = paired_bootstrap(base, head, b=2000, seed=1)
    assert 0.0 < result.p_value <= 1.0
    assert result.ci_low <= result.ci_high


def test_smaller_alpha_gives_a_wider_interval() -> None:
    base, head = _shifted(50, shift=0.3, spread=1.0, seed=6)
    narrow = paired_bootstrap(base, head, b=2000, alpha=0.05, seed=1)
    wide = paired_bootstrap(base, head, b=2000, alpha=0.01, seed=1)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_paired_bootstrap_echoes_n_b_seed() -> None:
    base, head = _shifted(12, shift=0.0, spread=1.0, seed=2)
    result = paired_bootstrap(base, head, b=333, seed=77)
    assert result.n == 12
    assert result.b == 333
    assert result.seed == 77


# -- mde -----------------------------------------------------------------


def test_mde_rejects_alpha_out_of_range() -> None:
    for bad_alpha in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="alpha"):
            mde(1.0, alpha=bad_alpha, power=0.80)


def test_mde_rejects_power_out_of_range() -> None:
    for bad_power in (0.5, 1.0, 0.4):
        with pytest.raises(ValueError, match="power"):
            mde(1.0, alpha=0.05, power=bad_power)


def test_mde_known_value() -> None:
    # z_{0.975} + z_{0.80} = 1.959964 + 0.841621
    assert mde(1.0, alpha=0.05, power=0.80) == pytest.approx(2.80158, abs=1e-4)


def test_mde_strictly_increasing_in_se() -> None:
    assert mde(2.0, alpha=0.05, power=0.80) > mde(1.0, alpha=0.05, power=0.80)


def test_mde_strictly_decreasing_in_alpha() -> None:
    # a smaller alpha demands more evidence to reject, which raises (not
    # lowers) the detection floor.
    assert mde(1.0, alpha=0.01, power=0.80) > mde(1.0, alpha=0.05, power=0.80)


# -- holm_bonferroni -------------------------------------------------------


def test_holm_bonferroni_empty_input() -> None:
    assert holm_bonferroni([]) == ()


def test_holm_bonferroni_worked_example_pins_the_running_maximum() -> None:
    # Sorted ranks (already ascending): 3*0.01=0.03, 2*0.02=0.04, 1*0.03=0.03;
    # the running max carries 0.04 forward over the third value. Without the
    # running max the third value would be 0.03 -- smaller than the second,
    # which is the incoherence the carry-forward prevents.
    result = holm_bonferroni([0.01, 0.02, 0.03])
    assert result == pytest.approx((0.03, 0.04, 0.04))


def test_holm_bonferroni_preserves_input_order() -> None:
    # Same three raw values as the worked-example test, permuted; the
    # adjusted values must follow the *input's* positions, not be re-sorted.
    result = holm_bonferroni([0.03, 0.01, 0.02])
    assert result == pytest.approx((0.04, 0.03, 0.04))


def test_holm_bonferroni_every_adjusted_value_is_at_least_the_raw_value() -> None:
    raw = [0.2, 0.05, 0.9, 0.01]
    adjusted = holm_bonferroni(raw)
    for a, r in zip(adjusted, raw, strict=True):
        assert a >= r


def test_holm_bonferroni_clamps_at_one() -> None:
    assert holm_bonferroni([0.6, 0.7]) == (1.0, 1.0)


def test_holm_bonferroni_single_value_is_unchanged() -> None:
    assert holm_bonferroni([0.37]) == pytest.approx((0.37,))


def test_holm_bonferroni_sequence_is_monotone_in_ascending_raw_order() -> None:
    raw = [0.5, 0.1, 0.3, 0.01]
    adjusted = holm_bonferroni(raw)
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    in_rank_order = [adjusted[i] for i in order]
    assert in_rank_order == sorted(in_rank_order)


# -- _metric_seed ----------------------------------------------------------


def test_metric_seed_matches_the_documented_digest_scheme() -> None:
    # Independently re-derived from `_metric_seed`'s own docstring (blake2b
    # of the metric name, digest_size=4, read big-endian, XORed with the
    # report-level seed) rather than called through the function itself, so
    # this pins the digest scheme and would fail loudly if it ever changed.
    expected = 0 ^ int.from_bytes(hashlib.blake2b(b"correctness", digest_size=4).digest(), "big")
    assert _metric_seed(0, "correctness") == expected
    # And pinned against a literal as well: re-derivation alone would still pass
    # if the digest scheme and this test were changed together, whereas the
    # literal fixes the actual value one released version of fireassay used.
    assert _metric_seed(0, "correctness") == 1601583508


def test_metric_seed_differs_by_metric_name() -> None:
    assert _metric_seed(0, "correctness") != _metric_seed(0, "safety")


def test_metric_seed_is_stable_across_calls() -> None:
    assert _metric_seed(42, "correctness") == _metric_seed(42, "correctness")


# -- evaluate_gate: validation ----------------------------------------------


def test_evaluate_gate_rejects_empty_specs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_gate(
            base_run_id="base",
            head_run_id="head",
            suite_id="suite",
            per_item={},
            specs=[],
        )


def test_evaluate_gate_rejects_metric_missing_from_per_item() -> None:
    specs = [MetricSpec(metric="correctness", threshold=0.1)]
    with pytest.raises(ValueError, match="missing from per_item"):
        evaluate_gate(
            base_run_id="base",
            head_run_id="head",
            suite_id="suite",
            per_item={},
            specs=specs,
        )


def test_evaluate_gate_rejects_non_positive_threshold() -> None:
    base, head = _shifted(20, shift=0.0, seed=1)
    for bad_threshold in (0.0, -0.1):
        specs = [MetricSpec(metric="correctness", threshold=bad_threshold)]
        with pytest.raises(ValueError):
            evaluate_gate(
                base_run_id="base",
                head_run_id="head",
                suite_id="suite",
                per_item={"correctness": (base, head)},
                specs=specs,
            )


# -- evaluate_gate: refusal --------------------------------------------------


def test_gate_refuses_threshold_below_mde() -> None:
    base, head = _shifted(10, shift=0.0, spread=1.0, seed=1)
    specs = [MetricSpec(metric="correctness", threshold=1e-9)]

    with pytest.raises(ThresholdBelowMDEError) as exc_info:
        evaluate_gate(
            base_run_id="base",
            head_run_id="head",
            suite_id="suite",
            per_item={"correctness": (base, head)},
            specs=specs,
            b=2000,
            seed=0,
        )

    assert exc_info.value.metric == "correctness"
    assert exc_info.value.threshold == 1e-9
    assert exc_info.value.threshold < exc_info.value.mde


def test_refusal_precedes_verdict_even_with_a_real_regression() -> None:
    """A large, genuine regression does not buy the gate permission to
    answer a question it cannot resolve: refusing is not the same as
    passing, and no `GateReport` is produced."""
    base, head = _shifted(10, shift=-10.0, spread=1.0, seed=1)
    specs = [MetricSpec(metric="correctness", threshold=1e-9)]

    with pytest.raises(ThresholdBelowMDEError):
        # If this ever returned instead of raising, the return value would
        # be a `GateReport` -- the point of this test is that it never does.
        evaluate_gate(
            base_run_id="base",
            head_run_id="head",
            suite_id="suite",
            per_item={"correctness": (base, head)},
            specs=specs,
            b=2000,
            seed=0,
        )


def test_refusal_names_the_first_sub_mde_metric_in_specs_order() -> None:
    base_a, head_a = _shifted(10, shift=0.0, spread=1.0, seed=1)
    base_b, head_b = _shifted(10, shift=0.0, spread=1.0, seed=2)
    specs = [
        MetricSpec(metric="metric_a", threshold=1e-9),
        MetricSpec(metric="metric_b", threshold=1e-9),
    ]
    per_item = {"metric_a": (base_a, head_a), "metric_b": (base_b, head_b)}

    with pytest.raises(ThresholdBelowMDEError) as exc_info:
        evaluate_gate(
            base_run_id="base",
            head_run_id="head",
            suite_id="suite",
            per_item=per_item,
            specs=specs,
            b=2000,
            seed=0,
        )

    assert exc_info.value.metric == "metric_a"


def test_mde_grows_when_metrics_are_added() -> None:
    """The MDE is computed at `alpha / m`: adding metrics to the same gate
    call tightens every metric's corrected alpha, which can only raise
    (never lower) the detection floor reported for a metric whose own data
    did not change at all."""
    shared_base, shared_head = _shifted(100, shift=0.0, spread=1.0, seed=10)
    extra_base, extra_head = _shifted(50, shift=0.0, spread=1.0, seed=11)
    shared_spec = MetricSpec(metric="correctness", threshold=1.0)

    report_m1 = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (shared_base, shared_head)},
        specs=[shared_spec],
        b=2000,
        seed=0,
    )

    extra_specs = [
        MetricSpec(metric="metric_b", threshold=1.0),
        MetricSpec(metric="metric_c", threshold=1.0),
    ]
    report_m3 = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={
            "correctness": (shared_base, shared_head),
            "metric_b": (extra_base, extra_head),
            "metric_c": (extra_base, extra_head),
        },
        specs=[shared_spec, *extra_specs],
        b=2000,
        seed=0,
    )

    mde_m1 = next(r.mde for r in report_m1.results if r.metric == "correctness")
    mde_m3 = next(r.mde for r in report_m3.results if r.metric == "correctness")
    assert mde_m3 > mde_m1


# -- evaluate_gate: the CI correction (Part A's new behaviour) ---------------


def test_reported_ci_is_multiplicity_corrected() -> None:
    shared_base, shared_head = _shifted(100, shift=0.0, spread=1.0, seed=20)
    extra_base, extra_head = _shifted(50, shift=0.0, spread=1.0, seed=21)
    shared_spec = MetricSpec(metric="correctness", threshold=1.0)

    report_m1 = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (shared_base, shared_head)},
        specs=[shared_spec],
        b=2000,
        seed=0,
    )
    extra_specs = [
        MetricSpec(metric="metric_b", threshold=1.0),
        MetricSpec(metric="metric_c", threshold=1.0),
    ]
    report_m3 = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={
            "correctness": (shared_base, shared_head),
            "metric_b": (extra_base, extra_head),
            "metric_c": (extra_base, extra_head),
        },
        specs=[shared_spec, *extra_specs],
        b=2000,
        seed=0,
    )

    r1 = next(r for r in report_m1.results if r.metric == "correctness")
    r3 = next(r for r in report_m3.results if r.metric == "correctness")
    assert (r3.ci_high - r3.ci_low) > (r1.ci_high - r1.ci_low)


def test_ci_matches_paired_bootstrap_at_corrected_alpha() -> None:
    """Pins the corrected *level*, not merely "some correction happened":
    recomputes each metric's interval directly via `paired_bootstrap` at
    `alpha / m` and checks it against what the report actually stored."""
    base_a, head_a = _shifted(60, shift=0.1, spread=1.0, seed=30)
    base_b, head_b = _shifted(60, shift=0.0, spread=1.0, seed=31)
    base_c, head_c = _shifted(60, shift=0.0, spread=1.0, seed=32)
    specs = [
        MetricSpec(metric="correctness", threshold=1.0),
        MetricSpec(metric="metric_b", threshold=1.0),
        MetricSpec(metric="metric_c", threshold=1.0),
    ]
    per_item = {
        "correctness": (base_a, head_a),
        "metric_b": (base_b, head_b),
        "metric_c": (base_c, head_c),
    }
    seed = 5
    b = 2000
    alpha = 0.05

    report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item=per_item,
        specs=specs,
        alpha=alpha,
        b=b,
        seed=seed,
    )

    m = len(specs)
    for spec in specs:
        base_values, head_values = per_item[spec.metric]
        direct = paired_bootstrap(
            base_values, head_values, b=b, alpha=alpha / m, seed=_metric_seed(seed, spec.metric)
        )
        reported = next(r for r in report.results if r.metric == spec.metric)
        assert reported.ci_low == pytest.approx(direct.ci_low, abs=1e-12)
        assert reported.ci_high == pytest.approx(direct.ci_high, abs=1e-12)


def test_se_and_p_value_are_unaffected_by_the_ci_correction() -> None:
    """`se` (visible only via `paired_bootstrap` -- `GateMetricResult` does
    not carry it) and `p_value` are both computed from the bootstrap
    replicate means alone; `alpha` only feeds the percentile CI and, via
    `mde`, the refusal floor. At m=1 and m=3 the shared metric's reported
    `p_value`, and the underlying `se`, must therefore be identical; only
    `ci_low`/`ci_high`/`mde` may move."""
    shared_base, shared_head = _shifted(80, shift=0.0, spread=1.0, seed=40)
    extra_base, extra_head = _shifted(40, shift=0.0, spread=1.0, seed=41)
    seed = 0
    b = 2000
    shared_spec = MetricSpec(metric="correctness", threshold=1.0)
    extra_specs = [
        MetricSpec(metric="metric_b", threshold=1.0),
        MetricSpec(metric="metric_c", threshold=1.0),
    ]

    report_m1 = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (shared_base, shared_head)},
        specs=[shared_spec],
        b=b,
        seed=seed,
    )
    report_m3 = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={
            "correctness": (shared_base, shared_head),
            "metric_b": (extra_base, extra_head),
            "metric_c": (extra_base, extra_head),
        },
        specs=[shared_spec, *extra_specs],
        b=b,
        seed=seed,
    )

    r1 = next(r for r in report_m1.results if r.metric == "correctness")
    r3 = next(r for r in report_m3.results if r.metric == "correctness")
    assert r1.p_value == pytest.approx(r3.p_value, abs=1e-15)

    metric_seed = _metric_seed(seed, "correctness")
    se_at_m1 = paired_bootstrap(shared_base, shared_head, b=b, alpha=0.05 / 1, seed=metric_seed).se
    se_at_m3 = paired_bootstrap(shared_base, shared_head, b=b, alpha=0.05 / 3, seed=metric_seed).se
    assert se_at_m1 == pytest.approx(se_at_m3, abs=1e-15)

    assert r1.ci_low != pytest.approx(r3.ci_low)
    assert r1.mde != pytest.approx(r3.mde)


# -- evaluate_gate: the negative control pair --------------------------------


def test_gate_blocks_an_injected_regression() -> None:
    base, head = _shifted(200, shift=-5.0, spread=1.0, seed=50)
    threshold = 1.0
    specs = [MetricSpec(metric="correctness", threshold=threshold)]

    report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (base, head)},
        specs=specs,
        b=2000,
        seed=0,
    )

    result = report.results[0]
    assert report.blocked is True
    assert result.verdict == "block"
    assert result.delta < 0.0
    assert result.delta <= -threshold
    assert result.p_adjusted < report.alpha


def test_gate_passes_when_head_is_identical_to_base() -> None:
    rng = np.random.default_rng(60)
    base = rng.normal(loc=0.0, scale=1.0, size=30).tolist()
    head = list(base)
    specs = [MetricSpec(metric="correctness", threshold=1.0)]

    report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (base, head)},
        specs=specs,
        b=2000,
        seed=0,
    )

    result = report.results[0]
    # se is exactly 0.0 here (the diff vector is exactly constant at 0.0),
    # so mde is 0.0 too, and any positive threshold clears the refusal
    # check trivially.
    assert result.delta == pytest.approx(0.0, abs=1e-15)
    assert report.blocked is False
    assert result.verdict == "pass"


def test_gate_passes_on_noise_with_no_true_effect() -> None:
    rng = np.random.default_rng(70)
    base = rng.normal(loc=0.0, scale=1.0, size=100).tolist()
    head = rng.normal(loc=0.0, scale=1.0, size=100).tolist()
    specs = [MetricSpec(metric="correctness", threshold=1.0)]

    report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (base, head)},
        specs=specs,
        b=2000,
        seed=0,
    )

    assert report.blocked is False


# -- evaluate_gate: verdict logic (docstring step 6) -------------------------


def test_significant_delta_smaller_than_threshold_still_passes() -> None:
    """Significance alone does not block: the gate also requires the delta
    to reach `threshold` in the harmful direction."""
    base, head = _shifted(1000, shift=-0.2, spread=0.5, seed=80)
    specs = [MetricSpec(metric="correctness", threshold=1.0)]

    report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (base, head)},
        specs=specs,
        b=2000,
        seed=0,
    )

    result = report.results[0]
    assert result.p_adjusted < report.alpha
    assert result.verdict == "pass"


def test_threshold_sized_regression_that_is_not_significant_still_passes() -> None:
    """A regression whose *true* size is threshold-sized is not, by itself,
    enough to block: with only a handful of noisy items, the observed
    effect can land far short of significance even though its intended size
    matches `threshold`. Significance is necessary, not just magnitude.

    The seed is not arbitrary. The MDE guarantee means that once
    `threshold >= mde` an effect of size `threshold` is detected with
    `power` (80%) probability, so a draw where the *observed* delta lands
    short of significance without also tripping the MDE refusal is the
    ~20% case by construction. Seeds 1-399 were searched (2026-09-10):
    26 satisfy both `p_adjusted >= alpha` and no `ThresholdBelowMDEError`;
    seed 18 is the first with a comfortable margin on both (observed
    delta -0.28, p 0.42, mde 0.95 < threshold 1.0). Every one of those 26
    also has `|delta| < threshold`, which is the point: on a suite this
    small, an effect that is both threshold-sized *and* non-significant
    cannot exist alongside `threshold >= mde` -- that is what MDE means.
    """
    threshold = 1.0
    base, head = _shifted(8, shift=-threshold, spread=0.9, seed=18)
    specs = [MetricSpec(metric="correctness", threshold=threshold)]

    report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (base, head)},
        specs=specs,
        b=2000,
        seed=0,
    )

    result = report.results[0]
    assert result.p_adjusted >= report.alpha
    assert result.verdict == "pass"


def test_higher_is_better_false_blocks_on_rise_and_passes_on_fall() -> None:
    threshold = 1.0
    rise_base, rise_head = _shifted(200, shift=5.0, spread=1.0, seed=100)
    fall_base, fall_head = _shifted(200, shift=-5.0, spread=1.0, seed=100)
    spec = MetricSpec(metric="cost", threshold=threshold, higher_is_better=False)

    rise_report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"cost": (rise_base, rise_head)},
        specs=[spec],
        b=2000,
        seed=0,
    )
    fall_report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"cost": (fall_base, fall_head)},
        specs=[spec],
        b=2000,
        seed=0,
    )

    assert rise_report.results[0].verdict == "block"
    assert fall_report.results[0].verdict == "pass"


def test_blocked_is_true_iff_any_metric_blocks() -> None:
    block_base, block_head = _shifted(200, shift=-5.0, spread=1.0, seed=110)
    pass_base, pass_head = _shifted(200, shift=0.0, spread=1.0, seed=111)
    specs = [
        MetricSpec(metric="correctness", threshold=1.0),
        MetricSpec(metric="safety", threshold=1.0),
    ]

    report = evaluate_gate(
        base_run_id="base",
        head_run_id="head",
        suite_id="suite",
        per_item={"correctness": (block_base, block_head), "safety": (pass_base, pass_head)},
        specs=specs,
        b=2000,
        seed=0,
    )

    by_metric = {r.metric: r for r in report.results}
    assert by_metric["correctness"].verdict == "block"
    assert by_metric["safety"].verdict == "pass"
    assert report.blocked is True


def test_evaluate_gate_is_deterministic_given_the_same_seed() -> None:
    base, head = _shifted(50, shift=-0.3, spread=1.0, seed=120)
    specs = [MetricSpec(metric="correctness", threshold=1.0)]
    per_item = {"correctness": (base, head)}

    report_a = evaluate_gate(
        base_run_id="base_run",
        head_run_id="head_run",
        suite_id="suite",
        per_item=per_item,
        specs=specs,
        b=2000,
        seed=42,
    )
    report_b = evaluate_gate(
        base_run_id="base_run",
        head_run_id="head_run",
        suite_id="suite",
        per_item=per_item,
        specs=specs,
        b=2000,
        seed=42,
    )

    assert report_a == report_b


def test_evaluate_gate_echoes_identity_and_config_fields() -> None:
    base, head = _shifted(50, shift=0.0, spread=1.0, seed=130)
    specs = [
        MetricSpec(metric="correctness", threshold=2.0),
        MetricSpec(metric="safety", threshold=2.0),
    ]
    per_item = {"correctness": (base, head), "safety": (base, head)}

    report = evaluate_gate(
        base_run_id="base_run_1",
        head_run_id="head_run_1",
        suite_id="suite_1",
        per_item=per_item,
        specs=specs,
        alpha=0.1,
        power=0.9,
        b=2000,
        seed=7,
    )

    assert report.base_run_id == "base_run_1"
    assert report.head_run_id == "head_run_1"
    assert report.suite_id == "suite_1"
    assert report.alpha == 0.1
    assert report.power == 0.9
    assert report.b == 2000
    assert report.seed == 7
    assert len(report.results) == len(specs)
    assert [r.metric for r in report.results] == [s.metric for s in specs]


# -- gate persistence (store/db.py) ------------------------------------------


def _two_runs(store: Store) -> tuple[str, str, str]:
    """Create a suite and two completed runs against it. `gate_check`'s
    foreign keys need real `run`/`suite` rows to reference, even though the
    gate's own statistics are computed over `per_item` data supplied
    separately -- mirrors the run-creation pattern `_seed_run` uses in
    tests/test_compare.py and `_start_open_run` uses in tests/test_store.py.
    """
    suite, _questions = setup_suite(store)
    config = store.put_config({"top_k": 5})
    env = {"python": "3.12", "fireassay": "0.1.0", "affects_results": {}}
    base_run = store.start_run(suite, config, env)
    store.finish_run(base_run.id, "complete")
    head_run = store.start_run(suite, config, env)
    store.finish_run(head_run.id, "complete")
    return suite.id, base_run.id, head_run.id


def test_put_gate_report_round_trips(store: Store) -> None:
    suite_id, base_run_id, head_run_id = _two_runs(store)
    base_a, head_a = _shifted(30, shift=-5.0, spread=1.0, seed=200)
    base_b, head_b = _shifted(30, shift=0.0, spread=1.0, seed=201)
    specs = [
        MetricSpec(metric="correctness", threshold=1.0),
        MetricSpec(metric="safety", threshold=1.0),
    ]

    report = evaluate_gate(
        base_run_id=base_run_id,
        head_run_id=head_run_id,
        suite_id=suite_id,
        per_item={"correctness": (base_a, head_a), "safety": (base_b, head_b)},
        specs=specs,
        b=2000,
        seed=0,
    )
    store.put_gate_report(report)

    rows = store.get_gate_checks(base_run_id, head_run_id)
    assert [r.metric for r in rows] == ["correctness", "safety"]

    by_metric = {r.metric: r for r in rows}
    for result in report.results:
        row = by_metric[result.metric]
        assert row.base_run_id == base_run_id
        assert row.head_run_id == head_run_id
        assert row.suite_id == suite_id
        assert row.n_items == result.n
        assert row.base_mean == pytest.approx(result.base_mean)
        assert row.head_mean == pytest.approx(result.head_mean)
        assert row.delta == pytest.approx(result.delta)
        assert row.ci_low == pytest.approx(result.ci_low)
        assert row.ci_high == pytest.approx(result.ci_high)
        assert row.p_value == pytest.approx(result.p_value)
        assert row.p_adjusted == pytest.approx(result.p_adjusted)
        assert row.mde == pytest.approx(result.mde)
        assert row.threshold == pytest.approx(result.threshold)
        assert row.verdict == result.verdict


def test_gate_report_cannot_be_recorded_twice(store: Store) -> None:
    """The anti-p-hacking constraint: re-running a gate on the same pair of
    runs until it happens to come back green must not be possible."""
    suite_id, base_run_id, head_run_id = _two_runs(store)
    base, head = _shifted(30, shift=0.0, spread=1.0, seed=210)
    specs = [MetricSpec(metric="correctness", threshold=1.0)]

    report = evaluate_gate(
        base_run_id=base_run_id,
        head_run_id=head_run_id,
        suite_id=suite_id,
        per_item={"correctness": (base, head)},
        specs=specs,
        b=2000,
        seed=0,
    )
    store.put_gate_report(report)

    with pytest.raises(GateAlreadyCheckedError):
        store.put_gate_report(report)


def test_second_put_rolls_back_completely(store: Store) -> None:
    """A partially-persisted `GateReport` (some metrics recorded, others
    not) would be exactly as misleading as the repeated check the unique
    constraint forbids -- `put_gate_report` must roll the whole call back."""
    suite_id, base_run_id, head_run_id = _two_runs(store)
    base_shared, head_shared = _shifted(40, shift=0.0, spread=1.0, seed=220)
    base_new, head_new = _shifted(40, shift=0.0, spread=1.0, seed=221)

    first = evaluate_gate(
        base_run_id=base_run_id,
        head_run_id=head_run_id,
        suite_id=suite_id,
        per_item={"correctness": (base_shared, head_shared)},
        specs=[MetricSpec(metric="correctness", threshold=1.0)],
        b=2000,
        seed=0,
    )
    store.put_gate_report(first)

    # "new" ordered before "correctness" so that, absent the documented
    # rollback, the new metric's row would already be committed by the time
    # the conflicting "correctness" insert raises.
    second = evaluate_gate(
        base_run_id=base_run_id,
        head_run_id=head_run_id,
        suite_id=suite_id,
        per_item={
            "new": (base_new, head_new),
            "correctness": (base_shared, head_shared),
        },
        specs=[
            MetricSpec(metric="new", threshold=1.0),
            MetricSpec(metric="correctness", threshold=1.0),
        ],
        b=2000,
        seed=0,
    )

    with pytest.raises(GateAlreadyCheckedError):
        store.put_gate_report(second)

    rows = store.get_gate_checks(base_run_id, head_run_id)
    assert [r.metric for r in rows] == ["correctness"]


def test_gate_check_rows_cannot_be_updated_or_deleted(store: Store, tmp_path: Path) -> None:
    suite_id, base_run_id, head_run_id = _two_runs(store)
    base, head = _shifted(30, shift=0.0, spread=1.0, seed=230)

    report = evaluate_gate(
        base_run_id=base_run_id,
        head_run_id=head_run_id,
        suite_id=suite_id,
        per_item={"correctness": (base, head)},
        specs=[MetricSpec(metric="correctness", threshold=1.0)],
        b=2000,
        seed=0,
    )
    store.put_gate_report(report)
    row_id = store.get_gate_checks(base_run_id, head_run_id)[0].id

    # Bypass the Store class entirely: a raw connection on the same file
    # must still be blocked by the trigger (mirrors
    # test_sealed_run_rejects_writes_via_sql_trigger in test_store.py).
    raw = sqlite3.connect(str(tmp_path / "test.db"))
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("UPDATE gate_check SET verdict = 'pass' WHERE id = ?", (row_id,))
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("DELETE FROM gate_check WHERE id = ?", (row_id,))
    raw.close()
