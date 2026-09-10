"""The gate — M5's release-blocking check, and the pillar the project's
whole thesis is written to prove out: an eval harness that "knows what it
cannot measure, and says so" (docs/SPEC.md §10.7) has to apply that
standard to itself before it applies it to a config.

Three ideas, each a documented, load-bearing choice:

**1. A gate refuses thresholds it cannot detect, before it reaches a
verdict.** `evaluate_gate` computes each metric's minimum detectable
effect (`mde`) from the same paired bootstrap it uses for the verdict, and
raises `ThresholdBelowMDEError` — never a `GateReport` — the moment any
`MetricSpec.threshold` is smaller than what this run of this suite can
actually resolve. docs/SPEC.md §8's own example is the cautionary tale:
"v2 set a 1-point threshold on a planned 1,000-question suite —
enforcing a threshold physically undetectable at that size, which is a
green whose cause was never established." A refusal is not a `pass`; it
is a statement that the question asked cannot honestly be answered yet,
and `evaluate_gate` never lets that statement get silently swallowed into
either verdict.

**2. Multiple-comparison correction is in the gate path, not bolted on
after.** Seven metrics checked at `alpha=0.05` each gives a per-PR false
positive rate of `1 - 0.95**7 ≈ 30%` (docs/SPEC.md §8) — enough to get a
gate switched off within a month. `evaluate_gate` Holm-adjusts every
metric's p-value (`holm_bonferroni`) before deciding `block`/`pass`, and —
the detail that is easy to get wrong — computes both the *refusal*
threshold (`mde`) and the *reported* confidence interval (`ci_low`/
`ci_high`) at the same corrected level, `alpha / m`, not the raw `alpha`,
while the verdict itself is decided by Holm. A gate that corrects its
verdict for multiplicity but still reports an uncorrected MDE would
understate its own detection floor by exactly the amount the correction
cost it; see `evaluate_gate`'s docstring, point 2, for why that would be
a second, quieter version of the same v2 mistake.

**3. The bootstrap is paired, and the pairing is the whole point.**
`paired_bootstrap` resamples *item indices* once per replicate and reads
both a metric's base and head values through that same resampled index —
never resampling the two arms independently. Two systems scored on the
same fixed suite are correlated question-by-question (a hard question is
hard for both), and a paired design exploits exactly that correlation to
shrink the standard error relative to an unpaired comparison of the same
size. An unpaired resample would silently throw that power away and
report a needlessly conservative (larger) `se` and `mde` — see
`test_pairing_is_preserved` in `tests/test_gate.py`, which is the test
that would catch this bug if it were ever reintroduced.

This module computes; it does not persist. `store/db.py`'s
`put_gate_report`/`get_gate_checks` persist a `GateReport` (once — see
`gate_check`'s `UNIQUE (base_run_id, head_run_id, metric)` constraint and
migration 0006's header comment for why re-running a gate until it goes
green is p-hacking the schema itself refuses to allow) and are the only
part of the M5 gate wired to a store; nothing here imports `Store`.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict


class BootstrapResult(BaseModel):
    """The output of one `paired_bootstrap` call: a delta, its bootstrap
    confidence interval and standard error, and a bootstrap p-value —
    everything `mde` and `evaluate_gate`'s verdict logic need, and nothing
    that depends on any particular gate metric's threshold or direction
    (those live on `MetricSpec`/`GateMetricResult`, one layer up)."""

    model_config = ConfigDict(frozen=True)

    n: int
    delta: float
    ci_low: float
    ci_high: float
    se: float
    p_value: float
    b: int
    seed: int


def paired_bootstrap(
    base: Sequence[float],
    head: Sequence[float],
    *,
    b: int = 10000,
    alpha: float = 0.05,
    seed: int,
) -> BootstrapResult:
    """Paired bootstrap over per-item metric values, aligned by index —
    `base[i]` and `head[i]` must be the same item.

    Raises `ValueError` if `base` and `head` differ in length or are
    empty: a bootstrap over zero items, or over two arms that do not name
    the same items pairwise, is not a measurement of anything.

    `delta = mean(head - base)` is the observed effect. Each of the `b`
    bootstrap replicates resamples *item indices* — not `base` and `head`
    separately — with replacement via `numpy.random.default_rng(seed)`,
    and reads both arms of the difference vector `d = head - base` through
    that one shared resampled index. This is what "paired" means
    operationally: preserving the pairing is what lets the bootstrap
    exploit the base/head correlation on a shared suite rather than
    treating the two arms as independent samples (see this module's
    docstring, point 3).

    The confidence interval is the plain percentile method: the
    `[100*alpha/2, 100*(1-alpha/2)]` percentiles of the `b` bootstrap
    replicate means. `se` is the bootstrap standard deviation of those
    same replicate means (`ddof=1`).

    The p-value is two-sided and null-centred: `shifted = boot_means -
    delta` re-centres the bootstrap distribution on zero (the null of "no
    effect"), and `p_value = (1 + count(|shifted| >= |delta|)) / (b + 1)`
    counts how often a null-centred replicate is at least as extreme as
    the observed `delta` itself. **The `+1` in both numerator and
    denominator is deliberate, not a smoothing nicety**: without it, a
    `delta` more extreme than every one of the `b` replicates would
    report `p_value = 0`, which both overstates the evidence (a bootstrap
    with finite `b` can never truly rule out `p = 0`) and is a value the
    Holm-Bonferroni step downstream cannot sensibly scale — `0` stays `0`
    under any multiplier. Adding the observed statistic itself as one more
    member of its own reference distribution makes `p_value = 0`
    structurally impossible while leaving the p-value's usual
    interpretation and calibration intact for any `b` worth running.
    """
    if len(base) != len(head):
        raise ValueError(
            f"paired_bootstrap: base and head must be the same length, got {len(base)} and {len(head)}"
        )
    n = len(base)
    if n == 0:
        raise ValueError("paired_bootstrap: base and head must not be empty")

    base_arr = np.asarray(base, dtype=float)
    head_arr = np.asarray(head, dtype=float)
    d = head_arr - base_arr
    delta = float(np.mean(d))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    boot_means = d[idx].mean(axis=1)

    percentiles = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ci_low = float(percentiles[0])
    ci_high = float(percentiles[1])
    se = float(np.std(boot_means, ddof=1))

    shifted = boot_means - delta
    extreme_count = int(np.sum(np.abs(shifted) >= abs(delta)))
    p_value = (1 + extreme_count) / (b + 1)

    return BootstrapResult(
        n=n, delta=delta, ci_low=ci_low, ci_high=ci_high, se=se, p_value=p_value, b=b, seed=seed
    )


def mde(se: float, *, alpha: float, power: float = 0.80) -> float:
    """Minimum detectable effect: `(z(1 - alpha/2) + z(power)) * se`, the
    standard paired-design power formula, `z` read from the standard
    normal quantile function (`statistics.NormalDist().inv_cdf` — stdlib,
    not scipy, matching `items.ppi`'s own purity rule).

    **Plain interpretation, stated because it is the number a threshold
    gets checked against: an effect smaller than this cannot be reliably
    distinguished from noise on this suite at this sample size.** A gate
    threshold set below `mde` is therefore not a threshold the suite can
    enforce — it is a threshold that will pass or block on noise with
    unacceptable frequency regardless of what the system under test
    actually did, which is exactly the failure docs/SPEC.md §8 documents
    (a 1-point threshold on a suite whose detection floor was several
    points). `evaluate_gate` calls this function and refuses outright
    (`ThresholdBelowMDEError`) rather than silently reporting a verdict
    computed against an unenforceable threshold.

    Raises `ValueError` unless `0 < alpha < 1` and `0.5 < power < 1` —
    outside those ranges `inv_cdf` is undefined or the "power" input no
    longer means a power (a detection probability below 0.5 is not a
    meaningful design target).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"mde: alpha must be in (0, 1), got {alpha!r}")
    if not (0.5 < power < 1.0):
        raise ValueError(f"mde: power must be in (0.5, 1), got {power!r}")
    dist = statistics.NormalDist()
    z_alpha = dist.inv_cdf(1 - alpha / 2)
    z_power = dist.inv_cdf(power)
    return (z_alpha + z_power) * se


def holm_bonferroni(p_values: Sequence[float]) -> tuple[float, ...]:
    """Standard Holm (step-down Bonferroni) adjustment, returned in the
    same order as `p_values` (not sorted) so a caller can zip the result
    straight back against the metrics it came from.

    For `m = len(p_values)`, sorted ascending as `p_(0) <= ... <=
    p_(m-1)` (0-based rank `j`), the raw step-down adjustment is
    `adj_(j) = (m - j) * p_(j)`. Each raw value is clamped to `1.0` (a
    p-value can never exceed 1) and then carried forward via a running
    maximum over increasing rank, which is what makes the *sequence* of
    adjusted p-values monotone non-decreasing — required for Holm's
    step-down procedure to control the family-wise error rate correctly;
    without it, a smaller raw multiplier later in the sorted order could
    produce an adjusted p-value *smaller* than an earlier, more
    significant rank's, which would make "reject while `p_adjusted <
    alpha`" an incoherent rule to apply metric-by-metric. Clamping before
    the running max rather than after is equivalent (`min(max(a, b), c) ==
    max(min(a, c), min(b, c))` for the clamp `c = 1.0`) but is done first
    here to keep every intermediate value already inside `[0, 1]`.

    Every adjusted value is `>= ` its corresponding raw `p_value` (the
    smallest possible multiplier, at the largest rank, is `1`, and the
    running max can only move an adjusted value up from there). Empty
    input returns an empty tuple.
    """
    m = len(p_values)
    if m == 0:
        return ()
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        capped = min((m - rank) * p_values[i], 1.0)
        running_max = max(running_max, capped)
        adjusted[i] = running_max
    return tuple(adjusted)


class MetricSpec(BaseModel):
    """One metric `evaluate_gate` is asked to check.

    `threshold` is the regression size the gate claims to catch, in
    metric units (`0.05` means "block a 5-point drop"), and must be `>
    0` — `evaluate_gate` enforces that (see its docstring, refusal step
    1), since a non-positive threshold has no meaningful "regression at
    least this large" reading. `higher_is_better` names which direction
    of `delta` (`head - base`) is a regression: `True` (the default,
    e.g. a quality metric like `correctness`) means a drop of at least
    `threshold` is a regression; `False` (e.g. a cost or latency metric)
    means a *rise* of at least `threshold` is.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    threshold: float
    higher_is_better: bool = True


class GateMetricResult(BaseModel):
    """One metric's full gate result: the paired-bootstrap statistics,
    the Holm-adjusted p-value, the MDE it was checked against, and the
    resulting per-metric verdict — everything `put_gate_report` persists
    as one `gate_check` row (see `store/db.py`).

    `ci_low`/`ci_high` are the family-wise (simultaneous) interval at the
    multiplicity-corrected `alpha / m`, not a per-comparison 95% interval
    — see `evaluate_gate`'s docstring, step 2, for why that is the level
    that belongs beside a Holm-corrected verdict.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    n: int
    base_mean: float
    head_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    mde: float
    threshold: float
    verdict: Literal["pass", "block"]


class GateReport(BaseModel):
    """The full result of one `evaluate_gate` call: every metric's
    `GateMetricResult` plus the run/suite identity and the statistical
    configuration (`alpha`, `power`, `b`, `seed`) that produced it, so the
    report is self-describing and reproducible without any external
    context. `blocked` is `True` iff any `results[i].verdict == "block"`
    — the single bit a CI job needs to decide pass/fail."""

    model_config = ConfigDict(frozen=True)

    base_run_id: str
    head_run_id: str
    suite_id: str
    alpha: float
    power: float
    b: int
    seed: int
    results: tuple[GateMetricResult, ...]
    blocked: bool


class ThresholdBelowMDEError(Exception):
    """Raised by `evaluate_gate` when a `MetricSpec.threshold` is smaller
    than what this run's sample size can actually resolve
    (`threshold < mde`, `mde` computed at the multiplicity-corrected
    `alpha / m` — see `evaluate_gate`'s docstring, point 2).

    Carries `metric`, `threshold` and `mde` so a caller can report exactly
    which metric refused and by how much, without re-deriving anything.
    Raising this means no `GateReport` was produced for this call at
    all — refusing is not the same as passing, and must never be
    reported as one (see this module's docstring, point 1, and
    `test_gate_refuses_threshold_below_mde` /
    `test_refusal_precedes_verdict_even_with_a_real_regression` in
    `tests/test_gate.py`).
    """

    def __init__(self, metric: str, threshold: float, mde: float) -> None:
        self.metric = metric
        self.threshold = threshold
        self.mde = mde
        super().__init__(
            f"gate refuses to check metric {metric!r}: threshold {threshold!r} is below the "
            f"minimum detectable effect {mde!r} this suite can resolve at this sample size and "
            "multiplicity-corrected alpha -- a threshold the suite cannot distinguish from noise "
            "is not one this gate can honestly enforce, so no verdict was reached"
        )


def _metric_seed(seed: int, metric: str) -> int:
    """Derive a per-metric bootstrap seed from the report-level `seed`
    and the metric name, so each metric's `paired_bootstrap` resamples
    independently while the whole `GateReport` stays reproducible from
    one integer.

    Deliberately not Python's builtin `hash()`: `hash(str)` is randomly
    salted per process (for hash-flooding resistance), so it gives a
    *different* answer for the same metric name in two different
    processes — the opposite of what a reproducible seed needs.
    `hashlib.blake2b` is a stable, unsalted digest: the metric name is
    hashed to 4 bytes, read as a big-endian unsigned int, and XORed with
    `seed`. XOR keeps the result a well-formed integer seed and only needs
    to satisfy one property here — a different metric name reliably gives
    a different seed — which a 32-bit digest already provides for any
    realistic number of metrics on one gate.
    """
    digest = hashlib.blake2b(metric.encode("utf-8"), digest_size=4).digest()
    return seed ^ int.from_bytes(digest, "big")


def evaluate_gate(
    *,
    base_run_id: str,
    head_run_id: str,
    suite_id: str,
    per_item: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    specs: Sequence[MetricSpec],
    alpha: float = 0.05,
    power: float = 0.80,
    b: int = 10000,
    seed: int = 0,
) -> GateReport:
    """Run the gate: paired-bootstrap significance test + Holm-Bonferroni
    correction + MDE refusal, over every metric in `specs`.

    `per_item[metric]` is `(base_values, head_values)`, aligned by item
    index — the same contract `paired_bootstrap` takes, one pair per
    metric in `specs`.

    In order:

    1. **Validate.** Raises `ValueError` if `specs` is empty, if any
       `spec.metric` is missing from `per_item`, or if any
       `spec.threshold <= 0`. (Per-metric length/emptiness of
       `per_item[metric]`'s two sequences is validated by
       `paired_bootstrap` itself, below.)
    2. **Compute the multiplicity-corrected MDE.** With `m = len(specs)`,
       every metric's MDE is computed at `alpha / m`, not the raw
       `alpha` — the whole point being that a gate which Holm-corrects
       its verdict for `m` comparisons has strictly less power per metric
       than an uncorrected single-metric gate would, and its *honestly
       reported* detection floor must reflect that cost, not the floor of
       a hypothetical uncorrected gate it is not running (see this
       module's docstring, point 2, and
       `test_mde_grows_when_metrics_are_added` in `tests/test_gate.py`).
       The bootstrap CI reported on each `GateMetricResult` is computed at
       that same corrected `alpha / m`, making it a *simultaneous*
       (family-wise) interval — the interval that belongs beside a
       family-wise-corrected verdict, not a per-comparison one. This buys
       a coherence guarantee in the direction that matters: a Bonferroni
       interval at `alpha / m` excludes zero exactly when a Bonferroni
       test rejects at `alpha / m`, and Holm is uniformly at least as
       powerful as Bonferroni, so **an interval that excludes zero can
       never sit beside a metric the gate declined to find significant**
       — a reader can never catch the report contradicting its own
       verdict in that direction. The converse is not guaranteed, and is
       documented rather than hidden: Holm can reject where the
       `alpha / m` interval still contains zero, because Holm has no
       closed-form simultaneous interval. This residual asymmetry errs
       safe — the interval stays conservative about what was measured
       while the verdict uses the more powerful test. A `block` beside an
       interval containing zero is therefore possible and is not a bug.
       The CI level is not stored as its own field; it is
       `alpha / len(results)` and is recoverable from any persisted
       report.
    3. **Bootstrap each metric.** For each `spec` (in order), run
       `paired_bootstrap` at level `alpha / m`, seeded independently per
       metric via `_metric_seed(seed, spec.metric)`, and compute
       `mde(result.se, alpha=alpha / m, power=power)`.
    4. **Refuse before any verdict.** If `spec.threshold < mde` for *any*
       metric, raise `ThresholdBelowMDEError` for the first such metric
       in `specs` order. No `GateReport` is produced and nothing is
       persisted — refusing is not the same as passing (this module's
       docstring, point 1).
    5. **Holm-adjust.** `holm_bonferroni` over every metric's `p_value`,
       in `specs` order.
    6. **Verdict per metric.** `block` iff `p_adjusted < alpha` **and**
       the regression is at least as large as `threshold` in the harmful
       direction: `delta <= -threshold` when `higher_is_better`, else
       `delta >= threshold`. A statistically significant delta that is
       smaller than `threshold` is a `pass` — significance alone is not
       what the gate blocks on; a `threshold`-sized-or-larger regression
       that is not yet statistically significant is also a `pass`, since
       reporting `block` on an effect indistinguishable from noise is
       exactly the failure mode this whole module exists to prevent.
    7. `blocked = any(result.verdict == "block" for result in results)`.
    """
    if not specs:
        raise ValueError("evaluate_gate: specs must not be empty")
    for spec in specs:
        if spec.metric not in per_item:
            raise ValueError(f"evaluate_gate: metric {spec.metric!r} is missing from per_item")
        if spec.threshold <= 0:
            raise ValueError(
                f"evaluate_gate: metric {spec.metric!r} has threshold={spec.threshold!r}, must be > 0"
            )

    m = len(specs)
    corrected_alpha = alpha / m

    boot_results: list[BootstrapResult] = []
    mdes: list[float] = []
    for spec in specs:
        base_values, head_values = per_item[spec.metric]
        metric_seed = _metric_seed(seed, spec.metric)
        boot = paired_bootstrap(base_values, head_values, b=b, alpha=corrected_alpha, seed=metric_seed)
        boot_results.append(boot)
        mdes.append(mde(boot.se, alpha=corrected_alpha, power=power))

    for spec, metric_mde in zip(specs, mdes, strict=True):
        if spec.threshold < metric_mde:
            raise ThresholdBelowMDEError(spec.metric, spec.threshold, metric_mde)

    p_adjusted = holm_bonferroni([boot.p_value for boot in boot_results])

    results: list[GateMetricResult] = []
    for spec, boot, metric_mde, p_adj in zip(specs, boot_results, mdes, p_adjusted, strict=True):
        base_values, head_values = per_item[spec.metric]
        base_mean = float(np.mean(np.asarray(base_values, dtype=float)))
        head_mean = float(np.mean(np.asarray(head_values, dtype=float)))
        if spec.higher_is_better:
            is_regression = boot.delta <= -spec.threshold
        else:
            is_regression = boot.delta >= spec.threshold
        verdict: Literal["pass", "block"] = "block" if (p_adj < alpha and is_regression) else "pass"
        results.append(
            GateMetricResult(
                metric=spec.metric,
                n=boot.n,
                base_mean=base_mean,
                head_mean=head_mean,
                delta=boot.delta,
                ci_low=boot.ci_low,
                ci_high=boot.ci_high,
                p_value=boot.p_value,
                p_adjusted=p_adj,
                mde=metric_mde,
                threshold=spec.threshold,
                verdict=verdict,
            )
        )

    blocked = any(result.verdict == "block" for result in results)
    return GateReport(
        base_run_id=base_run_id,
        head_run_id=head_run_id,
        suite_id=suite_id,
        alpha=alpha,
        power=power,
        b=b,
        seed=seed,
        results=tuple(results),
        blocked=blocked,
    )
