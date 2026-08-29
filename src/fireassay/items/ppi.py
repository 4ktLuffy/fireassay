"""Prediction-Powered Inference (PPI / PPI++) -- combines a small
human-labelled sample with a large judge-labelled one into a single
estimate that is *provably unbiased regardless of how wrong the judge is*
(`~/ObitosBrain/handbook/eval-of-evals.md` §9b). This is the mechanism
that makes a mediocre judge worth running at all: a judge alone is large but
biased (measured at roughly 65% precision / 65% recall here, over all 2,364
items); a human sample alone is unbiased but small (176 labels); PPI
combines them into an estimate that beats either used alone, with a
tighter interval than the human sample can give by itself.

**Pure, per §7b.** Imports only `items.core` (for `Estimate`) plus
numpy/pydantic (and the stdlib) -- `test_items_no_store_import.py` walks
this module's static import graph and asserts it never reaches
`fireassay.store` or `fireassay.llm`.

## The estimator

The classical PPI point estimate for a mean:

    theta_hat = mean(f_unlabeled) + mean(y_labeled - f_labeled)
                └ judge on the big set ┘  └───── the rectifier ─────┘

`f_unlabeled` is the judge's prediction on everything -- large, cheap, and
possibly badly biased. The rectifier is measured only on the small
human-labelled sample: how wrong the judge is, on average, where the truth
is known -- and subtracts that bias back off. **The rectifier's expectation
is exactly the judge's own bias on the labelled sample, so `theta_hat`'s
expectation is `E[f_unlabeled] + (E[y] - E[f_labeled]) = E[y]` regardless of
how biased or noisy `f` is**, as long as `f_labeled` and `f_unlabeled` share
the same distribution (see "The one assumption" below) -- that is the
"provably unbiased regardless of the judge's error profile" claim, and it
is what tests 2 and 6 in `tests/test_items_ppi.py` demonstrate under
simulation, not merely assert.

PPI++ generalises this with a tuning parameter `lam`:

    theta_hat(lam) = lam * mean(f_unlabeled) + mean(y_labeled - lam * f_labeled)

`lam = 1` recovers classical PPI exactly. `lam = 0` recovers the plain
human-sample mean (the judge is ignored entirely) -- a useful sanity bound,
since a well-tuned PPI estimate should never do *worse* than that. When
`lam` is not supplied, it is chosen to minimise the estimator's variance
(`_optimal_lambda`) -- the "++" half of the name.

**Variance.** Treating the labelled and unlabelled samples as independent
draws (they always are here -- disjoint items), `Var(theta_hat(lam))` is the
sum of two independent terms: the judge's own variance on the big
unlabelled set, scaled by `lam**2/N`, plus the rectifier's variance on the
small labelled set, scaled by `1/n`:

    Var(theta_hat(lam)) = lam**2 * Var(f_unlabeled)/N + Var(y_labeled - lam*f_labeled)/n

Minimising this over `lam` (set its derivative to zero and solve) gives the
closed form used whenever `lam=None`:

    lam* = Cov(y_labeled, f_labeled) / (Var(f_labeled) + (n/N)*Var(f_unlabeled))

-- population covariance/variance throughout (`ddof=0`; see
`_population_var`/`_population_cov`), the plug-in estimator the `/N`/`/n`
asymptotic variance formula above assumes. `lam* = 0` when the denominator
is exactly `0`: a judge with zero variance on both arms carries no
exploitable signal, and every `lam` gives the same `theta_hat` in that
degenerate case (see `_optimal_lambda`'s docstring), so `0.0` is a safe,
non-arbitrary fallback rather than a division by zero.

**The interval is asymptotic -- a normal approximation, `value +/- z * se`,
`z` the standard normal two-sided critical value at `alpha`**
(`statistics.NormalDist`, stdlib, not scipy -- see the purity rule above).
This is a materially different guarantee from `items.core.wilson_ci`, which
stays inside `[0, 1]` and is well-behaved even at very small `n` --
`ppi_mean_ci`'s interval has no such small-sample correction and can, in
principle, exceed `[0, 1]` or misbehave at very small `n`. Treat it as a
large-`n` approximation, the same way the PPI/PPI++ literature does; do not
read it as "exact-ish" the way `wilson_ci` is documented to be.

## The one assumption that silently invalidates everything

**`f_labeled` and `f_unlabeled` must come from the *same judge under the
same configuration*.** The rectifier `mean(y_labeled - lam*f_labeled)` is
only a valid estimate of the judge's bias *on the unlabelled set* if the
judge that produced `f_labeled` is the judge that produced `f_unlabeled` --
same model, same effort/reasoning setting, same prompt. A different model,
a different effort setting, or a reworded prompt on the two arms silently
invalidates the rectifier: the resulting `theta_hat` is still a real number
with a real-looking confidence interval, and nothing downstream would
notice that it no longer estimates anything true. This module cannot check
it -- there is no signal in two plain float arrays that could distinguish
"same judge, two arms" from "two different judges" -- so it is stated here,
prominently, instead.

## Hand-rolled, cross-checked

Decision 5 in the project notes: hand-roll the estimator (the same
discipline `pytrec_eval` is used for elsewhere -- a reference
implementation cross-checks, it does not run in production), cross-checked
against `ppi_py` (Angelopoulos et al.) kept **test-only**
(`tests/test_items_ppi.py`, `pytest.importorskip("ppi_py")`, so the suite
still runs without it installed). `ppi-py` is a dev-only dependency
(`pyproject.toml` `[project.optional-dependencies].dev`) -- never imported
by this module, or anything else at runtime.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from fireassay.items.core import Estimate

#: Tolerance for `stratified_ppi_mean_ci`'s "weights must sum to 1" guard
#: rail -- a documented convention for float summation drift, not a
#: measured threshold, mirroring `items.core._MIN_LIVE_ITEMS_PER_SPLIT` /
#: `items.calibration._MIN_DENOMINATOR`.
_WEIGHT_SUM_TOLERANCE = 1e-6


class PPIStratum(BaseModel):
    """One stratum's weight and PPI inputs for `stratified_ppi_mean_ci`.

    Named `PPIStratum`, not `Stratum`: `items.calibration.Stratum` is an
    unrelated type already living in this package (a
    `Literal["flagged", "unflagged", "calibration"]` review-sampling tag,
    nothing to do with this module's population strata) -- two unrelated
    types sharing a bare name reads fine in isolation and very badly the
    day something imports both."""

    model_config = ConfigDict(frozen=True)

    #: This stratum's share of the population (`w_k` in the handbook's
    #: notation). Every stratum's `weight` passed to
    #: `stratified_ppi_mean_ci` together must sum to `1.0` within
    #: `_WEIGHT_SUM_TOLERANCE`.
    weight: float
    y_labeled: tuple[float, ...]
    f_labeled: tuple[float, ...]
    f_unlabeled: tuple[float, ...]


def _population_var(x: np.ndarray) -> float:
    """Population variance (`ddof=0`) -- the plug-in estimator the
    asymptotic `/N`/`/n` variance formula in the module docstring assumes,
    not the `ddof=1` small-sample correction."""
    return float(np.var(x))


def _population_cov(x: np.ndarray, y: np.ndarray) -> float:
    """Population covariance (`ddof=0`) between two equal-length arrays."""
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def _optimal_lambda(y: np.ndarray, f: np.ndarray, f_unlabeled: np.ndarray) -> float:
    """The PPI++ variance-minimising `lam` (see the module docstring's
    derivation). `0.0` when the denominator is exactly `0` -- a judge with
    no variance on either arm carries no exploitable signal, and every
    `lam` gives an identical `theta_hat` in that case (see the module
    docstring), so `0.0` is a safe, non-arbitrary choice rather than a
    division by zero."""
    n = len(y)
    n_unlabeled = len(f_unlabeled)
    denom = _population_var(f) + (n / n_unlabeled) * _population_var(f_unlabeled)
    if denom == 0.0:
        return 0.0
    return _population_cov(y, f) / denom


def _validate_ppi_inputs(
    y_labeled: Sequence[float],
    f_labeled: Sequence[float],
    f_unlabeled: Sequence[float],
    *,
    fn_name: str,
) -> None:
    """Guard rails shared by `ppi_mean_ci` and `stratified_ppi_mean_ci`
    (applied per-stratum in the latter) -- see `ppi_mean_ci`'s docstring
    for what each one means."""
    if len(y_labeled) != len(f_labeled):
        raise ValueError(
            f"{fn_name}: y_labeled has {len(y_labeled)} item(s) but f_labeled has "
            f"{len(f_labeled)} -- they must be aligned item-for-item"
        )
    if len(y_labeled) == 0:
        raise ValueError(
            f"{fn_name}: y_labeled is empty -- there is no rectifier without at least one "
            "labeled item, so this is not PPI (there is no signal to correct the judge's bias "
            "with)"
        )
    if len(f_unlabeled) == 0:
        raise ValueError(
            f"{fn_name}: f_unlabeled is empty -- degenerate; the caller wants a plain "
            "human-sample CI over y_labeled instead, not PPI"
        )


def _ppi_point_and_variance(
    y: np.ndarray, f: np.ndarray, f_unlabeled: np.ndarray, lam: float | None
) -> tuple[float, float, float]:
    """`(theta_hat, variance, lam_used)` for one PPI/PPI++ arm -- shared by
    `ppi_mean_ci` (which wraps this in a normal-approximate CI) and
    `stratified_ppi_mean_ci` (which combines several arms' point estimates
    and variances directly, so it must not round-trip through a CI and
    reconstruct the variance from an already-rounded interval)."""
    n = len(y)
    n_unlabeled = len(f_unlabeled)
    effective_lam = _optimal_lambda(y, f, f_unlabeled) if lam is None else lam

    theta_hat = effective_lam * float(f_unlabeled.mean()) + float(np.mean(y - effective_lam * f))

    var_f_unlabeled = _population_var(f_unlabeled)
    var_rectifier = _population_var(y - effective_lam * f)
    variance = (effective_lam**2) * var_f_unlabeled / n_unlabeled + var_rectifier / n

    return theta_hat, variance, effective_lam


def _z_for_alpha(alpha: float) -> float:
    """Standard normal two-sided critical value at `alpha`, from the
    stdlib (`statistics.NormalDist`) rather than scipy -- see the module's
    purity rule."""
    return statistics.NormalDist().inv_cdf(1 - alpha / 2)


def ppi_mean_ci(
    y_labeled: Sequence[float],
    f_labeled: Sequence[float],
    f_unlabeled: Sequence[float],
    *,
    alpha: float = 0.05,
    lam: float | None = None,
) -> Estimate:
    """Prediction-Powered Inference estimate of a population mean (see the
    module docstring for the estimator, its variance, and -- most
    important -- the same-judge-same-configuration assumption that
    silently invalidates it if violated).

    `y_labeled` is the human label on each labelled item; `f_labeled` is
    the *same judge, under the same configuration* used for
    `f_unlabeled`'s predictions on those *same* labelled items, aligned
    index-for-index; `f_unlabeled` is that judge's predictions on
    everything else.

    `n` on the returned `Estimate` is `len(y_labeled)` -- the labelled
    count, not `len(f_unlabeled)`, because the labelled count is what
    bounds the rectifier and therefore the whole estimate's validity.

    `verdict` is always `"measured"`: every guard rail below raises before
    an `Estimate` is ever constructed, so there is no partial-success state
    to flag -- unlike `items.calibration.evaluate_detector`, which can
    report `"too_few_labels"` or `"no_denominator"` for an otherwise-valid
    call, `ppi_mean_ci` never partially succeeds.

    Raises `ValueError`, naming the offending value, when:

    - `len(y_labeled) != len(f_labeled)` -- they must be aligned
      item-for-item.
    - `y_labeled` (equivalently `f_labeled`) is empty -- there is no
      rectifier, so this is not PPI.
    - `f_unlabeled` is empty -- degenerate; the caller wants a plain
      human-sample CI instead.
    """
    _validate_ppi_inputs(y_labeled, f_labeled, f_unlabeled, fn_name="ppi_mean_ci")

    y = np.asarray(y_labeled, dtype=float)
    f = np.asarray(f_labeled, dtype=float)
    f_unlabeled_arr = np.asarray(f_unlabeled, dtype=float)

    theta_hat, variance, _lam_used = _ppi_point_and_variance(y, f, f_unlabeled_arr, lam)
    se = math.sqrt(max(variance, 0.0))
    z = _z_for_alpha(alpha)

    return Estimate(
        value=theta_hat,
        ci=(theta_hat - z * se, theta_hat + z * se),
        n=len(y_labeled),
        verdict="measured",
    )


def stratified_ppi_mean_ci(strata: Sequence[PPIStratum], *, alpha: float = 0.05) -> Estimate:
    """StratPPI (handbook §9b): a separate PPI estimate per stratum,
    combined -- the fix for a judge whose accuracy varies sharply by
    stratum, which ours demonstrably does (7.3% / 58.8% / 83.3% defect
    rates across its three verdicts). A single global `lam` fit by pooling
    every stratum together is a compromise that cannot be optimal for more
    than one of them; fitting `lam` separately per stratum (each
    `PPIStratum` uses `ppi_mean_ci`'s own `lam=None` variance-optimal
    tuning internally) and then combining is tighter.

    The combined point estimate is the weight-weighted sum of the
    per-stratum point estimates; the combined variance is the sum of
    `weight**2 * per_stratum_variance` (the strata are independent
    samples), from which the normal-approximate CI is built exactly as in
    `ppi_mean_ci`.

    Every `PPIStratum` in `strata` is independently subject to
    `ppi_mean_ci`'s own guard rails (aligned labelled arrays, non-empty
    labelled and unlabelled data) -- a violation in any one stratum raises
    from within that stratum's own validation.

    `strata`'s weights must sum to `1.0` within `_WEIGHT_SUM_TOLERANCE`;
    raises `ValueError` naming the actual sum otherwise. An empty `strata`
    sums to `0.0` and is caught by this same check -- there is no separate
    "at least one stratum" guard beyond it.

    `n` on the returned `Estimate` is the sum of every stratum's labelled
    count -- the natural generalisation of `ppi_mean_ci`'s own `n`
    convention (see its docstring) to more than one arm. With exactly one
    stratum (`weight=1.0`), this reduces exactly to calling `ppi_mean_ci`
    directly on that stratum's data: same `theta_hat`, same variance, same
    `n`, same interval."""
    weight_sum = sum(stratum.weight for stratum in strata)
    if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"stratified_ppi_mean_ci: stratum weights must sum to 1.0, got {weight_sum!r}")

    value = 0.0
    variance = 0.0
    n_total = 0
    for stratum in strata:
        _validate_ppi_inputs(
            stratum.y_labeled,
            stratum.f_labeled,
            stratum.f_unlabeled,
            fn_name="stratified_ppi_mean_ci",
        )
        y = np.asarray(stratum.y_labeled, dtype=float)
        f = np.asarray(stratum.f_labeled, dtype=float)
        f_unlabeled_arr = np.asarray(stratum.f_unlabeled, dtype=float)

        theta_hat, stratum_variance, _lam_used = _ppi_point_and_variance(y, f, f_unlabeled_arr, lam=None)
        value += stratum.weight * theta_hat
        variance += (stratum.weight**2) * stratum_variance
        n_total += len(stratum.y_labeled)

    se = math.sqrt(max(variance, 0.0))
    z = _z_for_alpha(alpha)

    return Estimate(
        value=value,
        ci=(value - z * se, value + z * se),
        n=n_total,
        verdict="measured",
    )
