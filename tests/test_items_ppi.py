"""`test_items_ppi.py`: `items.ppi` implements Prediction-Powered Inference
(handbook §9b) -- the mechanism that makes a ~65%-precision/recall judge
usable at all by combining it with a small unbiased human sample.

Tests 2 and 6 are the point of the module: a deliberately *biased* judge and
a deliberately *useless* one must both still yield an unbiased estimate
under simulation. This is demonstrated here, not asserted in a docstring --
see each test's own docstring for its seed and tolerance.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from fireassay.items.ppi import PPIStratum, ppi_mean_ci, stratified_ppi_mean_ci

# Matches `_z_for_alpha`'s own stdlib computation -- used only to build an
# independent classical (normal-approximate) interval in test 4, never
# imported from `items.ppi` itself.
_Z_95 = 1.959963984540054


# -- 1. cross-check against ppi_py (test-only reference implementation) -----


def test_matches_ppi_py_reference_implementation() -> None:
    """Classical PPI (`lam=1.0`, fixed explicitly on both sides) so the
    comparison isolates the point-estimate/CI formula itself from any
    difference in how the two packages auto-tune `lam` when it is left
    unset. Seed 1, n=200 labeled, n_unlab=2000 unlabeled, alpha=0.1."""
    ppi_py = pytest.importorskip("ppi_py")
    rng = np.random.default_rng(1)
    n, n_unlab = 200, 2000
    y_true = rng.normal(loc=5.0, scale=2.0, size=n_unlab + n)
    judge_noise = rng.normal(loc=0.3, scale=1.0, size=n_unlab + n)  # biased +0.3, noisy
    f_all = y_true + judge_noise
    y_labeled = y_true[:n]
    f_labeled = f_all[:n]
    f_unlabeled = f_all[n:]

    ours = ppi_mean_ci(y_labeled.tolist(), f_labeled.tolist(), f_unlabeled.tolist(), alpha=0.1, lam=1.0)
    theirs_lo, theirs_hi = ppi_py.ppi_mean_ci(y_labeled, f_labeled, f_unlabeled, alpha=0.1, lam=1.0)

    assert ours.value is not None
    expected_point = float(np.mean(f_unlabeled)) + float(np.mean(y_labeled - f_labeled))
    assert ours.value == pytest.approx(expected_point, abs=1e-9)

    assert ours.ci is not None
    assert ours.ci[0] == pytest.approx(theirs_lo, rel=1e-2)
    assert ours.ci[1] == pytest.approx(theirs_hi, rel=1e-2)


# -- 2. unbiasedness under a DELIBERATELY BIASED judge -----------------------


def test_biased_judge_ppi_concentrates_on_truth_naive_mean_does_not() -> None:
    """Deliberately biased judge: flips the true binary label with
    probability 0.3. For `p_true != 0.5` this shifts `E[f]` away from
    `E[y]` -- with `p_true=0.2`: `E[f] = 0.3*(1-0.2) + 0.7*0.2 = 0.38`,
    nearly double the truth. Over many replications the naive judge-only
    mean concentrates on that biased 0.38, not on the truth; the PPI point
    estimate, built from the very same biased judge, concentrates on the
    truth. This is the property `items.ppi` exists for, demonstrated under
    simulation rather than asserted.

    Seed 2, 800 replications, n_labeled=100, N_unlabeled=2000. Tolerance
    0.03 on the truth (the naive judge's own gap is ~0.18, an order of
    magnitude larger) and 0.02 on the analytically predicted biased target
    -- both pinned once here, not tuned to whatever a run happens to
    print.
    """
    rng = np.random.default_rng(2)
    p_true = 0.2
    flip_prob = 0.3
    n, n_unlabeled = 100, 2000
    replications = 800

    # E[f] = flip_prob*(1 - p_true) + (1 - flip_prob)*p_true
    expected_naive_target = flip_prob * (1 - p_true) + (1 - flip_prob) * p_true

    ppi_estimates: list[float] = []
    naive_means: list[float] = []
    for _ in range(replications):
        y_labeled = rng.binomial(1, p_true, size=n).astype(float)
        y_unlabeled_truth = rng.binomial(1, p_true, size=n_unlabeled).astype(float)
        flip_labeled = rng.uniform(size=n) < flip_prob
        flip_unlabeled = rng.uniform(size=n_unlabeled) < flip_prob
        f_labeled = np.where(flip_labeled, 1.0 - y_labeled, y_labeled)
        f_unlabeled = np.where(flip_unlabeled, 1.0 - y_unlabeled_truth, y_unlabeled_truth)

        result = ppi_mean_ci(y_labeled.tolist(), f_labeled.tolist(), f_unlabeled.tolist())
        assert result.value is not None
        ppi_estimates.append(result.value)
        naive_means.append(float(np.mean(f_unlabeled)))

    mean_ppi = float(np.mean(ppi_estimates))
    mean_naive = float(np.mean(naive_means))

    assert abs(mean_ppi - p_true) < 0.03
    assert abs(mean_naive - expected_naive_target) < 0.02
    assert abs(mean_naive - p_true) > 0.1
    assert abs(mean_ppi - p_true) < abs(mean_naive - p_true)


# -- 3. coverage --------------------------------------------------------------


def test_coverage_of_95_percent_interval_is_approximately_95_percent() -> None:
    """Seed 3, 1000 replications, n_labeled=100, N_unlabeled=2000, a
    biased+noisy judge, alpha=0.05. Tolerance: the coverage estimate is a
    proportion over 1000 Bernoulli trials, so at true coverage 0.95 its own
    standard error is `sqrt(0.95*0.05/1000) ~= 0.0069`. The asserted band,
    0.95 +/- 0.03, is about 4.3 of those standard errors -- wide enough to
    absorb the normal approximation's own small expected under-coverage at
    this `n` without absorbing a real defect."""
    rng = np.random.default_rng(3)
    mu_true = 10.0
    n, n_unlabeled = 100, 2000
    replications = 1000
    covered = 0
    for _ in range(replications):
        y_true_all = rng.normal(loc=mu_true, scale=3.0, size=n + n_unlabeled)
        judge_noise = rng.normal(loc=0.5, scale=1.5, size=n + n_unlabeled)  # biased, noisy
        f_all = y_true_all + judge_noise
        y_labeled = y_true_all[:n]
        f_labeled = f_all[:n]
        f_unlabeled = f_all[n:]

        result = ppi_mean_ci(y_labeled.tolist(), f_labeled.tolist(), f_unlabeled.tolist(), alpha=0.05)
        assert result.ci is not None
        lo, hi = result.ci
        if lo <= mu_true <= hi:
            covered += 1

    coverage = covered / replications
    assert 0.92 <= coverage <= 0.98


# -- 4. beats the human-only estimate -----------------------------------------


def test_ppi_interval_narrower_than_human_only_classical_interval() -> None:
    """With a genuinely informative judge (low-noise, unbiased), the PPI
    interval must be narrower than the classical normal-approximate
    interval computed from `y_labeled` alone -- if it were not, the module
    would be pointless. The classical interval is computed independently
    here (`mean +/- z*sqrt(var(y_labeled)/n)`), not by calling into
    `items.ppi` with `lam=0`, so the comparison is not circular. Seed 4."""
    rng = np.random.default_rng(4)
    n, n_unlabeled = 60, 3000
    y_true_all = rng.normal(loc=10.0, scale=3.0, size=n + n_unlabeled)
    judge_noise = rng.normal(loc=0.0, scale=0.5, size=n + n_unlabeled)  # good judge
    f_all = y_true_all + judge_noise
    y_labeled = y_true_all[:n]
    f_labeled = f_all[:n]
    f_unlabeled = f_all[n:]

    result = ppi_mean_ci(y_labeled.tolist(), f_labeled.tolist(), f_unlabeled.tolist(), alpha=0.05)
    assert result.ci is not None

    human_only_se = math.sqrt(float(np.var(y_labeled)) / n)
    human_only_width = 2 * _Z_95 * human_only_se
    ppi_width = result.ci[1] - result.ci[0]

    assert ppi_width < human_only_width


# -- 5. perfect judge collapses the rectifier ---------------------------------


def test_perfect_judge_collapses_rectifier_to_mean_of_unlabeled() -> None:
    """`f == y` exactly on the labeled sample. `lam=1.0` is passed
    explicitly (classical PPI) rather than left to PPI++'s variance-optimal
    tuning: the "rectifier collapses to zero" property is a property of the
    classical rectifier `mean(y - f_labeled)`, not of whatever `lam*` the
    variance-minimiser happens to pick for a given `f_unlabeled` (which need
    not be 1)."""
    y_labeled = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    f_labeled = list(y_labeled)  # perfect judge on the labeled sample
    f_unlabeled = [0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.55, 0.35, 0.7, 0.25]

    result = ppi_mean_ci(y_labeled, f_labeled, f_unlabeled, lam=1.0)

    assert result.value == pytest.approx(float(np.mean(f_unlabeled)))


# -- 6. useless judge: still unbiased, wider, never wrong ---------------------


def test_useless_judge_still_unbiased_but_wider_than_an_informative_one() -> None:
    """A judge that is pure noise, independent of the truth, must still
    yield an unbiased PPI estimate under PPI++'s own variance-optimal
    tuning (`lam=None`, the default) -- the auto-tuned `lam` should shrink
    toward the human-only estimator rather than let a useless judge corrupt
    `theta_hat`. Compared, over the same replications, against a genuinely
    informative judge (85% accurate): both concentrate on the truth, but
    the useless-judge estimator's average interval is wider -- "wider, but
    not wrong", the "provably unbiased regardless of the judge's error
    profile" claim in its strongest form. Seed 6, 600 replications,
    n_labeled=80, N_unlabeled=2000."""
    rng = np.random.default_rng(6)
    p_true = 0.3
    n, n_unlabeled = 80, 2000
    replications = 600

    useless_estimates: list[float] = []
    useless_widths: list[float] = []
    informative_estimates: list[float] = []
    informative_widths: list[float] = []

    for _ in range(replications):
        y_labeled = rng.binomial(1, p_true, size=n).astype(float)
        y_unlabeled_truth = rng.binomial(1, p_true, size=n_unlabeled).astype(float)

        # useless judge: pure noise, unrelated to y, same noise-generating
        # process ("same judge, same configuration") on both arms.
        f_labeled_noise = rng.uniform(0.0, 1.0, size=n)
        f_unlabeled_noise = rng.uniform(0.0, 1.0, size=n_unlabeled)
        est_noise = ppi_mean_ci(y_labeled.tolist(), f_labeled_noise.tolist(), f_unlabeled_noise.tolist())
        assert est_noise.value is not None and est_noise.ci is not None
        useless_estimates.append(est_noise.value)
        useless_widths.append(est_noise.ci[1] - est_noise.ci[0])

        # informative judge: 85% accurate, same configuration both arms.
        flip_labeled = rng.uniform(size=n) < 0.15
        flip_unlabeled = rng.uniform(size=n_unlabeled) < 0.15
        f_labeled_good = np.where(flip_labeled, 1.0 - y_labeled, y_labeled)
        f_unlabeled_good = np.where(flip_unlabeled, 1.0 - y_unlabeled_truth, y_unlabeled_truth)
        est_good = ppi_mean_ci(y_labeled.tolist(), f_labeled_good.tolist(), f_unlabeled_good.tolist())
        assert est_good.value is not None and est_good.ci is not None
        informative_estimates.append(est_good.value)
        informative_widths.append(est_good.ci[1] - est_good.ci[0])

    assert abs(float(np.mean(useless_estimates)) - p_true) < 0.03
    assert abs(float(np.mean(informative_estimates)) - p_true) < 0.03
    assert float(np.mean(useless_widths)) > float(np.mean(informative_widths))


# -- 7. guard rails ------------------------------------------------------------


def test_guard_rail_length_mismatch_names_both_lengths() -> None:
    with pytest.raises(ValueError, match="y_labeled has 3 item"):
        ppi_mean_ci([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [5.0])


def test_guard_rail_labeled_empty() -> None:
    with pytest.raises(ValueError, match="y_labeled is empty"):
        ppi_mean_ci([], [], [1.0, 2.0])


def test_guard_rail_unlabeled_empty() -> None:
    with pytest.raises(ValueError, match="f_unlabeled is empty"):
        ppi_mean_ci([1.0], [1.0], [])


def test_guard_rail_stratified_weights_must_sum_to_one_names_actual_sum() -> None:
    expected_sum = 0.5 + 0.2
    strata = [
        PPIStratum(weight=0.5, y_labeled=(1.0,), f_labeled=(1.0,), f_unlabeled=(1.0, 2.0)),
        PPIStratum(weight=0.2, y_labeled=(0.0,), f_labeled=(0.0,), f_unlabeled=(0.0, 1.0)),
    ]
    with pytest.raises(ValueError, match=rf"got {re.escape(repr(expected_sum))}"):
        stratified_ppi_mean_ci(strata)


def test_guard_rail_stratified_propagates_per_stratum_labeled_empty() -> None:
    strata = [
        PPIStratum(weight=1.0, y_labeled=(), f_labeled=(), f_unlabeled=(1.0, 2.0)),
    ]
    with pytest.raises(ValueError, match="y_labeled is empty"):
        stratified_ppi_mean_ci(strata)


# -- 8. stratified with one stratum reduces exactly to ppi_mean_ci -----------


def test_stratified_with_one_stratum_matches_ppi_mean_ci_exactly() -> None:
    rng = np.random.default_rng(8)
    n, n_unlabeled = 40, 500
    y_true = rng.normal(loc=3.0, scale=1.0, size=n + n_unlabeled)
    f_all = y_true + rng.normal(loc=0.2, scale=0.7, size=n + n_unlabeled)
    y_labeled = tuple(y_true[:n].tolist())
    f_labeled = tuple(f_all[:n].tolist())
    f_unlabeled = tuple(f_all[n:].tolist())

    direct = ppi_mean_ci(y_labeled, f_labeled, f_unlabeled)
    stratified = stratified_ppi_mean_ci(
        [PPIStratum(weight=1.0, y_labeled=y_labeled, f_labeled=f_labeled, f_unlabeled=f_unlabeled)]
    )

    assert direct.value is not None and stratified.value is not None
    assert direct.ci is not None and stratified.ci is not None
    assert stratified.value == pytest.approx(direct.value)
    assert stratified.ci[0] == pytest.approx(direct.ci[0])
    assert stratified.ci[1] == pytest.approx(direct.ci[1])
    assert stratified.n == direct.n


# -- 9. stratified beats unstratified when judge accuracy differs sharply ----


def test_stratified_beats_unstratified_when_judge_accuracy_differs_sharply() -> None:
    """Two equal-size strata: stratum A's judge is highly accurate (low
    noise), stratum B's judge is nearly useless (high noise). A single
    global `lam`, fit by pooling both strata together, is a compromise that
    fits neither stratum's judge well; a separate `lam` per stratum fits
    each -- StratPPI's whole justification (handbook §9b: judge accuracy
    measured at 7.3%/58.8%/83.3% by verdict on the real data). Seed 9."""
    rng = np.random.default_rng(9)
    n, n_unlabeled = 60, 1500
    mu_true = 5.0

    y_true_a = rng.normal(loc=mu_true, scale=2.0, size=n + n_unlabeled)
    f_a = y_true_a + rng.normal(loc=0.0, scale=0.3, size=n + n_unlabeled)  # accurate judge
    y_true_b = rng.normal(loc=mu_true, scale=2.0, size=n + n_unlabeled)
    f_b = y_true_b + rng.normal(loc=0.0, scale=8.0, size=n + n_unlabeled)  # near-useless judge

    strata = [
        PPIStratum(
            weight=0.5,
            y_labeled=tuple(y_true_a[:n].tolist()),
            f_labeled=tuple(f_a[:n].tolist()),
            f_unlabeled=tuple(f_a[n:].tolist()),
        ),
        PPIStratum(
            weight=0.5,
            y_labeled=tuple(y_true_b[:n].tolist()),
            f_labeled=tuple(f_b[:n].tolist()),
            f_unlabeled=tuple(f_b[n:].tolist()),
        ),
    ]
    stratified = stratified_ppi_mean_ci(strata)

    y_pooled = y_true_a[:n].tolist() + y_true_b[:n].tolist()
    f_pooled = f_a[:n].tolist() + f_b[:n].tolist()
    f_unlabeled_pooled = f_a[n:].tolist() + f_b[n:].tolist()
    pooled = ppi_mean_ci(y_pooled, f_pooled, f_unlabeled_pooled)

    assert stratified.ci is not None and pooled.ci is not None
    stratified_width = stratified.ci[1] - stratified.ci[0]
    pooled_width = pooled.ci[1] - pooled.ci[0]
    assert stratified_width < pooled_width


# -- 10. purity: module graph excludes fireassay.store / fireassay.llm ------
#
# See `tests/test_items_no_store_import.py`:
# `test_items_ppi_module_graph_excludes_store_and_llm` and
# `test_sanity_the_walker_follows_items_ppis_own_cross_module_import` --
# following the existing pair pattern used for `items.calibration` and
# `items.answerability` in that file.
