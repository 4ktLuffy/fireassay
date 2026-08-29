"""`test_items_reliability.py` (M-ITEMS-SPEC.md §6): a matrix built to be
internally inconsistent yields `too_few_systems`; a consistent one yields
`usable`. Also the regression test for the zero-variance-inflation bug
found on the real 2,364-item panel (`_split_half_reliability`'s
docstring): padding a set of live items with any number of all-correct
items must not change the measured reliability.

**Consistent matrix**: a classical Guttman scale over 20 systems ranked
0 (ablest) .. 19 (weakest) -- item `k` passes exactly for the systems
whose rank is below a threshold spread evenly across `0..20`. Because
every item is a monotonic function of the *same* underlying rank, any
random half of the 20 systems still preserves that same monotonic
ordering internally, so each item's point-biserial correlation trends
strongly positive in *both* halves of *any* split -- this is about as
strong and reliable a split-half signal as a matrix can have, and is
expected to clear the default `reliability_floor=0.5` comfortably.

**Inconsistent matrix**: only 4 systems (the minimum `analyse` accepts)
and 8 items -- fewer than `core._MIN_LIVE_ITEMS_PER_SPLIT` (10), so every
split is excluded from the mean regardless of the items' own pattern
(`reliability` falls back to `0.0`). This is deliberately the more
robust, provably-correct way to exercise `too_few_systems`: a "genuinely
noisy at scale" matrix is not something this file asserts an exact
threshold for, since that would be a statistical claim about a specific
random draw that cannot be hand-verified the way a sample-size floor can.
"""

from __future__ import annotations

import pytest

from fireassay.items.core import ItemResponses, analyse


def _guttman_matrix(n_systems: int, n_items: int) -> list[ItemResponses]:
    """Item `k`'s pass set is the top `threshold` systems by rank (system
    index 0 = ablest), with `threshold` spread evenly across `1..n_systems
    - 1` -- a monotonic, noiseless "everyone above this ability level
    passes" scale."""
    responses = []
    for k in range(n_items):
        threshold = max(1, min(n_systems - 1, round((k + 1) * n_systems / (n_items + 1))))
        vec = tuple(i < threshold for i in range(n_systems))
        responses.append(ItemResponses(item_id=f"g{k}", responses=vec))
    return responses


def _contradictory_matrix() -> list[ItemResponses]:
    """4 systems, 8 items: half the items favour {s0, s1}, half favour
    {s2, s3}. At 8 items this is below `core._MIN_LIVE_ITEMS_PER_SPLIT`
    (10), so every split is excluded from the mean regardless of the
    pattern -- see this module's docstring for why that is the intended,
    provable way this file exercises `too_few_systems`."""
    responses = []
    for k in range(4):
        responses.append(ItemResponses(item_id=f"favours_01_{k}", responses=(True, True, False, False)))
    for k in range(4):
        responses.append(ItemResponses(item_id=f"favours_23_{k}", responses=(False, False, True, True)))
    return responses


def test_consistent_guttman_matrix_is_usable() -> None:
    matrix = _guttman_matrix(n_systems=20, n_items=15)
    _item_stats, panel = analyse(matrix, reliability_floor=0.5, n_splits=200, seed=0)
    assert panel.reliability_verdict == "usable"
    assert panel.split_half_reliability >= 0.5


def test_contradictory_matrix_is_too_few_systems() -> None:
    matrix = _contradictory_matrix()
    _item_stats, panel = analyse(matrix, reliability_floor=0.5, n_splits=200, seed=0)
    assert panel.reliability_verdict == "too_few_systems"
    # 8 items is below _MIN_LIVE_ITEMS_PER_SPLIT (10): every split is
    # excluded from the mean, so this is exactly 0.0 / 0, not merely low.
    assert panel.split_half_reliability == 0.0
    assert panel.n_items_used_for_reliability == 0


def test_reliability_floor_is_configurable() -> None:
    """The same matrix flips verdict when the floor moves past its
    measured reliability -- the verdict is a comparison against a
    caller-supplied floor, not a hardcoded classification."""
    matrix = _guttman_matrix(n_systems=20, n_items=15)
    _item_stats, panel = analyse(matrix, reliability_floor=0.5, n_splits=200, seed=0)
    assert panel.reliability_verdict == "usable"
    _item_stats2, panel2 = analyse(
        matrix, reliability_floor=panel.split_half_reliability + 0.01, n_splits=200, seed=0
    )
    assert panel2.reliability_verdict == "too_few_systems"


def test_classification_stands_regardless_of_reliability_verdict() -> None:
    """Zero-variance classification (dead_all_pass/dead_all_fail) does not
    depend on -- and is not suppressed by -- a low reliability verdict."""
    matrix = _contradictory_matrix() + [
        ItemResponses(item_id="dead_pass", responses=(True, True, True, True)),
        ItemResponses(item_id="dead_fail", responses=(False, False, False, False)),
    ]
    item_stats, panel = analyse(matrix, reliability_floor=0.5, n_splits=200, seed=0)
    assert panel.reliability_verdict == "too_few_systems"
    by_id = {s.item_id: s for s in item_stats}
    assert by_id["dead_pass"].classification == "dead_all_pass"
    assert by_id["dead_fail"].classification == "dead_all_fail"


def test_padding_with_dead_items_does_not_inflate_reliability() -> None:
    """The regression test for the class of bug this module's docstring
    describes -- mirrors the Cronbach's-alpha-padding experiment in
    `handbook/eval-of-evals.md` (87 live items alone: alpha=0.956; 87 live
    + 300 fake all-correct items: alpha=0.948 -- barely moved, because
    alpha cannot tell the difference). `split_half_reliability` must not
    merely be *insensitive* to padding the way alpha was; it must be
    **identical**, because every padding item here is provably invariant:
    an all-correct item contributes a constant +1 to every system's total
    in every split (Pearson correlation is unchanged by adding a constant
    to one side), and it is excluded by the zero-variance filter before
    correlating in any case -- so live items' point-biserial values, the
    live/dead partition of every split, and therefore the reliability
    figure itself, must come out bit-for-bit the same whether or not the
    padding is present.
    """
    live_matrix = _guttman_matrix(n_systems=20, n_items=15)
    padded_matrix = live_matrix + [
        ItemResponses(item_id=f"dead{k}", responses=tuple(True for _ in range(20))) for k in range(300)
    ]

    _live_stats, live_panel = analyse(live_matrix, reliability_floor=0.5, n_splits=200, seed=0)
    _padded_stats, padded_panel = analyse(padded_matrix, reliability_floor=0.5, n_splits=200, seed=0)

    assert padded_panel.n_items == 315
    assert live_panel.n_items == 15
    assert padded_panel.split_half_reliability == pytest.approx(live_panel.split_half_reliability, abs=1e-9)
    assert padded_panel.n_items_used_for_reliability == live_panel.n_items_used_for_reliability
    assert padded_panel.reliability_verdict == live_panel.reliability_verdict
