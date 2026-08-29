"""`test_items_core.py` (M-ITEMS-SPEC.md §6): hand-computed p, D,
point-biserial on a fixed matrix; classification boundaries; ragged matrix
raises; <4 systems raises.

**The fixed matrix**, 7 items x 4 systems (s0..s3):

    ramp1      = [T, F, F, F]
    ramp2      = [T, T, F, F]
    ramp3      = [T, T, T, F]
    dead_pass  = [T, T, T, T]
    dead_fail  = [F, F, F, F]
    live_pos   = [T, T, F, F]
    mislabel   = [F, F, T, T]

Per-system total score (sum of all 7 items' True count) is strictly
decreasing: s0=5, s1=4, s2=3, s3=2 -- so with `n_systems=4`,
`_extreme_group_size(4) == 1`, top group = {s0} unambiguously, bottom
group = {s3} unambiguously (no ties to break). `p`/`D`/point-biserial
below are hand-computed against these totals via the standard formula
`r_pbis = ((M1 - M0) / s_total) * sqrt(p * q)` (M1/M0 = mean total score
of those who got the item right/wrong, s_total = population std of
totals) and cross-checked against a direct Pearson-correlation
computation; both agree.
"""

from __future__ import annotations

import math

import pytest

from fireassay.items.core import ItemResponses, analyse

_SQRT_0_6 = math.sqrt(0.6)  # ramp1, ramp3: (M1-M0)/s_total * sqrt(pq)
_TWO_OVER_SQRT_5 = 2.0 / math.sqrt(5.0)  # ramp2, live_pos, mislabel (abs value)


def _fixed_matrix() -> list[ItemResponses]:
    return [
        ItemResponses(item_id="ramp1", responses=(True, False, False, False)),
        ItemResponses(item_id="ramp2", responses=(True, True, False, False)),
        ItemResponses(item_id="ramp3", responses=(True, True, True, False)),
        ItemResponses(item_id="dead_pass", responses=(True, True, True, True)),
        ItemResponses(item_id="dead_fail", responses=(False, False, False, False)),
        ItemResponses(item_id="live_pos", responses=(True, True, False, False)),
        ItemResponses(item_id="mislabel", responses=(False, False, True, True)),
    ]


def _stats_by_id(matrix: list[ItemResponses]) -> dict[str, object]:
    item_stats, _panel = analyse(matrix, reliability_floor=0.0)  # floor=0 -- not what this test checks
    return {s.item_id: s for s in item_stats}


def test_p_hand_computed() -> None:
    stats = _stats_by_id(_fixed_matrix())
    assert stats["ramp1"].p == pytest.approx(0.25)
    assert stats["ramp2"].p == pytest.approx(0.5)
    assert stats["ramp3"].p == pytest.approx(0.75)
    assert stats["dead_pass"].p == pytest.approx(1.0)
    assert stats["dead_fail"].p == pytest.approx(0.0)
    assert stats["live_pos"].p == pytest.approx(0.5)
    assert stats["mislabel"].p == pytest.approx(0.5)


def test_discrimination_d_hand_computed() -> None:
    stats = _stats_by_id(_fixed_matrix())
    # top group = {s0}, bottom group = {s3} (unambiguous, see module docstring)
    assert stats["ramp1"].discrimination_d == pytest.approx(1.0)  # s0=T, s3=F
    assert stats["ramp2"].discrimination_d == pytest.approx(1.0)
    assert stats["ramp3"].discrimination_d == pytest.approx(1.0)
    assert stats["dead_pass"].discrimination_d == pytest.approx(0.0)  # s0=T, s3=T
    assert stats["dead_fail"].discrimination_d == pytest.approx(0.0)  # s0=F, s3=F
    assert stats["live_pos"].discrimination_d == pytest.approx(1.0)
    assert stats["mislabel"].discrimination_d == pytest.approx(-1.0)  # s0=F, s3=T -- negative


def test_point_biserial_hand_computed() -> None:
    stats = _stats_by_id(_fixed_matrix())
    assert stats["ramp1"].point_biserial == pytest.approx(_SQRT_0_6, rel=1e-9)
    assert stats["ramp2"].point_biserial == pytest.approx(_TWO_OVER_SQRT_5, rel=1e-9)
    assert stats["ramp3"].point_biserial == pytest.approx(_SQRT_0_6, rel=1e-9)
    assert stats["live_pos"].point_biserial == pytest.approx(_TWO_OVER_SQRT_5, rel=1e-9)
    assert stats["mislabel"].point_biserial == pytest.approx(-_TWO_OVER_SQRT_5, rel=1e-9)
    # zero-variance items: point_biserial is 0.0 by definition, never nan
    assert stats["dead_pass"].point_biserial == pytest.approx(0.0)
    assert stats["dead_fail"].point_biserial == pytest.approx(0.0)


def test_classification_boundaries() -> None:
    stats = _stats_by_id(_fixed_matrix())
    assert stats["dead_pass"].classification == "dead_all_pass"
    assert stats["dead_fail"].classification == "dead_all_fail"
    assert stats["mislabel"].classification == "mislabel_suspect"
    for item_id in ("ramp1", "ramp2", "ramp3", "live_pos"):
        assert stats[item_id].classification == "live"


def test_panel_stats_class_counts_and_shape() -> None:
    item_stats, panel = analyse(_fixed_matrix(), reliability_floor=0.0)
    assert panel.n_items == 7
    assert panel.n_systems == 4
    assert panel.class_counts == {
        "live": 4,
        "dead_all_pass": 1,
        "dead_all_fail": 1,
        "mislabel_suspect": 1,
    }
    assert len(item_stats) == 7


def test_claimed_vs_measured_difficulty_r_is_none_without_meta() -> None:
    _item_stats, panel = analyse(_fixed_matrix(), meta=None, reliability_floor=0.0)
    assert panel.claimed_vs_measured_difficulty_r is None


def test_ragged_matrix_raises() -> None:
    matrix = [
        ItemResponses(item_id="a", responses=(True, False, True, False)),
        ItemResponses(item_id="b", responses=(True, False, True)),  # one short
    ]
    with pytest.raises(ValueError, match="ragged"):
        analyse(matrix)


def test_fewer_than_four_systems_raises() -> None:
    matrix = [
        ItemResponses(item_id="a", responses=(True, False, True)),
        ItemResponses(item_id="b", responses=(True, True, False)),
    ]
    with pytest.raises(ValueError, match="at least 4 systems"):
        analyse(matrix)


def test_empty_responses_raises() -> None:
    with pytest.raises(ValueError, match="no items"):
        analyse([])
