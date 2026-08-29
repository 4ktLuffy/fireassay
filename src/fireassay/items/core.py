"""Pure item analysis over a response matrix. **NO I/O, NO STORAGE IMPORT.**

`items.core` MUST NOT import `fireassay.store`, or anything that
transitively imports it (`test_items_no_store_import.py` walks this
module's import graph and asserts `fireassay.store` never appears in it).
The input here is a response matrix -- items x systems x binary
correctness -- plus optional metadata; that is the entire contract. This
is what lets the tool be usable standalone by someone who has never heard
of fireassay: `adapters/tabular.py` builds the same matrix from a plain
CSV/JSONL a stranger can produce with no fireassay code at all, and
`adapters/store.py` (the one module allowed to import the store) builds it
from fireassay's own DB. Both funnel into this one module.

**What this module deliberately does NOT do**, per the measurements in
`~/ObitosBrain/handbook/eval-of-evals.md` -- read that file before changing
any of the rules below, it records what was measured and falsified so this
module does not re-derive it:

- **No Cronbach's alpha.** Measured at 0.948 for a set that was 77% fake,
  all-correct padding -- it is blind to dead weight, a category error for
  this purpose.
- **No composite quality score.** Three of five surveyed sources proposed
  one, each with different invented weights, none justified. `analyse`
  emits a scorecard (`list[ItemStats]`) and panel-level diagnostics
  (`PanelStats`), never a single number.
- **Per-item discrimination/point-biserial are reported alongside their own
  reliability, not asserted as trustworthy on their own.** Split-half
  reliability over random system-halves was measured at mean r = 0.294
  across 18 systems (183/200 splits below 0.5) -- discrimination tells you
  roughly *how many* items discriminate, not reliably *which* ones, until
  there are enough systems. `PanelStats.split_half_reliability` is
  reported first, ahead of any per-item number, everywhere this module's
  output is displayed (see `report.text.render_items_analysis`) --
  `PanelStats.reliability_verdict == "too_few_systems"` does not suppress
  `ItemStats.discrimination_d`/`point_biserial` (zero-variance
  classification is robust regardless of system count), but a caller must
  see the verdict before trusting a per-item rank. **Zero-variance items
  are excluded from this reliability computation itself** (see
  `_split_half_reliability`'s docstring) -- two dead items agreeing on a
  point-biserial of `0.0` is not evidence of a reliable measurement, the
  same flaw as Cronbach's alpha above; measured on a real 2,364-item panel,
  including them inflated split-half reliability from `0.287` (correct,
  `"too_few_systems"`) to `0.585` (wrong, `"usable"`).
  `PanelStats.n_items_used_for_reliability` reports how many items the
  figure was actually measured over, so that distinction is never left for
  a reader to infer.
- **A claimed difficulty label is never trusted.** Measured correlation
  between a claimed difficulty and actual pass rate was +0.039 -- the
  wrong sign. `PanelStats.claimed_vs_measured_difficulty_r` reports the
  correlation as a diagnostic, never as a filter or a weight.

**`mislabel_suspect`** is the cheap, no-4PL-fit proxy for a 4PL model's
upper-asymptote parameter: a genuinely *hard* item is failed by weak
systems and passed by strong ones (positive discrimination); a
*mislabelled* item is failed by strong systems too, because they produce
the objectively correct answer, which does not match a wrong reference
label. Operationally: among non-degenerate items (`0 < p < 1`), an item is
`mislabel_suspect` when the top-scoring systems do *worse* on it than the
bottom-scoring systems (`discrimination_d < 0`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

#: Minimum number of systems `analyse` will accept at all -- below this,
#: even the coarse top/bottom-27% split needed for discrimination has no
#: meaningful "top" or "bottom".
_MIN_SYSTEMS = 4

#: Kelley's classical top/bottom split fraction for discrimination D.
_EXTREME_GROUP_FRACTION = 0.27

#: Ordinal encoding for the canonical difficulty vocabulary shared with
#: `fireassay.models.Difficulty`. `ItemMeta.claimed_difficulty` is a bare
#: `str | None` (not that Literal) because a standalone caller's tabular
#: metadata may use any vocabulary at all -- a claimed_difficulty outside
#: this set simply does not contribute to
#: `claimed_vs_measured_difficulty_r` (omitted, never coerced into an
#: assumed order the label never asserted).
_DIFFICULTY_ORDER: dict[str, float] = {"easy": 0.0, "medium": 1.0, "hard": 2.0}


class ItemResponses(BaseModel):
    """One item's correctness against every system, in a fixed, shared
    system order (`responses[i]` must refer to the same system for every
    `ItemResponses` passed to `analyse` together)."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    responses: tuple[bool, ...]


class ItemMeta(BaseModel):
    """Everything about an item beyond its response vector -- entirely
    optional beyond `item_id` so a caller with only a bare response matrix
    (no text at all) can still run `analyse`."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    question: str | None = None
    reference_answer: str | None = None
    evidence_quote: str | None = None
    source_doc_id: str | None = None
    claimed_difficulty: str | None = None
    claimed_qtype: str | None = None


Classification = Literal["live", "dead_all_pass", "dead_all_fail", "mislabel_suspect"]

_ALL_CLASSIFICATIONS: tuple[Classification, ...] = (
    "live",
    "dead_all_pass",
    "dead_all_fail",
    "mislabel_suspect",
)


class ItemStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    p: float
    discrimination_d: float
    point_biserial: float
    classification: Classification


class PanelStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_items: int
    n_systems: int
    split_half_reliability: float
    #: How many items `split_half_reliability` was actually measured over
    #: (mean, rounded, across the splits averaged into it) -- **not**
    #: `n_items`. Zero-variance items are excluded before correlating (see
    #: `_split_half_reliability`'s docstring for why); a reliability figure
    #: computed over 987 of 2,364 items is a different claim from one over
    #: all 2,364, and a reader must not have to infer which one they are
    #: looking at.
    n_items_used_for_reliability: int
    reliability_verdict: Literal["usable", "too_few_systems"]
    class_counts: dict[str, int]
    claimed_vs_measured_difficulty_r: float | None


def _totals(matrix: Sequence[ItemResponses], system_indices: Sequence[int]) -> np.ndarray:
    """Per-system total correct count, restricted to `system_indices`,
    summed over every item in `matrix`."""
    totals = np.zeros(len(system_indices), dtype=float)
    for item in matrix:
        totals += np.array([1.0 if item.responses[i] else 0.0 for i in system_indices], dtype=float)
    return totals


def _item_vector(item: ItemResponses, system_indices: Sequence[int]) -> np.ndarray:
    return np.array([1.0 if item.responses[i] else 0.0 for i in system_indices], dtype=float)


def _point_biserial(item_vec: np.ndarray, totals: np.ndarray) -> float:
    """Pearson correlation between one item's binary response vector and
    per-system total score, restricted to whatever system subset both
    arrays were already built over. `0.0` when either side has no
    variance -- an item with no variance (`p` in {0, 1}) or a system
    subset whose totals never differ correlates with nothing, and `0.0` is
    the honest answer, not `numpy.corrcoef`'s `nan`."""
    if np.std(item_vec) == 0.0 or np.std(totals) == 0.0:
        return 0.0
    r = float(np.corrcoef(item_vec, totals)[0, 1])
    return 0.0 if np.isnan(r) else r


def _extreme_group_size(n_systems: int) -> int:
    n = max(1, round(n_systems * _EXTREME_GROUP_FRACTION))
    return min(n, n_systems // 2)


def _split_seed(seed: int, i: int) -> int:
    """A cheap, deterministic per-split integer seed derived from `seed`
    and the split index `i` -- plain arithmetic, not a cryptographic hash:
    `analyse`'s only requirement is that the same `(seed, i)` always
    produces the same split within one process, not cross-run content
    addressing (unlike `fireassay.hashing`, which this module must not
    import -- see the module docstring)."""
    return (seed * 1_000_003 + i * 7_919 + 104_729) & 0xFFFFFFFF


#: Below this many surviving (non-zero-variance) items, a split's
#: correlation is excluded from the mean -- the same "not a reliability
#: estimate" reasoning as the zero-std guard just below it, phrased as a
#: floor on sample size rather than on variance. See
#: `_split_half_reliability`'s docstring.
_MIN_LIVE_ITEMS_PER_SPLIT = 10


def _split_half_reliability(
    matrix: Sequence[ItemResponses], n_systems: int, n_splits: int, seed: int
) -> tuple[float, int]:
    """Mean Pearson correlation, over `n_splits` random disjoint halves of
    the systems, between the two halves' per-item point-biserial vectors --
    restricted, in every split, to items with measured variance in **at
    least one** half.

    **Why zero-variance items are dropped before correlating (read before
    "simplifying" this back out):** a zero-variance item -- `p` in {0, 1}
    globally, or merely zero-variance *within* one half by chance -- scores
    exactly `0.0` point-biserial in both halves, by `_point_biserial`'s own
    short-circuit. Two dead items "agreeing" that neither carries any
    signal is not evidence that the *live* items' discrimination estimates
    are reliable; it is a perfect, contentless 0-to-0 match that inflates
    the correlation for free, in direct proportion to how much of the
    matrix is dead weight. This is the exact same failure this module's
    docstring documents for Cronbach's alpha (0.948 for a set that was 77%
    fake, all-correct padding): a reliability statistic that improves when
    you add items carrying no information is measuring the padding, not
    the measurement -- and left uncorrected, this tool reproduced that
    flaw in the one number it leads every report with.

    Measured on the real 2,364-item panel (1,377 items, 58.2%,
    zero-variance): including dead items gave split-half reliability
    `0.585` (verdict `"usable"`); the 987 live items alone gave `0.287`
    (verdict `"too_few_systems"`) -- an inflation of `+0.298`, enough to
    flip the verdict this whole report leads with. Dropping zero-variance
    items here is that correction, not an arbitrary convenience.

    A split where fewer than `_MIN_LIVE_ITEMS_PER_SPLIT` items survive this
    filter is excluded from the mean entirely, on the same "not a
    reliability estimate" principle as the zero-std guard below it -- a
    correlation over a handful of points is not a reliability estimate,
    whether the reason it is small is degeneracy or filtering. If every
    split is excluded (degenerate or too few surviving items throughout),
    `(0.0, 0)` is returned: the honest reading that this panel cannot
    demonstrate any split-half consistency at all.

    Returns `(reliability, n_items_used)`: `n_items_used` is the mean,
    rounded, count of items that survived the filter across the splits
    actually averaged into `reliability` -- see
    `PanelStats.n_items_used_for_reliability`."""
    indices = list(range(n_systems))
    half = n_systems // 2
    correlations: list[float] = []
    live_counts: list[int] = []
    for i in range(n_splits):
        rng = random.Random(_split_seed(seed, i))
        shuffled = indices[:]
        rng.shuffle(shuffled)
        half_a, half_b = shuffled[:half], shuffled[half : 2 * half]

        totals_a, totals_b = _totals(matrix, half_a), _totals(matrix, half_b)
        pb_a = np.array([_point_biserial(_item_vector(item, half_a), totals_a) for item in matrix])
        pb_b = np.array([_point_biserial(_item_vector(item, half_b), totals_b) for item in matrix])

        live = (pb_a != 0.0) | (pb_b != 0.0)
        pb_a, pb_b = pb_a[live], pb_b[live]

        if len(pb_a) < _MIN_LIVE_ITEMS_PER_SPLIT:
            continue
        if np.std(pb_a) == 0.0 or np.std(pb_b) == 0.0:
            continue
        r = float(np.corrcoef(pb_a, pb_b)[0, 1])
        if not np.isnan(r):
            correlations.append(r)
            live_counts.append(int(len(pb_a)))

    reliability = float(np.mean(correlations)) if correlations else 0.0
    n_items_used = int(round(float(np.mean(live_counts)))) if live_counts else 0
    return reliability, n_items_used


def _claimed_vs_measured_r(
    item_stats: Sequence[ItemStats], meta: Mapping[str, ItemMeta] | None
) -> float | None:
    """Pearson r between claimed difficulty (canonical easy/medium/hard
    only, ordinal-encoded) and measured `p`, over items that supply a
    canonical claimed label. `None` -- never a fabricated `0.0` -- when no
    claimed labels were supplied at all, fewer than two items have a
    canonical label, or either side has no variance to correlate (mirrors
    `curate.coverage.difficulty_feature_correlation`'s omit-don't-zero
    rule)."""
    if not meta:
        return None
    pairs: list[tuple[float, float]] = []
    for stats in item_stats:
        m = meta.get(stats.item_id)
        if m is None:
            continue
        claimed = m.claimed_difficulty
        if claimed is None or claimed not in _DIFFICULTY_ORDER:
            continue
        pairs.append((_DIFFICULTY_ORDER[claimed], stats.p))
    if len(pairs) < 2:
        return None
    # distinct name from the `claimed` label above: reusing it shadows a
    # `str | None` with an ndarray, which reads fine and does not type-check
    claimed_rank = np.array([p[0] for p in pairs], dtype=float)
    measured = np.array([p[1] for p in pairs], dtype=float)
    if np.std(claimed_rank) == 0.0 or np.std(measured) == 0.0:
        return None
    r = float(np.corrcoef(claimed_rank, measured)[0, 1])
    return None if np.isnan(r) else r


def analyse(
    responses: Sequence[ItemResponses],
    meta: Mapping[str, ItemMeta] | None = None,
    *,
    reliability_floor: float = 0.5,
    n_splits: int = 200,
    seed: int = 0,
) -> tuple[list[ItemStats], PanelStats]:
    """Compute per-item statistics and panel-level diagnostics over
    `responses`.

    Raises `ValueError` if `responses` is empty, if any item's response
    count differs from the first item's (a ragged matrix), or if fewer
    than 4 systems are present -- see `_MIN_SYSTEMS`.
    """
    if not responses:
        raise ValueError("analyse: no items supplied")
    n_systems = len(responses[0].responses)
    for item in responses:
        if len(item.responses) != n_systems:
            raise ValueError(
                f"analyse: ragged response matrix -- item {item.item_id!r} has "
                f"{len(item.responses)} response(s), expected {n_systems} (from item "
                f"{responses[0].item_id!r}); every item must report exactly one response "
                "per system, in the same system order"
            )
    if n_systems < _MIN_SYSTEMS:
        raise ValueError(
            f"analyse: at least {_MIN_SYSTEMS} systems are required for item analysis, got {n_systems}"
        )

    all_indices = list(range(n_systems))
    totals_full = _totals(responses, all_indices)
    # kind="stable" -- not just any deterministic order, but a *documented*
    # one: ties broken by ascending system index, so top/bottom-group
    # membership (and therefore discrimination_d/classification for a
    # tied item) does not depend on an incidental property of the sort
    # algorithm.
    order = np.argsort(-totals_full, kind="stable")  # descending by total score
    n_extreme = _extreme_group_size(n_systems)
    top_idx = set(order[:n_extreme].tolist())
    bottom_idx = set(order[-n_extreme:].tolist())

    item_stats: list[ItemStats] = []
    for item in responses:
        vec = _item_vector(item, all_indices)
        p = float(vec.mean())
        pb = _point_biserial(vec, totals_full)
        p_top = float(np.mean([vec[i] for i in top_idx]))
        p_bottom = float(np.mean([vec[i] for i in bottom_idx]))
        d = p_top - p_bottom

        classification: Classification
        if p == 1.0:
            classification = "dead_all_pass"
        elif p == 0.0:
            classification = "dead_all_fail"
        elif d < 0.0:
            classification = "mislabel_suspect"
        else:
            classification = "live"

        item_stats.append(
            ItemStats(
                item_id=item.item_id,
                p=p,
                discrimination_d=d,
                point_biserial=pb,
                classification=classification,
            )
        )

    reliability, n_items_used = _split_half_reliability(responses, n_systems, n_splits, seed)
    reliability_verdict: Literal["usable", "too_few_systems"] = (
        "usable" if reliability >= reliability_floor else "too_few_systems"
    )

    class_counts: dict[str, int] = {c: 0 for c in _ALL_CLASSIFICATIONS}
    for stats in item_stats:
        class_counts[stats.classification] += 1

    panel_stats = PanelStats(
        n_items=len(responses),
        n_systems=n_systems,
        split_half_reliability=reliability,
        n_items_used_for_reliability=n_items_used,
        reliability_verdict=reliability_verdict,
        class_counts=class_counts,
        claimed_vs_measured_difficulty_r=_claimed_vs_measured_r(item_stats, meta),
    )

    return item_stats, panel_stats


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion -- chosen over
    the normal (Wald) approximation because it stays inside `[0, 1]` and
    stays sane at the small `n` a review batch typically has, where Wald
    routinely produces a nonsensical interval (e.g. a negative lower bound
    when `successes` is 0 or `n`)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))
