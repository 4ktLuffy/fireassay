"""Blind review: the step that turns a plausible detector
(`ItemStats.classification == "mislabel_suspect"`) into a measured one.

**A `ReviewItem` leaks nothing.** It exposes only `question`,
`reference_answer`, and `evidence_quote` -- never `classification`, never
any statistic (`p`, `discrimination_d`, `point_biserial`), never whether
the item was seeded. A reviewer who can see the verdict cannot measure it:
knowing an item was flagged (or seeded as known-bad) would change how
carefully -- or how skeptically -- a reviewer looks at it, contaminating
exactly the number this module exists to produce. `ReviewItem.review_id`
is an opaque per-batch position label, deliberately **not** the real
`item_id` -- the same precaution, one level further: even the id itself
must not let a reviewer who happens to also have access to `ItemStats`
(e.g. a team member who ran `analyse`) cross-reference a review item back
to its verdict mid-review.

The mapping from `review_id` back to ground truth (`item_id`, `source`,
whether it was seeded, which seed kind, and its measured `classification`)
lives only in `ReviewKeyEntry` / `build_review_key` -- written to a
**separate** file the reviewer never opens (`fireassay items review build
--key key.json`, distinct from `--out batch.jsonl`).

**Seeded recall is an upper bound**, never the number itself -- see
`items.seed`'s module docstring. `score_review` computes it anyway (it is
the only recall estimate available with no human time spent double-judging
natural items), but every place it is displayed must say so.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from fireassay.items.core import Classification, ItemMeta, ItemStats
from fireassay.items.seed import SeededItem, SeedKind

ReviewSource = Literal["flagged", "unflagged", "seeded"]


class ReviewItem(BaseModel):
    """What a reviewer sees. Deliberately excludes everything named in
    this module's docstring's leak rule."""

    model_config = ConfigDict(frozen=True)

    review_id: str
    question: str | None = None
    reference_answer: str | None = None
    evidence_quote: str | None = None


class ReviewKeyEntry(BaseModel):
    """One `ReviewItem`'s ground truth -- lives only in the key file."""

    model_config = ConfigDict(frozen=True)

    review_id: str
    item_id: str
    source: ReviewSource
    is_seeded: bool
    seed_kind: SeedKind | None = None
    classification: Classification | None = None


class ReviewLabel(BaseModel):
    """One reviewer verdict on one `ReviewItem`, keyed by `review_id`."""

    model_config = ConfigDict(frozen=True)

    review_id: str
    verdict: Literal["bad", "ok"]


class DetectorScore(BaseModel):
    """Precision (from the flagged items) and recall (from the seeded
    items), each with a confidence interval -- a bare point estimate is
    misleading at the sample sizes a blind review batch typically has.
    `precision`/`recall` and their CIs are `None` (never a fabricated
    `0.0`) when nothing in that subset was labelled."""

    model_config = ConfigDict(frozen=True)

    precision: float | None
    precision_ci: tuple[float, float] | None
    precision_n: int
    recall: float | None
    recall_ci: tuple[float, float] | None
    recall_n: int


@dataclass(frozen=True)
class _Slot:
    item_id: str
    source: ReviewSource
    is_seeded: bool
    seed_kind: SeedKind | None
    classification: Classification | None
    question: str | None
    reference_answer: str | None
    evidence_quote: str | None


def _plan_batch(
    stats: Sequence[ItemStats],
    meta: Mapping[str, ItemMeta],
    *,
    n_flagged: int,
    n_unflagged: int,
    seeded: Sequence[SeededItem],
    seed: int,
) -> list[_Slot]:
    """The single deterministic selection/shuffle plan both
    `build_review_batch` and `build_review_key` derive their respective,
    consistent views from -- called independently by each (rather than
    threading shared state between them) so the pair stays in sync purely
    because both are pure functions of the same `(stats, meta, seed, ...)`
    inputs, the same recompute-don't-share-mutable-state discipline
    `curate.honeypots` uses for its own corrupted renderings."""
    rng = random.Random(seed)
    flagged_pool = [s for s in stats if s.classification == "mislabel_suspect"]
    unflagged_pool = [s for s in stats if s.classification != "mislabel_suspect"]
    picked_flagged = rng.sample(flagged_pool, k=min(n_flagged, len(flagged_pool)))
    picked_unflagged = rng.sample(unflagged_pool, k=min(n_unflagged, len(unflagged_pool)))

    slots: list[_Slot] = []
    for s in picked_flagged:
        m = meta.get(s.item_id)
        slots.append(
            _Slot(
                s.item_id, "flagged", False, None, s.classification,
                m.question if m else None, m.reference_answer if m else None,
                m.evidence_quote if m else None,
            )
        )
    for s in picked_unflagged:
        m = meta.get(s.item_id)
        slots.append(
            _Slot(
                s.item_id, "unflagged", False, None, s.classification,
                m.question if m else None, m.reference_answer if m else None,
                m.evidence_quote if m else None,
            )
        )
    for si in seeded:
        slots.append(
            _Slot(
                si.item_id, "seeded", True, si.kind, None,
                si.question, si.reference_answer, si.evidence_quote,
            )
        )

    rng.shuffle(slots)
    return slots


def build_review_batch(
    stats: Sequence[ItemStats],
    meta: Mapping[str, ItemMeta],
    *,
    n_flagged: int,
    n_unflagged: int,
    seeded: Sequence[SeededItem] = (),
    seed: int,
) -> list[ReviewItem]:
    """Mix up to `n_flagged` `mislabel_suspect` items, up to `n_unflagged`
    items sampled from every other classification, and all of `seeded`
    into one shuffled batch, deterministic given `seed`. If fewer than
    `n_flagged`/`n_unflagged` eligible items exist, the actual count is
    smaller -- never padded or erroring."""
    plan = _plan_batch(stats, meta, n_flagged=n_flagged, n_unflagged=n_unflagged, seeded=seeded, seed=seed)
    return [
        ReviewItem(
            review_id=f"r{i:04d}",
            question=slot.question,
            reference_answer=slot.reference_answer,
            evidence_quote=slot.evidence_quote,
        )
        for i, slot in enumerate(plan)
    ]


def build_review_key(
    stats: Sequence[ItemStats],
    meta: Mapping[str, ItemMeta],
    *,
    n_flagged: int,
    n_unflagged: int,
    seeded: Sequence[SeededItem] = (),
    seed: int,
) -> list[ReviewKeyEntry]:
    """The ground-truth mapping for the batch `build_review_batch` builds
    from the identical arguments -- same seed, same `_plan_batch` call, so
    `build_review_key(...)[i].review_id == build_review_batch(...)[i].review_id`
    for every `i`, without the two functions sharing any state."""
    plan = _plan_batch(stats, meta, n_flagged=n_flagged, n_unflagged=n_unflagged, seeded=seeded, seed=seed)
    return [
        ReviewKeyEntry(
            review_id=f"r{i:04d}",
            item_id=slot.item_id,
            source=slot.source,
            is_seeded=slot.is_seeded,
            seed_kind=slot.seed_kind,
            classification=slot.classification,
        )
        for i, slot in enumerate(plan)
    ]


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
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


def score_review(labels: Sequence[ReviewLabel], key: Sequence[ReviewKeyEntry]) -> DetectorScore:
    """Precision from `source == "flagged"` items (of what the detector
    flagged as `mislabel_suspect`, how many did a blind reviewer
    independently call bad?); recall from `is_seeded` items (of the
    known-bad seeded items, how many did the reviewer catch? -- an upper
    bound, see the module docstring). A review item with no matching
    label is simply excluded from both its subset's numerator and
    denominator, not counted as either verdict.
    """
    labels_by_id = {label.review_id: label for label in labels}

    flagged_bad = flagged_n = 0
    seeded_caught = seeded_n = 0
    for entry in key:
        label = labels_by_id.get(entry.review_id)
        if label is None:
            continue
        if entry.source == "flagged":
            flagged_n += 1
            if label.verdict == "bad":
                flagged_bad += 1
        if entry.is_seeded:
            seeded_n += 1
            if label.verdict == "bad":
                seeded_caught += 1

    return DetectorScore(
        precision=(flagged_bad / flagged_n) if flagged_n > 0 else None,
        precision_ci=_wilson_ci(flagged_bad, flagged_n) if flagged_n > 0 else None,
        precision_n=flagged_n,
        recall=(seeded_caught / seeded_n) if seeded_n > 0 else None,
        recall_ci=_wilson_ci(seeded_caught, seeded_n) if seeded_n > 0 else None,
        recall_n=seeded_n,
    )
