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

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from fireassay.items.core import Classification, ItemMeta, ItemStats, wilson_ci
from fireassay.items.seed import SeededItem, SeedKind

ReviewSource = Literal["flagged", "unflagged", "seeded", "calibration"]


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
    """One reviewer verdict on one `ReviewItem`, keyed by `review_id`.

    `verdict` is the `handbook/eval-of-evals.md` §6 disposition vocabulary
    (the three of its five dispositions that make sense for a single blind
    label, rather than a full lifecycle decision): `keep` (the item is
    fine as written), `rewrite` (salvageable but currently wrong,
    ambiguous, or unanswerable as written), `purge` (discard). This is NOT
    a score and must never be mapped to one -- `rewrite` is exactly the
    middle a binary bad/ok verdict throws away, and it is what a later
    ambiguity or gold-answerability detector needs to be measured against.
    Every item in the deck gets the same three choices regardless of
    `source` -- flagged, unflagged, seeded, or calibration -- since a
    richer or narrower choice set for one source would let a reviewer
    tell the groups apart and break the blind."""

    model_config = ConfigDict(frozen=True)

    review_id: str
    verdict: Literal["keep", "rewrite", "purge"]


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


def _stats_slot(s: ItemStats, meta: Mapping[str, ItemMeta], source: ReviewSource) -> _Slot:
    m = meta.get(s.item_id)
    return _Slot(
        s.item_id, source, False, None, s.classification,
        m.question if m else None, m.reference_answer if m else None,
        m.evidence_quote if m else None,
    )


def _plan_batch(
    stats: Sequence[ItemStats],
    meta: Mapping[str, ItemMeta],
    *,
    n_flagged: int,
    n_unflagged: int,
    n_calibration: int = 0,
    seeded: Sequence[SeededItem],
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> list[_Slot]:
    """The single deterministic selection/shuffle plan both
    `build_review_batch` and `build_review_key` derive their respective,
    consistent views from -- called independently by each (rather than
    threading shared state between them) so the pair stays in sync purely
    because both are pure functions of the same `(stats, meta, seed, ...)`
    inputs, the same recompute-don't-share-mutable-state discipline
    `curate.honeypots` uses for its own corrupted renderings.

    **Invariant: an `item_id` appears at most once per deck.** A reviewer
    who sees the same question twice -- once intact, once corrupted --
    has been handed the answer key (see this module's history: measured
    at 70% of decks colliding, on the real 2,364-item pool, before this
    invariant existed). Claim order, in priority, and why:

    1. `flagged` -- sampled from `mislabel_suspect`, claims its ids
       first. The caller asks for a specific count and the precision
       denominator depends on getting it; nothing may displace a flagged
       item.
    2. `seeded` -- any `SeededItem` whose `item_id` is already claimed is
       dropped, as is any second `SeededItem` for an id another seed
       already claimed (the first in `seeded`'s own given order wins).
       Seeded items are scarce and purposeful (a human deliberately
       picked which corruptions to spend on), so they outrank the merely
       sampled groups that follow.
    3. `unflagged` -- sampled from every non-`mislabel_suspect` item,
       excluding anything already claimed.
    4. `calibration` -- sampled from the entire pool, every
       classification, excluding anything already claimed.

    Dropping a seed (or sampling fewer flagged/unflagged/calibration
    items than requested because the pool ran out) is never padded and
    never an error -- the actual per-source counts are always recoverable
    by counting the returned slots.

    `exclude` is a second, composable exclusion mechanism -- for a second
    (or later) review pass drawing only from items nobody has labelled
    yet -- and is applied to every pool, in the same order of importance
    as the claim order above: `flagged`, then `seeded`, then `unflagged`,
    then `calibration`. The `seeded` case is the one that matters most
    and is easiest to forget: unlike the `stats`-derived pools, `seeded`
    is passed in **pre-built** -- each `SeededItem` already carries its
    own (possibly corrupted) question/reference/evidence text. Filtering
    only `stats` would leave an excluded id free to reappear via
    `seeded`, showing a reviewer who already read that item in an earlier
    pass the exact same question again, this time deliberately broken --
    the same leak the at-most-once invariant above exists to close, one
    pass later. `exclude` composes with that invariant rather than
    replacing it: an id may be excluded *and* independently claimed: both
    checks apply, and dropping a seed for either reason follows the same
    never-pad, never-error rule as the rest of this function.
    """
    rng = random.Random(seed)
    flagged_pool = [
        s for s in stats if s.classification == "mislabel_suspect" and s.item_id not in exclude
    ]
    picked_flagged = rng.sample(flagged_pool, k=min(n_flagged, len(flagged_pool)))

    slots: list[_Slot] = [_stats_slot(s, meta, "flagged") for s in picked_flagged]
    claimed: set[str] = {s.item_id for s in picked_flagged}

    seeded_claimed: set[str] = set()
    for si in seeded:
        if si.item_id in exclude or si.item_id in claimed or si.item_id in seeded_claimed:
            continue
        seeded_claimed.add(si.item_id)
        slots.append(
            _Slot(
                si.item_id, "seeded", True, si.kind, None,
                si.question, si.reference_answer, si.evidence_quote,
            )
        )
    claimed |= seeded_claimed

    unflagged_pool = [
        s
        for s in stats
        if s.classification != "mislabel_suspect"
        and s.item_id not in claimed
        and s.item_id not in exclude
    ]
    picked_unflagged = rng.sample(unflagged_pool, k=min(n_unflagged, len(unflagged_pool)))
    slots.extend(_stats_slot(s, meta, "unflagged") for s in picked_unflagged)
    claimed |= {s.item_id for s in picked_unflagged}

    calibration_pool = [s for s in stats if s.item_id not in claimed and s.item_id not in exclude]
    picked_calibration = rng.sample(calibration_pool, k=min(n_calibration, len(calibration_pool)))
    slots.extend(_stats_slot(s, meta, "calibration") for s in picked_calibration)

    rng.shuffle(slots)
    return slots


def build_review_batch(
    stats: Sequence[ItemStats],
    meta: Mapping[str, ItemMeta],
    *,
    n_flagged: int,
    n_unflagged: int,
    n_calibration: int = 0,
    seeded: Sequence[SeededItem] = (),
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> list[ReviewItem]:
    """Mix up to `n_flagged` `mislabel_suspect` items, up to `n_unflagged`
    items sampled from every other classification, up to `n_calibration`
    items sampled uniformly from the entire pool (every classification,
    never filtered or stratified -- see `_plan_batch`'s docstring), and
    whatever of `seeded` survives the dedupe rule into one shuffled batch,
    deterministic given `seed`. If fewer than `n_flagged`/`n_unflagged`/
    `n_calibration` eligible items exist, the actual count is smaller --
    never padded or erroring. **No `item_id` appears twice** -- see
    `_plan_batch`'s docstring for the claim order that guarantees it.
    `exclude`, if given, removes those `item_id`s from every pool
    (including `seeded`, whose members are pre-built and would otherwise
    be able to reappear across passes -- see `_plan_batch`'s docstring)
    and composes with, rather than replaces, that same at-most-once
    invariant."""
    plan = _plan_batch(
        stats, meta, n_flagged=n_flagged, n_unflagged=n_unflagged, n_calibration=n_calibration,
        seeded=seeded, seed=seed, exclude=exclude,
    )
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
    n_calibration: int = 0,
    seeded: Sequence[SeededItem] = (),
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> list[ReviewKeyEntry]:
    """The ground-truth mapping for the batch `build_review_batch` builds
    from the identical arguments -- same seed, same `_plan_batch` call, so
    `build_review_key(...)[i].review_id == build_review_batch(...)[i].review_id`
    for every `i`, without the two functions sharing any state. A
    `calibration` entry's `classification` is its real, measured one --
    that fact lives only here, never in the `ReviewItem` the reviewer
    sees. `exclude` behaves exactly as in `build_review_batch`/
    `_plan_batch`."""
    plan = _plan_batch(
        stats, meta, n_flagged=n_flagged, n_unflagged=n_unflagged, n_calibration=n_calibration,
        seeded=seeded, seed=seed, exclude=exclude,
    )
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


def score_review(
    labels: Sequence[ReviewLabel],
    key: Sequence[ReviewKeyEntry],
    *,
    bad_verdicts: frozenset[str] = frozenset({"purge"}),
) -> DetectorScore:
    """Precision from `source == "flagged"` items (of what the detector
    flagged as `mislabel_suspect`, how many did a blind reviewer
    independently call bad?); recall from `is_seeded` items (of the
    known-bad seeded items, how many did the reviewer catch? -- an upper
    bound, see the module docstring). A review item with no matching
    label is simply excluded from both its subset's numerator and
    denominator, not counted as either verdict.

    An item counts toward its subset's numerator iff `label.verdict in
    bad_verdicts`. `bad_verdicts` defaults to the strict reading
    (`{"purge"}` only); pass `{"purge", "rewrite"}` for the lenient one.
    Call this twice for both -- `DetectorScore` deliberately carries no
    second set of fields for a lenient count, and no default here is
    "the" threshold: which of the two matters is the reader's call, and
    picking one for them would be exactly the unmeasured-threshold
    failure this tool exists to refuse."""
    labels_by_id = {label.review_id: label for label in labels}

    flagged_bad = flagged_n = 0
    seeded_caught = seeded_n = 0
    for entry in key:
        label = labels_by_id.get(entry.review_id)
        if label is None:
            continue
        if entry.source == "flagged":
            flagged_n += 1
            if label.verdict in bad_verdicts:
                flagged_bad += 1
        if entry.is_seeded:
            seeded_n += 1
            if label.verdict in bad_verdicts:
                seeded_caught += 1

    return DetectorScore(
        precision=(flagged_bad / flagged_n) if flagged_n > 0 else None,
        precision_ci=wilson_ci(flagged_bad, flagged_n) if flagged_n > 0 else None,
        precision_n=flagged_n,
        recall=(seeded_caught / seeded_n) if seeded_n > 0 else None,
        recall_ci=wilson_ci(seeded_caught, seeded_n) if seeded_n > 0 else None,
        recall_n=seeded_n,
    )
