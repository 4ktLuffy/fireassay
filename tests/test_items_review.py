"""`test_items_review.py` (M-ITEMS-SPEC.md §6): a `ReviewItem` leaks
neither classification, statistics, nor seeded status; batch order is
seed-deterministic."""

from __future__ import annotations

from fireassay.items.core import Classification, ItemMeta, ItemStats
from fireassay.items.review import (
    ReviewItem,
    ReviewKeyEntry,
    ReviewLabel,
    build_review_batch,
    build_review_key,
    score_review,
)
from fireassay.items.seed import SeededItem, seed_batch


def _stats(item_id: str, p: float, d: float, pb: float, classification: Classification) -> ItemStats:
    return ItemStats(
        item_id=item_id, p=p, discrimination_d=d, point_biserial=pb, classification=classification
    )


def _meta(item_id: str) -> ItemMeta:
    return ItemMeta(
        item_id=item_id,
        question=f"Q {item_id}?",
        reference_answer=f"A {item_id}",
        evidence_quote=f"E {item_id}",
    )


_STATS = [
    _stats("flag1", 0.5, -0.6, -0.5, "mislabel_suspect"),
    _stats("flag2", 0.4, -0.3, -0.4, "mislabel_suspect"),
    _stats("live1", 0.5, 0.6, 0.5, "live"),
    _stats("live2", 0.6, 0.3, 0.4, "live"),
    _stats("dead1", 1.0, 0.0, 0.0, "dead_all_pass"),
]

_META = {item_id: _meta(item_id) for item_id in ("flag1", "flag2", "live1", "live2", "dead1")}

_SEEDED = [
    SeededItem(
        item_id="live1",
        kind="swapped_reference",
        question="Q live1?",
        reference_answer="A live2",  # swapped
        evidence_quote="E live1",
        source_doc_id=None,
        detail="reference_answer swapped from item 'live2'",
    ),
]


def test_review_item_model_only_exposes_question_reference_evidence() -> None:
    """Structural guard: `ReviewItem`'s schema itself cannot carry a
    classification, a statistic, or a seeded flag -- there is no field to
    accidentally populate with one."""
    assert set(ReviewItem.model_fields) == {"review_id", "question", "reference_answer", "evidence_quote"}


def test_review_batch_leaks_nothing_at_the_instance_level() -> None:
    batch = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=_SEEDED, seed=42)
    for item in batch:
        dumped = item.model_dump()
        assert set(dumped) == {"review_id", "question", "reference_answer", "evidence_quote"}
        # the real item_id must not leak through review_id
        assert dumped["review_id"] not in {"flag1", "flag2", "live1", "live2", "dead1"}


def test_review_batch_composition() -> None:
    batch = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=_SEEDED, seed=42)
    key = build_review_key(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=_SEEDED, seed=42)
    # 2 flagged (all of the mislabel_suspect pool) + 2 unflagged + 1 seeded = 5 review items
    assert len(batch) == 5
    assert sum(1 for k in key if k.source == "flagged") == 2
    assert sum(1 for k in key if k.source == "unflagged") == 2
    assert sum(1 for k in key if k.source == "seeded") == 1
    # the flagged pool is exactly {flag1, flag2} -- both are picked since
    # n_flagged (2) equals the full flagged pool size, so this is
    # deterministic regardless of the RNG draw.
    assert {k.item_id for k in key if k.source == "flagged"} == {"flag1", "flag2"}


def test_review_key_maps_back_to_ground_truth() -> None:
    batch = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=_SEEDED, seed=42)
    key = build_review_key(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=_SEEDED, seed=42)

    assert [k.review_id for k in key] == [item.review_id for item in batch]

    sources = {k.source for k in key}
    assert sources == {"flagged", "unflagged", "seeded"}

    seeded_entries = [k for k in key if k.is_seeded]
    assert len(seeded_entries) == 1
    assert seeded_entries[0].item_id == "live1"
    assert seeded_entries[0].seed_kind == "swapped_reference"
    assert seeded_entries[0].source == "seeded"

    flagged_entries = [k for k in key if k.source == "flagged"]
    assert {e.item_id for e in flagged_entries} == {"flag1", "flag2"}
    assert all(e.classification == "mislabel_suspect" for e in flagged_entries)
    assert all(not e.is_seeded for e in flagged_entries)


def test_batch_order_is_seed_deterministic() -> None:
    batch_a = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=_SEEDED, seed=7)
    batch_b = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=_SEEDED, seed=7)
    assert batch_a == batch_b


def test_different_seeds_select_the_same_saturated_pool() -> None:
    """`n_flagged`/`n_unflagged` set to the *full* size of each pool (2
    flagged, 3 non-flagged) makes selection itself deterministic
    regardless of `seed` -- sampling an entire population always returns
    every member, whatever order the shuffle puts them in."""
    batch_a = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=3, seeded=(), seed=1)
    batch_b = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=3, seeded=(), seed=2)
    content_a = sorted((i.question, i.reference_answer, i.evidence_quote) for i in batch_a)
    content_b = sorted((i.question, i.reference_answer, i.evidence_quote) for i in batch_b)
    expected = sorted((m.question, m.reference_answer, m.evidence_quote) for m in _META.values())
    assert content_a == content_b == expected


def test_requesting_more_than_available_takes_what_exists_without_erroring() -> None:
    batch = build_review_batch(_STATS, _META, n_flagged=100, n_unflagged=100, seeded=(), seed=0)
    # only 2 mislabel_suspect + 3 non-flagged items exist in _STATS
    assert len(batch) == 5


# -- no-duplicate-item_id invariant (defect fix) ------------------------------


def _collision_pool(n: int = 40) -> tuple[list[ItemStats], dict[str, ItemMeta]]:
    """A pool realistic enough to reproduce the pre-fix collision: real
    question/reference text `seed_batch`'s corruptors can actually work
    on (long enough questions to truncate, reference answers with a
    negatable verb), varied `source_doc_id` for `foreign_evidence`
    donors, and a mix of every classification so flagged/unflagged/
    calibration all draw from one shared, overlapping pool rather than
    disjoint synthetic ones with no collision risk at all."""
    classifications: list[Classification] = (
        ["mislabel_suspect"] * 10 + ["live"] * 20 + ["dead_all_pass"] * 5 + ["dead_all_fail"] * 5
    )
    assert len(classifications) == n
    stats: list[ItemStats] = []
    meta: dict[str, ItemMeta] = {}
    for i, classification in enumerate(classifications):
        item_id = f"item{i:03d}"
        d = -0.4 if classification == "mislabel_suspect" else 0.4
        stats.append(
            ItemStats(
                item_id=item_id, p=0.5, discrimination_d=d, point_biserial=d, classification=classification
            )
        )
        meta[item_id] = ItemMeta(
            item_id=item_id,
            question=f"What is the fee for service {i} and who is responsible for paying it",
            reference_answer=f"The fee is {i} pounds and the applicant pays it",
            evidence_quote=f"Fee schedule entry {i} describing the applicable charge",
            source_doc_id=f"doc{i % 5}",
        )
    return stats, meta


def test_no_duplicate_item_id_in_deck_across_many_seeds() -> None:
    """The collision experiment IS the regression test (mirrors how
    defect 42's own regression test is the padding experiment itself):
    at this batch composition, on the real 2,364-item pool, 70% of decks
    contained a duplicated `item_id` before `_plan_batch`'s dedupe
    invariant existed."""
    stats, meta = _collision_pool()
    for trial_seed in range(50):
        seeded = seed_batch(list(meta.values()), n_per_kind=4, seed=trial_seed)
        key = build_review_key(
            stats, meta, n_flagged=5, n_unflagged=20, n_calibration=5, seeded=seeded, seed=trial_seed
        )
        ids = [entry.item_id for entry in key]
        assert len(set(ids)) == len(ids), f"seed={trial_seed}: duplicate item_id in deck: {ids}"


def test_flagged_items_survive_a_seed_colliding_with_a_flagged_item() -> None:
    """A `SeededItem` whose source item is one of the flagged picks must
    be dropped, never displace the flagged item -- claim order puts
    `flagged` first."""
    colliding_seed = SeededItem(
        item_id="flag1",
        kind="swapped_reference",
        question="Q flag1 corrupted?",
        reference_answer="corrupted reference",
        evidence_quote="E flag1",
        source_doc_id=None,
        detail="deliberately collides with a flagged item_id",
    )
    batch = build_review_batch(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=[colliding_seed], seed=42)
    key = build_review_key(_STATS, _META, n_flagged=2, n_unflagged=2, seeded=[colliding_seed], seed=42)
    # both requested flagged items are present -- the colliding seed did
    # not displace either (n_flagged=2 equals the full flagged pool size,
    # so this is deterministic regardless of the RNG draw)
    flagged_ids = {k.item_id for k in key if k.source == "flagged"}
    assert flagged_ids == {"flag1", "flag2"}
    # the colliding seed was dropped, not substituted for another id
    assert not any(k.source == "seeded" for k in key)
    assert len(batch) == len(key) == 4


def test_two_seeded_items_sharing_an_item_id_yield_at_most_one_slot() -> None:
    seed_a = SeededItem(
        item_id="live1", kind="swapped_reference", question="Q live1 a?",
        reference_answer="A a", evidence_quote="E a", source_doc_id=None, detail="a",
    )
    seed_b = SeededItem(
        item_id="live1", kind="negated_reference", question="Q live1 b?",
        reference_answer="A b", evidence_quote="E b", source_doc_id=None, detail="b",
    )
    key = build_review_key(_STATS, _META, n_flagged=0, n_unflagged=0, seeded=[seed_a, seed_b], seed=1)
    live1_entries = [k for k in key if k.item_id == "live1"]
    assert len(live1_entries) == 1
    # the first SeededItem in the given order wins
    assert live1_entries[0].seed_kind == "swapped_reference"


# -- calibration source --------------------------------------------------------


def _calibration_pool() -> tuple[list[ItemStats], dict[str, ItemMeta]]:
    ids_and_classes: list[tuple[str, Classification]] = [
        ("flagA", "mislabel_suspect"),
        ("flagB", "mislabel_suspect"),
        ("liveA", "live"),
        ("liveB", "live"),
        ("deadA", "dead_all_pass"),
    ]
    stats = [
        ItemStats(
            item_id=item_id, p=0.5, discrimination_d=-0.4 if c == "mislabel_suspect" else 0.4,
            point_biserial=0.0, classification=c,
        )
        for item_id, c in ids_and_classes
    ]
    meta = {item_id: _meta(item_id) for item_id, _ in ids_and_classes}
    return stats, meta


def test_calibration_draws_from_every_classification_not_only_unflagged() -> None:
    """Statistical: `n_flagged=1`/`n_unflagged=1` leaves one item of each
    pool unclaimed, so over many seeds `n_calibration=1` must sometimes
    land on the unclaimed `mislabel_suspect` item and sometimes on the
    `dead_all_pass` one -- calibration is never filtered to unflagged."""
    stats, meta = _calibration_pool()
    seen_classifications: set[str] = set()
    for trial_seed in range(300):
        key = build_review_key(
            stats, meta, n_flagged=1, n_unflagged=1, n_calibration=1, seeded=(), seed=trial_seed
        )
        seen_classifications.update(
            k.classification for k in key if k.source == "calibration" and k.classification is not None
        )
    assert "dead_all_pass" in seen_classifications
    assert "mislabel_suspect" in seen_classifications


def test_review_batch_leaks_nothing_with_calibration_present() -> None:
    stats, meta = _calibration_pool()
    batch = build_review_batch(stats, meta, n_flagged=1, n_unflagged=1, n_calibration=2, seeded=(), seed=3)
    for item in batch:
        dumped = item.model_dump()
        assert set(dumped) == {"review_id", "question", "reference_answer", "evidence_quote"}


# -- exclude: multi-pass calibration -------------------------------------------


def test_exclude_removes_item_from_every_pool() -> None:
    """An excluded item_id must not appear in the batch or the key, from
    any source -- flagged, unflagged, or calibration (seeded is its own
    named test below, `test_exclude_prevents_reappearance_via_seeded_pool`,
    since seeds are passed in pre-built and are the leak that matters
    most)."""
    exclude = frozenset({"flag1", "live1", "dead1"})
    key = build_review_key(
        _STATS, _META, n_flagged=2, n_unflagged=2, n_calibration=1, seeded=(), seed=42, exclude=exclude
    )
    batch = build_review_batch(
        _STATS, _META, n_flagged=2, n_unflagged=2, n_calibration=1, seeded=(), seed=42, exclude=exclude
    )
    item_ids = {k.item_id for k in key}
    assert item_ids.isdisjoint(exclude)
    # what remains after excluding one id from each of the three pools:
    # flag2 (flagged), live2 (unflagged) -- nothing left for calibration,
    # since claiming both leaves only the three excluded ids in the pool
    assert item_ids == {"flag2", "live2"}
    assert len(batch) == len(key) == 2


def test_exclude_prevents_reappearance_via_seeded_pool() -> None:
    """The cross-pass leak this feature exists to close: `seeded` is
    passed in pre-built, so excluding an id must be applied to the seed
    list itself, not only to the `stats`-derived pools -- otherwise an
    item a reviewer already saw (and excluded for that reason) could
    reappear in pass two, this time deliberately corrupted, which is
    worse than reappearing intact."""
    exclude = frozenset(si.item_id for si in _SEEDED)
    key = build_review_key(
        _STATS, _META, n_flagged=0, n_unflagged=0, seeded=_SEEDED, seed=42, exclude=exclude
    )
    batch = build_review_batch(
        _STATS, _META, n_flagged=0, n_unflagged=0, seeded=_SEEDED, seed=42, exclude=exclude
    )
    assert not any(k.is_seeded for k in key)
    assert not any(k.source == "seeded" for k in key)
    assert len(batch) == len(key) == 0


def test_exclude_composes_with_no_duplicate_invariant() -> None:
    """`exclude` and the at-most-once invariant are independent
    mechanisms that must both hold at once -- excluding some ids must
    not reopen the duplicate-item_id defect (see
    `test_no_duplicate_item_id_in_deck_across_many_seeds`) for the ids
    that remain."""
    stats, meta = _collision_pool()
    exclude = frozenset(f"item{i:03d}" for i in range(0, 40, 3))  # every third id
    for trial_seed in range(50):
        seeded = seed_batch(list(meta.values()), n_per_kind=4, seed=trial_seed)
        key = build_review_key(
            stats, meta, n_flagged=5, n_unflagged=20, n_calibration=5,
            seeded=seeded, seed=trial_seed, exclude=exclude,
        )
        ids = [entry.item_id for entry in key]
        assert len(set(ids)) == len(ids), f"seed={trial_seed}: duplicate item_id in deck: {ids}"
        assert exclude.isdisjoint(ids), f"seed={trial_seed}: excluded item_id leaked into deck: {ids}"


def test_excluding_everything_yields_empty_deck_not_an_error() -> None:
    exclude = frozenset(_META)
    key = build_review_key(
        _STATS, _META, n_flagged=2, n_unflagged=2, n_calibration=1, seeded=_SEEDED, seed=42, exclude=exclude
    )
    batch = build_review_batch(
        _STATS, _META, n_flagged=2, n_unflagged=2, n_calibration=1, seeded=_SEEDED, seed=42, exclude=exclude
    )
    assert key == []
    assert batch == []


# -- score_review: strict vs lenient dispositions -----------------------------


def test_score_review_strict_vs_lenient_gives_different_numerators() -> None:
    key = [
        ReviewKeyEntry(review_id="r0", item_id="flag1", source="flagged", is_seeded=False),
        ReviewKeyEntry(review_id="r1", item_id="flag2", source="flagged", is_seeded=False),
        ReviewKeyEntry(review_id="r2", item_id="flag3", source="flagged", is_seeded=False),
    ]
    labels = [
        ReviewLabel(review_id="r0", verdict="purge"),
        ReviewLabel(review_id="r1", verdict="rewrite"),
        # r2 has no matching label at all -- excluded from both numerator
        # and denominator, not counted as either verdict
    ]
    strict = score_review(labels, key)  # default bad_verdicts={"purge"}
    lenient = score_review(labels, key, bad_verdicts=frozenset({"purge", "rewrite"}))
    assert strict.precision_n == lenient.precision_n == 2  # r2 excluded from both denominators
    assert strict.precision == 0.5  # only r0 ("purge") counts
    assert lenient.precision == 1.0  # r0 and r1 ("purge"/"rewrite") both count
