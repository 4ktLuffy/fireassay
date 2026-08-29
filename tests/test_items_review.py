"""`test_items_review.py` (M-ITEMS-SPEC.md §6): a `ReviewItem` leaks
neither classification, statistics, nor seeded status; batch order is
seed-deterministic."""

from __future__ import annotations

from fireassay.items.core import Classification, ItemMeta, ItemStats
from fireassay.items.review import ReviewItem, build_review_batch, build_review_key
from fireassay.items.seed import SeededItem


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
