"""`test_items_seed.py` (M-ITEMS-SPEC.md §6): each corruption is applied
and recorded; deterministic under a seed."""

from __future__ import annotations

import pytest

from fireassay.items.core import ItemMeta
from fireassay.items.seed import (
    ALL_SEED_KINDS,
    foreign_evidence,
    negate_reference,
    seed_batch,
    swap_reference,
    truncate_question,
)

_ITEM_A = ItemMeta(
    item_id="a",
    question="What is the capital of France and why does it matter here?",
    reference_answer="Paris is the capital.",
    evidence_quote="Paris has been the capital since 987.",
    source_doc_id="doc-france",
)
_ITEM_B = ItemMeta(
    item_id="b",
    question="What is the capital of Germany?",
    reference_answer="Berlin is the capital.",
    evidence_quote="Berlin has been the capital since reunification.",
    source_doc_id="doc-germany",
)
_ITEM_C = ItemMeta(
    item_id="c",
    question="What is the capital of Italy?",
    reference_answer="Rome is the capital.",
    evidence_quote="Rome has been the capital since 1871.",
    source_doc_id="doc-italy",
)
_POOL = [_ITEM_A, _ITEM_B, _ITEM_C]


# -- swapped_reference --------------------------------------------------


def test_swap_reference_takes_a_different_items_reference_answer() -> None:
    seeded = swap_reference(_ITEM_A, _POOL, seed=0)
    assert seeded.kind == "swapped_reference"
    assert seeded.item_id == "a"
    assert seeded.reference_answer in {_ITEM_B.reference_answer, _ITEM_C.reference_answer}
    assert seeded.reference_answer != _ITEM_A.reference_answer
    # everything else about the item is untouched
    assert seeded.question == _ITEM_A.question
    assert seeded.evidence_quote == _ITEM_A.evidence_quote
    assert "swapped from item" in seeded.detail


def test_swap_reference_deterministic_under_a_seed() -> None:
    first = swap_reference(_ITEM_A, _POOL, seed=123)
    second = swap_reference(_ITEM_A, _POOL, seed=123)
    assert first == second


def test_swap_reference_raises_without_a_donor() -> None:
    lone = ItemMeta(item_id="only", reference_answer="X")
    with pytest.raises(ValueError, match="no donor"):
        swap_reference(lone, [lone], seed=0)


# -- foreign_evidence -----------------------------------------------------


def test_foreign_evidence_takes_quote_and_doc_from_a_different_document() -> None:
    seeded = foreign_evidence(_ITEM_A, _POOL, seed=0)
    assert seeded.kind == "foreign_evidence"
    assert seeded.source_doc_id in {_ITEM_B.source_doc_id, _ITEM_C.source_doc_id}
    assert seeded.source_doc_id != _ITEM_A.source_doc_id
    assert seeded.evidence_quote != _ITEM_A.evidence_quote
    assert seeded.question == _ITEM_A.question
    assert seeded.reference_answer == _ITEM_A.reference_answer
    assert "swapped from item" in seeded.detail


def test_foreign_evidence_deterministic_under_a_seed() -> None:
    first = foreign_evidence(_ITEM_A, _POOL, seed=99)
    second = foreign_evidence(_ITEM_A, _POOL, seed=99)
    assert first == second


def test_foreign_evidence_raises_without_a_different_document_in_pool() -> None:
    same_doc = ItemMeta(item_id="same", evidence_quote="x", source_doc_id="doc-france")
    with pytest.raises(ValueError, match="no donor"):
        foreign_evidence(_ITEM_A, [_ITEM_A, same_doc], seed=0)


# -- truncated_question ---------------------------------------------------


def test_truncate_question_cuts_mid_clause() -> None:
    seeded = truncate_question(_ITEM_A, _POOL, seed=0)
    assert seeded.kind == "truncated_question"
    assert seeded.question is not None
    assert seeded.question != _ITEM_A.question
    assert _ITEM_A.question is not None
    assert _ITEM_A.question.startswith(seeded.question)
    truncated_words = seeded.question.split()
    original_words = _ITEM_A.question.split()
    assert 0 < len(truncated_words) < len(original_words)
    assert "truncated to" in seeded.detail


def test_truncate_question_deterministic_under_a_seed() -> None:
    first = truncate_question(_ITEM_A, _POOL, seed=5)
    second = truncate_question(_ITEM_A, _POOL, seed=5)
    assert first == second


def test_truncate_question_raises_on_a_too_short_question() -> None:
    short = ItemMeta(item_id="short", question="Hi there")
    with pytest.raises(ValueError, match="too short"):
        truncate_question(short, _POOL, seed=0)


def test_truncate_question_raises_with_no_question() -> None:
    blank = ItemMeta(item_id="blank")
    with pytest.raises(ValueError, match="no question"):
        truncate_question(blank, _POOL, seed=0)


# -- negated_reference ------------------------------------------------------


def test_negate_reference_inserts_not_when_an_auxiliary_verb_is_present() -> None:
    item = ItemMeta(item_id="yn", reference_answer="Yes, the treaty was ratified.")
    seeded = negate_reference(item, _POOL, seed=0)
    assert seeded.kind == "negated_reference"
    assert seeded.reference_answer == "Yes, the treaty was not ratified."
    assert "polarity flipped" in seeded.detail


def test_negate_reference_inserts_not_after_an_auxiliary_verb() -> None:
    item = ItemMeta(item_id="aux", reference_answer="The treaty is valid today.")
    seeded = negate_reference(item, _POOL, seed=0)
    assert seeded.reference_answer == "The treaty is not valid today."


def test_negate_reference_removes_an_existing_negation_word() -> None:
    item = ItemMeta(item_id="neg", reference_answer="The treaty is not valid today.")
    seeded = negate_reference(item, _POOL, seed=0)
    assert seeded.reference_answer == "The treaty is valid today."


def test_negate_reference_deterministic() -> None:
    item = ItemMeta(item_id="x", reference_answer="The treaty is valid today.")
    first = negate_reference(item, _POOL, seed=0)
    second = negate_reference(item, _POOL, seed=0)
    assert first == second


def test_negate_reference_raises_without_a_reference_answer() -> None:
    blank = ItemMeta(item_id="blank")
    with pytest.raises(ValueError, match="no reference_answer"):
        negate_reference(blank, _POOL, seed=0)


# -- seed_batch -------------------------------------------------------------


def test_seed_batch_covers_every_requested_kind() -> None:
    batch = seed_batch(_POOL, n_per_kind=1, seed=0)
    kinds_seen = {s.kind for s in batch}
    assert kinds_seen == set(ALL_SEED_KINDS)


def test_seed_batch_deterministic_under_a_seed() -> None:
    first = seed_batch(_POOL, n_per_kind=2, seed=42)
    second = seed_batch(_POOL, n_per_kind=2, seed=42)
    assert first == second


def test_seed_batch_respects_n_per_kind_cap() -> None:
    batch = seed_batch(_POOL, kinds=("swapped_reference",), n_per_kind=2, seed=0)
    assert len(batch) == 2
    assert all(s.kind == "swapped_reference" for s in batch)


def test_seed_batch_skips_items_that_cannot_satisfy_a_kind_rather_than_erroring() -> None:
    """A pool of a single item with no field `truncated_question` needs
    (too-short question) yields zero seeds for that kind, not an
    exception."""
    tiny_pool = [ItemMeta(item_id="only", question="Hi", reference_answer="x")]
    batch = seed_batch(tiny_pool, kinds=("truncated_question",), n_per_kind=3, seed=0)
    assert batch == []
