"""`test_items_seed.py` (M-ITEMS-SPEC.md §6): each corruption is applied
and recorded; deterministic under a seed."""

from __future__ import annotations

import pytest

from fireassay.items.core import ItemMeta
from fireassay.items.seed import (
    ALL_SEED_KINDS,
    _seeded_item,
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


def test_swap_reference_never_returns_an_item_with_an_unchanged_reference_answer() -> None:
    """A donor whose `reference_answer` happens to equal the item's own
    is not a corruption -- it must be excluded, never swapped in
    unchanged (the `swap_reference` half of the invariant defect)."""
    same_a = ItemMeta(item_id="same_a", reference_answer="Same answer")
    same_b = ItemMeta(item_id="same_b", reference_answer="Same answer")
    # same_a and same_b are each other's only possible donor, and share
    # the exact same reference_answer -- no valid swap exists
    with pytest.raises(ValueError, match="no donor"):
        swap_reference(same_a, [same_a, same_b], seed=0)

    # adding a genuinely different donor makes the swap succeed, and it
    # is that donor's answer that is picked, never the unchanged one
    different = ItemMeta(item_id="different", reference_answer="A different answer entirely")
    seeded = swap_reference(same_a, [same_a, same_b, different], seed=0)
    assert seeded.reference_answer == "A different answer entirely"
    assert seeded.reference_answer != same_a.reference_answer


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


def test_negate_reference_bare_no_yields_yes_not_an_unchanged_no() -> None:
    """A bare `"No"` reference answer used to fall through to being
    returned unchanged (removing its only word left nothing, and the old
    code silently returned the input) while still claiming a negation was
    removed -- see the module docstring."""
    item = ItemMeta(item_id="bareno", reference_answer="No")
    seeded = negate_reference(item, _POOL, seed=0)
    assert seeded.reference_answer == "Yes"
    assert seeded.reference_answer != "No"
    assert "substituted" in seeded.detail
    assert "'Yes'" in seeded.detail


def test_negate_reference_bare_lowercase_no_yields_lowercase_yes() -> None:
    item = ItemMeta(item_id="barenolower", reference_answer="no")
    seeded = negate_reference(item, _POOL, seed=0)
    assert seeded.reference_answer == "yes"
    assert seeded.reference_answer != "no"


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


# -- invariant: no corruptor may emit an unchanged item ----------------------


def _invariant_pool() -> list[ItemMeta]:
    """A pool built specifically to hit both root causes the invariant
    fixes: several items whose entire `reference_answer` is exactly
    `"No"`/`"no"` (the case that broke `_negate_text`), and a pair
    sharing the exact same `reference_answer` as each other (the case
    that broke `swap_reference`'s donor filter) -- plus enough ordinary
    items that every kind has real material to work with."""
    items = [
        ItemMeta(
            item_id=f"item{i}",
            question=f"Is service {i} available on weekends and during public holidays too",
            reference_answer=f"Service {i} is available on weekends.",
            evidence_quote=f"Service {i} operates every day including weekends.",
            source_doc_id=f"doc{i % 4}",
        )
        for i in range(20)
    ]
    for i, text in enumerate(["No", "no", "No", "no"]):
        items.append(
            ItemMeta(
                item_id=f"bareno{i}",
                question=f"Is feature {i} enabled by default in the standard configuration",
                reference_answer=text,
                evidence_quote=f"Feature {i} is disabled by default in every configuration.",
                source_doc_id=f"doc{i % 4}",
            )
        )
    items.append(
        ItemMeta(
            item_id="dup1", question="Is the premium tier included at no extra cost",
            reference_answer="No", evidence_quote="quote for dup1", source_doc_id="docA",
        )
    )
    items.append(
        ItemMeta(
            item_id="dup2", question="Is priority support included at no extra cost",
            reference_answer="No", evidence_quote="quote for dup2", source_doc_id="docA",
        )
    )
    return items


def test_no_corruptor_ever_emits_an_unchanged_item() -> None:
    """The invariant IS the regression test (mirrors how defect 42's own
    regression test is the padding experiment itself): seed every item
    with every kind, over many seeds, and assert no emitted `SeededItem`
    equals its source on all three of question/reference_answer/
    evidence_quote. Measured on the real 2,364-item pool x 4 kinds
    (9,456 seeds) before this fix: 47 `negated_reference` and 1
    `swapped_reference` seed were byte-identical to their source."""
    pool = _invariant_pool()
    by_id = {m.item_id: m for m in pool}
    total_seeded = 0
    for trial_seed in range(20):
        batch = seed_batch(pool, n_per_kind=len(pool), seed=trial_seed)
        total_seeded += len(batch)
        for seeded in batch:
            source = by_id[seeded.item_id]
            unchanged = (
                seeded.question == source.question
                and seeded.reference_answer == source.reference_answer
                and seeded.evidence_quote == source.evidence_quote
            )
            assert not unchanged, (
                f"seed={trial_seed} kind={seeded.kind} item={seeded.item_id} "
                f"was emitted unchanged from its source"
            )
    assert total_seeded > 0  # not a vacuous pass


def test_seeded_item_refuses_to_construct_an_unchanged_result() -> None:
    """The central enforcement point directly: any corruptor -- present
    or future -- that hands it back the source's own fields raises,
    naming the item and kind, rather than silently constructing a
    `SeededItem` that changed nothing."""
    item = ItemMeta(item_id="x", question="Q", reference_answer="A", evidence_quote="E")
    with pytest.raises(ValueError, match=r"truncated_question.*'x'.*no change"):
        _seeded_item(
            item,
            kind="truncated_question",
            question="Q",
            reference_answer="A",
            evidence_quote="E",
            source_doc_id=None,
            detail="no-op",
        )


def test_seed_batch_skips_an_item_ruled_out_by_the_invariant_rather_than_erroring() -> None:
    """`same_a`/`same_b` share the exact same `reference_answer` and are
    each other's only possible donor -- `swap_reference`'s invariant-
    driven donor filter rules both out for `swapped_reference`, and
    `seed_batch` must not propagate the resulting `ValueError`, just skip
    them and continue: `truncated_question` does not depend on
    `reference_answer` at all, and still succeeds for both in the same
    call."""
    same_a = ItemMeta(
        item_id="same_a",
        question="Is the refund policy the same for every region worldwide today",
        reference_answer="Same answer",
    )
    same_b = ItemMeta(
        item_id="same_b",
        question="Is the return policy identical across every store location today",
        reference_answer="Same answer",
    )
    pool = [same_a, same_b]
    batch = seed_batch(pool, kinds=("swapped_reference", "truncated_question"), n_per_kind=2, seed=0)

    assert not any(s.kind == "swapped_reference" for s in batch)
    truncated = {s.item_id for s in batch if s.kind == "truncated_question"}
    assert truncated == {"same_a", "same_b"}
