"""Controls for `tools/measure_detector_recall.py`'s unmeasurability check.

A response-matrix cell there is built from exactly four inputs --
`question`, `doc_id`, `char_start`, `char_end` -- via `responses_for`. A
corruption that leaves all four unchanged (`swapped_reference`,
`negated_reference`: both touch only `reference_answer`) cannot move a
single cell, so its measured recall is `0%` by construction, not by
detector weakness; the tool's last real run reported exactly that `0%`
for both kinds and mislabelled it as a result. These were, until now,
throwaway scripts -- landing them as tests is what makes the finding
re-runnable rather than unfalsifiable prose.

Uses a tiny, hermetic, two-document corpus and a 3-config BM25 panel
(never `data/corpus.jsonl` or `run/golden.db`, so this stays fast and
independent of the real corpus)."""

from __future__ import annotations

from fireassay.items.core import ItemMeta
from fireassay.items.seed import (
    SeededItem,
    foreign_evidence,
    negate_reference,
    swap_reference,
    truncate_question,
)
from fireassay.models import Question
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Doc, chunk_corpus
from helpers import load_measure_detector_recall

mdr = load_measure_detector_recall()

# -- a tiny, hermetic corpus: doc_a shares every content word with the
# query below, doc_b shares none, so BM25 ranks doc_a first regardless of
# chunking or top_k -- deterministic without relying on real corpus data.
_DOC_A_TEXT = "Photosynthesis converts sunlight into chemical energy in plants and algae."
_DOC_B_TEXT = "Interest rates influence borrowing costs across the entire economy today."
_DOCS = [
    Doc(doc_id="doc_a", title="", text=_DOC_A_TEXT),
    Doc(doc_id="doc_b", title="", text=_DOC_B_TEXT),
]

_ITEM1 = ItemMeta(
    item_id="item1",
    question="What process converts sunlight into energy in plants?",
    reference_answer="Photosynthesis is the process that converts sunlight into energy.",
    evidence_quote="Photosynthesis converts sunlight into chemical energy",
    source_doc_id="doc_a",
)
_ITEM2 = ItemMeta(
    item_id="item2",
    question="What influences borrowing costs across the economy?",
    reference_answer="Interest rates influence borrowing costs.",
    evidence_quote="Interest rates influence borrowing costs across the economy",
    source_doc_id="doc_b",
)
_POOL = [_ITEM1, _ITEM2]
_META_BY_ID = {_ITEM1.item_id: _ITEM1, _ITEM2.item_id: _ITEM2}

# `item1`'s gold span: chars [0, 20) of doc_a -- well inside doc_a's single
# chunk under every panel config below, whatever doc it ends up paired
# with once a corruption swaps `doc_id` out from under it.
_SPANS: dict[str, tuple[str, int, int]] = {"item1": ("doc_a", 0, 20), "item2": ("doc_b", 0, 20)}


def _build_panel() -> list[BM25System]:
    """3 configs, differing only in chunk size -- each doc is short enough
    to stay a single chunk under every size, so `top_k=1` always retrieves
    whichever document actually scores higher against the query, making
    the panel's behaviour depend only on BM25 ranking, never on chunk
    boundaries."""
    panel: list[BM25System] = []
    for size in (100, 150, 200):
        chunks = list(chunk_corpus(_DOCS, size=size, overlap=0))
        panel.append(BM25System(chunks, top_k=1))
    return panel


def _responses_for(
    panel: list[BM25System], question: str, doc_id: str, cs: int, ce: int
) -> tuple[bool, ...]:
    """The same "did any retrieved chunk overlap the gold span" cell
    `tools/measure_detector_recall.py`'s `responses_for` computes --
    reimplemented here (that function is a closure private to `main()`,
    not importable) rather than mocked, so these controls measure the
    real instrument's behaviour, not a stand-in for it."""
    q = Question(
        text=question, qtype="factual", difficulty="easy", reference_answer=None,
        provenance="synthetic", generator="probe@1",
    )
    return tuple(
        any(ch.doc_id == doc_id and ch.char_start < ce and ch.char_end > cs for ch in s.answer(q).retrieved)
        for s in panel
    )


# -- 1. negative control -----------------------------------------------------


def _assert_reference_only_corruption_is_unmeasurable(seeded: SeededItem) -> None:
    """Shared body for the two reference-only corruptions below:
    `seeded`'s response vector must come out byte-identical to the
    uncorrupted item's, since neither `question` nor `doc_id`/span moved.
    Also asserts the reference answers genuinely differ: without that,
    this could pass because the corruption silently did nothing (the
    exact bug the `seed.py` invariant now prevents), not because the
    instrument is blind to it."""
    assert seeded.reference_answer != _ITEM1.reference_answer

    orig_inputs = mdr.instrument_inputs(_ITEM1.item_id, None, _META_BY_ID, _SPANS)
    seed_inputs = mdr.instrument_inputs(_ITEM1.item_id, seeded, _META_BY_ID, _SPANS)
    assert seed_inputs == orig_inputs
    assert mdr.reached_instrument(seeded, _META_BY_ID, _SPANS) is False

    panel = _build_panel()
    orig_vector = _responses_for(panel, *orig_inputs)
    seed_vector = _responses_for(panel, *seed_inputs)
    assert seed_vector == orig_vector


def test_swapped_reference_yields_a_byte_identical_response_vector() -> None:
    _assert_reference_only_corruption_is_unmeasurable(swap_reference(_ITEM1, _POOL, seed=0))


def test_negated_reference_yields_a_byte_identical_response_vector() -> None:
    _assert_reference_only_corruption_is_unmeasurable(negate_reference(_ITEM1, _POOL, seed=0))


# -- 2. positive control ------------------------------------------------------


def test_foreign_evidence_changes_the_response_vector() -> None:
    """`foreign_evidence` moves `doc_id` to a different document -- unlike
    the reference-only kinds above, it must change the response vector.
    Without this control, the negative control above could pass with a
    broken panel that returns a constant vector regardless of input."""
    seeded = foreign_evidence(_ITEM1, _POOL, seed=0)
    assert seeded.source_doc_id == "doc_b"

    orig_inputs = mdr.instrument_inputs(_ITEM1.item_id, None, _META_BY_ID, _SPANS)
    seed_inputs = mdr.instrument_inputs(_ITEM1.item_id, seeded, _META_BY_ID, _SPANS)
    assert seed_inputs != orig_inputs
    assert mdr.reached_instrument(seeded, _META_BY_ID, _SPANS) is True

    panel = _build_panel()
    orig_vector = _responses_for(panel, *orig_inputs)
    seed_vector = _responses_for(panel, *seed_inputs)
    assert orig_vector != seed_vector
    # doc_a always outranks doc_b for this query (shares every content
    # word with doc_a, none with doc_b): the uncorrupted item's cell is
    # true everywhere, the foreign-evidence one false everywhere.
    assert all(orig_vector)
    assert not any(seed_vector)


# -- 3. the verdict -----------------------------------------------------------


def test_verdict_is_unmeasurable_for_reference_only_kinds_and_measurable_otherwise() -> None:
    """`count_reached_instrument`/`verdict_for_kind` -- Change 1's own
    unmeasurability check -- must report `UNMEASURABLE` (`0` seeds
    reached the instrument) for the two reference-only kinds and a real,
    nonzero count for the two kinds that move `question` or `doc_id`."""
    seeded = (
        [swap_reference(_ITEM1, _POOL, seed=s) for s in range(5)]
        + [negate_reference(_ITEM1, _POOL, seed=s) for s in range(5)]
        + [foreign_evidence(_ITEM1, _POOL, seed=s) for s in range(5)]
        + [truncate_question(_ITEM1, _POOL, seed=s) for s in range(5)]
    )
    reached_counts = mdr.count_reached_instrument(seeded, _META_BY_ID, _SPANS)

    assert reached_counts["swapped_reference"] == (0, 5)
    assert mdr.verdict_for_kind(*reached_counts["swapped_reference"]) == "UNMEASURABLE"

    assert reached_counts["negated_reference"] == (0, 5)
    assert mdr.verdict_for_kind(*reached_counts["negated_reference"]) == "UNMEASURABLE"

    assert reached_counts["foreign_evidence"] == (5, 5)
    assert mdr.verdict_for_kind(*reached_counts["foreign_evidence"]) == "measurable"

    assert reached_counts["truncated_question"] == (5, 5)
    assert mdr.verdict_for_kind(*reached_counts["truncated_question"]) == "measurable"
