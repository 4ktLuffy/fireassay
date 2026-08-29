"""`test_items_adapters_retrieval.py`: `lexical_decoy` seed construction
(`items.adapters.retrieval`) -- hermetic, a small synthetic corpus built
directly as `Chunk`s, no `data/corpus.jsonl`, no `run/golden.db`.

Vocabulary is deliberately made of nonsense tokens (`zeta`, `yentl`, ...)
so BM25's ranking is controlled entirely by how many of them each chunk
repeats, not by any real-world semantic content.

Also covers `tools/measure_detector_recall.py`'s `lexical_decoy_report` --
the denominator logic for that kind's recall line -- since this task's
write-scope has no dedicated `tools/` test file; loaded via `helpers.
load_measure_detector_recall` (the one shared loader for that module --
see its docstring), not a loader of this file's own."""

from __future__ import annotations

import pytest

from fireassay.items.adapters.retrieval import build_lexical_decoy_seeds, select_decoy_chunk
from fireassay.items.core import ItemMeta, ItemStats
from fireassay.system.corpus import Chunk
from helpers import load_measure_detector_recall

mdr = load_measure_detector_recall()

# Three single-chunk documents, each exactly 15 words (so BM25's length
# normalisation term is identical across all three and cannot perturb the
# ranking below), with a known BM25 ranking for the 5-term query: `source`
# and `decoyb` both repeat all 5 query terms; `farc` repeats only "zeta"
# (the one term all three documents share). `source`/`decoyb`/`farc`
# therefore rank, among non-source documents, `decoyb` clearly above
# `farc` -- the "known ranking, pick the middle one, not the lowest"
# fixture requirement 1 asks for -- for both the 5-term query and (via the
# doc_id tie-break, since all three then score identically) the 1-term
# query `test_no_outcome_conditioning...` uses.
_SOURCE_TEXT = "zeta yentl vor quixx meboo alpha bravo charlie delta echo foxtrot golf hotel india juliet"
_DECOYB_TEXT = "zeta yentl vor quixx meboo kilo lima mike november oscar papa quebec romeo sierra tango"
_FARC_TEXT = "zeta uniform victor whiskey xray yankee zulu apple banana cherry date fig grape honeydew ice"

_CHUNKS = [
    Chunk(
        doc_id="source", chunk_id="source#0000", text=_SOURCE_TEXT, char_start=0, char_end=len(_SOURCE_TEXT)
    ),
    Chunk(
        doc_id="decoyb", chunk_id="decoyb#0000", text=_DECOYB_TEXT, char_start=0, char_end=len(_DECOYB_TEXT)
    ),
    Chunk(doc_id="farc", chunk_id="farc#0000", text=_FARC_TEXT, char_start=0, char_end=len(_FARC_TEXT)),
]

_FULL_QUERY = "zeta yentl vor quixx meboo"


def _item(item_id: str = "q1", question: str = _FULL_QUERY, source_doc_id: str = "source") -> ItemMeta:
    return ItemMeta(
        item_id=item_id,
        question=question,
        reference_answer="the answer to q1",
        evidence_quote="original evidence text, from the source document itself",
        source_doc_id=source_doc_id,
    )


# -- requirement 1: highest-ranked NON-SOURCE chunk, not merely a different one ---


def test_select_decoy_chunk_picks_the_top_non_source_chunk_not_the_lowest() -> None:
    decoy = select_decoy_chunk(_item(), _CHUNKS)
    assert decoy.chunk.doc_id == "decoyb"
    assert decoy.chunk.doc_id != "farc"
    assert decoy.chunk.doc_id != "source"  # never the source, even though it likely scores highest


def test_build_lexical_decoy_seeds_relocates_to_the_top_non_source_document() -> None:
    seeded = build_lexical_decoy_seeds([_item()], _CHUNKS, n=1, seed=0)
    assert len(seeded) == 1
    assert seeded[0].source_doc_id == "decoyb"


# -- requirement 2: question/reference_answer untouched; span + quote moved -------


def test_build_lexical_decoy_seeds_leaves_question_and_reference_answer_untouched() -> None:
    item = _item()
    seeded = build_lexical_decoy_seeds([item], _CHUNKS, n=1, seed=0)[0]
    assert seeded.kind == "lexical_decoy"
    assert seeded.question == item.question
    assert seeded.reference_answer == item.reference_answer


def test_build_lexical_decoy_seeds_moves_source_doc_id_and_evidence_quote() -> None:
    item = _item()
    seeded = build_lexical_decoy_seeds([item], _CHUNKS, n=1, seed=0)[0]
    assert seeded.source_doc_id != item.source_doc_id
    assert seeded.source_doc_id == "decoyb"
    assert seeded.evidence_quote != item.evidence_quote
    assert seeded.evidence_quote == _DECOYB_TEXT


# -- requirement 3: deterministic in seed --------------------------------------


def test_build_lexical_decoy_seeds_deterministic_under_the_same_seed() -> None:
    first = build_lexical_decoy_seeds([_item()], _CHUNKS, n=1, seed=42)
    second = build_lexical_decoy_seeds([_item()], _CHUNKS, n=1, seed=42)
    assert first == second


def test_build_lexical_decoy_seeds_selection_differs_across_seeds() -> None:
    """A pool larger than `n` -- which items get picked must vary with
    `seed` (never all identical across a spread of seeds), and must not
    depend on anything about the items themselves (they are otherwise
    interchangeable here, differing only in `item_id`)."""
    pool = [_item(item_id=f"q{i}") for i in range(6)]
    selections = {
        frozenset(s.item_id for s in build_lexical_decoy_seeds(pool, _CHUNKS, n=2, seed=trial_seed))
        for trial_seed in range(10)
    }
    assert len(selections) > 1, "selection was identical across every seed tried -- seed is not driving it"


# -- requirement 4: ValueError when every retrieved chunk is the source's own -----

_SOURCE_ONLY_CHUNKS = [
    Chunk(
        doc_id="source", chunk_id="source#0000", text=_SOURCE_TEXT, char_start=0, char_end=len(_SOURCE_TEXT)
    ),
    Chunk(
        doc_id="source",
        chunk_id="source#0001",
        text=_DECOYB_TEXT,  # same doc_id as the item's source -- no non-source chunk exists at all
        char_start=len(_SOURCE_TEXT),
        char_end=len(_SOURCE_TEXT) + len(_DECOYB_TEXT),
    ),
]


def test_select_decoy_chunk_raises_when_every_retrieved_chunk_is_the_source_doc() -> None:
    with pytest.raises(ValueError, match="no non-source chunk"):
        select_decoy_chunk(_item(), _SOURCE_ONLY_CHUNKS)


def test_build_lexical_decoy_seeds_skips_an_item_it_cannot_build_a_decoy_for() -> None:
    """The caller skips a `ValueError`-raising item rather than
    propagating it or substituting something else, mirroring
    `items.seed.seed_batch`'s per-item skip-and-continue contract."""
    seeded = build_lexical_decoy_seeds([_item()], _SOURCE_ONLY_CHUNKS, n=1, seed=0)
    assert seeded == []


# -- requirement 5: detail names the decoy doc, rank, and score -------------------


def test_detail_names_decoy_doc_rank_and_score() -> None:
    seeded = build_lexical_decoy_seeds([_item()], _CHUNKS, n=1, seed=0)[0]
    assert "decoyb" in seeded.detail
    assert "rank" in seeded.detail
    assert "score" in seeded.detail


def test_select_decoy_chunk_returns_the_actual_rank_and_score() -> None:
    decoy = select_decoy_chunk(_item(), _CHUNKS)
    assert decoy.rank >= 1
    assert decoy.score > 0.0


# -- requirement 6: THE TRAP -- no outcome conditioning, both seeds emitted -------


def test_no_outcome_conditioning_a_strong_and_a_weak_decoy_are_both_emitted() -> None:
    """Regression test for the trap the module docstring warns about: a
    seed whose decoy is a strong, high-scoring lexical match (`"strong"`,
    querying with the full 5-term vocabulary) and one whose decoy is a
    weak, marginal match (`"weak"`, querying with a single shared term)
    must BOTH come back from `build_lexical_decoy_seeds` -- nothing in
    this module is allowed to prefer one over the other because one
    "worked better" than the other. If a future change adds a "did this
    decoy work?" filter, the weak item disappears and this test fails."""
    strong = _item(item_id="strong", question=_FULL_QUERY)
    weak = _item(item_id="weak", question="zeta")

    seeded = build_lexical_decoy_seeds([strong, weak], _CHUNKS, n=2, seed=0)

    assert {s.item_id for s in seeded} == {"strong", "weak"}

    by_id = {s.item_id: s for s in seeded}
    strong_decoy = select_decoy_chunk(strong, _CHUNKS)
    weak_decoy = select_decoy_chunk(weak, _CHUNKS)
    assert strong_decoy.score > weak_decoy.score  # confirms the two really were asymmetric
    assert by_id["strong"].source_doc_id == "decoyb"
    assert by_id["weak"].source_doc_id == "decoyb"


# -- regression: an unretrieved lexical_decoy seed stays in the recall
# denominator (tools/measure_detector_recall.py's report, not the builder
# above) -- the instruction this guards against was given and then
# retracted; see `LexicalDecoyReport`'s docstring for the full reasoning.


def test_lexical_decoy_report_keeps_an_unretrieved_seed_in_the_denominator() -> None:
    """A `lexical_decoy` seed no panel config ever retrieved (`p == 0`,
    classified `dead_all_fail` -- its decoy degenerated into
    `foreign_evidence`) is a genuine observation, not a non-observation:
    it must stay in `n_total`/`class_counts`, the PRIMARY recall
    denominator -- never excluded the way "reached the instrument" (a
    structural, outcome-independent fact) correctly excludes a
    byte-identical `swapped_reference`/`negated_reference` seed."""
    by_id: dict[str, ItemStats] = {
        "never_retrieved": ItemStats(
            item_id="never_retrieved",
            p=0.0,
            discrimination_d=0.0,
            point_biserial=0.0,
            classification="dead_all_fail",
        ),
        "retrieved_and_flagged": ItemStats(
            item_id="retrieved_and_flagged",
            p=0.5,
            discrimination_d=-0.3,
            point_biserial=-0.2,
            classification="mislabel_suspect",
        ),
    }
    decoy_ids = ["never_retrieved", "retrieved_and_flagged"]

    report = mdr.lexical_decoy_report(decoy_ids, by_id)

    assert report.n_total == 2  # BOTH seeds counted -- not narrowed to the retrieved one
    assert report.n_reached == 1  # diagnostic only: just one was ever retrieved by any config
    assert report.class_counts == {"dead_all_fail": 1, "mislabel_suspect": 1}

    anomalous = report.class_counts.get("mislabel_suspect", 0) + report.class_counts.get("dead_all_fail", 0)
    assert anomalous == 2
    # the primary recall divides by n_total (2), including the never-retrieved
    # seed -- dividing by n_reached (1) instead is exactly the excluded-
    # denominator mistake this test guards against, and would (wrongly)
    # report 200% recall
    assert anomalous / report.n_total == 1.0
    assert anomalous / report.n_reached == 2.0
