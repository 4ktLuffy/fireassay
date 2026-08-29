from __future__ import annotations

import pytest

from fireassay.models import Question
from fireassay.system.corpus import Chunk
from fireassay.system.tfidf import TfidfSystem


def _chunk(doc_id: str, index: int, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{index:04d}", text=text, char_start=0, char_end=len(text))


def _question(text: str) -> Question:
    return Question(text=text, qtype="factual", difficulty="easy", provenance="synthetic")


def test_hand_computed_ranking() -> None:
    """N=3 chunks:
        chunk0 "cat dog"      -> cat tf=1, dog tf=1
        chunk1 "cat cat cat"  -> cat tf=3
        chunk2 "bird fish"    -> bird tf=1, fish tf=1

    df(cat)=2 (chunk0, chunk1); df(dog)=df(bird)=df(fish)=1.
    idf(t) = log(N/df(t)):
        idf(cat)               = log(3/2) = 0.405465108
        idf(dog)=idf(bird)=idf(fish) = log(3/1) = 1.098612289

    L2-normalised chunk vectors:
        chunk0 raw (cat, dog) = (0.405465108, 1.098612289)
            norm = 1.171046931
            normalised = (0.346241553, 0.938145398)
        chunk1 raw (cat) = (1.216395324,) -- a single-term vector always
            normalises to weight 1.0 in that dimension, regardless of tf.
        chunk2 raw (bird, fish) = (1.098612289, 1.098612289)
            norm = 1.553512261
            normalised = (0.707106781, 0.707106781)

    Query "cat dog bird" (tf=1 each):
        query_vec = (cat: 0.405465108, dog: 1.098612289, bird: 1.098612289)
        query_norm = 1.605708528

    Cosine scores (dot(query_vec, normalised chunk_vec) / query_norm):
        chunk0 numerator = 0.405465108*0.346241553 + 1.098612289*0.938145398
                          = 1.171046931
            score = 1.171046931 / 1.605708528 = 0.729302305
        chunk1 numerator = 0.405465108*1.0 = 0.405465108
            score = 0.405465108 / 1.605708528 = 0.252514763
        chunk2 numerator = 1.098612289*0.707106781 = 0.776836199
            score = 0.776836199 / 1.605708528 = 0.483796521

    Expected order (descending score): chunk0, chunk2, chunk1.
    """
    chunks = [
        _chunk("doc0", 0, "cat dog"),
        _chunk("doc1", 0, "cat cat cat"),
        _chunk("doc2", 0, "bird fish"),
    ]
    system = TfidfSystem(chunks, top_k=3)
    output = system.answer(_question("cat dog bird"))
    assert [rc.chunk_id for rc in output.retrieved] == ["doc0#0000", "doc2#0000", "doc1#0000"]
    scores = {rc.chunk_id: rc.score for rc in output.retrieved}
    assert scores["doc0#0000"] == pytest.approx(0.729302305, abs=1e-5)
    assert scores["doc2#0000"] == pytest.approx(0.483796521, abs=1e-5)
    assert scores["doc1#0000"] == pytest.approx(0.252514763, abs=1e-5)


def test_deterministic_across_two_runs() -> None:
    chunks = [
        _chunk("doc0", 0, "cat dog"),
        _chunk("doc1", 0, "cat cat cat"),
        _chunk("doc2", 0, "bird fish"),
    ]
    system = TfidfSystem(chunks, top_k=3)
    question = _question("cat dog bird")
    out1 = system.answer(question)
    out2 = system.answer(question)
    assert [(rc.chunk_id, rc.rank, rc.score) for rc in out1.retrieved] == [
        (rc.chunk_id, rc.rank, rc.score) for rc in out2.retrieved
    ]


def test_tie_break_is_by_ascending_index_in_the_given_sequence() -> None:
    """Two chunks with identical text produce identical tf-idf vectors (the
    idf table is corpus-wide, and their term counts are identical), so they
    tie on cosine score. Per this module's documented contract, the tie is
    broken by ascending position in the `chunks` sequence passed to the
    constructor, not by `chunk_id` or any other identity -- so reordering
    the input sequence changes which one ranks first."""
    chunk_a = _chunk("docA", 0, "cat dog")
    chunk_b = _chunk("docB", 0, "cat dog")
    question = _question("cat dog")

    forward = TfidfSystem([chunk_a, chunk_b], top_k=2)
    out_forward = forward.answer(question)
    assert [rc.chunk_id for rc in out_forward.retrieved] == [chunk_a.chunk_id, chunk_b.chunk_id]

    reversed_system = TfidfSystem([chunk_b, chunk_a], top_k=2)
    out_reversed = reversed_system.answer(question)
    assert [rc.chunk_id for rc in out_reversed.retrieved] == [chunk_b.chunk_id, chunk_a.chunk_id]


def test_top_k_limits_number_of_results() -> None:
    chunks = [_chunk(f"doc{i}", 0, f"term{i} shared") for i in range(5)]
    system = TfidfSystem(chunks, top_k=3)
    output = system.answer(_question("shared"))
    assert len(output.retrieved) == 3


def test_top_k_larger_than_corpus_is_not_an_error() -> None:
    chunks = [_chunk("doc0", 0, "one term"), _chunk("doc1", 0, "two terms")]
    system = TfidfSystem(chunks, top_k=10)
    output = system.answer(_question("one two"))
    assert len(output.retrieved) == 2


def test_empty_query_returns_all_zero_scores_without_raising() -> None:
    chunks = [_chunk("doc0", 0, "cats"), _chunk("doc1", 0, "dogs")]
    system = TfidfSystem(chunks, top_k=2)
    output = system.answer(_question(""))
    assert len(output.retrieved) == 2
    assert all(rc.score == 0.0 for rc in output.retrieved)


def test_query_sharing_no_terms_returns_all_zero_scores_without_raising() -> None:
    chunks = [_chunk("doc0", 0, "cats"), _chunk("doc1", 0, "dogs")]
    system = TfidfSystem(chunks, top_k=2)
    output = system.answer(_question("zzzznonexistentqueryword"))
    assert len(output.retrieved) == 2
    assert all(rc.score == 0.0 for rc in output.retrieved)


def test_generates_answers_is_false() -> None:
    system = TfidfSystem([_chunk("doc0", 0, "cats")], top_k=1)
    assert system.generates_answers is False
