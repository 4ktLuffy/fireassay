from __future__ import annotations

import pytest

from fireassay.models import Question
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk
from fireassay.system.coverage import CoverageSystem


def _chunk(doc_id: str, index: int, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{index:04d}", text=text, char_start=0, char_end=len(text))


def _question(text: str) -> Question:
    return Question(text=text, qtype="factual", difficulty="easy", provenance="synthetic")


def test_hand_computed_ranking() -> None:
    """Query "cats like dogs" -> query_terms = {cats, like, dogs} (3 distinct).

    chunk0 "cats like fish"           -> terms {cats, like, fish}
        coverage = |{cats, like}| / 3 = 2/3 ~= 0.667
    chunk1 "dogs like bones and cats" -> terms {dogs, like, bones, and, cats}
        coverage = |{dogs, like, cats}| / 3 = 3/3 = 1.0
    chunk2 "birds fly"                -> terms {birds, fly}
        coverage = |{}| / 3 = 0/3 = 0.0

    Expected order (descending score): chunk1, chunk0, chunk2.
    """
    chunks = [
        _chunk("doc0", 0, "cats like fish"),
        _chunk("doc1", 0, "dogs like bones and cats"),
        _chunk("doc2", 0, "birds fly"),
    ]
    system = CoverageSystem(chunks, top_k=3)
    output = system.answer(_question("cats like dogs"))
    assert [rc.chunk_id for rc in output.retrieved] == ["doc1#0000", "doc0#0000", "doc2#0000"]
    scores = {rc.chunk_id: rc.score for rc in output.retrieved}
    assert scores["doc1#0000"] == pytest.approx(1.0)
    assert scores["doc0#0000"] == pytest.approx(2 / 3)
    assert scores["doc2#0000"] == pytest.approx(0.0)


def test_deterministic_across_two_runs() -> None:
    chunks = [
        _chunk("doc0", 0, "cats like fish"),
        _chunk("doc1", 0, "dogs like bones and cats"),
        _chunk("doc2", 0, "birds fly"),
    ]
    system = CoverageSystem(chunks, top_k=3)
    question = _question("cats like dogs")
    out1 = system.answer(question)
    out2 = system.answer(question)
    assert [(rc.chunk_id, rc.rank, rc.score) for rc in out1.retrieved] == [
        (rc.chunk_id, rc.rank, rc.score) for rc in out2.retrieved
    ]


def test_tie_break_is_by_ascending_index_in_the_given_sequence() -> None:
    """Two chunks with identical term sets tie on score. Per this module's
    documented contract, the tie is broken by ascending position in the
    `chunks` sequence passed to the constructor, not by `chunk_id` or any
    other identity -- so reordering the input sequence changes which one
    ranks first."""
    chunk_a = _chunk("docA", 0, "cats dogs")
    chunk_b = _chunk("docB", 0, "cats dogs")
    question = _question("cats dogs")

    forward = CoverageSystem([chunk_a, chunk_b], top_k=2)
    out_forward = forward.answer(question)
    assert [rc.chunk_id for rc in out_forward.retrieved] == [chunk_a.chunk_id, chunk_b.chunk_id]

    reversed_system = CoverageSystem([chunk_b, chunk_a], top_k=2)
    out_reversed = reversed_system.answer(question)
    assert [rc.chunk_id for rc in out_reversed.retrieved] == [chunk_b.chunk_id, chunk_a.chunk_id]


def test_top_k_limits_number_of_results() -> None:
    chunks = [_chunk(f"doc{i}", 0, f"term{i} shared") for i in range(5)]
    system = CoverageSystem(chunks, top_k=3)
    output = system.answer(_question("shared"))
    assert len(output.retrieved) == 3


def test_top_k_larger_than_corpus_is_not_an_error() -> None:
    chunks = [_chunk("doc0", 0, "one term"), _chunk("doc1", 0, "two terms")]
    system = CoverageSystem(chunks, top_k=10)
    output = system.answer(_question("one two"))
    assert len(output.retrieved) == 2


def test_empty_query_returns_all_zero_scores_without_raising() -> None:
    chunks = [_chunk("doc0", 0, "cats"), _chunk("doc1", 0, "dogs")]
    system = CoverageSystem(chunks, top_k=2)
    output = system.answer(_question(""))
    assert len(output.retrieved) == 2
    assert all(rc.score == 0.0 for rc in output.retrieved)


def test_query_sharing_no_terms_returns_all_zero_scores_without_raising() -> None:
    chunks = [_chunk("doc0", 0, "cats"), _chunk("doc1", 0, "dogs")]
    system = CoverageSystem(chunks, top_k=2)
    output = system.answer(_question("zzzznonexistentqueryword"))
    assert len(output.retrieved) == 2
    assert all(rc.score == 0.0 for rc in output.retrieved)


def test_generates_answers_is_false() -> None:
    system = CoverageSystem([_chunk("doc0", 0, "cats")], top_k=1)
    assert system.generates_answers is False


def test_disagrees_with_bm25_on_top_1() -> None:
    """A chunk that repeats one rare query term many times (high BM25 idf
    x near-saturated term frequency) versus a chunk that mentions several
    common query terms once each (low BM25 idf per term, but full term
    coverage). BM25 must prefer the repetition chunk; CoverageSystem must
    prefer the breadth chunk -- this is the property the whole change
    exists to obtain.

    N=5 docs. idf here is BM25's own smoothed formula (bm25.py):
    idf(xenon) = log(1 + (5-1+0.5)/(1+0.5)) = log(4) ~= 1.386  (df=1/5)
    idf(common1) = idf(common2)
        = log(1 + (5-4+0.5)/(4+0.5)) = log(1.333) ~= 0.288    (df=4/5)
    avg_len = (8 + 3 + 5 + 5 + 5) / 5 = 5.2

    BM25(doc0, "xenon" x8, dl=8), only "xenon" matches:
        denom = 8 + 1.5*(0.25 + 0.75*8/5.2) ~= 10.106
        score = 1.386 * (8*2.5 / 10.106) ~= 2.743
    BM25(doc1, "common1 common2 filler", dl=3), both commons match:
        denom = 1 + 1.5*(0.25 + 0.75*3/5.2) ~= 2.024
        score = 2 * 0.288 * (1*2.5 / 2.024) ~= 0.711
    -> BM25 top-1 = doc0.

    Coverage(doc0) = |{xenon}| / 3 ~= 0.333
    Coverage(doc1) = |{common1, common2}| / 3 ~= 0.667 (tied with doc2-4,
        doc1 wins the tie: lowest index among them)
    -> CoverageSystem top-1 = doc1.
    """
    chunks = [
        _chunk("doc0", 0, "xenon xenon xenon xenon xenon xenon xenon xenon"),
        _chunk("doc1", 0, "common1 common2 filler"),
        _chunk("doc2", 0, "common1 common2 other words here"),
        _chunk("doc3", 0, "common1 common2 more filler text"),
        _chunk("doc4", 0, "common1 common2 yet another filler"),
    ]
    question = _question("xenon common1 common2")

    bm25_top1 = BM25System(chunks, top_k=1).answer(question).retrieved[0].doc_id
    coverage_top1 = CoverageSystem(chunks, top_k=1).answer(question).retrieved[0].doc_id

    assert bm25_top1 == "doc0"
    assert coverage_top1 == "doc1"
    assert bm25_top1 != coverage_top1
