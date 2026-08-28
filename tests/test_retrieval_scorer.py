from __future__ import annotations

import math

import pytest

from fireassay.models import EvidenceSpan, Question, RetrievedChunk, Score, SystemOutput
from fireassay.score.base import ScoringContext
from fireassay.score.retrieval import RetrievalScorer


def _question_with_gold(spans: list[EvidenceSpan]) -> Question:
    return Question(
        text="q", qtype="factual", difficulty="easy", provenance="synthetic", evidence_spans=tuple(spans)
    )


def _chunk(doc_id: str, chunk_id: str, rank: int, char_start: int, char_end: int) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        score=1.0 / rank,
        rank=rank,
        char_start=char_start,
        char_end=char_end,
    )


def _output(chunks: list[RetrievedChunk]) -> SystemOutput:
    return SystemOutput(answer=None, abstained=True, retrieved=tuple(chunks), latency_ms={"total": 1.0})


def _span(doc_id: str, start: int, end: int) -> EvidenceSpan:
    return EvidenceSpan(doc_id=doc_id, chunk_id="unused", char_start=start, char_end=end, quote="x")


def _by_metric(scores: list[Score]) -> dict[str, float]:
    return {s.metric: s.value for s in scores}


def test_no_gold_evidence_spans_yields_zero_score_objects() -> None:
    """The load-bearing 'omit rather than zero' rule (M1-SPEC.md §7): a
    question with no evidence spans carries no information about
    retrieval and must not appear in a retrieval average at all."""
    question = _question_with_gold([])
    output = _output([_chunk("d", "d#0000", 1, 0, 50)])
    ctx = ScoringContext()

    scores = RetrievalScorer().score(question, output, ctx)
    assert scores == []


def test_recall_ndcg_mrr_hand_computed_at_k5() -> None:
    # Gold spans: [0,10) and [40,50) in doc d.
    # Retrieved: rank1 [100,110) no overlap; rank2 [0,10) covers span A;
    #            rank3 [200,210) no overlap; rank4 [40,50) covers span B; rank5 no overlap.
    gold = [_span("d", 0, 10), _span("d", 40, 50)]
    question = _question_with_gold(gold)
    output = _output(
        [
            _chunk("d", "d#0100", 1, 100, 110),
            _chunk("d", "d#0000", 2, 0, 10),
            _chunk("d", "d#0200", 3, 200, 210),
            _chunk("d", "d#0040", 4, 40, 50),
            _chunk("d", "d#0300", 5, 300, 310),
        ]
    )
    ctx = ScoringContext(eval_ks=(5,))

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))

    assert by_metric["retrieval.recall@5"] == pytest.approx(1.0)  # both spans covered

    dcg = 1.0 / math.log2(2 + 1) + 1.0 / math.log2(4 + 1)  # relevant chunks at ranks 2 and 4
    idcg = 1.0 / math.log2(1 + 1) + 1.0 / math.log2(2 + 1)  # best case: ranks 1 and 2
    assert by_metric["retrieval.ndcg@5"] == pytest.approx(dcg / idcg)

    assert by_metric["retrieval.mrr"] == pytest.approx(0.5)  # first relevant chunk at rank 2


def test_sweeps_every_k_in_eval_ks_and_truncates_to_retrieved_length() -> None:
    gold = [_span("d", 0, 10)]
    question = _question_with_gold(gold)
    output = _output([_chunk("d", "d#0000", 1, 0, 10), _chunk("d", "e#0000", 2, 500, 510)])
    ctx = ScoringContext(eval_ks=(1, 3, 5, 10))

    scores = RetrievalScorer().score(question, output, ctx)
    by_metric = _by_metric(scores)

    # Every requested k produces a recall/ndcg/judged_fraction metric, even
    # k's larger than the 2-chunk retrieved list (truncated to
    # len(retrieved) internally), plus the single non-@k metrics.
    assert set(by_metric) == {
        "retrieval.recall@1",
        "retrieval.recall@3",
        "retrieval.recall@5",
        "retrieval.recall@10",
        "retrieval.ndcg@1",
        "retrieval.ndcg@3",
        "retrieval.ndcg@5",
        "retrieval.ndcg@10",
        "retrieval.judged_fraction@1",
        "retrieval.judged_fraction@3",
        "retrieval.judged_fraction@5",
        "retrieval.judged_fraction@10",
        "retrieval.mrr",
        "retrieval.bpref",
    }
    # The gold span is covered by the rank-1 chunk, so every k >= 1 sees it.
    for k in (1, 3, 5, 10):
        assert by_metric[f"retrieval.recall@{k}"] == pytest.approx(1.0)


def test_recall_monotonically_non_decreasing_in_k() -> None:
    gold = [_span("d", 0, 10), _span("d", 1000, 1010)]
    question = _question_with_gold(gold)
    output = _output(
        [
            _chunk("d", "d#a", 1, 500, 510),  # no coverage
            _chunk("d", "d#b", 2, 0, 10),  # covers span 1
            _chunk("d", "d#c", 3, 600, 610),  # no coverage
            _chunk("d", "d#d", 4, 1000, 1010),  # covers span 2
        ]
    )
    ctx = ScoringContext(eval_ks=(1, 2, 3, 4))
    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))

    values = [by_metric[f"retrieval.recall@{k}"] for k in (1, 2, 3, 4)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)


def test_no_hits_yields_zero_recall_ndcg_mrr() -> None:
    question = _question_with_gold([_span("d", 0, 10)])
    output = _output([_chunk("d", "d#0999", 1, 999, 1010)])  # no overlap
    ctx = ScoringContext(eval_ks=(5,))

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.recall@5"] == pytest.approx(0.0)
    assert by_metric["retrieval.ndcg@5"] == pytest.approx(0.0)
    assert by_metric["retrieval.mrr"] == pytest.approx(0.0)


def test_overlap_requires_same_doc_id() -> None:
    question = _question_with_gold([_span("docA", 0, 10)])
    output = _output([_chunk("docB", "docB#0000", 1, 0, 10)])  # same range, different doc
    ctx = ScoringContext(eval_ks=(5,))

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.recall@5"] == pytest.approx(0.0)


def test_one_chunk_covering_multiple_gold_spans_scores_full_recall() -> None:
    # A single wide chunk [0, 100) covers three separate gold spans.
    gold = [_span("d", 0, 10), _span("d", 40, 50), _span("d", 80, 90)]
    question = _question_with_gold(gold)
    output = _output([_chunk("d", "d#wide", 1, 0, 100)])
    ctx = ScoringContext(eval_ks=(1,))

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.recall@1"] == pytest.approx(1.0)


def test_two_chunks_covering_the_same_span_count_it_once() -> None:
    gold = [_span("d", 0, 10)]
    question = _question_with_gold(gold)
    output = _output([_chunk("d", "d#a", 1, 0, 5), _chunk("d", "d#b", 2, 5, 10)])
    ctx = ScoringContext(eval_ks=(2,))

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.recall@2"] == pytest.approx(1.0)  # not 2.0


def test_recall_identical_across_two_chunking_strategies_full_coverage() -> None:
    """Regression test for M1-SPEC.md correction 1: retrieval relevance
    must be judged by character-range overlap, not chunk_id equality, so
    that recall is comparable across a chunking config axis. Two entirely
    different chunkings of the same underlying text, both of which happen
    to fully cover the same gold spans, must produce identical recall
    (and, since both retrieve the covering chunk at rank 1, identical
    nDCG and MRR too) -- even though their chunk_ids are completely
    disjoint.
    """
    gold = [_span("d", 10, 20), _span("d", 200, 210)]
    question = _question_with_gold(gold)

    # Chunking "512": one big chunk per span-containing region.
    output_512 = _output(
        [
            _chunk("d", "d#0000", 1, 0, 512),  # covers both spans (10-20 and 200-210)
        ]
    )
    # Chunking "1024": totally different boundaries and chunk_ids, but
    # still fully covers both spans within a single retrieved chunk.
    output_1024 = _output(
        [
            _chunk("d", "d#chunkX", 1, 0, 1024),
        ]
    )

    ctx = ScoringContext(eval_ks=(1, 3, 5))
    scores_512 = _by_metric(RetrievalScorer().score(question, output_512, ctx))
    scores_1024 = _by_metric(RetrievalScorer().score(question, output_1024, ctx))

    for k in (1, 3, 5):
        assert scores_512[f"retrieval.recall@{k}"] == pytest.approx(1.0)
        assert scores_1024[f"retrieval.recall@{k}"] == pytest.approx(1.0)
        assert scores_512[f"retrieval.recall@{k}"] == pytest.approx(scores_1024[f"retrieval.recall@{k}"])
        assert scores_512[f"retrieval.ndcg@{k}"] == pytest.approx(scores_1024[f"retrieval.ndcg@{k}"])
    assert scores_512["retrieval.mrr"] == pytest.approx(scores_1024["retrieval.mrr"]) == pytest.approx(1.0)


def test_ndcg_never_exceeds_one_when_more_relevant_chunks_than_gold_spans() -> None:
    """Regression test: relevance is per-chunk overlap, not per-span
    chunk_id membership, so a chunking strategy can retrieve *more*
    distinct relevant chunks than there are gold spans (e.g. two adjacent
    chunks each partially overlapping one wide gold span). ndcg@k must
    still never exceed 1.0."""
    gold = [_span("d", 0, 100)]  # a single, wide gold span
    question = _question_with_gold(gold)
    # Three different retrieved chunks all independently overlap the one
    # gold span.
    output = _output(
        [
            _chunk("d", "d#a", 1, 0, 40),
            _chunk("d", "d#b", 2, 40, 70),
            _chunk("d", "d#c", 3, 70, 100),
        ]
    )
    ctx = ScoringContext(eval_ks=(1, 2, 3))

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    for k in (1, 2, 3):
        assert 0.0 <= by_metric[f"retrieval.ndcg@{k}"] <= 1.0
    # All three ranks are relevant, and they occupy ranks 1..3, which is
    # the maximum achievable arrangement, so ndcg@3 == 1.0 exactly.
    assert by_metric["retrieval.ndcg@3"] == pytest.approx(1.0)


def test_overlap_min_chars_gates_relevance() -> None:
    gold = [_span("d", 0, 10)]
    question = _question_with_gold(gold)
    output = _output([_chunk("d", "d#0000", 1, 9, 20)])  # overlaps [0,10) by exactly 1 char: [9,10)
    ctx_loose = ScoringContext(eval_ks=(1,), overlap_min_chars=1)
    ctx_strict = ScoringContext(eval_ks=(1,), overlap_min_chars=2)

    loose = _by_metric(RetrievalScorer().score(question, output, ctx_loose))
    strict = _by_metric(RetrievalScorer().score(question, output, ctx_strict))

    assert loose["retrieval.recall@1"] == pytest.approx(1.0)
    assert strict["retrieval.recall@1"] == pytest.approx(0.0)


# -- retrieval.judged_fraction ----------------------------------------------


def test_judged_fraction_defaults_to_zero_when_pool_is_empty() -> None:
    """A ScoringContext with no judged_spans_by_doc (the default, used by
    every other test in this file) means nothing is judged, so
    judged_fraction is 0.0 -- even for a chunk that happens to cover this
    question's own gold span, because judged_fraction is defined purely
    off ctx.judged_spans_by_doc, not off the question's own gold spans."""
    gold = [_span("d", 0, 10)]
    question = _question_with_gold(gold)
    output = _output([_chunk("d", "d#0000", 1, 0, 10)])
    ctx = ScoringContext(eval_ks=(1,))  # judged_spans_by_doc defaults to {}

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.judged_fraction@1"] == pytest.approx(0.0)


def test_judged_fraction_counts_suite_wide_judged_pool() -> None:
    """judged_fraction uses ctx.judged_spans_by_doc -- the union of every
    question's evidence spans in the suite -- not just this question's own
    gold spans, so a chunk judged by a *different* question still counts
    as judged here."""
    own_span = _span("d", 0, 10)
    other_question_span = _span("d", 200, 210)
    question = _question_with_gold([own_span])
    output = _output(
        [
            _chunk("d", "d#a", 1, 0, 10),  # judged: matches own_span
            _chunk("d", "d#b", 2, 200, 210),  # judged: matches other_question_span
            _chunk("d", "d#c", 3, 9999, 10010),  # not judged by anyone
        ]
    )
    ctx = ScoringContext(eval_ks=(3,), judged_spans_by_doc={"d": (own_span, other_question_span)})

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.judged_fraction@3"] == pytest.approx(2 / 3)


# -- retrieval.bpref ----------------------------------------------------------


def test_bpref_equals_one_when_no_nonrelevant_judgments() -> None:
    """Expected, correct M1 behaviour, not a bug: with no negative
    judgments available (the only source in M1), bpref is 1.0 whenever
    anything relevant was retrieved."""
    gold = [_span("d", 0, 10)]
    question = _question_with_gold(gold)
    output = _output([_chunk("d", "d#0000", 1, 0, 10)])
    ctx = ScoringContext(eval_ks=(1,))  # judged_nonrelevant_spans_by_question defaults to {}

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.bpref"] == pytest.approx(1.0)


def test_bpref_zero_when_nothing_relevant_retrieved() -> None:
    gold = [_span("d", 0, 10)]
    question = _question_with_gold(gold)
    output = _output([_chunk("d", "d#0000", 1, 999, 1010)])  # no overlap
    ctx = ScoringContext(eval_ks=(1,))

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.bpref"] == pytest.approx(0.0)


def test_bpref_hand_computed_with_nonrelevant_judgments() -> None:
    """Hand-computed bpref against nonrelevant judgments supplied directly
    in the test fixture (M1 has no production data source for these; this
    exercises the formula so a later milestone's pooling work is a
    drop-in).

    Gold (relevant): [0,10) and [500,510).
    Nonrelevant (hand-supplied): [1000,1010) and [1500,1510).

    Retrieved, ranked:
      1: [1000,1010) -- nonrelevant (N)
      2: [0,10)       -- relevant (R)
      3: [1500,1510) -- nonrelevant (N)
      4: [500,510)    -- relevant (R)

    R = {rank2, rank4}, N = {rank1, rank3}, |R| = |N| = 2, denom = 2.
      r=rank2: n ranked above (rank < 2) = {rank1} -> 1/2 -> term = 1 - 0.5 = 0.5
      r=rank4: n ranked above (rank < 4) = {rank1, rank3} -> 2/2 -> term = 1 - 1.0 = 0.0
    bpref = (0.5 + 0.0) / 2 = 0.25
    """
    gold = [_span("d", 0, 10), _span("d", 500, 510)]
    question = _question_with_gold(gold)
    nonrelevant = (_span("d", 1000, 1010), _span("d", 1500, 1510))
    output = _output(
        [
            _chunk("d", "d#a", 1, 1000, 1010),
            _chunk("d", "d#b", 2, 0, 10),
            _chunk("d", "d#c", 3, 1500, 1510),
            _chunk("d", "d#d", 4, 500, 510),
        ]
    )
    ctx = ScoringContext(eval_ks=(1,), judged_nonrelevant_spans_by_question={question.id: nonrelevant})

    by_metric = _by_metric(RetrievalScorer().score(question, output, ctx))
    assert by_metric["retrieval.bpref"] == pytest.approx(0.25)
