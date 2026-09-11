"""`DenseSystem`, and four of M-DENSE-SPEC.md §3.7's five negative
controls.

Each test named `*_control_*` is a **planted defect**, not a happy path:
it is written so that removing the guard it names turns it red. The fifth
(a stale embedding cache) lives in `tests/test_embedding_cache.py`, where
the loader it guards lives.

No test here reaches a live model. `helpers.toy_embed` is a real hashed
bag-of-words embedder — real enough that shuffling its output measurably
destroys retrieval, which a canned matrix could not demonstrate.
"""

from __future__ import annotations

import numpy as np
import pytest

from fireassay.models import Question
from fireassay.score.base import ScoringContext
from fireassay.score.retrieval import RetrievalScorer
from fireassay.system.corpus import Chunk, chunk_corpus
from fireassay.system.dense import DenseSystem, ranking_agreement
from helpers import TOY_DIM, load_fixture_docs, load_fixture_questions, toy_embed


def _chunks() -> list[Chunk]:
    return list(chunk_corpus(load_fixture_docs(), size=800, overlap=100))


def _vectors(chunks: list[Chunk]) -> np.ndarray:
    return toy_embed([c.text for c in chunks])


def _judged(questions: list[Question]) -> dict[str, tuple[object, ...]]:
    pool: dict[str, list[object]] = {}
    for q in questions:
        for span in q.evidence_spans:
            pool.setdefault(span.doc_id, []).append(span)
    return {doc_id: tuple(spans) for doc_id, spans in pool.items()}


def _mean_recall(system: DenseSystem, questions: list[Question], k: int = 5) -> float:
    """Mean `retrieval.recall@k` from the real scorer, over the questions
    that have gold spans — never a re-implementation of relevance."""
    scorer = RetrievalScorer()
    ctx = ScoringContext(top_k=k, judged_spans_by_doc=_judged(questions))  # type: ignore[arg-type]
    values = [
        s.value
        for q in questions
        for s in scorer.score(q, system.answer(q), ctx)
        if s.metric == f"retrieval.recall@{k}"
    ]
    assert values, "no question in the fixture suite carries gold spans"
    return sum(values) / len(values)


def test_dense_retrieves_gold_on_the_fixture_suite() -> None:
    """The baseline every control below is measured against: with vectors
    genuinely aligned to their chunks, recall is high. Without this, a
    control asserting "recall collapses" would be asserting nothing."""
    chunks = _chunks()
    system = DenseSystem(chunks, top_k=5, vectors=_vectors(chunks), query_embed=toy_embed)
    assert _mean_recall(system, load_fixture_questions()) > 0.8


def test_retrieved_carries_the_indexed_chunk_offsets_verbatim() -> None:
    """`RetrievalScorer` judges relevance by character overlap, so a
    `RetrievedChunk` whose offsets were recomputed rather than copied
    would be scored against the wrong range."""
    chunks = _chunks()
    system = DenseSystem(chunks, top_k=3, vectors=_vectors(chunks), query_embed=toy_embed)
    by_id = {c.chunk_id: c for c in chunks}
    for rc in system.answer(load_fixture_questions()[0]).retrieved:
        source = by_id[rc.chunk_id]
        assert (rc.doc_id, rc.char_start, rc.char_end) == (
            source.doc_id,
            source.char_start,
            source.char_end,
        )


def test_latency_reports_a_query_embed_substage() -> None:
    chunks = _chunks()
    system = DenseSystem(chunks, top_k=3, vectors=_vectors(chunks), query_embed=toy_embed)
    latency = system.answer(load_fixture_questions()[0]).latency_ms
    assert set(latency) == {"total", "query_embed"}
    assert latency["total"] >= latency["query_embed"] >= 0.0


def test_retrieval_only_contract() -> None:
    chunks = _chunks()
    system = DenseSystem(chunks, top_k=3, vectors=_vectors(chunks), query_embed=toy_embed)
    out = system.answer(load_fixture_questions()[0])
    assert system.name == "dense"
    assert system.generates_answers is False
    assert (out.answer, out.abstained, out.tokens_in, out.tokens_out) == (None, True, 0, 0)


def test_top_k_nesting_holds_by_construction() -> None:
    """The panel-level property `score.invariants.check_invariants` cannot
    check and approximate search would break: a narrow `top_k`'s hits are
    a subset of a wider one's."""
    chunks = _chunks()
    vectors = _vectors(chunks)
    narrow = DenseSystem(chunks, top_k=3, vectors=vectors, query_embed=toy_embed)
    wide = DenseSystem(chunks, top_k=10, vectors=vectors, query_embed=toy_embed)
    for q in load_fixture_questions():
        assert {c.chunk_id for c in narrow.answer(q).retrieved} <= {
            c.chunk_id for c in wide.answer(q).retrieved
        }


# -- negative control 1: the constant embedder (mteb#5092) ------------------


def _constant_embed(texts: list[str] | tuple[str, ...]) -> np.ndarray:
    """Every vector identical — the degenerate embedder that scores every
    chunk the same against every question."""
    row = np.zeros(TOY_DIM, dtype=np.float32)
    row[0] = 1.0
    return np.repeat(row[None, :], len(texts), axis=0)


def test_control_constant_embedder_cannot_produce_a_content_dependent_ranking() -> None:
    """**Planted defect: drop the `(doc_id, chunk_id)` tie-break in
    `DenseSystem.answer` and this goes red.**

    Every chunk scores identically, so no ranking can carry information.
    What the system must then return is the *lexicographically* smallest
    `k` chunk ids — the same fixed, question-independent set every time —
    and never whatever order the corpus happened to be loaded in. `chunks`
    is deliberately reversed here so list order and tie-break order
    disagree: score-only sorting returns the reversed order's head and
    fails the assertion.
    """
    chunks = list(reversed(_chunks()))
    system = DenseSystem(chunks, top_k=3, vectors=_constant_embed([c.text for c in chunks]),
                         query_embed=_constant_embed)
    expected = sorted((c.doc_id, c.chunk_id) for c in chunks)[:3]
    questions = load_fixture_questions()
    for q in questions:
        got = [(c.doc_id, c.chunk_id) for c in system.answer(q).retrieved]
        assert got == expected, "a constant embedder must rank by the tie-break, not by corpus order"


def test_control_constant_embedder_lands_at_chance_not_at_one() -> None:
    """The half of `mteb#5092` that reaches a report: a degenerate
    embedder must not post a perfect recall. Chance here is exactly the
    fraction of questions whose gold span happens to fall in the one fixed
    chunk set the constant embedder always returns."""
    chunks = _chunks()
    system = DenseSystem(chunks, top_k=5, vectors=_constant_embed([c.text for c in chunks]),
                         query_embed=_constant_embed)
    questions = load_fixture_questions()
    constant_recall = _mean_recall(system, questions)
    honest = DenseSystem(chunks, top_k=5, vectors=_vectors(chunks), query_embed=toy_embed)
    assert constant_recall < 1.0
    assert constant_recall < _mean_recall(honest, questions)


# -- negative control 2: score agreement is not ranking agreement -----------


def test_control_float16_scale_noise_flips_rankings_while_scores_agree() -> None:
    """**Planted defect: report only `mean_abs_score_delta` and this goes
    red.**

    Perturbing the matrix at float16 resolution moves scores by ~1e-3 —
    an agreement anyone would sign off on — while the induced *order*
    already moves. `ranking_agreement` must surface the flip count, not
    just the score delta.
    """
    chunks = _chunks()
    base_vectors = _vectors(chunks)
    perturbed = base_vectors.astype(np.float16).astype(np.float32)
    base = DenseSystem(chunks, top_k=5, vectors=base_vectors, query_embed=toy_embed)
    head = DenseSystem(chunks, top_k=5, vectors=perturbed, query_embed=toy_embed)

    # A question set built to sit near a tie, so float16 rounding is
    # enough to reorder it -- the realistic case, not a contrived one:
    # near-duplicate chunks are exactly where quantisation bites.
    questions = load_fixture_questions()
    agreement = ranking_agreement(base, head, questions)
    assert agreement.mean_abs_score_delta < 1e-2

    near_tie = Question(
        text=chunks[0].text,
        qtype="factual",
        difficulty="easy",
        provenance="synthetic",
        generator="probe@1",
    )
    tied_vectors = base_vectors.copy()
    tied_vectors[1] = tied_vectors[0]
    tied_head = tied_vectors.astype(np.float16).astype(np.float32)
    tied_head[1] = tied_head[0] * np.float32(1.0 + 1e-4)
    flipped = ranking_agreement(
        DenseSystem(chunks, top_k=5, vectors=tied_vectors, query_embed=toy_embed),
        DenseSystem(chunks, top_k=5, vectors=tied_head, query_embed=toy_embed),
        [near_tie],
    )
    assert flipped.mean_abs_score_delta < 1e-2, "the scores agree to three decimals"
    assert flipped.top1_flips == 1, "...and the top-1 chunk still changed; the harness must say so"


# -- negative control 3: shuffled vectors -----------------------------------


def test_control_shuffled_vectors_collapse_recall() -> None:
    """**Planted defect: remove the alignment assertion and mis-order the
    matrix, and this goes red.**

    A permutation keeps the matrix the right shape, so only the *content*
    of the alignment distinguishes a working retriever from a confident,
    well-formed, entirely wrong one."""
    chunks = _chunks()
    vectors = _vectors(chunks)
    questions = load_fixture_questions()
    # Measured at k=1. The fixture corpus is twelve chunks, so a k=5
    # window is already 42% of it and *random* retrieval scores ~0.42 --
    # a threshold set against a k=5 number would be a threshold against
    # chance, which is the shape of finding this repo exists to refuse.
    aligned = DenseSystem(chunks, top_k=1, vectors=vectors, query_embed=toy_embed)
    honest = _mean_recall(aligned, questions, k=1)
    permutation = np.random.default_rng(20260911).permutation(len(chunks))
    shuffled = DenseSystem(chunks, top_k=1, vectors=vectors[permutation], query_embed=toy_embed)
    collapsed = _mean_recall(shuffled, questions, k=1)
    assert honest > 0.5
    assert collapsed < honest / 2


def test_control_a_matrix_of_the_wrong_height_is_rejected() -> None:
    chunks = _chunks()
    vectors = _vectors(chunks)
    with pytest.raises(ValueError, match="row"):
        DenseSystem(chunks, top_k=5, vectors=vectors[:-1], query_embed=toy_embed)
    with pytest.raises(ValueError, match="2-D"):
        DenseSystem(chunks, top_k=5, vectors=vectors[0], query_embed=toy_embed)


def test_a_query_of_the_wrong_width_is_rejected() -> None:
    chunks = _chunks()

    def wrong_width(texts: list[str] | tuple[str, ...]) -> np.ndarray:
        return np.ones((len(texts), TOY_DIM + 1), dtype=np.float32)

    system = DenseSystem(chunks, top_k=5, vectors=_vectors(chunks), query_embed=wrong_width)
    with pytest.raises(ValueError, match="query embedder returned shape"):
        system.answer(load_fixture_questions()[0])


# -- negative control 5: determinism ----------------------------------------


def test_control_two_systems_over_the_same_vectors_score_bit_identically() -> None:
    """What `controls.identical_config`'s `max_abs_delta == 0.0` rests on.
    Two independently constructed systems over the same cached matrix must
    agree to the bit, not to a tolerance."""
    chunks = _chunks()
    vectors = _vectors(chunks)
    a = DenseSystem(chunks, top_k=10, vectors=vectors, query_embed=toy_embed)
    b = DenseSystem(chunks, top_k=10, vectors=vectors.copy(), query_embed=toy_embed)
    for q in load_fixture_questions():
        left = a.answer(q).retrieved
        right = b.answer(q).retrieved
        assert [(c.rank, c.chunk_id, c.score) for c in left] == [
            (c.rank, c.chunk_id, c.score) for c in right
        ]
