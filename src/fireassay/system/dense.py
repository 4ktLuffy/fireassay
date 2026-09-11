"""`DenseSystem` — exact dense retrieval over a cached embedding matrix,
the fourth `System` alongside `bm25`, `tfidf` and `coverage`.

**Exact, not approximate, and that is an engineering choice with a
measurement reason.** At 1,181 documents (12,513 chunks at 1024/128) a
full dot product is a single BLAS call of a few milliseconds. An ANN index
would add a recall loss that then has to be measured and reported
separately — a second source of error inside the very comparison this
milestone exists to make. It would also break the one panel-level property
`score.invariants` cannot check: a narrower `top_k`'s hits being a subset
of a wider one's holds by construction for exact search and fails for
approximate search.

**The tie-break is the bug guard, not a formality.** Ranking is by
`(-score, doc_id, chunk_id)` — byte-identical to `BM25System`'s — so a
*constant* embedder, whose every chunk scores the same, produces a fixed,
content-independent ranking rather than whatever order `np.argpartition`
happened to emit. Without it, a degenerate embedder can post a perfect
recall on a suite whose gold chunk happens to sit early in corpus load
order (the `mteb#5092` shape). `tests/test_dense.py`'s constant-embedder
control is that bug, planted, and must stay red without this sort key.

**float32 on disk, float64 in the reduction.** The cache stores float32
(half the bytes, and the precision an embedding actually carries); the
matrix is widened once in `__init__` and every dot product is accumulated
in float64. A float32 reduction over a matrix this size has been observed
returning a cosine of 1.0106, which is not a cosine.

Vectors arrive already loaded and aligned index-for-index with `chunks`
(see `system.embedding_cache`); the constructor asserts that alignment
rather than trusting it, because a permuted matrix produces confident,
well-formed, entirely wrong rankings — `tests/test_dense.py`'s
shuffled-vector control.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.system.base import System
from fireassay.system.corpus import Chunk
from fireassay.system.embedding import Embed


class DenseSystem:
    """Rank a fixed chunk collection by dot product against the question's
    embedding. Retrieval-only, like every other `System` in this repo:
    `answer=None`, `abstained=True`, `tokens_in=tokens_out=0`.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        top_k: int = 5,
        *,
        vectors: np.ndarray,
        query_embed: Embed,
    ) -> None:
        self.name = "dense"
        # Retrieval only; it never generates a natural-language answer —
        # see `System.generates_answers`'s docstring and `BM25System`'s
        # identical choice.
        self.generates_answers = False
        self.top_k = top_k
        self._chunks: list[Chunk] = list(chunks)
        matrix = np.asarray(vectors)
        if matrix.ndim != 2:
            raise ValueError(f"DenseSystem: vectors must be a 2-D (n_chunks, d) matrix, got {matrix.shape}")
        if matrix.shape[0] != len(self._chunks):
            raise ValueError(
                f"DenseSystem: vectors has {matrix.shape[0]} row(s) but {len(self._chunks)} chunk(s) "
                "were given; row i must be the embedding of chunks[i]. A matrix that is merely the "
                "right *size* for a different chunking ranks confidently and wrongly, which is why "
                "this is an error rather than a warning."
            )
        self._vectors = np.ascontiguousarray(matrix, dtype=np.float64)
        self._query_embed = query_embed

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    def answer(self, question: Question) -> SystemOutput:
        """Embed `question.text`, score every chunk by dot product, and
        return the top `top_k`.

        Ranking ties are broken by `(doc_id, chunk_id)`, never by score
        alone — the same rule, for the same reason, as
        `BM25System.answer`. The candidate set handed to that sort is
        every index whose score is at least the `k`-th largest, not the
        `np.argpartition` selection itself: partition order among equal
        scores is unspecified, so slicing it directly would reintroduce
        exactly the arbitrary, corpus-order-dependent ranking the sort key
        exists to remove.

        `latency_ms` carries `total` and the `query_embed` sub-stage, so a
        report can separate "the model was slow" from "the search was
        slow" — for a dense retriever those are different systems' costs
        and only one of them is ours.
        """
        start = time.perf_counter()
        embedded = np.asarray(self._query_embed([question.text]), dtype=np.float64)
        embed_ms = (time.perf_counter() - start) * 1000
        if embedded.shape != (1, self._vectors.shape[1]):
            raise ValueError(
                f"DenseSystem: query embedder returned shape {embedded.shape}, expected "
                f"(1, {self._vectors.shape[1]})"
            )
        query = embedded[0]

        n = len(self._chunks)
        k = min(self.top_k, n)
        if k <= 0:
            top: list[int] = []
            scores = np.zeros(n, dtype=np.float64)
        else:
            scores = self._vectors @ query
            selected = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
            cutoff = float(scores[selected].min())
            candidates = np.flatnonzero(scores >= cutoff)
            top = sorted(
                (int(i) for i in candidates),
                key=lambda i: (-scores[i], self._chunks[i].doc_id, self._chunks[i].chunk_id),
            )[:k]

        retrieved = tuple(
            RetrievedChunk(
                doc_id=self._chunks[i].doc_id,
                chunk_id=self._chunks[i].chunk_id,
                score=float(scores[i]),
                rank=rank,
                char_start=self._chunks[i].char_start,
                char_end=self._chunks[i].char_end,
            )
            for rank, i in enumerate(top, start=1)
        )
        total_ms = (time.perf_counter() - start) * 1000
        return SystemOutput(
            answer=None,
            abstained=True,
            retrieved=retrieved,
            latency_ms={"total": total_ms, "query_embed": embed_ms},
            tokens_in=0,
            tokens_out=0,
        )


@dataclass(frozen=True)
class RankingAgreement:
    """How two retrievers differ — **in ranks, not only in scores.**

    A mean absolute score delta is the number it is tempting to report
    when comparing a perturbed embedder (a different quantisation, a
    float16 store, a re-pulled model) against its baseline, and it is the
    number that hides the failure: score deltas at float16 scale are
    around 1e-3 and look like agreement, while the *order* those scores
    induce has already moved and every downstream metric moves with it.
    `top1_flips` and `topk_set_changes` are what a caller must look at;
    `mean_abs_score_delta` is reported beside them precisely so the
    contrast is visible rather than so it can be quoted alone.

    `tests/test_dense.py::test_float16_scale_noise_flips_rankings_while_scores_agree`
    is that contrast, planted: it asserts the score delta is tiny *and*
    that the flip count is nonzero, and goes red if this type ever stops
    reporting the second.
    """

    n_questions: int
    mean_abs_score_delta: float
    max_abs_score_delta: float
    top1_flips: int
    topk_set_changes: int


def ranking_agreement(base: System, head: System, questions: Sequence[Question]) -> RankingAgreement:
    """Compare two retrievers over `questions`, rank-first.

    Score deltas are compared rank-position by rank-position (the score at
    rank 1 against the score at rank 1), which is what makes "the scores
    barely moved" and "the top-1 chunk changed" separately observable on
    the same pair of runs.
    """
    if not questions:
        raise ValueError("ranking_agreement: questions must not be empty")
    deltas: list[float] = []
    top1_flips = 0
    set_changes = 0
    for question in questions:
        a = base.answer(question).retrieved
        b = head.answer(question).retrieved
        for x, y in zip(sorted(a, key=lambda c: c.rank), sorted(b, key=lambda c: c.rank), strict=False):
            deltas.append(abs(x.score - y.score))
        first_a = min(a, key=lambda c: c.rank).chunk_id if a else None
        first_b = min(b, key=lambda c: c.rank).chunk_id if b else None
        if first_a != first_b:
            top1_flips += 1
        if {c.chunk_id for c in a} != {c.chunk_id for c in b}:
            set_changes += 1
    return RankingAgreement(
        n_questions=len(questions),
        mean_abs_score_delta=(sum(deltas) / len(deltas)) if deltas else 0.0,
        max_abs_score_delta=max(deltas) if deltas else 0.0,
        top1_flips=top1_flips,
        topk_set_changes=set_changes,
    )
