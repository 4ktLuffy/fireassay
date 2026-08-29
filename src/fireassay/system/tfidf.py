"""A pure-Python, deterministic classic tf-idf cosine retriever.

Standard vector-space tf-idf: raw term counts, `idf(t) = log(N / df(t))`,
L2-normalised chunk vectors, ranked by cosine similarity with the query
vector. Chosen alongside `CoverageSystem` (see that module's docstring for
the panel-level "why") because its scoring differs structurally from
`BM25System` in two ways: it has no term-frequency saturation (BM25's
`k1`), so a chunk repeating a term keeps accumulating weight rather than
diminishing returns, and it has no explicit document-length-normalisation
parameter (BM25's `b`) -- a long chunk's inflated term counts are instead
tempered by the L2 norm dividing through the whole vector, a materially
different mechanism.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence

from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.system.corpus import Chunk
from fireassay.text import tokenize

# `tokenize` is re-exported here for the same reason `bm25.py` re-exports
# it: a stable import path for this module, while the implementation
# itself lives in exactly one place, `fireassay.text`. Do not redefine a
# tokenizing regex here.


class TfidfSystem:
    """Classic tf-idf cosine similarity over a fixed, in-memory chunk
    collection.

    Each chunk's vector is `tf(t, chunk) * idf(t)` for every term `t` in
    the chunk, then L2-normalised to unit length. The query vector is
    built the same way, `tf(t, query) * idf(t)`, but is *not*
    L2-normalised (normalising it would not change the ranking, since it
    is a shared denominator across every chunk's score for a given
    query). A term with `idf(t) == 0` (it appears in every chunk) or a
    query term absent from the corpus vocabulary (no `idf` entry at all,
    treated as `idf(t) == 0`) contributes zero weight -- it cannot help
    distinguish one chunk from another either way.

    Ties are broken by ascending chunk index -- the position of the chunk
    in the `chunks` sequence passed to the constructor -- via a stable
    sort on score alone, the same reasoning `CoverageSystem` documents
    (itself following `items.core`'s `kind="stable"` argsort): a stable
    sort preserves the original (index) order among equal-scoring
    elements by construction, so no explicit secondary sort key is
    needed.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        top_k: int = 5,
    ) -> None:
        self.name = "tfidf"
        # This system only ever retrieves; it never generates a
        # natural-language answer (see `System.generates_answers`'s
        # docstring, and `BM25System`'s identical choice).
        self.generates_answers = False
        self.top_k = top_k
        self._chunks: list[Chunk] = list(chunks)

        tokenized = [tokenize(c.text) for c in self._chunks]
        n_docs = len(self._chunks)
        doc_freq: dict[str, int] = {}
        for toks in tokenized:
            for term in set(toks):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        self._idf: dict[str, float] = {term: math.log(n_docs / freq) for term, freq in doc_freq.items()}

        self._chunk_vectors: list[dict[str, float]] = []
        for toks in tokenized:
            tf = Counter(toks)
            vec = {term: count * self._idf[term] for term, count in tf.items()}
            norm = math.sqrt(sum(w * w for w in vec.values()))
            if norm > 0.0:
                vec = {term: w / norm for term, w in vec.items()}
            self._chunk_vectors.append(vec)

    def _score(self, query_vec: dict[str, float], query_norm: float, doc_index: int) -> float:
        # A zero-norm query vector (empty query, or every query term
        # absent from the corpus vocabulary) makes cosine similarity
        # undefined (0/0); scoring it as 0.0 for every chunk keeps this
        # contract identical to BM25's for a query with no signal: a
        # deterministic, fully-tied ranking, not an exception.
        if query_norm == 0.0:
            return 0.0
        chunk_vec = self._chunk_vectors[doc_index]
        # `chunk_vec` is already L2-normalised to unit length, so cosine
        # similarity reduces to `dot(query_vec, chunk_vec) / ||query_vec||`.
        dot = sum(weight * chunk_vec.get(term, 0.0) for term, weight in query_vec.items())
        return dot / query_norm

    def answer(self, question: Question) -> SystemOutput:
        """Rank every chunk by tf-idf cosine similarity against
        `question.text` and return the top `top_k`.

        Returns retrieved chunks only: `answer=None`, `abstained=True`,
        `tokens_in=tokens_out=0` -- the same retrieval-only contract as
        `BM25System.answer` (see that docstring for the full reasoning).
        """
        start = time.perf_counter()
        query_terms = tokenize(question.text)
        tf_q = Counter(query_terms)
        query_vec = {term: count * self._idf.get(term, 0.0) for term, count in tf_q.items()}
        query_norm = math.sqrt(sum(w * w for w in query_vec.values()))
        scores = [self._score(query_vec, query_norm, i) for i in range(len(self._chunks))]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        top = ranked[: self.top_k]
        retrieved = tuple(
            RetrievedChunk(
                doc_id=self._chunks[i].doc_id,
                chunk_id=self._chunks[i].chunk_id,
                score=scores[i],
                rank=rank,
                char_start=self._chunks[i].char_start,
                char_end=self._chunks[i].char_end,
            )
            for rank, i in enumerate(top, start=1)
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SystemOutput(
            answer=None,
            abstained=True,
            retrieved=retrieved,
            latency_ms={"total": elapsed_ms},
            tokens_in=0,
            tokens_out=0,
        )
