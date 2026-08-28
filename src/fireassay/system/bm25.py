"""A pure-Python, deterministic BM25 retriever — the M1 system under test.

This is real retrieval, not a stub: it builds an inverted index over a
chunked corpus and ranks by the Okapi BM25 score. It needs no API key, no
network access, and no LLM call, which is the point of choosing it for
M1 — a stranger with nothing but this repo and the fixture corpus can
reproduce every retrieval number fireassay reports.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence

from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.system.corpus import Chunk
from fireassay.text import tokenize

# `tokenize` is re-exported here (rather than each caller reaching into
# `fireassay.text` directly) purely for a stable import path on this
# module; the implementation itself lives in exactly one place —
# `fireassay.text` — per that module's docstring. Do not redefine it here.


class BM25System:
    """Okapi BM25 over a fixed, in-memory chunk collection.

    `k1` (term-frequency saturation) and `b` (length normalisation) default
    to the standard Okapi BM25 values (`k1=1.5`, `b=0.75`) used by
    Elasticsearch/Lucene's default similarity, rather than values tuned for
    any particular corpus — M1's purpose is a deterministic, reproducible
    baseline, not a maximally-tuned retriever.

    idf uses the smoothed, non-negative variant
    ``idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))``
    (the same formula used by the widely-used `rank_bm25` reference
    implementation) rather than the original Robertson-Sparck Jones form,
    which can go negative for terms that appear in more than half the
    corpus and would then *penalise* a chunk for containing a very common
    query term — clearly wrong for a ranking function.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.name = "bm25"
        # This system only ever retrieves; it never generates a
        # natural-language answer (see `answer` below and
        # `System.generates_answers`'s docstring).
        self.generates_answers = False
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = list(chunks)

        tokenized = [tokenize(c.text) for c in self._chunks]
        self._doc_lens = [len(toks) for toks in tokenized]
        self._avg_len = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0
        self._term_freqs: list[Counter[str]] = [Counter(toks) for toks in tokenized]

        doc_freq: dict[str, int] = {}
        for toks in tokenized:
            for term in set(toks):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        n_docs = len(self._chunks)
        self._idf: dict[str, float] = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()
        }

    def _score(self, query_terms: Sequence[str], doc_index: int) -> float:
        tf = self._term_freqs[doc_index]
        dl = self._doc_lens[doc_index]
        avg_len = self._avg_len or 1.0
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf.get(term, 0.0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * dl / avg_len)
            score += idf * (numerator / denominator)
        return score

    def answer(self, question: Question) -> SystemOutput:
        """Rank every chunk by BM25 score against `question.text` and
        return the top `top_k`.

        Returns retrieved chunks only: `answer=None`, `abstained=True`,
        `tokens_in=tokens_out=0`. M1 measures retrieval, not generation —
        there is no LLM anywhere in this milestone, so there is nothing to
        report as a generated answer or token spend. Downstream, this means
        `AbstentionScorer` will score this system as *always* abstaining,
        which is correct: this system genuinely never generates an answer.

        Ranking ties are broken by `(doc_id, chunk_id)`, never by score
        alone: Python's `sorted` is stable, so sorting on score alone would
        leave tied chunks in whatever order they happen to sit in
        `self._chunks` -- which is corpus *load* order, not a property of
        relevance, and is not guaranteed reproducible across a corpus file
        re-ordered or re-generated on a different machine. A constant
        (all-tied) scorer with an unbroken tie order can silently produce
        internally-inconsistent numbers, e.g. `retrieval.mrr == 1.0` with
        `retrieval.ndcg@10 == 0.0`, if the "lucky" first-place chunk in
        insertion order happens not to be gold on one run but is on another.
        """
        start = time.perf_counter()
        query_terms = tokenize(question.text)
        scores = [self._score(query_terms, i) for i in range(len(self._chunks))]
        ranked = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], self._chunks[i].doc_id, self._chunks[i].chunk_id),
        )
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
