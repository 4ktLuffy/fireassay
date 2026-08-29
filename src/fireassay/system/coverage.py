"""A pure-Python, deterministic term-coverage retriever.

Ranks chunks by the fraction of *distinct* query terms they contain,
ignoring term frequency and idf entirely. That is what makes it disagree
with `BM25System` by construction: BM25 rewards a chunk that repeats one
rare query term, while this system rewards a chunk that mentions many
different query words even once each. Added alongside `TfidfSystem`
(see that module) to give the evaluation panel a genuine second axis of
retrieval strength, rather than one where every config is a strict subset
of a wider `top_k` from the same chunking (see `system.panel`'s module
docstring for the full "why").
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.system.corpus import Chunk
from fireassay.text import tokenize

# `tokenize` is re-exported here for the same reason `bm25.py` re-exports
# it: a stable import path for this module, while the implementation
# itself lives in exactly one place, `fireassay.text`. Do not redefine a
# tokenizing regex here.


class CoverageSystem:
    """Ranks each chunk by `|query_terms ∩ chunk_terms| / |query_terms|`,
    where both sets are *distinct* tokens (a `set`, not a `Counter`) --
    term frequency and idf play no role at all.

    Ties are broken by ascending chunk index -- the position of the chunk
    in the `chunks` sequence passed to the constructor -- via a stable
    sort on score alone. The same reasoning `items.core` documents for its
    `kind="stable"` argsort applies here: a stable sort preserves the
    original (index) order among equal-scoring elements by construction,
    so ranking never depends on dict or sort incidentals, and no explicit
    secondary sort key is needed.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        top_k: int = 5,
    ) -> None:
        self.name = "coverage"
        # This system only ever retrieves; it never generates a
        # natural-language answer (see `System.generates_answers`'s
        # docstring, and `BM25System`'s identical choice).
        self.generates_answers = False
        self.top_k = top_k
        self._chunks: list[Chunk] = list(chunks)
        self._chunk_terms: list[set[str]] = [set(tokenize(c.text)) for c in self._chunks]

    def _score(self, query_terms: set[str], doc_index: int) -> float:
        # An empty query shares no terms with anything by definition;
        # scoring it as 0.0 for every chunk (rather than raising a
        # ZeroDivisionError) keeps `answer()`'s contract identical to
        # BM25's for a query with no signal: a deterministic, fully-tied
        # ranking, not an exception.
        if not query_terms:
            return 0.0
        matched = len(query_terms & self._chunk_terms[doc_index])
        return matched / len(query_terms)

    def answer(self, question: Question) -> SystemOutput:
        """Rank every chunk by term-coverage score against `question.text`
        and return the top `top_k`.

        Returns retrieved chunks only: `answer=None`, `abstained=True`,
        `tokens_in=tokens_out=0` -- the same retrieval-only contract as
        `BM25System.answer` (see that docstring for the full reasoning).
        """
        start = time.perf_counter()
        query_terms = set(tokenize(question.text))
        scores = [self._score(query_terms, i) for i in range(len(self._chunks))]
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
