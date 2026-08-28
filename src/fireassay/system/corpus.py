"""Corpus loading and chunking for the BM25 system under test.

No dependency beyond stdlib: a corpus is just JSON Lines, and chunking is a
fixed-size character window. Keeping this deterministic and dependency-free
is part of the point of the M1 system-under-test — anyone can reproduce it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from fireassay.hashing import content_hash

_CORPUS_HASH_PREFIX = "fa.corpus.1"


@dataclass(frozen=True)
class Doc:
    """One document from the corpus, before chunking."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """One fixed-size character window of a document."""

    doc_id: str
    chunk_id: str
    text: str
    char_start: int
    char_end: int


def load_corpus(path: Path | str) -> list[Doc]:
    """Load a corpus from a JSONL file, one `{"doc_id", "title", "text"}`
    object per line. Blank lines are skipped."""
    docs: list[Doc] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Doc(doc_id=obj["doc_id"], title=obj.get("title", ""), text=obj["text"]))
    return docs


def chunk_corpus(docs: Iterable[Doc], size: int, overlap: int) -> Iterator[Chunk]:
    """Chunk each document into fixed-size, overlapping character windows.

    Windows are `[start, start + size)` (clipped to the document length),
    stepping by `size - overlap` each time, so consecutive chunks share
    `overlap` characters — the standard fixed-size-chunk RAG pattern, kept
    deliberately simple (no sentence/token boundary awareness) because M1
    measures whether retrieval finds the gold chunk, not chunk quality.

    `chunk_id` is `f"{doc_id}#{index:04d}"`, where `index` is the chunk's
    0-based position within its document — stable and human-legible, and
    matches the id scheme evidence spans reference.

    Raises `ValueError` if `size <= 0` or `overlap` is not in `[0, size)`
    (an overlap >= size would step by <= 0 characters and never terminate).
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap} size={size}")
    step = size - overlap
    for doc in docs:
        n = len(doc.text)
        if n == 0:
            continue
        index = 0
        start = 0
        while start < n:
            end = min(start + size, n)
            yield Chunk(
                doc_id=doc.doc_id,
                chunk_id=f"{doc.doc_id}#{index:04d}",
                text=doc.text[start:end],
                char_start=start,
                char_end=end,
            )
            index += 1
            if end == n:
                break
            start += step


def corpus_hash(docs: Iterable[Doc]) -> str:
    """Content-address the raw corpus, for `run.env_json["affects_results"]`.

    `fa.corpus.1` over `"\\n".join(sorted(f"{doc_id}:{sha256(doc.text)}" for each
    doc))`. Wired into `runner.run_matrix` (via a caller-supplied,
    zero-argument `corpus_hash_factory`) so that if the source corpus is
    edited underneath a suite — the gov.uk content this is meant to run
    against is revised over time — `integrity.assert_comparable`'s
    existing `ENV_MISMATCH` check catches it, rather than two runs
    silently being compared as if against the same evidence.

    **Resolved tension (was flagged for review, now decided):** an earlier
    version of this hash was computed over *chunk* boundaries and
    per-chunk text, which made it sensitive to chunking-strategy changes
    as well as genuine content drift — two configs differing only in
    chunk size got different `corpus_hash` values and tripped
    `ENV_MISMATCH` between them, directly working against
    `score.retrieval.RetrievalScorer`'s character-overlap-based relevance,
    whose entire purpose is making recall/nDCG/MRR comparable *across*
    chunking strategies (M1-SPEC.md correction 1). The resolution: chunking
    is a property of the **config**, not of the **corpus** — and it is
    already captured in `config_hash` (chunk size/overlap are config axis
    values). `corpus_hash` therefore hashes raw per-document text only, is
    computed once per corpus rather than once per config, and no longer
    varies with chunking at all. `runner.run_matrix`'s
    `corpus_hash_factory` takes no arguments for the same reason: the
    corpus's identity does not depend on which config is being run.
    """
    parts = sorted(f"{doc.doc_id}:{hashlib.sha256(doc.text.encode('utf-8')).hexdigest()}" for doc in docs)
    return content_hash(_CORPUS_HASH_PREFIX, "\n".join(parts))
