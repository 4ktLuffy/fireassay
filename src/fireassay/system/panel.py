"""The shared evaluation panel: every (ranker, chunking, top_k) config the
recall-detector tool and its tests evaluate against.

Why a panel needs more than one ranker (measured, not asserted): with only
`BM25System` varying `top_k`, a `k=3` retrieval set is always a subset of
that same chunking's `k=10` set -- 0 monotonicity violations were found
across 14,184 (item, chunking) pairs. A narrow config's successes are a
strict subset of a wide config's, so the widest `top_k` configs always
score highest and a weak config can never outperform a strong one from the
same chunking family. That made `discrimination_d < 0` nearly unproducible
and `split_half_reliability` sit at 0.29 -- 18 configs behaved like about
6, because `top_k` is a monotone ladder, not an independent axis.

The fix: rank by a genuinely different principle. `CoverageSystem` (term
coverage, ignoring frequency/idf) and `TfidfSystem` (idf-weighted cosine,
no BM25-style saturation or length normalisation) were chosen because
their scoring *structurally* differs from BM25's and from each other's --
not because they were found, after the fact, to produce ranking
inversions. If they turn out not to break the nesting, that is a result
to report, not a defect in this module to fix.

`build_panel` is the single place the recall tool, any future script, and
the tests all construct this panel from, so a config's identity is never
implicit in list position again: the old panel labelled its 18 systems
`sys00..sys17`, positionally, which is exactly why the `top_k` ladder went
unnoticed. Every system id here names its ranker, chunking and `top_k`
unambiguously, e.g. `bm25/512-32/k5`, `coverage/1024-128/k10`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, Doc, chunk_corpus
from fireassay.system.coverage import CoverageSystem
from fireassay.system.tfidf import TfidfSystem

#: The six (chunk_size, overlap) pairs the panel has used since M1 -- kept
#: exactly as they were so the new 54-config panel is a strict superset of
#: the old 18, and the old configs remain identifiable within it.
CHUNKINGS: tuple[tuple[int, int], ...] = (
    (256, 32),
    (256, 128),
    (512, 32),
    (512, 128),
    (1024, 32),
    (1024, 128),
)

#: The three `top_k` values the panel has used since M1, unchanged.
TOP_KS: tuple[int, ...] = (3, 5, 10)

#: `(system_id prefix, constructor)` pairs, in the order `build_panel`
#: iterates them. `bm25` first, so that with `TOP_KS` unchanged and
#: `CHUNKINGS` in this order, the panel's first 18 entries are exactly the
#: old 18-config panel (see `tests/test_panel.py`).
RANKERS: tuple[tuple[str, Callable[[Sequence[Chunk], int], System]], ...] = (
    ("bm25", BM25System),
    ("coverage", CoverageSystem),
    ("tfidf", TfidfSystem),
)


def build_chunks_by_config(
    docs: Sequence[Doc],
    chunkings: Sequence[tuple[int, int]] = CHUNKINGS,
) -> dict[tuple[int, int], list[Chunk]]:
    """Chunk `docs` once per `(size, overlap)` pair in `chunkings`, keyed
    by that pair -- shared across all three rankers at a given chunking
    (`build_panel` builds three systems per entry), rather than
    re-chunking the same corpus once per ranker."""
    by_config: dict[tuple[int, int], list[Chunk]] = {}
    for size, overlap in chunkings:
        by_config[size, overlap] = list(chunk_corpus(docs, size=size, overlap=overlap))
    return by_config


def build_panel(
    chunks_by_config: Mapping[tuple[int, int], Sequence[Chunk]],
    top_ks: Sequence[int] = TOP_KS,
) -> list[tuple[str, System]]:
    """Build the full panel: one `System` per `(ranker, chunking, top_k)`
    triple, i.e. `len(RANKERS) * len(chunks_by_config) * len(top_ks)`
    entries -- 54 with the defaults (3 rankers x 6 chunkings x 3 top_ks).

    `system_id` is `f"{ranker_name}/{size}-{overlap}/k{top_k}"`, e.g.
    `"bm25/512-32/k5"` -- unambiguous and human-legible, unlike the
    positional `sys00..sys17` ids the panel used before this change.

    `chunks_by_config` is caller-supplied (typically `build_chunks_by_config`'s
    result) rather than built from raw docs here, so the same chunked
    corpus is reused across all three rankers at a given chunking instead
    of being rebuilt once per ranker.
    """
    panel: list[tuple[str, System]] = []
    for ranker_name, ranker_ctor in RANKERS:
        for (size, overlap), chunks in chunks_by_config.items():
            for top_k in top_ks:
                system_id = f"{ranker_name}/{size}-{overlap}/k{top_k}"
                panel.append((system_id, ranker_ctor(chunks, top_k)))
    return panel
