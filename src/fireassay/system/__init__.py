"""Systems under test.

`base.System` is the protocol every system implements. `bm25.BM25System` is
the M1 system under test: a real, deterministic, pure-Python retriever with
no API key and no network call, so a stranger with no credentials can
reproduce every number fireassay reports about it. `coverage.CoverageSystem`
and `tfidf.TfidfSystem` rank by principles that structurally differ from
BM25's and from each other's, giving the evaluation panel (`panel.build_panel`)
a genuine second and third axis of retrieval strength.
"""

from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, chunk_corpus, corpus_hash, load_corpus
from fireassay.system.coverage import CoverageSystem
from fireassay.system.panel import CHUNKINGS, RANKERS, TOP_KS, build_chunks_by_config, build_panel
from fireassay.system.tfidf import TfidfSystem

__all__ = [
    "BM25System",
    "CHUNKINGS",
    "Chunk",
    "CoverageSystem",
    "RANKERS",
    "System",
    "TOP_KS",
    "TfidfSystem",
    "build_chunks_by_config",
    "build_panel",
    "chunk_corpus",
    "corpus_hash",
    "load_corpus",
]
