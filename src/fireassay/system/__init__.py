"""Systems under test.

`base.System` is the protocol every system implements. `bm25.BM25System` is
the M1 system under test: a real, deterministic, pure-Python retriever with
no API key and no network call, so a stranger with no credentials can
reproduce every number fireassay reports about it.
"""

from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, chunk_corpus, corpus_hash, load_corpus

__all__ = ["BM25System", "Chunk", "System", "chunk_corpus", "corpus_hash", "load_corpus"]
