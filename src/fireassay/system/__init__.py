"""Systems under test.

`base.System` is the protocol every system implements. `bm25.BM25System` is
the M1 system under test: a real, deterministic, pure-Python retriever with
no API key and no network call, so a stranger with no credentials can
reproduce every number fireassay reports about it. `coverage.CoverageSystem`
and `tfidf.TfidfSystem` rank by principles that structurally differ from
BM25's and from each other's, giving the evaluation panel (`panel.build_panel`)
a genuine second and third axis of retrieval strength.

`rag.RagSystem` is the first system for which `generates_answers` is `True`
(see that module's docstring): it wraps any retriever above with an
injected `rag.Generate` call to produce an actual answer. It is additive --
`panel.build_panel`'s 54-config retrieval panel is unchanged and does not
use it. `adapters.ollama_generate` is the thin factory that wires a real
`OllamaClient` into a `rag.Generate`; `rag.py` itself imports no LLM client.
"""

from fireassay.system.adapters import ollama_generate
from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, chunk_corpus, corpus_hash, load_corpus
from fireassay.system.coverage import CoverageSystem
from fireassay.system.panel import CHUNKINGS, RANKERS, TOP_KS, build_chunks_by_config, build_panel
from fireassay.system.rag import Generate, RagSystem, build_prompt
from fireassay.system.tfidf import TfidfSystem

__all__ = [
    "BM25System",
    "CHUNKINGS",
    "Chunk",
    "CoverageSystem",
    "Generate",
    "RANKERS",
    "RagSystem",
    "System",
    "TOP_KS",
    "TfidfSystem",
    "build_chunks_by_config",
    "build_panel",
    "build_prompt",
    "chunk_corpus",
    "corpus_hash",
    "load_corpus",
    "ollama_generate",
]
