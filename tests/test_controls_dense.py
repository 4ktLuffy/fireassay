"""The four deterministic controls, run against `DenseSystem` rather than
BM25 — offline.

M-DENSE-SPEC.md §2 calls `identical_config` "at risk" for a dense
retriever, because it asserts `max_abs_delta == 0.0` across every
non-latency metric and a retriever that calls a live model per build
cannot satisfy that. This module is where that risk is discharged in CI
rather than only in a one-off report: the vectors come from a real
`EmbeddingCache` on disk and the query vectors from a real `QueryCache`,
exactly as they do in a live run, with `helpers.toy_embed` standing in for
Ollama so no test touches the network (M3-SPEC.md §7).

`corpus_ablation` is the interesting one: it drops documents and rebuilds,
so the ablated corpus has a different `corpus_hash` and no cache of its
own. The factory here resolves its vectors the same way `cli._build_dense`
does — `embedding_cache.resolve_vectors`, which selects the surviving
chunks' rows out of the full corpus's matrix. That is exact, because
`chunk_corpus` windows each document independently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from fireassay.controls.base import ControlContext, ControlOutcome
from fireassay.controls.corpus_ablation import CorpusAblationControl
from fireassay.controls.identical_config import IdenticalConfigControl
from fireassay.controls.no_retrieval import NoRetrievalControl
from fireassay.controls.shuffled_gold import ShuffledGoldControl
from fireassay.llm.ollama import ModelRef
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.corpus import Doc, chunk_corpus, corpus_hash
from fireassay.system.dense import DenseSystem
from fireassay.system.embedding_cache import CacheKey, EmbeddingCache, QueryCache, resolve_vectors
from helpers import load_fixture_docs, make_control_context, toy_embed

MODEL = ModelRef(name="qwen3-embedding:0.6b", digest="ac6da0dfba84a81f")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _key(docs: Sequence[Doc]) -> CacheKey:
    chunks = list(chunk_corpus(docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
    return CacheKey(
        model_name=MODEL.name,
        model_digest=MODEL.digest,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        corpus_hash=corpus_hash(docs),
        chunk_ids=tuple(c.chunk_id for c in chunks),
    )


def dense_factory(root: Path) -> Callable[[Mapping[str, object], Sequence[Doc]], System]:
    """The same construction `cli._build_dense` performs, over `root`."""

    def factory(config: Mapping[str, object], docs: Sequence[Doc]) -> System:
        chunks = list(chunk_corpus(docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
        vectors, _provenance = resolve_vectors(_key(docs), root=root)
        query_embed = QueryCache(root / "queries", MODEL).wrap(toy_embed)
        top_k = int(config.get("top_k", 5))  # type: ignore[arg-type]
        return DenseSystem(chunks, top_k=top_k, vectors=vectors, query_embed=query_embed)

    return factory


@pytest.fixture
def dense_ctx(store: Store, tmp_path: Path) -> ControlContext:
    docs = load_fixture_docs()
    key = _key(docs)
    texts = [c.text for c in chunk_corpus(docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)]
    EmbeddingCache.open(key, root=tmp_path).build(texts, toy_embed)
    return make_control_context(store, system_factory=dense_factory(tmp_path))


def _assert_passed(outcome: ControlOutcome) -> None:
    assert outcome.status == "PASSED", outcome.detail
    assert outcome.twin_ok, outcome.detail
    assert all(outcome.cause_assertions.values()), outcome.cause_assertions


def test_no_retrieval_passes_against_dense(dense_ctx: ControlContext) -> None:
    outcome = NoRetrievalControl().run(dense_ctx)
    _assert_passed(outcome)
    assert outcome.observed["recall_at_k"] == 0.0


def test_shuffled_gold_passes_against_dense(dense_ctx: ControlContext) -> None:
    _assert_passed(ShuffledGoldControl().run(dense_ctx))


def test_corpus_ablation_passes_against_dense(dense_ctx: ControlContext) -> None:
    """Exercises the derived-subset path: the ablated corpus has no cache
    of its own and its vectors are selected out of the full matrix."""
    outcome = CorpusAblationControl().run(dense_ctx)
    _assert_passed(outcome)
    assert outcome.observed["recall_at_k_drop"] > 0.0


def test_identical_config_passes_at_exactly_zero_delta(dense_ctx: ControlContext) -> None:
    """The control M-DENSE-SPEC.md §2 flagged as at risk. `0.0`, not a
    tolerance: two independently built dense systems reading the same
    cached vectors must agree to the bit."""
    outcome = IdenticalConfigControl().run(dense_ctx)
    _assert_passed(outcome)
    assert outcome.observed["max_abs_delta"] == 0.0


def test_identical_config_would_fail_on_a_non_deterministic_embedder(
    store: Store, tmp_path: Path
) -> None:
    """**Planted defect: a query embedder that does not come from a
    cache.** This is the failure mode the cache exists to prevent, shown
    failing, so the passing case above is evidence rather than a
    coincidence of a suite too small to notice.

    The perturbation is applied to the *direction*, not the magnitude. A
    query vector that merely drifts in scale leaves every score scaled by
    the same factor and therefore every ranking — and so every
    `retrieval.*` metric — untouched; a control that fired on that would
    be reporting arithmetic, not nondeterminism. Only a change of
    direction can move a rank, which is the same distinction
    `dense.ranking_agreement` exists to keep visible.
    """
    import numpy as np

    calls = iter(range(10_000))

    def wobbling_embed(texts: Sequence[str]) -> np.ndarray:
        base = np.asarray(toy_embed(texts), dtype=np.float64)
        rng = np.random.default_rng(next(calls))
        noisy = base + rng.normal(scale=0.25, size=base.shape)
        return (noisy / np.linalg.norm(noisy, axis=1, keepdims=True)).astype(np.float32)

    def factory(config: Mapping[str, object], factory_docs: Sequence[Doc]) -> System:
        factory_chunks = list(chunk_corpus(factory_docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
        top_k = int(config.get("top_k", 5))  # type: ignore[arg-type]
        return DenseSystem(
            factory_chunks,
            top_k=top_k,
            vectors=toy_embed([c.text for c in factory_chunks]),
            query_embed=wobbling_embed,
        )

    ctx = make_control_context(store, system_factory=factory)
    outcome = IdenticalConfigControl().run(ctx)
    assert outcome.status == "FAILED"
    assert outcome.observed["max_abs_delta"] > 0.0
