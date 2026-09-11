"""`system.embedding_cache`: the resumable shard store, the query cache,
and M-DENSE-SPEC.md §3.7's fifth negative control — **a stale cache must
raise, not be silently reused or silently rebuilt.**

The cache is the mechanism that makes a dense retriever bit-deterministic
and therefore a legitimate subject for `controls.identical_config`. A test
that only checked "the cache round-trips" would miss the half that
matters: what it does when the cache does *not* belong to this run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fireassay.llm.ollama import ModelRef
from fireassay.system.corpus import chunk_corpus, corpus_hash
from fireassay.system.embedding_cache import (
    CacheKey,
    EmbeddingCache,
    QueryCache,
    StaleCacheError,
    resolve_vectors,
)
from helpers import load_fixture_docs, toy_embed

MODEL = ModelRef(name="qwen3-embedding:0.6b", digest="ac6da0dfba84a81f")
OTHER_MODEL = ModelRef(name="qwen3-embedding:0.6b", digest="0000000000000000")


def _key(size: int = 800, overlap: int = 100) -> tuple[CacheKey, list[str]]:
    docs = load_fixture_docs()
    chunks = list(chunk_corpus(docs, size=size, overlap=overlap))
    key = CacheKey(
        model_name=MODEL.name,
        model_digest=MODEL.digest,
        chunk_size=size,
        chunk_overlap=overlap,
        corpus_hash=corpus_hash(docs),
        chunk_ids=tuple(c.chunk_id for c in chunks),
    )
    return key, [c.text for c in chunks]


def test_build_then_load_round_trips_in_chunk_order(tmp_path: Path) -> None:
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    metadata = cache.build(texts, toy_embed, shard_size=5)
    assert metadata.n_chunks == len(texts)
    loaded = cache.load()
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, toy_embed(texts))


def test_shards_are_written_at_the_requested_size(tmp_path: Path) -> None:
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)
    names = sorted(p.name for p in cache.shards_path.glob("*.npy"))
    assert names == ["0-5.npy", "10-12.npy", "5-10.npy"]
    assert not list(cache.shards_path.glob("*.tmp.npy")), "no temp file may survive a completed build"


def test_build_is_resumable_and_re_embeds_only_the_missing_shard(tmp_path: Path) -> None:
    """A kill costs one shard, not an hour — the reason shards exist."""
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)
    (cache.shards_path / "5-10.npy").unlink()

    embedded: list[int] = []

    def counting(batch: Sequence[str]) -> np.ndarray:
        embedded.append(len(batch))
        return toy_embed(batch)

    cache.build(texts, counting, shard_size=5)
    assert embedded == [5], "only the deleted shard may be re-embedded"
    assert np.array_equal(cache.load(), toy_embed(texts))


def test_progress_reports_work_not_files(tmp_path: Path) -> None:
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    seen: list[tuple[int, int]] = []
    cache.build(texts, toy_embed, shard_size=5, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(5, 12), (10, 12), (12, 12)]
    seen.clear()
    cache.build(texts, toy_embed, shard_size=5, progress=lambda d, t: seen.append((d, t)))
    assert seen == [], "a fully cached rebuild does no work and must report none"


# -- negative control 4: a stale cache is refused ---------------------------


def test_control_a_cache_built_at_a_different_chunking_is_refused(tmp_path: Path) -> None:
    """**Planted defect: drop `EmbeddingCache._check` and this goes red.**

    The realistic shape of the bug: the chunking moves, the cache
    directory is still there, and every question gets ranked against
    another chunking's vectors. The numbers that come out are well-formed
    and wrong, which is the only kind worth a guard.
    """
    key, texts = _key(size=800, overlap=100)
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)

    other_key, _ = _key(size=400, overlap=50)
    # Point a *different* chunking's key at the same directory -- what a
    # shared or hand-copied cache dir does in practice.
    stale = EmbeddingCache(cache.path, other_key)
    with pytest.raises(StaleCacheError, match="chunk_size"):
        stale.load()
    with pytest.raises(StaleCacheError, match="chunk_size"):
        stale.build(texts, toy_embed, shard_size=5)


def test_control_a_cache_built_by_a_different_model_is_refused(tmp_path: Path) -> None:
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)
    other = replace(key, model_digest=OTHER_MODEL.digest)
    with pytest.raises(StaleCacheError, match="model_digest"):
        EmbeddingCache(cache.path, other).load()


def test_control_a_cache_over_different_chunk_ids_is_refused(tmp_path: Path) -> None:
    """The count can match while the *identity* does not — a corpus
    re-ordered, or one document swapped for another of the same shape."""
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)
    other = replace(key, chunk_ids=("nope", *key.chunk_ids[1:]))
    with pytest.raises(StaleCacheError, match="chunk_ids_sha256"):
        EmbeddingCache(cache.path, other).load()


def test_a_missing_cache_raises_rather_than_re_embedding_inside_a_run(tmp_path: Path) -> None:
    key, _texts = _key()
    with pytest.raises(StaleCacheError, match="no embedding cache"):
        EmbeddingCache.open(key, root=tmp_path).load()


def test_a_missing_shard_raises_rather_than_loading_a_short_matrix(tmp_path: Path) -> None:
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed)
    next(iter(cache.shards_path.glob("*.npy"))).unlink()
    with pytest.raises(StaleCacheError, match="missing or truncated"):
        cache.load()


def test_a_truncated_shard_raises(tmp_path: Path) -> None:
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)
    shard = cache.shards_path / "0-5.npy"
    np.save(shard, np.load(shard)[:2])
    with pytest.raises(StaleCacheError, match="expected 5 row"):
        cache.load()


def test_build_rejects_a_text_count_that_disagrees_with_its_key(tmp_path: Path) -> None:
    key, texts = _key()
    with pytest.raises(ValueError, match="names 12 chunk"):
        EmbeddingCache.open(key, root=tmp_path).build(texts[:-1], toy_embed)


def test_the_directory_name_is_content_addressed_on_the_four_key_fields() -> None:
    key, _ = _key(size=800, overlap=100)
    same = replace(key, model_name="a-different-tag-same-weights")
    different = replace(key, chunk_overlap=99)
    assert key.digest() == same.digest(), "the tag is not the identity; the digest is"
    assert key.digest() != different.digest()


# -- the query cache: what makes two runs byte-identical --------------------


def test_query_cache_serves_the_second_call_from_disk(tmp_path: Path) -> None:
    calls: list[int] = []

    def counting(texts: Sequence[str]) -> np.ndarray:
        calls.append(len(texts))
        return toy_embed(texts)

    first = QueryCache(tmp_path, MODEL).wrap(counting)
    a = first(["how do I reset my password?"])
    assert calls == [1]

    # A *fresh* cache object over the same directory, as a second process
    # (or a second `system_factory` call inside `identical_config`) has.
    second = QueryCache(tmp_path, MODEL).wrap(counting)
    b = second(["how do I reset my password?"])
    assert calls == [1], "a cached query must never reach the model again"
    assert np.array_equal(a, b)


def test_query_cache_embeds_only_the_misses_in_a_mixed_batch(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def counting(texts: Sequence[str]) -> np.ndarray:
        calls.append(list(texts))
        return toy_embed(texts)

    cached = QueryCache(tmp_path, MODEL).wrap(counting)
    cached(["one"])
    out = cached(["one", "two", "three"])
    assert calls[1] == ["two", "three"]
    assert out.shape == (3, toy_embed(["one"]).shape[1])
    assert np.array_equal(out[0], cached(["one"])[0])


def test_query_cache_is_partitioned_by_model_digest(tmp_path: Path) -> None:
    """Two models' vectors for the same text must never collide — the
    same rule `llm.cache.ResponseCache` follows for generations."""
    a = QueryCache(tmp_path, MODEL)
    b = QueryCache(tmp_path, OTHER_MODEL)
    assert a.path != b.path


# -- deriving a subset corpus's matrix (what corpus_ablation needs) ---------


def _ablated_key(dropped: set[str], size: int = 800, overlap: int = 100) -> tuple[CacheKey, list[str]]:
    docs = [d for d in load_fixture_docs() if d.doc_id not in dropped]
    chunks = list(chunk_corpus(docs, size=size, overlap=overlap))
    key = CacheKey(
        model_name=MODEL.name,
        model_digest=MODEL.digest,
        chunk_size=size,
        chunk_overlap=overlap,
        corpus_hash=corpus_hash(docs),
        chunk_ids=tuple(c.chunk_id for c in chunks),
    )
    return key, [c.text for c in chunks]


def test_derived_subset_equals_a_freshly_built_subset(tmp_path: Path) -> None:
    """`controls.corpus_ablation` drops documents and rebuilds, so the
    ablated corpus has a different `corpus_hash` and no cache of its own.
    Selecting its rows out of the full corpus's matrix must be **exact**,
    not approximately right: `chunk_corpus` windows each document
    independently, so a surviving chunk is byte-identical either way.

    Checked against a real rebuild rather than asserted.
    """
    full_key, full_texts = _key()
    EmbeddingCache.open(full_key, root=tmp_path).build(full_texts, toy_embed, shard_size=5)

    dropped = {"doc03", "doc07"}
    ablated_key, ablated_texts = _ablated_key(dropped)
    derived, provenance = resolve_vectors(ablated_key, root=tmp_path)
    assert provenance.startswith("derived from ")

    fresh = EmbeddingCache.open(ablated_key, root=tmp_path / "fresh")
    fresh.build(ablated_texts, toy_embed, shard_size=5)
    assert np.array_equal(derived, fresh.load())


def test_an_exact_cache_is_preferred_over_a_derivation(tmp_path: Path) -> None:
    key, texts = _key()
    EmbeddingCache.open(key, root=tmp_path).build(texts, toy_embed, shard_size=5)
    vectors, provenance = resolve_vectors(key, root=tmp_path)
    assert provenance == "exact"
    assert np.array_equal(vectors, toy_embed(texts))


def test_a_superset_missing_one_chunk_id_is_not_derivable(tmp_path: Path) -> None:
    """A subset that is not actually a subset must raise, never return a
    short or a reordered matrix."""
    partial_key, partial_texts = _ablated_key({"doc03"})
    EmbeddingCache.open(partial_key, root=tmp_path).build(partial_texts, toy_embed, shard_size=5)
    full_key, _ = _key()
    with pytest.raises(StaleCacheError, match="holds every one of"):
        resolve_vectors(full_key, root=tmp_path)


def test_a_cache_at_a_different_chunking_is_never_derived_from(tmp_path: Path) -> None:
    """Same corpus, different chunking: the chunk ids would *look*
    reusable (`doc01#0000` exists under both) while the text behind them
    differs. The namespace check is what stops that."""
    coarse_key, coarse_texts = _key(size=800, overlap=100)
    EmbeddingCache.open(coarse_key, root=tmp_path).build(coarse_texts, toy_embed, shard_size=5)
    fine_key, _ = _key(size=64, overlap=8)
    with pytest.raises(StaleCacheError, match="holds every one of"):
        resolve_vectors(fine_key, root=tmp_path)


def test_load_subset_returns_rows_in_the_requested_order(tmp_path: Path) -> None:
    """`load_subset`'s contract is "these chunk ids, in this order".

    Written after a planted defect got away: replacing the row lookup
    with `sorted(index[c] for c in chunk_ids)` -- a plausible "make fancy
    indexing sequential" optimisation that silently misaligns every
    vector -- survived `test_derived_subset_equals_a_freshly_built_subset`,
    because a document-ablated corpus keeps its surviving chunks in
    ascending order and `sorted` is therefore a no-op on it. Only a
    request whose order differs from the cache's own can tell the two
    apart, so that is what this asks for.
    """
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)
    full = cache.load()

    reversed_ids = list(reversed(key.chunk_ids))
    subset = cache.load_subset(reversed_ids)
    assert subset is not None
    assert np.array_equal(subset, full[::-1])

    picked = [key.chunk_ids[7], key.chunk_ids[1], key.chunk_ids[4]]
    rows = cache.load_subset(picked)
    assert rows is not None
    assert np.array_equal(rows, full[[7, 1, 4]])


def test_load_subset_is_none_when_an_id_is_absent(tmp_path: Path) -> None:
    key, texts = _key()
    cache = EmbeddingCache.open(key, root=tmp_path)
    cache.build(texts, toy_embed, shard_size=5)
    assert cache.load_subset([*key.chunk_ids, "doc99#0000"]) is None
