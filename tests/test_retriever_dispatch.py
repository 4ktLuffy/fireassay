"""The `retriever` config key — and the compatibility contract that makes
adding it safe.

**The contract:** a config dict that does not mention `retriever` must
behave exactly as it did before the key existed. Every stored run, every
`config_hash` in `run/golden.db`, and every number in `EVIDENCE.md` was
produced from such a dict; if adding a key with a default moved any of
them, the whole panel would have to be re-run to stay comparable
(`integrity.assert_comparable`'s `CONFIG_MISMATCH`). `config_hash` is
computed over the dict as written, so the guarantee is structural — and
`test_absent_retriever_key_leaves_config_hash_untouched` pins it anyway,
because a structural guarantee nobody checks is a comment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from fireassay.cli import RETRIEVERS, _build_retriever, _docs_system_factory
from fireassay.hashing import config_hash
from fireassay.llm.ollama import ModelRef
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import chunk_corpus, corpus_hash
from fireassay.system.coverage import CoverageSystem
from fireassay.system.dense import DenseSystem
from fireassay.system.embedding import OllamaEmbedder
from fireassay.system.embedding_cache import CacheKey, EmbeddingCache, StaleCacheError
from fireassay.system.tfidf import TfidfSystem
from helpers import load_fixture_docs, load_fixture_questions, toy_embed


def _chunks() -> list:
    return list(chunk_corpus(load_fixture_docs(), size=800, overlap=100))


def test_an_absent_retriever_key_still_builds_bm25() -> None:
    system = _build_retriever({"top_k": 5}, _chunks(), load_fixture_docs())
    assert isinstance(system, BM25System)
    assert system.name == "bm25"


def test_absent_retriever_key_leaves_config_hash_untouched() -> None:
    """The actual compatibility claim: no stored run's identity moves."""
    spec: Mapping[str, object] = {"top_k": 5, "chunk_size": 800, "chunk_overlap": 100}
    # The hash of the exact dict shape every existing run was recorded
    # under, recomputed here. Adding `retriever` to the *dispatch* must
    # not add it to the *dict*.
    assert config_hash(spec) == config_hash({"chunk_overlap": 100, "chunk_size": 800, "top_k": 5})
    assert config_hash(spec) != config_hash({**spec, "retriever": "bm25"})


def test_explicit_bm25_is_the_same_system_as_the_default() -> None:
    chunks = _chunks()
    docs = load_fixture_docs()
    default = _build_retriever({"top_k": 3}, chunks, docs)
    explicit = _build_retriever({"top_k": 3, "retriever": "bm25"}, chunks, docs)
    assert type(default) is type(explicit)
    question = load_fixture_questions()[0]
    assert [c.chunk_id for c in default.answer(question).retrieved] == [
        c.chunk_id for c in explicit.answer(question).retrieved
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("bm25", BM25System), ("tfidf", TfidfSystem), ("coverage", CoverageSystem)],
)
def test_every_offline_retriever_name_dispatches(name: str, expected: type) -> None:
    system = _build_retriever({"retriever": name, "top_k": 3}, _chunks(), load_fixture_docs())
    assert isinstance(system, expected)
    assert system.top_k == 3


def test_bm25_parameters_still_reach_the_system() -> None:
    system = _build_retriever({"retriever": "bm25", "k1": 2.0, "b": 0.5}, _chunks(), load_fixture_docs())
    assert isinstance(system, BM25System)
    assert (system.k1, system.b) == (2.0, 0.5)


def test_an_unknown_retriever_is_an_error_not_a_silent_bm25() -> None:
    with pytest.raises(ValueError, match="is not one of"):
        _build_retriever({"retriever": "faiss"}, _chunks(), load_fixture_docs())


def test_a_non_string_retriever_is_an_error() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        _build_retriever({"retriever": 7}, _chunks(), load_fixture_docs())


def test_dense_is_registered_but_needs_no_import_of_it_to_use_the_others() -> None:
    assert RETRIEVERS == ("bm25", "tfidf", "coverage", "dense")


def test_docs_system_factory_honours_the_retriever_key() -> None:
    """`_docs_system_factory` is the shape `controls.base.ControlContext`
    and `mutation.run` take, so the controls see the same dispatch the
    `run` command does."""
    factory = _docs_system_factory(800, 100)
    docs = load_fixture_docs()
    assert isinstance(factory({"top_k": 3}, docs), BM25System)
    assert isinstance(factory({"top_k": 3, "retriever": "tfidf"}, docs), TfidfSystem)


def test_dense_dispatch_builds_from_the_cache_and_never_embeds_the_corpus_in_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cli._build_dense`, end to end, with `helpers.toy_embed` standing in
    for Ollama (M3-SPEC.md §7: no test calls a live model).

    Two things are pinned: the vectors come off disk, and a config whose
    cache has not been built raises rather than quietly spending an hour
    of model time inside a run.
    """
    docs = load_fixture_docs()
    chunks = list(chunk_corpus(docs, size=800, overlap=100))
    model = ModelRef(name="qwen3-embedding:0.6b", digest="ac6da0dfba84a81f")

    class _Stub:
        def __init__(self) -> None:
            self.model = model
            self.corpus_calls = 0

        def __call__(self, texts: Sequence[str]) -> object:
            self.corpus_calls += 1
            return toy_embed(texts)

    stub = _Stub()
    monkeypatch.setattr(OllamaEmbedder, "resolve", classmethod(lambda cls, *a, **kw: stub))

    config = {
        "retriever": "dense",
        "top_k": 3,
        "chunk_size": 800,
        "chunk_overlap": 100,
        "embed_cache": str(tmp_path),
    }
    with pytest.raises(StaleCacheError, match="no embedding cache"):
        _build_retriever(config, chunks, docs)

    key = CacheKey(
        model_name=model.name,
        model_digest=model.digest,
        chunk_size=800,
        chunk_overlap=100,
        corpus_hash=corpus_hash(docs),
        chunk_ids=tuple(c.chunk_id for c in chunks),
    )
    EmbeddingCache.open(key, root=tmp_path).build([c.text for c in chunks], toy_embed)
    stub.corpus_calls = 0

    system = _build_retriever(config, chunks, docs)
    assert isinstance(system, DenseSystem)
    assert stub.corpus_calls == 0, "building a dense system must not re-embed the corpus"
    retrieved = system.answer(load_fixture_questions()[0]).retrieved
    assert len(retrieved) == 3
    assert stub.corpus_calls == 1, "only the question is embedded live, and only on a cache miss"
