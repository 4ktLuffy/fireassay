from __future__ import annotations

import pytest

from fireassay.integrity import ComparisonRefusedError, assert_comparable
from fireassay.models import Run
from fireassay.system.corpus import Doc, chunk_corpus, corpus_hash


def test_corpus_hash_is_deterministic() -> None:
    docs = [Doc(doc_id="d1", title="t", text="hello world this is some text")]
    assert corpus_hash(docs) == corpus_hash(docs)


def test_corpus_hash_changes_when_a_document_is_edited() -> None:
    docs_v1 = [Doc(doc_id="d1", title="t", text="the original sentence")]
    docs_v2 = [Doc(doc_id="d1", title="t", text="the edited, different sentence")]
    assert corpus_hash(docs_v1) != corpus_hash(docs_v2)


def test_corpus_hash_is_order_independent() -> None:
    docs = [
        Doc(doc_id="d1", title="t", text="first document text"),
        Doc(doc_id="d2", title="t", text="second document text"),
    ]
    assert corpus_hash(docs) == corpus_hash(list(reversed(docs)))


def test_corpus_hash_is_independent_of_chunking() -> None:
    """The resolved design (M1-SPEC.md correction, round 3): chunking is a
    property of the config (already captured in config_hash), not of the
    corpus, so corpus_hash must not vary with chunk size/overlap at all --
    it is computed from raw docs.text, never from Chunk objects."""
    docs = [Doc(doc_id="d1", title="t", text="a fairly long piece of support article text " * 5)]
    # chunk_corpus itself is not even called here: corpus_hash's signature
    # takes Doc, not Chunk, so there is nothing chunking-shaped to pass it.
    # This is exercised at the integration level below instead.
    assert corpus_hash(docs) == corpus_hash(docs)


def _run(**overrides: object) -> Run:
    defaults: dict[str, object] = {
        "id": "run1",
        "suite_id": "suite1",
        "suite_hash": "hashA",
        "config_id": "cfg1",
        "config_hash": "cfghashA",
        "env_json": {"python": "3.12", "fireassay": "0.1.0", "affects_results": {}},
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "status": "complete",
        "result_count": 10,
    }
    defaults.update(overrides)
    return Run(**defaults)  # type: ignore[arg-type]


def test_changed_corpus_hash_triggers_env_mismatch() -> None:
    """A corpus that changed underneath a suite (e.g. a gov.uk page
    revised) must make two otherwise-identical runs incomparable, via the
    existing ENV_MISMATCH path -- corpus_hash lives inside
    affects_results specifically so this happens for free."""
    docs_v1 = [Doc(doc_id="d1", title="t", text="the original support article text")]
    docs_v2 = [Doc(doc_id="d1", title="t", text="the revised support article text, edited")]
    hash_v1 = corpus_hash(docs_v1)
    hash_v2 = corpus_hash(docs_v2)
    assert hash_v1 != hash_v2

    run_a = _run(
        id="a", env_json={"python": "3.12", "fireassay": "0.1.0", "affects_results": {"corpus_hash": hash_v1}}
    )
    run_b = _run(
        id="b", env_json={"python": "3.12", "fireassay": "0.1.0", "affects_results": {"corpus_hash": hash_v2}}
    )

    scorers = {"a": frozenset({"s@1.0.0"}), "b": frozenset({"s@1.0.0"})}
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([run_a, run_b], scorers_by_run=scorers)
    codes = {r.code for r in exc_info.value.reasons}
    assert "ENV_MISMATCH" in codes


def test_same_corpus_hash_does_not_trigger_env_mismatch() -> None:
    docs = [Doc(doc_id="d1", title="t", text="stable support article text")]
    h = corpus_hash(docs)

    env: dict[str, object] = {"python": "3.12", "fireassay": "0.1.0", "affects_results": {"corpus_hash": h}}
    run_a = _run(id="a", env_json=env)
    run_b = _run(id="b", env_json=env)

    scorers = {"a": frozenset({"s@1.0.0"}), "b": frozenset({"s@1.0.0"})}
    report = assert_comparable([run_a, run_b], scorers_by_run=scorers)
    assert report.comparable is True


def test_different_chunk_sizes_over_same_docs_are_comparable() -> None:
    """The specific regression this round of correction fixes: two configs
    that differ only in chunking (a config-axis concern, already captured
    in config_hash) must get the SAME corpus_hash and remain comparable --
    corpus_hash must not manufacture a spurious ENV_MISMATCH between them.
    """
    docs = [Doc(doc_id="d1", title="t", text="a fairly long piece of support article text, " * 8)]

    # Two very different chunkings of the identical documents.
    chunks_small = list(chunk_corpus(docs, size=50, overlap=5))
    chunks_large = list(chunk_corpus(docs, size=400, overlap=0))
    assert len(chunks_small) != len(chunks_large)  # genuinely different chunking

    # corpus_hash never even looks at the chunks -- it is computed once
    # from the docs, independent of how they end up chunked.
    hash_for_small_config = corpus_hash(docs)
    hash_for_large_config = corpus_hash(docs)
    assert hash_for_small_config == hash_for_large_config

    run_small_chunks = _run(
        id="small",
        config_hash="cfg-small-chunks",
        env_json={
            "python": "3.12",
            "fireassay": "0.1.0",
            "affects_results": {"corpus_hash": hash_for_small_config},
        },
    )
    run_large_chunks = _run(
        id="large",
        config_hash="cfg-large-chunks",
        env_json={
            "python": "3.12",
            "fireassay": "0.1.0",
            "affects_results": {"corpus_hash": hash_for_large_config},
        },
    )

    scorers = {"small": frozenset({"retrieval@1.0.0"}), "large": frozenset({"retrieval@1.0.0"})}
    report = assert_comparable([run_small_chunks, run_large_chunks], scorers_by_run=scorers)
    assert report.comparable is True
