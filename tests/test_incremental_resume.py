"""M3b-SPEC.md Part 1: a second run over the same db adds no duplicate
candidates; a run over a partially populated db processes only the
missing chunks; skipped counts are reported.

Uses a minimal stand-in client (mirrors `test_generate_spans.py`'s own
`_StubClient`, duplicated rather than imported since each test file's
helpers are private to it) so these tests exercise only
`generate_candidates`'s own resume/skip logic, never the network or a real
cache replay. `build_prompt` itself is monkeypatched to a simple,
per-chunk-unique scheme (`f"prompt-for-{chunk_id}"`) purely so the stub
client's response table can be keyed unambiguously -- these tests assert
nothing about real prompt *content*, only about which chunks a call was
actually made for, so byte-identical real prompts are not needed here the
way they are for `test_generate_spans.py`'s cache-fixture tests.
"""

from __future__ import annotations

import pytest

from fireassay.generate import pipeline as pipeline_module
from fireassay.generate.models import Candidate, CandidateBatch, CandidateFeatures, ResolvedCandidate
from fireassay.generate.pipeline import GenerationStats, generate_candidates, target_cell_for
from fireassay.llm.ollama import ModelRef
from fireassay.store.db import Store
from fireassay.system.corpus import Chunk, Doc


class _StubClient:
    """Returns a canned `CandidateBatch` per prompt and records every
    prompt it was actually asked to generate -- what these tests check is
    that a resumed chunk's prompt is never even attempted a second time,
    not just that its final candidate count happens to match."""

    def __init__(self, responses: dict[str, CandidateBatch]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def generate_json(self, model: object, prompt: str, schema: object) -> CandidateBatch:
        self.calls.append(prompt)
        return self._responses[prompt]


def _docs_and_chunks(n: int) -> tuple[list[Doc], list[Chunk]]:
    docs = [
        Doc(
            doc_id=f"d{i}",
            title=f"Title {i}",
            text=f"This is document number {i} with some unique content about topic {i}.",
        )
        for i in range(n)
    ]
    chunks = [
        Chunk(
            doc_id=f"d{i}", chunk_id=f"d{i}#0000", text=docs[i].text, char_start=0, char_end=len(docs[i].text)
        )
        for i in range(n)
    ]
    return docs, chunks


def _prompt_for_chunk(chunk: Chunk) -> str:
    return f"prompt-for-{chunk.chunk_id}"


def _response_for(i: int, chunk: Chunk) -> CandidateBatch:
    target_qtype, target_difficulty = target_cell_for(chunk.chunk_id, 0)
    return CandidateBatch(
        candidates=(
            Candidate(
                text=f"What is document number {i} about?",
                qtype=target_qtype,
                difficulty=target_difficulty,
                reference_answer=f"Topic {i}.",
                quote=f"document number {i}",
            ),
        )
    )


def _client_for(chunks: list[Chunk]) -> _StubClient:
    responses = {_prompt_for_chunk(chunk): _response_for(i, chunk) for i, chunk in enumerate(chunks)}
    return _StubClient(responses)


def _run(
    store: Store,
    client: _StubClient,
    docs: list[Doc],
    chunks: list[Chunk],
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_id: str,
) -> GenerationStats:
    chunk_text_to_prompt = {c.text: _prompt_for_chunk(c) for c in chunks}

    def fake_build_prompt(chunk_text: str, title: str, qtype: object, difficulty: object) -> str:
        return chunk_text_to_prompt[chunk_text]

    monkeypatch.setattr(pipeline_module, "build_prompt", fake_build_prompt)
    model = ModelRef(name="test-model", digest="digest-resume")
    return generate_candidates(store, client, model, docs, chunks, n_per_chunk=1, batch_id=batch_id)


def test_second_run_over_the_same_db_adds_no_duplicate_candidates(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs, chunks = _docs_and_chunks(3)
    client = _client_for(chunks)

    first = _run(store, client, docs, chunks, monkeypatch, batch_id="batch-1")
    assert first.kept == 3
    assert first.resumed == 0

    second = _run(store, client, docs, chunks, monkeypatch, batch_id="batch-2")
    assert second.kept == 0
    assert second.resumed == 3

    all_candidates = store.get_all_candidates()
    assert len(all_candidates) == 3  # not 6: no duplicates from the second run
    texts = [c.text for c in all_candidates]
    assert len(texts) == len(set(texts)), "no duplicate question text across the two runs"


def test_partially_populated_db_processes_only_missing_chunks(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs, chunks = _docs_and_chunks(5)
    client = _client_for(chunks)

    # First run only over the first two chunks -- simulates a database
    # that already has candidates for a subset of the full chunk list.
    first_stats = _run(store, client, docs[:2], chunks[:2], monkeypatch, batch_id="batch-partial-1")
    assert first_stats.kept == 2
    assert len(client.calls) == 2

    # Second run over the FULL chunk list: only the 3 new chunks should
    # ever reach the client.
    client.calls.clear()
    second_stats = _run(store, client, docs, chunks, monkeypatch, batch_id="batch-partial-2")
    assert second_stats.kept == 3
    assert second_stats.resumed == 2
    assert len(client.calls) == 3

    all_candidates = store.get_all_candidates()
    assert len(all_candidates) == 5
    assert {c.chunk_id for c in all_candidates} == {c.chunk_id for c in chunks}


def test_skip_counts_are_reported_in_generation_stats(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    docs, chunks = _docs_and_chunks(4)
    client = _client_for(chunks)

    _run(store, client, docs[:1], chunks[:1], monkeypatch, batch_id="batch-a")
    stats = _run(store, client, docs, chunks, monkeypatch, batch_id="batch-b")

    assert stats.resumed == 1
    assert stats.kept == 3
    # resumed is deliberately not folded into `generated` -- a resumed
    # chunk contributes zero filter_result rows this run (see
    # GenerationStats.resumed's own docstring).
    assert stats.generated == 3


def _bare_candidate(candidate_id: str, chunk_id: str = "") -> ResolvedCandidate:
    return ResolvedCandidate(
        id=candidate_id,
        batch_id="b0",
        text=f"a question {candidate_id}?",
        qtype="factual",
        difficulty="easy",
        target_qtype="factual",
        target_difficulty="easy",
        reference_answer="an answer",
        quote="a quote",
        source_doc_id="doc1",
        char_start=0,
        char_end=1,
        features=CandidateFeatures(title_overlap=0.0, quote_overlap=0.0, question_len_tokens=1),
        model_digest="digest",
        prompt_hash="prompt-hash",
        created_at="2026-01-01T00:00:00",
        chunk_id=chunk_id,
    )


def test_candidate_source_chunks_excludes_pre_migration_empty_chunk_id(store: Store) -> None:
    """A `chunk_id=''` candidate (the migration-0005 default for a
    pre-existing row) must never appear in the resume skip set -- it is
    not "chunk ''", it is "no chunk recorded"."""
    store.put_candidate(_bare_candidate("legacy-1"))  # chunk_id defaults to ""
    assert store.candidate_source_chunks() == set()


def test_candidate_source_chunks_reflects_real_chunk_ids(store: Store) -> None:
    store.put_candidate(_bare_candidate("c1", chunk_id="doc1#0000"))
    assert store.candidate_source_chunks() == {"doc1#0000"}
