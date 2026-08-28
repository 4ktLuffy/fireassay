"""M3-SPEC.md §7: quote not in document -> QUOTE_NOT_FOUND; quote appearing
twice -> QUOTE_AMBIGUOUS; no fuzzy matching anywhere. Also covers the
coordinator's prompt-injection-resistance addition: an imperative,
non-interrogative model response is discarded as NOT_A_QUESTION rather
than stored."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fireassay.generate.models import Candidate, CandidateBatch
from fireassay.generate.pipeline import (
    GenerationFailureRateExceededError,
    _build_gold_rank_retriever,
    _gold_doc_rank,
    generate_candidates,
    select_chunks_for_target,
    target_cell_for,
)
from fireassay.generate.prompts import build_prompt
from fireassay.generate.spans import ResolvedSpan, SpanRejection, resolve_span
from fireassay.generate.validate import is_imperative
from fireassay.llm.cache import ResponseCache
from fireassay.llm.ollama import GenerationExhaustedError, ModelRef, OllamaClient
from fireassay.store.db import Store
from fireassay.system.corpus import Chunk, Doc

# -- resolve_span -----------------------------------------------------------


def test_quote_not_found_is_rejected() -> None:
    result = resolve_span("a phrase that does not appear", "some unrelated document text")
    assert isinstance(result, SpanRejection)
    assert result.reason == "QUOTE_NOT_FOUND"


def test_quote_appearing_twice_is_ambiguous() -> None:
    doc = "cats are great pets. many people agree that cats are great pets indeed."
    result = resolve_span("cats are great pets", doc)
    assert isinstance(result, SpanRejection)
    assert result.reason == "QUOTE_AMBIGUOUS"


def test_quote_appearing_once_resolves_exact_offsets() -> None:
    doc = "the quick brown fox jumps over the lazy dog"
    result = resolve_span("quick brown fox", doc)
    assert isinstance(result, ResolvedSpan)
    assert doc[result.char_start : result.char_end] == "quick brown fox"


def test_no_fuzzy_matching_a_near_miss_quote_is_not_found() -> None:
    """A one-character typo must never resolve to "close enough" — a span
    pointing at the wrong occurrence (or no occurrence) is silently wrong
    ground truth."""
    doc = "the cat sat on the mat"
    result = resolve_span("the cat sit on the mat", doc)  # sat -> sit
    assert isinstance(result, SpanRejection)
    assert result.reason == "QUOTE_NOT_FOUND"


def test_empty_quote_is_rejected_not_matched_everywhere() -> None:
    result = resolve_span("", "any document text")
    assert isinstance(result, SpanRejection)
    assert result.reason == "QUOTE_NOT_FOUND"


# -- is_imperative ------------------------------------------------------


def test_is_imperative_true_for_bare_command_no_question_mark() -> None:
    assert is_imperative("Start now.") is True
    assert is_imperative("Sign in before continuing") is True


def test_is_imperative_false_for_genuine_question() -> None:
    assert is_imperative("What must you do before continuing?") is False


def test_is_imperative_false_when_question_mark_present_even_if_command_shaped() -> None:
    assert is_imperative("Start now?") is False


# -- end-to-end: an imperative model response is discarded, not stored ------


def test_imperative_candidate_is_discarded_as_not_a_question(store: Store, tmp_path: Path) -> None:
    """Coordinator addition: gov.uk service pages are wall-to-wall
    imperatives ("Start now", "You must sign in before continuing"). A
    generator that echoes one back as its `text` must be caught and
    counted, never persisted as a candidate."""
    doc = Doc(
        doc_id="d1",
        title="Apply for a licence",
        text="Start now. You must sign in before continuing. The fee is 20 pounds.",
    )
    chunk = Chunk(doc_id="d1", chunk_id="d1#0000", text=doc.text, char_start=0, char_end=len(doc.text))
    model = ModelRef(name="test-model", digest="digest-a")

    cache = ResponseCache(tmp_path / "cache.jsonl")
    target_qtype, target_difficulty = target_cell_for(chunk.chunk_id, 0)
    prompt = build_prompt(chunk.text, doc.title, target_qtype, target_difficulty)
    bad_response = json.dumps(
        {
            "candidates": [
                {
                    "text": "Start now. You must sign in before continuing.",
                    "qtype": "procedural",
                    "difficulty": "easy",
                    "reference_answer": "Sign in and start now.",
                    "quote": "Start now.",
                }
            ]
        }
    )
    cache.put(model, prompt, bad_response)
    # base_url would fail if actually called -- proves the cache hit is
    # what served this response, never the network.
    client = OllamaClient(base_url="http://127.0.0.1:1", cache=cache)

    stats = generate_candidates(store, client, model, [doc], [chunk], n_per_chunk=1, batch_id="batch-1")

    assert stats.generated == 1
    assert stats.not_a_question == 1
    assert stats.kept == 0
    assert stats.quote_not_found == 0
    assert stats.quote_ambiguous == 0
    assert store.get_candidates_by_batch("batch-1") == []

    filter_results = store.get_all_filter_results()
    not_a_question_rows = [r for r in filter_results if r.stage == "not_a_question"]
    assert len(not_a_question_rows) == 1
    assert not_a_question_rows[0].kept is False
    assert not_a_question_rows[0].reason == "NOT_A_QUESTION"


# -- end-to-end: gold_doc_rank is measured and stored on a kept candidate ---


def test_kept_candidate_gets_a_measured_gold_doc_rank(store: Store, tmp_path: Path) -> None:
    """`generate_candidates` builds one BM25 index over the whole corpus
    and records, once, at insert time, the rank at which the candidate's
    own source document is found for its own question text -- `candidate`
    rows are append-only, so this is the only point this feature can ever
    be measured (see generate/pipeline.py's module docstring)."""
    doc = Doc(
        doc_id="d1",
        title="Refund Policy",
        text="Refunds are issued within fourteen days of purchase if the product is unused.",
    )
    other_doc = Doc(
        doc_id="d2",
        title="Shipping Times",
        text="Standard shipping takes three to five business days for delivery.",
    )
    chunk = Chunk(doc_id="d1", chunk_id="d1#0000", text=doc.text, char_start=0, char_end=len(doc.text))
    model = ModelRef(name="test-model", digest="digest-b")

    cache = ResponseCache(tmp_path / "cache.jsonl")
    target_qtype, target_difficulty = target_cell_for(chunk.chunk_id, 0)
    prompt = build_prompt(chunk.text, doc.title, target_qtype, target_difficulty)
    good_response = json.dumps(
        {
            "candidates": [
                {
                    "text": "How many days do I have to request a refund on an unused product?",
                    "qtype": "factual",
                    "difficulty": "easy",
                    "reference_answer": "Fourteen days.",
                    "quote": "Refunds are issued within fourteen days of purchase",
                }
            ]
        }
    )
    cache.put(model, prompt, good_response)
    client = OllamaClient(base_url="http://127.0.0.1:1", cache=cache)

    stats = generate_candidates(
        store, client, model, [doc, other_doc], [chunk], n_per_chunk=1, batch_id="batch-2"
    )

    assert stats.kept == 1
    candidates = store.get_candidates_by_batch("batch-2")
    assert len(candidates) == 1
    rank = candidates[0].features.gold_doc_rank
    # d1 is the only document sharing "refund"/"days"/"unused" with the
    # question, against a two-document corpus -- it must be found, and
    # found first.
    assert rank == 1


# -- chunk selection: stable and independent of n ---------------------


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(doc_id=f"d{i}", chunk_id=f"d{i}#0000", text="x", char_start=0, char_end=1) for i in range(n)
    ]


def test_select_chunks_for_target_smaller_n_is_a_prefix_of_larger_n() -> None:
    """The coordinator's pre-flight fix: `--n 4000` topping up to
    `--n 5000` must reuse every prompt the smaller run already cached,
    not re-stride the whole corpus and miss cache on almost everything."""
    chunks = _chunks(50)
    smaller = select_chunks_for_target(chunks, target_candidates=10, n_per_chunk=1)
    larger = select_chunks_for_target(chunks, target_candidates=20, n_per_chunk=1)
    assert larger[: len(smaller)] == smaller


def test_select_chunks_for_target_order_does_not_depend_on_corpus_file_order() -> None:
    chunks = _chunks(50)
    reordered = list(reversed(chunks))
    assert select_chunks_for_target(chunks, 10, 1) == select_chunks_for_target(reordered, 10, 1)


def test_select_chunks_for_target_returns_the_requested_count() -> None:
    chunks = _chunks(50)
    assert len(select_chunks_for_target(chunks, target_candidates=15, n_per_chunk=1)) == 15


def test_select_chunks_for_target_caps_at_available_chunks() -> None:
    chunks = _chunks(5)
    assert len(select_chunks_for_target(chunks, target_candidates=100, n_per_chunk=1)) == 5


def test_select_chunks_for_target_empty_chunks_is_empty() -> None:
    assert select_chunks_for_target([], target_candidates=10, n_per_chunk=1) == []


def test_select_chunks_for_target_rejects_non_positive_n_per_chunk() -> None:
    with pytest.raises(ValueError, match="n_per_chunk"):
        select_chunks_for_target(_chunks(5), target_candidates=10, n_per_chunk=0)


# -- stratified generation: taxonomy coverage is not left to the model ----


def test_target_cell_for_is_deterministic_given_chunk_id() -> None:
    assert target_cell_for("d1#0000", 0) == target_cell_for("d1#0000", 0)


def test_target_cell_for_cycles_within_a_chunk() -> None:
    """Multiple candidates requested from the same chunk (n_per_chunk > 1)
    must spread across different cells, not repeat the same one."""
    cells = [target_cell_for("d1#0000", i) for i in range(12)]
    assert len(set(cells)) == 12  # a full cycle of all 12 cells, no repeats


def test_target_cell_for_is_always_one_of_the_twelve_in_scope_cells() -> None:
    valid_qtypes = {"factual", "procedural", "comparative", "policy_sensitive"}
    valid_difficulties = {"easy", "medium", "hard"}
    for i in range(50):
        qtype, difficulty = target_cell_for(f"chunk-{i}", 0)
        assert qtype in valid_qtypes
        assert difficulty in valid_difficulties


def test_generation_spans_more_than_one_cell_across_many_chunks(store: Store, tmp_path: Path) -> None:
    """Coordinator's pre-flight fix: an unstratified prompt was measured
    to collapse generation to ~2 of 21 taxonomy cells. Stratified
    generation across a handful of different chunks must span more than
    one cell, and every kept candidate must carry the target cell it was
    actually asked for."""
    n = 12
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
    model = ModelRef(name="test-model", digest="digest-multi")
    cache = ResponseCache(tmp_path / "cache.jsonl")

    expected_cells: dict[str, tuple[str, str]] = {}
    for i, chunk in enumerate(chunks):
        target_qtype, target_difficulty = target_cell_for(chunk.chunk_id, 0)
        expected_cells[chunk.chunk_id] = (target_qtype, target_difficulty)
        prompt = build_prompt(chunk.text, docs[i].title, target_qtype, target_difficulty)
        response = json.dumps(
            {
                "candidates": [
                    {
                        "text": f"What is document number {i} about?",
                        "qtype": target_qtype,
                        "difficulty": target_difficulty,
                        "reference_answer": f"Topic {i}.",
                        "quote": f"document number {i}",
                    }
                ]
            }
        )
        cache.put(model, prompt, response)

    client = OllamaClient(base_url="http://127.0.0.1:1", cache=cache)
    stats = generate_candidates(store, client, model, docs, chunks, n_per_chunk=1, batch_id="batch-multi")

    assert stats.kept == n
    candidates = store.get_candidates_by_batch("batch-multi")
    assert len(candidates) == n

    recorded_cells = {(c.target_qtype, c.target_difficulty) for c in candidates}
    assert len(recorded_cells) > 1, "generation must not collapse to a single taxonomy cell"

    for c in candidates:
        assert (c.target_qtype, c.target_difficulty) == expected_cells[f"{c.source_doc_id}#0000"]


def test_gold_doc_rank_is_none_with_no_retriever_and_never_raises() -> None:
    """`_gold_doc_rank` degrades to `None`, never raises, when there is no
    retriever to measure against (`_build_gold_rank_retriever` returns
    `None` for an empty corpus) -- exercised directly rather than through
    `generate_candidates`, whose per-chunk `docs_by_id` lookup requires
    every chunk's own document to be present regardless of gold-rank
    concerns, so a genuinely empty `docs` cannot reach this path any other
    way."""
    retriever = _build_gold_rank_retriever([])
    assert retriever is None
    assert _gold_doc_rank(retriever, "any question text", "d1", "factual", "easy") is None


# -- generation failures: caught per attempt, not per batch -----------------


class _StubClient:
    """A minimal stand-in for `OllamaClient`: `generate_json` raises
    `GenerationExhaustedError` for prompts in `failing_prompts` and
    returns the matching `CandidateBatch` from `responses` otherwise --
    no network, no cache, no retry machinery, so these tests exercise
    only `generate_candidates`'s own try/except and failure-rate logic."""

    def __init__(self, failing_prompts: set[str], responses: dict[str, CandidateBatch]) -> None:
        self._failing_prompts = failing_prompts
        self._responses = responses
        self.calls: list[str] = []

    def generate_json(self, model: object, prompt: str, schema: object) -> CandidateBatch:
        self.calls.append(prompt)
        if prompt in self._failing_prompts:
            raise GenerationExhaustedError(f"stub failure for prompt starting {prompt[:40]!r}")
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


def _prompt_for(doc: Doc, chunk: Chunk) -> str:
    target_qtype, target_difficulty = target_cell_for(chunk.chunk_id, 0)
    return build_prompt(chunk.text, doc.title, target_qtype, target_difficulty)


def _good_response_for(i: int, chunk: Chunk) -> CandidateBatch:
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


def test_one_failing_chunk_yields_the_others_and_records_one_generation_failed(store: Store) -> None:
    docs, chunks = _docs_and_chunks(3)
    model = ModelRef(name="test-model", digest="digest-stub")

    prompts = [_prompt_for(docs[i], chunks[i]) for i in range(3)]
    failing_prompts = {prompts[1]}
    responses = {prompts[i]: _good_response_for(i, chunks[i]) for i in (0, 2)}
    client = _StubClient(failing_prompts, responses)

    stats = generate_candidates(store, client, model, docs, chunks, n_per_chunk=1, batch_id="batch-fail")

    assert stats.kept == 2
    assert stats.generation_failed == 1
    assert stats.generated == 3  # reconciles: 2 kept + 1 generation_failed

    generation_rows = [r for r in store.get_all_filter_results() if r.stage == "generation"]
    failed_rows = [r for r in generation_rows if not r.kept]
    assert len(failed_rows) == 1
    assert failed_rows[0].reason == "GENERATION_FAILED"


def test_exceeding_max_generation_failure_rate_aborts_early_with_a_clear_message(store: Store) -> None:
    """Every attempt fails; the check trips as soon as the minimum sample
    size is reached, so the run aborts before grinding through the rest
    of the (here, deliberately larger) chunk list."""
    docs, chunks = _docs_and_chunks(10)
    model = ModelRef(name="test-model", digest="digest-stub")
    prompts = [_prompt_for(docs[i], chunks[i]) for i in range(10)]
    client = _StubClient(set(prompts), {})

    with pytest.raises(GenerationFailureRateExceededError) as exc_info:
        generate_candidates(
            store, client, model, docs, chunks, n_per_chunk=1, batch_id="batch-abort",
            max_generation_failure_rate=0.2, min_attempts_before_failure_rate_check=2,
        )

    message = str(exc_info.value)
    assert "2/2" in message  # names the counts: 2 failed of 2 attempts
    assert "20.0%" in message  # names the configured threshold
    # Aborted at the minimum sample size, not after grinding through all 10.
    assert len(client.calls) == 2


def test_failure_rate_below_threshold_does_not_abort(store: Store) -> None:
    docs, chunks = _docs_and_chunks(10)
    model = ModelRef(name="test-model", digest="digest-stub")
    prompts = [_prompt_for(docs[i], chunks[i]) for i in range(10)]
    # 1 of 10 fails -> 10%, below the 20% default threshold.
    failing_prompts = {prompts[0]}
    responses = {prompts[i]: _good_response_for(i, chunks[i]) for i in range(1, 10)}
    client = _StubClient(failing_prompts, responses)

    stats = generate_candidates(
        store, client, model, docs, chunks, n_per_chunk=1, batch_id="batch-ok",
        max_generation_failure_rate=0.2, min_attempts_before_failure_rate_check=2,
    )

    assert stats.generation_failed == 1
    assert stats.kept == 9
    assert len(client.calls) == 10  # ran to completion, never aborted
