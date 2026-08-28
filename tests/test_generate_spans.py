"""M3-SPEC.md §7: quote not in document -> QUOTE_NOT_FOUND; quote appearing
twice -> QUOTE_AMBIGUOUS; no fuzzy matching anywhere. Also covers the
coordinator's prompt-injection-resistance addition: an imperative,
non-interrogative model response is discarded as NOT_A_QUESTION rather
than stored."""

from __future__ import annotations

import json
from pathlib import Path

from fireassay.generate.pipeline import generate_candidates
from fireassay.generate.prompts import build_prompt
from fireassay.generate.spans import ResolvedSpan, SpanRejection, resolve_span
from fireassay.generate.validate import is_imperative
from fireassay.llm.cache import ResponseCache
from fireassay.llm.ollama import ModelRef, OllamaClient
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
    prompt = build_prompt(chunk.text, doc.title, 1)
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
