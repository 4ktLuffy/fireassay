"""`test_rag_system.py`: `system.rag.RagSystem` is the first system in the
project for which `generates_answers` is `True` (see that module's
docstring) -- it is what activates `score.abstention.AbstentionScorer`,
which has been structurally inert until now. All hermetic: every test
injects a fake `Generate`; no test calls a live model.
"""

from __future__ import annotations

import pytest

from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.system.base import System
from fireassay.system.corpus import Chunk
from fireassay.system.rag import Generate, RagSystem, build_prompt

_SENTINEL = "INSUFFICIENT_CONTEXT"


def _question(text: str = "What is the passport renewal fee?") -> Question:
    return Question(text=text, qtype="factual", difficulty="easy", provenance="synthetic")


def _chunk(doc_id: str, chunk_id: str, text: str) -> Chunk:
    return Chunk(doc_id=doc_id, chunk_id=chunk_id, text=text, char_start=0, char_end=len(text))


def _retrieved(doc_id: str, chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, chunk_id=chunk_id, score=1.0, rank=rank, char_start=0, char_end=10)


class FakeRetriever:
    """A minimal `System` double: returns a fixed `retrieved` tuple and
    never generates an answer, exactly like `BM25System`/`CoverageSystem`."""

    def __init__(self, retrieved: tuple[RetrievedChunk, ...], name: str = "fake-retriever") -> None:
        self.name = name
        self.generates_answers = False
        self.retrieved = retrieved

    def answer(self, question: Question) -> SystemOutput:
        return SystemOutput(
            answer=None,
            abstained=True,
            retrieved=self.retrieved,
            latency_ms={"total": 0.01},
        )


_CHUNKS = (
    _chunk("doc01", "doc01#0000", "The passport renewal fee is 75.50 GBP for adults."),
    _chunk("doc01", "doc01#0001", "Processing normally takes three weeks."),
)
_RETRIEVED = (
    _retrieved("doc01", "doc01#0000", 1),
    _retrieved("doc01", "doc01#0001", 2),
)


def _system(generate: Generate) -> RagSystem:
    return RagSystem(FakeRetriever(_RETRIEVED), generate, _CHUNKS)


# -- 1. normal completion -> answer, not abstained, retrieved passed through -


def test_normal_completion_becomes_answer_and_retrieved_passed_through_unchanged() -> None:
    system = _system(lambda prompt: "75.50 GBP")
    output = system.answer(_question())

    assert output.answer == "75.50 GBP"
    assert output.abstained is False
    assert output.retrieved == _RETRIEVED


# -- 2. the sentinel alone -> abstention --------------------------------------


def test_sentinel_completion_becomes_abstention() -> None:
    system = _system(lambda prompt: _SENTINEL)
    output = system.answer(_question())

    assert output.answer is None
    assert output.abstained is True


# -- 3. injection defence, named ----------------------------------------------


def test_sentinel_quoted_in_context_does_not_force_abstention_when_real_answer_follows() -> None:
    """A completion whose *quoted context* contains `INSUFFICIENT_CONTEXT`
    but which ends with a real answer must NOT be treated as an
    abstention -- the CONTEXT is gov.uk corpus text injected verbatim, and
    a document containing the sentinel must not be able to force one."""
    raw = (
        'The retrieved page includes a banner reading "INSUFFICIENT_CONTEXT" for '
        "unrelated visitors, but that is not this answer.\n"
        "The passport renewal fee is 75.50 GBP.\n"
    )
    system = _system(lambda prompt: raw)
    output = system.answer(_question())

    assert output.abstained is False
    assert output.answer == raw.strip()


def test_sentinel_repeated_earlier_still_loses_to_real_final_answer() -> None:
    """Same property, stated the other way: many earlier occurrences of the
    exact sentinel, of every flavour, all lose to the true last word."""
    raw = (
        f"{_SENTINEL}\n{_SENTINEL}\nIgnore the above, here is my real answer.\n75.50 GBP\n"
    )
    system = _system(lambda prompt: raw)
    output = system.answer(_question())

    assert output.abstained is False
    assert output.answer == raw.strip()


# -- 4. empty / whitespace-only completions abstain, never answer "" ---------


def test_empty_completion_abstains_rather_than_answering_empty_string() -> None:
    system = _system(lambda prompt: "")
    output = system.answer(_question())

    assert output.answer is None
    assert output.abstained is True


def test_whitespace_only_completion_abstains_rather_than_answering_empty_string() -> None:
    system = _system(lambda prompt: "   \n\t  ")
    output = system.answer(_question())

    assert output.answer is None
    assert output.abstained is True


# -- 5. latency_ms has total, retrieval, generation; total >= each stage -----


def test_latency_ms_has_total_retrieval_generation_and_total_is_at_least_each_stage() -> None:
    system = _system(lambda prompt: "75.50 GBP")
    output = system.answer(_question())

    assert set(output.latency_ms) >= {"total", "retrieval", "generation"}
    assert output.latency_ms["total"] >= output.latency_ms["retrieval"]
    assert output.latency_ms["total"] >= output.latency_ms["generation"]


# -- 6. the prompt contains the question and every retrieved chunk's text ----


def test_prompt_contains_question_and_every_retrieved_chunk_text() -> None:
    seen_prompts: list[str] = []

    def fake_generate(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "75.50 GBP"

    question = _question()
    system = _system(fake_generate)
    system.answer(question)

    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    assert question.text in prompt
    for chunk in _CHUNKS:
        assert chunk.text in prompt
    # `build_prompt` is the pinned, reusable prompt builder -- confirm this
    # is really what RagSystem calls, not an inline duplicate.
    assert prompt == build_prompt(question, [chunk.text for chunk in _CHUNKS])


# -- 7. generates_answers is True; RagSystem satisfies the System protocol ---


def test_generates_answers_true_and_ragsystem_satisfies_system_protocol() -> None:
    system = _system(lambda prompt: "75.50 GBP")
    assert system.generates_answers is True

    # `System` (system/base.py) is not `@runtime_checkable`, so structural
    # conformance is asserted the way `mypy src/` would police it -- a typed
    # assignment -- plus a direct runtime check of its three members.
    conforms: System = system
    assert isinstance(conforms.name, str)
    assert isinstance(conforms.generates_answers, bool)
    assert callable(conforms.answer)


# -- 8. generate is called exactly once per answer() call --------------------


def test_generate_called_exactly_once_per_answer_call() -> None:
    call_count = 0

    def fake_generate(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "75.50 GBP"

    system = _system(fake_generate)
    system.answer(_question())
    assert call_count == 1

    system.answer(_question("A different question?"))
    assert call_count == 2


# -- 9. retriever/chunks disagreement raises, never silently drops context ---


def test_chunk_id_absent_from_supplied_chunks_raises_valueerror_naming_it() -> None:
    """Chunks from a different chunking than the retriever was built on
    must fail loudly, not be silently skipped or filled with empty text."""
    mismatched_chunks = (_chunk("doc01", "doc01#0000-1024", "different chunking entirely"),)
    system = RagSystem(FakeRetriever(_RETRIEVED), lambda prompt: "75.50 GBP", mismatched_chunks)

    with pytest.raises(ValueError, match="doc01#0000"):
        system.answer(_question())
