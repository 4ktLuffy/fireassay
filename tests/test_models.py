from __future__ import annotations

import pytest
from pydantic import ValidationError

from fireassay.models import EvidenceSpan, Question, SystemOutput


def test_generator_rejected_for_real_traffic() -> None:
    with pytest.raises(ValidationError):
        Question(
            text="q",
            qtype="factual",
            difficulty="easy",
            provenance="real_traffic",
            generator="gen@1.0.0",
        )


def test_generator_none_allowed_for_real_traffic() -> None:
    q = Question(text="q", qtype="factual", difficulty="easy", provenance="real_traffic")
    assert q.generator is None


def test_generator_allowed_for_synthetic() -> None:
    q = Question(
        text="q", qtype="factual", difficulty="easy", provenance="synthetic", generator="gen@1.0.0"
    )
    assert q.generator == "gen@1.0.0"


def test_latency_ms_without_total_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOutput(answer="a", abstained=False, latency_ms={"retrieval": 10.0})


def test_latency_ms_with_total_accepted() -> None:
    out = SystemOutput(answer="a", abstained=False, latency_ms={"total": 10.0, "retrieval": 4.0})
    assert out.latency_ms["total"] == 10.0


def test_question_id_auto_computed_when_omitted() -> None:
    q = Question(text="q", qtype="factual", difficulty="easy", provenance="synthetic")
    assert q.id != ""
    assert len(q.id) == 64


def test_question_id_depends_only_on_text_reference_answer_and_spans() -> None:
    q1 = Question(text="q", qtype="factual", difficulty="easy", provenance="synthetic")
    q2 = Question(text="q", qtype="procedural", difficulty="hard", provenance="synthetic")
    assert q1.id == q2.id


def test_question_id_changes_with_evidence_spans() -> None:
    span = EvidenceSpan(doc_id="d", chunk_id="d#0000", char_start=0, char_end=5, quote="hello")
    without = Question(text="q", qtype="factual", difficulty="easy", provenance="synthetic")
    with_span = Question(
        text="q", qtype="factual", difficulty="easy", provenance="synthetic", evidence_spans=(span,)
    )
    assert without.id != with_span.id


def test_question_explicit_id_is_preserved() -> None:
    q = Question(id="custom-id", text="q", qtype="factual", difficulty="easy", provenance="synthetic")
    assert q.id == "custom-id"


def test_models_are_frozen() -> None:
    q = Question(text="q", qtype="factual", difficulty="easy", provenance="synthetic")
    with pytest.raises(ValidationError):
        q.text = "changed"  # type: ignore[misc]


def test_evidence_span_key_excludes_page_and_quote() -> None:
    a = EvidenceSpan(doc_id="d", chunk_id="c", page=1, char_start=0, char_end=5, quote="hello")
    b = EvidenceSpan(doc_id="d", chunk_id="c", page=2, char_start=0, char_end=5, quote="different quote")
    assert a.key() == b.key() == "d:c:0:5"
