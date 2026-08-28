"""M3-SPEC.md §7: a corrupted candidate's expected reason is recorded;
accuracy computed correctly; a curator who accepts a known-bad item scores
0 for it."""

from __future__ import annotations

import pytest

from fireassay.curate.honeypots import (
    choose_honeypot_reason,
    corrupt_candidate_view,
    honeypot_accuracy,
    is_honeypot_correct,
)
from fireassay.curate.models import Decision, QueueItem, RubricVerdict
from fireassay.generate.models import LexicalFeatures, ResolvedCandidate


def _candidate(candidate_id: str, source_doc_id: str = "doc1") -> ResolvedCandidate:
    return ResolvedCandidate(
        id=candidate_id,
        batch_id="b1",
        text=f"question {candidate_id}?",
        qtype="factual",
        difficulty="easy",
        reference_answer=f"answer for {candidate_id}",
        quote=f"quote for {candidate_id}",
        source_doc_id=source_doc_id,
        char_start=0,
        char_end=8,
        features=LexicalFeatures(title_overlap=0.0, quote_overlap=0.0, question_len_tokens=2),
        model_digest="digest",
        prompt_hash="prompt-hash",
        created_at="2026-01-01T00:00:00",
    )


_RUBRIC = RubricVerdict(
    answerable_from_kb="yes",
    self_contained=True,
    reference_answer_correct="yes",
    evidence_sufficient=True,
    difficulty_agrees=True,
    qtype_agrees=True,
)


def _decision(
    candidate_id: str, curator_id: str, decision: str, reject_reason: str | None = None
) -> Decision:
    return Decision(
        candidate_id=candidate_id,
        curator_id=curator_id,
        decision=decision,  # type: ignore[arg-type]
        reject_reason=reject_reason,  # type: ignore[arg-type]
        rubric=_RUBRIC,
        duration_ms=5000,
        decided_at="2026-01-01T00:00:00",
    )


def test_choose_honeypot_reason_is_deterministic() -> None:
    assert choose_honeypot_reason("c1", 42) == choose_honeypot_reason("c1", 42)


def test_choose_honeypot_reason_is_one_of_the_two_known_bad_mechanisms() -> None:
    for i in range(20):
        assert choose_honeypot_reason(f"c{i}", 42) in ("WRONG_REFERENCE", "INSUFFICIENT_EVIDENCE")


def test_corrupt_candidate_view_wrong_reference_swaps_only_the_answer() -> None:
    target = _candidate("c1")
    pool = [target, _candidate("c2"), _candidate("c3")]
    view = corrupt_candidate_view(target, pool, "WRONG_REFERENCE", seed=1)

    assert view["text"] == target.text
    assert view["quote"] == target.quote
    assert view["reference_answer"] != target.reference_answer
    assert view["reference_answer"] in (c.reference_answer for c in pool if c.id != target.id)


def test_corrupt_candidate_view_insufficient_evidence_swaps_only_the_span() -> None:
    target = _candidate("c1", source_doc_id="docA")
    other = _candidate("c2", source_doc_id="docB")
    pool = [target, other]
    view = corrupt_candidate_view(target, pool, "INSUFFICIENT_EVIDENCE", seed=1)

    assert view["text"] == target.text
    assert view["reference_answer"] == target.reference_answer
    assert view["source_doc_id"] == "docB"
    assert view["quote"] == other.quote


def test_corrupt_candidate_view_is_deterministic() -> None:
    target = _candidate("c1", source_doc_id="docA")
    pool = [target, _candidate("c2", source_doc_id="docB"), _candidate("c3", source_doc_id="docC")]
    first = corrupt_candidate_view(target, pool, "WRONG_REFERENCE", seed=7)
    second = corrupt_candidate_view(target, pool, "WRONG_REFERENCE", seed=7)
    assert first == second


def test_corrupt_candidate_view_insufficient_evidence_requires_another_document() -> None:
    target = _candidate("c1", source_doc_id="docA")
    pool = [target, _candidate("c2", source_doc_id="docA")]  # everything from the same doc
    with pytest.raises(ValueError, match="different source document"):
        corrupt_candidate_view(target, pool, "INSUFFICIENT_EVIDENCE", seed=1)


# -- accuracy ---------------------------------------------------------------


def test_is_honeypot_correct_true_only_when_rejected() -> None:
    assert is_honeypot_correct(_decision("c1", "alice", "reject", "WRONG_REFERENCE")) is True
    assert is_honeypot_correct(_decision("c1", "alice", "reject", "INSUFFICIENT_EVIDENCE")) is True


def test_curator_who_accepts_a_known_bad_item_scores_zero() -> None:
    accepted = _decision("c1", "alice", "accept")
    assert is_honeypot_correct(accepted) is False


def test_curator_who_edits_a_known_bad_item_also_scores_incorrect() -> None:
    edited = _decision("c1", "alice", "edit")
    assert is_honeypot_correct(edited) is False


def test_honeypot_accuracy_computed_correctly() -> None:
    hp1 = QueueItem(
        id="q1", queue_id="qu", curator_id="alice", candidate_id="c1", position=1,
        is_honeypot=True, honeypot_expected_reason="WRONG_REFERENCE", is_double_review=False,
    )
    hp2 = QueueItem(
        id="q2", queue_id="qu", curator_id="alice", candidate_id="c2", position=2,
        is_honeypot=True, honeypot_expected_reason="INSUFFICIENT_EVIDENCE", is_double_review=False,
    )
    real = QueueItem(
        id="q3", queue_id="qu", curator_id="alice", candidate_id="c3", position=3,
        is_honeypot=False, honeypot_expected_reason=None, is_double_review=False,
    )
    pairs = [
        (hp1, _decision("c1", "alice", "reject", "WRONG_REFERENCE")),  # correct
        (hp2, _decision("c2", "alice", "accept")),  # incorrect -- accepted a known-bad item
        (real, _decision("c3", "alice", "accept")),  # not a honeypot, ignored
    ]
    assert honeypot_accuracy(pairs) == pytest.approx(0.5)


def test_honeypot_accuracy_is_none_when_no_honeypot_seen_yet() -> None:
    real = QueueItem(
        id="q1", queue_id="qu", curator_id="alice", candidate_id="c1", position=1,
        is_honeypot=False, honeypot_expected_reason=None, is_double_review=False,
    )
    assert honeypot_accuracy([(real, _decision("c1", "alice", "accept"))]) is None
    assert honeypot_accuracy([]) is None
