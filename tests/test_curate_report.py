"""M3-SPEC.md §7: funnel counts reconcile -- generated == kept + every
rejection reason, with no silent losses. This is the required
reconciliation test: a funnel that loses candidates without naming a
reason is the reporting equivalent of a check that cannot run reading as
a pass.
"""

from __future__ import annotations

import pytest

from fireassay.curate.funnel import compute_funnel
from fireassay.curate.models import Decision, QueueItem, RubricVerdict
from fireassay.curate.report import build_report
from fireassay.generate.models import LexicalFeatures, ResolvedCandidate
from fireassay.store.db import FilterResultRow
from fireassay.system.corpus import Doc


def _row(candidate_id: str, stage: str, kept: bool, reason: str | None = None) -> FilterResultRow:
    return FilterResultRow(
        candidate_id=candidate_id, stage=stage, kept=kept, reason=reason, checked_at="2026-01-01T00:00:00"
    )


def test_funnel_reconciles_generated_equals_kept_plus_every_reason() -> None:
    rows = [
        # c1: survives every stage.
        _row("c1", "not_a_question", True),
        _row("c1", "span_resolution", True),
        _row("c1", "degeneracy", True),
        _row("c1", "self_containment", True),
        _row("c1", "near_duplicate", True),
        _row("c1", "generic", True),
        _row("c1", "balance", True),
        # c2: rejected immediately as NOT_A_QUESTION.
        _row("c2", "not_a_question", False, "NOT_A_QUESTION"),
        # c3: passes not_a_question, rejected at span resolution.
        _row("c3", "not_a_question", True),
        _row("c3", "span_resolution", False, "QUOTE_NOT_FOUND"),
        # c4: rejected at degeneracy.
        _row("c4", "not_a_question", True),
        _row("c4", "span_resolution", True),
        _row("c4", "degeneracy", False, "DEGENERATE"),
        # c5: passes every filter stage except balance (cell full).
        _row("c5", "not_a_question", True),
        _row("c5", "span_resolution", True),
        _row("c5", "degeneracy", True),
        _row("c5", "self_containment", True),
        _row("c5", "near_duplicate", True),
        _row("c5", "generic", True),
        _row("c5", "balance", False, "CELL_FULL"),
        # c6: rejected as QUOTE_AMBIGUOUS.
        _row("c6", "not_a_question", True),
        _row("c6", "span_resolution", False, "QUOTE_AMBIGUOUS"),
    ]

    funnel = compute_funnel(rows, decisions=[])

    assert funnel.generated == 6
    assert funnel.filter_kept == 1
    assert funnel.total_rejected_by_filter() == 5
    assert funnel.reconciles() is True

    reasons = {
        stage.stage: stage.rejected_by_reason for stage in funnel.stages if stage.rejected_by_reason
    }
    assert reasons["not_a_question"] == {"NOT_A_QUESTION": 1}
    assert reasons["span_resolution"] == {"QUOTE_NOT_FOUND": 1, "QUOTE_AMBIGUOUS": 1}
    assert reasons["degeneracy"] == {"DEGENERATE": 1}
    assert reasons["balance"] == {"CELL_FULL": 1}


def test_funnel_fails_to_reconcile_when_a_candidate_is_silently_lost() -> None:
    """A candidate that entered the pipeline but has no terminal row at
    all (neither kept nor rejected-with-a-reason at any stage) is exactly
    the bug reconciliation exists to catch."""
    rows = [
        _row("c1", "not_a_question", True),
        _row("c1", "span_resolution", True),
        # c1 vanishes here -- no degeneracy row, no reason, nothing.
    ]
    funnel = compute_funnel(rows, decisions=[])
    assert funnel.generated == 1
    assert funnel.filter_kept == 0
    assert funnel.total_rejected_by_filter() == 0
    assert funnel.reconciles() is False


def test_funnel_reconciles_on_the_empty_case() -> None:
    funnel = compute_funnel([], decisions=[])
    assert funnel.generated == 0
    assert funnel.filter_kept == 0
    assert funnel.reconciles() is True


def test_funnel_curation_counts_use_latest_decision_per_pair() -> None:
    rows = [
        _row("c1", "not_a_question", True),
        _row("c1", "span_resolution", True),
        _row("c1", "degeneracy", True),
        _row("c1", "self_containment", True),
        _row("c1", "near_duplicate", True),
        _row("c1", "generic", True),
        _row("c1", "balance", True),
    ]
    rubric = RubricVerdict(
        answerable_from_kb="yes", self_contained=True, reference_answer_correct="yes",
        evidence_sufficient=True, difficulty_agrees=True, qtype_agrees=True,
    )
    first = Decision(
        id="d1", candidate_id="c1", curator_id="alice", decision="reject", reject_reason="TRIVIAL",
        rubric=rubric, duration_ms=1000, decided_at="2026-01-01T00:00:00",
    )
    changed_mind = Decision(
        id="d2", candidate_id="c1", curator_id="alice", decision="accept", rubric=rubric,
        duration_ms=1000, decided_at="2026-01-01T00:05:00",
    )
    funnel = compute_funnel(rows, decisions=[first, changed_mind])
    assert funnel.accepted == 1
    assert funnel.rejected == 0
    assert funnel.reject_reason_counts == {}


# -- build_report -----------------------------------------------------------


def _candidate(
    candidate_id: str, source_doc_id: str, qtype: str = "factual", difficulty: str = "easy"
) -> ResolvedCandidate:
    return ResolvedCandidate(
        id=candidate_id,
        batch_id="b1",
        text=f"question about {source_doc_id}?",
        qtype=qtype,  # type: ignore[arg-type]
        difficulty=difficulty,  # type: ignore[arg-type]
        reference_answer="ans",
        quote="q",
        source_doc_id=source_doc_id, char_start=0, char_end=1,
        features=LexicalFeatures(title_overlap=0.2, quote_overlap=0.3, question_len_tokens=5),
        model_digest="digest", prompt_hash="prompt", created_at="2026-01-01T00:00:00",
    )


def _rubric() -> RubricVerdict:
    return RubricVerdict(
        answerable_from_kb="yes", self_contained=True, reference_answer_correct="yes",
        evidence_sufficient=True, difficulty_agrees=True, qtype_agrees=True,
    )


def test_build_report_computes_reviewer_hours_and_median_seconds() -> None:
    decisions = [
        Decision(
            id=f"d{i}", candidate_id=f"c{i}", curator_id="alice", decision="accept", rubric=_rubric(),
            duration_ms=duration, decided_at=f"2026-01-01T00:0{i}:00",
        )
        for i, duration in enumerate([1000, 2000, 3000])
    ]
    candidates = [_candidate(f"c{i}", "doc1") for i in range(3)]
    report = build_report(filter_results=[], decisions=decisions, queue_items=[], candidates=candidates)

    assert report.reviewer_hours == pytest.approx(6000 / 1000 / 3600)
    assert report.median_seconds_per_item == pytest.approx(2.0)


def test_build_report_content_coverage_requires_corpus() -> None:
    candidates = [_candidate("c1", "doc1")]
    report_without_corpus = build_report([], [], [], candidates, docs=None)
    assert report_without_corpus.coverage is None

    docs = [Doc(doc_id="doc1", title="t1", text="hello"), Doc(doc_id="doc2", title="t2", text="world")]
    decisions = [
        Decision(
            id="d1", candidate_id="c1", curator_id="alice", decision="accept", rubric=_rubric(),
            duration_ms=1000, decided_at="2026-01-01T00:00:00",
        )
    ]
    report_with_corpus = build_report([], decisions, [], candidates, docs=docs)
    assert report_with_corpus.coverage is not None
    assert report_with_corpus.coverage.documents_with_questions == 1
    assert report_with_corpus.coverage.total_documents == 2
    assert report_with_corpus.coverage.doc_coverage_fraction == pytest.approx(0.5)


def test_build_report_honeypot_accuracy_by_curator() -> None:
    candidates = [_candidate("c1", "doc1"), _candidate("c2", "doc1")]
    queue_items = [
        QueueItem(
            id="q1", queue_id="qu", curator_id="alice", candidate_id="c1", position=1,
            is_honeypot=True, honeypot_expected_reason="WRONG_REFERENCE", is_double_review=False,
        ),
        QueueItem(
            id="q2", queue_id="qu", curator_id="alice", candidate_id="c2", position=2,
            is_honeypot=False, honeypot_expected_reason=None, is_double_review=False,
        ),
    ]
    decisions = [
        Decision(
            id="d1", candidate_id="c1", curator_id="alice", decision="reject",
            reject_reason="WRONG_REFERENCE",
            rubric=_rubric(), duration_ms=1000, decided_at="2026-01-01T00:00:00",
        ),
        Decision(
            id="d2", candidate_id="c2", curator_id="alice", decision="accept", rubric=_rubric(),
            duration_ms=1000, decided_at="2026-01-01T00:01:00",
        ),
    ]
    report = build_report([], decisions, queue_items, candidates)
    assert report.honeypot_accuracy_by_curator["alice"] == pytest.approx(1.0)


# -- agreement status: unmeasurable vs low-agreement must never collapse ----


def _rubric_with(self_contained: bool = True, ref_correct: str = "yes") -> RubricVerdict:
    return RubricVerdict(
        answerable_from_kb="yes",
        self_contained=self_contained,
        reference_answer_correct=ref_correct,  # type: ignore[arg-type]
        evidence_sufficient=True,
        difficulty_agrees=True,
        qtype_agrees=True,
    )


def test_build_report_flags_unmeasurable_agreement_not_silently_absent() -> None:
    """Two curators agreeing on every shared item makes
    `krippendorff.alpha` raise internally -- the report must surface that
    as UNMEASURABLE and list the criterion as flagged, never omit it or
    let it read as "agreement is fine"."""
    candidates = [_candidate("c1", "doc1"), _candidate("c2", "doc1")]
    decisions = [
        Decision(
            id="d1", candidate_id="c1", curator_id="alice", decision="accept", rubric=_rubric(),
            duration_ms=1000, decided_at="2026-01-01T00:00:00",
        ),
        Decision(
            id="d2", candidate_id="c2", curator_id="alice", decision="accept", rubric=_rubric(),
            duration_ms=1000, decided_at="2026-01-01T00:01:00",
        ),
        Decision(
            id="d3", candidate_id="c1", curator_id="bob", decision="accept", rubric=_rubric(),
            duration_ms=1000, decided_at="2026-01-01T00:02:00",
        ),
        Decision(
            id="d4", candidate_id="c2", curator_id="bob", decision="accept", rubric=_rubric(),
            duration_ms=1000, decided_at="2026-01-01T00:03:00",
        ),
    ]

    report = build_report([], decisions, [], candidates)

    result = report.agreement_by_criterion["self_contained"]
    assert result.status == "UNMEASURABLE"
    assert "self_contained" in report.unmeasurable_criteria
    assert "self_contained" not in report.low_agreement_criteria


def test_build_report_flags_genuine_disagreement_as_low_not_unmeasurable() -> None:
    """A real, measured disagreement (the hand-computed alpha=-0.5 case
    from test_agreement.py) must produce status OK and land in
    low_agreement_criteria -- never in unmeasurable_criteria."""
    candidates = [_candidate("c1", "doc1"), _candidate("c2", "doc1")]
    decisions = [
        Decision(
            id="d1", candidate_id="c1", curator_id="alice", decision="accept",
            rubric=_rubric_with(self_contained=False), duration_ms=1000, decided_at="2026-01-01T00:00:00",
        ),
        Decision(
            id="d2", candidate_id="c2", curator_id="alice", decision="accept",
            rubric=_rubric_with(self_contained=True), duration_ms=1000, decided_at="2026-01-01T00:01:00",
        ),
        Decision(
            id="d3", candidate_id="c1", curator_id="bob", decision="accept",
            rubric=_rubric_with(self_contained=True), duration_ms=1000, decided_at="2026-01-01T00:02:00",
        ),
        Decision(
            id="d4", candidate_id="c2", curator_id="bob", decision="accept",
            rubric=_rubric_with(self_contained=False), duration_ms=1000, decided_at="2026-01-01T00:03:00",
        ),
    ]

    report = build_report([], decisions, [], candidates)

    result = report.agreement_by_criterion["self_contained"]
    assert result.status == "OK"
    assert result.alpha is not None
    assert result.alpha == pytest.approx(-0.5, abs=1e-6)
    assert "self_contained" in report.low_agreement_criteria
    assert "self_contained" not in report.unmeasurable_criteria
