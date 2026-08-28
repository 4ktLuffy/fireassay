"""`fireassay curate report` (M3-SPEC.md §4) — assembles the funnel,
reviewer hours, Krippendorff's α, honeypot accuracy, autopilot/speeding
flags, content coverage and difficulty validation into one report.

This module is a pure function of already-fetched data (`build_report`
takes plain sequences, not a `Store`) so it is testable without a database
in every test file that exercises it; `cli.py`'s `curate report` command is
the only place that reads those sequences out of a `Store`.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from fireassay.curate.agreement import (
    LOW_AGREEMENT_THRESHOLD,
    AgreementResult,
    RubricCriterion,
    krippendorff_alpha,
)
from fireassay.curate.coverage import (
    CoverageReport,
    GeneratorObedience,
    content_coverage,
    difficulty_feature_correlation,
    generator_obedience,
)
from fireassay.curate.funnel import FunnelReport, compute_funnel
from fireassay.curate.honeypots import is_honeypot_correct
from fireassay.curate.models import Decision, QueueItem, RubricVerdict
from fireassay.curate.quality import (
    DEFAULT_AUTOPILOT_RUN_THRESHOLD,
    DEFAULT_SESSION_GAP_MINUTES,
    DEFAULT_SPEEDUP_FLAG_PCT,
    RECOMMENDED_MAX_SESSION_MINUTES,
    Session,
    decile_medians,
    detect_sessions,
    is_autopilot,
    is_speeding,
)
from fireassay.generate.models import ResolvedCandidate
from fireassay.store.db import FilterResultRow
from fireassay.system.corpus import Doc

_RUBRIC_CRITERIA: tuple[RubricCriterion, ...] = (
    "answerable_from_kb",
    "self_contained",
    "reference_answer_correct",
    "evidence_sufficient",
    "difficulty_agrees",
    "qtype_agrees",
)


@dataclass(frozen=True)
class SessionFlag:
    curator_id: str
    session_index: int
    n_decisions: int


@dataclass(frozen=True)
class CurateReport:
    funnel: FunnelReport
    reviewer_hours: float
    median_seconds_per_item: float | None
    agreement_by_criterion: dict[str, AgreementResult]
    low_agreement_criteria: tuple[str, ...]
    unmeasurable_criteria: tuple[str, ...]
    honeypot_accuracy_by_curator: dict[str, float | None]
    honeypot_accuracy_by_session_decile: dict[int, float | None]
    autopilot_flags: tuple[SessionFlag, ...]
    speeding_flags: tuple[SessionFlag, ...]
    coverage: CoverageReport | None
    difficulty_correlation: dict[str, float]
    generator_obedience: GeneratorObedience
    #: Queue composition, needed by the renderer to tell "no honeypots
    #: were scheduled" apart from "the curator scored zero" -- an empty
    #: `honeypot_accuracy_by_curator` is ambiguous between those two on
    #: its own (M3-SPEC.md's rule: `None` is not `0`, applied to a queue
    #: rather than a score).
    total_queue_items: int
    total_honeypot_items: int
    recommended_max_session_minutes: int = field(default=RECOMMENDED_MAX_SESSION_MINUTES)


def _latest_decision_by_pair(decisions: Sequence[Decision]) -> dict[tuple[str, str], Decision]:
    latest: dict[tuple[str, str], Decision] = {}
    for d in decisions:
        key = (d.candidate_id, d.curator_id)
        existing = latest.get(key)
        if existing is None or d.decided_at > existing.decided_at:
            latest[key] = d
    return latest


def _rubrics_by_curator(decisions: Sequence[Decision]) -> dict[str, dict[str, RubricVerdict]]:
    latest = _latest_decision_by_pair(decisions)
    by_curator: dict[str, dict[str, RubricVerdict]] = {}
    for (candidate_id, curator_id), d in latest.items():
        by_curator.setdefault(curator_id, {})[candidate_id] = d.rubric
    return by_curator


def _honeypot_accuracy_by_curator(
    queue_items: Sequence[QueueItem], decisions: Sequence[Decision]
) -> dict[str, float | None]:
    latest = _latest_decision_by_pair(decisions)
    by_curator: dict[str, list[bool]] = {}
    for item in queue_items:
        if not item.is_honeypot:
            continue
        decision = latest.get((item.candidate_id, item.curator_id))
        if decision is None:
            continue
        by_curator.setdefault(item.curator_id, []).append(is_honeypot_correct(decision))
    return {curator: (sum(v) / len(v) if v else None) for curator, v in by_curator.items()}


def _honeypot_accuracy_by_session_decile(
    sessions: Sequence[Session],
    queue_items_by_pair: dict[tuple[str, str], QueueItem],
    n_deciles: int = 10,
) -> dict[int, float | None]:
    """Pools every session's decisions into `n_deciles` time-ordered
    buckets (bucket boundaries computed per-session, so a short and a long
    session both contribute to all ten deciles) and reports honeypot
    accuracy within each bucket, across every curator/session — this is
    what makes drift within a sitting visible (M3-SPEC.md §4: "curator
    accuracy tracked over session time")."""
    buckets: dict[int, list[bool]] = {i: [] for i in range(n_deciles)}
    for session in sessions:
        n = len(session.decisions)
        if n == 0:
            continue
        for idx, decision in enumerate(session.decisions):
            decile = min(n_deciles - 1, (idx * n_deciles) // n)
            item = queue_items_by_pair.get((decision.candidate_id, decision.curator_id))
            if item is not None and item.is_honeypot:
                buckets[decile].append(is_honeypot_correct(decision))
    return {i: (sum(v) / len(v) if v else None) for i, v in buckets.items()}


def build_report(
    filter_results: Sequence[FilterResultRow],
    decisions: Sequence[Decision],
    queue_items: Sequence[QueueItem],
    candidates: Sequence[ResolvedCandidate],
    docs: Sequence[Doc] | None = None,
    *,
    session_gap_minutes: float = DEFAULT_SESSION_GAP_MINUTES,
    autopilot_run_threshold: int = DEFAULT_AUTOPILOT_RUN_THRESHOLD,
    speedup_flag_pct: float = DEFAULT_SPEEDUP_FLAG_PCT,
) -> CurateReport:
    funnel = compute_funnel(filter_results, decisions)

    latest = _latest_decision_by_pair(decisions)
    durations_ms = [d.duration_ms for d in latest.values()]
    reviewer_hours = sum(durations_ms) / 1000.0 / 3600.0
    median_seconds_per_item = (
        statistics.median(durations_ms) / 1000.0 if durations_ms else None
    )

    rubrics_by_curator = _rubrics_by_curator(decisions)
    agreement_by_criterion: dict[str, AgreementResult] = {
        criterion: krippendorff_alpha(rubrics_by_curator, criterion) for criterion in _RUBRIC_CRITERIA
    }
    # "OK but low" and "not OK at all" are different claims and must be
    # reported in different lists -- collapsing them is exactly the bug
    # this type exists to prevent (see agreement.py's module docstring).
    low_agreement_criteria = tuple(
        criterion
        for criterion, result in agreement_by_criterion.items()
        if result.status == "OK" and result.alpha is not None and result.alpha < LOW_AGREEMENT_THRESHOLD
    )
    unmeasurable_criteria = tuple(
        criterion for criterion, result in agreement_by_criterion.items() if result.status != "OK"
    )

    sessions = detect_sessions(decisions, session_gap_minutes=session_gap_minutes)
    queue_items_by_pair = {(qi.candidate_id, qi.curator_id): qi for qi in queue_items}

    autopilot_flags = tuple(
        SessionFlag(session.curator_id, i, len(session.decisions))
        for i, session in enumerate(sessions)
        if is_autopilot(session, threshold=autopilot_run_threshold)
    )
    speeding_flags = tuple(
        SessionFlag(session.curator_id, i, len(session.decisions))
        for i, session in enumerate(sessions)
        if is_speeding(session, speedup_flag_pct=speedup_flag_pct)
    )

    honeypot_accuracy_by_curator = _honeypot_accuracy_by_curator(queue_items, decisions)
    honeypot_accuracy_by_session_decile = _honeypot_accuracy_by_session_decile(
        sessions, queue_items_by_pair
    )

    accepted_or_edited = [
        d.candidate_id for d in latest.values() if d.decision in ("accept", "edit")
    ]
    candidates_by_id = {c.id: c for c in candidates}
    accepted_candidates = [
        candidates_by_id[cid] for cid in accepted_or_edited if cid in candidates_by_id
    ]
    coverage = content_coverage(accepted_candidates, docs) if docs is not None else None

    difficulty_correlation = difficulty_feature_correlation(candidates)
    obedience = generator_obedience(candidates)

    return CurateReport(
        funnel=funnel,
        reviewer_hours=reviewer_hours,
        median_seconds_per_item=median_seconds_per_item,
        agreement_by_criterion=agreement_by_criterion,
        low_agreement_criteria=low_agreement_criteria,
        unmeasurable_criteria=unmeasurable_criteria,
        honeypot_accuracy_by_curator=honeypot_accuracy_by_curator,
        honeypot_accuracy_by_session_decile=honeypot_accuracy_by_session_decile,
        autopilot_flags=autopilot_flags,
        speeding_flags=speeding_flags,
        coverage=coverage,
        difficulty_correlation=difficulty_correlation,
        generator_obedience=obedience,
        total_queue_items=len(queue_items),
        total_honeypot_items=sum(1 for qi in queue_items if qi.is_honeypot),
    )


__all__ = [
    "CurateReport",
    "SessionFlag",
    "build_report",
    "RECOMMENDED_MAX_SESSION_MINUTES",
    "decile_medians",
]
