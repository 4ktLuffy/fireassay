"""The funnel report (M3-SPEC.md §4/§7) — and the reconciliation rule that
governs it:

    generated == kept + every rejection reason

A funnel that loses candidates without naming a reason is the reporting
equivalent of a check that cannot run reading as a pass (M2-SPEC.md §1's
rule, applied to reporting instead of controls). This module computes the
funnel from the full, stage-by-stage `filter_result` audit trail — three
stages recorded by `generate/` (`generation`, `not_a_question`,
`span_resolution`) and five recorded by `filter/` (`degeneracy`,
`self_containment`, `near_duplicate`, `unretrievable`, `balance`) — and
`reconciles()` is checked by `test_curate_report.py`'s required
reconciliation test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from fireassay.curate.models import Decision
from fireassay.filter.pipeline import STAGE_ORDER as _FILTER_STAGE_ORDER
from fireassay.generate.pipeline import STAGE_GENERATION, STAGE_NOT_A_QUESTION, STAGE_SPAN_RESOLUTION
from fireassay.store.db import FilterResultRow

#: Full pipeline order: generate/'s three stages, then filter/'s five.
#: `generation` (a failed LLM call, GENERATION_FAILED) comes first: it is
#: the earliest possible failure point, before there is even a raw
#: candidate to check for imperativeness.
FUNNEL_STAGE_ORDER = (STAGE_GENERATION, STAGE_NOT_A_QUESTION, STAGE_SPAN_RESOLUTION, *_FILTER_STAGE_ORDER)

_FINAL_STAGE = _FILTER_STAGE_ORDER[-1]


@dataclass(frozen=True)
class StageFunnel:
    stage: str
    kept: int
    rejected_by_reason: dict[str, int]

    @property
    def rejected(self) -> int:
        return sum(self.rejected_by_reason.values())


@dataclass(frozen=True)
class FunnelReport:
    generated: int
    filter_kept: int  # survivors of the filter pipeline (pre-curation)
    stages: tuple[StageFunnel, ...]
    accepted: int
    edited: int
    rejected: int
    reject_reason_counts: dict[str, int]

    def total_rejected_by_filter(self) -> int:
        return sum(stage.rejected for stage in self.stages)

    def reconciles(self) -> bool:
        """`generated == filter_kept + every filter-stage rejection` — the
        required funnel invariant. Curation-stage accept/edit/reject counts
        are reported separately (see `FunnelReport`'s own fields) rather
        than folded into this check: curation operates only on
        `filter_kept` survivors and a candidate can legitimately still be
        awaiting a curator decision, so `accepted + edited + rejected` is
        not expected to equal `filter_kept` at every point in time — only
        the generation/filter half of the funnel is a closed pipeline
        that must reconcile exactly at all times.
        """
        return self.generated == self.filter_kept + self.total_rejected_by_filter()


def compute_funnel(
    filter_results: Sequence[FilterResultRow], decisions: Sequence[Decision]
) -> FunnelReport:
    """Build the `FunnelReport` from every persisted `filter_result` row
    (across `generate/`'s three stages and `filter/`'s five) and the
    latest `Decision` per `(candidate_id, curator_id)`.

    `generated` is the count of distinct `candidate_id`s appearing in
    `filter_results` at all — every generation attempt gets a `generation`
    row (a failed LLM call gets one for a throwaway id with no raw
    candidate behind it; a successful one gets one per raw candidate
    returned), whether or not it survives, so this is exactly the
    funnel's starting count regardless of how many later stages a given
    candidate reached.
    """
    by_candidate: dict[str, list[FilterResultRow]] = {}
    for row in filter_results:
        by_candidate.setdefault(row.candidate_id, []).append(row)

    stages: list[StageFunnel] = []
    for stage_name in FUNNEL_STAGE_ORDER:
        kept = 0
        rejected_by_reason: Counter[str] = Counter()
        for row in filter_results:
            if row.stage != stage_name:
                continue
            if row.kept:
                kept += 1
            elif row.reason is not None:
                rejected_by_reason[row.reason] += 1
        stages.append(StageFunnel(stage=stage_name, kept=kept, rejected_by_reason=dict(rejected_by_reason)))

    generated = len(by_candidate)
    filter_kept = sum(
        1
        for candidate_id, rows in by_candidate.items()
        if any(r.stage == _FINAL_STAGE and r.kept for r in rows)
    )

    # Latest decision per (candidate_id, curator_id), then latest overall
    # per candidate_id (a candidate double-reviewed by two curators
    # contributes once per curator's own verdict to accept/edit/reject
    # counts -- each curator's decision is a distinct labelling act).
    latest_by_pair: dict[tuple[str, str], Decision] = {}
    for d in decisions:
        key = (d.candidate_id, d.curator_id)
        existing = latest_by_pair.get(key)
        if existing is None or d.decided_at > existing.decided_at:
            latest_by_pair[key] = d

    accepted = sum(1 for d in latest_by_pair.values() if d.decision == "accept")
    edited = sum(1 for d in latest_by_pair.values() if d.decision == "edit")
    rejected = sum(1 for d in latest_by_pair.values() if d.decision == "reject")
    reject_reason_counts: Counter[str] = Counter(
        d.reject_reason for d in latest_by_pair.values() if d.decision == "reject" and d.reject_reason
    )

    return FunnelReport(
        generated=generated,
        filter_kept=filter_kept,
        stages=tuple(stages),
        accepted=accepted,
        edited=edited,
        rejected=rejected,
        reject_reason_counts=dict(reject_reason_counts),
    )
