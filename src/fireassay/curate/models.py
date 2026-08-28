"""`RubricVerdict`, `Decision`, `RejectReason`, `QueueItem` (M3-SPEC.md §4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from fireassay.models import Difficulty, QType

RejectReason = Literal[
    "DUPLICATE",
    "NOT_ANSWERABLE",
    "AMBIGUOUS",
    "WRONG_REFERENCE",
    "INSUFFICIENT_EVIDENCE",
    "LEADING_QUESTION",
    "TRIVIAL",
    "OUT_OF_SCOPE",
    "PII_RISK",
]


class RubricVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    answerable_from_kb: Literal["yes", "no", "partially"]
    self_contained: bool
    reference_answer_correct: Literal["yes", "no", "incomplete"]
    evidence_sufficient: bool
    difficulty_agrees: bool
    difficulty_reassigned: Difficulty | None = None
    qtype_agrees: bool
    qtype_reassigned: QType | None = None


class Decision(BaseModel):
    """One curator's verdict on one candidate.

    `id`/`decided_at` are minted by `Store.put_decision` at insert time
    (mirroring `Run.id`/`started_at`, minted by `Store.start_run`), not by
    the caller — a `Decision` built from a submitted `verdict.json` never
    needs to supply either. Decisions are append-only: a curator changing
    their mind about a candidate is a **new** `Decision` row, never an
    edit to the old one (M3-SPEC.md §5: "a `decision` may not be
    overwritten"), so every report reads the latest per
    `(candidate_id, curator_id)`.
    """

    model_config = ConfigDict(frozen=True)

    id: str = ""
    candidate_id: str
    curator_id: str
    decision: Literal["accept", "edit", "reject"]
    reject_reason: RejectReason | None = None
    rubric: RubricVerdict
    edited_text: str | None = None
    edited_answer: str | None = None
    notes: str | None = None
    duration_ms: int
    decided_at: str = ""

    @model_validator(mode="after")
    def _reject_reason_iff_rejected(self) -> Decision:
        if self.decision == "reject" and self.reject_reason is None:
            raise ValueError("reject_reason is required when decision == 'reject'")
        if self.decision != "reject" and self.reject_reason is not None:
            raise ValueError("reject_reason must be omitted unless decision == 'reject'")
        return self


class QueueItem(BaseModel):
    """One slot in one curator's queue (M3-SPEC.md §4).

    `is_honeypot`/`honeypot_expected_reason`/`is_double_review` are
    recorded for later analysis (`honeypots.py`, `agreement.py`) but are
    deliberately **never** included in what `curate next` shows a curator
    (see `curate.serve.next_item`) — a curator who could see them would
    behave differently on flagged items, defeating the point of both
    mechanisms (M3-SPEC.md §4: "interleaved indistinguishably").
    """

    model_config = ConfigDict(frozen=True)

    id: str
    queue_id: str
    curator_id: str
    candidate_id: str
    position: int
    is_honeypot: bool
    honeypot_expected_reason: RejectReason | None
    is_double_review: bool
