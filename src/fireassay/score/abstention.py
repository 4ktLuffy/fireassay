"""`AbstentionScorer` (`abstention@1.0.0`) — did the system abstain
correctly, and did it abstain when it should not have?

Two metrics, partitioned by qtype so every question contributes to exactly
one of them:

- `abstention.correct` — only for `qtype == "unanswerable"`.
- `abstention.wrongly_abstained` — only for `qtype != "unanswerable"`.

Neither is ever folded into `correctness` (per fireassay-SPEC.md §9) —
mixing them would hide a system that abstains its way to a good-looking
score.

Both are suppressed entirely — not applicable, not zeroed — when
`ctx.system_generates_answers` is `False` (a retrieval-only system such as
`BM25System`): see `AbstentionScorer.score` for why.
"""

from __future__ import annotations

from fireassay.models import Question, Score, SystemOutput
from fireassay.score.base import ScoringContext

_NAME = "abstention"
_VERSION = "1.0.0"


class AbstentionScorer:
    name = _NAME
    version = _VERSION
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        """Emit exactly one Score, whose metric depends on `question.qtype`
        — or none at all, for a retrieval-only system.

        If `ctx.system_generates_answers` is `False`, this returns an
        **empty list**, for both qtypes, not a Score of `0.0`/`1.0`. A
        system that structurally never generates an answer (e.g.
        `BM25System`) *always* reports `output.abstained = True` — that is
        not a per-question decision it made, it is a constant property of
        what the system is. `abstention.correct`/`abstention.wrongly_
        abstained` exist to measure a *choice* to abstain; scoring a
        system that has no such choice would either always read as
        "perfectly cautious" (`abstention.correct` pinned at 1.0) or
        "wrongly abstains on everything" (`abstention.wrongly_abstained`
        pinned at 1.0) — neither is a real signal about the system, and
        both would make a retrieval-only leaderboard entry read as if it
        had failed a test it was never trying to pass. This is the same
        "not applicable" rule `RetrievalScorer` applies to a question with
        no evidence spans (see score/base.py's documented invariant): the
        metric genuinely does not apply, so it is omitted, not zeroed.

        Otherwise: for `qtype == "unanswerable"`, `abstention.correct` =
        1.0 if `output.abstained` else 0.0. For every other qtype,
        `abstention.wrongly_abstained` = 1.0 if `output.abstained` else
        0.0 — see the module docstring for why the second metric exists
        (never omitting a real failure to abstain-correctly).
        """
        if not ctx.system_generates_answers:
            return []

        scorer_tag = f"{self.name}@{self.version}"
        value = 1.0 if output.abstained else 0.0
        if question.qtype == "unanswerable":
            return [Score(metric="abstention.correct", value=value, scorer=scorer_tag)]
        return [Score(metric="abstention.wrongly_abstained", value=value, scorer=scorer_tag)]
