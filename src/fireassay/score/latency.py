"""`LatencyScorer` (`latency@1.0.0`) — per-question latency, raw.

Emits one Score per timed stage, per question. Percentiles (p50/p95) are
deliberately *not* computed here: a percentile is only meaningful over a
population of questions, and this scorer only ever sees one question at a
time. Aggregation into p50/p95 happens once, across a whole run, in
compare.leaderboard.
"""

from __future__ import annotations

from fireassay.models import Question, Score, SystemOutput
from fireassay.score.base import ScoringContext

_NAME = "latency"
_VERSION = "1.0.0"


class LatencyScorer:
    name = _NAME
    version = _VERSION
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        """Emit `latency.total_ms` from `output.latency_ms["total"]`
        (guaranteed present by `SystemOutput` validation), plus
        `latency.{stage}_ms` for every other key in `output.latency_ms`."""
        scorer_tag = f"{self.name}@{self.version}"
        scores = [Score(metric="latency.total_ms", value=output.latency_ms["total"], scorer=scorer_tag)]
        for stage, value in output.latency_ms.items():
            if stage == "total":
                continue
            scores.append(Score(metric=f"latency.{stage}_ms", value=value, scorer=scorer_tag))
        return scores
