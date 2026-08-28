"""`CostScorer` (`cost@1.0.0`) — token cost in USD.

Prices in `ScoringContext` (`price_in_per_mtok`, `price_out_per_mtok`) are
**per million tokens**, not per token and not per thousand tokens. This is
the single most consequential unit in the repo to get wrong: a units error
here would silently corrupt every cost number fireassay ever reports, and
nothing downstream would catch it (a cost that is 1,000,000x too small or
too large is still a plausible-looking float). State the unit everywhere
this value is read or written, not just here.
"""

from __future__ import annotations

from fireassay.models import Question, Score, SystemOutput
from fireassay.score.base import ScoringContext

_NAME = "cost"
_VERSION = "1.0.0"
_TOKENS_PER_MILLION = 1_000_000


class CostScorer:
    name = _NAME
    version = _VERSION
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        """`cost.usd = tokens_in / 1e6 * price_in_per_mtok + tokens_out / 1e6 * price_out_per_mtok`."""
        cost_usd = (
            output.tokens_in / _TOKENS_PER_MILLION * ctx.price_in_per_mtok
            + output.tokens_out / _TOKENS_PER_MILLION * ctx.price_out_per_mtok
        )
        return [Score(metric="cost.usd", value=cost_usd, scorer=f"{self.name}@{self.version}")]
