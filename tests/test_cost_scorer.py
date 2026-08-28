from __future__ import annotations

import pytest

from fireassay.models import Question, SystemOutput
from fireassay.score.base import ScoringContext
from fireassay.score.cost import CostScorer


def _question() -> Question:
    return Question(text="q", qtype="factual", difficulty="easy", provenance="synthetic")


def test_cost_per_million_token_arithmetic_worked_example() -> None:
    # 2,000,000 input tokens @ $3/M = $6.00 ; 500,000 output tokens @ $15/M = $7.50
    output = SystemOutput(
        answer="a", abstained=False, latency_ms={"total": 1.0}, tokens_in=2_000_000, tokens_out=500_000
    )
    ctx = ScoringContext(price_in_per_mtok=3.0, price_out_per_mtok=15.0)

    scores = CostScorer().score(_question(), output, ctx)
    assert len(scores) == 1
    assert scores[0].metric == "cost.usd"
    assert scores[0].value == pytest.approx(13.50)
    assert scores[0].scorer == "cost@1.0.0"


def test_cost_is_zero_when_no_tokens_spent() -> None:
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    ctx = ScoringContext(price_in_per_mtok=3.0, price_out_per_mtok=15.0)
    scores = CostScorer().score(_question(), output, ctx)
    assert scores[0].value == pytest.approx(0.0)


def test_cost_is_zero_when_prices_are_zero() -> None:
    output = SystemOutput(
        answer="a", abstained=False, latency_ms={"total": 1.0}, tokens_in=1_000_000, tokens_out=1_000_000
    )
    ctx = ScoringContext()  # default prices are 0.0
    scores = CostScorer().score(_question(), output, ctx)
    assert scores[0].value == pytest.approx(0.0)


def test_one_million_tokens_costs_exactly_the_per_mtok_price() -> None:
    output = SystemOutput(answer="a", abstained=False, latency_ms={"total": 1.0}, tokens_in=1_000_000)
    ctx = ScoringContext(price_in_per_mtok=7.25)
    scores = CostScorer().score(_question(), output, ctx)
    assert scores[0].value == pytest.approx(7.25)
