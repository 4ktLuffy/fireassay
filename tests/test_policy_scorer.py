from __future__ import annotations

from fireassay.models import Question, SystemOutput
from fireassay.score.base import PolicyRule, ScoringContext
from fireassay.score.policy import PolicyScorer

_EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"


def _question() -> Question:
    return Question(text="q", qtype="factual", difficulty="easy", provenance="synthetic")


def _rule(rule_id: str, pattern: str, applies_to: str = "answer") -> PolicyRule:
    return PolicyRule(id=rule_id, pattern=pattern, applies_to=applies_to, severity="high")


def test_rule_fires_and_rationale_lists_the_rule_id() -> None:
    output = SystemOutput(answer="contact me at a@b.com", abstained=False, latency_ms={"total": 1.0})
    ctx = ScoringContext(policy_rules=(_rule("no_email", _EMAIL_PATTERN),))

    scores = PolicyScorer().score(_question(), output, ctx)
    assert len(scores) == 1
    assert scores[0].metric == "policy.violations"
    assert scores[0].value == 1.0
    assert scores[0].rationale == "no_email"
    assert scores[0].scorer == "policy@1.0.0"


def test_violations_counts_rules_fired_not_regex_matches() -> None:
    output = SystemOutput(
        answer="a@b.com and c@d.com and e@f.com", abstained=False, latency_ms={"total": 1.0}
    )
    ctx = ScoringContext(policy_rules=(_rule("no_email", _EMAIL_PATTERN),))

    scores = PolicyScorer().score(_question(), output, ctx)
    # Three regex matches, but one rule -> violations == 1, not 3.
    assert scores[0].value == 1.0


def test_multiple_rules_each_count_once() -> None:
    output = SystemOutput(answer="a@b.com, call 555-123-4567", abstained=False, latency_ms={"total": 1.0})
    ctx = ScoringContext(
        policy_rules=(
            _rule("no_email", _EMAIL_PATTERN),
            _rule("no_phone", r"\d{3}-\d{3}-\d{4}"),
        )
    )
    scores = PolicyScorer().score(_question(), output, ctx)
    assert scores[0].value == 2.0
    assert scores[0].rationale is not None
    assert "no_email" in scores[0].rationale
    assert "no_phone" in scores[0].rationale


def test_zero_on_none_answer() -> None:
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    ctx = ScoringContext(policy_rules=(_rule("no_email", _EMAIL_PATTERN),))

    scores = PolicyScorer().score(_question(), output, ctx)
    assert len(scores) == 1
    assert scores[0].value == 0.0
    assert scores[0].rationale is None


def test_no_rules_fire_gives_zero_and_none_rationale() -> None:
    output = SystemOutput(answer="a perfectly clean answer", abstained=False, latency_ms={"total": 1.0})
    ctx = ScoringContext(policy_rules=(_rule("no_email", _EMAIL_PATTERN),))

    scores = PolicyScorer().score(_question(), output, ctx)
    assert scores[0].value == 0.0
    assert scores[0].rationale is None


def test_no_configured_rules_never_fires() -> None:
    output = SystemOutput(answer="a@b.com", abstained=False, latency_ms={"total": 1.0})
    ctx = ScoringContext(policy_rules=())
    scores = PolicyScorer().score(_question(), output, ctx)
    assert scores[0].value == 0.0
