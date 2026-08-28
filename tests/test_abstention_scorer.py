from __future__ import annotations

from fireassay.models import Question, SystemOutput
from fireassay.score.abstention import AbstentionScorer
from fireassay.score.base import ScoringContext


def _question(qtype: str) -> Question:
    return Question(text="q", qtype=qtype, difficulty="medium", provenance="synthetic")


def test_correct_abstention_scores_one() -> None:
    ctx = ScoringContext()
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    scores = AbstentionScorer().score(_question("unanswerable"), output, ctx)
    assert len(scores) == 1
    assert scores[0].metric == "abstention.correct"
    assert scores[0].value == 1.0
    assert scores[0].scorer == "abstention@1.0.0"


def test_failure_to_abstain_on_unanswerable_scores_zero() -> None:
    ctx = ScoringContext()
    output = SystemOutput(answer="a confident guess", abstained=False, latency_ms={"total": 1.0})
    scores = AbstentionScorer().score(_question("unanswerable"), output, ctx)
    assert len(scores) == 1
    assert scores[0].metric == "abstention.correct"
    assert scores[0].value == 0.0


def test_wrongly_abstained_emitted_for_every_non_unanswerable_qtype() -> None:
    """Load-bearing per the 'never omit to represent a failure' invariant
    (score/base.py): abstaining on an answerable question is always
    scored, never silently dropped."""
    ctx = ScoringContext()
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    for qtype in ["factual", "procedural", "comparative", "multi_hop", "ambiguous", "policy_sensitive"]:
        scores = AbstentionScorer().score(_question(qtype), output, ctx)
        assert len(scores) == 1
        assert scores[0].metric == "abstention.wrongly_abstained"
        assert scores[0].value == 1.0


def test_answering_an_answerable_question_scores_wrongly_abstained_zero() -> None:
    ctx = ScoringContext()
    output = SystemOutput(answer="a real answer", abstained=False, latency_ms={"total": 1.0})
    scores = AbstentionScorer().score(_question("factual"), output, ctx)
    assert scores[0].metric == "abstention.wrongly_abstained"
    assert scores[0].value == 0.0


def test_every_question_gets_exactly_one_of_the_two_metrics_never_both_never_neither() -> None:
    ctx = ScoringContext()
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    unanswerable_scores = AbstentionScorer().score(_question("unanswerable"), output, ctx)
    answerable_scores = AbstentionScorer().score(_question("factual"), output, ctx)
    assert {s.metric for s in unanswerable_scores} == {"abstention.correct"}
    assert {s.metric for s in answerable_scores} == {"abstention.wrongly_abstained"}


def test_abstaining_on_everything_does_not_look_better_than_answering() -> None:
    """The concrete failure mode the second metric exists to catch: a
    system that abstains on every question must not come out ahead of a
    system that actually attempts answers, once abstention on answerable
    questions is itself a scored (and visible) failure."""
    ctx = ScoringContext()

    always_abstains = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    always_answers = SystemOutput(answer="a real answer", abstained=False, latency_ms={"total": 1.0})

    answerable_questions = [_question(qt) for qt in ["factual", "procedural", "multi_hop"]]

    abstain_scores = [
        AbstentionScorer().score(q, always_abstains, ctx)[0].value for q in answerable_questions
    ]
    answer_scores = [
        AbstentionScorer().score(q, always_answers, ctx)[0].value for q in answerable_questions
    ]

    # "wrongly_abstained" mean for the always-abstaining system is 1.0
    # (worst possible); for the always-answering system it is 0.0 (best
    # possible). Lower is better here, so the abstaining system's score is
    # never mistaken for an improvement.
    assert sum(abstain_scores) / len(abstain_scores) == 1.0
    assert sum(answer_scores) / len(answer_scores) == 0.0
    assert sum(abstain_scores) > sum(answer_scores)


# -- system_generates_answers == False (retrieval-only systems) -------------


def test_no_scores_for_unanswerable_when_system_never_generates_answers() -> None:
    """A retrieval-only system (e.g. BM25System) always reports
    abstained=True -- that is a constant property of what it is, not a
    per-question choice, so it must not be scored at all, not even
    abstention.correct == 1.0 (which would look like the system is
    'perfectly cautious' when it never had a caution decision to make)."""
    ctx = ScoringContext(system_generates_answers=False)
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    assert AbstentionScorer().score(_question("unanswerable"), output, ctx) == []


def test_no_scores_for_answerable_when_system_never_generates_answers() -> None:
    """Same rule for abstention.wrongly_abstained: a retrieval-only system
    is not 'wrongly abstaining' on every answerable question, it simply
    was never trying to answer any of them."""
    ctx = ScoringContext(system_generates_answers=False)
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    assert AbstentionScorer().score(_question("factual"), output, ctx) == []


def test_system_generates_answers_true_is_the_default_and_scores_normally() -> None:
    ctx = ScoringContext()  # system_generates_answers defaults to True
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    assert AbstentionScorer().score(_question("unanswerable"), output, ctx) != []
    assert AbstentionScorer().score(_question("factual"), output, ctx) != []
