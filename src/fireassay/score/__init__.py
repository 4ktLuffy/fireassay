"""The five M1 deterministic scorers.

Every scorer here has `requires_llm = False`: this is the credibility
anchor of M1 — a stranger with no API key can reproduce these numbers.
LLM-backed scorers (correctness, groundedness, judge calibration) are
explicitly out of scope until M4.
"""

from fireassay.score.abstention import AbstentionScorer
from fireassay.score.base import PolicyRule, Scorer, ScoringContext
from fireassay.score.cost import CostScorer
from fireassay.score.invariants import InvariantViolation, check_invariants
from fireassay.score.latency import LatencyScorer
from fireassay.score.policy import PolicyScorer, load_policy_rules
from fireassay.score.retrieval import RetrievalScorer

__all__ = [
    "AbstentionScorer",
    "CostScorer",
    "InvariantViolation",
    "LatencyScorer",
    "PolicyRule",
    "PolicyScorer",
    "RetrievalScorer",
    "Scorer",
    "ScoringContext",
    "check_invariants",
    "load_policy_rules",
]
