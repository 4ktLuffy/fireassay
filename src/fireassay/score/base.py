"""The `Scorer` protocol and the shared context every scorer receives.

All five M1 scorers (`retrieval`, `latency`, `cost`, `policy`, `abstention`)
have `requires_llm = False` — that is the credibility anchor of M1: a
stranger with no API key can reproduce every number these scorers produce.

**Documented invariant, binding on every `Scorer` implementation:**

    A Scorer emits a Score when the metric is APPLICABLE and MEASURABLE.
    It emits nothing when the metric is NOT APPLICABLE to the question.
    It must NEVER omit a Score to represent a failure, and a consumer
    must never treat an absent Score as a pass.

This is why `RetrievalScorer` omits (rather than zeroes) a question with no
gold evidence spans — the metric genuinely does not apply — but
`AbstentionScorer` emits `abstention.wrongly_abstained` for every
*answerable* question rather than omitting anything for a system that
abstains on it: an abstention on an answerable question is a failure, not
an inapplicable case, and a failure must always be a visible Score, never
silence. A scorer that got this backwards — omitting on failure — would let
a system that abstains on everything "win" simply by having nothing scored
against it, while every metric that *does* still fire for it looks
unnaturally good because the hard, correctly-scored-badly questions quietly
dropped out of its mean.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from fireassay.models import EvidenceSpan, Question, Score, SystemOutput


class PolicyRule(BaseModel):
    """One rule from a policy rule pack (see configs/policy.example.yaml).

    `pattern` is a Python regular expression. `applies_to` selects what the
    rule is checked against: `"answer"` checks `output.answer`;
    `"retrieved"` is accepted for forward compatibility with a future
    milestone but never matches anything in M1, because `RetrievedChunk`
    (models.py) intentionally carries no chunk text — see the docstring on
    `PolicyScorer.score` for why.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    pattern: str
    applies_to: Literal["answer", "retrieved"]
    severity: str


class ScoringContext(BaseModel):
    """Shared, run-level context passed to every `Scorer.score` call.

    Values here are properties of *how the run was scored*, not of any one
    question — e.g. the same price table applies to every question in a
    run, which is why they live on a context object rather than being
    threaded through each scorer's constructor.
    """

    model_config = ConfigDict(frozen=True)

    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    top_k: int = 5  # informational; no longer drives RetrievalScorer, see eval_ks
    policy_rules: tuple[PolicyRule, ...] = ()

    #: Whether the system under test ever generates a natural-language
    #: answer (copied from `system.System.generates_answers` by
    #: `runner.run_matrix`; defaults `True` so a `ScoringContext`
    #: constructed directly, e.g. in a test, behaves like an
    #: answer-generating system unless told otherwise). `False` for a
    #: retrieval-only system like `BM25System` — see
    #: `score.abstention.AbstentionScorer` for what that suppresses and why.
    system_generates_answers: bool = True

    #: Minimum character overlap between a retrieved chunk and a gold
    #: evidence span for the chunk to count as covering that span (see
    #: `score.retrieval.RetrievalScorer`). Default 1: any overlap at all.
    overlap_min_chars: int = 1

    #: The cutoffs `RetrievalScorer` sweeps `retrieval.recall@k` and
    #: `retrieval.ndcg@k` over, so that `score.invariants.check_invariants`
    #: has more than one k per question to check monotonicity against.
    #: Each k is truncated to `len(output.retrieved)` per question (a
    #: system that only returned 3 chunks cannot be evaluated at k=10).
    eval_ks: tuple[int, ...] = (1, 3, 5, 10)

    #: Every evidence span from every question in the suite, grouped by
    #: `doc_id` — the suite-wide "judged" pool used by
    #: `retrieval.judged_fraction`. Populated once per suite by
    #: `runner.run_matrix` (evidence spans are a property of the suite,
    #: not of any one config, so this does not vary per config the way
    #: `price_in_per_mtok` etc. can). Defaults to empty so a `ScoringContext`
    #: can still be constructed directly (e.g. in tests) without a `Store`.
    judged_spans_by_doc: dict[str, tuple[EvidenceSpan, ...]] = Field(default_factory=dict)

    #: Per-question spans explicitly judged **not** relevant, keyed by
    #: `question.id` — the "N" set `retrieval.bpref` needs.
    #:
    #: M1 has no data source for this: nothing in the current pipeline
    #: marks a passage irrelevant (only `Question.evidence_spans` marks
    #: passages *relevant*), so `runner.run_matrix` never populates this
    #: field and it is always `{}` in a real M1 run — meaning
    #: `retrieval.bpref` is always `1.0` whenever anything relevant is
    #: retrieved, by construction (see `RetrievalScorer.score`). Negative
    #: judgments are expected to arrive in a later milestone via pooling
    #: (judging a sample of each config's unique top results); this field
    #: exists now, and `bpref` is implemented and tested against it now
    #: (with hand-supplied nonrelevant judgments in the test), specifically
    #: so that pooling work is a drop-in later rather than a scorer rewrite.
    judged_nonrelevant_spans_by_question: dict[str, tuple[EvidenceSpan, ...]] = Field(default_factory=dict)


class Scorer(Protocol):
    name: str
    version: str
    requires_llm: bool

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]: ...
