"""Pydantic v2 data models for fireassay.

All models are frozen (``model_config = ConfigDict(frozen=True)``): once
constructed they cannot be mutated in place. This matches the append-only
design of the store (see store/db.py) and is load-bearing for content
addressing — hashing.py assumes an id is a pure function of a model's
content forever, which a mutable model would not guarantee.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fireassay.hashing import question_id as _question_id


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    chunk_id: str
    page: int | None = None
    char_start: int
    char_end: int
    quote: str

    def key(self) -> str:
        """Stable identity of this span, used both for question-id hashing
        and as the gold-set membership key in RetrievalScorer.

        Deliberately excludes `page` and `quote`: those are display/debug
        aids that can be re-derived from (doc_id, chunk_id, char_start,
        char_end). Including them would let a purely cosmetic edit (e.g.
        fixing a typo in the quoted excerpt) change a question's content
        hash, which should only change when the actual evidence location
        changes.
        """
        return f"{self.doc_id}:{self.chunk_id}:{self.char_start}:{self.char_end}"


QType = Literal[
    "factual",
    "procedural",
    "comparative",
    "multi_hop",
    "unanswerable",
    "ambiguous",
    "policy_sensitive",
]
Difficulty = Literal["easy", "medium", "hard"]
Provenance = Literal["synthetic", "real_traffic"]


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = ""
    text: str
    qtype: QType
    difficulty: Difficulty
    reference_answer: str | None = None
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    provenance: Provenance
    generator: str | None = None  # name@version; must be None when provenance == "real_traffic"
    source_doc_id: str | None = None

    @model_validator(mode="after")
    def _validate_generator(self) -> Question:
        if self.provenance == "real_traffic" and self.generator is not None:
            raise ValueError(
                "generator must be None when provenance is 'real_traffic': "
                "real-traffic questions were captured, not generated, so "
                "attributing them to a generator would misrepresent their "
                "provenance in every report split by provenance"
            )
        return self

    def model_post_init(self, __context: object) -> None:
        """Auto-compute `id` as a content hash when the caller did not
        supply one explicitly.

        Content addressing (see hashing.py) requires `id` to be a pure
        function of (text, reference_answer, evidence spans). Computing it
        here — rather than requiring every caller that builds a Question
        from raw import data to remember to hash it themselves — is the
        difference between a contract that is enforced and one that is
        merely documented.
        """
        if self.id == "":
            computed = _question_id(
                self.text,
                self.reference_answer,
                (span.key() for span in self.evidence_spans),
            )
            # Bypass the frozen-model guard: this is the one place a
            # Question is allowed to set its own id, exactly once, during
            # construction.
            object.__setattr__(self, "id", computed)


class RetrievedChunk(BaseModel):
    """One chunk a system retrieved for a question.

    `char_start`/`char_end` (added post-M1-SPEC.md, see spec correction 1)
    are the chunk's character range within its document, copied from the
    `Chunk` the system retrieved. `RetrievalScorer` needs these to judge
    relevance by character-range overlap against gold evidence spans
    rather than by `chunk_id` equality — `chunk_id` embeds a chunking
    strategy's index (`f"{doc_id}#{index:04d}"`), so two configs with
    different chunk sizes produce disjoint chunk_id spaces even when they
    retrieve the exact same underlying text, which made recall
    incomparable across a chunking config axis.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    chunk_id: str
    score: float
    rank: int  # 1-based
    char_start: int
    char_end: int


class SystemOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str | None
    abstained: bool
    retrieved: tuple[RetrievedChunk, ...] = ()
    latency_ms: dict[str, float]  # keys: any stage names + required "total"
    tokens_in: int = 0
    tokens_out: int = 0

    @model_validator(mode="after")
    def _validate_latency_total(self) -> SystemOutput:
        if "total" not in self.latency_ms:
            raise ValueError(
                "latency_ms must include a 'total' key: every scorer and "
                "aggregation downstream (LatencyScorer, compare.leaderboard) "
                "reads output.latency_ms['total'] unconditionally"
            )
        return self


class Score(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    value: float
    scorer: str  # "name@version"
    rationale: str | None = None


class Suite(BaseModel):
    """A frozen, named, versioned, content-addressed set of questions.

    `id` is the suite's `suite_hash`: suites, like questions and configs,
    are content-addressed, so two freezes of the identical question set
    (even under different (name, version) labels) share the same `id`.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    version: str
    suite_hash: str
    frozen_at: str
    question_count: int


class Config(BaseModel):
    """A stored config spec, keyed by its content hash (`config_hash`)."""

    model_config = ConfigDict(frozen=True)

    id: str
    config_hash: str
    label: str | None
    spec: dict[str, object]


class Run(BaseModel):
    """A single execution of a system against a frozen suite under a config.

    Unlike Suite/Config, a Run is *not* content-addressed: two runs can
    share the same (suite_hash, config_hash) yet be different events (e.g.
    re-runs, or runs on different days/environments), so `id` is a random
    identifier rather than a hash. `suite_hash`/`config_hash` are
    denormalised onto the run row so integrity checks and leaderboard
    grouping never need a join back to `suite`/`config` to compare runs.

    `result_count` is likewise denormalised (computed by Store.get_run) so
    that `integrity.assert_comparable` can detect an EMPTY_RUN directly
    from the `Run` objects it is handed, without also needing a `Store` or
    a separate result-count lookup in its signature.

    `admissible` (added post-M1-SPEC.md) records whether
    `score.invariants.check_invariants` found any violation across this
    run's scores. It defaults to `True` for a run still in progress; the
    runner sets it before `finish_run` seals the run. A run with
    `admissible=False` ran to completion but produced metric values that
    are internally inconsistent (e.g. `retrieval.mrr > 0` while
    `retrieval.recall@k_max == 0`) — a bug in the metric code or the
    stored result, never a property of the system under test.
    `admissible=False` is a comparison-refusal ground in its own right:
    `integrity.assert_comparable` raises `RUN_INADMISSIBLE` for it, on the
    same footing as the other refusal codes — a run whose own numbers are
    self-contradictory must not be silently treated as comparable
    evidence.

    `admissibility_json` (added in M2, migration 0002) records the fuller
    verdict `admissibility.assess` computes once M2 controls have run
    against this run's suite/config: invariant violations plus control
    PASSED/FAILED/NOT_RUN results, folded together. It defaults to `{}`
    (the M1 shape — a run controls have never assessed) so that
    `store.get_run` and every M1 test constructing a `Run` directly keep
    working unchanged; only `store.set_admissibility` (M2) ever populates
    it, exactly once per run.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    suite_id: str
    suite_hash: str
    config_id: str
    config_hash: str
    env_json: dict[str, object]
    started_at: str
    finished_at: str | None
    status: Literal["running", "complete", "failed"]
    result_count: int = 0
    admissible: bool = True
    admissibility_json: dict[str, object] = Field(default_factory=dict)
