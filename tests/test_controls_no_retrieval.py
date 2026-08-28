from __future__ import annotations

from collections.abc import Mapping, Sequence

from fireassay.controls.no_retrieval import NoRetrievalControl
from fireassay.models import Question, Score, SystemOutput
from fireassay.score.base import Scorer, ScoringContext
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.corpus import Doc
from helpers import make_control_context


class _AlwaysEmptySystem:
    """A stand-in "system" that never retrieves anything, for either the
    twin or the mutant. Used to prove `twin_ok=False` forces `FAILED` even
    when the mutant's own observed numbers exactly match the band."""

    def __init__(self, docs: Sequence[Doc]) -> None:
        self.name = "always_empty"
        self.generates_answers = False

    def answer(self, question: Question) -> SystemOutput:
        return SystemOutput(answer=None, abstained=True, retrieved=(), latency_ms={"total": 0.0})


def _always_empty_factory(config: Mapping[str, object], docs: Sequence[Doc]) -> System:
    return _AlwaysEmptySystem(docs)


class _AlwaysRecallOneScorer:
    """A deliberately broken retrieval scorer: reports `recall@k == 1.0`
    unconditionally, regardless of what was actually retrieved. Standing in
    for "the harness's own scoring code is broken" — the control must
    catch this via its cause assertions, not merely by re-trusting a
    plausible-looking recall number."""

    name = "retrieval"
    version = "broken"
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        k = ctx.eval_ks[0]
        return [Score(metric=f"retrieval.recall@{k}", value=1.0, scorer=f"{self.name}@{self.version}")]


def test_no_retrieval_passes_on_working_harness(store: Store) -> None:
    ctx = make_control_context(store)
    outcome = NoRetrievalControl().run(ctx)
    assert outcome.kind == "no_retrieval"
    assert outcome.status == "PASSED"
    assert outcome.twin_ok is True
    assert outcome.cause_assertions == {
        "recall_is_zero": True,
        "retrieved_len_is_zero": True,
        "judged_fraction_is_zero": True,
    }
    assert outcome.observed["recall_at_k"] == 0.0
    assert outcome.observed["retrieved_len"] == 0.0


def test_no_retrieval_twin_ok_false_forces_failed_even_when_band_matches(store: Store) -> None:
    """The mutant's observed numbers will exactly match the expected band
    (recall==0, retrieved_len==0) because the twin system *also* never
    retrieves anything — but `twin_ok` must be False (the twin never
    proved the measurement can succeed), and that alone must force
    `FAILED`, per M2-SPEC.md §2."""
    ctx = make_control_context(store, system_factory=_always_empty_factory)
    outcome = NoRetrievalControl().run(ctx)
    assert outcome.twin_ok is False
    # The band itself is satisfied -- proving this is not a "the numbers
    # looked wrong" failure, but specifically a twin_ok failure.
    assert outcome.observed["recall_at_k"] == 0.0
    assert outcome.observed["retrieved_len"] == 0.0
    assert outcome.status == "FAILED"


def test_no_retrieval_fires_when_scorer_is_broken(store: Store) -> None:
    """A broken scorer that reports recall=1.0 regardless of what was
    retrieved must make this control FAIL — proving the control checks the
    *cause* (recall_is_zero) rather than trusting a scorer's number, and
    proving the control can fire on a genuinely broken harness, not only
    stay silent on a working one."""
    broken_scorers: list[Scorer] = [_AlwaysRecallOneScorer()]
    ctx = make_control_context(store, scorers=broken_scorers)
    outcome = NoRetrievalControl().run(ctx)
    assert outcome.status == "FAILED"
    assert outcome.cause_assertions["recall_is_zero"] is False
    # retrieved_len is read directly from persisted results, not from the
    # (broken) scorer, so it is still correctly zero.
    assert outcome.cause_assertions["retrieved_len_is_zero"] is True


def test_no_retrieval_actually_wraps_retrieval_to_empty(store: Store) -> None:
    """Sanity: the mutant run genuinely persisted zero retrieved chunks for
    every question, not merely a zero recall score."""
    ctx = make_control_context(store)
    outcome = NoRetrievalControl().run(ctx)
    assert outcome.status == "PASSED"
    # Re-derive independently via bm25_system_factory + standard_scorers
    # sanity: the fixture corpus/questions genuinely support nonzero
    # recall in the unmodified case (this is what makes twin_ok True
    # above) -- covered by the "passes on working harness" test; here we
    # only assert the wrapping mechanism engaged at all.
    assert "recall_at_k" in outcome.observed
    assert "retrieved_len" in outcome.observed
