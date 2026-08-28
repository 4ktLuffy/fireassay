from __future__ import annotations

from fireassay.controls.corpus_ablation import CorpusAblationControl
from fireassay.models import Question, Score, SystemOutput
from fireassay.score.base import Scorer, ScoringContext
from fireassay.store.db import Store
from helpers import make_control_context


def test_corpus_ablation_drops_recall_by_at_least_min_drop(store: Store) -> None:
    ctx = make_control_context(store)
    outcome = CorpusAblationControl().run(ctx)
    assert outcome.kind == "corpus_ablation"
    assert outcome.status == "PASSED"
    assert outcome.twin_ok is True
    assert outcome.cause_assertions["documents_were_dropped"] is True
    assert outcome.cause_assertions["recall_dropped"] is True
    assert outcome.cause_assertions["drop_meets_min"] is True
    assert outcome.observed["recall_at_k_drop"] >= 0.15


class _ConstantRecallScorer:
    """Reports the same recall regardless of corpus content -- a stand-in
    for a scoring/system pipeline that does not actually respond to the
    ablated corpus, which this control must catch."""

    name = "retrieval"
    version = "broken"
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        if not question.evidence_spans:
            return []
        k = ctx.eval_ks[0]
        return [Score(metric=f"retrieval.recall@{k}", value=0.5, scorer=f"{self.name}@{self.version}")]


def test_corpus_ablation_fails_when_recall_does_not_respond_to_ablation(store: Store) -> None:
    broken_scorers: list[Scorer] = [_ConstantRecallScorer()]
    ctx = make_control_context(store, scorers=broken_scorers)
    outcome = CorpusAblationControl().run(ctx)
    assert outcome.status == "FAILED"
    assert outcome.cause_assertions["recall_dropped"] is False
    assert outcome.observed["recall_at_k_drop"] == 0.0
