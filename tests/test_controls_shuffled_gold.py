from __future__ import annotations

import math
from pathlib import Path

import pytest

from fireassay.controls.base import ControlContext, build_control_context
from fireassay.controls.shuffled_gold import ShuffledGoldControl
from fireassay.models import Question, Score, SystemOutput
from fireassay.score.base import Scorer, ScoringContext
from fireassay.store.db import Store
from helpers import (
    bm25_system_factory,
    load_expected,
    load_fixture_docs,
    load_fixture_questions,
    scoring_ctx_factory,
    standard_scorers,
)


def _make_context(
    store: Store, *, salt: str | None = None, scorers: list[Scorer] | None = None
) -> ControlContext:
    """Build a `ControlContext` over the fixture corpus/questions, exactly
    like `helpers.make_control_context`, except it can optionally perturb
    `suite_hash` with one extra, non-gold-bearing "salt" question -- giving
    `test_shuffled_gold_passes_every_time_across_different_seed_sets` a way
    to exercise genuinely different primary/calibration seed sets without
    changing the real gold-bearing question set or corpus at all. Defined
    locally (not added to `tests/helpers.py`) because this need is
    specific to `shuffled_gold`'s tests.
    """
    questions = load_fixture_questions()
    if salt is not None:
        questions = [
            *questions,
            Question(
                text=f"unrelated salt question {salt}",
                qtype="ambiguous",
                difficulty="easy",
                provenance="synthetic",
            ),
        ]
    store.put_questions(questions)
    suite = store.freeze_suite("kb", "1.0.0", [q.id for q in questions])
    docs = load_fixture_docs()
    base_config = store.put_config({"top_k": 5})
    expected = load_expected()
    return build_control_context(
        store,
        suite,
        base_config,
        questions,
        docs,
        bm25_system_factory,
        scoring_ctx_factory,
        scorers if scorers is not None else standard_scorers(),
        expected,
    )


def test_shuffled_gold_self_calibrates_and_passes_on_the_real_fixture(store: Store) -> None:
    """On the real fixture (BM25 over 12 documents), chance-level recall
    under a random gold-span permutation is *not* near zero -- ~40% of
    the corpus is retrieved on every query, so a meaningless (permuted)
    gold span still overlaps something roughly a third to a half of the
    time. The control measures that chance level and spread itself
    (`chance_level`/`chance_sd`) instead of assuming a fixed band, and
    compares an *estimate* of the permuted recall (the mean of
    `m_primary` independent permutations) against a standard-error-based
    tolerance built from that measurement -- PASSED, deterministically
    (see the module docstring for why a single draw and a raw-sd
    tolerance were both measured to still carry a real false-alarm rate on
    this fixture before this criterion was adopted).
    """
    ctx = _make_context(store)
    outcome = ShuffledGoldControl().run(ctx)
    assert outcome.kind == "shuffled_gold"
    assert outcome.cause_assertions["gold_spans_were_permuted"] is True
    assert outcome.cause_assertions["question_ids_preserved"] is True
    assert outcome.cause_assertions["permuted_recall_within_tolerance"] is True
    assert outcome.twin_ok is True
    assert outcome.status == "PASSED"
    assert 0.0 < outcome.observed["chance_level"] < 1.0
    assert outcome.observed["chance_sd"] >= 0.0
    assert outcome.observed["se_diff"] >= 0.0
    assert outcome.observed["upper_tolerance"] >= outcome.observed["chance_level"]
    assert outcome.observed["m_primary"] == 5.0
    assert outcome.observed["n_seeds"] == 50.0


def test_shuffled_gold_is_deterministic_across_two_runs(store: Store, tmp_path: Path) -> None:
    """Two independent stores over the identical fixture suite (same
    suite_hash) must derive the identical permutation seeds and therefore
    the identical observed recall/chance measurements."""
    ctx1 = _make_context(store)
    outcome1 = ShuffledGoldControl().run(ctx1)

    store2 = Store(tmp_path / "second.db")
    store2.migrate()
    ctx2 = _make_context(store2)
    outcome2 = ShuffledGoldControl().run(ctx2)
    store2.close()

    assert ctx1.suite.suite_hash == ctx2.suite.suite_hash
    assert outcome1.observed == outcome2.observed
    assert outcome1.status == outcome2.status == "PASSED"


def test_shuffled_gold_passes_every_time_across_different_seed_sets() -> None:
    """Proves the false-alarm property is gone, empirically, not just by
    simulation.

    A single-draw criterion (whether against the raw sample maximum or a
    raw-sd tolerance interval at `k=3`) was measured at a **1.0%**
    false-alarm rate across 200 independently-seeded suites on this
    fixture -- 12 trials (an earlier version of this test) is far too few
    to distinguish a 1% true rate from 0% (`0.99**12 ≈ 89%` chance of
    seeing zero failures out of 12 even at a real 1% rate, i.e. that
    version of this test would have looked green either way). This uses
    200 trials, matching the sample size the false alarm was originally
    measured at, and asserts PASSED on every one.

    Each trial shares the exact same real questions/corpus/system but
    differs in one unrelated, non-gold-bearing "salt" question, giving
    each trial a distinct `suite_hash` and therefore a genuinely distinct
    primary + calibration seed set. Stores are in-memory (`:memory:`) so
    200 trials x 56 BM25 executions each stays fast -- no data needs to
    survive past each trial's assertion.
    """
    n_trials = 200
    for i in range(n_trials):
        store = Store(":memory:")
        store.migrate()
        try:
            ctx = _make_context(store, salt=f"trial-{i}")
            outcome = ShuffledGoldControl().run(ctx)
            assert outcome.status == "PASSED", (
                f"trial {i}: expected PASSED, got {outcome.status} ({outcome.detail})"
            )
        finally:
            store.close()


class _ConstantRecallScorer:
    """A scorer that always reports the same recall regardless of what
    evidence spans a question actually has -- a stand-in for a scoring
    pipeline that never responds to gold-span permutation at all. Fully
    deterministic (no dependence on which specific permutation any given
    seed produces): both the twin and every permuted run see recall
    pinned at 1.0, so the twin can never clear "measured chance + margin"
    (chance is *also* pinned at 1.0) -- `shuffled_gold` must catch this
    via `twin_ok=False`."""

    name = "retrieval"
    version = "broken"
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        if not question.evidence_spans:
            return []
        k = ctx.eval_ks[0]
        return [Score(metric=f"retrieval.recall@{k}", value=1.0, scorer=f"{self.name}@{self.version}")]


def test_shuffled_gold_fails_when_recall_never_responds_to_permutation(store: Store) -> None:
    broken_scorers: list[Scorer] = [_ConstantRecallScorer()]
    ctx = _make_context(store, scorers=broken_scorers)
    outcome = ShuffledGoldControl().run(ctx)
    assert outcome.status == "FAILED"
    assert outcome.twin_ok is False
    assert outcome.observed["chance_level"] == 1.0
    assert outcome.observed["chance_sd"] == 0.0
    # se_diff = sd_floor * sqrt(1/m_primary + 1/n_seeds), since the
    # constant scorer gives calibration sd == 0 and sd_floor (0.02) wins.
    assert outcome.observed["se_diff"] == pytest.approx(0.02 * math.sqrt(1 / 5 + 1 / 50))
    assert outcome.observed["recall_at_k"] == 1.0
