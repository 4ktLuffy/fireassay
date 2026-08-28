from __future__ import annotations

from fireassay.admissibility import assess
from fireassay.controls.null_questions import NullQuestionsControl
from fireassay.models import Run
from fireassay.store.db import Store
from helpers import make_control_context


def test_null_questions_always_reports_not_run(store: Store) -> None:
    """`null_questions` needs a generating system to measure abstention
    against, which M1/M2's retrieval-only BM25 system is not (see the
    module docstring for why the earlier retrieval-shaped proxy was
    ill-posed). It is registered now, exactly like `judge_calibration`,
    and always reports NOT_RUN until M4."""
    ctx = make_control_context(store)
    outcome = NullQuestionsControl().run(ctx)
    assert outcome.kind == "null_questions"
    assert outcome.status == "NOT_RUN"
    assert outcome.twin_ok is False
    assert outcome.cause_assertions == {}
    assert outcome.observed == {}
    assert "M4" in outcome.detail


def test_null_questions_not_run_makes_a_run_inadmissible_unless_allowed(store: Store) -> None:
    """Per M2-SPEC.md §1, NOT_RUN is never a pass: a run whose controls
    include null_questions' guaranteed NOT_RUN is inadmissible unless the
    caller explicitly allows it -- same treatment as judge_calibration."""
    ctx = make_control_context(store)
    outcome = NullQuestionsControl().run(ctx)

    run = Run(
        id="run1",
        suite_id="suite1",
        suite_hash="hash1",
        config_id="config1",
        config_hash="confighash1",
        env_json={},
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        status="complete",
        result_count=10,
    )

    verdict_disallowed = assess(run, [], [outcome])
    assert verdict_disallowed.admissible is False
    assert verdict_disallowed.not_run_controls == ("null_questions",)

    verdict_allowed = assess(run, [], [outcome], allow_not_run=["null_questions"])
    assert verdict_allowed.admissible is True
    assert verdict_allowed.allowed_not_run == ("null_questions",)
