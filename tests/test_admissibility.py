from __future__ import annotations

from fireassay.admissibility import assess
from fireassay.controls.base import ControlOutcome
from fireassay.models import Run
from fireassay.score.invariants import InvariantViolation


def _make_run(admissible: bool = True) -> Run:
    return Run(
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
        admissible=admissible,
    )


def _outcome(kind: str, status: str) -> ControlOutcome:
    return ControlOutcome(
        kind=kind, status=status, observed={}, expected={}, twin_ok=(status == "PASSED"),
        cause_assertions={}, detail="",
    )


def test_all_passed_no_violations_is_admissible() -> None:
    run = _make_run()
    outcomes = [_outcome(k, "PASSED") for k in ("no_retrieval", "shuffled_gold")]
    verdict = assess(run, [], outcomes)
    assert verdict.admissible is True
    assert verdict.detail == "admissible"


def test_any_failed_control_makes_run_inadmissible() -> None:
    run = _make_run()
    outcomes = [_outcome("no_retrieval", "PASSED"), _outcome("shuffled_gold", "FAILED")]
    verdict = assess(run, [], outcomes)
    assert verdict.admissible is False
    assert verdict.failed_controls == ("shuffled_gold",)


def test_any_not_run_makes_run_inadmissible_unless_allowed() -> None:
    run = _make_run()
    outcomes = [_outcome("no_retrieval", "PASSED"), _outcome("judge_calibration", "NOT_RUN")]

    verdict_disallowed = assess(run, [], outcomes)
    assert verdict_disallowed.admissible is False
    assert verdict_disallowed.not_run_controls == ("judge_calibration",)
    assert verdict_disallowed.allowed_not_run == ()

    verdict_allowed = assess(run, [], outcomes, allow_not_run=["judge_calibration"])
    assert verdict_allowed.admissible is True
    assert verdict_allowed.allowed_not_run == ("judge_calibration",)


def test_allowed_not_run_is_recorded_on_the_verdict_even_when_other_kinds_are_not() -> None:
    run = _make_run()
    outcomes = [
        _outcome("judge_calibration", "NOT_RUN"),
        _outcome("no_retrieval", "PASSED"),
    ]
    verdict = assess(run, [], outcomes, allow_not_run=["judge_calibration", "some_other_kind"])
    assert verdict.admissible is True
    # Only kinds that actually appeared as NOT_RUN are recorded as allowed,
    # even though `allow_not_run` names a broader set -- allow_not_run is a
    # permission, not a claim about what happened.
    assert verdict.allowed_not_run == ("judge_calibration",)


def test_failed_control_beats_allow_not_run() -> None:
    """A FAILED control is never tolerated by allow_not_run -- that flag
    only ever widens what NOT_RUN is tolerated for."""
    run = _make_run()
    outcomes = [_outcome("no_retrieval", "FAILED"), _outcome("judge_calibration", "NOT_RUN")]
    verdict = assess(run, [], outcomes, allow_not_run=["no_retrieval", "judge_calibration"])
    assert verdict.admissible is False
    assert verdict.failed_controls == ("no_retrieval",)


def test_invariant_violations_make_run_inadmissible_even_with_all_controls_passed() -> None:
    run = _make_run()
    outcomes = [_outcome("no_retrieval", "PASSED")]
    violations = [InvariantViolation(rule="range_0_1", question_id="q1", detail="out of range")]
    verdict = assess(run, violations, outcomes)
    assert verdict.admissible is False
    assert verdict.invariant_violation_count == 1
