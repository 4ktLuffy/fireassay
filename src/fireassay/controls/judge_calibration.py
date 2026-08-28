"""`judge_calibration` — registered now, deferred to M4.

Always reports `status="NOT_RUN"`. Per M2-SPEC.md §1, `NOT_RUN` is not a
pass: a run containing this control is **inadmissible** until M4 lands and
gives it a real implementation, unless the caller explicitly passes
`judge_calibration` to `--allow-not-run` (CLI) / `allow_not_run`
(`admissibility.assess`) — which stamps `admissibility_json.allowed_not_run`
on the run and must be surfaced in every report built on it. That every run
is inadmissible-by-default until this control has a judge is intended
behaviour, not a bug to work around.
"""

from __future__ import annotations

from fireassay.controls.base import ControlContext, ControlOutcome


class JudgeCalibrationControl:
    kind = "judge_calibration"
    version = "0.0.0"

    def run(self, ctx: ControlContext) -> ControlOutcome:
        return ControlOutcome(
            kind=self.kind,
            status="NOT_RUN",
            observed={},
            expected={},
            twin_ok=False,
            cause_assertions={},
            detail="requires a judge; arrives in M4",
        )
