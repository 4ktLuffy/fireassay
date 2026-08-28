"""`null_questions` — registered now, deferred to M4.

Always reports `status="NOT_RUN"`, on the same footing as
`judge_calibration` (`controls/judge_calibration.py`).

**Why (corrected after an implementation-time review, following the same
"the control's own proxy exposed that its measurement was ill-posed"
principle this project's thesis is about):** the property this control is
supposed to check is "a question with no answer in the corpus should make
the system *decline*" — which is a statement about **abstention**, and
requires a system that generates an answer at all. M1/M2's system under
test is `BM25System`, which is retrieval-only
(`System.generates_answers = False`) and never makes an abstain/answer
choice for anything to measure.

An earlier version of this control worked around that by defining a
control-local, retrieval-shaped proxy: "does anything retrieved for an
unanswerable question overlap *any* known-relevant span elsewhere in the
suite". That number is measurable today, but it does not actually answer
the question this control exists to ask. On the fixture corpus it reports
`recall_at_k=1.0` against an expected `eq: 0.0` — not because retrieval is
broken, but because BM25 always returns its `top_k`, and with only 12
documents those chunks trivially overlap *some* question's gold span. The
proxy measures "did retrieval return anything at all", which is always
true, and produces a confident-looking number about nothing. That is
exactly the failure mode this project's thesis (fireassay-SPEC.md §1) is
about: a measurement with no external referent.

So, like `judge_calibration`, this control is honestly `NOT_RUN` until a
generating system exists to measure abstention against (M4) — which also
makes every run inadmissible unless explicitly allowed via
`--allow-not-run null_questions`, correctly surfacing that this property
is not yet checked rather than reporting a number that looks like it is.
"""

from __future__ import annotations

from fireassay.controls.base import ControlContext, ControlOutcome


class NullQuestionsControl:
    kind = "null_questions"
    version = "0.0.0"

    def run(self, ctx: ControlContext) -> ControlOutcome:
        return ControlOutcome(
            kind=self.kind,
            status="NOT_RUN",
            observed={},
            expected={},
            twin_ok=False,
            cause_assertions={},
            detail=(
                "requires a generating system; abstention cannot be measured on a "
                "retrieval-only system. Arrives in M4."
            ),
        )
