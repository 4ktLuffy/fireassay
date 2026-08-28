"""The control framework: `ControlOutcome`, the `Control` protocol, and
`ControlContext` — the shared wiring every control in `controls/*.py`
receives.

Governing rule (M2-SPEC.md §1): **a check that cannot run must never read
as a check that passed.** `ControlOutcome.status` is one of `PASSED` /
`FAILED` / `NOT_RUN` — never a fourth, implicitly-passing value.
`admissibility.assess` (built on this module) treats `NOT_RUN` exactly
like `FAILED` unless the caller explicitly allows it.

Corollary (same section): any control whose expected result is "nothing
happens" needs independent proof that something was genuinely attempted —
`ControlOutcome.twin_ok`. `twin_ok=False` must force `status="FAILED"` in
every control's own `run` implementation; this module does not enforce
that centrally (each control's mechanism for producing a twin differs too
much to generalise), but every control below applies it, and
`tests/test_controls_no_retrieval.py` pins the rule with a dedicated test.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from fireassay.controls.expected import Band
from fireassay.models import Config, EvidenceSpan, Question, Suite
from fireassay.score.base import Scorer, ScoringContext
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.corpus import Doc

ControlStatus = Literal["PASSED", "FAILED", "NOT_RUN"]


class ControlOutcome(BaseModel):
    """The result of running one control.

    `observed` and `expected` are both flat `{metric_name: value}` maps —
    `expected` is literally the band(s) from `controls/expected.yaml` for
    this control (as plain JSON, via `Band.model_dump`), not re-derived or
    summarised, so a report can show exactly what was checked against what
    was observed. `cause_assertions` are specific, named boolean facts
    (e.g. `recall_is_zero`, `retrieved_len_is_zero`) — "assert the cause,
    not the outcome" (M2-SPEC.md §2): a control that only checked "the
    number dropped" without checking *why* is not evidence.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    status: ControlStatus
    observed: dict[str, float]
    expected: dict[str, object]
    twin_ok: bool
    cause_assertions: dict[str, bool]
    detail: str


@dataclass(frozen=True)
class ControlContext:
    """Shared wiring passed to every `Control.run`.

    `system_factory` takes `(config_spec, docs)` rather than only
    `config_spec` (contrast `runner.run_matrix`'s `system_factory`, which
    closes over one fixed document set) because `corpus_ablation` must be
    able to build a system over a *different*, ablated document set
    without mutating `base_config` at all — chunking is a config-axis
    concern, but *which documents exist* is not something `base_config`'s
    spec encodes.

    `judged_spans_by_doc` is the suite's true, unmutated judged pool
    (every evidence span from every question in the suite, grouped by
    `doc_id`) — computed once by `build_control_context`, and always used
    as-is by every control execution (see `controls._common.execute`),
    even when a control scores a mutated or partial view of the suite's
    questions.
    """

    store: Store
    suite: Suite
    base_config: Config
    questions: tuple[Question, ...]
    corpus_docs: tuple[Doc, ...]
    system_factory: Callable[[Mapping[str, object], Sequence[Doc]], System]
    scoring_ctx_factory: Callable[[Mapping[str, object]], ScoringContext]
    scorers: tuple[Scorer, ...]
    expected: Mapping[str, Mapping[str, Band]]
    judged_spans_by_doc: Mapping[str, tuple[EvidenceSpan, ...]]


def build_control_context(
    store: Store,
    suite: Suite,
    base_config: Config,
    questions: Sequence[Question],
    corpus_docs: Sequence[Doc],
    system_factory: Callable[[Mapping[str, object], Sequence[Doc]], System],
    scoring_ctx_factory: Callable[[Mapping[str, object]], ScoringContext],
    scorers: Sequence[Scorer],
    expected: Mapping[str, Mapping[str, Band]],
) -> ControlContext:
    """Build a `ControlContext`, computing the suite-wide judged pool once
    from `questions` — the same computation `runner.run_matrix` does for an
    ordinary run (see its docstring) — so every control execution shares
    one true, unmutated pool regardless of which subset/mutation of
    `questions` any individual control run happens to score.
    """
    judged: dict[str, list[EvidenceSpan]] = {}
    for q in questions:
        for span in q.evidence_spans:
            judged.setdefault(span.doc_id, []).append(span)
    judged_spans_by_doc = {doc_id: tuple(spans) for doc_id, spans in judged.items()}
    return ControlContext(
        store=store,
        suite=suite,
        base_config=base_config,
        questions=tuple(questions),
        corpus_docs=tuple(corpus_docs),
        system_factory=system_factory,
        scoring_ctx_factory=scoring_ctx_factory,
        scorers=tuple(scorers),
        expected=expected,
        judged_spans_by_doc=judged_spans_by_doc,
    )


class Control(Protocol):
    kind: str
    version: str

    def run(self, ctx: ControlContext) -> ControlOutcome: ...
