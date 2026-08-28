"""End-to-end mutation pipeline: baseline run -> N mutants -> detector ->
`gate_mutation_score` (M2-SPEC.md §6, §9's `fireassay mutate` command).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from fireassay.controls._common import seed_from_suite_hash
from fireassay.models import Config, EvidenceSpan, Question, Suite
from fireassay.mutation.detector import MutationDetector
from fireassay.mutation.operators import MutationOperator
from fireassay.mutation.score import MutantResult, MutationScoreResult, compute_score
from fireassay.runner import run_once
from fireassay.score.base import Scorer, ScoringContext
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.corpus import Doc


def run_mutation(
    store: Store,
    suite: Suite,
    config: Config,
    questions: Sequence[Question],
    docs: Sequence[Doc],
    system_factory: Callable[[Mapping[str, object], Sequence[Doc]], System],
    scorers: Sequence[Scorer],
    scoring_ctx: ScoringContext,
    operators: Sequence[MutationOperator],
    detector: MutationDetector,
) -> MutationScoreResult:
    """Run `config`'s system once as the baseline, then once per operator
    in `operators` (each wrapping a freshly built baseline system), check
    every mutant with `detector`, and return the aggregate
    `MutationScoreResult` (persistence is the CLI's job, via
    `Store.put_mutation_run`/`put_mutant`, mirroring how `run_once`'s
    caller — not `run_once` itself — decides what to do with a `Run`).

    `system_factory` has the same `(config_spec, docs) -> System` shape as
    `controls.base.ControlContext.system_factory`, so the CLI can build one
    closure and hand it to both `controls run` and `mutate`.

    `seed` for every operator is derived once from `suite.suite_hash` (see
    `controls._common.seed_from_suite_hash`) — the same deterministic-seed
    convention the controls use — so a mutation run is exactly reproducible
    for a given suite.
    """
    judged: dict[str, list[EvidenceSpan]] = {}
    for q in questions:
        for span in q.evidence_spans:
            judged.setdefault(span.doc_id, []).append(span)
    judged_spans_by_doc: dict[str, tuple[EvidenceSpan, ...]] = {
        doc_id: tuple(spans) for doc_id, spans in judged.items()
    }

    baseline_system = system_factory(dict(config.spec), docs)
    baseline_run = run_once(
        store, suite, config, baseline_system, list(scorers), scoring_ctx, list(questions),
        judged_spans_by_doc=judged_spans_by_doc,
    )
    baseline_retrieved = store.run_retrieved(baseline_run.id)
    baseline_lengths = [len(chunks) for chunks in baseline_retrieved.values()]

    seed = seed_from_suite_hash(suite.suite_hash)

    mutants: list[MutantResult] = []
    for operator in operators:
        equivalent, reason = operator.check_equivalence(baseline_lengths)
        mutant_system = operator.wrap(system_factory(dict(config.spec), docs), seed)
        mutant_config_spec = dict(
            config.spec, _mutation_operator=operator.name, _mutation_params=dict(operator.params)
        )
        mutant_config = store.put_config(mutant_config_spec)
        mutant_run = run_once(
            store, suite, mutant_config, mutant_system, list(scorers), scoring_ctx, list(questions),
            judged_spans_by_doc=judged_spans_by_doc,
        )
        killed, detail = detector.detects(baseline_run, mutant_run, store)
        mutants.append(
            MutantResult(
                operator=operator.name,
                params=dict(operator.params),
                mutant_run_id=mutant_run.id,
                killed=killed,
                equivalent=equivalent,
                equivalent_reason=reason if equivalent else None,
                detail=detail,
            )
        )

    killed_n, total_n, equivalent_n, score = compute_score(mutants)
    return MutationScoreResult(
        suite_id=suite.id,
        suite_hash=suite.suite_hash,
        base_config_id=config.id,
        detector=detector.name,
        killed=killed_n,
        total=total_n,
        equivalent=equivalent_n,
        score=score,
        mutants=tuple(mutants),
    )
