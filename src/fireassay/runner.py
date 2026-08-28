"""Sequential runner: expands a config matrix, executes each config's
system against a frozen suite, scores it, and persists everything.

Sequential and synchronous by design — no threads, no async. Concurrency is
explicitly M5 (fireassay-SPEC.md §15); M1's job is correctness of the spine,
not throughput.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from fireassay import __version__
from fireassay.config import MatrixSpec, expand
from fireassay.models import EvidenceSpan, Question, Run, Score, SystemOutput
from fireassay.score.base import Scorer, ScoringContext
from fireassay.score.invariants import check_invariants
from fireassay.store.db import Store
from fireassay.system.base import System

_FAILURE_THRESHOLD = 0.01
_TOKENS_PER_MILLION = 1_000_000


def _parse_suite_ref(suite_ref: str) -> tuple[str, str]:
    name, sep, version = suite_ref.partition("@")
    if not sep or not version:
        raise ValueError(f"suite ref must be 'name@version', got {suite_ref!r}")
    return name, version


def _answer_one(system: System, question: Question) -> tuple[SystemOutput, bool]:
    """Call `system.answer(question)`, returning `(output, failed)`.

    On success, `output` is exactly what the system returned (trusted to
    carry its own `latency_ms["total"]`, per `SystemOutput` validation). On
    an exception, `output` is a synthetic failed result — `answer=None`,
    `abstained=False`, no retrieval, no tokens — whose `latency_ms["total"]`
    is the wall-clock time of the failing call, since a system that raised
    produced no `SystemOutput` of its own to read a latency from.
    """
    start = time.perf_counter()
    try:
        return system.answer(question), False
    except Exception:  # noqa: BLE001 - a crashing system must not abort the whole run
        elapsed_ms = (time.perf_counter() - start) * 1000
        failed_output = SystemOutput(
            answer=None,
            abstained=False,
            retrieved=(),
            latency_ms={"total": elapsed_ms},
            tokens_in=0,
            tokens_out=0,
        )
        return failed_output, True


def _cost_usd(output: SystemOutput, ctx: ScoringContext) -> float:
    """The same per-million-token formula as `score.cost.CostScorer`,
    computed independently here because `Store.put_result` persists
    `cost_usd` as a denormalised column on `result` regardless of whether
    `CostScorer` is present in the caller's `scorers` list."""
    return (
        output.tokens_in / _TOKENS_PER_MILLION * ctx.price_in_per_mtok
        + output.tokens_out / _TOKENS_PER_MILLION * ctx.price_out_per_mtok
    )


def run_matrix(
    store: Store,
    spec: MatrixSpec,
    system_factory: Callable[[Mapping[str, object]], System],
    scorers: Sequence[Scorer],
    ctx_factory: Callable[[Mapping[str, object]], ScoringContext],
    *,
    rerun: bool = False,
    corpus_hash_factory: Callable[[], str] | None = None,
) -> list[Run]:
    """Run every config in `expand(spec)` against `spec.suite`, sequentially.

    For each expanded config dict:

    1. `store.put_config(config)`.
    2. Skip starting a new run if a **complete** run already exists for
       `(suite_hash, config_hash)`, unless `rerun=True` — this makes
       `run_matrix` idempotent and safe to call repeatedly (e.g. from CI)
       without wasting time or creating duplicate runs. The existing run
       is still included in the returned list.
    3. Otherwise: build the system once via `system_factory(config)` and
       the scoring context once via `ctx_factory(config)`, `start_run`,
       then for every question in the suite call `system.answer(question)`,
       timing it (see `_answer_one`), score the result with every scorer
       in `scorers`, and persist via `put_result`/`put_scores`.
    4. Run `score.invariants.check_invariants` over every score this run
       produced. `finish_run` with status `'complete'`, unless more than 1%
       of questions raised an exception, in which case status is
       `'failed'` (a raising question does *not* abort the run — every
       other question still gets a chance to run — it only affects the
       final status), and with `admissible=False` if any invariant
       violation was found, regardless of status.

    `corpus_hash_factory`, if given, takes **no arguments** and is called
    once (not once per config): chunking is a property of the *config*
    (already captured in `config_hash`), not of the corpus, so
    `system.corpus.corpus_hash` no longer varies by config — see its
    docstring for the full resolution of that earlier tension. Its result
    is stored at `run.env_json["affects_results"]["corpus_hash"]` on every
    run this call produces, so `integrity.assert_comparable`'s existing
    `ENV_MISMATCH` check can catch a corpus that changed underneath a
    suite, while two configs that differ only in chunking still get the
    *same* `corpus_hash` and remain comparable. Without it (the default),
    `affects_results` is empty: `run_matrix` otherwise receives only an
    opaque `System` via `system_factory` and has no principled way to
    fingerprint what is inside it.

    `ctx_factory`'s result has two fields overlaid (via `model_copy(update=
    ...)`, since `ScoringContext` is frozen) that `ctx_factory` itself
    cannot know:
    `judged_spans_by_doc` — every evidence span from every question in the
    suite, grouped by `doc_id`, the suite-wide "judged" pool
    `retrieval.judged_fraction` needs, since `ctx_factory`'s signature only
    receives one config dict, not the suite's question list; and
    `system_generates_answers` — copied from the just-built `system`'s own
    `generates_answers`, so `score.abstention.AbstentionScorer` can tell a
    retrieval-only system apart from one that chooses to abstain, since
    `ctx_factory` is called independently of `system_factory` and has no
    other way to know what kind of system it is scoring for.

    Returns every run touched by this call — newly executed and
    skip-because-already-complete alike — in `expand(spec)` order.
    """
    name, version = _parse_suite_ref(spec.suite)
    suite = store.get_suite(name, version)
    questions = list(store.iter_questions(suite.id))

    # The suite-wide "judged" pool for retrieval.judged_fraction: every
    # evidence span from every question in the suite, grouped by doc_id.
    # This is a property of the suite, not of any one config, so it is
    # computed once here rather than inside ctx_factory (whose signature
    # only receives one config dict, not the question list).
    judged_spans_lists: dict[str, list[EvidenceSpan]] = {}
    for q in questions:
        for span in q.evidence_spans:
            judged_spans_lists.setdefault(span.doc_id, []).append(span)
    judged_spans_by_doc: dict[str, tuple[EvidenceSpan, ...]] = {
        doc_id: tuple(spans) for doc_id, spans in judged_spans_lists.items()
    }

    # Computed once for the whole matrix, not once per config: the corpus's
    # identity does not depend on which config is being run (see
    # system.corpus.corpus_hash's docstring).
    corpus_hash_value: str | None = corpus_hash_factory() if corpus_hash_factory is not None else None

    runs: list[Run] = []
    for config_spec in expand(spec):
        config = store.put_config(config_spec)

        if not rerun:
            existing = store.find_complete_run(suite.suite_hash, config.config_hash)
            if existing is not None:
                runs.append(existing)
                continue

        affects_results: dict[str, object] = {}
        if corpus_hash_value is not None:
            affects_results["corpus_hash"] = corpus_hash_value
        env: dict[str, object] = {
            "python": platform.python_version(),
            "fireassay": __version__,
            "affects_results": affects_results,
        }

        run = store.start_run(suite, config, env)
        system = system_factory(config_spec)
        ctx = ctx_factory(config_spec).model_copy(
            update={
                "judged_spans_by_doc": judged_spans_by_doc,
                "system_generates_answers": system.generates_answers,
            }
        )

        failure_count = 0
        scores_by_question: dict[str, list[Score]] = {}
        for question in questions:
            output, failed = _answer_one(system, question)
            if failed:
                failure_count += 1
            store.put_result(run.id, question.id, output, _cost_usd(output, ctx))

            scores: list[Score] = []
            for scorer in scorers:
                scores.extend(scorer.score(question, output, ctx))
            if scores:
                store.put_scores(run.id, question.id, scores)
            scores_by_question[question.id] = scores

        violations = check_invariants(scores_by_question)

        total = len(questions)
        status: Literal["complete", "failed"] = (
            "failed" if total > 0 and (failure_count / total) > _FAILURE_THRESHOLD else "complete"
        )
        store.finish_run(run.id, status, admissible=not violations, invariant_violations=violations)
        runs.append(store.get_run(run.id))

    return runs
