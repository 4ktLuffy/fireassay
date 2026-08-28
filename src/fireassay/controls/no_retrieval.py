"""`no_retrieval` — wraps the system so it always retrieves nothing, and
checks that every retrieval metric collapses to its floor **for the reason
we expect** (nothing was retrieved), not merely that a number went to zero.

Writable twin: the unmodified config must itself score `recall@k > 0` on
at least one question first. If it does not, the corpus/suite pairing is
broken and this control's silence on the mutant tells us nothing about the
harness (M2-SPEC.md §2) — `twin_ok=False` forces `status="FAILED"` even
when the mutant's observed numbers exactly match the expected band.
"""

from __future__ import annotations

from typing import Literal

from fireassay.controls._common import (
    band_map,
    base_scoring_ctx,
    canonical_k,
    execute,
    fixed_k_ctx,
    mean,
    mean_metric,
    or_nan,
)
from fireassay.controls.base import ControlContext, ControlOutcome
from fireassay.models import Question, SystemOutput
from fireassay.system.base import System


class _EmptyRetrievalSystem:
    """Wraps a `System` so `answer` always reports `retrieved=()`;
    everything else (`answer`/`abstained`/`latency_ms`/tokens) is passed
    through unchanged. This is the entire `no_retrieval` mechanism: the
    inner system still runs and spends whatever time it spends, but its
    retrieved-chunk list is discarded before scoring ever sees it."""

    def __init__(self, inner: System) -> None:
        self._inner = inner
        self.name = f"no_retrieval({inner.name})"
        self.generates_answers = inner.generates_answers

    def answer(self, question: Question) -> SystemOutput:
        real = self._inner.answer(question)
        return real.model_copy(update={"retrieved": ()})


class NoRetrievalControl:
    kind = "no_retrieval"
    version = "1.0.0"

    def run(self, ctx: ControlContext) -> ControlOutcome:
        bands = band_map(ctx, self.kind)
        k = canonical_k(ctx)
        scoring_ctx = fixed_k_ctx(base_scoring_ctx(ctx), k)

        # Writable twin: prove the measurement genuinely works unmodified
        # before trusting its silence on the mutant.
        twin_system = ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs)
        twin_run = execute(
            ctx,
            config_spec=dict(ctx.base_config.spec, _control=f"{self.kind}_twin"),
            system=twin_system,
            questions=ctx.questions,
            scoring_ctx=scoring_ctx,
        )
        twin_recall = mean_metric(ctx, twin_run.id, f"retrieval.recall@{k}")
        twin_ok = twin_recall is not None and twin_recall > 0.0

        mutant_system = _EmptyRetrievalSystem(ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs))
        mutant_run = execute(
            ctx,
            config_spec=dict(ctx.base_config.spec, _control=self.kind),
            system=mutant_system,
            questions=ctx.questions,
            scoring_ctx=scoring_ctx,
        )

        rows = ctx.store.run_scores(mutant_run.id)
        recall_mean = mean([v for _q, m, v in rows if m == f"retrieval.recall@{k}"])
        judged_mean = mean([v for _q, m, v in rows if m == f"retrieval.judged_fraction@{k}"])
        retrieved_by_question = ctx.store.run_retrieved(mutant_run.id)
        retrieved_lens = [len(chunks) for chunks in retrieved_by_question.values()]
        retrieved_len_mean = mean([float(n) for n in retrieved_lens])

        observed = {
            "recall_at_k": or_nan(recall_mean),
            "retrieved_len": or_nan(retrieved_len_mean),
        }
        cause_assertions = {
            "recall_is_zero": recall_mean is not None and recall_mean == 0.0,
            "retrieved_len_is_zero": len(retrieved_lens) > 0 and all(n == 0 for n in retrieved_lens),
            "judged_fraction_is_zero": judged_mean is not None and judged_mean == 0.0,
        }
        bands_ok = (
            recall_mean is not None
            and bands["recall_at_k"].check(recall_mean)
            and retrieved_len_mean is not None
            and bands["retrieved_len"].check(retrieved_len_mean)
        )

        status: Literal["PASSED", "FAILED"] = (
            "PASSED" if (twin_ok and bands_ok and all(cause_assertions.values())) else "FAILED"
        )
        detail = (
            f"twin retrieval.recall@{k}={twin_recall!r} (twin_ok={twin_ok}); "
            f"mutant retrieval.recall@{k}={recall_mean!r}, retrieved_len={retrieved_len_mean!r}, "
            f"retrieval.judged_fraction@{k}={judged_mean!r}"
        )
        return ControlOutcome(
            kind=self.kind,
            status=status,
            observed=observed,
            expected={name: b.model_dump(exclude_none=True) for name, b in bands.items()},
            twin_ok=twin_ok,
            cause_assertions=cause_assertions,
            detail=detail,
        )
