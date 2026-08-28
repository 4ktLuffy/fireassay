"""`identical_config` — runs the same config twice (two independently
built `System` instances) and checks that every *quality* metric's per-run
mean is byte-for-byte identical. M1's scoring pipeline is deterministic
end to end (pure BM25 over a fixed corpus, no LLM, no randomness), so a
nonzero delta on any of `retrieval.*`, `abstention.*`, `policy.*` is a bug,
not noise — unlike a typical mutation-testing "identical config" check, no
tolerance band is appropriate for those.

**`latency.*` is deliberately excluded** from the comparison:
`score.latency.LatencyScorer` reports genuine wall-clock timing
(`time.perf_counter()` inside `BM25System.answer`), which is *never*
bit-for-bit reproducible between two runs regardless of how deterministic
the retrieval logic itself is — comparing it here would make this control
fail on every healthy harness, for a reason that has nothing to do with
whether the harness is broken. `score/invariants.py` draws the same line
(`_RANGE_EXEMPT_PREFIXES = ("latency.", "cost.")`) for exactly this
reason. `cost.*` is left in: at zero configured token prices (the default)
it is exactly `0.0` on every question and so is still a valid determinism
check; a caller running this control with nonzero prices against a system
that reports genuinely varying token counts would need to exclude it too,
but M1's BM25 system always reports `tokens_in=tokens_out=0`, so this is
not yet a real conflict.
"""

from __future__ import annotations

from typing import Literal

from fireassay.controls._common import band_map, base_scoring_ctx, execute, or_nan
from fireassay.controls.base import ControlContext, ControlOutcome

_EXCLUDED_METRIC_PREFIXES = ("latency.",)


def _mean_by_metric(rows: list[tuple[str, str, float]]) -> dict[str, float]:
    by_metric: dict[str, list[float]] = {}
    for _question_id, metric, value in rows:
        if metric.startswith(_EXCLUDED_METRIC_PREFIXES):
            continue
        by_metric.setdefault(metric, []).append(value)
    return {metric: sum(values) / len(values) for metric, values in by_metric.items()}


class IdenticalConfigControl:
    kind = "identical_config"
    version = "1.0.0"

    def run(self, ctx: ControlContext) -> ControlOutcome:
        bands = band_map(ctx, self.kind)
        delta_band = bands["max_abs_delta"]
        scoring_ctx = base_scoring_ctx(ctx)

        system_a = ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs)
        run_a = execute(
            ctx,
            config_spec=dict(ctx.base_config.spec, _control=f"{self.kind}_a"),
            system=system_a,
            questions=ctx.questions,
            scoring_ctx=scoring_ctx,
        )
        system_b = ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs)
        run_b = execute(
            ctx,
            config_spec=dict(ctx.base_config.spec, _control=f"{self.kind}_b"),
            system=system_b,
            questions=ctx.questions,
            scoring_ctx=scoring_ctx,
        )

        # Writable twin: "silence" here (delta == 0.0) is only evidence of
        # determinism if both runs genuinely executed against the whole
        # suite — an empty run trivially has delta 0.0 for having measured
        # nothing at all.
        twin_ok = run_a.result_count > 0 and run_b.result_count > 0

        means_a = _mean_by_metric(ctx.store.run_scores(run_a.id))
        means_b = _mean_by_metric(ctx.store.run_scores(run_b.id))
        metric_sets_match = set(means_a) == set(means_b)
        common_metrics = sorted(set(means_a) & set(means_b))
        deltas = {m: abs(means_a[m] - means_b[m]) for m in common_metrics}
        max_abs_delta = max(deltas.values()) if deltas else None

        observed = {"max_abs_delta": or_nan(max_abs_delta)}
        cause_assertions = {
            "metric_sets_match": metric_sets_match,
            "all_deltas_zero": bool(deltas) and all(d == 0.0 for d in deltas.values()),
        }
        bands_ok = max_abs_delta is not None and delta_band.check(max_abs_delta)
        status: Literal["PASSED", "FAILED"] = (
            "PASSED" if (twin_ok and bands_ok and all(cause_assertions.values())) else "FAILED"
        )
        detail = (
            f"run_a.result_count={run_a.result_count}, run_b.result_count={run_b.result_count}; "
            f"{len(common_metrics)} common metric(s), max_abs_delta={max_abs_delta!r}"
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
