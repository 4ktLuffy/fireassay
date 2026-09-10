"""`MutationDetector` (M2-SPEC.md §6): decides whether a mutant run is
distinguishable from the baseline run it was mutated from — i.e. whether
this eval setup would have caught the injected regression.

M2 ships exactly one detector, `ThresholdDetector` — a metric moved beyond
a configured threshold. This is a **naive, explicitly labelled** detector,
not a statistical test: no significance, no correction, no baseline
variance estimate. **M5 will add `GateDetector`**, using the real
statistical gate (paired bootstrap, Holm-Bonferroni — fireassay-SPEC.md
§8); the mutation score will then be recomputed under both detectors and
both reported side by side, so a reader can see how much of the naive
detector's apparent sensitivity survives once noise is accounted for.
Doing anything statistical here in M2 would be dishonest about what a
metric-threshold check actually proves.
"""

from __future__ import annotations

from typing import Protocol

from fireassay.models import Run


class ScoreSource(Protocol):
    """The one thing a detector reads: per-question scores for a run, in
    `Store.run_scores`'s `(question_id, metric, value)` shape. Declared as
    a Protocol rather than taking `Store` itself so a consumer that keeps
    its scores elsewhere (agent-assay wraps its own store in a view with
    exactly this method) can use the detector without a fireassay
    database, and so this module does not import `fireassay.store` at all
    -- the same rule `items.core` follows."""

    def run_scores(self, run_id: str) -> list[tuple[str, str, float]]: ...


class MutationDetector(Protocol):
    name: str

    def detects(self, baseline: Run, mutant: Run, store: ScoreSource) -> tuple[bool, str]:
        """Whether the mutant is distinguishable from the baseline.
        Returns `(killed, detail)` — `detail` explains the comparison that
        was made, regardless of the outcome."""
        ...


def _mean_metric(store: ScoreSource, run_id: str, metric: str) -> float | None:
    rows = store.run_scores(run_id)
    values = [v for _q, m, v in rows if m == metric]
    if not values:
        return None
    return sum(values) / len(values)


class ThresholdDetector:
    """Kills a mutant when `metric`'s mean moves beyond a fixed threshold
    relative to the baseline's mean for the same metric.

    `max_drop`/`max_rise` are absolute deltas (not percentages), at least
    one of which must be given. A `metric` missing from either run (e.g.
    every question happened to have zero gold spans, so `RetrievalScorer`
    emitted nothing) means the detector cannot compare anything —
    `detects` then returns `(False, ...)`, not `(True, ...)`: an
    undetectable comparison must never silently read as "caught it".
    """

    name = "threshold@1.0.0"

    def __init__(self, metric: str, *, max_drop: float | None = None, max_rise: float | None = None) -> None:
        if max_drop is None and max_rise is None:
            raise ValueError("ThresholdDetector requires at least one of max_drop/max_rise")
        self.metric = metric
        self.max_drop = max_drop
        self.max_rise = max_rise

    def detects(self, baseline: Run, mutant: Run, store: ScoreSource) -> tuple[bool, str]:
        baseline_mean = _mean_metric(store, baseline.id, self.metric)
        mutant_mean = _mean_metric(store, mutant.id, self.metric)
        if baseline_mean is None or mutant_mean is None:
            return False, f"{self.metric} missing on baseline or mutant run — nothing to compare"

        delta = mutant_mean - baseline_mean
        if self.max_drop is not None and delta <= -self.max_drop:
            return True, (
                f"{self.metric} dropped by {-delta:.4f} (baseline={baseline_mean:.4f}, "
                f"mutant={mutant_mean:.4f}) >= max_drop {self.max_drop}"
            )
        if self.max_rise is not None and delta >= self.max_rise:
            return True, (
                f"{self.metric} rose by {delta:.4f} (baseline={baseline_mean:.4f}, "
                f"mutant={mutant_mean:.4f}) >= max_rise {self.max_rise}"
            )
        return False, (
            f"{self.metric} delta {delta:.4f} (baseline={baseline_mean:.4f}, mutant={mutant_mean:.4f}) "
            "within threshold"
        )
