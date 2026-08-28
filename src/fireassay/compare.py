"""Leaderboard aggregation.

`leaderboard` is the second choke point (after `integrity.assert_comparable`,
which it calls first) every reported number passes through: it turns raw
per-question scores into per-run, per-metric summaries without ever
inventing a data point that was not actually measured.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from fireassay.integrity import ComparabilityReport, assert_comparable
from fireassay.models import Run
from fireassay.store.db import Store

_LATENCY_PREFIX = "latency."
_SPLIT_DIMENSIONS = ("qtype", "difficulty")  # + "provenance", gated separately by split_by_provenance


class MetricSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    n: int
    #: How many questions (in this run/slice) this metric COULD have
    #: applied to — see `_build_entry`'s docstring for exactly how this is
    #: computed and why it is a documented approximation, not a per-scorer
    #: ground truth. Always `n <= applicable_n`; `n < applicable_n` means
    #: this metric's mean is based on a strict subset of the population it
    #: could have covered, and MUST be surfaced by any renderer (e.g.
    #: "0.82 (n=430/512)") rather than shown as a bare mean — a metric
    #: silently measured on a shrinking subset is the single most common
    #: way an eval flatters a config.
    applicable_n: int
    mean: float
    p50: float | None = None  # latency.* only
    p95: float | None = None  # latency.* only


class RunLeaderboardEntry(BaseModel):
    """Per-run metric summaries.

    `slice` is `{}` (and there is exactly one entry per run) unless
    `leaderboard(..., split_by_provenance=True)` and/or
    `leaderboard(..., split_by=[...])` is used, in which case there is one
    entry per distinct combination of slice-dimension values actually
    present among that run's scored questions, e.g.
    `{"provenance": "synthetic", "qtype": "multi_hop"}`.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    config_hash: str
    slice: dict[str, str] = Field(default_factory=dict)
    metrics: tuple[MetricSummary, ...]


class Leaderboard(BaseModel):
    model_config = ConfigDict(frozen=True)

    report: ComparabilityReport
    entries: tuple[RunLeaderboardEntry, ...]


def _percentile_nearest_rank(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: `rank = ceil(p / 100 * n)`, 1-based,
    clamped to `[1, n]`, read off the ascending-sorted `values`.

    Chosen over an interpolated method (e.g. numpy's default 'linear')
    because a nearest-rank percentile is always one of the values actually
    observed — never an interpolated number nobody measured, which matters
    when a latency p95 might get quoted verbatim in a report.
    """
    arr = np.sort(np.asarray(values, dtype=float))
    n = int(arr.size)
    if n == 0:
        raise ValueError("cannot compute a percentile of zero values")
    rank = max(1, math.ceil(p / 100 * n))
    rank = min(rank, n)
    return float(arr[rank - 1])


def _summarize(values: Sequence[float], metric: str, applicable_n: int) -> MetricSummary:
    n = len(values)
    assert n <= applicable_n, (
        f"{metric}: n={n} exceeds applicable_n={applicable_n} — a metric cannot have been "
        "measured on more questions than were in scope for it"
    )
    mean = float(np.mean(np.asarray(values, dtype=float)))
    p50: float | None = None
    p95: float | None = None
    if metric.startswith(_LATENCY_PREFIX):
        p50 = _percentile_nearest_rank(values, 50)
        p95 = _percentile_nearest_rank(values, 95)
    return MetricSummary(metric=metric, n=n, applicable_n=applicable_n, mean=mean, p50=p50, p95=p95)


def _build_entry(
    run: Run, rows: Sequence[tuple[str, str, float]], *, slice_values: dict[str, str]
) -> RunLeaderboardEntry:
    """Aggregate one run's (question_id, metric, value) rows into a
    `RunLeaderboardEntry`.

    `applicable_n` (see `MetricSummary`) is the number of **distinct
    questions with any recorded score at all** in this run/slice — every
    question in M1 gets at least a `latency.total_ms` score (see
    `score.latency.LatencyScorer`, which never omits), so this equals the
    full question population of the run/slice. This is a deliberately
    simple, scorer-agnostic proxy: `compare.py` works from raw score rows
    and has no per-scorer knowledge of what "applicable" means for an
    arbitrary metric (e.g. retrieval's true population is "questions with
    gold spans", abstention.correct's is "questions with qtype ==
    unanswerable"), so it cannot compute the metric-exact denominator
    without either hardcoding scorer-specific rules here (coupling
    compare.py to every scorer's internals) or having scorers persist
    applicability alongside their scores (a bigger schema change, not made
    in M1). This proxy still catches the dangerous case the invariant is
    aimed at — a metric silently measured on a shrinking subset of the
    *attempted* population, e.g. a future LLM-judge `correctness` scorer
    that only fires when the system actually answered.
    """
    all_question_ids = {question_id for question_id, _metric, _value in rows}
    applicable_n = len(all_question_ids)

    by_metric: dict[str, list[float]] = {}
    for _question_id, metric, value in rows:
        by_metric.setdefault(metric, []).append(value)
    metrics = tuple(
        _summarize(values, metric, applicable_n) for metric, values in sorted(by_metric.items())
    )
    return RunLeaderboardEntry(
        run_id=run.id, config_hash=run.config_hash, slice=slice_values, metrics=metrics
    )


def leaderboard(
    store: Store,
    runs: Sequence[Run],
    *,
    force: bool = False,
    split_by_provenance: bool = False,
    split_by: Sequence[str] = (),
) -> Leaderboard:
    """Build a leaderboard over `runs`.

    Calls `integrity.assert_comparable` first — raising `ComparisonRefusedError`
    unless `force=True` — then aggregates each run's scores per metric.

    - The mean for a metric is computed only over the questions that *have*
      that metric recorded for that run; `n` reports that count, and
      `applicable_n` reports the population it could have applied to (see
      `_build_entry`). `n` will legitimately differ between metrics on the
      same run — e.g. `abstention.correct`'s `n` equals the number of
      `unanswerable` questions. A missing score is never imputed as 0; it
      is simply outside the population `n` counts.
    - `latency.*` metrics additionally get `p50`/`p95` via
      `_percentile_nearest_rank` on that run's sorted per-question values.
    - `split_by_provenance=True` and/or `split_by=("qtype", "difficulty")`
      (any subset/order of those two) produce one `RunLeaderboardEntry`
      per distinct combination of slice-dimension values actually present,
      instead of one per run — so a config that wins overall but fails
      every `multi_hop` question, or wins on synthetic questions and loses
      on real traffic, is visible rather than averaged away.

    Raises `ValueError` if `split_by` names anything other than `"qtype"`
    or `"difficulty"`.
    """
    for dim in split_by:
        if dim not in _SPLIT_DIMENSIONS:
            raise ValueError(f"split_by only supports {_SPLIT_DIMENSIONS!r}, got {dim!r}")

    scorers_by_run = {run.id: store.run_scorers(run.id) for run in runs}
    report = assert_comparable(runs, scorers_by_run=scorers_by_run, force=force)

    dims: list[str] = (["provenance"] if split_by_provenance else []) + list(split_by)

    question_meta: dict[str, dict[str, str]] = {}
    if dims and runs:
        # assert_comparable has already required a shared suite_hash
        # (or the caller forced past that refusal); any run's suite_id
        # resolves the same question -> metadata mapping.
        for q in store.iter_questions(runs[0].suite_id):
            question_meta[q.id] = {"provenance": q.provenance, "qtype": q.qtype, "difficulty": q.difficulty}

    entries: list[RunLeaderboardEntry] = []
    for run in runs:
        rows = store.run_scores(run.id)
        if dims:
            groups: dict[tuple[str, ...], list[tuple[str, str, float]]] = {}
            for question_id, metric, value in rows:
                meta = question_meta.get(question_id, {})
                key = tuple(meta.get(dim, "unknown") for dim in dims)
                groups.setdefault(key, []).append((question_id, metric, value))
            for key, group_rows in sorted(groups.items()):
                slice_values = dict(zip(dims, key, strict=True))
                entries.append(_build_entry(run, group_rows, slice_values=slice_values))
        else:
            entries.append(_build_entry(run, rows, slice_values={}))

    return Leaderboard(report=report, entries=tuple(entries))
