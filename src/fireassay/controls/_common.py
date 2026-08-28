"""Shared helpers for the five deterministic controls.

Kept separate from `controls/base.py` (the public `Protocol`/model surface)
purely to avoid every control module reaching into `base.py` for both the
data shapes and the plumbing; nothing here is part of the public API.

`stable_seed`/`seed_from_suite_hash` are also imported by `mutation/run.py`
and `mutation/operators.py` — determinism seeding is one concern shared by
both packages, not something worth duplicating.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from fireassay.controls.base import ControlContext
from fireassay.controls.expected import Band
from fireassay.models import Question, Run
from fireassay.runner import run_once
from fireassay.score.base import ScoringContext
from fireassay.system.base import System


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or `None` if `values` is empty.

    Never silently returns `0.0` for an empty input: `0.0` is a real,
    meaningful recall/delta value and must stay distinguishable from
    "there was nothing to average" — the same "omit, don't zero" principle
    `RetrievalScorer` applies (see `score/base.py`'s documented invariant).
    """
    if not values:
        return None
    return sum(values) / len(values)


def or_nan(value: float | None) -> float:
    """`value`, or `float('nan')` for display when there was nothing to
    measure — used only in `ControlOutcome.observed`, which is typed
    `dict[str, float]` and so cannot hold `None` directly."""
    return value if value is not None else float("nan")


def mean_metric(ctx: ControlContext, run_id: str, metric: str) -> float | None:
    """Mean of every score recorded for `metric` in `run_id`."""
    rows = ctx.store.run_scores(run_id)
    return mean([v for _q, m, v in rows if m == metric])


def band_map(ctx: ControlContext, kind: str) -> Mapping[str, Band]:
    bands = ctx.expected.get(kind)
    if not bands:
        raise KeyError(f"controls/expected.yaml: no band registered for control kind {kind!r}")
    return bands


def base_scoring_ctx(ctx: ControlContext) -> ScoringContext:
    return ctx.scoring_ctx_factory(dict(ctx.base_config.spec))


def canonical_k(ctx: ControlContext) -> int:
    """The single retrieval cutoff every control measures against, so
    `expected.yaml`'s `recall_at_k`-shaped keys refer to one unambiguous k
    rather than a sweep across `ScoringContext.eval_ks`. Taken from
    `ScoringContext.top_k` — the config's own declared primary cutoff."""
    return base_scoring_ctx(ctx).top_k


def fixed_k_ctx(scoring_ctx: ScoringContext, k: int) -> ScoringContext:
    """`scoring_ctx` restricted to evaluate at exactly one k, so a
    control's recall/nDCG/judged_fraction observations are all keyed by
    the same, single `@{k}` metric name."""
    return scoring_ctx.model_copy(update={"eval_ks": (k,)})


def stable_seed(*parts: object) -> int:
    """A stable integer seed derived from `parts` via sha256 — deliberately
    **not** Python's builtin `hash()`, which is randomised per process for
    `str` (`PYTHONHASHSEED`, on by default since Python 3.3). Using
    `hash()` here would make a "deterministic" permutation differ between
    two separate interpreter runs while still looking correct within any
    one process — exactly the kind of self-consistency-only check that
    would not catch its own bug (a determinism test that only compares two
    calls inside one process would not have caught this).

    Each part is stringified and joined with the ASCII unit separator
    (matching `hashing.content_hash`'s convention) before hashing, so
    `stable_seed(1, "a")` and `stable_seed("1", "a")` do not collide with
    `stable_seed("1a")`.
    """
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def seed_from_suite_hash(suite_hash: str) -> int:
    """Deterministic integer seed derived from a suite's content hash, so
    every control needing a seeded permutation (`shuffled_gold`,
    `corpus_ablation`) reproduces the exact same permutation across
    repeated runs of the same suite — required by
    `test_controls_shuffled_gold.py`'s determinism assertion — without
    depending on wall-clock time or a caller-supplied seed."""
    return stable_seed(suite_hash)


def overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Character overlap between two half-open ranges `[a_start, a_end)`
    and `[b_start, b_end)`.

    Deliberately duplicated (not imported) from `score.retrieval`'s
    private `_overlap_chars`: it is a one-line, essentially-frozen
    definition, and importing a leading-underscore name across modules
    would couple `controls` to `score.retrieval`'s internals for no real
    benefit.
    """
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def execute(
    ctx: ControlContext,
    *,
    config_spec: Mapping[str, object],
    system: System,
    questions: Sequence[Question],
    scoring_ctx: ScoringContext,
) -> Run:
    """Store `config_spec` as its own config row and execute `system`
    against `questions` via `runner.run_once`, returning the sealed `Run`.

    Always scores against `ctx.judged_spans_by_doc` — the suite's true,
    unmutated judged pool — regardless of which (possibly mutated or
    partial) `questions` this particular execution scores, so a control
    that only exercises part of the suite does not also shrink or corrupt
    what `retrieval.judged_fraction` measures against.
    """
    config = ctx.store.put_config(dict(config_spec))
    return run_once(
        ctx.store,
        ctx.suite,
        config,
        system,
        list(ctx.scorers),
        scoring_ctx,
        list(questions),
        judged_spans_by_doc=ctx.judged_spans_by_doc,
    )
