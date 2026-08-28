"""`run_filter` — the ordered filter pipeline (M3-SPEC.md §3).

Each candidate is evaluated stage by stage, **in the fixed order** given in
M3-SPEC.md §3's table (degeneracy -> self_containment -> near_duplicate ->
unretrievable -> balance); a rejection at any stage stops that candidate's
progress through the rest — it is never double-counted against a later
stage. A `StageResult` row is recorded for *every* stage a candidate
reaches, whether it passes (`kept=True, reason=None`) or is rejected
(`kept=False, reason=<CODE>`), so a survivor has exactly one `StageResult`
per stage in the pipeline and a rejected candidate has exactly one
`StageResult` per stage up to and including the one that rejected it. This
is what lets the funnel reconcile exactly: summing `kept=False` counts by
reason, across every stage (including generate/'s own `generation`/
`not_a_question`/`span_resolution` stages), plus the final survivor
count, equals the number of candidates the pipeline was handed.

Candidates are processed in `(created_at, id)` order — the order they were
generated in — so "keeping the earliest" (near-duplicate, balance) means
exactly what it says.

`unretrievable` (`filter.stages.check_unretrievable`) needs a live BM25
index over the whole corpus, built **once per `run_filter` call**, not
once per candidate — `docs` must be non-empty for this pipeline to mean
anything, and `run_filter` raises `ValueError` immediately, rather than
silently rejecting every candidate against a degenerate empty index, if it
is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fireassay.filter.config import FilterConfig
from fireassay.filter.stages import (
    check_degeneracy,
    check_near_duplicate,
    check_self_containment,
    check_unretrievable,
)
from fireassay.generate.models import ResolvedCandidate
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Doc, chunk_corpus
from fireassay.text import tokenize

STAGE_DEGENERACY = "degeneracy"
STAGE_SELF_CONTAINMENT = "self_containment"
STAGE_NEAR_DUPLICATE = "near_duplicate"
STAGE_UNRETRIEVABLE = "unretrievable"
STAGE_BALANCE = "balance"

#: Fixed pipeline order (M3-SPEC.md §3's table, minus the two
#: generate/-owned span-resolution reason codes).
STAGE_ORDER = (
    STAGE_DEGENERACY,
    STAGE_SELF_CONTAINMENT,
    STAGE_NEAR_DUPLICATE,
    STAGE_UNRETRIEVABLE,
    STAGE_BALANCE,
)

#: Chunking used to build the filter's own BM25 index — matches the
#: defaults used elsewhere in this codebase for chunking a corpus (`run`,
#: `controls run`, and `generate.pipeline`'s independent, generation-time
#: index). Not (yet) exposed via `FilterConfig`; only `unretrievable_top_n`
#: is (see that field's docstring).
_RETRIEVAL_CHUNK_SIZE = 800
_RETRIEVAL_CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class StageResult:
    candidate_id: str
    stage: str
    kept: bool
    reason: str | None


@dataclass(frozen=True)
class FilterRunResult:
    kept: tuple[ResolvedCandidate, ...]
    stage_results: tuple[StageResult, ...]


def _build_retriever(docs: Sequence[Doc], top_k: int) -> BM25System:
    chunks = list(chunk_corpus(docs, size=_RETRIEVAL_CHUNK_SIZE, overlap=_RETRIEVAL_CHUNK_OVERLAP))
    return BM25System(chunks, top_k=top_k)


def run_filter(
    candidates: Sequence[ResolvedCandidate], docs: Sequence[Doc], config: FilterConfig
) -> FilterRunResult:
    """Run every candidate in `candidates` through the ordered filter
    pipeline, returning the survivors and the full, stage-by-stage audit
    trail (`stage_results`) needed to persist `filter_result` rows and to
    reconcile the funnel.

    Raises `ValueError` if `docs` is empty: the `unretrievable` stage's
    BM25 index would otherwise build over zero chunks and reject every
    candidate by construction, which is a broken run silently producing
    output that looks like a real (if harsh) filtering decision — exactly
    the "check that cannot run must never read as a pass" failure mode
    this project exists to rule out, applied to a corpus precondition
    instead of a control.
    """
    if not docs:
        raise ValueError(
            "run_filter requires a non-empty corpus: the unretrievable stage builds a BM25 "
            "index over it, and an empty corpus would silently reject every candidate rather "
            "than fail clearly"
        )

    retriever = _build_retriever(docs, top_k=config.unretrievable_top_n)
    ordered = sorted(candidates, key=lambda c: (c.created_at, c.id))

    stage_results: list[StageResult] = []
    kept: list[ResolvedCandidate] = []
    kept_token_sets: list[frozenset[str]] = []
    cell_counts: dict[tuple[str, str], int] = {}

    for candidate in ordered:
        ok, reason = check_degeneracy(candidate, config)
        stage_results.append(StageResult(candidate.id, STAGE_DEGENERACY, ok, reason))
        if not ok:
            continue

        ok, reason = check_self_containment(candidate)
        stage_results.append(StageResult(candidate.id, STAGE_SELF_CONTAINMENT, ok, reason))
        if not ok:
            continue

        ok, reason = check_near_duplicate(candidate, kept_token_sets, config.near_dup_jaccard_threshold)
        stage_results.append(StageResult(candidate.id, STAGE_NEAR_DUPLICATE, ok, reason))
        if not ok:
            continue

        ok, reason = check_unretrievable(candidate, retriever, config.unretrievable_top_n)
        stage_results.append(StageResult(candidate.id, STAGE_UNRETRIEVABLE, ok, reason))
        if not ok:
            continue

        cell = (candidate.qtype, candidate.difficulty)
        count = cell_counts.get(cell, 0)
        if count >= config.cell_cap:
            stage_results.append(StageResult(candidate.id, STAGE_BALANCE, False, "CELL_FULL"))
            continue
        cell_counts[cell] = count + 1
        stage_results.append(StageResult(candidate.id, STAGE_BALANCE, True, None))
        kept.append(candidate)
        kept_token_sets.append(frozenset(tokenize(candidate.text)))

    return FilterRunResult(kept=tuple(kept), stage_results=tuple(stage_results))
