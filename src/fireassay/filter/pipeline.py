"""`run_filter` — the ordered filter pipeline (M3-SPEC.md §3).

Each candidate is evaluated stage by stage, **in the fixed order** given in
M3-SPEC.md §3's table (degeneracy -> self_containment -> near_duplicate ->
generic -> balance); a rejection at any stage stops that candidate's
progress through the rest — it is never double-counted against a later
stage. A `StageResult` row is recorded for *every* stage a candidate
reaches, whether it passes (`kept=True, reason=None`) or is rejected
(`kept=False, reason=<CODE>`), so a survivor has exactly one `StageResult`
per stage in the pipeline and a rejected candidate has exactly one
`StageResult` per stage up to and including the one that rejected it. This
is what lets the funnel reconcile exactly: summing `kept=False` counts by
reason, across every stage (including generate/'s own
`not_a_question`/`span_resolution` stages), plus the final survivor count,
equals the number of candidates the pipeline was handed.

Candidates are processed in `(created_at, id)` order — the order they were
generated in — so "keeping the earliest" (near-duplicate, balance) means
exactly what it says.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fireassay.filter.config import FilterConfig
from fireassay.filter.stages import (
    check_degeneracy,
    check_generic,
    check_near_duplicate,
    check_self_containment,
)
from fireassay.generate.models import ResolvedCandidate
from fireassay.system.corpus import Doc
from fireassay.text import tokenize

STAGE_DEGENERACY = "degeneracy"
STAGE_SELF_CONTAINMENT = "self_containment"
STAGE_NEAR_DUPLICATE = "near_duplicate"
STAGE_GENERIC = "generic"
STAGE_BALANCE = "balance"

#: Fixed pipeline order (M3-SPEC.md §3's table, minus the two
#: generate/-owned span-resolution reason codes).
STAGE_ORDER = (STAGE_DEGENERACY, STAGE_SELF_CONTAINMENT, STAGE_NEAR_DUPLICATE, STAGE_GENERIC, STAGE_BALANCE)


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


def _build_doc_frequency(docs: Sequence[Doc]) -> tuple[dict[str, int], int]:
    """Document frequency of every content-bearing token across the full
    corpus, for `check_generic`. Built from full document text, not just
    the chunks candidates were generated from — genericness is a property
    of the corpus a question would have to be distinguished against, which
    is the whole corpus, not only the sampled chunks a `generate` run
    happened to touch."""
    doc_freq: dict[str, int] = {}
    for doc in docs:
        for token in set(tokenize(doc.text)):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return doc_freq, len(docs)


def run_filter(
    candidates: Sequence[ResolvedCandidate], docs: Sequence[Doc], config: FilterConfig
) -> FilterRunResult:
    """Run every candidate in `candidates` through the ordered filter
    pipeline, returning the survivors and the full, stage-by-stage audit
    trail (`stage_results`) needed to persist `filter_result` rows and to
    reconcile the funnel."""
    doc_freq, n_docs = _build_doc_frequency(docs)
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

        ok, reason = check_generic(candidate, doc_freq, n_docs, config.generic_df_pct)
        stage_results.append(StageResult(candidate.id, STAGE_GENERIC, ok, reason))
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
