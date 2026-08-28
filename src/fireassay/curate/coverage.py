"""Content validity (`docs/SPEC.md` §7, M3-SPEC.md §4): does the curated
set actually spread across the corpus and the `(qtype, difficulty)` grid,
or is it concentrated on a fraction of both?

Also: **difficulty validation** — the spike (`spike/RESULTS.md`) found a
generator's proposed `difficulty` label can be *inverted* relative to what
actually drives retrieval (easy: 0.176 recall, hard: 0.588). `difficulty
_feature_correlation` reports Pearson r between the proposed label
(ordinal: easy=0, medium=1, hard=2) and each measured `CandidateFeatures`
field, over every generated candidate — "a difficulty label that does not
correlate with anything measurable is a label, not a difficulty" (M3-SPEC.md
§2), and this is the number that lets a report say so. `gold_doc_rank` — a
*measured*, empirical difficulty signal (1 = easy to find, 30+ = hard) — is
checked alongside the lexical features for exactly this reason: a
generator's proposed label should be validated against retrieval reality,
not only against lexical overlap with the title/quote.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fireassay.generate.models import ResolvedCandidate
from fireassay.system.corpus import Doc

_DIFFICULTY_ORDER = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
#: `gold_doc_rank` is nullable (a candidate may have no measured rank, or
#: none within the search depth) unlike the other three, which are always
#: present -- `difficulty_feature_correlation` filters it to only the
#: candidates that have a value before computing anything, per-feature,
#: rather than requiring every candidate to have every feature.
_FEATURE_NAMES = ("title_overlap", "quote_overlap", "question_len_tokens", "gold_doc_rank")


@dataclass(frozen=True)
class GeneratorObedience:
    """How often the model's own proposed `qtype`/`difficulty` matched
    the specific cell `generate.pipeline` asked it for (M3-SPEC.md's
    pre-flight finding: an unstratified prompt let the model default to
    easy factual questions almost every time). A free measure of whether
    the generator is actually listening to what it is asked for, distinct
    from whether its proposal is *correct* — a model can honestly report
    that it produced something other than what was requested, and that is
    a different fact from disagreeing about the label on a question it
    was correctly asked to attempt.

    `None` fields mean no candidates were given to compute a rate from —
    never a misleading `0.0`, the same omit-don't-zero rule this project
    applies everywhere a mean can be computed over an empty population.
    """

    n_candidates: int
    qtype_match_rate: float | None
    difficulty_match_rate: float | None
    cell_match_rate: float | None


def generator_obedience(candidates: Sequence[ResolvedCandidate]) -> GeneratorObedience:
    """Compute `GeneratorObedience` over `candidates`."""
    n = len(candidates)
    if n == 0:
        return GeneratorObedience(
            n_candidates=0, qtype_match_rate=None, difficulty_match_rate=None, cell_match_rate=None
        )
    qtype_matches = sum(1 for c in candidates if c.qtype == c.target_qtype)
    difficulty_matches = sum(1 for c in candidates if c.difficulty == c.target_difficulty)
    cell_matches = sum(
        1 for c in candidates if c.qtype == c.target_qtype and c.difficulty == c.target_difficulty
    )
    return GeneratorObedience(
        n_candidates=n,
        qtype_match_rate=qtype_matches / n,
        difficulty_match_rate=difficulty_matches / n,
        cell_match_rate=cell_matches / n,
    )


@dataclass(frozen=True)
class CoverageReport:
    #: fraction of corpus documents with >= 1 curated question.
    doc_coverage_fraction: float
    documents_with_questions: int
    total_documents: int
    #: {(qtype, difficulty): count}, over every (qtype, difficulty)
    #: combination that appears at least once — cells with zero questions
    #: are simply absent, which is itself the signal a sparse suite shows.
    cell_occupancy: dict[tuple[str, str], int]


def content_coverage(candidates: Sequence[ResolvedCandidate], docs: Sequence[Doc]) -> CoverageReport:
    """Coverage over `candidates` — the caller decides which population
    this means (M3-SPEC.md leaves this to the report: `curate report`
    passes the currently accepted/edited set, since "the funnel reports
    how many questions survived; it must also report coverage" is framed
    in terms of survivors, not raw generation output)."""
    doc_ids_with_questions = {c.source_doc_id for c in candidates}
    total_documents = len(docs)
    documents_with_questions = sum(1 for d in docs if d.doc_id in doc_ids_with_questions)
    doc_coverage_fraction = (documents_with_questions / total_documents) if total_documents else 0.0

    cell_occupancy: dict[tuple[str, str], int] = {}
    for c in candidates:
        cell = (c.qtype, c.difficulty)
        cell_occupancy[cell] = cell_occupancy.get(cell, 0) + 1

    return CoverageReport(
        doc_coverage_fraction=doc_coverage_fraction,
        documents_with_questions=documents_with_questions,
        total_documents=total_documents,
        cell_occupancy=cell_occupancy,
    )


def difficulty_feature_correlation(candidates: Sequence[ResolvedCandidate]) -> dict[str, float]:
    """Pearson r between proposed `difficulty` (ordinal 0/1/2) and each
    `CandidateFeatures` field, over `candidates`.

    Each feature is filtered to the candidates that actually have a value
    for it *before* correlating (`gold_doc_rank` is `None` for a candidate
    with no measured rank; the other three are always present) — a
    per-feature filter, not a single global one, so one nullable feature
    does not shrink the population every other feature is correlated over.

    A feature is simply absent from the result — never a misleading `0.0`
    — when fewer than 2 candidates have a value for it, or when
    `difficulty` (or the feature itself) is constant across that subset:
    `numpy.corrcoef` returns `nan` for a zero-variance input, which would
    otherwise silently read as "no correlation" when the honest answer is
    "undefined, not enough variation to compute one".
    """
    result: dict[str, float] = {}
    for feature_name in _FEATURE_NAMES:
        pairs = [
            (_DIFFICULTY_ORDER[c.difficulty], getattr(c.features, feature_name))
            for c in candidates
            if getattr(c.features, feature_name) is not None
        ]
        if len(pairs) < 2:
            continue
        difficulties = np.array([p[0] for p in pairs], dtype=float)
        values = np.array([p[1] for p in pairs], dtype=float)
        if np.std(difficulties) == 0.0 or np.std(values) == 0.0:
            continue
        r = float(np.corrcoef(difficulties, values)[0, 1])
        if not np.isnan(r):
            result[feature_name] = r
    return result
