"""Krippendorff's α (M3-SPEC.md §4), via the `krippendorff` package — the
one new runtime dependency M3 adds (no pandas, no scikit-learn).

**Agreement is not accuracy.** α says whether two curators agree with each
other; it says nothing about whether either of them is *right* — two
curators can agree confidently and both be wrong. `honeypots.py` measures
correctness against a known-bad item; the two are deliberately separate
instruments and neither substitutes for the other (M3-SPEC.md §4).

**Ordinal** metric for `answerable_from_kb` (no/partially/yes) and
`reference_answer_correct` (no/incomplete/yes) — both have a genuine
order, and treating "no" vs "yes" as equally different from "partially"/
"incomplete" as they are from each other would understate agreement on a
near-miss. **Nominal** for the four rubric booleans
(`self_contained`, `evidence_sufficient`, `difficulty_agrees`,
`qtype_agrees`) — a boolean has no order to preserve.

Missing observations (a candidate only one curator happened to review) are
passed through as `NaN`, never imputed — imputing a value one curator never
gave would be inventing agreement data, exactly the kind of measurement
with no external referent this whole project exists to catch. This is also
*why* Krippendorff's α rather than Cohen's κ: κ has no principled way to
handle a partially-overlapping rater set at all.

**"Unmeasurable" is a state, not a number (M2-SPEC.md §1's rule, applied
here).** `krippendorff.alpha` can raise (`ValueError: There has to be more
than one value in the domain` — an entirely ordinary outcome: two curators
agreeing on every shared item), and separately can return `NaN` without
raising (a sparse, missing-heavy reliability matrix). Both were previously
collapsed into looking like "no problem": a caller comparing a `NaN` alpha
against `LOW_AGREEMENT_THRESHOLD` gets `False` (`nan < 0.6` is `False`),
so a criterion whose agreement is mathematically undefined silently read as
acceptable. `AgreementResult.status` makes the three states — computed,
unmeasurable, insufficient data — mutually exclusive and explicit, so
"we could not tell" can never be reported as if it were "agreement is
fine".
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from fireassay.curate.models import RubricVerdict

RubricCriterion = Literal[
    "answerable_from_kb",
    "self_contained",
    "reference_answer_correct",
    "evidence_sufficient",
    "difficulty_agrees",
    "qtype_agrees",
]

AgreementStatus = Literal["OK", "UNMEASURABLE", "INSUFFICIENT_DATA"]

_ORDINAL_LEVELS: dict[str, dict[str, float]] = {
    "answerable_from_kb": {"no": 0.0, "partially": 1.0, "yes": 2.0},
    "reference_answer_correct": {"no": 0.0, "incomplete": 1.0, "yes": 2.0},
}
_ORDINAL_FIELDS = frozenset(_ORDINAL_LEVELS)
_NOMINAL_BOOL_FIELDS = frozenset(
    {"self_contained", "evidence_sufficient", "difficulty_agrees", "qtype_agrees"}
)

#: α below this on `reference_answer_correct` is the flag M3-SPEC.md §4
#: says must be recorded in `suite.agreement_json` and surfaced in every
#: report built on that suite, permanently.
LOW_AGREEMENT_THRESHOLD = 0.6


@dataclass(frozen=True)
class AgreementResult:
    """The outcome of asking for α on one rubric criterion.

    `alpha` is only ever a real, finite number when `status == "OK"` — a
    consumer must switch on `status` first and never compare `alpha`
    against a threshold without checking it is `"OK"`, precisely the bug
    this type exists to make impossible (`NaN < threshold` is `False`,
    which silently reads as "not low agreement" if `status` is ignored).
    """

    criterion: RubricCriterion
    alpha: float | None
    status: AgreementStatus
    detail: str


def level_of_measurement(criterion: str) -> Literal["ordinal", "nominal"]:
    if criterion in _ORDINAL_FIELDS:
        return "ordinal"
    if criterion in _NOMINAL_BOOL_FIELDS:
        return "nominal"
    raise ValueError(f"unknown rubric criterion {criterion!r}")


def _numeric_value(rubric: RubricVerdict, criterion: str) -> float:
    if criterion in _ORDINAL_FIELDS:
        raw = getattr(rubric, criterion)
        return _ORDINAL_LEVELS[criterion][raw]
    value = getattr(rubric, criterion)
    if not isinstance(value, bool):
        raise ValueError(f"rubric criterion {criterion!r} is not boolean-valued: {value!r}")
    return 1.0 if value else 0.0


def _import_krippendorff() -> Any:
    """`krippendorff` is imported here, at call time, not at module import.

    The `curate` package is imported by the CLI (`curate report`) and by
    `report/text.py`, so a module-level import made `krippendorff` a hard
    dependency of *every* `fireassay` install -- including a consumer that
    only wants `fireassay.hashing`/`integrity`/`items.core` as a library
    (agent-assay is one; see pyproject's `[curate]` extra). The package
    itself is tiny and pure Python; the point is that the spine must not
    require the curation stack, and the import error, when it happens,
    must say which extra to install rather than surface as a bare
    ModuleNotFoundError from inside a rubric report.
    """
    try:
        import krippendorff
    except ImportError as exc:  # pragma: no cover - exercised only on a spine-only install
        raise ImportError(
            "Krippendorff's alpha needs the optional 'krippendorff' package: "
            "pip install 'fireassay[curate]'"
        ) from exc
    return krippendorff


def krippendorff_alpha(
    rubrics_by_curator: Mapping[str, Mapping[str, RubricVerdict]], criterion: RubricCriterion
) -> AgreementResult:
    """α for `criterion`, over `{curator_id: {candidate_id: RubricVerdict}}`
    (one `RubricVerdict` per curator per candidate — the caller is
    responsible for reducing multiple decisions on the same candidate down
    to each curator's *latest*, per `Decision`'s append-only semantics).

    Returns an `AgreementResult` whose `status` is:

    - `INSUFFICIENT_DATA` — fewer than two curators have rated anything,
      or no candidate was rated at all. α is undefined with a single
      rater; this is not a measurement attempt that failed, it is one
      that never had enough data to run.
    - `UNMEASURABLE` — a measurement was attempted and could not produce a
      real number: `krippendorff.alpha` raised (most commonly `ValueError:
      There has to be more than one value in the domain` — an entirely
      ordinary case, e.g. two curators who agree on every shared item),
      or it returned `NaN` (a sparse, missing-heavy reliability matrix).
      The package's `RuntimeWarning` for the NaN case is suppressed at
      this call site specifically because we check for it ourselves
      immediately after — a warning must not be the only signal for a
      condition the return value already reports structurally.
    - `OK` — a real, finite α.
    """
    level = level_of_measurement(criterion)
    curators = sorted(rubrics_by_curator)
    if len(curators) < 2:
        return AgreementResult(
            criterion=criterion,
            alpha=None,
            status="INSUFFICIENT_DATA",
            detail="fewer than two curators have rated anything for this criterion",
        )
    candidate_ids = sorted({cid for by_cand in rubrics_by_curator.values() for cid in by_cand})
    if not candidate_ids:
        return AgreementResult(
            criterion=criterion, alpha=None, status="INSUFFICIENT_DATA", detail="no candidate rated"
        )

    matrix = np.full((len(curators), len(candidate_ids)), np.nan, dtype=float)
    for row, curator in enumerate(curators):
        by_cand = rubrics_by_curator[curator]
        for col, cid in enumerate(candidate_ids):
            if cid in by_cand:
                matrix[row, col] = _numeric_value(by_cand[cid], criterion)

    krippendorff = _import_krippendorff()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            alpha = float(krippendorff.alpha(reliability_data=matrix, level_of_measurement=level))
    except ValueError as exc:
        return AgreementResult(criterion=criterion, alpha=None, status="UNMEASURABLE", detail=str(exc))

    if math.isnan(alpha):
        return AgreementResult(
            criterion=criterion,
            alpha=None,
            status="UNMEASURABLE",
            detail="krippendorff.alpha returned NaN (degenerate or too-sparse reliability data)",
        )

    return AgreementResult(criterion=criterion, alpha=alpha, status="OK", detail="")
