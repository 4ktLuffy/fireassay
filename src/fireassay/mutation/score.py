"""`gate_mutation_score` arithmetic (M2-SPEC.md §6):

    gate_mutation_score = killed / (total - equivalent)

A mutant is **killed** when the detector reports it, and **equivalent**
when it provably does not degrade quality (see each operator's
`check_equivalence` in `mutation/operators.py`). Equivalent mutants are
excluded from both the numerator and the denominator — never merely
assumed equivalent: `MutantResult.equivalent_reason` is required whenever
`equivalent` is `True` (`__post_init__` enforces it), so an inflated score
from an unexplained exclusion cannot happen silently.

A **surviving** mutant (`not equivalent and not killed`) is the
interesting output — it names a regression this eval setup would not have
caught. `MutationScoreResult.mutants` and any report built on it list
survivors first.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, model_validator


class MutantResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    operator: str
    params: dict[str, object]
    mutant_run_id: str
    killed: bool
    equivalent: bool
    equivalent_reason: str | None
    detail: str

    @model_validator(mode="after")
    def _require_reason_when_equivalent(self) -> MutantResult:
        if self.equivalent and not self.equivalent_reason:
            raise ValueError(
                f"mutant {self.operator}({self.params}) is marked equivalent with no "
                "equivalent_reason — an unexplained exclusion is how a mutation score gets inflated"
            )
        return self


class MutationScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    suite_id: str
    suite_hash: str
    base_config_id: str
    detector: str
    killed: int
    total: int
    equivalent: int
    score: float
    mutants: tuple[MutantResult, ...]

    def survivors(self) -> tuple[MutantResult, ...]:
        """Non-equivalent mutants the detector did not kill — the
        interesting output: each one names a regression this eval setup
        would not have caught."""
        return tuple(m for m in self.mutants if not m.equivalent and not m.killed)

    def sorted_for_report(self) -> tuple[MutantResult, ...]:
        """`mutants`, survivors first (M2-SPEC.md §6: "the report lists
        survivors first"), then killed mutants, then equivalent
        (excluded) mutants, each group in original operator order."""
        survivors = [m for m in self.mutants if not m.equivalent and not m.killed]
        killed = [m for m in self.mutants if not m.equivalent and m.killed]
        equivalent = [m for m in self.mutants if m.equivalent]
        return tuple(survivors + killed + equivalent)


def compute_score(mutants: Sequence[MutantResult]) -> tuple[int, int, int, float]:
    """Return `(killed, total, equivalent, score)` for `mutants`.

    `killed`/`total`/`equivalent` are counts; `score = killed / (total -
    equivalent)`, or `0.0` if every mutant was equivalent (an empty
    denominator must not raise — and reports 0.0 rather than a
    misleadingly perfect score, since "no mutants were even eligible to be
    killed" is not evidence the harness catches anything).
    """
    total = len(mutants)
    equivalent = sum(1 for m in mutants if m.equivalent)
    killed = sum(1 for m in mutants if m.killed and not m.equivalent)
    denom = total - equivalent
    score = (killed / denom) if denom > 0 else 0.0
    return killed, total, equivalent, score
