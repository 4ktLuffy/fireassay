"""The admissibility engine (M2-SPEC.md §5) — the second choke point (after
`integrity.assert_comparable`) every run must pass through before its
numbers are trusted.

`assess` is a **pure function**, deliberately mirroring
`score.invariants.check_invariants`: it computes a verdict and returns it;
persisting that verdict onto a `Run` (`run.admissible`,
`run.admissibility_json`) is the caller's responsibility, via
`Store.set_admissibility` — exactly how `check_invariants`'s result is
persisted by `runner.run_matrix` calling `Store.finish_run`, not by
`check_invariants` itself.

Governing rule (M2-SPEC.md §1): a run is admissible **iff** it has no
invariant violations **and** every control is `PASSED` (or `NOT_RUN` and
explicitly allowed via `allow_not_run`). `NOT_RUN` is never silently
treated as a pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from fireassay.controls.base import ControlOutcome
from fireassay.models import Run


class InvariantViolationLike(Protocol):
    """What `assess` needs from an invariant violation: a rule name and a
    detail string. `fireassay.score.invariants.InvariantViolation` is one
    such thing; a consumer with its own violation model (agent-assay's is
    scoped to an *item* rather than a *question*, so it is a different
    class on purpose) is another. `assess` only ever counts these and
    reports the count, so the Protocol is deliberately this small."""

    @property
    def rule(self) -> str: ...

    @property
    def detail(self) -> str: ...


class Admissibility(BaseModel):
    """The verdict `assess` computes. `detail` is a human-readable summary
    of every reason admissibility failed, or `"admissible"` when it did
    not fail for any reason."""

    model_config = ConfigDict(frozen=True)

    admissible: bool
    invariant_violation_count: int
    failed_controls: tuple[str, ...]
    not_run_controls: tuple[str, ...]
    allowed_not_run: tuple[str, ...]
    detail: str


def assess(
    run: Run,
    invariant_violations: Sequence[InvariantViolationLike],
    controls: Sequence[ControlOutcome],
    *,
    allow_not_run: Sequence[str] = (),
) -> Admissibility:
    """Decide whether `run` is admissible.

    A run is admissible **iff**:

    - `invariant_violations` is empty, and
    - every control in `controls` is `PASSED`, or `NOT_RUN` **and** its
      `kind` appears in `allow_not_run`.

    Any `FAILED` control makes the run inadmissible unconditionally —
    `allow_not_run` only ever widens what `NOT_RUN` is tolerated for, never
    what `FAILED` is tolerated for. A control kind repeated in `controls`
    (e.g. checked against more than one config) contributes once per
    occurrence to `failed_controls`/`not_run_controls`, but each set is
    de-duplicated by `kind` in the returned `Admissibility` — the caller
    only needs to know *which* kinds are the problem, not how many times.
    """
    failed = tuple(sorted({c.kind for c in controls if c.status == "FAILED"}))
    not_run = tuple(sorted({c.kind for c in controls if c.status == "NOT_RUN"}))
    allowed = frozenset(allow_not_run)
    disallowed_not_run = tuple(sorted(kind for kind in not_run if kind not in allowed))
    allowed_not_run = tuple(sorted(kind for kind in not_run if kind in allowed))

    admissible = not invariant_violations and not failed and not disallowed_not_run

    reasons: list[str] = []
    if invariant_violations:
        reasons.append(f"{len(invariant_violations)} invariant violation(s)")
    if failed:
        reasons.append(f"failed control(s): {', '.join(failed)}")
    if disallowed_not_run:
        reasons.append(f"disallowed NOT_RUN control(s): {', '.join(disallowed_not_run)}")
    detail = "admissible" if admissible else "; ".join(reasons)

    return Admissibility(
        admissible=admissible,
        invariant_violation_count=len(invariant_violations),
        failed_controls=failed,
        not_run_controls=not_run,
        allowed_not_run=allowed_not_run,
        detail=detail,
    )
