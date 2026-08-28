"""The comparability engine — the core of fireassay.

A green evaluation is only evidence if the harness refuses to report a
number it cannot stand behind. `assert_comparable` is the single choke
point every comparison (`compare`, `diff`, and later `gate`) must pass
through before it is allowed to render a table: it decides whether a set
of runs is soundly comparable, and if not, why not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from fireassay.models import Run

RefusalCode = Literal[
    "SUITE_MISMATCH",
    "SCORER_MISMATCH",
    "ENV_MISMATCH",
    "RUN_INCOMPLETE",
    "EMPTY_RUN",
    "RUN_INADMISSIBLE",
]


class RefusalReason(BaseModel):
    """One specific, human-readable reason a comparison was refused."""

    model_config = ConfigDict(frozen=True)

    code: RefusalCode
    detail: str


class ComparisonRefusedError(Exception):
    """Raised by `assert_comparable` (force=False) when one or more runs are
    not soundly comparable. Carries the full list of reasons, not just the
    first one found, so a caller can fix every problem in one pass rather
    than rediscovering them one at a time."""

    def __init__(self, reasons: Sequence[RefusalReason]) -> None:
        self.reasons = tuple(reasons)
        summary = "; ".join(f"{r.code}: {r.detail}" for r in self.reasons)
        super().__init__(f"comparison refused: {summary}")


class ComparabilityReport(BaseModel):
    """The verdict of `assert_comparable`.

    `forced` is True only when the caller explicitly passed `force=True`
    *and* there were reasons that would otherwise have caused a refusal.
    Every renderer (report/text.py and any future HTML/JSON renderer) MUST
    check `.forced` and, if True, prefix its output with
    'UNSOUND COMPARISON' — the report is preserved deliberately, not raised
    as an exception, so `--force` output still explains exactly what is
    unsound about it.
    """

    model_config = ConfigDict(frozen=True)

    comparable: bool
    reasons: tuple[RefusalReason, ...] = ()
    forced: bool = False


def env_fingerprint(env_json: Mapping[str, object]) -> dict[str, object]:
    """Extract the sub-dict of `run.env_json` that participates in
    ENV_MISMATCH.

    `run.env_json` is shaped `{"python": ..., "fireassay": ..., "affects_results": {...}}`.
    Only `affects_results` is compared across runs: `python`/`fireassay`
    version strings are recorded for audit/debugging but are deliberately
    *not* grounds for refusal on their own — a patch-level Python bump that
    does not change results should not block a comparison. Which fields
    belong under `affects_results` (e.g. embedding model, chunker
    parameters, tokenizer) is a decision made by whoever constructs the
    run's environment record, not by this function.
    """
    value = env_json.get("affects_results", {})
    if not isinstance(value, dict):
        raise ValueError("run.env_json['affects_results'] must be a JSON object")
    return value


def assert_comparable(
    runs: Sequence[Run],
    *,
    scorers_by_run: Mapping[str, frozenset[str]],
    force: bool = False,
) -> ComparabilityReport:
    """Refuse to compare runs that are not soundly comparable.

    Refuses (collects every applicable `RefusalReason`, not just the first)
    when, across any two runs in `runs`:

    - `suite_hash` differs (`SUITE_MISMATCH`) — the runs were not scored
      against the same frozen question set, so a delta could just be a
      different, easier or harder, set of questions.
    - the set of `scorer` values ("name@version") differs
      (`SCORER_MISMATCH`) — a "recall" produced by `retrieval@1.0.0` and one
      produced by `retrieval@1.1.0` are not the same measurement, even if
      the metric name matches.
    - the `affects_results` sub-dict of `env_json` differs
      (`ENV_MISMATCH`) — e.g. a different embedding model or chunk size,
      which would make an apparent quality delta actually a setup delta.

    Also refuses, per individual run:

    - any run has `status != "complete"` (`RUN_INCOMPLETE`) — an in-flight
      or failed run's numbers are not final.
    - any run has zero results (`EMPTY_RUN`) — nothing to compare.
    - any run has `admissible is False` (`RUN_INADMISSIBLE`) — `runner.
      run_matrix` found a metric self-consistency violation
      (`score.invariants.check_invariants`) somewhere in that run. This is
      the whole thesis applied to the harness itself: a run whose own
      numbers are internally contradictory must not be silently treated as
      comparable evidence, no matter how the comparison would otherwise
      have come out.

    With `force=True`, does not raise `ComparisonRefusedError`; instead returns
    a `ComparabilityReport` with `forced=True` carrying the same reasons,
    so a caller can deliberately view an unsound comparison as long as
    every renderer stamps it as such.

    Raises `ComparisonRefusedError` (force=False) with the same reasons if the
    comparison is unsound. A comparison with fewer than two runs is always
    comparable in the pairwise sense (there is nothing to compare against),
    but per-run checks (RUN_INCOMPLETE, EMPTY_RUN) still apply — a
    single-run "leaderboard" of an incomplete run is still refused.
    """
    reasons: list[RefusalReason] = []

    for run in runs:
        if run.status != "complete":
            reasons.append(
                RefusalReason(
                    code="RUN_INCOMPLETE",
                    detail=f"run {run.id} has status '{run.status}', not 'complete'",
                )
            )
        if run.result_count == 0:
            reasons.append(
                RefusalReason(code="EMPTY_RUN", detail=f"run {run.id} has zero results")
            )
        if not run.admissible:
            reasons.append(
                RefusalReason(
                    code="RUN_INADMISSIBLE",
                    detail=f"run {run.id} failed a metric self-consistency invariant (admissible=False)",
                )
            )

    suite_hashes = {run.suite_hash for run in runs}
    if len(suite_hashes) > 1:
        detail = ", ".join(f"{run.id}={run.suite_hash}" for run in runs)
        reasons.append(
            RefusalReason(code="SUITE_MISMATCH", detail=f"runs disagree on suite_hash: {detail}")
        )

    scorer_sets = {run.id: scorers_by_run.get(run.id, frozenset()) for run in runs}
    distinct_scorer_sets = {frozenset(s) for s in scorer_sets.values()}
    if len(distinct_scorer_sets) > 1:
        detail = ", ".join(f"{run_id}={sorted(scorers)}" for run_id, scorers in scorer_sets.items())
        reasons.append(
            RefusalReason(code="SCORER_MISMATCH", detail=f"runs disagree on scorer set: {detail}")
        )

    fingerprints = {run.id: env_fingerprint(run.env_json) for run in runs}
    distinct_fingerprints = []
    for fp in fingerprints.values():
        if fp not in distinct_fingerprints:
            distinct_fingerprints.append(fp)
    if len(distinct_fingerprints) > 1:
        detail = ", ".join(f"{run_id}={fp}" for run_id, fp in fingerprints.items())
        reasons.append(
            RefusalReason(
                code="ENV_MISMATCH",
                detail=f"runs disagree on affects_results env fields: {detail}",
            )
        )

    if not reasons:
        return ComparabilityReport(comparable=True, reasons=(), forced=False)

    if force:
        return ComparabilityReport(comparable=False, reasons=tuple(reasons), forced=True)

    raise ComparisonRefusedError(reasons)
