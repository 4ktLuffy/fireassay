"""Loader for `controls/expected.yaml` — the version-controlled expected
bands every control checks its observed metrics against (M2-SPEC.md §4).

Two rejections are load-bearing: an unknown control kind is rejected (a
typo'd kind would silently register no constraint at all for that control),
and an empty band — a control kind, or a metric within it, with no
`eq`/`min`/`max` key — is rejected, because a band with no constraint keys
always "passes" regardless of what was observed. That is precisely the
failure mode M2's governing rule (M2-SPEC.md §1: "a check that cannot run
must never read as a check that passed") exists to rule out, applied to the
expected-band file itself rather than to a control's runtime behaviour.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

#: The four controls this file may declare a band for. `judge_calibration`
#: and `null_questions` deliberately have no entry: both always report
#: `NOT_RUN` (each needs something M2 does not have -- a judge, and a
#: generating system to measure abstention against, respectively; both
#: arrive in M4 -- see their modules) and so have nothing to check a band
#: against.
VALID_CONTROL_KINDS = frozenset(
    {"no_retrieval", "shuffled_gold", "corpus_ablation", "identical_config"}
)


class Band(BaseModel):
    """One constraint on one observed metric.

    `eq`/`min`/`max` may be combined (e.g. a band with both `min` and
    `max`); at least one must be present — see `_reject_empty`.
    """

    model_config = ConfigDict(frozen=True)

    eq: float | None = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _reject_empty(self) -> Band:
        if self.eq is None and self.min is None and self.max is None:
            raise ValueError(
                "expected.yaml band has no constraint keys (eq/min/max) — "
                "an empty band always passes regardless of what was observed"
            )
        return self

    def check(self, value: float) -> bool:
        """Whether `value` satisfies every constraint this band declares."""
        if self.eq is not None and value != self.eq:
            return False
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)


def load_expected_bands(path: Path | str) -> dict[str, dict[str, Band]]:
    """Load `controls/expected.yaml` into `{control_kind: {metric_name: Band}}`.

    Raises `ValueError` for:
    - a control kind not in `VALID_CONTROL_KINDS`;
    - a control kind with no metric bands at all (an empty mapping);
    - a metric whose band has no constraint keys (`{}` or `null`).
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"expected.yaml must be a mapping of control kind -> metric bands, got {type(raw).__name__}"
        )

    result: dict[str, dict[str, Band]] = {}
    for kind, metrics in raw.items():
        if kind not in VALID_CONTROL_KINDS:
            raise ValueError(
                f"expected.yaml: unknown control kind {kind!r}; must be one of {sorted(VALID_CONTROL_KINDS)}"
            )
        if not metrics:
            raise ValueError(f"expected.yaml: {kind!r} has no metric bands — an empty band always passes")
        band_for_metric: dict[str, Band] = {}
        for metric_name, band_spec in metrics.items():
            if not band_spec:
                raise ValueError(
                    f"expected.yaml: {kind}.{metric_name} has no constraint keys — "
                    "an empty band always passes regardless of what was observed"
                )
            band_for_metric[metric_name] = Band(**band_spec)
        result[kind] = band_for_metric
    return result
