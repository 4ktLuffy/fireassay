"""`FilterConfig` and its YAML loader.

Unknown keys are rejected outright — the same rule `controls.expected` and
`mutation.operators` already apply to their own config files, and for the
same reason the spike found on the mutants YAML's `detector:` block: a
config key that is silently accepted and ignored is how a documented
setting ends up having no effect at all.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class FilterConfig(BaseModel):
    """Tunables for `filter.pipeline.run_filter`, one field per stage in
    M3-SPEC.md §3's table (span resolution and its two reason codes are
    generate/'s responsibility, not configured here)."""

    model_config = ConfigDict(frozen=True)

    #: degeneracy: question token-count band.
    min_question_tokens: int = 5
    max_question_tokens: int = 60
    #: near-duplicate: token-set Jaccard threshold against any kept candidate.
    near_dup_jaccard_threshold: float = 0.85
    #: generic: a question is TOO_GENERIC when every one of its content
    #: words appears in more than this fraction of corpus documents.
    generic_df_pct: float = 0.05
    #: balance: max candidates kept per (qtype, difficulty) cell.
    cell_cap: int = 200


_VALID_KEYS = frozenset(FilterConfig.model_fields)


def load_filter_config(path: Path | str) -> FilterConfig:
    """Load a `filter.yaml` (see `configs/filter.example.yaml`). Raises
    `ValueError` for a top-level key outside `FilterConfig`'s fields."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"filter.yaml must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _VALID_KEYS
    if unknown:
        raise ValueError(
            f"filter.yaml: unknown key(s) {sorted(unknown)}; must be a subset of {sorted(_VALID_KEYS)}"
        )
    return FilterConfig(**raw)
