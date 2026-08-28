"""fireassay M2 — negative controls: the harness proves it can fail.

Five deterministic controls (`no_retrieval`, `shuffled_gold`,
`corpus_ablation`, `identical_config`, `null_questions`) plus a `NOT_RUN`
stub for `judge_calibration` (arrives in M4). See `controls/base.py` for
the governing rule and `controls/registry.py` for the full list.
"""

from fireassay.controls.base import Control, ControlContext, ControlOutcome, build_control_context
from fireassay.controls.expected import Band, load_expected_bands
from fireassay.controls.registry import ALL_CONTROLS

__all__ = [
    "ALL_CONTROLS",
    "Band",
    "Control",
    "ControlContext",
    "ControlOutcome",
    "build_control_context",
    "load_expected_bands",
]
