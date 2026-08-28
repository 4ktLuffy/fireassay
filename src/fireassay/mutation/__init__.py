"""fireassay M2 — system mutation: does this eval setup catch injected
regressions?

Classical mutation testing mutates *source code*; this module mutates the
**system under test** (`mutation/operators.py`), asks whether a
`MutationDetector` (`mutation/detector.py`) notices, and reports
`gate_mutation_score = killed / (total - equivalent)`
(`mutation/score.py`). See `mutation/run.py` for the end-to-end
baseline -> mutants -> detector -> score pipeline.
"""

from fireassay.mutation.detector import MutationDetector, ThresholdDetector
from fireassay.mutation.operators import (
    ALL_OPERATOR_KINDS,
    CorruptQueryOperator,
    DetectorSpec,
    DropResultsOperator,
    MutationConfig,
    MutationOperator,
    ShuffleTopkOperator,
    SwapRankingOperator,
    TruncateTopkOperator,
    load_mutation_config,
    load_operators,
)
from fireassay.mutation.run import run_mutation
from fireassay.mutation.score import MutantResult, MutationScoreResult, compute_score

__all__ = [
    "ALL_OPERATOR_KINDS",
    "CorruptQueryOperator",
    "DetectorSpec",
    "DropResultsOperator",
    "MutantResult",
    "MutationConfig",
    "MutationDetector",
    "MutationOperator",
    "MutationScoreResult",
    "ShuffleTopkOperator",
    "SwapRankingOperator",
    "ThresholdDetector",
    "TruncateTopkOperator",
    "compute_score",
    "load_mutation_config",
    "load_operators",
    "run_mutation",
]
