"""SQLite-backed persistence for fireassay.

See db.py for the Store class. migrations/0001_initial.sql holds the M1
schema (question, evidence_span, suite, suite_question, config, run,
result, score); migrations/0002_controls_mutation.sql (M2) adds
control_check, mutation_run, mutant, and run.admissibility_json.
`curation` and `gate_check` remain out of scope (M3/M5) and are
intentionally omitted.
"""

from fireassay.store.db import (
    AdmissibilityAlreadySetError,
    ControlCheckRow,
    MutantRow,
    MutationRunRow,
    RunSealedError,
    Store,
    SuiteExistsError,
)

__all__ = [
    "AdmissibilityAlreadySetError",
    "ControlCheckRow",
    "MutantRow",
    "MutationRunRow",
    "RunSealedError",
    "Store",
    "SuiteExistsError",
]
