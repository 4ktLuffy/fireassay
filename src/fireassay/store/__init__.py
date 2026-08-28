"""SQLite-backed persistence for fireassay (M1 tables only).

See db.py for the Store class, and migrations/0001_initial.sql for the
schema (question, evidence_span, suite, suite_question, config, run,
result, score). `curation`, `control_check` and `gate_check` are
out of scope for M1 and intentionally omitted.
"""

from fireassay.store.db import RunSealedError, Store, SuiteExistsError

__all__ = ["RunSealedError", "Store", "SuiteExistsError"]
