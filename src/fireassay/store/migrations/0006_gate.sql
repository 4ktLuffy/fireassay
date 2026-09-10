-- fireassay M5 schema addition: gate_check.
--
-- Adds gate_check, the persisted record of one `fireassay.gate.
-- evaluate_gate` result for one metric of one (base_run, head_run) pair
-- (M5, `fireassay.gate.GateReport` / `GateMetricResult`). One row per
-- metric, written in a single transaction by `Store.put_gate_report`.
--
-- Append-only enforcement mirrors migration 0002's control_check /
-- mutation_run / mutant tables exactly, for the same reason: a
-- gate_check row is, by construction, only ever written *after* the
-- gate has already computed a verdict for that (base_run, head_run,
-- metric) triple, so there is no "parent run still open" state to key
-- sealing off of the way result/score are. The guarantee is instead
-- literal immutability -- once inserted, a gate_check row can never be
-- UPDATEd or DELETEd, enforced here by trigger and in Python by Store
-- never issuing an UPDATE/DELETE against this table.
--
-- **The `UNIQUE (base_run_id, head_run_id, metric)` constraint is
-- load-bearing, not incidental.** A gate check for a given (base, head,
-- metric) triple may be recorded exactly once. Re-running a gate on the
-- same pair of runs until it happens to come back green is p-hacking --
-- the same "counts how many times you asked" problem docs/SPEC.md §8
-- names for repeated CI attempts -- and this schema forbids it outright
-- rather than trusting the operator (or a retry loop) not to do it.
-- `Store.put_gate_report` surfaces a second attempt on the same triple
-- as `GateAlreadyCheckedError`, translated from this constraint's
-- `sqlite3.IntegrityError`, not as a silent overwrite.

CREATE TABLE gate_check (
    id            TEXT PRIMARY KEY,
    base_run_id   TEXT NOT NULL REFERENCES run(id),
    head_run_id   TEXT NOT NULL REFERENCES run(id),
    suite_id      TEXT NOT NULL REFERENCES suite(id),
    metric        TEXT NOT NULL,
    n_items       INTEGER NOT NULL,
    base_mean     REAL NOT NULL,
    head_mean     REAL NOT NULL,
    delta         REAL NOT NULL,
    ci_low        REAL NOT NULL,
    ci_high       REAL NOT NULL,
    p_value       REAL NOT NULL,
    p_adjusted    REAL NOT NULL,
    mde           REAL NOT NULL,
    threshold     REAL NOT NULL,
    alpha         REAL NOT NULL,
    power         REAL NOT NULL,
    bootstrap_b   INTEGER NOT NULL,
    seed          INTEGER NOT NULL,
    verdict       TEXT NOT NULL,
    checked_at    TEXT NOT NULL,
    UNIQUE (base_run_id, head_run_id, metric)
);

CREATE TRIGGER gate_check_seal_update
BEFORE UPDATE ON gate_check
BEGIN
    SELECT RAISE(ABORT, 'gate_check is append-only: rows cannot be updated');
END;

CREATE TRIGGER gate_check_seal_delete
BEFORE DELETE ON gate_check
BEGIN
    SELECT RAISE(ABORT, 'gate_check is append-only: rows cannot be deleted');
END;
