-- fireassay M2 schema additions: controls, admissibility, mutation.
--
-- Adds control_check (per M2-SPEC.md §7), mutation_run, mutant, and one
-- column on the existing `run` table (`admissibility_json`) that
-- M2-SPEC.md §5 requires `admissibility.assess`'s caller to be able to
-- write ("`assess` writes `run.admissible` and `run.admissibility_json`").
-- `run.admissible` already exists (M1 migration 0001, populated from
-- `score.invariants.check_invariants`); §5's admissibility verdict folds
-- that same column in as one more input, alongside control results.
--
-- Append-only enforcement for the three new tables: M1's `result`/`score`
-- sealing is keyed to the *parent run's* status (no writes once the run is
-- 'complete'/'failed'), which does not fit `control_check`/`mutation_run`/
-- `mutant` -- those rows are, by construction, only ever written *after*
-- the run(s) they describe have already completed. Their append-only
-- guarantee is instead literal immutability: once inserted, a row in any
-- of these three tables can never be UPDATEd or DELETEd, enforced here by
-- trigger and in Python by Store never issuing an UPDATE/DELETE against
-- them (see store/db.py).

ALTER TABLE run ADD COLUMN admissibility_json TEXT NOT NULL DEFAULT '{}';

CREATE TABLE control_check (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES run(id),
    kind          TEXT NOT NULL,
    status        TEXT NOT NULL,   -- PASSED | FAILED | NOT_RUN
    observed_json TEXT NOT NULL,
    expected_json TEXT NOT NULL,
    twin_ok       INTEGER NOT NULL,   -- 0/1
    cause_json    TEXT NOT NULL,
    detail        TEXT NOT NULL,
    checked_at    TEXT NOT NULL
);

CREATE TABLE mutation_run (
    id              TEXT PRIMARY KEY,
    suite_id        TEXT NOT NULL REFERENCES suite(id),
    suite_hash      TEXT NOT NULL,
    base_config_id  TEXT NOT NULL REFERENCES config(id),
    detector        TEXT NOT NULL,
    killed          INTEGER NOT NULL,
    total           INTEGER NOT NULL,
    equivalent      INTEGER NOT NULL,
    score           REAL NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE mutant (
    id                TEXT PRIMARY KEY,
    mutation_run_id   TEXT NOT NULL REFERENCES mutation_run(id),
    operator          TEXT NOT NULL,
    params_json       TEXT NOT NULL,
    mutant_run_id     TEXT NOT NULL REFERENCES run(id),
    killed            INTEGER NOT NULL,   -- 0/1
    equivalent        INTEGER NOT NULL,   -- 0/1
    equivalent_reason TEXT,
    detail            TEXT NOT NULL
);

CREATE TRIGGER control_check_no_update
BEFORE UPDATE ON control_check
BEGIN
    SELECT RAISE(ABORT, 'control_check is append-only: rows cannot be updated');
END;

CREATE TRIGGER control_check_no_delete
BEFORE DELETE ON control_check
BEGIN
    SELECT RAISE(ABORT, 'control_check is append-only: rows cannot be deleted');
END;

CREATE TRIGGER mutation_run_no_update
BEFORE UPDATE ON mutation_run
BEGIN
    SELECT RAISE(ABORT, 'mutation_run is append-only: rows cannot be updated');
END;

CREATE TRIGGER mutation_run_no_delete
BEFORE DELETE ON mutation_run
BEGIN
    SELECT RAISE(ABORT, 'mutation_run is append-only: rows cannot be deleted');
END;

CREATE TRIGGER mutant_no_update
BEFORE UPDATE ON mutant
BEGIN
    SELECT RAISE(ABORT, 'mutant is append-only: rows cannot be updated');
END;

CREATE TRIGGER mutant_no_delete
BEFORE DELETE ON mutant
BEGIN
    SELECT RAISE(ABORT, 'mutant is append-only: rows cannot be deleted');
END;
