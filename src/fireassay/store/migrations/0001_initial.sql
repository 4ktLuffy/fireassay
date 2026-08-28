-- fireassay M1 schema.
--
-- Restricted to the M1 tables per M1-SPEC.md §3: question, evidence_span,
-- suite, suite_question, config, run, result, score. `curation`,
-- `control_check` and `gate_check` belong to later milestones (negative
-- controls, curation, release gate) and are intentionally not created here.
--
-- Compared to the full schema in fireassay-SPEC.md §5, the `run` table also
-- drops `code_version`, `is_control`, `control_kind`, `admissible` and
-- `admissibility_json`: those columns exist to support negative controls
-- (M2), which M1 has no logic to populate. Adding them now with no writer
-- would just be dead, misleading columns; they will be added back in the M2
-- migration that introduces controls.
--
-- Append-only enforcement: the triggers below reject any INSERT into
-- `result` or `score` whose parent run is no longer 'running' (i.e. status
-- is 'complete' or 'failed' — both are terminal). This is a second line of
-- defense behind the Python-level check in store/db.py; it protects the
-- invariant even against a caller that bypasses the Store class.

CREATE TABLE question (
    id               TEXT PRIMARY KEY,
    text             TEXT NOT NULL,
    qtype            TEXT NOT NULL,
    difficulty       TEXT NOT NULL,
    reference_answer TEXT,
    provenance       TEXT NOT NULL,
    generator        TEXT,
    source_doc_id    TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE evidence_span (
    question_id TEXT NOT NULL REFERENCES question(id),
    doc_id      TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    page        INTEGER,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    quote       TEXT NOT NULL,
    PRIMARY KEY (question_id, doc_id, chunk_id, char_start, char_end)
);

CREATE TABLE suite (
    id             TEXT PRIMARY KEY,   -- suite_hash; suites are content-addressed
    name           TEXT NOT NULL,
    version        TEXT NOT NULL,
    suite_hash     TEXT NOT NULL,
    frozen_at      TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE suite_question (
    suite_id    TEXT NOT NULL REFERENCES suite(id),
    question_id TEXT NOT NULL REFERENCES question(id),
    PRIMARY KEY (suite_id, question_id)
);

CREATE TABLE config (
    id          TEXT PRIMARY KEY,   -- config_hash; configs are content-addressed
    config_hash TEXT NOT NULL UNIQUE,
    label       TEXT,
    spec_json   TEXT NOT NULL
);

CREATE TABLE run (
    id          TEXT PRIMARY KEY,   -- random id; runs are events, not content-addressed
    suite_id    TEXT NOT NULL REFERENCES suite(id),
    suite_hash  TEXT NOT NULL,      -- denormalised for integrity checks without a join
    config_id   TEXT NOT NULL REFERENCES config(id),
    config_hash TEXT NOT NULL,      -- denormalised for integrity checks without a join
    env_json    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',   -- running | complete | failed
    -- Added post-M1-SPEC.md: score.invariants.check_invariants runs at the
    -- end of every run; a run with any violation is admissible = 0. Not
    -- (yet) wired into integrity.assert_comparable's refusal codes -- see
    -- Run.admissible's docstring in models.py.
    admissible                 INTEGER NOT NULL DEFAULT 1,
    invariant_violations_json  TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE result (
    run_id         TEXT NOT NULL REFERENCES run(id),
    question_id    TEXT NOT NULL REFERENCES question(id),
    answer         TEXT,
    abstained      INTEGER NOT NULL,   -- 0/1
    retrieved_json TEXT NOT NULL,
    latency_json   TEXT NOT NULL,
    tokens_in      INTEGER NOT NULL,
    tokens_out     INTEGER NOT NULL,
    cost_usd       REAL NOT NULL,
    PRIMARY KEY (run_id, question_id)
);

CREATE TABLE score (
    run_id      TEXT NOT NULL REFERENCES run(id),
    question_id TEXT NOT NULL REFERENCES question(id),
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    scorer      TEXT NOT NULL,   -- "name@version"
    rationale   TEXT,
    PRIMARY KEY (run_id, question_id, metric, scorer)
);

CREATE TRIGGER result_seal_insert
BEFORE INSERT ON result
FOR EACH ROW
WHEN (SELECT status FROM run WHERE id = NEW.run_id) != 'running'
BEGIN
    SELECT RAISE(ABORT, 'run is sealed: cannot write result');
END;

CREATE TRIGGER result_seal_update
BEFORE UPDATE ON result
FOR EACH ROW
WHEN (SELECT status FROM run WHERE id = NEW.run_id) != 'running'
BEGIN
    SELECT RAISE(ABORT, 'run is sealed: cannot write result');
END;

CREATE TRIGGER score_seal_insert
BEFORE INSERT ON score
FOR EACH ROW
WHEN (SELECT status FROM run WHERE id = NEW.run_id) != 'running'
BEGIN
    SELECT RAISE(ABORT, 'run is sealed: cannot write score');
END;

CREATE TRIGGER score_seal_update
BEFORE UPDATE ON score
FOR EACH ROW
WHEN (SELECT status FROM run WHERE id = NEW.run_id) != 'running'
BEGIN
    SELECT RAISE(ABORT, 'run is sealed: cannot write score');
END;
