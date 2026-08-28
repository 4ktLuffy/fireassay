-- Canonical, human-readable copy of the current fireassay schema.
--
-- This file is documentation: the source of truth that Store.migrate()
-- actually applies is migrations/0001_initial.sql (and any later
-- 000N_*.sql files). Keep this file in sync with the latest migration
-- whenever the schema changes, so a reader can see the full current
-- schema in one place without reconstructing it by replaying migrations.

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
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    version        TEXT NOT NULL,
    suite_hash     TEXT NOT NULL,
    frozen_at      TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    agreement_json TEXT NOT NULL DEFAULT '{}',   -- added in migration 0003 (M3)
    UNIQUE (name, version)
);

CREATE TABLE suite_question (
    suite_id    TEXT NOT NULL REFERENCES suite(id),
    question_id TEXT NOT NULL REFERENCES question(id),
    PRIMARY KEY (suite_id, question_id)
);

CREATE TABLE config (
    id          TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL UNIQUE,
    label       TEXT,
    spec_json   TEXT NOT NULL
);

CREATE TABLE run (
    id          TEXT PRIMARY KEY,
    suite_id    TEXT NOT NULL REFERENCES suite(id),
    suite_hash  TEXT NOT NULL,
    config_id   TEXT NOT NULL REFERENCES config(id),
    config_hash TEXT NOT NULL,
    env_json    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    admissible                 INTEGER NOT NULL DEFAULT 1,
    invariant_violations_json  TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE result (
    run_id         TEXT NOT NULL REFERENCES run(id),
    question_id    TEXT NOT NULL REFERENCES question(id),
    answer         TEXT,
    abstained      INTEGER NOT NULL,
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
    scorer      TEXT NOT NULL,
    rationale   TEXT,
    PRIMARY KEY (run_id, question_id, metric, scorer)
);

-- See migrations/0001_initial.sql for the append-only trigger definitions
-- (result_seal_insert, result_seal_update, score_seal_insert,
-- score_seal_update). See migrations/0002_controls_mutation.sql for
-- control_check / mutation_run / mutant (M2), and
-- migrations/0003_generation_curation.sql for candidate / filter_result /
-- queue_item / decision (M3) and their own append-only triggers.

CREATE TABLE candidate (
    id               TEXT PRIMARY KEY,
    batch_id         TEXT NOT NULL,
    text             TEXT NOT NULL,
    qtype            TEXT NOT NULL,
    difficulty       TEXT NOT NULL,
    reference_answer TEXT NOT NULL,
    quote            TEXT NOT NULL,
    source_doc_id    TEXT NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    features_json    TEXT NOT NULL,
    model_digest     TEXT NOT NULL,
    prompt_hash      TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE TABLE filter_result (
    candidate_id TEXT NOT NULL,
    stage        TEXT NOT NULL,
    kept         INTEGER NOT NULL,
    reason       TEXT,
    checked_at   TEXT NOT NULL,
    PRIMARY KEY (candidate_id, stage)
);

CREATE TABLE queue_item (
    id                       TEXT PRIMARY KEY,
    queue_id                 TEXT NOT NULL,
    curator_id               TEXT NOT NULL,
    candidate_id             TEXT NOT NULL,
    position                 INTEGER NOT NULL,
    is_honeypot              INTEGER NOT NULL,
    honeypot_expected_reason TEXT,
    is_double_review         INTEGER NOT NULL
);

CREATE TABLE decision (
    id            TEXT PRIMARY KEY,
    candidate_id  TEXT NOT NULL,
    curator_id    TEXT NOT NULL,
    decision      TEXT NOT NULL,
    reject_reason TEXT,
    rubric_json   TEXT NOT NULL,
    edited_text   TEXT,
    edited_answer TEXT,
    notes         TEXT,
    duration_ms   INTEGER NOT NULL,
    decided_at    TEXT NOT NULL
);
