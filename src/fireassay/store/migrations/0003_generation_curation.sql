-- fireassay M3 schema additions: generation, filtering, curation.
--
-- Adds `candidate`, `filter_result`, `queue_item`, `decision` (M3-SPEC.md
-- §5), plus `suite.agreement_json` (M3-SPEC.md §4's Krippendorff's-α flag,
-- mirroring `run.admissibility_json`'s M2-era ALTER TABLE).
--
-- `filter_result.candidate_id` is deliberately **not** a `REFERENCES
-- candidate(id)` foreign key: `generate/`'s span-resolution stage
-- (`not_a_question`, `span_resolution`) assigns a candidate_id to every
-- raw LLM proposal *before* knowing whether it survives, and a proposal
-- that is discarded (NOT_A_QUESTION / QUOTE_NOT_FOUND / QUOTE_AMBIGUOUS)
-- never gets a `candidate` row at all — only a `filter_result` row
-- recording why. This is what lets the funnel reconcile exactly
-- (`generated == kept + every rejection reason`, see curate/funnel.py):
-- `generated` is counted directly off `filter_result`, independent of
-- whether a given candidate_id ever made it into `candidate`.
--
-- Append-only: `candidate`/`filter_result`/`queue_item` are only ever
-- INSERTed by their respective pipelines and never revisited; `decision`
-- is explicitly append-only by design (M3-SPEC.md §5: "a decision may not
-- be overwritten -- a changed mind is a new row"). All four get the same
-- UPDATE/DELETE-blocking triggers M2's `control_check`/`mutation_run`/
-- `mutant` already established for literal (not run-status-keyed)
-- immutability.

ALTER TABLE suite ADD COLUMN agreement_json TEXT NOT NULL DEFAULT '{}';

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

CREATE INDEX candidate_batch_id_idx ON candidate (batch_id);

CREATE TABLE filter_result (
    candidate_id TEXT NOT NULL,
    stage        TEXT NOT NULL,
    kept         INTEGER NOT NULL,   -- 0/1
    reason       TEXT,               -- NULL iff kept = 1
    checked_at   TEXT NOT NULL,
    PRIMARY KEY (candidate_id, stage)
);

CREATE TABLE queue_item (
    id                       TEXT PRIMARY KEY,
    queue_id                 TEXT NOT NULL,
    curator_id               TEXT NOT NULL,
    candidate_id             TEXT NOT NULL,
    position                 INTEGER NOT NULL,
    is_honeypot              INTEGER NOT NULL,   -- 0/1
    honeypot_expected_reason TEXT,               -- NULL iff is_honeypot = 0
    is_double_review         INTEGER NOT NULL    -- 0/1
);

CREATE INDEX queue_item_curator_idx ON queue_item (curator_id, position);

CREATE TABLE decision (
    id            TEXT PRIMARY KEY,
    candidate_id  TEXT NOT NULL,
    curator_id    TEXT NOT NULL,
    decision      TEXT NOT NULL,   -- accept | edit | reject
    reject_reason TEXT,            -- NOT NULL iff decision = 'reject'
    rubric_json   TEXT NOT NULL,
    edited_text   TEXT,
    edited_answer TEXT,
    notes         TEXT,
    duration_ms   INTEGER NOT NULL,
    decided_at    TEXT NOT NULL
);

CREATE INDEX decision_candidate_curator_idx ON decision (candidate_id, curator_id, decided_at);

CREATE TRIGGER candidate_no_update BEFORE UPDATE ON candidate
BEGIN SELECT RAISE(ABORT, 'candidate is append-only: rows cannot be updated'); END;
CREATE TRIGGER candidate_no_delete BEFORE DELETE ON candidate
BEGIN SELECT RAISE(ABORT, 'candidate is append-only: rows cannot be deleted'); END;

CREATE TRIGGER filter_result_no_update BEFORE UPDATE ON filter_result
BEGIN SELECT RAISE(ABORT, 'filter_result is append-only: rows cannot be updated'); END;
CREATE TRIGGER filter_result_no_delete BEFORE DELETE ON filter_result
BEGIN SELECT RAISE(ABORT, 'filter_result is append-only: rows cannot be deleted'); END;

CREATE TRIGGER queue_item_no_update BEFORE UPDATE ON queue_item
BEGIN SELECT RAISE(ABORT, 'queue_item is append-only: rows cannot be updated'); END;
CREATE TRIGGER queue_item_no_delete BEFORE DELETE ON queue_item
BEGIN SELECT RAISE(ABORT, 'queue_item is append-only: rows cannot be deleted'); END;

CREATE TRIGGER decision_no_update BEFORE UPDATE ON decision
BEGIN SELECT RAISE(ABORT, 'decision is append-only: a changed mind is a new row, never an edit'); END;
CREATE TRIGGER decision_no_delete BEFORE DELETE ON decision
BEGIN SELECT RAISE(ABORT, 'decision is append-only: a changed mind is a new row, never an edit'); END;
