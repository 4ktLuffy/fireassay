-- fireassay M3b addendum: chunk-level provenance on candidate, for
-- incremental resume (M3b-SPEC.md Part 1).
--
-- Adds candidate.chunk_id: the specific chunk a candidate was generated
-- from. `source_doc_id` alone is not enough to resume safely -- several
-- chunks share one document, so "skip any chunk whose document already
-- has a candidate" would silently skip chunks of the same document that
-- were never actually attempted. `Store.candidate_source_chunks()` reads
-- this column to build the skip set `generate.pipeline.generate_candidates`
-- checks before spending an LLM call (or, on a warm response cache, the
-- ~1s/candidate replay cost measured in spike/RESULTS.md and M3b-SPEC.md
-- Part 1) on a chunk that already has >= 1 candidate row.
--
-- DEFAULT '' for the same reason migration 0004 defaults
-- target_qtype/target_difficulty to '': SQLite's `ALTER TABLE ADD COLUMN`
-- requires a default against a (possibly non-empty) existing table.
-- Unlike 0004, `''` is not treated as an error state here -- a
-- pre-migration-0005 candidate row simply does not contribute to the
-- resume skip set (`Store.candidate_source_chunks()` excludes empty
-- values), so a run resumed against a database that predates this column
-- may re-process a handful of chunks generated before it existed. That is
-- a one-time, bounded cost (at most one extra generation attempt per
-- pre-existing chunk), not a correctness hazard the way a stratification-
-- less row is (`PreStratificationCandidateError`), so
-- `ResolvedCandidate.chunk_id` defaults to `""` rather than being required
-- the way `target_qtype`/`target_difficulty` are.

ALTER TABLE candidate ADD COLUMN chunk_id TEXT NOT NULL DEFAULT '';

CREATE INDEX candidate_chunk_id_idx ON candidate (chunk_id);
