-- fireassay M3 addendum: stratified generation target cell.
--
-- Adds candidate.target_qtype / candidate.target_difficulty: the
-- (qtype, difficulty) cell generate.pipeline explicitly asked the model
-- for, recorded alongside the model's own *proposed* qtype/difficulty
-- (the existing candidate.qtype/difficulty columns). Pre-flight
-- measurement on real generation found an unstratified prompt collapses
-- to ~2 of 21 taxonomy cells -- see generate/pipeline.py's module
-- docstring for the full finding and generate/prompts.py for the
-- stratified prompt itself.
--
-- DEFAULT '' exists only to satisfy SQLite's NOT NULL requirement on
-- ALTER TABLE ADD COLUMN against a (possibly non-empty) existing table;
-- '' is not a valid QType/Difficulty value and no code path ever writes
-- it deliberately -- any candidate row predating this migration was
-- generated before stratification existed and should be regenerated,
-- not read back through Store as if it had a real target cell.

ALTER TABLE candidate ADD COLUMN target_qtype TEXT NOT NULL DEFAULT '';
ALTER TABLE candidate ADD COLUMN target_difficulty TEXT NOT NULL DEFAULT '';
