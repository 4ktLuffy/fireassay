# fireassay — M1

An evaluation integrity layer: content-addressed suites that refuse to report unsound
comparisons. This is **M1 only** — the deterministic spine. Read this section honestly before
assuming fireassay does more than it does.

## What M1 actually is

- A SQLite store for questions, evidence spans, suites, configs, runs, results and scores —
  **append-only**: once a run is sealed, writes to it are rejected in Python
  (`RunSealedError`) and at the database level (SQL triggers), not just by convention.
- Content-addressed suites and configs (`hashing.py`): a question's id, a suite's
  `suite_hash`, and a config's `config_hash` are all pure functions of their content, sha256
  under a versioned scheme prefix.
- A refusal rule (`integrity.py`): `compare` and `diff` refuse to report a comparison across
  runs that differ in suite, scorer versions, or environment fields marked `affects_results`,
  or that are incomplete, empty, or individually inadmissible — unless you pass `--force`, in
  which case every render is stamped `UNSOUND COMPARISON`. Six refusal codes: `SUITE_MISMATCH`,
  `SCORER_MISMATCH`, `ENV_MISMATCH`, `RUN_INCOMPLETE`, `EMPTY_RUN`, `RUN_INADMISSIBLE`.
- A config matrix (`config.py`): a Cartesian product over named axes, with `include`/`exclude`.
- A real system under test (`system/bm25.py`): a pure-Python Okapi BM25 retriever, no API key,
  no network call. This is deliberate — a stranger with nothing but this repo can reproduce
  every retrieval number fireassay reports.
- Five deterministic scorers (`score/`): `retrieval`, `latency`, `cost`, `policy`,
  `abstention`. **None of them call an LLM.** That is the credibility anchor of this
  milestone.
- A leaderboard (`compare.py`) and a CLI (`cli.py`) to drive all of the above.

## What M1 is honestly *not*

The following are explicitly out of scope and **not built**, regardless of what the parent
spec (`../fireassay-SPEC.md`) describes for the finished project:

- **No question generation, no curation TUI, no reject taxonomy, no inter-annotator
  agreement.** Questions are imported from a JSONL file you provide.
- **No negative controls** (the six controls in the parent spec's Pillar 2) beyond the metric
  self-consistency invariants described below, which check the *scorer*, not the *system*.
- **No LLM judges** (`correctness`, `groundedness`) — five of the eventual eight metrics are
  built; the three that need a judge are M4.
- **No CJE integration**, no calibration.
- **No release gate**, no paired bootstrap significance testing, no GitHub Action.
- **No HTML report.** Rendering is `rich` tables to a terminal (`report/text.py`).
- **No corpus scraping.** You provide `corpus.jsonl`.

If you came here looking for the finished product described in `fireassay-SPEC.md`, it is not
here yet. What *is* here is real: a working store, a real refusal mechanism, a real retriever,
and real deterministic metrics — not a mockup of any of them.

## Beyond the original M1 spec

Several rounds of correction were applied to the original M1 spec during implementation; the
following exist because of them, not because the original spec asked for them:

- **Retrieval relevance is judged by character-range overlap, not `chunk_id` equality**
  (`score/retrieval.py`). Chunking is a config axis; matching on `chunk_id` made recall
  incomparable across chunk sizes. `RetrievedChunk` now carries `char_start`/`char_end`.
- **`retrieval.judged_fraction@k` and `retrieval.bpref`** exist because gold evidence spans are
  known-incomplete: a chunk that genuinely answers a question but was never annotated as gold
  is scored as a miss by recall/nDCG/MRR. `judged_fraction` makes that incompleteness visible;
  `bpref` (Buckley & Voorhees 2004) is designed for incomplete judgments — it is `1.0` whenever
  anything relevant is retrieved in M1, because M1 has no data source for negative judgments
  yet (`ScoringContext.judged_nonrelevant_spans_by_question` is always empty in a real run);
  the formula is implemented and tested now so a later milestone's pooling work is a drop-in.
- **A missing Score is never a pass.** `AbstentionScorer` emits `abstention.wrongly_abstained`
  for every answerable question an answer-generating system abstains on, so an
  abstain-everything system cannot win by having nothing scored against it.
  `compare.MetricSummary.applicable_n` surfaces when a metric's mean is based on a shrinking
  subset of its population. Conversely, a retrieval-only system (`System.generates_answers ==
  False`, e.g. `BM25System`) gets **no** abstention metrics at all — not `0.0`/`1.0` — because
  it never made an abstain/answer choice for either metric to measure.
- **Metric self-consistency invariants** (`score/invariants.py`) run automatically at the end of
  every run: value ranges, `recall@k` monotonicity, `mrr >= recall@1`, and consistency between
  `ndcg@k`/`mrr` and `recall@k` on whether anything relevant was found at all. (An earlier
  `ndcg@k` monotonicity rule was removed — it was mathematically wrong: IDCG@k grows with k, so
  a perfect `ndcg@1` can legitimately fall by `ndcg@3`.) A run that violates one is marked
  `admissible=False` and is a first-class refusal ground: `integrity.assert_comparable` raises
  `RUN_INADMISSIBLE` for it.
- **BM25 ranking ties are broken by `(doc_id, chunk_id)`, never by score alone**, so a fully
  tied ranking is reproducible regardless of corpus load order.
- **The corpus is content-hashed from raw documents** (`system/corpus.corpus_hash`, hashing
  `doc.text` only — not chunk boundaries) into `run.env_json["affects_results"]["corpus_hash"]`,
  so a corpus that changes underneath a suite trips `ENV_MISMATCH` instead of silently comparing
  against stale evidence, while two configs that differ only in chunk size still hash identically
  and remain comparable — chunking is a property of the *config* (already in `config_hash`), not
  of the corpus.
- **`compare.leaderboard` supports `split_by=("qtype", "difficulty")`** alongside
  `split_by_provenance`, so a config that wins overall but fails every `multi_hop` question is
  visible in the leaderboard rather than averaged away.
- **Tokenisation is centralised** in `text.py`; `BM25System` imports it rather than defining its
  own regex, so a future text-matching component (dedup, curation) cannot silently diverge from
  what the retriever considers a token.

## Install

```
pip install -e ".[dev]"
```

Dependencies are exactly `pydantic>=2`, `typer`, `pyyaml`, `numpy`, `rich` (dev:
`pytest`, `pytest-cov`, `ruff`, `mypy`). No SQLAlchemy, no ORM, no network library — SQLite is
stdlib `sqlite3`.

## Quick tour

```
fireassay init                --db bench.db
fireassay questions import    questions.jsonl --db bench.db
fireassay suite freeze        --name support-kb --version 1.0.0 --db bench.db
fireassay suite show          support-kb@1.0.0 --db bench.db
fireassay run                 --matrix configs/matrix.example.yaml --corpus corpus.jsonl --db bench.db
fireassay compare             --suite support-kb@1.0.0 --db bench.db [--force] [--by-provenance] [--split-by qtype]
fireassay diff                --base <run_id> --head <run_id> --db bench.db [--force]
```

`compare`/`diff` exit **2** when the comparison is refused as unsound (`ComparisonRefused`),
distinct from exit 1 for a plain CLI usage error.

`make demo` runs this whole sequence against the fixture corpus/questions in `tests/fixtures/`.

## Tests

```
make test       # pytest, coverage on
make lint       # ruff
make typecheck  # mypy --strict on src/
```

## Repo layout

```
src/fireassay/
├── cli.py               # typer CLI
├── hashing.py            # canonical_json, content_hash, and the three id schemes
├── models.py              # pydantic v2 models (Question, Score, Run, ...)
├── integrity.py           # assert_comparable — the refusal engine
├── config.py               # MatrixSpec, expand()
├── text.py                  # the one tokenizer
├── runner.py                  # sequential run_matrix
├── compare.py                   # leaderboard aggregation
├── store/                     # schema.sql, migrations/, db.py (Store)
├── system/                     # bm25.py, corpus.py, base.py (System protocol)
├── score/                       # retrieval, latency, cost, policy, abstention, invariants
└── report/                       # text.py — rich table rendering
tests/
├── fixtures/            # corpus.jsonl (12 docs), questions.jsonl (20 questions)
└── test_*.py
```

## Prices are per million tokens

`ScoringContext.price_in_per_mtok` / `price_out_per_mtok` are USD **per million tokens**, not
per token and not per thousand. Getting this unit wrong silently corrupts every cost number in
the repo — see the docstring on `score/cost.py`.
