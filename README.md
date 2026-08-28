# fireassay — M1 + M2 + M3

An evaluation integrity layer: content-addressed suites that refuse to report unsound
comparisons, and a harness that proves it can fail. This covers **M1, M2 and M3** — the
deterministic spine, controls and mutation scores, and question generation / deterministic
filtering / human curation. Read this section honestly before assuming fireassay does more
than it does.

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

## What M2 adds

The centre of the product (fireassay-SPEC.md §2, pillar 2): **a green evaluation is only
evidence if the harness can be shown to go red, in both directions, with a number.**

- **Four deterministic negative controls** (`controls/`): `no_retrieval`, `shuffled_gold`,
  `corpus_ablation`, `identical_config`. Each wraps or mutates the system/data, proves a
  **writable twin** (the unmodified measurement genuinely works) before trusting its own
  silence, and asserts specific **cause** facts (e.g. `recall == 0.0` *and* `len(retrieved) ==
  0` *and* `judged_fraction == 0.0`), never just "the number dropped".
- **`shuffled_gold`'s chance band is measured, not assumed — and measured as an estimate, not a
  single draw.** A fixed threshold (e.g. `max: 0.05`) silently encodes one corpus size — on this
  repo's 12-document fixture with `top_k=5`, ~40% of the corpus is retrieved on every query, so
  a *meaningless* (permuted) gold span still overlaps something roughly a third to a half of the
  time (measured: mean ≈0.376, sd ≈0.105 across 300 draws, only 17 distinct values — `recall@5`
  over 14 gold-bearing questions is a coarse, discrete statistic); chance would only look like
  `0.05` on a much larger corpus (~1,181 docs ≈ 0.004). Two narrower fixes were tried and
  measured, not assumed, to still be wrong: comparing one permutation's recall against the raw
  sample maximum of `n_seeds` calibration draws has a `1/(n_seeds+1)` false-alarm rate on a
  *perfectly healthy* system, and a normal-theory tolerance interval (`chance_level + k *
  chance_sd`, `k=3`) still measured a 1.0% false-alarm rate across 200 independently-seeded
  suites, because this fixture's recall distribution is coarse and discrete rather than smoothly
  normal. The control instead averages `m_primary` independent permutations into one estimate
  and compares *that* against the calibration mean using the standard error of the difference
  between two means (`chance_sd * sqrt(1/m_primary + 1/n_seeds)`) — simulated at a 0.0000%
  false-alarm rate over 40,000 trials against the fixture's real empirical distribution, and
  confirmed empirically (not just simulated) at 200/200 PASSED across independently-seeded
  suites. PASSES only when the twin clears the measured chance level by `min_margin` *and* the
  averaged primary estimate stays within the measured tolerance — the same "measure the
  baseline, never assume it" principle a paired-bootstrap gate applies to its own noise floor,
  applied a second time to the measurement of the noise floor itself once the first attempt
  turned out to still be an assumption in disguise.
- **`judge_calibration` and `null_questions` are registered now, always `NOT_RUN`.**
  `judge_calibration` needs a judge (M4). `null_questions` was originally specified as a
  retrieval-shaped check ("does a question with no answer retrieve nothing relevant"), but that
  property is actually about **abstention** — whether a system that *generates* an answer
  correctly declines — and M1/M2's system under test (`BM25System`) is retrieval-only and never
  makes that choice. An earlier implementation forced a retrieval-only proxy onto it and it
  reported a confident `recall_at_k=1.0` against an expected `0.0`, for no reason connected to
  whether retrieval was healthy — exactly the "measurement with no external referent" failure
  mode this project exists to catch (fireassay-SPEC.md §1). `null_questions` will get a real
  implementation once a generating system exists to measure abstention against (M4).
- **`NOT_RUN` is never a pass** (`admissibility.py`): a run is admissible iff it has zero
  invariant violations and every control is `PASSED` (or `NOT_RUN` and explicitly allowed via
  `--allow-not-run`). `fireassay controls run` exits non-zero on any `FAILED` or disallowed
  `NOT_RUN`, regardless of how well the real configs scored.
- **System mutation** (`mutation/`): five operators (`drop_results`, `truncate_topk`,
  `shuffle_topk`, `corrupt_query`, `swap_ranking`) degrade the system under test
  deterministically; `ThresholdDetector` (naive, explicitly labelled — the real statistical gate
  is M5) decides whether each mutant is distinguishable from a baseline; `gate_mutation_score =
  killed / (total - equivalent)`. Equivalent mutants are **computed, not assumed** — every
  exclusion carries a `equivalent_reason`, printed in every report.
- **`harness_mutation_score`**: `cosmic-ray` (dev-only dependency) mutates fireassay's *own*
  source (`integrity.py`, `hashing.py`, `admissibility.py`, `score/`, `controls/`) — a different
  question from `gate_mutation_score` ("does this eval setup catch injected regressions in the
  *system under test*"): "do our tests catch bugs in *fireassay itself*". `make mutants` runs
  it; see `cosmic-ray.toml` and `.github/workflows/mutants.yml` (scheduled, not per-PR — slow).
  Target: 80%+ on the modules above; lower is a finding to report, not to quietly lower.

```
fireassay controls run     --matrix configs/matrix.example.yaml --corpus corpus.jsonl --db bench.db \
                            [--allow-not-run judge_calibration --allow-not-run null_questions]
fireassay controls show    <run_id> --db bench.db
fireassay mutate           --suite support-kb@1.0.0 --config <config_id> \
                            --operators configs/mutants.example.yaml --corpus corpus.jsonl --db bench.db
fireassay mutation score   --mutation-run <id> --db bench.db
```

`controls run` exits **2** on any control `FAILED`, **3** on any disallowed `NOT_RUN`. `mutate`
always exits **0** — a low `gate_mutation_score` is a finding to report, not a command failure.
On the M1 fixture (retrieval-only `BM25System`), `judge_calibration` and `null_questions` are
*always* `NOT_RUN`, so `controls run` needs both named in `--allow-not-run` to exit 0.

## What M3 adds

Turning a corpus into a curated golden set (fireassay-SPEC.md §2, pillar 3), with a rubric and
a measured error rate — not a funnel that trusts its own inputs.

- **Generation** (`generate/`, `llm/`) is the only part of fireassay allowed to call an LLM, via
  a thin `OllamaClient` (`urllib.request` only, no new HTTP dependency) that pins a model by
  **digest, from `GET /api/tags`, never `/api/show`** (verified against a live server: `/api/show`
  carries no `digest` key at all) and never by tag, and that **raises rather than returns a
  partial result** when every retry produces invalid JSON. Every LLM response is content-addressed
  on `(digest, prompt)` in a `ResponseCache`, so tests replay fixtures and never touch a live
  model. A candidate's evidence span is resolved by locating its quote **verbatim, under
  whitespace normalisation only** (never fuzzy matching) in the source document; a quote that
  does not appear, or appears more than once, is discarded and counted, never guessed at.
- **Filtering** (`filter/`) is entirely deterministic and LLM-free: degeneracy, self-containment,
  near-duplication (token-set Jaccard via the one shared tokenizer), an `unretrievable` check, and
  a per-`(qtype, difficulty)` balance cap, in that fixed order, with every candidate's full
  stage-by-stage trail persisted so the funnel reconciles exactly (`generated == kept + every
  rejection reason`).
  **`unretrievable` replaces an earlier per-word document-frequency check (`TOO_GENERIC`)**,
  measured on the real 1,181-document gov.uk corpus to have a ~50% false-positive rate: on a
  topically narrow corpus, ordinary words are common simply because the whole corpus shares a
  topic, so "every content word is common" rejects specific, well-formed questions. `unretrievable`
  instead asks the more direct question an answerability filter asks — can the retriever that will
  eventually have to answer this find its own source document at all? — by running the question
  through the BM25 retriever already in this repo and rejecting it only if its own source document
  fails to appear in the top `unretrievable_top_n` (default 50 of ~24,584 chunks — a floor, not a
  selection criterion) of the whole corpus. This is defensible for filtering configs that are
  later evaluated with the same retriever family only because it is (1) a wide floor, not a
  ranking criterion, and (2) applied identically to every config, so it cannot differentially
  favour one. **The bias that remains is real and is not hidden: the curated set will
  under-represent questions that require semantic rather than lexical matching.** When a second,
  differently-biased retriever family exists in this repo, filtering with it instead of the one
  under evaluation is the correct fix; until then this is a documented limitation of the set's
  composition, not of any comparison drawn from it.
- **Curation** (`curate/`) is the non-interactive core (`curate next` / `curate submit`) a future
  TUI (M3b) will drive with no new logic, only keystrokes: a six-question rubric, a queue that
  interleaves invisible known-bad honeypots and double-reviews, Krippendorff's α (ordinal for the
  two ordinal rubric fields, nominal for the four booleans, missing observations passed through as
  `NaN` rather than imputed), and a funnel/coverage/difficulty-validation report.
  **Agreement and accuracy are different instruments and neither substitutes for the other**: α
  says whether curators agree with each other, honeypots say whether a curator is right against a
  known-bad item planted invisibly in the queue. **"Unmeasurable" is a state, not a number**:
  `krippendorff.alpha` can raise (two curators agreeing on every shared item is an entirely
  ordinary case) or return `NaN` (sparse, missing-heavy data) without raising — both are reported
  as a distinct `UNMEASURABLE` status, never silently compared against the low-agreement threshold
  as if they were a real, if low, α.
- **`gold_doc_rank`** — the BM25 rank at which a candidate's own source document is first found,
  measured once at generation time — is recorded as a feature alongside the lexical ones, and
  `curate report`'s difficulty-validation correlation checks the generator's *proposed* difficulty
  label against it: the spike found a generator's difficulty labels can be **inverted** relative to
  what actually drives retrieval (`easy` probes scored 0.176 recall, `hard` scored 0.588, because
  what drove it was document-title overlap, not passage overlap). A difficulty label that does not
  correlate with anything measurable is a label, not a difficulty.

```
fireassay generate  --corpus corpus.jsonl --model qwen2.5:7b --n 3000 --cache-dir .cache/llm --db bench.db
fireassay filter    --batch <batch_id> --config configs/filter.example.yaml --corpus corpus.jsonl --db bench.db
fireassay curate next    --curator alice --db bench.db
fireassay curate submit  verdict.json --db bench.db
fireassay curate report  --db bench.db [--corpus corpus.jsonl] [--suite support-kb@1.0.0]
fireassay suite freeze --name support-kb --version 1.0.0 --db bench.db --from-curated
```

## What M1+M2+M3 are honestly *not*

The following are explicitly out of scope and **not built**, regardless of what the parent
spec (`../fireassay-SPEC.md`) describes for the finished project:

- **No curation TUI** (M3b will drive `curate next`/`curate submit` interactively; both are
  already fully usable non-interactively). No crowd-kit/Dawid-Skene aggregation — pointless with
  one curator, revisit when there are several.
- **No LLM judges** (`correctness`, `groundedness`), no `judge_calibration` control with a real
  judge — five of the eventual eight metrics are built; the three that need a judge, and the
  sixth control, are M4.
- **No CJE integration**, no calibration, no pooling.
- **No release gate** (`GateDetector`, paired bootstrap significance testing, Holm–Bonferroni,
  power check, `confseq` re-runs), no GitHub Action gating PRs — M5.
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

Runtime dependencies are `pydantic>=2`, `typer`, `pyyaml`, `numpy`, `rich`, plus **`krippendorff`**
(M3's one new dependency — no pandas, no scikit-learn). Dev-only: `pytest`, `pytest-cov`, `ruff`,
`mypy`, `cosmic-ray` (the last is never imported at runtime; it powers `make mutants` only). No
SQLAlchemy, no ORM — SQLite is stdlib `sqlite3`; Ollama access is stdlib `urllib.request`, no HTTP
library. No LLM call and no network anywhere in M1 or M2, and nowhere in M3 outside `generate/`.

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
make mutants    # cosmic-ray over integrity/hashing/admissibility/score/controls (slow; M2 §8)
```

## Repo layout

```
src/fireassay/
├── cli.py               # typer CLI
├── hashing.py            # canonical_json, content_hash, and the three id schemes
├── models.py              # pydantic v2 models (Question, Score, Run, ...)
├── integrity.py           # assert_comparable — the refusal engine
├── admissibility.py        # M2: assess() — controls + invariants -> admissible verdict
├── config.py                 # MatrixSpec, expand()
├── text.py                    # the one tokenizer
├── runner.py                    # sequential run_matrix, run_once (M2)
├── compare.py                     # leaderboard aggregation
├── store/                           # migrations/ (0001 M1, 0002 M2, 0003 M3), db.py
├── system/                           # bm25.py, corpus.py, base.py (System protocol)
├── score/                             # retrieval, latency, cost, policy, abstention, invariants
├── controls/                           # M2: 5 deterministic controls + expected.yaml + registry
├── mutation/                            # M2: operators, ThresholdDetector, score, run_mutation
├── llm/                                  # M3: OllamaClient, ResponseCache — the only LLM boundary
├── generate/                              # M3: candidate generation, span resolution, features
├── filter/                                 # M3: deterministic filter pipeline (no LLM)
├── curate/                                  # M3: rubric, queue, honeypots, agreement, report
└── report/                                   # text.py — rich table rendering (M1 + M2 + M3)
tests/
├── fixtures/            # corpus.jsonl (12 docs), questions.jsonl (20 questions), llm/ (cache fixtures)
└── test_*.py
```

## Prices are per million tokens

`ScoringContext.price_in_per_mtok` / `price_out_per_mtok` are USD **per million tokens**, not
per token and not per thousand. Getting this unit wrong silently corrupts every cost number in
the repo — see the docstring on `score/cost.py`.
