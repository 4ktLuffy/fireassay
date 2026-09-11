# fireassay

An evaluation integrity layer for RAG and LLM golden sets. The thesis, in one line:

> **A green evaluation is only evidence if the harness can be shown to go red.**

fireassay is the machinery that makes that showable. Content-addressed suites that refuse to
report a comparison across mismatched questions, scorers or environments. Negative controls
that must fail, with a measured chance band, before a run is admissible. An item-analysis
tool whose detectors are scored against blind human labels and shipped as `UNVALIDATED` when
they lose to a random draw. And a release gate that computes the smallest regression a suite
can resolve and refuses any threshold below it, instead of rubber-stamping a green.

**Every number in this file is a claim in [`EVIDENCE.md`](EVIDENCE.md)**, mapped to the
committed artifact it is read from and the command that recomputes it. `make evidence`
recomputes all of them and fails on any drift; it never rewrites a number to match. CI runs it
on every push, together with the gate demo below.

## What was measured

All of it on one corpus (1,181 gov.uk documents, [OGL v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/))
and one generated candidate set (2,364 questions from a local `qwen2.5:7b`). Human labels are
Henos's, blind, on a uniform-random stratum.

| Finding | Number | Claim |
|---|---|---|
| Generated items that are broken (blind human review, 132 uniform-random labels) | **17.4%** [11.9, 24.8]; 40.9% if "needs rewrite" counts | `labels.base_rate`, `labels.lenient_rate` |
| Defect rate over all 2,364 items, judge + 132 labels combined (prediction-powered inference) | **20.7%** [15.3, 26.1] | `ppi.defect_rate` |
| Best judge vs human labels: gpt-5.4/low | precision 0.652, recall 0.652 | `judge.gpt54` |
| Local judge: granite4 7B | precision 0.571, recall 0.174 | `judge.granite4` |
| DeBERTa-v3-large NLI as a judge | precision 0.224 | `judge.nli` |
| No-model token-overlap floor every judge has to beat | precision 0.75 (n=8) | `judge.token_overlap` |
| `mislabel_suspect` detector vs a random draw of the same size | 0.219 vs 0.250, Fisher p=0.79 — **shipped as UNVALIDATED** | `detector.mislabel_suspect` |
| Split-half reliability of per-item statistics, 54-config panel | 0.503 (usable); 0.286 on the 18-config panel (too few systems) | `panel.reliability`, `panel18.reliability` |
| Monotonicity inside one retriever family | 0 violations in 42,552 pairs — a `top_k` ladder cannot distinguish strength from breadth | `panel.monotonic` |
| Closed-book, forced-guess: items answerable with no retrieval | 2 of 149 (1.3%) | `closedbook.forced` |
| **Dense retrieval vs BM25**, 2,364 items, matched chunking and top_k | dense loses all 6 pairs: **-8.5 to -6.3 points** at 1024/128 (gate **blocks**), -2.0 to -1.1 at 512/128 | `dense.vs_bm25` |
| The frozen panel's lost `correct` definition, recovered by reproduction | 14,184 cells regenerated, **0 mismatched** | `dense.bm25_twin` |
| **The gate blocking a planted regression** | exit 4 on `drop_results`; exit 0 on a benign change; **exit 5 (refused)** on a real 0.39 drop asked at threshold 0.3, because 14 items resolve only 0.35 | `gate.*` |

Three of those rows are why this project exists rather than a metric library. The calibration
set's first act was to kill one of the tool's own detectors: `mislabel_suspect` looked like a
weak-but-real detector at 0.219 precision until it was put beside the 0.25 a random draw
scores. And the gate's third case is a regression that is real, larger than the threshold
asked for, and still refused, because a threshold the suite cannot distinguish from noise is
not one a gate can honestly enforce. And the dense row is the result nobody sets out to get: a
2026 embedding model, run properly, beaten by a 1994 lexical baseline on every matched
configuration — which is only a finding at all because the baseline was there to lose to.

The full defect log, 50 entries, each a plausible number that was wrong, is in
`~/ObitosBrain` (private). The ones that changed the code are in the module docstrings.

## The gate (M5)

```bash
fireassay gate --base <run_id> --head <run_id> --db bench.db \
    --metric "retrieval.recall@5=0.05" --metric "latency.total_ms=50:lower"
```

For each `--metric NAME=THRESHOLD[:lower]` the gate pairs the two runs question by question,
runs a paired bootstrap, Holm-corrects the p-values across every metric checked, and computes
each metric's minimum detectable effect at the corrected alpha. It refuses before it decides:
the two runs must be soundly comparable (same suite, scorers and environment — the same rule
`compare` and `diff` apply), and every threshold must be at least the MDE. A verdict is
`block` only when the regression is at least THRESHOLD in size **and** significant after
correction. A `(base, head, metric)` triple can be recorded exactly once; the schema forbids
re-running a gate on the same pair until it comes back green.

| exit | meaning |
|---|---|
| 0 | every metric passed |
| 2 | comparison refused (`SUITE_MISMATCH`, `SCORER_MISMATCH`, `ENV_MISMATCH`, `RUN_INCOMPLETE`, `EMPTY_RUN`, `RUN_INADMISSIBLE`) |
| 4 | at least one metric **blocked** |
| 5 | a threshold is below the MDE — no verdict was reached; refusing is not passing |
| 6 | this pair has already been gate-checked |

`make gate-demo` (`tools/gate_demo.py`) shows all of it from an empty database on the
fixture suite, through the real CLI, and writes `run/gate_demo.json` so the three verdicts are
recomputable claims. `tools/mde.py` reports a suite's MDE as a distribution over system pairs
at candidate suite sizes, for pre-registering a threshold before the suite is built.

## What is in the box

| Pillar | Where | What it does |
|---|---|---|
| Suite integrity | `hashing.py`, `integrity.py`, `store/` | Content-addressed questions, suites and configs; append-only runs enforced by SQL triggers; `assert_comparable` with six refusal codes |
| Controls, both directions | `controls/`, `mutation/`, `admissibility.py` | Four deterministic negative controls with a *measured* chance band and a writable twin; five mutation operators and `gate_mutation_score`; `NOT_RUN` is never a pass |
| Curation with a measured error rate | `generate/`, `filter/`, `curate/` | Generation (the only LLM boundary, digest-pinned, content-cached), a deterministic filter with an exactly reconciling funnel, a six-question rubric, honeypots, Krippendorff's α with `UNMEASURABLE` as a state |
| Item analysis | `items/` | `fireassay items analyse` on any item × system response matrix — usable with no store, by someone who has never heard of fireassay; split-half reliability, discrimination, blind review decks, seeded corruptions, PPI |
| The gate | `gate.py`, `store/migrations/0006_gate.sql` | Paired bootstrap, Holm-Bonferroni, MDE refusal, once-only persistence |
| System under test | `system/` | Pure-Python BM25, TF-IDF and coverage (a panel of 54 configs) — no API key, no network, so a stranger can reproduce every lexical retrieval number here — plus a dense retriever over a local embedding model (below) |

Deterministic scorers (`score/`): `retrieval` (recall/nDCG/MRR by character-range overlap, plus
`judged_fraction` and `bpref` because gold spans are known-incomplete), `latency`, `cost`,
`policy`, `abstention`. None of them calls a model.

## Quick tour

```bash
pip install -e ".[dev]"
make demo          # init, import fixtures, freeze, run a 4-config matrix, compare
make gate-demo     # the gate: block, pass, refuse -- from an empty db
make test          # pytest, coverage on
make evidence      # recompute every claim in EVIDENCE.md
```

```bash
fireassay init                --db bench.db
fireassay questions import    questions.jsonl --db bench.db
fireassay suite freeze        --name support-kb --version 1.0.0 --db bench.db
fireassay run                 --matrix configs/matrix.example.yaml --corpus corpus.jsonl --db bench.db
fireassay compare             --suite support-kb@1.0.0 --db bench.db        # exit 2 if unsound
fireassay controls run        --matrix ... --corpus ... --db bench.db        # exit 2/3 on FAILED/NOT_RUN
fireassay mutate              --suite ... --config <id> --operators configs/mutants.example.yaml ...
fireassay items analyse       --matrix panel.csv                             # no db needed
fireassay gate                --base <run> --head <run> --metric "retrieval.recall@5=0.05" --db bench.db
```

`fireassay generate / filter / curate next / curate submit / curate tui / curate report` turn
a corpus into a curated set; see [`docs/MILESTONES.md`](docs/MILESTONES.md) for each one.

## Dense retrieval (M-DENSE), and what it cost

A fourth retriever sits beside `bm25`, `tfidf` and `coverage`: exact dot product over a local
embedding model (`qwen3-embedding:0.6b`, 1024-d, pinned by digest, served by Ollama). Exact,
not approximate — 12,513 chunks is one BLAS call, and an ANN index would add a recall loss
that then has to be measured inside the very comparison this exists to make.

```yaml
# configs/matrix.dense.yaml
axes:
  retriever: ["dense"]     # absent => "bm25", so every pre-existing config hash is unchanged
```

**The result is a loss.** Against BM25 at matched chunking and `top_k`, over the same 2,364
items, dense is worse on all six pairs — by 6.3 to 8.5 points at 1024/128, where the gate
blocks, and by 1.1 to 2.0 at 512/128, inside the pre-registered 0.05 threshold. The same
comparison asked at a 0.01 threshold is **refused**: this suite's minimum detectable effect at
n=2,364, after correcting for six comparisons, is 0.027.

**Before any of that was believed, the instrument was checked against the frozen panel.** The
script that produced `run/panel54_matrix.csv` was never committed and is lost, so the meaning
of its `correct` column existed nowhere. `tools/panel_dense.py --verify-bm25` recovers it by
reproduction: it regenerates the six BM25 columns and compares 14,184 cells against the frozen
file. They agree exactly, which is what licenses comparing anything to that panel.

```bash
ollama serve &
.venv/bin/python tools/embed_corpus.py --chunkings 1024-128,512-128   # ~64 min, 178 MB
.venv/bin/python tools/panel_dense.py --verify-bm25                   # must pass first
.venv/bin/python tools/panel_dense.py --retrievers bm25,dense --resume
.venv/bin/python tools/dense_vs_bm25.py
```

**Reproducing a dense number needs a model and about an hour**; reproducing every other number
in this repo needs only the repo. That asymmetry is why `.cache/embeddings/` is gitignored and
the *response matrix* is committed instead: `dense.vs_bm25` recomputes from
`run/panel_dense_matrix.csv`, never from the vectors. The cache is content-addressed on
`(model digest, chunk size, overlap, corpus hash)`, resumable one shard at a time, and refuses
a mismatched cache rather than reusing or silently rebuilding it. It is also what lets the
`identical_config` control pass at `max_abs_delta == 0.0` against a dense retriever, which a
live model call per build could never do.

## What this is honestly not, yet

- **The golden set is not curated.** The generation run produced 3,054 candidates; the
  curation TUI works; no suite has been frozen from curated decisions. The pre-registered
  acceptance check for this project names "a golden set that is really uncurated candidates"
  as a failure condition, and as of this README it is still triggered. The gate above is
  demonstrated on the 20-question fixture suite, not on a curated set.
- **No LLM judge scores answer correctness.** The three judges measured here score *item
  quality* (is this question answerable from its evidence). `correctness` and `groundedness`,
  and the `judge_calibration` control, remain unbuilt. `null_questions` is registered and
  always `NOT_RUN` because the system under test is retrieval-only.
- **No pooling, no CJE, no HTML report, single-turn only.** Rendering is `rich` tables.
- The curated set will under-represent questions needing semantic rather than lexical
  matching: the `unretrievable` filter uses the same BM25 family the panel evaluates. Stated,
  not hidden — and it is the strongest caveat on the dense result above. Items that only a
  semantic retriever could reach were filtered out *before* the panel saw them, by a lexical
  filter, so the comparison is run on ground the lexical baseline helped choose. The honest
  reading of "dense loses" is therefore "dense loses on this set, which was selected
  lexically", not "dense loses". Re-running the filter with the dense retriever in the
  `unretrievable` stage, and re-measuring, is the experiment that would settle it; it is not
  in this milestone.
- **One embedding model, one size.** `qwen3-embedding:0.6b` is 0.6B parameters and quantised
  to Q8_0. Nothing here shows a larger or unquantised model would also lose.

## Using fireassay as a library

[`agent-assay`](https://github.com/4ktLuffy/agent-assay), the sibling layer for tool-calling
agents, imports the spine rather than copying it: `hashing`, `integrity.assert_comparable`,
`admissibility.assess`, `controls.base.ControlOutcome`, `mutation.detector`, `items.core.wilson_ci`,
and the cost and latency scorers.

```bash
pip install "fireassay @ git+https://github.com/4ktLuffy/fireassay"            # the spine
pip install "fireassay[curate] @ git+https://github.com/4ktLuffy/fireassay"    # + Krippendorff, Textual TUI
```

The spine depends on `pydantic`, `numpy`, `pyyaml`, `typer`, `rich` and nothing else. The
package ships `py.typed`. `items.core` imports nothing from `fireassay.store`, by test.

## Tests and CI

```bash
make test       # 716 tests, 91% coverage
make lint       # ruff
make typecheck  # mypy --strict on src/
make mutants    # cosmic-ray over integrity/hashing/admissibility/score/controls (slow; weekly in CI)
```

`.github/workflows/ci.yml` runs lint, types, tests, `evidence.py --check` and the gate demo on
every push. `mutants.yml` runs the harness mutation score weekly; its target is 80%+ on the
five modules above, and a lower number is a finding to report, not a target to lower.

## Repo layout

```
src/fireassay/
├── cli.py                  # typer CLI (every command above)
├── hashing.py  integrity.py  admissibility.py  config.py  text.py  models.py
├── gate.py                 # M5: paired bootstrap, Holm, MDE refusal
├── runner.py  compare.py
├── store/                  # sqlite3, migrations 0001-0006, append-only triggers
├── system/                 # bm25, tfidf, coverage, dense (+ embedding, embedding_cache), corpus, panel
├── score/                  # the five deterministic scorers + invariants
├── controls/  mutation/    # negative controls; mutation operators and detector
├── llm/  generate/  filter/  curate/   # the LLM boundary and the curation funnel
├── items/                  # item analysis, review decks, seeding, calibration, PPI
└── report/text.py          # rich renderers
tools/        evidence.py (EVIDENCE.md's generator/verifier), gate_demo.py, mde.py,
              panel_dense.py, embed_corpus.py, dense_vs_bm25.py, measurement scripts
run/          committed artifacts every claim reads (response matrices, labels, judge outputs)
docs/         SPEC.md (the full v3 specification), MILESTONES.md (M1-M3b history)
tests/        fixtures/ (12 docs, 20 questions) and test_*.py
```

Milestone specs: `M1-SPEC.md`, `M2-SPEC.md`, `M3-SPEC.md`, `M3b-SPEC.md`, `M-ITEMS-SPEC.md`,
`M-DENSE-SPEC.md`.

## Two units that will silently corrupt numbers

`ScoringContext.price_in_per_mtok` / `price_out_per_mtok` are USD **per million tokens**. The
corpus JSONL contains a U+2028 character inside one string; read it by iterating the file
handle, never with `read_text().splitlines()`, which treats it as a line break and truncates
one record.

## Licence

Apache-2.0 (`LICENSE`). The corpus under `data/` is Crown copyright, reused under the Open
Government Licence v3.0.
