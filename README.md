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
| **The gate blocking a planted regression** | exit 4 on `drop_results`; exit 0 on a benign change; **exit 5 (refused)** on a real 0.39 drop asked at threshold 0.3, because 14 items resolve only 0.35 | `gate.*` |

Two of those rows are why this project exists rather than a metric library. The calibration
set's first act was to kill one of the tool's own detectors: `mislabel_suspect` looked like a
weak-but-real detector at 0.219 precision until it was put beside the 0.25 a random draw
scores. And the gate's third case is a regression that is real, larger than the threshold
asked for, and still refused, because a threshold the suite cannot distinguish from noise is
not one a gate can honestly enforce.

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
| System under test | `system/` | Pure-Python BM25 (and TF-IDF, a panel of 54 configs) — no API key, no network, so a stranger can reproduce every retrieval number here |

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
  not hidden.

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
make test       # 644 tests, 91% coverage
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
├── system/                 # bm25, tfidf, corpus, panel, coverage
├── score/                  # the five deterministic scorers + invariants
├── controls/  mutation/    # negative controls; mutation operators and detector
├── llm/  generate/  filter/  curate/   # the LLM boundary and the curation funnel
├── items/                  # item analysis, review decks, seeding, calibration, PPI
└── report/text.py          # rich renderers
tools/        evidence.py (EVIDENCE.md's generator/verifier), gate_demo.py, mde.py, measurement scripts
run/          committed artifacts every claim reads (response matrices, labels, judge outputs)
docs/         SPEC.md (the full v3 specification), MILESTONES.md (M1-M3b history)
tests/        fixtures/ (12 docs, 20 questions) and test_*.py
```

Milestone specs: `M1-SPEC.md`, `M2-SPEC.md`, `M3-SPEC.md`, `M3b-SPEC.md`, `M-ITEMS-SPEC.md`.

## Two units that will silently corrupt numbers

`ScoringContext.price_in_per_mtok` / `price_out_per_mtok` are USD **per million tokens**. The
corpus JSONL contains a U+2028 character inside one string; read it by iterating the file
handle, never with `read_text().splitlines()`, which treats it as a line break and truncates
one record.

## Licence

Apache-2.0 (`LICENSE`). The corpus under `data/` is Crown copyright, reused under the Open
Government Licence v3.0.
