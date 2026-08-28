# fireassay — specification (v3)

**One line:** an evaluation integrity layer — the thing that tells you whether to believe your evals.

Fire assay is the reference method for determining precious-metal purity: the assay against
which other assays are validated. That is the positioning. fireassay does not compute quality
metrics, generate synthetic data, or optimise prompts — five well-funded projects do all three.
It makes a *result* admissible.

`last-verified: 2026-08-28` · supersedes benchgate-SPEC.md v1 (a full eval platform, discarded)
and v2 (correct thesis, but with six defects found since).

---

## 1. Thesis

> A green evaluation is only evidence if the harness can be shown to go red.

Same principle as `gradcheck`'s negative control, applied to evaluation instead of gradients.

Every defect found while writing this spec is one idea in different costumes: **a measurement
with no external referent**. Groundedness checks the answer against the retrieved context —
self-consistency. Inter-annotator agreement checks labellers against each other — self-consistency.
"Unjudged means irrelevant" assumes our own judgments are complete. Re-running CI until it
passes tests the same hypothesis until it agrees. Each needs an outside anchor: a reference
answer, a honeypot, a pooled judgment, a recorded attempt count.

## 2. Four pillars, each verified unbuilt

| Pillar | Verification (2026-08-28) |
|---|---|
| **1. Suite integrity** — content-addressed sets that refuse unsound comparison | 3 search phrasings → **0 repos** |
| **2. Controls + mutation score** — the harness proves it can fail, with a number | 3 phrasings → **0 repos** |
| **3. Curation with a measured error rate** | Kiln rates ~160 items for judge calibration, not a funnel. Argilla has annotation, no rubric framework / no agreement stats |
| **4. Base-vs-head gating with honest statistics** | promptfoo: "current checkout only". Langfuse: fixed threshold, no baseline fetch, no significance. Evidently: baselines a *dataset*, not a *run* |

**v1 of this spec claimed five gaps; three were wrong.** Kiln calibrates judges against humans
(Kendall's Tau/Spearman/Pearson/MAE); HELM measures efficiency alongside accuracy; promptfoo,
Kiln, deepeval (`optimizer/`) and ragas (`optimizers/`) all sweep configs. Recorded because the
error is instructive: ~90 repos were judged from description and stars, and the conclusion was
wrong in three of five places. Ten read properly overturned it. **Descriptions describe ambition;
source describes capability.**

## 3. Prior art (ships in the README)

| Project | Licence | Better than us at | We use it |
|---|---|---|---|
| ragas | Apache-2.0 | KG test generation, RAG metrics | scorer + generator adapter |
| deepeval | Apache-2.0 | metric breadth, prompt optimisation | scorer adapter |
| CJE | MIT | judge calibration, statistically serious | calibration backend |
| ranx / pytrec_eval | MIT | IR metrics **validated against trec_eval** | **test-only differential oracle** |
| crowd-kit | Apache-2.0 | Dawid-Skene, MACE, gold-majority-vote | annotation aggregation |
| confseq | MIT | Howard–Ramdas time-uniform confidence sequences | anytime-valid gate re-runs |
| promptfoo | MIT | config matrix, assertions, DX | config import |
| Kiln | Other (not OSI) | end-to-end workbench, non-technical UX | — closest competitor |
| HELM | Apache-2.0 | model benchmarking, efficiency metrics | — different layer |
| Langfuse / Opik / Evidently | Other / Apache-2.0 | tracing, experiment tracking | optional export |
| mutmut / cosmic-ray | BSD-3 / MIT | source-code mutation | CI job on our own tests |

fireassay ships Apache-2.0 and depends only on Apache-2.0/MIT. Nothing vendored.

## 4. Non-goals

Not a metric library (we define a `Scorer` protocol). Not a judge-calibration library (CJE).
Not a prompt optimiser (Kiln, deepeval). Not observability (Langfuse, Opik). No hosted service.

---

## 5. Pillar 1 — suite integrity

```
question_id = sha256("bgq1" ‖ text ‖ reference_answer ‖ sorted(evidence_span_keys))
suite_hash  = sha256("bgs1" ‖ sorted(question_ids))
config_hash = sha256("bgc1" ‖ canonical_json(spec))
corpus_hash = sha256("bgx1" ‖ sorted(doc_id:chunk_id:sha256(text)))
```

Prefixes are scheme-version tags so the hash scheme can change without colliding with old ids.

**The refusal rule.** `compare`, `diff` and `gate` refuse when: `suite_hash` differs; a
leaderboard would mix suite hashes; `scorer` name@version differs; any `env_json.affects_results`
field differs (**including `corpus_hash`**); a run is incomplete; a run has zero results.
`--force` proceeds but stamps `UNSOUND COMPARISON` on every output and records `forced: true`
permanently.

The most common lie in LLM evaluation is an improvement measured against a test set that quietly
changed. Discipline does not prevent it; structure does.

**Why `corpus_hash` is load-bearing:** of 1,181 gov.uk documents fetched, 258 were revised in 2026
and 182 in 2025. A re-fetch months later is a different corpus. Without the hash, nothing notices.

Schema: as v2 §5, plus `run.admissible`, `run.admissibility_json`, `control_check`, and
`gate_attempt` (§8). Runs are append-only, sealed in Python **and** by SQL trigger.

---

## 6. Pillar 2 — controls, in both directions

v2 had only negative controls. **Firing on a known bug proves sensitivity and nothing else**; a
detector that fires on everything and one that works are indistinguishable from the positive
control alone. Both directions are required.

### Negative controls — the harness must go red

| Control | Mechanism | Required |
|---|---|---|
| `no_retrieval` | empty context | groundedness → floor; recall = 0 |
| `shuffled_reference` | references permuted | correctness → chance band |
| `null_questions` | no answer in corpus | high `abstention.correct` |
| `identical_config` | same config twice | deltas inside declared noise band |
| `corpus_ablation` | 50% of gold docs removed | recall drops measurably |
| `judge_calibration` | judge vs human labels via CJE | calibrated, not extrapolating |
| `inference_noise_floor` | same config run N times against a stochastic system | spread measured, not assumed; becomes the detection floor |
| `corpus_injection` | canary documents carrying injection payloads planted in the corpus and deliberately retrieved | judge scores do not move |

Expected bands live in `controls/expected.yaml`, version-controlled, so a band cannot be quietly
widened to make a run pass.

**Silence is only evidence if something was attempted.** Any control whose expected result is
"nothing happens" must prove the mechanism ran — a *writable twin* that performs the same
operation where it should succeed, and demonstrably succeeds. A silent control with no successful
twin is vacuous and reports **red, not green**. Controls assert the *cause* (recall = 0 because
retrieval returned empty, not because scoring raised), never just the outcome.

**Taxonomy coverage, not case coverage.** For every refusal code and every metric: one case that
must fire, one that must stay silent. A single control against a five-class table is not coverage.

### Positive controls — the mutation score

Mutation testing is mature: inject faults, measure the fraction killed, report
`killed ÷ (total − equivalent)`. Thresholds 80–90% critical, 60–70% otherwise.

**fireassay reports two mutation scores, and the distinction is the point:**

| Score | Mutants | Answers |
|---|---|---|
| `harness_mutation_score` | source-code mutants of fireassay itself, via `cosmic-ray` | *do our tests catch bugs in fireassay?* |
| `gate_mutation_score` | *system-config* mutants — degraded retrieval, truncated context, weakened model, planted prompt regression | *does this eval setup catch injected regressions?* |

`gate_mutation_score` is the artifact that makes the whole project legible in one line:
**"this eval setup catches 87% of injected regressions."** It converts controls from binary to
continuous, supplies the missing positive direction, and no repo found does it for eval harnesses.

Equivalent mutants (a change that genuinely does not degrade quality) are excluded from the
denominator, as in classical mutation testing, and each exclusion is recorded with its reason.

**A run whose controls did not behave as required is inadmissible, and `run` exits non-zero
regardless of how well the real configs scored.**

---

## 7. Pillar 3 — curation with a measured error rate

```
generate 10–20k ─▶ machine filter ─▶ human curation ─▶ blend real traffic ─▶ frozen suite
```

The funnel is a reported artifact: counts and reject reasons at every step, plus reviewer hours.

### Rubric — six questions, then accept / edit / reject

`answerable_from_kb` · `self_contained` · `reference_answer_correct` · `evidence_sufficient` ·
`difficulty_agrees` · `qtype_agrees`

Reject reason codes: `DUPLICATE` `NOT_ANSWERABLE` `AMBIGUOUS` `WRONG_REFERENCE`
`INSUFFICIENT_EVIDENCE` `LEADING_QUESTION` `TRIVIAL` `OUT_OF_SCOPE` `PII_RISK`

### Agreement is not correctness

κ measures whether annotators agree, never whether they are right — two can agree and both be
wrong. Both instruments are required:

- **Agreement:** 10% double-reviewed. **Krippendorff's α**, not Cohen's κ — the rubric is ordinal
  (`yes/no/partially`) and will have missing observations; κ handles neither. α < 0.6 on
  `reference_answer_correct` is recorded in `suite.agreement_json` and surfaced in **every**
  report built on that suite, permanently.
- **Accuracy:** **honeypots** — 3–10% of the queue is items with known-correct verdicts, injected
  invisibly. This measures each curator against truth, not against each other.
- **Aggregation:** `crowd-kit` — Dawid-Skene, or `gold_majority_vote` where honeypots exist.
  Models each annotator's confusion matrix rather than taking a majority.
- **Drift:** curator accuracy tracked over session time; `curation.duration_ms` already recorded.

### Sealed predictions

Before any human-labelling batch, a structural prediction of the expected reject-reason
distribution is written and committed unopened. Otherwise the labels get read as confirming
whatever was expected. (On a prior project a sealed prediction was flatly wrong — 71.4% against
82.6% — and the refinement it was testing was dropped on that evidence. Unsealed, it would have
been read as confirmation.)

### Content validity

The funnel reports how many questions survived; it must also report **coverage**. What fraction of
corpus documents, and of `(qtype × difficulty)` cells, carry at least one question? A
1,000-question suite concentrated on 200 of 1,181 documents is a different instrument from one
spread evenly, and nothing currently distinguishes them. Vocabulary from construct-validity
literature (Cronbach & Meehl 1955; *Measuring what Matters*, NeurIPS 2025, which found
construct-validity weaknesses in nearly all of 445 reviewed ML/NLP papers).

### Interface

v1 terminal (Textual TUI, ~5s/item target). v2 Label Studio / Argilla adapters.
**Calibration labelling must not show the judge's verdict** — anchoring bias.

### Provenance

Real questions enter as `provenance='real_traffic'` through the same rubric. Every report splits
by provenance: a config that wins on synthetic questions and loses on real traffic is the exact
failure this exists to catch.

---

## 8. Pillar 4 — the gate

```yaml
baseline: release/v1.4        # a RUN, not a number
require_same_suite: true
require_admissible: true

primary:                      # uncorrected
  correctness:          {min_delta: -0.01}
guardrails:                   # Holm–Bonferroni across this family
  groundedness:         {min_delta: -0.01}
  retrieval.recall@5:   {min_delta: -0.02}
  abstention.correct:   {min_delta: -0.05}
  service_time.p95_ms:  {max_delta_pct: 15}
  cost.usd_per_1k:      {max_delta_pct: 20}
  policy.violations:    {max_absolute: 0}

significance:
  method: paired_bootstrap
  cluster_by: source_doc_id   # questions cluster by document
  n: 10000
  alpha: 0.05
  correction: holm
  anytime_valid: true         # confseq, for repeated runs on one commit
```

Four things here that no surveyed tool does:

**1. The baseline is a previous run, not a static threshold.** Langfuse raises `RegressionError`
against a fixed number; recovering a baseline is left to the user.

**2. Multiple-comparison correction.** Seven metrics at α=0.05 each is a false-positive rate of
**1 − 0.95⁷ ≈ 30% per PR**. One clean PR in three would be blocked by noise and the gate would be
switched off within a month. Holm–Bonferroni across the guardrail family, one primary metric
uncorrected.

**3. The gate refuses thresholds it cannot detect.** Paired-design power, α=0.05, power 0.8,
σ_d assumed 0.3 *(assumed — measure on the first real run; every number below moves with it)*:

| suite n | smallest detectable drop |
|---|---|
| 1,000 | ~2.7 pts |
| ~7,000 | ~1.0 pt |
| ~11,000 | ~1.0 pt with correction for 7 metrics |

v2 set a 1-point threshold on a planned 1,000-question suite — enforcing a threshold physically
undetectable at that size, which is a green whose cause was never established. `fireassay power`
computes required n from desired MDE, α and metric count; the gate checks its own thresholds
against the suite's MDE and **declines rather than reporting noise as a regression**.

*Corollary worth stating publicly: the Safaricom 5,000-question set was the right order of
magnitude for ~1% detection. Not "a big set" — the correct set.*

**4. The gate counts how many times you asked.** 5–7 looks at α=0.05 doubles real Type I error to
~10%; twenty reaches ~25–40%. A developer re-running CI until it passes is p-hacking with extra
steps, and with any temperature above zero it works. Every attempt is recorded in `gate_attempt`
against `(commit_sha, baseline_run_id)`; repeated attempts without an intervening code change
widen the interval via `confseq`'s always-valid p-values, and the PR comment says
**"this gate has been run 4 times on this commit."**

Bootstrap clusters by source document — a cluster bootstrap with too few clusters undercovers
(measured elsewhere at 9.0% rejection against a nominal 5% at 15 clusters). ~1,181 documents means
many clusters, so the effect is likely mild, but it is unverified and must be checked by
simulation on our own resampling scheme, not assumed.

---

## 9. Statistical efficiency (M6)

Motivated by a real constraint: judging 5,000 questions × 8 configs on a local 7B model is an
overnight job, and §8 says we need ~7,000 questions for 1-point detection. The answer is not a
bigger set — it is a more informative one.

| Technique | What it buys | Caveat |
|---|---|---|
| **CUPED** (Microsoft 2013; Netflix, Booking, BBC) — regression-adjust the delta using the baseline run's per-question score as covariate | **20–50% sample-size reduction.** The baseline score is an unusually good covariate: measured before the change, uncorrelated with it, highly correlated with the outcome | needs a baseline run; degrades to no-op if correlation is weak |
| **PPI** — prediction-powered inference (Science, 2023): combine few human labels with many judge labels, retaining valid CIs via a bias "rectifier" | exactly our judge situation; dramatically larger effective sample size | needs a labelled subset — we have one, via honeypots and curation |
| **IRT** — item difficulty + discrimination; compress a benchmark while preserving rankings (tinyBenchmarks: full-benchmark estimates within 2% MAE) | tells us **which questions actually discriminate between configs**. Many will not — every config gets them right, so they cost judge calls and carry no signal | **fits item parameters from many systems.** tinyBenchmarks used 319 models; we will have ~8 configs. At that scale only a 1-parameter (Rasch) model is defensible, used descriptively. **Report this limitation rather than overclaiming** |

Together these reframe the golden set: **it does not need to be big, it needs to be informative.**
Curate 5,000; IRT identifies the subset carrying the signal; CUPED and PPI detect smaller
regressions from fewer judge calls. None of the surveyed tools does any of the three.

---

## 10. Metrics

| Metric | LLM? | Source |
|---|---|---|
| `retrieval.recall@k` / `ndcg@k` / `mrr` / `bpref` | no | ours — relevance by **character-range overlap**, not chunk_id |
| `retrieval.judged_fraction` | no | ours |
| `service_time.{p50,p95,p99}_ms` | no | ours — **service time, not latency**; see §10.1 |
| `cost.usd_per_1k` | no | ours — prices **per million tokens** |
| `policy.violations` | no | ours — deterministic rule pack |
| `abstention.correct` / `abstention.wrongly_abstained` | no | ours |
| `correctness`, `groundedness` | yes | ragas / deepeval adapter |
| judge calibration | yes | CJE |

**Six of ten need no LLM.** A stranger with no API key reproduces most of the numbers.

Three metric defects fixed since v2, each of which would have produced a plausible wrong number:

1. **Relevance was matched by `chunk_id`** — but chunking is a config axis, so two chunk sizes give
   disjoint id spaces and any non-matching config scores recall 0.0, silently. Now character-range
   overlap within a document.
2. **A missing score read as a pass** — a system abstaining on everything scored best, because
   abstentions dropped out of the mean. Now `abstention.wrongly_abstained`, and every metric
   reports `n / applicable_n` so a shrinking denominator is visible. *Never aggregate quality
   without also scoring coverage, or extracting less wins.*
3. **Unjudged meant irrelevant** — the classic TREC pooling bias. Gold spans mark what we *know* is
   relevant, not all that *is*, which penalises configs retrieving correct-but-different passages.
   Now `judged_fraction` and `bpref` expose it; **pooling** (judge the union of top-k across all
   configs, adjudicate the unjudged) lands at M4, reusing the curation loop.

Also: `sorted()` on score alone leaves ties in load order — a constant scorer once produced
MRR@10 = 1.0 with nDCG@10 = 0. Sort by `(-score, doc_id, chunk_id)`.

And **one tokenizer**, in `text.py`, imported by BM25 and the dedup filter alike. A divergent
tokenizer between components produces disagreement for reasons unrelated to what is measured, and
both sides return plausible numbers. This has cost a prior project three fixes in three files in
one day.

## 10.1 Service time is not latency

Our runner is strictly sequential: send, wait, send. That is a **closed-loop** measurement,
which coordinates with the system being measured and never observes the queue — coordinated
omission, which underestimates p99 by **up to 25×** in published measurements.

The metric is therefore named `service_time`, not `latency`, and documented as per-request
service time under no concurrency. Both numbers are useful; only one is the name we gave it.
Open-loop measurement under a fixed arrival schedule is out of scope and stated as such.

## 10.2 Two noise sources, both measured

Temperature 0 does not make an LLM deterministic. The dominant cause is not floating-point
non-associativity but **batch-size dependence of reduction kernels**: the same prompt under a
different dynamic batch size takes a different reduction tree. Thinking Machines Lab observed
**80 distinct completions from 1,000 identical requests** on Qwen3-235B; accuracy swings up to
9% have been measured from GPU-count and batch-size changes alone. Batch-invariant kernels now
ship in vLLM and SGLang at ~61.5% throughput cost.

Consequences:

1. **`identical_config` asserts exact zero only for deterministic systems** (M1's BM25). For any
   model-backed system the expected band comes from `inference_noise_floor` — measured, never
   assumed.
2. **There are two independent noise sources**: sampling noise across questions (§8, drives the
   MDE table) and inference nondeterminism on the same question. The detection floor is the
   larger of the two. v3 accounted for only the first.
3. `env_fingerprint` records inference-server settings including batch size, because they change
   results and therefore belong in `affects_results`.

## 10.3 Judge integrity (M4)

Three documented judge biases, each with a required countermeasure:

| Bias | Magnitude | Countermeasure |
|---|---|---|
| **position** — prefers the response in a given slot | 10–15 percentage points | **swap augmentation**: judge A-vs-B and B-vs-A, accept consistent verdicts, record inconsistent as ties. Plus a `judge_position_bias` control reporting flip rate |
| **verbosity** — prefers longer answers even when the extra content is irrelevant | — | report the **correlation between score and answer length**; a strong correlation is evidence, and it is deterministic and free |
| **self-preference** — favours its own outputs | — | **the judge model must differ from the question-generator model.** Both are recorded in `env_fingerprint` and a run where they match is **refused** |

Self-preference settles an open question from v3 on evidence rather than analogy: if one model
writes the questions and grades the answers, that bias contaminates every correctness score.

## 10.4 A corpus can attack the judge

Indirect prompt injection research is blunt: retrieval is the universal failure point — once
malicious text enters the top-k it is fed to the model through the system-controlled path and
trusted implicitly, and nearly any model can be hijacked.

fireassay retrieves documents and feeds them to a judge, so **a poisoned corpus document can move
a score.** For a tool whose purpose is trustworthy measurement this is an attack surface, and no
surveyed eval framework treats it as one.

Hence the `corpus_injection` control: plant canary documents carrying injection payloads, retrieve
them deliberately, verify the judge's scores do not move. Mutation testing for the judge's
injection resistance. gov.uk is a safe corpus; the tool is meant for anyone's.

## 11. Differential oracle on our own metrics

`ranx` states its metrics are tested against `trec_eval`; `pytrec_eval` binds the reference
implementation directly. We keep our own implementations — relevance resolution is ours and no
library does character-overlap — but resolve relevance our way, emit standard qrels + run, and
**assert our nDCG/MRR/recall/bpref equal the reference to 1e-9 across randomised cases generated
with Hypothesis.**

Both are **test-only** dependencies (ranx pulls Numba/LLVM; unacceptable at runtime for a repo
that should clone-and-run). This is a differential oracle applied to our own code, and it buys a
claim few eval repos can make.

## 12. Evidence (`EVIDENCE.md`)

| # | Artifact | Proves |
|---|---|---|
| 1 | Leaderboard, ~8 configs, all metrics, split by provenance/qtype/difficulty | it runs at scale |
| 2 | Refusal transcript — comparison declined across a changed suite and a changed corpus | pillar 1 |
| 3 | Six control results **+ both mutation scores** + a deliberately broken harness the controls catch | pillar 2 |
| 4 | Funnel report: reject reasons, reviewer hours, Krippendorff's α, honeypot accuracy, content coverage | pillar 3 |
| 5 | A blocked PR in this repo's own history — a planted regression the gate caught | pillar 4 |
| 6 | Differential test: our metrics == `trec_eval` to 1e-9 | §11 |
| 7 | `make reproduce` regenerating 1, 2, 3, 6 with no API key | reproducibility |

Artifact 3 is the one that matters most: a harness that catches its own sabotage, with a number.

## 13. Build order

| # | Milestone | Contents |
|---|---|---|
| **M1** | deterministic spine | store, hashing, refusal, BM25, deterministic scorers, differential oracle. **No LLM.** |
| **M2** | controls + admissibility + **both mutation scores** | the centre of the product |
| **M3** | curation: TUI, rubric, honeypots, crowd-kit, Krippendorff, funnel, coverage | most human time — start early |
| **M4** | judges via ragas/deepeval + CJE calibration + **pooling** | judges enter only after they can be proven |
| **M5** | gate: base-vs-head, Holm–Bonferroni, cluster bootstrap, power check, confseq re-runs, GH Action | |
| **M6** | statistical efficiency: CUPED, PPI, IRT | optional, and the most impressive |

M1 contains no model call at all.

## 14. Open questions

- σ_d for correctness is **assumed 0.3**. Measure it on the first real run.
- Cluster-bootstrap coverage must be validated by simulation on our own resampling scheme.
- Must the judge model differ from the question-generator model? Probably yes — same
  self-consistency argument as everything else. Not yet decided.
- IRT with ~8 configs is thin. Rasch-only, descriptive, limitation stated.
- Corpus: gov.uk, OGL v3, 1,181 docs / 10.8M chars / ~21k chunks @512 (fetched 2026-08-28).
  README must carry: *"Contains public sector information licensed under the Open Government
  Licence v3.0."*
