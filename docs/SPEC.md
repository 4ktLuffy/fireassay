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

### Staleness — free from content addressing

*A stale evaluation set does not fail loudly. It fails politely* — it keeps producing numbers
that no longer describe reality. Our corpus is already content-hashed, so staleness is
computable rather than a matter of vigilance:

- `evidence_span` records the **hash of its source text at freeze time**.
- `fireassay suite staleness` re-hashes the current corpus and reports every question whose
  supporting evidence has moved. Its reference answer may no longer be correct.
- Every question carries `created_at`; the funnel report includes the **age distribution** of
  the suite alongside coverage.
- Retirement is deliberate and recorded, never silent: a retired question is marked with a
  reason, and retiring one produces a new `suite` version with a new `suite_hash`.

This is not hypothetical — **258 of our 1,181 gov.uk documents were revised in 2026**. A suite
frozen today will have questions whose ground truth has silently moved within months.

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

## 10.5 Contamination: the closed-book floor is not zero

gov.uk is public and old, so it is in every model's training data. A model asked "how do I
renew my driving licence" answers from parametric memory with no retrieval at all.

**This breaks `no_retrieval` the moment a judge exists.** M2's version is safe — it asserts
`recall/ndcg/mrr == 0`, which is mechanical and contamination-proof. But from M4, correctness
and groundedness under `no_retrieval` will NOT collapse to zero, and a band expecting zero
would fail on a working harness.

The floor must therefore be **measured, not assumed**: run the system with retrieval disabled
and record what it gets right from memory. That closed-book run becomes:

1. the expected band for `no_retrieval` at M4, and
2. a first-class reported metric —

    **RAG Gain = accuracy(with retrieval) − accuracy(closed book)**

Most eval tools report absolute accuracy and never ask whether retrieval contributed anything.
A config can score 85% with retrieval worth three points of it. RAG Gain is the number that
decides whether the retrieval stack earns its cost, and it is the generation-side twin of
`handbook/embeddings/04-evaluation.md`'s rule: *a baseline the model can lose to.*

Contamination-detection literature (n-gram overlap, perturbation analysis, backdoor "dye pack"
canaries) is out of scope — we do not need to prove contamination, only to stop assuming its
absence.

## 10.6 Counterfactual mutants isolate the decoder (M5)

Fix the retrieved passage, apply a controlled semantic edit to it, and re-run. Any accuracy drop
is a **decoder faithfulness** failure, not a retrieval failure — the two are otherwise conflated
in every end-to-end number. This is a generation-side mutation operator for M5 and it gives the
mutation framework a second, independent axis.

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

## 10.7 The instrument has three limits, and fireassay publishes all three

A gate can only detect a change larger than its own measurement error. There are three
independent sources, and v3 accounted for only the first:

| Limit | Source | Measured by |
|---|---|---|
| **MDE** | sampling noise across questions | suite size + σ_d (§8) |
| **Noise floor** | inference nondeterminism on the same question | `inference_noise_floor` control (§10.2) |
| **Ceiling** | error in the reference answers themselves | honeypot accuracy + Krippendorff's α (§7) |

The third is new and it is a hard cap. Real-world labelling rarely exceeds ~97% consistency,
and audits of "gold standard" ML benchmarks find label error rates of **3–6%**. The literature
calls this *label convergence*: the highest score achievable given contradictory annotations.

Concretely: if honeypots show curators are 94% accurate, a config scoring 95% correctness is
**indistinguishable** from one scoring 100% — the instrument cannot resolve the difference.
That number belongs on the leaderboard beside the scores, not in a footnote.

The gate refuses a threshold that violates **any** of the three. This is the integrity thesis
turned on the instrument itself: fireassay knows what it cannot measure, and says so.

## 10.8 Abstention deserves a curve, not a bit

`abstention.correct` is binary and therefore compares two configs at one arbitrary operating
point on two different curves. A system that can abstain has a *tunable* threshold, and the
honest comparison is the whole trade-off:

- sweep the confidence threshold τ, recording **coverage** (fraction answered) against
  **selective risk** (error rate among answered)
- report **AURC** — area under the risk-coverage curve, lower is better — and **E-AURC**, the
  excess over an oracle ranking, which separates "the model is wrong" from "the model does not
  know when it is wrong"

Full coverage is maximally useful and maximally unsafe; abstaining always is safe and useless.
The curve is that trade-off, and it is how an abstention threshold should actually be chosen —
including for a system that must decline rather than guess.

## 10.9 A failing gate must say *which* slice

*Eval scores tell you that quality regressed, but not why.* A gate that reports "correctness
dropped 3 points" blocks a PR without helping anyone fix it.

`fireassay diff --explain` embeds the questions that newly fail, clusters them (HDBSCAN over
local embeddings — `qwen3-embedding:0.6b` is already available offline), and names the clusters
in the PR comment alongside the fixed slices we already have (`qtype`, `difficulty`,
`provenance`). The target output is not *"correctness −3.1"* but *"correctness −3.1,
concentrated in multi-hop questions about tax deadlines."*

Fixed slices catch what we thought to ask about; clustering catches what we did not.

## 10.10 Metamorphic controls — a third category, needing no golden set

Metamorphic testing (Chen, Cheung & Yiu, 1998) replaces the ground-truth oracle with a
*relation* oracle. Instead of "what is the correct output?", ask "**how must the output change,
or not change, when the input is systematically altered?**" MetaRAG (2025) applies this to RAG
hallucination detection in an unsupervised, black-box setting needing neither references nor
model internals.

That gives fireassay a third control category alongside negative (sabotage must be caught) and
positive (mutants must be killed): **relations that must hold on any corpus, with no curated
questions at all.**

| Relation | Must hold | Needs a model? |
|---|---|---|
| `corpus_reorder_invariance` | shuffling corpus load order does not change results | no |
| `duplicate_document_idempotence` | duplicating a document does not materially change retrieval | no |
| `irrelevant_insertion` | adding an unrelated document does not change answers to unrelated questions | no |
| `query_paraphrase_invariance` | paraphrasing a question retrieves substantially the same documents; large divergence means brittle retrieval | yes (M4) |
| `factoid_antonym_contradiction` | decompose the answer into atomic factoids, substitute antonyms — the context must contradict them (MetaRAG) | yes (M4) |

The first three are deterministic and land with M5's mutation work. `corpus_reorder_invariance`
already exists in M1 as the BM25 tie-breaking test — it was a metamorphic relation before we had
the word for it.

This matters because metamorphic controls work where the golden set does not: a new corpus, a
domain nobody has curated, day one of adoption.

## 10.11 A golden set is a test suite — import the flaky-test playbook

Software engineering has twenty years of practice on nondeterministic tests. The LLM evaluation
world has not imported it. The vocabulary transfers exactly:

| SE practice | fireassay |
|---|---|
| flake rate: % of runs with inconsistent results, per test | **per-question flake rate**, measured by `inference_noise_floor` (§10.2) — run N times on identical inputs, count flips |
| "detection is a statistics problem, not a debugging problem" | exactly why the noise floor is measured rather than assumed |
| quarantine lane: flaky tests still run, reported, but do not block | **chronically flaky questions are reported but excluded from gating** |
| a new test with a 10% flake rate goes back to the author before entering the gating suite | **admission policy**: a question above the flake threshold does not enter the frozen suite |
| keep quarantine under 5% of the suite; over 10% signals systemic design failure | a suite with >10% flaky questions is flagged — the problem is the suite, not the questions |
| **"retry-until-green is an antipattern"** | independent confirmation of §8's attempt counting. One rerun for data collection is fine; retrying until it passes is p-hacking |

This also makes the project legible to any software engineer in one sentence: *your golden set is
a test suite, so here is its flake rate and its quarantine list.*

## 10.12 APFD closes the subsetting question

Test-case prioritisation has a standard metric — **APFD**, Average Percentage of Faults Detected —
measuring how *quickly* an ordering finds faults, not merely whether it does.

We already generate faults: the mutants of §6. So **APFD over the mutant set** answers the
question M6 raises but cannot otherwise settle — *can I run 800 questions instead of 5,000?*

- `gate_mutation_score` asks **whether** a suite catches injected regressions.
- **APFD** asks **how quickly**, which is what justifies a reduced suite.

An IRT-selected subset that preserves APFD is a defensible reduction. One that does not is a
cheaper suite pretending to be the same instrument.

## 10.13 Divisions — MLPerf already solved refusal-vs-usability

§5 refuses to compare runs across a changed suite. That is correct and insufficient: real users
grow their suite, and a flat refusal makes the tool unusable the first time they do.

MLPerf's submission rules solve this with **divisions**, and the structure transfers exactly:

| Division | Rule | Comparable to |
|---|---|---|
| **closed** | frozen suite, fixed scorer versions, matching `corpus_hash` and `affects_results` | other closed runs on the same `suite_hash` |
| **open** | anything — grown suite, swapped corpus, custom scorers — but **every deviation is disclosed** | other open runs only; never to closed results |

An open-division result is *valid and publishable*, just not comparable to a closed one. That
converts a refusal into a labelled lane, which is what makes the integrity rule survive contact
with users. `--force` (§5) remains for the genuinely unsound case; divisions cover the legitimate
one.

MLPerf also calls its mandatory pre-submission checks **compliance tests** that every submitter
must run for a result to count. That is precisely §6's controls, under an older name and in a
field with fifteen years of adversarial use — useful validation that the mechanism is sound, and
better vocabulary for explaining it.

## 10.14 `env_fingerprint` — adopt MLPerf's disclosure shape

§5 hand-waves `affects_results`. MLPerf's submission metadata gives it a concrete shape:
hardware, full software stack with framework versions, model metadata (starting weights,
transformations, data types), and source sufficient for replication with **commit hashes**.

fireassay's `affects_results` therefore records: package versions (ours and every scorer's),
model identity **by digest, not tag**, inference-server settings **including batch size**
(§10.2), corpus hash, suite hash, scorer name@version, and the git commit of the run.
Everything else — wall-clock, hostname, run id — is metadata and does not participate in
`ENV_MISMATCH`.

The distinction matters: a fingerprint that includes too much refuses everything, and one that
includes too little compares the incomparable.

## 10.15 An Annex IV evidence bundle (M6, and the commercial angle)

The EU AI Act (Article 11, Annex IV) requires providers of high-risk systems to document
metrics and thresholds for **accuracy and robustness**, quantify accuracy **for specific groups**,
and justify **the appropriateness of the performance metrics** for the system's purpose.
Conformity assessment evaluates *a comprehensive evidence chain rather than isolated documents*.

fireassay already produces every component:

| Annex IV requirement | fireassay artifact |
|---|---|
| metrics used, and why they are appropriate | metric definitions + §10.7's three limits — **we can justify appropriateness because we measure what the instrument cannot resolve** |
| accuracy thresholds and their justification | gate thresholds checked against MDE, noise floor and ceiling |
| accuracy for specific groups | provenance / qtype / difficulty slices (§7) and `diff --explain` clusters (§10.9) |
| robustness testing | controls, mutation scores, metamorphic relations |
| evidence chain, versioned | content-addressed suite/corpus/config + append-only runs |

`fireassay report --annex-iv` assembles them. Almost nobody can currently answer "why is this
metric appropriate?" with evidence rather than assertion. We can, and that is a direct
consequence of §10.7 rather than a bolt-on.

## 10.16 Single-turn only — stated, not papered over

Our suite is single-turn. Real customer support is conversational, and the literature is explicit
that **single-turn estimation overstates dialogue ability**; multi-turn adds evolving intent,
conversational noise, non-standalone questions and context limits (MTRAG, CORAL).

This is a scope limitation, not a gap to quietly fill. The README says so plainly. Extending the
schema to multi-turn is a coherent future milestone, but claiming single-turn results describe
conversational quality would be exactly the overclaim this project exists to prevent.

## 10.17 The strongest critique of this project, and what it changes

Practitioner evidence, taken at face value rather than argued with:

- **"Start with error analysis, not infrastructure."** (Husain/Shankar) Error analysis — sampling
  real traces, writing notes on what went wrong, clustering into a failure taxonomy, counting
  frequencies — is described as *the most important activity in evals*. fireassay is
  infrastructure. That is the sharpest available criticism of it.
- **The results-actionability gap: 17 of 19 studied practitioners** could gather evaluation data
  but not turn it into concrete improvements. Teams do not struggle to *measure*; they struggle
  to *act*.
- **Engineers abandon evals within weeks**, usually because they cannot show ROI, and because
  they run evals that are not tied to business goals — *"metrics sound convincing"* while
  measuring the wrong thing.
- **Quality gates get bypassed entirely** when they block releases for things that do not affect
  users. Industry practice has moved to *risk-based* gating. Gates on the whole codebase
  *"fail forever on legacy projects and get disabled"*.
- **93% of published evaluation happens pre-deployment**; continuous post-deployment evaluation
  is barely explored.

### What survives

The thesis is orthogonal to the critique. Error analysis tells you *what to measure*; fireassay
tells you *whether to believe the measurement*. And the practitioner advice to
*"compare your LLM judge's outputs against your hand-labelled data"* **is** §7's calibration
loop — the same conclusion reached from a different direction.

### What changes

1. **Error analysis becomes a first-class workflow, not a downstream feature.** The curation TUI
   (§7) already does 80% of it: review an item, write a note, assign a code, count frequencies.
   Adding a trace-review mode that clusters free-text notes into a failure taxonomy costs little
   and addresses the single most-cited practitioner need. It also feeds §10.9's `diff --explain`.
2. **The gate defaults to ONE gated metric, not seven.** Everything else is report-only until a
   user promotes it. Holm–Bonferroni (§8) fixes the statistics of gating seven metrics; it does
   not fix the *social* problem that a gate firing often gets switched off. Risk-based by
   default: block on what matters, report the rest.
3. **Gate the delta, never the standing state.** A gate that can fail forever will be disabled.
   Thresholds are on change relative to the baseline run — which §8 already does — and the
   README must say why absolute-threshold gating is the wrong shape.
4. **`diff --explain` (§10.9) is core, not optional.** The results-actionability gap is the
   number one reported failure. A gate that says *which* slice regressed is the difference
   between infrastructure and a tool someone keeps.
5. **The README leads with what a practitioner recognises** — judge calibration, error analysis,
   a gate that explains itself — and presents integrity as what makes those trustworthy. Leading
   with the integrity machinery reads as ceremony to exactly the audience most able to judge it.

None of this weakens the thesis. It changes the order in which it is presented, and it adds the
one workflow the evidence says users actually need.

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
