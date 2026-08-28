# fireassay M2 — controls, admissibility, mutation scores

Repo root: `/Users/Twinkle/AI Engineering/fireassay`
Read `M1-SPEC.md` for the existing shape and `../fireassay-SPEC.md` §6 for intent.
This document governs M2 and overrides both on detail.

M1 is complete and green: 134 tests, 95% coverage, ruff + mypy --strict clean. Do not
regress it.

## M2 goal

The harness proves it can fail — in both directions, with a number.

M1 answers "what did this config score." M2 answers **"should you believe that score."**

### In scope
control framework · 5 deterministic controls · writable twins · expected bands ·
admissibility engine · system-mutation framework · `gate_mutation_score` ·
`harness_mutation_score` config · CLI · tests

### Out of scope (do not build)
LLM judges · CJE · curation · pooling · the statistical gate (M5) · HTML report

## Constraints

- **No LLM call and no network anywhere in M2**, same as M1.
- Runtime dependencies unchanged: pydantic, typer, pyyaml, numpy, rich. `cosmic-ray` is a
  **dev/test** dependency only.
- `mypy --strict` and `ruff` clean. Full annotations.
- No git, no shell (you have neither); tests will be run for you.

---

## 1. The rule that governs everything in M2

> **A check that cannot run must never read as a check that passed.**

Every control result is one of `PASSED` / `FAILED` / **`NOT_RUN`**. `NOT_RUN` is not a pass:
a run containing any `NOT_RUN` control is **inadmissible**, exactly like a failure. There is
no third state that quietly means "fine."

Corollary, and the reason `twin_ok` exists below:

> **Any check whose expected result is "nothing happens" needs independent proof that
> something was genuinely attempted.**

## 2. `src/fireassay/controls/base.py`

```python
class ControlOutcome(BaseModel):
    kind: str
    status: Literal["PASSED", "FAILED", "NOT_RUN"]
    observed: dict[str, float]
    expected: dict[str, object]      # the band from expected.yaml
    twin_ok: bool                    # did the writable twin demonstrably succeed?
    cause_assertions: dict[str, bool]  # specific facts, not just the outcome
    detail: str

class Control(Protocol):
    kind: str
    version: str
    def run(self, ctx: ControlContext) -> ControlOutcome: ...
```

`ControlContext` carries the store, the suite, the base config, the corpus documents, the
system factory, the scorers and the loaded expected bands.

### Writable twin

Every control that expects a metric to *collapse* must first demonstrate the same
measurement *working* on an unmodified run. `twin_ok = False` forces `status = FAILED`,
never `PASSED` — a silent control with no successful twin is vacuous.

Concretely for `no_retrieval`: the twin is the unmodified config, which must score
`retrieval.recall@k > 0` on at least one question. If it does not, the corpus/suite pairing
is broken and the control's silence tells us nothing about the harness.

### Cause assertions

Assert *why*, not just *what*. `no_retrieval` asserts `recall == 0.0` **and**
`len(retrieved) == 0` **and** `judged_fraction == 0.0`. "It went to zero" and "it went to
zero for the reason I think" are different claims, and only the second is evidence.

## 3. The five deterministic controls

`src/fireassay/controls/` — one module each. All are LLM-free.

| kind | Mechanism | Must observe | Twin |
|---|---|---|---|
| `no_retrieval` | system wrapper returns empty `retrieved` | `recall@k == 0`, `ndcg@k == 0`, `mrr == 0`, `judged_fraction == 0`, `len(retrieved) == 0` | unmodified config scores `recall@k > 0` |
| `shuffled_gold` | permute `evidence_spans` across questions (deterministic seed from `suite_hash`) | `recall@k` falls into the chance band | unmodified config beats the chance band |
| `corpus_ablation` | drop 50% of documents that carry gold spans (seeded) | `recall@k` drops by at least `min_drop` | unmodified config scores higher |
| `identical_config` | run the same config twice | every metric delta `== 0.0` exactly (M1 is deterministic; a nonzero delta is a bug, not noise) | both runs complete with `n > 0` |
| `null_questions` | questions whose `qtype == "unanswerable"` | `recall@k == 0` — nothing relevant exists to find | the answerable subset scores `> 0` |

`judge_calibration` is **M4**. Register it now returning `status = "NOT_RUN"` with
`detail = "requires a judge; arrives in M4"`. Per §1 this makes runs inadmissible until M4
lands — that is correct and intended. Add `--allow-not-run` to the CLI to proceed anyway,
which stamps `admissibility_json.allowed_not_run = [...]` on the run and is surfaced in
every report built on it.

**`shuffled_gold` replaces v2's `shuffled_reference`.** Permuting reference *answers* only
moves a judge-based correctness metric, which does not exist yet. Permuting gold *spans*
collapses retrieval, which is measurable today. Same idea, deterministic form.

## 4. `controls/expected.yaml`

Version-controlled so a band cannot be quietly widened to make a run pass.

```yaml
no_retrieval:
  recall_at_k:      {eq: 0.0}
  retrieved_len:    {eq: 0}
shuffled_gold:
  recall_at_k:      {max: 0.05}      # chance band
corpus_ablation:
  recall_at_k_drop: {min: 0.15}
identical_config:
  max_abs_delta:    {eq: 0.0}
null_questions:
  recall_at_k:      {eq: 0.0}
```

The loader must reject an unknown control kind and reject a band with no constraint keys —
an empty band is a band that always passes.

## 5. Admissibility engine — `src/fireassay/admissibility.py`

```python
def assess(run: Run, invariant_violations: Sequence[InvariantViolation],
           controls: Sequence[ControlOutcome], *, allow_not_run: Sequence[str] = ()) -> Admissibility
```

A run is admissible **iff** no invariant violations **and** every control is `PASSED`
(or `NOT_RUN` and explicitly allowed). `assess` writes `run.admissible` and
`run.admissibility_json`.

`fireassay run` exits **non-zero when any control fails, regardless of how well the real
configs scored.** A harness that cannot detect a sabotaged config is not measuring
anything, and no result it produces is admissible.

## 6. System mutation — `src/fireassay/mutation/`

Classical mutation testing mutates *source code*. Ours mutates the **system under test**, to
ask whether the evaluation setup would notice a real regression.

### Operators (`mutation/operators.py`)

Each wraps a `System` and degrades it deterministically (seed from `suite_hash` + params):

| operator | params | effect |
|---|---|---|
| `drop_results` | `n: int` | delete `n` chunks from the top of the ranking |
| `truncate_topk` | `k: int` | return only the first `k` results |
| `shuffle_topk` | — | randomise the order of the retrieved list |
| `corrupt_query` | `pct: float` | drop `pct` of query tokens before retrieval |
| `swap_ranking` | — | reverse the retrieved list |

`MutationOperator` is a Protocol so M5 can add generation-side operators (weakened model,
truncated context, planted prompt regression) without touching this module.

### Detector (`mutation/detector.py`)

```python
class MutationDetector(Protocol):
    name: str
    def detects(self, baseline: Run, mutant: Run, store: Store) -> tuple[bool, str]: ...
```

M2 ships `ThresholdDetector` — a metric moved beyond a configured threshold. **M5 will add
`GateDetector` using the real statistical gate; the score must then be recomputed and both
reported.** Do not attempt statistics here; a naive threshold is honest and clearly labelled.

### Score (`mutation/score.py`)

```
gate_mutation_score = killed / (total - equivalent)
```

A mutant is **killed** when the detector reports it. A mutant is **equivalent** when it
provably does not degrade quality — e.g. `truncate_topk(k)` where `k >= len(retrieved)` for
every question, or `drop_results(0)`. Equivalence must be **computed and justified**, never
assumed: store `equivalent_reason` on every excluded mutant and print the exclusions in the
report. An unexplained exclusion is how a mutation score gets inflated.

A **surviving** mutant is the interesting output — it names a regression this eval setup
would not have caught. The report lists survivors first.

## 7. Schema additions — `store/migrations/0002_controls_mutation.sql`

```sql
control_check(id, run_id, kind, status, observed_json, expected_json,
              twin_ok, cause_json, detail, checked_at)

mutation_run(id, suite_id, suite_hash, base_config_id, detector,
             killed, total, equivalent, score, created_at)

mutant(id, mutation_run_id, operator, params_json, mutant_run_id,
       killed, equivalent, equivalent_reason, detail)
```

Same append-only sealing as M1, enforced in Python and by SQL trigger.

## 8. `harness_mutation_score` — mutating our own source

Not our code: configuration plus a CI job.

- `cosmic-ray.toml` targeting `src/fireassay/{integrity,hashing,admissibility}.py`,
  `src/fireassay/score/`, `src/fireassay/controls/`.
- `Makefile`: `make mutants` runs it and writes `artifacts/harness-mutation.json`.
- `.github/workflows/mutants.yml` runs it on a schedule (not per-PR; it is slow).
- README documents the target: **80%+ on those modules**, anything lower is a finding.

A repo whose thesis is that a harness must be shown to go red should publish whether its own
tests do.

## 9. CLI additions

```
fireassay controls run     --matrix configs/matrix.yaml --corpus corpus.jsonl [--allow-not-run judge_calibration]
fireassay controls show    --run <run_id>
fireassay mutate           --suite name@ver --config <config_id> --operators configs/mutants.yaml
fireassay mutation score   --mutation-run <id>
```

`controls run` exits 2 on any `FAILED`, 3 on any disallowed `NOT_RUN`. `mutate` exits 0
regardless of score — a low score is a finding to report, not a command failure.

## 10. Tests

| File | Must prove |
|---|---|
| `test_controls_no_retrieval.py` | fires when retrieval is empty; **`twin_ok=False` forces FAILED even when the observation matches the band** |
| `test_controls_shuffled_gold.py` | recall collapses to the chance band; deterministic across two runs with the same `suite_hash` |
| `test_controls_corpus_ablation.py` | recall drops by at least `min_drop`; twin scores higher |
| `test_controls_identical_config.py` | zero deltas on M1's deterministic system; a deliberately nondeterministic stub FAILS it |
| `test_controls_null_questions.py` | recall == 0 on unanswerable; answerable twin > 0 |
| `test_expected_bands.py` | unknown kind rejected; **empty band rejected**; a widened band is visible in a diff |
| `test_admissibility.py` | any FAILED → inadmissible; any NOT_RUN → inadmissible unless allowed; allowed NOT_RUN is recorded on the run |
| `test_mutation_operators.py` | each operator degrades a known ranking in the expected way; deterministic under a fixed seed |
| `test_mutation_score.py` | score arithmetic; **equivalent mutants excluded from the denominator AND carry a reason**; survivors listed first |
| `test_mutation_e2e.py` | full path: baseline run → 5 mutants → detector → score, on the M1 fixtures |
| `test_controls_cli.py` | exit 2 on FAILED, exit 3 on disallowed NOT_RUN |

**Every control needs a test proving it FAILS on a broken harness, not only that it passes
on a working one.** A control that has never fired is untested. Taxonomy coverage: for each
of the five, one case that must fire and one that must stay silent.

## 11. If anything is ambiguous

Stop and ask. Especially: the `NOT_RUN` ≠ passed rule (§1), the writable-twin requirement
(§2), and equivalent-mutant justification (§6). Those three are load-bearing and a
plausible guess is worse than a question.
