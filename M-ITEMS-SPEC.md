# `fireassay items` — item analysis for evaluation sets

Repo root: `/Users/Twinkle/AI Engineering/fireassay`
**Read `~/ObitosBrain/handbook/eval-of-evals.md` first — it records what was measured, what was
falsified, and why. Do not re-derive it.**

M1–M3b are green: 372 tests, 90% coverage, ruff + mypy --strict clean. Do not regress them.

## What this is

A tool that answers *"is this eval question worth keeping?"* — usable **inside** fireassay and
**standalone by someone who has never heard of fireassay**. That dual use is a hard requirement,
not packaging.

## ⚠ The constraint that shapes everything

**`items/core.py` MUST NOT import `fireassay.store`, or anything that does.**

The core takes a response matrix and returns statistics. The moment it reaches for storage the
tool is welded into fireassay and cannot be extracted. If something in the core seems to need
the store, the interface is wrong — stop and ask.

## What was already measured — encode these, do not re-litigate

| Finding | Consequence for this tool |
|---|---|
| Cronbach's alpha reports 0.948 for a set padded with 300 fake all-correct items | **Do not implement it.** Blind to padding; a category error here |
| Per-item discrimination split-half r = 0.294 at 18 systems (183/200 splits < 0.5) | **Always report split-half reliability**, and refuse to rank items when it is low |
| Labelled difficulty vs measured: corr = +0.039, wrong sign | **Never trust a claimed difficulty.** Report claimed-vs-measured correlation as a diagnostic |
| 43 discriminating items ranked systems better than 100 random | Set-level filtering works where per-item ranking does not |
| corr(D, point-biserial) = 0.938 | Both are worth computing; they agree |
| 3 of 5 surveyed sources proposed a composite score, all with different invented weights | **No composite quality score.** Emit a scorecard and a disposition |

## 1. `src/fireassay/items/core.py` — pure, no I/O

```python
class ItemResponses(BaseModel):      # frozen
    item_id: str
    responses: tuple[bool, ...]      # one per system; order matches across all items

class ItemMeta(BaseModel):           # frozen, all optional beyond the id
    item_id: str
    question: str | None = None
    reference_answer: str | None = None
    evidence_quote: str | None = None
    source_doc_id: str | None = None
    claimed_difficulty: str | None = None
    claimed_qtype: str | None = None

Classification = Literal["live", "dead_all_pass", "dead_all_fail", "mislabel_suspect"]

class ItemStats(BaseModel):
    item_id: str
    p: float                    # fraction of systems correct
    discrimination_d: float     # top 27% minus bottom 27%
    point_biserial: float       # 0.0 when the item has no variance
    classification: Classification

class PanelStats(BaseModel):
    n_items: int
    n_systems: int
    split_half_reliability: float          # mean Kendall/Pearson over random half-splits
    reliability_verdict: Literal["usable", "too_few_systems"]
    class_counts: dict[str, int]
    claimed_vs_measured_difficulty_r: float | None   # None when no claimed labels supplied

def analyse(responses: Sequence[ItemResponses], meta: Mapping[str, ItemMeta] | None = None,
            *, reliability_floor: float = 0.5, n_splits: int = 200,
            seed: int = 0) -> tuple[list[ItemStats], PanelStats]: ...
```

Rules:

- **`mislabel_suspect`**: top systems fail it while bottom systems pass — the cheap proxy for the
  4PL upper asymptote. A genuinely hard item is failed by weak systems and passed by strong ones;
  a *mislabelled* item is failed by strong ones too, because they give the right answer to a wrong
  label.
- **`split_half_reliability`**: split systems into disjoint halves `n_splits` times, compute
  per-item point-biserial in each half, correlate. Below `reliability_floor` the verdict is
  `too_few_systems` and **`ItemStats.point_biserial` and `discrimination_d` must be reported with
  an explicit warning** — the panel cannot support per-item ranking. Classification still stands:
  zero-variance detection is robust where fine-grained ranking is not.
- **Raise** on a ragged matrix (items with differing response counts) or fewer than 4 systems.

## 2. `src/fireassay/items/adapters/`

`store.py` — read from fireassay's DB (this file may import the store; the core may not).
`tabular.py` — read a CSV or JSONL that anyone can produce:

    item_id,system_id,correct
    q1,bm25-k3,1
    q1,bm25-k5,0

Plus an optional metadata file keyed on `item_id`. Document both formats in the module docstring;
they are the public contract for standalone use.

## 3. Blind review — `src/fireassay/items/review.py`

The step that turns plausible detectors into measured ones.

```python
def build_review_batch(stats, meta, *, n_flagged: int, n_unflagged: int,
                       n_calibration: int = 0, seeded: Sequence[SeededItem] = (),
                       seed: int) -> list[ReviewItem]
```

- Mixes flagged (`mislabel_suspect`), unflagged (sampled from other classes), **seeded**
  known-bad, and **calibration** items into one shuffled batch.
- **Calibration** items are drawn uniformly at random from the whole pool, every
  classification, never filtered. They are labelled for overall quality rather than to score
  one detector, which makes them the only external referent the tool has: every other error
  rate it reports is measured against corruptions we invented ourselves. They serve detectors
  that do not exist yet, at no extra cost in reviewer time.
- **Invariant: an `item_id` appears at most once per deck.** Claim order is flagged -> seeded
  -> unflagged -> calibration, each excluding what is already claimed. Without this, the same
  question can appear once intact and once corrupted; measured at **70% of decks** on the real
  2,364-item pool before the invariant existed. That does not merely leak, it hands the
  reviewer the answer key and double-counts one item in the recall denominator.
- `ReviewItem` exposes **only** the question, reference answer and evidence. It must **not**
  carry the classification, the statistics, or whether the item was seeded. A reviewer who can
  see the verdict cannot measure it.
- The mapping from review position back to ground truth is written to a **separate** file the
  reviewer never opens.

A reviewer records one **disposition** per item — `keep` / `rewrite` / `purge`, the
`handbook/eval-of-evals.md` §6 vocabulary. Never a score. Every item offers the same three
choices whatever its source: if calibration items asked a richer question than flagged ones, a
reviewer could tell the groups apart and the blind would be gone. `rewrite` is the middle a
binary throws away, and is what a later ambiguity or gold-answerability detector gets measured
against.

```python
def score_review(labels, key, *, bad_verdicts=frozenset({"purge"})) -> DetectorScore
```

Recall comes from the seeded items, precision from the flagged ones. **`DetectorScore` must carry
a confidence interval on both** — at these sample sizes a bare point estimate is misleading, and
this tool exists to object to exactly that. Strict (`{"purge"}`) versus lenient
(`{"purge", "rewrite"}`) is the caller's call, obtained by calling twice; blessing one as
canonical would be the unmeasured threshold this tool refuses to produce.

`review build --worksheet out.csv` writes a fill-in CSV — `review_id,verdict,question,
reference_answer,evidence_quote`, verdict blank. Bound by the identical leak rule. `review
score --labels` accepts it directly, so the filled worksheet *is* the labels file; a blank
verdict is skipped entirely rather than becoming a fourth verdict.

## 4. Seeding — `src/fireassay/items/seed.py`

Deliberate corruptions with known ground truth, for measuring recall without human time:

| kind | corruption |
|---|---|
| `swapped_reference` | reference answer taken from a different item |
| `foreign_evidence` | evidence span pointing at an unrelated document |
| `truncated_question` | question cut mid-clause |
| `negated_reference` | reference answer's polarity flipped |

Deterministic given a seed. Each records what was done.

**No corruptor may emit an item identical to its source.** Enforce centrally, and raise
`ValueError` so `seed_batch` skips that item — a corruption that changed nothing is not a
known-bad, and a reviewer who correctly calls it `keep` is then scored as a recall miss,
biasing the exact number seeding exists to produce. Measured before the invariant: **48 of
9,456** possible seeds were sound items, 47 of them bare `"No"` reference answers where
removing the only negation word left nothing and the text came back unchanged — while the
recorded `detail` asserted a polarity flip that never happened.

**State the limitation in the module docstring:** seeded flaws may be easier to detect than
natural ones, so seeded recall is an **upper bound**. Say so in the report too.

## 5. CLI

    fireassay items analyse --matrix results.csv [--meta meta.jsonl]      # standalone
    fireassay items analyse --db run/golden.db --suite name@ver           # in-project
    fireassay items review build --matrix results.csv --meta meta.jsonl \
        --out batch.jsonl --key key.json [--worksheet sheet.csv]         # standalone
    fireassay items review build --db ... --out batch.jsonl --key key.json
    fireassay items review score --labels labels.jsonl|sheet.csv --key key.json
    fireassay items review export-calibration --labels ... --key ... --out calib.jsonl
    fireassay items detector-score --flags flags.txt --calibration calib.jsonl

`analyse` prints the scorecard and the panel stats, **leading with `split_half_reliability`** —
a reader must see how much to trust the per-item numbers before they see the numbers.

## 6. Tests

| File | Must prove |
|---|---|
| `test_items_core.py` | hand-computed p, D, point-biserial on a fixed matrix; classification boundaries; ragged matrix raises; <4 systems raises |
| `test_items_reliability.py` | a matrix built to be internally inconsistent yields `too_few_systems`; a consistent one yields `usable` |
| `test_items_no_store_import.py` | **`items.core` imports no storage** — walk its module graph and assert `fireassay.store` is absent. This is the guard on the whole design |
| `test_items_tabular.py` | CSV round-trip; malformed rows rejected with a clear message |
| `test_items_review.py` | a `ReviewItem` leaks neither classification, statistics, nor seeded status; batch order is seed-deterministic |
| `test_items_seed.py` | each corruption is applied and recorded; deterministic under a seed |
| `test_items_score.py` | precision/recall arithmetic on a hand-worked example; CIs widen as n shrinks |
| `test_items_review_cli.py` | standalone `--matrix`/`--meta` and in-project `--db` produce identical batch and key; worksheet round-trip; `export-calibration` emits no seeded item |
| `test_items_calibration.py` | **the stratum rule** — pooling a flagged census with a random sample gives a visibly wrong base rate, and the test fails loudly if anyone ever pools them; a structural zero is reported as `needs_sampling_design`, never as `0.0` |

## 7. Out of scope

Closed-book / retrieval contribution, gold answerability, ambiguity-by-disagreement,
counterfactual stability — all need a judge and arrive with M4. Leave interfaces, build nothing.
No composite score. No Cronbach's alpha.

The **calibration consumer** (`items/calibration.py`, `detector-score`) is deliberately *in*
scope despite scoring detectors that do not exist yet. It is what lets each of the four above
ship with a measured error rate the day it lands, against labels already collected, costing no
further reviewer time. Building it after those detectors would mean asking for the human pass
twice. It is `handbook/eval-of-evals.md` §8 — *every detector this tool ships must itself be
evaluated* — made runnable rather than aspirational.

## 8. If anything is ambiguous

Stop and ask — especially the no-storage-import rule (§the constraint) and the blind-review
leak rule (§3). A plausible guess in either defeats the point of the tool.
