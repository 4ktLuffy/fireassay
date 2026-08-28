# fireassay M3b — the curation TUI, and incremental resume

Repo root: `/Users/Twinkle/AI Engineering/fireassay`
Read `M3-SPEC.md` §4 and `spike/RESULTS.md` first. M1–M3 are green: **332 tests, 92%
coverage, ruff + mypy --strict clean.** Do not regress them.

## Why this exists

One curator is about to review **1,000 items by hand**. That is 2–3 hours of a real person's
attention, and it is the one part of this project that cannot be automated. Every second per
item costs ~17 minutes across the set.

The research is specific about what makes annotation fast and what makes it degrade:

- **keyboard shortcuts are the dominant factor** in real annotation throughput
- **eliminate scrolling** — it causes fatigue and wastes time
- **consistent element placement** minimises eye movement
- quality degrades measurably past **~45 minutes** in a session
- **autopilot** shows up as long runs of identical verdicts

## ⚠ Constraint that overrides everything

**DO NOT MODIFY `src/fireassay/generate/prompts.py` OR ANY PROMPT TEXT.**

A generation run is in flight right now, and its cache holds hundreds of responses keyed on
byte-identical prompts. Changing a prompt by one character invalidates the checkpoint and
throws away hours of model time. If a change seems to require touching a prompt, stop and ask.

---

## Part 1 — incremental resume (`generate/pipeline.py`)

Two measured problems with one fix.

**Re-running into an existing db duplicates candidates.** Verified: 5 candidates became 10,
two batches, duplicate question texts. Candidate ids are random uuid4s, so reprocessing a
chunk mints a new row.

**Replaying a cache into a fresh db costs ~1s/candidate even with zero model calls** — about
an hour at 4,000. Cause unisolated: BM25 at gold-rank depth is 35ms and the retriever is built
once outside the loop, so neither explains it. **Profile it before optimising; do not guess.**
The per-row SQLite write pattern is the leading suspect.

The fix for both: **skip chunks the target database already has candidates for.**

    already = store.candidate_source_chunks()   # set of chunk_ids with >=1 candidate row
    ... skip any selected chunk already in that set

- Record `chunk_id` on the `candidate` row (migration `0005`) — `source_doc_id` is not enough,
  since several chunks share a document.
- A resumed run reports what it skipped: `resumed=N chunks already present`.
- **`rm -f run/golden.db` stops being necessary.** Update the README's resume instructions.
- If profiling shows the write pattern is the cost, batch inserts into one transaction per
  chunk. Report the before/after numbers; do not claim an improvement you did not measure.

Tests: a second run over the same db adds no duplicate candidates; a run over a partially
populated db processes only the missing chunks; skipped counts are reported.

---

## Part 2 — the TUI (`src/fireassay/curate/tui.py`)

**Textual.** Add `textual` as a runtime dependency. It is the only new one.

### The rule that shapes the whole design

**The fast path is one keystroke.** Most candidates are fine. A curator who must answer six
rubric questions on every item will average 15s and quit at item 300. So:

- **`a` accepts** with the rubric defaulted to "everything agrees" — `answerable_from_kb=yes`,
  `self_contained=true`, `reference_answer_correct=yes`, `evidence_sufficient=true`,
  `difficulty_agrees=true`, `qtype_agrees=true`.
- The six criteria are only touched when something is actually wrong, via `d` (drill in).
- `r` rejects, then a single keystroke picks the reason.

That is the difference between ~5s and ~15s per item — between two hours and five.

### Layout — fixed, no scrolling, ever

Everything fits one screen at 80x24. Long text is **truncated with an explicit marker**, never
scrolled. `v` opens the full evidence in a modal for the rare case it is needed.

    ┌ fireassay curate ──────────── henos · 127/1000 · session 23m ─┐
    │ QUESTION                                                      │
    │   What must you do if you cancel your Direct Debit before a   │
    │   monthly payment is taken?                                   │
    │                                                               │
    │ REFERENCE ANSWER                                              │
    │   You must contact HMRC to arrange another way to pay.        │
    │                                                               │
    │ EVIDENCE   self-assessment-tax-returns · chars 5495–5698      │
    │   ...you must tell HMRC if you cancel a Direct Debit so that  │
    │   another payment method can be arranged... [+312 chars, v]   │
    │                                                               │
    │ PROPOSED   policy_sensitive / hard          gold rank 2       │
    ├───────────────────────────────────────────────────────────────┤
    │ a accept   r reject   e edit   d rubric   v evidence   ? help │
    └───────────────────────────────────────────────────────────────┘

Element positions are **identical on every item** — the eye should never hunt.

### Keys

| key | action |
|---|---|
| `a` | accept with default rubric, advance |
| `r` | reject → reason picker (single keystroke per code) |
| `e` | edit question text and/or reference answer, then accept |
| `d` | drill into the six rubric criteria |
| `v` | full evidence in a modal |
| `u` | undo the last submission (see below) |
| `?` | help |
| `q` | save and quit |

Reason picker, one keystroke each: `d`uplicate · `n`ot answerable · `a`mbiguous ·
`w`rong reference · `i`nsufficient evidence · `l`eading · `t`rivial · `o`ut of scope · `p`ii.
Show the letters on screen; a curator must never have to remember them.

### Undo

`u` re-opens the previous item for a corrected verdict. Decisions are append-only (M3), so this
writes a **new** decision row; the report already takes the latest per `(candidate, curator)`.
Do not delete anything. Undo depth of 1 is enough.

### Honeypots stay invisible

The TUI renders a honeypot **identically** to a real item and never hints. A visible honeypot
measures nothing.

### Session awareness

- Header shows elapsed session time. At **45 minutes**, a non-blocking banner suggests a break —
  the documented threshold past which annotation quality measurably degrades.
- After **15 consecutive identical verdicts**, a one-line note: *"15 accepts in a row — still
  reading?"* Autopilot is a documented failure mode and the data to detect it is already stored.
- Neither ever blocks input. A curator who wants to continue continues.

### It adds no logic

The TUI calls the existing `curate.serve.next_item` / `submit_decision`. **Every rule lives in
the core and is already tested.** If the TUI needs a decision the core cannot express, that is a
gap in the core — fix it there, with tests, not in the UI.

`duration_ms` is measured from item render to keypress.

### CLI

    fireassay curate tui --curator henos --db run/golden.db

## Tests

A Textual app cannot be fully tested headlessly here, which is exactly why the logic lives in
the core. Test what is testable and say plainly what is not:

| File | Must prove |
|---|---|
| `test_tui_logic.py` | the default-accept rubric is exactly the six "agrees" values; the reason-key map is a bijection onto the nine codes; truncation marks how much was hidden and never silently drops text |
| `test_tui_session.py` | 45-minute banner fires on time; the identical-verdict run counter fires at 15 and resets on a different verdict |
| `test_incremental_resume.py` | no duplicates on re-run; only missing chunks processed; skip counts reported |

Pull every decision worth testing out of the widget classes into plain functions and test those.

## Out of scope

LLM judges, CJE, pooling, the gate, `unanswerable` generation, HTML reports. And, again: **no
prompt text changes.**

## If anything is ambiguous

Stop and ask. Especially the one-keystroke default-accept rubric (§Part 2) and the
skip-already-present rule (§Part 1).
