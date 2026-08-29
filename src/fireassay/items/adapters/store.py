"""Build `items.core`'s input from fireassay's own SQLite store.

This is the **only** module in `fireassay.items` allowed to import
`fireassay.store` (M-ITEMS-SPEC.md's constraint; `items.core` may not,
directly or transitively -- see `test_items_no_store_import.py`).

**Why `metric`/`threshold` are required, with no default (read before
changing this):** fireassay's own runs record *continuous* retrieval
metrics (`retrieval.recall@k`, `ndcg@k`, `mrr`, `bpref`, ...); there is no
metric anywhere in the store that already means "this system answered
this item correctly" as a plain boolean -- that would need an LLM judge,
which arrives with M4. Turning a continuous score into `ItemResponses`'
boolean `correct` therefore requires picking both a metric name and a
cutoff, and different reasonable choices (e.g. `retrieval.recall@5 >
0.999` -- "every gold span covered" -- vs `> 0.0` -- "found at least
something") produce materially different item statistics. Inventing a
default here would be exactly the kind of unmeasured, invented threshold this tool
exists to refuse to produce (see `items.core`'s module docstring on
Cronbach's alpha and the composite score). Callers -- including
`fireassay items analyse --db ...`'s CLI wiring -- must state their
criterion explicitly.

**The convention behind the recorded measurements** (`handbook/eval-of-
evals.md`'s 2,364-item panel numbers, and the split-half-reliability
figures in `items.core`'s docstrings) is `metric="retrieval.recall@5"`,
`threshold=0.0` -- an item counts as correct iff at least one retrieved
chunk overlapped its gold evidence span at all (`recall@5 > 0.0`). This is
documented here as the convention those specific numbers were produced
under, so it is discoverable rather than folklore -- **it is not a
default**, and nothing in this module applies it automatically; a caller
who wants numbers comparable to those must pass `metric`/`threshold`
themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

from fireassay.items.core import ItemMeta, ItemResponses
from fireassay.models import Run
from fireassay.store.db import Store


def load_matrix(store: Store, runs: Sequence[Run], *, metric: str, threshold: float) -> list[ItemResponses]:
    """Build a response matrix over `runs` -- one "system" per run, in the
    order given (callers typically pass `store.runs_for_suite(suite.id)`
    or some subset of it), `system_id` implicitly `run.id`. An item scores
    `correct = (value > threshold)` for `metric` in a given run.

    Strict `>`, not `>=`: `retrieval.recall@k` and friends are bounded
    below at exactly `0.0` with no fixed smallest-positive-increment (the
    granularity depends on a question's own gold-span count), so `>=` with
    `threshold=0.0` cannot express "found at least something" at all --
    every item is `recall >= 0.0` trivially. `>` makes `threshold=0.0`
    (see "the convention behind the recorded measurements" above) mean
    exactly what it says.

    An item missing `metric` in **any** of `runs` (e.g. a question with no
    gold evidence spans, which `score.retrieval.RetrievalScorer` omits
    entirely rather than scoring `0.0` -- see its module docstring) is
    dropped from the matrix rather than backfilled: `items.core.analyse`
    requires a rectangular matrix, and inventing a `0`/`1` for a metric
    that was never computed would misrepresent "not applicable" as
    "measured and failed".
    """
    if not runs:
        raise ValueError("load_matrix: no runs given")

    per_run_scores: list[dict[str, float]] = []
    for run in runs:
        rows = store.run_scores(run.id)
        per_run_scores.append({question_id: value for question_id, m, value in rows if m == metric})

    item_ids = set(per_run_scores[0])
    for by_item in per_run_scores[1:]:
        item_ids &= set(by_item)
    if not item_ids:
        raise ValueError(
            f"load_matrix: no item has metric {metric!r} recorded in every one of {len(runs)} run(s)"
        )

    return [
        ItemResponses(
            item_id=item_id,
            responses=tuple(scores[item_id] > threshold for scores in per_run_scores),
        )
        for item_id in sorted(item_ids)
    ]


def load_meta(store: Store, suite_id: str) -> dict[str, ItemMeta]:
    """Build `ItemMeta` for every question in `suite_id` -- for blind
    review/seeding (`items.review`, `items.seed`) and for
    `PanelStats.claimed_vs_measured_difficulty_r`.

    `evidence_quote`/`source_doc_id` are taken from the question's first
    evidence span (an arbitrary but deterministic tie-break for a
    multi-span question -- review/seeding only need one representative
    quote to show a reviewer, not the full span set). `claimed_difficulty`/
    `claimed_qtype` are the question's own `difficulty`/`qtype` fields --
    fireassay's *generator's proposal or curator's rubric call*, never a
    measured property (see `items.core`'s "never trust a claimed
    difficulty" rule).
    """
    out: dict[str, ItemMeta] = {}
    for q in store.iter_questions(suite_id):
        span = q.evidence_spans[0] if q.evidence_spans else None
        out[q.id] = ItemMeta(
            item_id=q.id,
            question=q.text,
            reference_answer=q.reference_answer,
            evidence_quote=span.quote if span else None,
            source_doc_id=q.source_doc_id,
            claimed_difficulty=q.difficulty,
            claimed_qtype=q.qtype,
        )
    return out
