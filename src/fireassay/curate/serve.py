"""`curate next` / `curate submit`'s core logic (M3-SPEC.md §6) — the
non-interactive primitives the M3b TUI will drive. No terminal I/O here;
`cli.py` is the only place that touches stdin/stdout for this package.
"""

from __future__ import annotations

from dataclasses import dataclass

from fireassay.curate.honeypots import corrupt_candidate_view
from fireassay.curate.models import Decision, QueueItem
from fireassay.curate.queue import build_queue, seed_from_candidates
from fireassay.generate.models import ResolvedCandidate
from fireassay.store.db import Store

DEFAULT_HONEYPOT_RATE = 0.05
DEFAULT_DOUBLE_REVIEW_RATE = 0.10


@dataclass(frozen=True)
class ServedItem:
    """What `curate next` emits as JSON. Deliberately excludes
    `QueueItem.is_honeypot`/`is_double_review`/`honeypot_expected_reason` —
    see `curate.models.QueueItem`'s docstring for why: a curator who could
    see them would behave differently on flagged items."""

    queue_item_id: str
    candidate_id: str
    curator_id: str
    view: dict[str, object]


def _candidate_view(candidate: ResolvedCandidate) -> dict[str, object]:
    return {
        "text": candidate.text,
        "qtype": candidate.qtype,
        "difficulty": candidate.difficulty,
        "reference_answer": candidate.reference_answer,
        "quote": candidate.quote,
        "source_doc_id": candidate.source_doc_id,
        "char_start": candidate.char_start,
        "char_end": candidate.char_end,
    }


def ensure_queue(
    store: Store,
    curator_id: str,
    *,
    honeypot_rate: float = DEFAULT_HONEYPOT_RATE,
    double_review_rate: float = DEFAULT_DOUBLE_REVIEW_RATE,
) -> None:
    """Lazily build `curator_id`'s queue, over every currently filter-kept
    candidate, the first time it is needed — there is no separate
    "build queue" command in M3-SPEC.md §6's CLI, so `curate next` is
    responsible for materialising a queue on first use. A no-op if
    `curator_id` already has queue items (queues are built once per
    curator, over whatever candidate pool existed at that time; a later,
    larger pool needs an explicit new curator_id/queue to pick up, not a
    silent extension of an existing one mid-review)."""
    if store.has_queue_items_for_curator(curator_id):
        return
    candidate_ids = store.get_kept_candidate_ids()
    if not candidate_ids:
        return
    candidates = [store.get_candidate(cid) for cid in sorted(candidate_ids)]
    seed = seed_from_candidates([c.id for c in candidates])
    items: list[QueueItem] = build_queue(
        candidates, curator_id=curator_id, honeypot_rate=honeypot_rate, double_review_rate=double_review_rate,
        seed=seed,
    )
    store.put_queue_items(items)


def next_item(
    store: Store,
    curator_id: str,
    *,
    honeypot_rate: float = DEFAULT_HONEYPOT_RATE,
    double_review_rate: float = DEFAULT_DOUBLE_REVIEW_RATE,
) -> ServedItem | None:
    """The next undecided item in `curator_id`'s queue, or `None` when the
    queue is exhausted (or empty — no filter-kept candidates exist yet)."""
    ensure_queue(store, curator_id, honeypot_rate=honeypot_rate, double_review_rate=double_review_rate)
    item = store.next_undecided_queue_item(curator_id)
    if item is None:
        return None

    candidate = store.get_candidate(item.candidate_id)
    if item.is_honeypot:
        if item.honeypot_expected_reason is None:
            raise ValueError(f"queue item {item.id} is_honeypot but has no honeypot_expected_reason")
        pool_ids = sorted(store.get_kept_candidate_ids())
        pool = [store.get_candidate(cid) for cid in pool_ids]
        seed = seed_from_candidates(pool_ids)
        view = corrupt_candidate_view(candidate, pool, item.honeypot_expected_reason, seed)
    else:
        view = _candidate_view(candidate)

    return ServedItem(queue_item_id=item.id, candidate_id=candidate.id, curator_id=curator_id, view=view)


def submit_decision(store: Store, decision: Decision) -> Decision:
    """Persist `decision` (minting `id`/`decided_at`) and return the
    stored row. A thin pass-through over `Store.put_decision` — kept as its
    own function so `cli.py`'s `curate submit` and any future TUI call the
    same, single code path rather than either reaching into `Store`
    directly."""
    return store.put_decision(decision)
