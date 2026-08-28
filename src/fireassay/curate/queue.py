"""`build_queue` (M3-SPEC.md §4).

Honeypots and double-reviews are **selected** independently of
`curator_id` — from `candidates` and `seed` alone — so that calling
`build_queue` once per curator over the same candidate set (and the same
seed) flags the *same* candidate ids as honeypots/double-reviews for every
curator: that overlap is what makes cross-curator agreement (`agreement.py`)
and per-curator honeypot accuracy (`honeypots.py`) measure the same items.
Only each curator's **position order** differs (seeded additionally by
`curator_id`), so two curators genuinely see the flagged items at
different points in their own queue, not lined up in lockstep.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from fireassay.curate._common import stable_seed
from fireassay.curate.honeypots import choose_honeypot_reason
from fireassay.curate.models import QueueItem
from fireassay.generate.models import ResolvedCandidate
from fireassay.hashing import content_hash

_QUEUE_ID_PREFIX = "fa.queue.1"
_QUEUE_ITEM_ID_PREFIX = "fa.queueitem.1"


def seed_from_candidates(candidate_ids: Sequence[str]) -> int:
    """A deterministic seed derived purely from the candidate-set content
    (M3-SPEC.md §4: "the seed derives from the candidate-set hash") — so
    re-building a queue over an unchanged candidate pool reproduces the
    exact same honeypot/double-review selection, and a pool that grows
    (more candidates curated in a later batch) produces a different,
    still-deterministic seed rather than silently reusing a stale one."""
    return stable_seed(content_hash("fa.queueseed.1", "\n".join(sorted(candidate_ids))))


def build_queue(
    candidates: Sequence[ResolvedCandidate],
    *,
    curator_id: str,
    honeypot_rate: float = 0.05,
    double_review_rate: float = 0.10,
    seed: int,
) -> list[QueueItem]:
    """Build one curator's queue over `candidates`.

    Deterministic given `seed`: the same `(candidates, seed)` always
    selects the same honeypot/double-review subset, and the same
    `(candidates, seed, curator_id)` always produces the same position
    order — required for `curate next` to serve the same item again if
    asked twice before a decision is recorded, and for `test_queue.py`'s
    determinism assertion.
    """
    ordered = sorted(candidates, key=lambda c: c.id)
    n = len(ordered)
    queue_id = content_hash(
        _QUEUE_ID_PREFIX, str(seed), "\n".join(c.id for c in ordered)
    )

    # Selection of *which* candidates are honeypots/double-reviews:
    # curator-independent, so every curator's queue over the same
    # candidate pool and seed flags the same items.
    selection_order = list(range(n))
    random.Random(seed).shuffle(selection_order)
    n_honeypot = round(n * honeypot_rate)
    n_double = round(n * double_review_rate)
    honeypot_indices = set(selection_order[:n_honeypot])
    double_review_indices = set(selection_order[n_honeypot : n_honeypot + n_double])

    # Position order: additionally seeded by curator_id, so different
    # curators see the (curator-independently) flagged items at different
    # points in their own queue.
    position_order = list(range(n))
    random.Random(stable_seed(seed, curator_id)).shuffle(position_order)

    items: list[QueueItem] = []
    for position, idx in enumerate(position_order, start=1):
        candidate = ordered[idx]
        is_honeypot = idx in honeypot_indices
        expected_reason = choose_honeypot_reason(candidate.id, seed) if is_honeypot else None
        item_id = content_hash(_QUEUE_ITEM_ID_PREFIX, queue_id, curator_id, candidate.id)
        items.append(
            QueueItem(
                id=item_id,
                queue_id=queue_id,
                curator_id=curator_id,
                candidate_id=candidate.id,
                position=position,
                is_honeypot=is_honeypot,
                honeypot_expected_reason=expected_reason,
                is_double_review=idx in double_review_indices,
            )
        )
    return items
