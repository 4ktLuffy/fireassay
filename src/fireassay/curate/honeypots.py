"""Known-bad honeypots (M3-SPEC.md §4).

A honeypot is a real candidate, deterministically corrupted, whose correct
verdict is known in advance: the curator **must reject it**. M3 ships only
"known-bad" honeypots — available immediately, with no bootstrap round —
and leaves a hook for "known-good" (real candidates a second curator has
already verified), which needs at least one completed curation round to
exist and so cannot ship in M3.

The corruption itself is **not stored**: only `honeypot_expected_reason` is
persisted on the `queue_item` row (see `queue.py`). The corrupted rendering
a curator actually sees is recomputed deterministically at serve time
(`curate.serve.next_item`) from `(candidate, seed)` — the same
determinism-over-storage trade every other seeded mechanism in this
codebase makes (`mutation.operators`, `controls.shuffled_gold`).

**Honeypots measure correctness, not agreement** — that is `agreement.py`'s
job, and the two are deliberately separate instruments (M3-SPEC.md §4).
"""

from __future__ import annotations

from collections.abc import Sequence

from fireassay.curate._common import stable_seed
from fireassay.curate.models import Decision, QueueItem, RejectReason
from fireassay.generate.models import ResolvedCandidate

#: Known-bad corruption is always one of these two — M3-SPEC.md §4 names
#: exactly these mechanisms as available with no bootstrap.
_KNOWN_BAD_REASONS: tuple[RejectReason, RejectReason] = ("WRONG_REFERENCE", "INSUFFICIENT_EVIDENCE")


def choose_honeypot_reason(candidate_id: str, seed: int) -> RejectReason:
    """Deterministically choose which known-bad corruption a given
    honeypot slot uses, alternating between the two available mechanisms
    so a report is not dominated by only one corruption type."""
    return _KNOWN_BAD_REASONS[stable_seed(seed, candidate_id, "honeypot_kind") % 2]


def corrupt_candidate_view(
    candidate: ResolvedCandidate,
    pool: Sequence[ResolvedCandidate],
    reason: RejectReason,
    seed: int,
) -> dict[str, object]:
    """Render `candidate` as JSON-serialisable curation-item fields, with
    exactly one field corrupted according to `reason`:

    - `WRONG_REFERENCE`: swap in another candidate's `reference_answer`.
    - `INSUFFICIENT_EVIDENCE`: swap in a quote/span from an unrelated
      document (any candidate whose `source_doc_id` differs from this
      one's).

    The "other candidate" is chosen deterministically from `pool` via
    `stable_seed`, so the same `(candidate, pool, reason, seed)` always
    produces the same corrupted view — required for `curate next` to
    re-serve an un-decided honeypot identically if asked twice.
    """
    base: dict[str, object] = {
        "text": candidate.text,
        "qtype": candidate.qtype,
        "difficulty": candidate.difficulty,
        "reference_answer": candidate.reference_answer,
        "quote": candidate.quote,
        "source_doc_id": candidate.source_doc_id,
        "char_start": candidate.char_start,
        "char_end": candidate.char_end,
    }

    if reason == "WRONG_REFERENCE":
        others = [c for c in pool if c.id != candidate.id]
        if not others:
            raise ValueError("cannot build a WRONG_REFERENCE honeypot from a pool of size 1")
        pick = others[stable_seed(seed, candidate.id, "wrong_reference") % len(others)]
        base["reference_answer"] = pick.reference_answer
        return base

    if reason == "INSUFFICIENT_EVIDENCE":
        others = [c for c in pool if c.source_doc_id != candidate.source_doc_id]
        if not others:
            raise ValueError(
                "cannot build an INSUFFICIENT_EVIDENCE honeypot: no candidate in the pool "
                "comes from a different source document"
            )
        pick = others[stable_seed(seed, candidate.id, "insufficient_evidence") % len(others)]
        base["quote"] = pick.quote
        base["source_doc_id"] = pick.source_doc_id
        base["char_start"] = pick.char_start
        base["char_end"] = pick.char_end
        return base

    raise ValueError(f"unsupported known-bad honeypot reason: {reason!r}")


def is_honeypot_correct(decision: Decision) -> bool:
    """A curator handles a known-bad honeypot correctly iff they reject
    it — M3-SPEC.md §4: "the curator must reject it". A curator who
    accepts (or edits, treating it as salvageable) a known-bad item scores
    incorrect for it, regardless of which specific reject_reason they
    picked: the corruption mechanism (`WRONG_REFERENCE` vs
    `INSUFFICIENT_EVIDENCE`) tells us what we injected, not the only
    legitimate way a human could correctly describe why the item is bad.
    """
    return decision.decision == "reject"


def honeypot_accuracy(pairs: Sequence[tuple[QueueItem, Decision | None]]) -> float | None:
    """`correct honeypot verdicts / honeypots seen`, over `(queue_item,
    decision)` pairs restricted internally to `is_honeypot` items with a
    recorded decision. `None` (not `0.0`) when no honeypot has been seen
    yet — "no honeypots decided" and "every honeypot decided wrongly" must
    stay distinguishable, the same omit-don't-zero rule `score.base`
    documents for scorers generally.
    """
    seen = [(item, decision) for item, decision in pairs if item.is_honeypot and decision is not None]
    if not seen:
        return None
    correct = sum(1 for _item, decision in seen if is_honeypot_correct(decision))
    return correct / len(seen)
