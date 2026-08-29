"""Deliberate, deterministic corruptions of real items, for measuring a
detector's recall without spending human review time on items known in
advance to be bad -- the same "known-bad honeypot" idea as
`fireassay.curate.honeypots`, adapted to items rather than curation
candidates (and duplicated rather than imported: `items/` must stay
dependency-free of `curate/`, `controls/`, and `mutation/`, exactly the
same "extractable standalone package" discipline `items.core`'s no-store
rule enforces for `fireassay.store` -- see `curate._common.stable_seed`'s
own docstring for why this codebase duplicates this one helper per
package rather than sharing it).

**Seeded recall is an upper bound, not an estimate of true recall.** A
corruption applied by this module is, by construction, exactly the kind of
flaw its own corruptor was designed to hide -- a detector (or a human
reviewer) may catch a seeded `swapped_reference` more easily than a
naturally occurring wrong reference that happens to be topically close to
the item it belongs to. Whatever recall `items.review.score_review`
reports from seeded items, report it as an upper bound on real-world
recall, never as the number itself -- both in code comments here and in
every place seeded recall reaches a human (see
`report.text.render_items_score`).

Four corruption kinds (M-ITEMS-SPEC.md §4):

| kind | corruption |
|---|---|
| `swapped_reference` | reference answer taken from a different item |
| `foreign_evidence` | evidence span/quote taken from an unrelated document |
| `truncated_question` | question cut mid-clause |
| `negated_reference` | reference answer's polarity mechanically flipped |

A fifth kind, `lexical_decoy` (evidence relocated into the corpus document a
retriever ranks highest for the item's question, excluding the item's own
source document), is declared in `SeedKind` below but has no corruptor
here and is **not** in `ALL_SEED_KINDS` -- picking a decoy needs a
retriever, which `items/` itself must stay dependency-free of (same rule
that keeps `fireassay.store` out, see `items.core`'s module docstring).
It is built by `items.adapters.retrieval.build_lexical_decoy_seeds`
instead, the sanctioned place `items/` meets the rest of fireassay.

Each corruptor raises `ValueError` when the source item lacks the field it
needs (e.g. `truncated_question` on an item with no `question`, or
`foreign_evidence` when no other item in the pool comes from a different
`source_doc_id`) -- `seed_batch` catches this per item and simply moves on
to the next candidate, since seeding is a best-effort augmentation over
whatever metadata happens to be complete enough to corrupt, not a
guarantee every item can satisfy every kind.

**A corruption that changes nothing is not a known-bad.** Measured on the
real 2,364-item pool x 4 kinds (9,456 seeds): 47 `negated_reference` seeds
(2.0%) and 1 `swapped_reference` seed were byte-identical to their source
item -- a bare `"No"` reference answer whose only negation word, once
removed, left nothing, silently returned unchanged by the old
`_negate_text`; and a donor whose `reference_answer` happened to already
equal the item's own. A sound item presented as a known-bad is not a
harder seed, it is a *wrong* one: a reviewer who correctly calls it `keep`
gets counted as a recall miss, silently biasing the one number this module
exists to produce. Every corruptor therefore builds its `SeededItem`
through `_seeded_item` below, never `SeededItem(...)` directly -- it
refuses (raises `ValueError`, caught the same as any other
can't-corrupt-this-item failure) to emit a `SeededItem` whose
`question`/`reference_answer`/`evidence_quote` all equal the source
`ItemMeta`'s.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from fireassay.items.core import ItemMeta

#: `lexical_decoy` is a declared kind with no corruptor in this module --
#: see the module docstring's "a fifth kind" paragraph.
SeedKind = Literal[
    "swapped_reference", "foreign_evidence", "truncated_question", "negated_reference", "lexical_decoy"
]

#: The kinds `seed_batch` can produce -- deliberately still the original
#: four; `lexical_decoy` is built elsewhere (see the module docstring).
ALL_SEED_KINDS: tuple[SeedKind, ...] = (
    "swapped_reference",
    "foreign_evidence",
    "truncated_question",
    "negated_reference",
)


class SeededItem(BaseModel):
    """A known-bad item: `item_id` names the real item this was derived
    from (its *original*, uncorrupted fields are never included here --
    only the corrupted rendering is), `detail` records exactly what was
    done, so a report or a test can say precisely what makes this item
    wrong without re-deriving it."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    kind: SeedKind
    question: str | None
    reference_answer: str | None
    evidence_quote: str | None
    source_doc_id: str | None
    detail: str


def _stable_seed(*parts: object) -> int:
    """Deliberately duplicated, not imported, from
    `curate._common.stable_seed` -- see this module's docstring."""
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def stable_seed(*parts: object) -> int:
    """Public forwarder to `_stable_seed` -- exposed for the same reason
    `seeded_item` is: `items.adapters.retrieval` needs a deterministic
    integer to seed `random.Random` with (`random.Random` does not accept
    a tuple -- only `None`/`int`/`float`/`str`/`bytes`/`bytearray`), and
    must not invent a second hashing scheme of its own for it."""
    return _stable_seed(*parts)


def _seeded_item(
    item: ItemMeta,
    *,
    kind: SeedKind,
    question: str | None,
    reference_answer: str | None,
    evidence_quote: str | None,
    source_doc_id: str | None,
    detail: str,
) -> SeededItem:
    """The single constructor every corruptor must build its `SeededItem`
    through -- see the module docstring's "a corruption that changes
    nothing is not a known-bad" rule. Centralised here rather than
    checked per-corruptor so the invariant holds regardless of which of
    the four (or a future fifth) corruptor is doing the constructing:
    refuses to emit a `SeededItem` whose `question`/`reference_answer`/
    `evidence_quote` all equal the source `ItemMeta`'s, the same
    `ValueError`-per-item, `seed_batch`-skips-and-continues contract every
    other can't-corrupt-this-item failure already uses."""
    if (
        question == item.question
        and reference_answer == item.reference_answer
        and evidence_quote == item.evidence_quote
    ):
        raise ValueError(
            f"{kind}: corruption of item {item.item_id!r} produced no change at all -- "
            "question/reference_answer/evidence_quote all equal the source; refusing to "
            "emit an unchanged item as a known-bad seed"
        )
    return SeededItem(
        item_id=item.item_id,
        kind=kind,
        question=question,
        reference_answer=reference_answer,
        evidence_quote=evidence_quote,
        source_doc_id=source_doc_id,
        detail=detail,
    )


def seeded_item(
    item: ItemMeta,
    *,
    kind: SeedKind,
    question: str | None,
    reference_answer: str | None,
    evidence_quote: str | None,
    source_doc_id: str | None,
    detail: str,
) -> SeededItem:
    """Public forwarder to `_seeded_item` (see its docstring for the
    invariant it enforces) -- exposed so `items.adapters.retrieval`,
    which lives outside this module and must not duplicate the
    never-emit-an-unchanged-item check, can build a `lexical_decoy`
    `SeededItem` through the exact same enforcement point every corruptor
    in this module already uses, rather than re-deriving the rule."""
    return _seeded_item(
        item,
        kind=kind,
        question=question,
        reference_answer=reference_answer,
        evidence_quote=evidence_quote,
        source_doc_id=source_doc_id,
        detail=detail,
    )


def swap_reference(item: ItemMeta, pool: Sequence[ItemMeta], seed: int) -> SeededItem:
    """`swapped_reference`: `reference_answer` taken from a different item
    in `pool`, chosen deterministically from `seed`. Donors whose
    `reference_answer` is exactly equal to this item's own (compared
    exactly, no normalisation -- an answer differing only in case or
    trailing punctuation is still a genuinely different string a reviewer
    would read as wrong) are excluded: swapping in an identical answer is
    not a corruption at all."""
    donors = [
        m
        for m in pool
        if m.item_id != item.item_id and m.reference_answer and m.reference_answer != item.reference_answer
    ]
    if not donors:
        raise ValueError(
            f"swap_reference: no donor with a reference_answer different from item "
            f"{item.item_id!r}'s own"
        )
    donor = donors[_stable_seed(seed, item.item_id, "swapped_reference") % len(donors)]
    return _seeded_item(
        item,
        kind="swapped_reference",
        question=item.question,
        reference_answer=donor.reference_answer,
        evidence_quote=item.evidence_quote,
        source_doc_id=item.source_doc_id,
        detail=f"reference_answer swapped from item {donor.item_id!r}",
    )


def foreign_evidence(item: ItemMeta, pool: Sequence[ItemMeta], seed: int) -> SeededItem:
    """`foreign_evidence`: `evidence_quote`/`source_doc_id` taken from a
    different item whose `source_doc_id` differs from this item's."""
    donors = [
        m
        for m in pool
        if m.item_id != item.item_id
        and m.evidence_quote
        and m.source_doc_id is not None
        and m.source_doc_id != item.source_doc_id
    ]
    if not donors:
        raise ValueError(
            f"foreign_evidence: no donor from a different source_doc_id with an "
            f"evidence_quote for item {item.item_id!r}"
        )
    donor = donors[_stable_seed(seed, item.item_id, "foreign_evidence") % len(donors)]
    return _seeded_item(
        item,
        kind="foreign_evidence",
        question=item.question,
        reference_answer=item.reference_answer,
        evidence_quote=donor.evidence_quote,
        source_doc_id=donor.source_doc_id,
        detail=(
            f"evidence_quote/source_doc_id swapped from item {donor.item_id!r} "
            f"(doc {donor.source_doc_id!r})"
        ),
    )


def truncate_question(item: ItemMeta, pool: Sequence[ItemMeta], seed: int) -> SeededItem:
    """`truncated_question`: `question` cut at a deterministically chosen
    word boundary within its middle third -- never at the very first or
    last word, which would be indistinguishable from an ordinarily short
    question. `pool` is accepted for signature uniformity with the other
    three corruptors but unused: truncation needs only the item itself."""
    if not item.question:
        raise ValueError(f"truncate_question: item {item.item_id!r} has no question to truncate")
    words = item.question.split()
    if len(words) < 4:
        raise ValueError(
            f"truncate_question: item {item.item_id!r} has only {len(words)} word(s), "
            "too short to cut mid-clause"
        )
    lo = max(1, len(words) // 3)
    hi = max(lo + 1, (2 * len(words)) // 3)
    cut = lo + _stable_seed(seed, item.item_id, "truncated_question") % (hi - lo)
    truncated = " ".join(words[:cut])
    return _seeded_item(
        item,
        kind="truncated_question",
        question=truncated,
        reference_answer=item.reference_answer,
        evidence_quote=item.evidence_quote,
        source_doc_id=item.source_doc_id,
        detail=f"question truncated to {cut}/{len(words)} words",
    )


#: Auxiliary/copula verbs `_negate_text` inserts "not" after, in priority
#: order, when no standalone negation word is present to remove.
_AUX_VERBS = (
    "is", "are", "was", "were", "does", "do", "did", "can", "will", "has", "have", "should", "would",
)
#: Standalone negation words `_negate_text` removes the first occurrence
#: of, before falling back to inserting one.
_NEGATION_WORDS = (
    "not", "never", "no", "cannot", "can't", "won't", "isn't", "doesn't",
    "don't", "didn't", "wasn't", "weren't", "hasn't", "haven't",
)


def _negate_text(text: str) -> tuple[str, str]:
    """Best-effort, dependency-free polarity flip: remove the first
    standalone negation word if one is present (double-negative ->
    affirmative-shaped), else insert "not" after the first auxiliary/copula
    verb, else prefix an explicit negation. Deliberately blunt -- this
    exists to make a reference answer reliably WRONG for a known-bad seeded
    item, not to produce fluent negation. Returns `(negated_text,
    rule_applied)` so `SeededItem.detail` can record exactly which rule
    fired.

    **A removal that would leave nothing is not a negation.** `text`
    being exactly the single word `"No"`/`"no"` (common in this corpus's
    reference answers) used to fall through to returning `text`
    unchanged while still claiming to have removed a negation word -- see
    the module docstring. `"No"`/`"no"` alone is substituted with its
    affirmative counterpart instead, preserving the original word's
    capitalisation; any other bare negation word (rare, and not one this
    corpus's answers use standalone) falls through to the "insert"/
    "prefix" rules below, which always produce a genuinely different
    string."""
    words = text.split()
    lowered = [w.strip(".,!?").lower() for w in words]
    for i, w in enumerate(lowered):
        if w in _NEGATION_WORDS:
            remaining = words[:i] + words[i + 1 :]
            if remaining:
                return " ".join(remaining), f"removed negation word {words[i]!r}"
            if w == "no":
                affirmative = "Yes" if words[i][:1].isupper() else "yes"
                return affirmative, (
                    f"{words[i]!r} was the entire text -- removing it would leave nothing, "
                    f"substituted {affirmative!r}"
                )
            break
    for i, w in enumerate(lowered):
        if w in _AUX_VERBS:
            new_words = words[: i + 1] + ["not"] + words[i + 1 :]
            return " ".join(new_words), f"inserted 'not' after {words[i]!r}"
    if not text:
        return text, "empty text, nothing to negate"
    return f"It is not true that {text[0].lower()}{text[1:]}", "prefixed explicit negation"


def negate_reference(item: ItemMeta, pool: Sequence[ItemMeta], seed: int) -> SeededItem:
    """`negated_reference`: `reference_answer`'s polarity mechanically
    flipped (see `_negate_text`). `pool`/`seed` are accepted for signature
    uniformity with the other three corruptors but unused: negation is a
    pure function of the reference text, with no choice to make."""
    if not item.reference_answer:
        raise ValueError(f"negate_reference: item {item.item_id!r} has no reference_answer to negate")
    negated, rule = _negate_text(item.reference_answer)
    return _seeded_item(
        item,
        kind="negated_reference",
        question=item.question,
        reference_answer=negated,
        evidence_quote=item.evidence_quote,
        source_doc_id=item.source_doc_id,
        detail=f"reference_answer polarity flipped ({rule})",
    )


_CORRUPTORS: dict[SeedKind, Callable[[ItemMeta, Sequence[ItemMeta], int], SeededItem]] = {
    "swapped_reference": swap_reference,
    "foreign_evidence": foreign_evidence,
    "truncated_question": truncate_question,
    "negated_reference": negate_reference,
}


def seed_batch(
    meta: Sequence[ItemMeta],
    *,
    kinds: Sequence[SeedKind] = ALL_SEED_KINDS,
    n_per_kind: int,
    seed: int,
) -> list[SeededItem]:
    """Deterministically choose up to `n_per_kind` source items per kind in
    `kinds` (sampled without replacement within a kind, order derived from
    `seed`) and apply that kind's corruption to each. An item that cannot
    satisfy a given kind (its corruptor raises `ValueError`) is skipped,
    not substituted -- so a kind's actual yield may be smaller than
    `n_per_kind` if `meta` does not have enough eligible items; this is
    reported implicitly by `len()` of the result, never padded."""
    pool = list(meta)
    out: list[SeededItem] = []
    for kind in kinds:
        rng = random.Random(_stable_seed(seed, kind))
        candidates = pool[:]
        rng.shuffle(candidates)
        picked = 0
        for item in candidates:
            if picked >= n_per_kind:
                break
            try:
                out.append(_CORRUPTORS[kind](item, pool, seed))
            except ValueError:
                continue
            picked += 1
    return out
