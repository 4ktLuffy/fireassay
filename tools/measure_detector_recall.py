#!/usr/bin/env python3
"""Measure the mislabel detector's recall by seeding known-bad items.

Recall cannot be measured by sampling: at ~1% prevalence a random sample of
200 unflagged items contains about two true positives. So we manufacture
positives instead - deliberate corruptions with known ground truth - and
count how many the detector catches. This is mutation testing pointed at our
own detector rather than at a retrieval system.

The result is an UPPER BOUND. Seeded flaws may be easier to detect than
natural ones.

    .venv/bin/python tools/measure_detector_recall.py
"""
from __future__ import annotations

import json
import sqlite3
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fireassay.items.adapters.retrieval import DEFAULT_TOP_K, build_lexical_decoy_seeds, select_decoy_chunk
from fireassay.items.core import ItemMeta, ItemResponses, ItemStats, analyse
from fireassay.items.seed import ALL_SEED_KINDS, SeededItem, SeedKind, seed_batch
from fireassay.models import Question
from fireassay.system.base import System
from fireassay.system.corpus import chunk_corpus, load_corpus
from fireassay.system.panel import build_chunks_by_config, build_panel

N_PER_KIND = 100
SEED = 20260829

#: All five kinds this report covers -- `ALL_SEED_KINDS` (the four
#: `seed_batch` can produce) plus `lexical_decoy`, built separately via
#: `build_lexical_decoy_seeds` (see `items.seed`'s module docstring for
#: why `lexical_decoy` is not in `ALL_SEED_KINDS` itself).
REPORT_KINDS: tuple[SeedKind, ...] = (*ALL_SEED_KINDS, "lexical_decoy")

#: The chunking `lexical_decoy` seeding picks its decoy under -- one of
#: `build_panel`'s own six (size, overlap) combinations (its mid-range
#: one), not a seventh, arbitrary pairing. Which panel config, if any,
#: later retrieves into that decoy is a separate, independently-varying
#: question -- exactly what this script measures, not something this
#: choice presupposes.
DECOY_CHUNK_SIZE = 512
DECOY_CHUNK_OVERLAP = 32

#: The four inputs a response-matrix cell is actually computed from -- see
#: `responses_for` in `main()` below. `char_start`/`char_end` are never
#: touched by any of the four corruption kinds; only `question` (via
#: `truncated_question`) and `doc_id` (via `foreign_evidence`'s
#: `source_doc_id`) ever move.
InstrumentInputs = tuple[str, str, int, int]

#: Human-readable names for the `SeededItem` fields `unmeasurable_reason`
#: checks, in the order it reports them.
_FIELD_LABELS: dict[str, str] = {
    "question": "question",
    "reference_answer": "reference answer",
    "evidence_quote": "evidence quote",
}


def instrument_inputs(
    item_id: str,
    seed_item: SeededItem | None,
    meta_by_id: dict[str, ItemMeta],
    spans: dict[str, tuple[str, int, int]],
) -> InstrumentInputs:
    """The `(question, doc_id, char_start, char_end)` a response-matrix
    cell for `item_id` is actually scored against -- `seed_item=None` for
    the real, uncorrupted item; otherwise the same construction `main()`
    feeds to `responses_for` for a `SeededItem` (its `question`/`doc_id`
    fall back to the source item's own whenever the corruption left that
    field `None`, i.e. did not touch it). `char_start`/`char_end` always
    come from the source item's own stored span in `spans`, since no
    corruption kind changes them."""
    orig_doc_id, cs, ce = spans[item_id]
    orig_question = meta_by_id[item_id].question or ""
    if seed_item is None:
        return (orig_question, orig_doc_id, cs, ce)
    question = seed_item.question if seed_item.question is not None else orig_question
    doc_id = seed_item.source_doc_id if seed_item.source_doc_id is not None else orig_doc_id
    return (question, doc_id, cs, ce)


def reached_instrument(
    seed_item: SeededItem,
    meta_by_id: dict[str, ItemMeta],
    spans: dict[str, tuple[str, int, int]],
) -> bool:
    """Whether `seed_item` changed at least one of the four instrument
    inputs relative to the uncorrupted item it was derived from -- i.e.
    whether it could possibly move a response-matrix cell. `False` means
    its response vector is guaranteed byte-identical to the uncorrupted
    item's: the seed never reached the instrument at all."""
    return instrument_inputs(seed_item.item_id, seed_item, meta_by_id, spans) != instrument_inputs(
        seed_item.item_id, None, meta_by_id, spans
    )


def count_reached_instrument(
    seeded: Sequence[SeededItem],
    meta_by_id: dict[str, ItemMeta],
    spans: dict[str, tuple[str, int, int]],
) -> dict[SeedKind, tuple[int, int]]:
    """Per corruption kind actually present in `seeded` (never a hardcoded
    list: a future kind shows up here automatically), `(seeds that reached
    the instrument, total seeds of that kind)`. A kind whose first element
    is `0` is `UNMEASURABLE` with this instrument -- see
    `verdict_for_kind`."""
    reached: Counter[SeedKind] = Counter()
    total: Counter[SeedKind] = Counter()
    for s in seeded:
        total[s.kind] += 1
        if reached_instrument(s, meta_by_id, spans):
            reached[s.kind] += 1
    return {kind: (reached[kind], total[kind]) for kind in total}


def verdict_for_kind(reached: int, total: int) -> Literal["measurable", "UNMEASURABLE"]:
    """The verdict Change 1 derives purely from the count of seeds that
    reached the instrument -- never from a hardcoded list of kind names,
    so a future corruption kind that does move a cell (e.g. a
    lexical-decoy seed relocating the gold span to a similar document) is
    recognised as measurable without editing this function."""
    if total == 0:
        raise ValueError("verdict_for_kind: no seeds of this kind were scored")
    return "UNMEASURABLE" if reached == 0 else "measurable"


def unmeasurable_reason(kind: SeedKind, seeded: Sequence[SeededItem], meta_by_id: dict[str, ItemMeta]) -> str:
    """Which of `question`/`reference_answer`/`evidence_quote` this kind's
    seeds actually touch, in human-readable form -- derived from what the
    seeds themselves changed relative to their source items, not from a
    hardcoded per-kind list, so a new `UNMEASURABLE` kind explains itself
    correctly without this function needing an edit."""
    touched: dict[str, bool] = dict.fromkeys(_FIELD_LABELS, False)
    for s in seeded:
        if s.kind != kind:
            continue
        m = meta_by_id[s.item_id]
        if s.question is not None and s.question != m.question:
            touched["question"] = True
        if s.reference_answer is not None and s.reference_answer != m.reference_answer:
            touched["reference_answer"] = True
        if s.evidence_quote is not None and s.evidence_quote != m.evidence_quote:
            touched["evidence_quote"] = True
    labels = [_FIELD_LABELS[field] for field, hit in touched.items() if hit]
    return " and ".join(labels) if labels else "nothing scoreable"


@dataclass(frozen=True)
class LexicalDecoyReport:
    """The `lexical_decoy` kind's report-block numbers, computed as a pure
    function of `(decoy_ids, by_id)` -- kept out of `main()` so the
    denominator behaviour below is directly testable without running the
    full pipeline.

    **`n_total`/`class_counts` are the PRIMARY, headline numbers, and
    `n_total` never shrinks to exclude a seed no panel config retrieved.**
    That is a deliberate reversal of an earlier version of this report,
    which excluded a `lexical_decoy` seed with `p == 0` from the
    denominator on the theory that it "never reached the instrument" --
    the same phrase used for `swapped_reference`/`negated_reference`
    above. That reasoning does not transfer: for those two kinds, "did
    not reach" is a *structural* fact -- the corruption changed no input
    the matrix is computed from, so there is nothing to observe, and that
    is independent of any outcome. A `lexical_decoy` seed with `p == 0`
    is the opposite: the corrupted item genuinely differs from its
    source, the panel genuinely responded, and the classifier genuinely
    called it `dead_all_fail` -- an *observation* (the decoy degenerated
    into `foreign_evidence`, because nothing retrieved it), not an
    absence of one. Excluding it from the denominator conditions recall
    on the outcome -- a softer version of the trap
    `items.adapters.retrieval`'s module docstring warns
    `build_lexical_decoy_seeds` itself against, one step downstream, in
    how the number is reported rather than in which seeds are emitted.

    `n_reached`/`class_counts_reached` are diagnostics only: how many
    decoys a panel config retrieved at all, and the classification
    breakdown restricted to that subset. `main()` must label any recall
    computed from `class_counts_reached` as conditioned on that subset,
    and must never present it as the headline number."""

    n_total: int
    n_reached: int
    class_counts: dict[str, int]
    class_counts_reached: dict[str, int]


def lexical_decoy_report(decoy_ids: Sequence[str], by_id: Mapping[str, ItemStats]) -> LexicalDecoyReport:
    reached_ids = [i for i in decoy_ids if by_id[i].p > 0.0]
    return LexicalDecoyReport(
        n_total=len(decoy_ids),
        n_reached=len(reached_ids),
        class_counts=dict(Counter(by_id[i].classification for i in decoy_ids)),
        class_counts_reached=dict(Counter(by_id[i].classification for i in reached_ids)),
    )


def main() -> None:
    conn = sqlite3.connect("run/golden.db")
    rows = list(
        conn.execute(
            "select id, text, reference_answer, quote, source_doc_id, char_start, char_end "
            "from candidate where id in (select candidate_id from filter_result "
            "where stage='balance' and kept=1)"
        )
    )
    spans = {r[0]: (r[4], r[5], r[6]) for r in rows}
    meta = [
        ItemMeta(item_id=r[0], question=r[1], reference_answer=r[2],
                 evidence_quote=r[3], source_doc_id=r[4])
        for r in rows
    ]
    meta_by_id = {m.item_id: m for m in meta}
    seeded_four = seed_batch(meta, kinds=ALL_SEED_KINDS, n_per_kind=N_PER_KIND, seed=SEED)

    docs = load_corpus(Path("data/corpus.jsonl"))
    panel: list[tuple[str, System]] = build_panel(build_chunks_by_config(docs))

    # lexical_decoy: a fifth kind, built separately (see items.seed's
    # module docstring for why it is not in ALL_SEED_KINDS/seed_batch) --
    # decoys are picked against one reference chunking (DECOY_CHUNK_SIZE/
    # DECOY_CHUNK_OVERLAP), independent of which of the panel's six
    # chunkings later does or does not retrieve into the decoy.
    decoy_chunks = list(chunk_corpus(docs, size=DECOY_CHUNK_SIZE, overlap=DECOY_CHUNK_OVERLAP))
    seeded_decoy = build_lexical_decoy_seeds(meta, decoy_chunks, n=N_PER_KIND, seed=SEED, top_k=DEFAULT_TOP_K)

    print(f"{len(rows)} real items; seeded {len(seeded_four) + len(seeded_decoy)} corruptions "
          f"({N_PER_KIND} per kind x {len(REPORT_KINDS)})", flush=True)

    def responses_for(question: str, doc_id: str, cs: int, ce: int) -> tuple[bool, ...]:
        q = Question(text=question, qtype="factual", difficulty="easy", reference_answer=None,
                     provenance="synthetic", generator="probe@1")
        return tuple(
            any(ch.doc_id == doc_id and ch.char_start < ce and ch.char_end > cs
                for ch in s.answer(q).retrieved)
            for _, s in panel
        )

    # real items: reuse the matrix already computed
    real = json.loads(Path("run/full_matrix.json").read_text())
    matrix = [ItemResponses(item_id=k, responses=tuple(bool(x) for x in v)) for k, v in real.items()]

    # seeded items (the four ALL_SEED_KINDS): re-run retrieval against the
    # corrupted question/evidence -- `instrument_inputs` is the exact same
    # construction the unmeasurability check below uses, so a seed is
    # scored against precisely the inputs that check inspects, not a
    # re-derived copy.
    for i, s in enumerate(seeded_four):
        question, doc_id, cs, ce = instrument_inputs(s.item_id, s, meta_by_id, spans)
        matrix.append(ItemResponses(item_id=f"SEED::{s.kind}::{i}",
                                    responses=responses_for(question, doc_id, cs, ce)))
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(seeded_four)} seeded", flush=True)

    # lexical_decoy seeded items: re-run retrieval against the *decoy's*
    # own char range, not the source item's -- `SeededItem` (a shape
    # shared with the four kinds above) carries no char-offset fields, so
    # `select_decoy_chunk` is called again here, on the same
    # (item, decoy_chunks, top_k) `build_lexical_decoy_seeds` already used,
    # to deterministically recover the exact same decoy chunk -- including
    # its char_start/char_end -- rather than reusing the source item's own
    # (meaningless once evidence has moved to an unrelated document).
    for i, s in enumerate(seeded_decoy):
        item = meta_by_id[s.item_id]
        decoy = select_decoy_chunk(item, decoy_chunks, top_k=DEFAULT_TOP_K)
        matrix.append(ItemResponses(
            item_id=f"SEED::lexical_decoy::{i}",
            responses=responses_for(item.question or "", decoy.chunk.doc_id,
                                     decoy.chunk.char_start, decoy.chunk.char_end),
        ))
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(seeded_decoy)} lexical_decoy seeded", flush=True)

    stats, panel_stats = analyse(matrix)
    by_id = {s.item_id: s for s in stats}
    print(f"\npanel: {panel_stats.n_items} items, reliability {panel_stats.split_half_reliability:.3f} "
          f"({panel_stats.reliability_verdict})", flush=True)

    reached_counts = count_reached_instrument(seeded_four, meta_by_id, spans)

    print("\nwhat the detector saw, by corruption kind:", flush=True)
    caught_any: Counter[str] = Counter()
    indent = f"  {'':<20} "
    for kind in ALL_SEED_KINDS:
        ids = [s.item_id for s in stats if s.item_id.startswith(f"SEED::{kind}::")]
        counts = Counter(by_id[i].classification for i in ids)
        flagged = counts.get("mislabel_suspect", 0)
        anomalous = flagged + counts.get("dead_all_fail", 0)
        caught_any[kind] = anomalous
        print(f"  {kind:<20} n={len(ids):>3}  "
              f"mislabel_suspect={flagged:<4} dead_all_fail={counts.get('dead_all_fail',0):<4} "
              f"live={counts.get('live',0):<4} dead_all_pass={counts.get('dead_all_pass',0):<4}", flush=True)
        reached, total = reached_counts[kind]
        print(f"  {'':<20} reached the instrument: {reached}/{total}", flush=True)
        if verdict_for_kind(reached, total) == "UNMEASURABLE":
            reason = unmeasurable_reason(kind, seeded_four, meta_by_id)
            message = (
                f"recall = UNMEASURABLE -- this corruption changes only the {reason}, and a "
                "matrix cell here is retrieval overlap with the gold span, so no cell can "
                "move. Not a measurement of the detector. Fix the seed, not the number: a "
                "corruption that relocates the gold span to a lexically SIMILAR document "
                "produces the ranking inversion the detector looks for."
            )
            print(textwrap.fill(message, width=78, initial_indent=indent, subsequent_indent=indent),
                  flush=True)
        else:
            print(f"  {'':<20} recall(mislabel_suspect)={flagged/len(ids):.0%}   "
                  f"recall(any anomaly)={anomalous/len(ids):.0%}", flush=True)
            if kind == "foreign_evidence":
                caveat = (
                    "note: this corruption relocates the gold span to a different document, "
                    "so by construction no config can retrieve it -- the 100% above is close "
                    "to tautological, not a measure of detector sensitivity."
                )
                print(textwrap.fill(caveat, width=78, initial_indent=indent, subsequent_indent=indent),
                      flush=True)

    # lexical_decoy: reported alongside the four kinds above. Its
    # denominator is NOT "reached the instrument" (see `LexicalDecoyReport`'s
    # docstring for why that field-diff concept, correct for the four
    # kinds above, does not transfer here) -- the primary recall below is
    # over every seed `build_lexical_decoy_seeds` emitted, full stop.
    # "Retrieved by >=1 config" is reported only as a diagnostic.
    decoy_ids = [f"SEED::lexical_decoy::{i}" for i in range(len(seeded_decoy))]
    decoy_report = lexical_decoy_report(decoy_ids, by_id)
    decoy_flagged = decoy_report.class_counts.get("mislabel_suspect", 0)
    decoy_anomalous = decoy_flagged + decoy_report.class_counts.get("dead_all_fail", 0)
    caught_any["lexical_decoy"] = decoy_anomalous
    print(f"  {'lexical_decoy':<20} n={decoy_report.n_total:>3}  "
          f"mislabel_suspect={decoy_flagged:<4} "
          f"dead_all_fail={decoy_report.class_counts.get('dead_all_fail', 0):<4} "
          f"live={decoy_report.class_counts.get('live', 0):<4} "
          f"dead_all_pass={decoy_report.class_counts.get('dead_all_pass', 0):<4}",
          flush=True)
    print(f"  {'':<20} decoys retrieved by >=1 config: {decoy_report.n_reached}/{decoy_report.n_total}"
          " (diagnostic, not the recall denominator)", flush=True)
    not_reached = decoy_report.n_total - decoy_report.n_reached
    if not_reached:
        note = (
            f"note: {not_reached}/{decoy_report.n_total} lexical_decoy seed(s) were never "
            "retrieved by any panel config -- their decoy degenerated into foreign_evidence "
            "(nothing could possibly find it). That is a real result about the seed, kept in "
            "the recall denominator below, not an absence of one to exclude."
        )
        print(textwrap.fill(note, width=78, initial_indent=indent, subsequent_indent=indent), flush=True)
    if decoy_report.n_total == 0:
        message = "recall = UNMEASURABLE -- no lexical_decoy seed was built at all."
        print(textwrap.fill(message, width=78, initial_indent=indent, subsequent_indent=indent), flush=True)
    else:
        print(f"  {'':<20} recall(mislabel_suspect)={decoy_flagged / decoy_report.n_total:.0%}   "
              f"recall(any anomaly)={decoy_anomalous / decoy_report.n_total:.0%}  "
              "[PRIMARY -- over all emitted seeds]", flush=True)
        if decoy_report.n_reached:
            reached_flagged = decoy_report.class_counts_reached.get("mislabel_suspect", 0)
            reached_anomalous = reached_flagged + decoy_report.class_counts_reached.get("dead_all_fail", 0)
            print(f"  {'':<20} recall(mislabel_suspect)={reached_flagged / decoy_report.n_reached:.0%}   "
                  f"recall(any anomaly)={reached_anomalous / decoy_report.n_reached:.0%}  "
                  "[conditioned on decoy retrieved by >=1 config -- NOT the headline number]",
                  flush=True)

    print("\nNOTE: seeded recall is an UPPER BOUND -- synthetic flaws may be easier to spot.",
          flush=True)


if __name__ == "__main__":
    main()
