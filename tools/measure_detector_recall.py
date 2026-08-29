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
from collections import Counter
from pathlib import Path

from fireassay.items.core import ItemMeta, ItemResponses, analyse
from fireassay.items.seed import ALL_SEED_KINDS, seed_batch
from fireassay.models import Question
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import chunk_corpus, load_corpus

N_PER_KIND = 100
SEED = 20260829


def build_panel(docs: list[object]) -> list[BM25System]:
    panel: list[BM25System] = []
    for size, overlap in [(256, 32), (256, 128), (512, 32), (512, 128), (1024, 32), (1024, 128)]:
        chunks = list(chunk_corpus(docs, size=size, overlap=overlap))  # type: ignore[arg-type]
        for k in (3, 5, 10):
            panel.append(BM25System(chunks, top_k=k))
    return panel


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
    seeded = seed_batch(meta, kinds=ALL_SEED_KINDS, n_per_kind=N_PER_KIND, seed=SEED)
    print(f"{len(rows)} real items; seeded {len(seeded)} corruptions "
          f"({N_PER_KIND} per kind x {len(ALL_SEED_KINDS)})", flush=True)

    docs = load_corpus(Path("data/corpus.jsonl"))
    panel = build_panel(docs)  # type: ignore[arg-type]

    def responses_for(question: str, doc_id: str, cs: int, ce: int) -> tuple[bool, ...]:
        q = Question(text=question, qtype="factual", difficulty="easy", reference_answer=None,
                     provenance="synthetic", generator="probe@1")
        return tuple(
            any(ch.doc_id == doc_id and ch.char_start < ce and ch.char_end > cs
                for ch in s.answer(q).retrieved)
            for s in panel
        )

    # real items: reuse the matrix already computed
    real = json.loads(Path("run/full_matrix.json").read_text())
    matrix = [ItemResponses(item_id=k, responses=tuple(bool(x) for x in v)) for k, v in real.items()]

    # seeded items: re-run retrieval against the corrupted question/evidence
    for i, s in enumerate(seeded):
        doc_id, cs, ce = spans[s.item_id]
        if s.source_doc_id is not None:        # foreign_evidence moved the span
            doc_id = s.source_doc_id
        question = s.question if s.question is not None else next(
            m.question for m in meta if m.item_id == s.item_id) or ""
        matrix.append(ItemResponses(item_id=f"SEED::{s.kind}::{i}",
                                    responses=responses_for(question, doc_id, cs, ce)))
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(seeded)} seeded", flush=True)

    stats, panel_stats = analyse(matrix)
    by_id = {s.item_id: s for s in stats}
    print(f"\npanel: {panel_stats.n_items} items, reliability {panel_stats.split_half_reliability:.3f} "
          f"({panel_stats.reliability_verdict})", flush=True)

    print("\nwhat the detector saw, by corruption kind:", flush=True)
    caught_any: Counter[str] = Counter()
    for kind in ALL_SEED_KINDS:
        ids = [s.item_id for s in stats if s.item_id.startswith(f"SEED::{kind}::")]
        counts = Counter(by_id[i].classification for i in ids)
        flagged = counts.get("mislabel_suspect", 0)
        anomalous = flagged + counts.get("dead_all_fail", 0)
        caught_any[kind] = anomalous
        print(f"  {kind:<20} n={len(ids):>3}  "
              f"mislabel_suspect={flagged:<4} dead_all_fail={counts.get('dead_all_fail',0):<4} "
              f"live={counts.get('live',0):<4} dead_all_pass={counts.get('dead_all_pass',0):<4}", flush=True)
        print(f"  {'':<20} recall(mislabel_suspect)={flagged/len(ids):.0%}   "
              f"recall(any anomaly)={anomalous/len(ids):.0%}", flush=True)
    print("\nNOTE: seeded recall is an UPPER BOUND -- synthetic flaws may be easier to spot.",
          flush=True)


if __name__ == "__main__":
    main()
