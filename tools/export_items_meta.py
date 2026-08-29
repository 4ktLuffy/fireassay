#!/usr/bin/env python3
"""Export `ItemMeta` JSONL from `run/golden.db` for the kept candidates.

This is the fireassay-side bridge to `items.adapters.tabular`'s standalone
contract, not part of `items` itself: `items/` must not know where its
input came from, and this script is exactly the kind of upstream glue that
stays out of it (see `items.core`'s module docstring). Point `fireassay
items analyse --matrix ... --meta` or `fireassay items review build
--matrix ... --meta` at this script's output to run the standalone path
against our own run.

    .venv/bin/python tools/export_items_meta.py --db run/golden.db --out run/items_meta.jsonl
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from fireassay.items.core import ItemMeta

_QUERY = (
    "select id, text, reference_answer, quote, source_doc_id from candidate "
    "where id in (select candidate_id from filter_result "
    "where stage='balance' and kept=1)"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("run/golden.db"))
    ap.add_argument("--out", type=Path, default=Path("run/items_meta.jsonl"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = list(conn.execute(_QUERY))
    conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item_id, question, reference_answer, evidence_quote, source_doc_id in rows:
            meta = ItemMeta(
                item_id=item_id,
                question=question,
                reference_answer=reference_answer,
                evidence_quote=evidence_quote,
                source_doc_id=source_doc_id,
            )
            f.write(meta.model_dump_json())
            f.write("\n")

    print(f"wrote {len(rows)} item(s) to {args.out}")


if __name__ == "__main__":
    main()
