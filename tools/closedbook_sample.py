#!/usr/bin/env python3
"""Closed-book screen: does a question need retrieval at all?

The strongest single signal for whether an item tests a RAG system, per the
literature: no-context accuracy on *original* RAG benchmarks runs 31-82%,
and leakage filtering drops it to 1.4-17%. A question the model answers from
memory is not testing your retrieval stack.

Samples rather than sweeping the whole set: 300 items gives roughly +/-5.7%
at 95% confidence for a proportion near 0.5, which is ample to decide whether
there is a problem, at a quarter of the cost of all 2,364.

Resumable. The response cache is the checkpoint - kill this any time and
re-run the identical command; completed calls replay for free.

    .venv/bin/python tools/closedbook_sample.py
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path

from pydantic import BaseModel

from fireassay.llm.cache import ResponseCache
from fireassay.llm.ollama import OllamaClient

DB = "run/golden.db"
CACHE = ".cache/closedbook/responses.jsonl"
OUT = Path("run/closedbook.json")
SAMPLE = 300
SEED = 20260829


class Answer(BaseModel):
    answer: str


class Verdict(BaseModel):
    matches: bool
    reason: str


def main() -> None:
    Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(cache=ResponseCache(CACHE))
    model = client.model_ref("qwen2.5:7b")
    print(f"model {model.name}@{model.digest[:12]} quant={model.quantization_level}", flush=True)

    conn = sqlite3.connect(DB)
    rows = list(
        conn.execute(
            "select id, text, reference_answer from candidate where id in "
            "(select candidate_id from filter_result where stage='balance' and kept=1)"
        )
    )
    random.Random(SEED).shuffle(rows)
    rows = rows[:SAMPLE]
    print(f"sampling {len(rows)} of the kept candidates", flush=True)

    out: list[dict[str, object]] = []
    t0 = time.time()
    for i, (cid, question, reference) in enumerate(rows):
        try:
            closed = client.generate_json(
                model,
                "Answer the question from your own knowledge. You have NO reference documents.\n"
                "If you do not know, answer exactly: I do not know.\n\n"
                f"Question: {question}\n\n"
                'Respond as JSON: {"answer": "..."}',
                Answer,
            )
            verdict = client.generate_json(
                model,
                "Does the CANDIDATE answer convey the same factual content as the REFERENCE?\n"
                "Ignore wording and level of detail. True only if the substance matches.\n\n"
                f"REFERENCE: {reference}\n\nCANDIDATE: {closed.answer}\n\n"
                'Respond as JSON: {"matches": true|false, "reason": "..."}',
                Verdict,
            )
            out.append(
                {
                    "id": cid,
                    "question": question,
                    "reference": reference,
                    "closed_book": closed.answer,
                    "matches": verdict.matches,
                    "reason": verdict.reason,
                }
            )
        except Exception as exc:  # noqa: BLE001 - record and continue, never abort the batch
            out.append({"id": cid, "question": question, "error": type(exc).__name__})
        if (i + 1) % 50 == 0:
            hits = sum(1 for o in out if o.get("matches"))
            print(f"  {i + 1}/{len(rows)}  answered without retrieval: {hits}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    scored = [o for o in out if "matches" in o]
    hits = sum(1 for o in scored if o["matches"])
    print(f"\nRESULT: {hits}/{len(scored)} = {hits / len(scored):.1%} answerable with NO retrieval",
          flush=True)
    print(f"errors: {len(out) - len(scored)}", flush=True)
    print("literature: 31-82% on original RAG benchmarks; 1.4-17% after leakage filtering",
          flush=True)


if __name__ == "__main__":
    main()
