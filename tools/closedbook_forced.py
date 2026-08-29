#!/usr/bin/env python3
"""Closed-book screen, FORCED-GUESS protocol: is the answer in the model's weights?

The strongest single signal for whether an item tests a RAG system, per the
literature: no-context accuracy on *original* RAG benchmarks runs 31-82%,
and leakage filtering drops it to 1.4-17%. A question the model answers from
memory is not testing your retrieval stack.

Two protocols answer different questions, and the difference is not cosmetic.
An earlier run permitted abstention ("if you do not know, say I do not know")
and the model abstained 82.7% of the time, yielding 0.7% matches -- which
measures how often an abstention-instructed model happens to match, NOT
whether the question is answerable from memory. Published figures (31-82% on
original RAG benchmarks) come from forced-answer protocols, so that number was
not comparable to anything.

This script forces a best guess. No escape hatch. That is the contamination
question, and it is the one the literature answers.

Samples rather than sweeping all 2,364: 150 items gives roughly +/-8% at 95%
confidence, ample to decide whether a full run is warranted.

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
CACHE = ".cache/closedbook_forced/responses.jsonl"
OUT = Path("run/closedbook_forced.json")
SAMPLE = 150
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
                "You MUST give your best specific answer even if unsure. Guessing is required.\n"
                "Do not say you do not know, and do not ask for more information.\n\n"
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
