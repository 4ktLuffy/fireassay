#!/usr/bin/env python3
"""Score eval-of-evals signal 1 -- gold answerability (handbook §6) -- over
the items named in a calibration file, writing raw `AnswerabilityVerdict`s
as JSONL. This is the ONE scoring run the protocol section of
`SPEC-answerability.md` describes: the prompt (`items.answerability.build_prompt`)
was written and committed before any labelled item was looked at, and is
scored exactly once against the 176 human labels in `run/calibration.jsonl`.
**If the resulting precision disappoints, the honest next step is a fresh
labelled sample -- not editing the prompt and re-running this script
against the same 176 labels.** That would burn the only external referent
this project has and stop the number being a measurement at all.

Judge model is `granite4:7b-a1b-h`, resolved through `client.model_ref` so
it is pinned by digest, never a re-pullable tag. It must never be the
generator model (`qwen2.5:7b`, which produced these items) -- defect 15
measured self-preference bias in judges, and the module-level check below
fails loudly, before any network call, if the two constants are ever made
to collide.

`temperature=0.0` is requested, but **temperature 0 is not determinism**
(defect 14: batch-size dependence of Ollama's/the backend's reduction
kernels can still perturb output). What actually makes a re-run
reproducible at zero additional cost is the cache, not the temperature:
`OllamaClient.generate_text` checks its own cache before ever calling the
model, so re-running this script after a partial or complete prior run
replays every already-answered item for free and only pays for the rest.

That cache is **its own `OllamaClient`/`ResponseCache` pair**, rooted at
`.cache/judge/`, entirely separate from `generate_json`'s generation
checkpoint at `.cache/llm/`. `_CacheProtocol` keys purely on
`(model.digest, prompt)`, with no notion of "text" vs "JSON" mode -- reusing
the generation cache here could replay a JSON-shaped generation response
for a free-text judge prompt that happens to coincide, or vice versa.
Keeping the two caches on separate files makes that collision structurally
impossible rather than merely unlikely.

Per item: `ask` failures (network errors, an unreachable server, anything
`OllamaClient.generate_text` raises) are recorded and skipped -- never
fatal to the batch (defect 39: raise at the call, record and continue at
the batch). Progress prints and flushes after every item, not batched,
so a long run's state is visible and killing it mid-run loses nothing the
cache has not already checkpointed.

The output is three verdicts (`supported` / `not_supported` / `unclear`)
per item, plus the quoted span and the raw completion -- **no threshold, no
score, no composite** is computed here. Comparing the verdicts against
`run/calibration.jsonl`'s human labels (via `items.calibration.evaluate_detector`)
is a separate, deliberate step, not folded into this script.

    .venv/bin/python tools/measure_answerability.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from fireassay.items.answerability import AnswerabilityVerdict, build_prompt, parse_verdict
from fireassay.items.calibration import load_calibration_jsonl
from fireassay.items.core import ItemMeta
from fireassay.llm.cache import ResponseCache
from fireassay.llm.ollama import OllamaClient

#: The judge -- pinned by digest via `client.model_ref`, never by this tag
#: alone. Must never equal `_GENERATOR_MODEL_NAME`; see the module-level
#: check below.
_JUDGE_MODEL_NAME = "granite4:7b-a1b-h"

#: The model that generated these items (`tools/closedbook_sample.py` uses
#: the same tag). Defect 15 measured self-preference bias when a judge
#: scores its own generator's output -- the judge must be a different model.
_GENERATOR_MODEL_NAME = "qwen2.5:7b"

if _JUDGE_MODEL_NAME == _GENERATOR_MODEL_NAME:
    raise ValueError(
        f"measure_answerability: judge model {_JUDGE_MODEL_NAME!r} must not equal the "
        f"generator model {_GENERATOR_MODEL_NAME!r} -- defect 15 measured self-preference "
        "bias in judges; scoring the generator's own items with itself as judge would "
        "reproduce exactly that bias, not measure answerability"
    )

_DEFAULT_CALIBRATION = Path("run/calibration.jsonl")
_DEFAULT_DB = Path("run/golden.db")
_DEFAULT_OUT = Path("run/answerability.jsonl")

#: A separate cache file from `generate_json`'s `.cache/llm/` checkpoint --
#: see the module docstring's caching section for why the two must never
#: share a file.
_DEFAULT_CACHE = Path(".cache/judge/responses.jsonl")

_META_QUERY = "select id, text, reference_answer, quote, source_doc_id from candidate where id in ({})"


def _load_meta(db_path: Path, item_ids: list[str]) -> dict[str, ItemMeta]:
    """`ItemMeta` for every id in `item_ids` that has a row in `db_path`'s
    `candidate` table -- an id with no matching row is simply absent from
    the returned mapping (the caller records and skips it, per the
    module docstring's defect-39 discipline, rather than this function
    raising for a single missing id)."""
    if not item_ids:
        return {}
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in item_ids)
        rows = list(conn.execute(_META_QUERY.format(placeholders), item_ids))
    finally:
        conn.close()
    return {
        row[0]: ItemMeta(item_id=row[0], question=row[1], reference_answer=row[2],
                          evidence_quote=row[3], source_doc_id=row[4])
        for row in rows
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path, default=_DEFAULT_CALIBRATION)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    args = ap.parse_args()

    labels = load_calibration_jsonl(args.calibration)
    item_ids = sorted({label.item_id for label in labels})
    print(f"{len(item_ids)} distinct item(s) named in {args.calibration}", flush=True)

    meta_by_id = _load_meta(args.db, item_ids)
    missing_ids = [i for i in item_ids if i not in meta_by_id]
    if missing_ids:
        print(
            f"WARNING: {len(missing_ids)}/{len(item_ids)} item(s) named in the calibration file "
            f"have no row in {args.db}'s candidate table -- recorded and skipped: {missing_ids[:10]}"
            + (" ..." if len(missing_ids) > 10 else ""),
            flush=True,
        )

    client = OllamaClient(cache=ResponseCache(args.cache))
    model = client.model_ref(_JUDGE_MODEL_NAME)
    print(f"judge {model.name}@{model.digest[:12]} quant={model.quantization_level}", flush=True)

    items = [meta_by_id[i] for i in item_ids if i in meta_by_id]
    verdicts: list[AnswerabilityVerdict] = []
    errors: list[dict[str, str]] = []

    for i, item in enumerate(items, start=1):
        try:
            raw = client.generate_text(model, build_prompt(item), temperature=0.0)
        except Exception as exc:  # noqa: BLE001 -- defect 39: record and continue, never abort
            errors.append({"item_id": item.item_id, "error": type(exc).__name__, "message": str(exc)})
            print(f"  [{i}/{len(items)}] {item.item_id}: ERROR ({type(exc).__name__})", flush=True)
            continue
        verdict = parse_verdict(item.item_id, raw)
        verdicts.append(verdict)
        print(f"  [{i}/{len(items)}] {item.item_id}: {verdict.verdict}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for verdict in verdicts:
            f.write(verdict.model_dump_json())
            f.write("\n")

    print(f"\nwrote {len(verdicts)} verdict(s) to {args.out}", flush=True)
    if missing_ids:
        print(f"{len(missing_ids)} item(s) skipped -- no ItemMeta in {args.db}", flush=True)
    if errors:
        print(f"{len(errors)} item(s) skipped -- ask() failed, see above", flush=True)
        errors_path = args.out.with_suffix(".errors.json")
        errors_path.write_text(json.dumps(errors, indent=1))
        print(f"error detail written to {errors_path}", flush=True)


if __name__ == "__main__":
    main()
