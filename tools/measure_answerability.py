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

Two judge backends, `--judge {ollama,codex}` (default `ollama`, so every
command line that predates this option keeps behaving exactly as before):

- **ollama** (default): the local `granite4:7b-a1b-h` judge, unchanged.
  Must never resolve to the generator model (`qwen2.5:7b`, which produced
  these items) -- defect 15 measured self-preference bias in judges, and
  `_check_not_self_preferring` fails loudly, before any network call, if
  the *resolved* judge model (today's default, or a `--model` override)
  ever collides with it.
- **codex**: OpenAI's `codex` CLI (`codex exec`), a genuinely independent
  judge -- a different provider from `qwen2.5:7b`, so defect 15's
  self-preference concern does not apply here. See `_codex_ask`'s
  docstring for the three load-bearing constraints on invoking it
  correctly: `/dev/null` stdin, a per-call timeout, and reading the
  completion from its `-o` file rather than stdout.

`--model`/`--effort` default to today's Ollama judge tag and `"low"`
respectively. `--effort` only has meaning for `codex` (Ollama has no
reasoning-effort concept) and is recorded as `null` for an `ollama` run.
Both are written into every output record, so a verdict is always
attributable to the judge that actually produced it -- previously the
output carried no such field at all.

`temperature=0.0` is requested for the Ollama backend, but **temperature 0
is not determinism** (defect 14: batch-size dependence of the backend's
reduction kernels can still perturb output). What actually makes a re-run
reproducible at zero additional cost is the cache, not the temperature:
both backends check their own cache before ever calling a model, so
re-running this script after a partial or complete prior run replays every
already-answered item for free and only pays for the rest.

Each backend has **its own cache directory** -- ollama at `.cache/judge/`,
codex at `.cache/judge-codex/` -- entirely separate from each other and
from `generate_json`'s generation checkpoint at `.cache/llm/`. The cache
is keyed on `(model, prompt)` with no notion of which backend produced the
entry, so sharing a file between two judges could silently replay one
judge's verdict as if it were the other's; keeping every judge on its own
cache file makes that collision structurally impossible rather than
merely unlikely.

Per item: `ask` failures (network errors, an unreachable Ollama server, a
`codex exec` timeout or non-zero exit -- anything a backend can raise) are
recorded and skipped -- never fatal to the batch (defect 39: raise at the
call, record and continue at the batch). Progress prints and flushes after
every item, not batched, so a long run's state is visible and killing it
mid-run loses nothing the cache has not already checkpointed.

`--limit N` and/or `--items <file>` scope a run to a handful of items
without editing code or `run/calibration.jsonl` -- useful for a quick,
cheap smoke test of a new judge (codex runs ~8s/item) before committing to
the full batch. `--items` (one item id per line) replaces `--calibration`
as the source of the id list to score; `--limit` truncates whichever list
was produced to its first `N` entries.

The output is three verdicts (`supported` / `not_supported` / `unclear`)
per item, plus the quoted span, the raw completion, and now the judge/
model/effort that produced it -- **no threshold, no score, no composite**
is computed here. Comparing the verdicts against `run/calibration.jsonl`'s
human labels (via `items.calibration.evaluate_detector`) is a separate,
deliberate step, not folded into this script.

    .venv/bin/python tools/measure_answerability.py
    .venv/bin/python tools/measure_answerability.py --judge codex --limit 5
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from fireassay.items.answerability import Ask, build_prompt, parse_verdict
from fireassay.items.calibration import load_calibration_jsonl
from fireassay.items.core import ItemMeta
from fireassay.llm.cache import ResponseCache
from fireassay.llm.ollama import ModelRef, OllamaClient

#: The Ollama judge's default tag -- pinned by digest via `client.model_ref`,
#: never by this tag alone, exactly as before. `--model` can override it,
#: which is why the self-preference check runs on the *resolved* value,
#: post-argument-parsing, rather than on this constant alone.
_DEFAULT_OLLAMA_MODEL = "granite4:7b-a1b-h"

#: codex's default model / reasoning effort -- the verified invocation is
#: `codex exec ... -m gpt-5.4 -c model_reasoning_effort='"low"'`.
_DEFAULT_CODEX_MODEL = "gpt-5.4"
_DEFAULT_CODEX_EFFORT = "low"
_DEFAULT_CODEX_TIMEOUT = 120.0

#: The model that generated these items (`tools/closedbook_sample.py` uses
#: the same tag). Defect 15 measured self-preference bias when a judge
#: scores its own generator's output -- see `_check_not_self_preferring`.
_GENERATOR_MODEL_NAME = "qwen2.5:7b"

_DEFAULT_CALIBRATION = Path("run/calibration.jsonl")
_DEFAULT_DB = Path("run/golden.db")
_DEFAULT_OUT = Path("run/answerability.jsonl")

#: One cache file per judge -- see the module docstring's caching section
#: for why the two must never share a file.
_DEFAULT_CACHE_OLLAMA = Path(".cache/judge/responses.jsonl")
_DEFAULT_CACHE_CODEX = Path(".cache/judge-codex/responses.jsonl")

_META_QUERY = "select id, text, reference_answer, quote, source_doc_id from candidate where id in ({})"


def _check_not_self_preferring(judge: str, model: str) -> None:
    """Fail loudly, before any network call, if the Ollama judge would
    resolve to the same model that generated these items -- checked on
    the resolved model (today's default, or a `--model` override), not
    just on `_DEFAULT_OLLAMA_MODEL`, now that the judge model is a runtime
    argument rather than a fixed constant."""
    if judge == "ollama" and model == _GENERATOR_MODEL_NAME:
        raise ValueError(
            f"measure_answerability: judge model {model!r} must not equal the generator "
            f"model {_GENERATOR_MODEL_NAME!r} -- defect 15 measured self-preference bias in "
            "judges; scoring the generator's own items with itself as judge would reproduce "
            "exactly that bias, not measure answerability"
        )


class CodexTimeoutError(Exception):
    """A single `codex exec` call exceeded its timeout (see `_codex_ask`).
    Caught by `main`'s existing defect-39 record-and-continue loop exactly
    like any other `ask` failure."""


class CodexInvocationError(Exception):
    """`codex exec` exited non-zero, or produced no (or an empty) `-o`
    output file. Never treated as an empty completion -- `items.
    answerability`'s empty-completion handling (`verdict="unclear"`) is
    for a real, empty *model* response, not a process that failed to
    run at all."""


def _codex_ask(model: str, effort: str, *, timeout: float = _DEFAULT_CODEX_TIMEOUT) -> Ask:
    """Build an `Ask` (`Callable[[str], str]`, `items.answerability`'s
    injected-model contract -- a new judge is a new adapter, never an edit
    to that module) that runs `prompt` through `codex exec` and returns
    the completion written to a fresh `-o` file, one per call.

    The exact invocation (verified working):

        codex exec --skip-git-repo-check -m {model} \\
          -c model_reasoning_effort='"{effort}"' \\
          -o <tmpfile> "<prompt>" </dev/null

    Three constraints here are load-bearing, not stylistic:

    1. **stdin is `subprocess.DEVNULL`, always, not configurable.** Without
       it, `codex exec` prints "Reading additional input from stdin..."
       and blocks forever whenever stdin is not a TTY -- which is always
       true for this subprocess call. There is no recovering from that
       once a multi-hour batch is hundreds of items in; it just silently
       wedges the run.
    2. **`timeout` bounds every call** (`--timeout` at the CLI, default
       120s). A single hung call must never stall the whole batch --
       expiry raises `CodexTimeoutError`, which `main`'s existing
       defect-39 loop already catches and records, continuing to the next
       item rather than aborting.
    3. **The completion comes from the `-o` file, never stdout.** `codex
       exec`'s stdout carries a header (`model:`, `reasoning effort:`,
       `session id:`), the echoed prompt, and a token count -- none of
       which is the free-text completion `items.answerability.
       parse_verdict` expects to parse.
    """

    def ask(prompt: str) -> str:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "codex_output.txt"
            cmd = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "-m",
                model,
                "-c",
                f'model_reasoning_effort="{effort}"',
                "-o",
                str(out_path),
                prompt,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,  # required, not optional -- see docstring point 1
                    capture_output=True,
                    timeout=timeout,
                    encoding="utf-8",
                )
            except subprocess.TimeoutExpired as exc:
                raise CodexTimeoutError(
                    f"codex exec exceeded its {timeout}s timeout (model={model!r}, effort={effort!r})"
                ) from exc

            if result.returncode != 0:
                raise CodexInvocationError(
                    f"codex exec exited {result.returncode} (model={model!r}, effort={effort!r}): "
                    f"{result.stderr.strip()[:500]}"
                )
            if not out_path.exists():
                raise CodexInvocationError(
                    f"codex exec produced no -o output file (model={model!r}, effort={effort!r})"
                )
            # Point 3: the completion lives in the -o file, never stdout.
            completion = out_path.read_text(encoding="utf-8")
            if not completion.strip():
                raise CodexInvocationError(
                    f"codex exec's -o file was empty (model={model!r}, effort={effort!r})"
                )
            return completion

    return ask


def _cached_ask(cache: ResponseCache, cache_model: ModelRef, ask: Ask) -> Ask:
    """Wrap `ask` with a get-then-call-then-put cache check -- the same
    sequence `OllamaClient.generate_text` already does internally for the
    Ollama backend. `codex exec` has no cache of its own, so `main` wraps
    `_codex_ask`'s result with this; the Ollama backend does not need it,
    since `client.generate_text` is already cache-aware."""

    def wrapped(prompt: str) -> str:
        cached = cache.get(cache_model, prompt)
        if cached is not None:
            return cached
        response = ask(prompt)
        cache.put(cache_model, prompt, response)
        return response

    return wrapped


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


def _load_item_ids_from_file(path: Path) -> list[str]:
    """One item id per line; blank lines skipped -- mirrors `system.corpus
    .load_corpus`'s handling of blank lines in a plain-text input file."""
    ids: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                ids.append(line)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path, default=_DEFAULT_CALIBRATION)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="default: .cache/judge/responses.jsonl (ollama) or "
        ".cache/judge-codex/responses.jsonl (codex)",
    )
    ap.add_argument("--judge", choices=["ollama", "codex"], default="ollama")
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"default: {_DEFAULT_OLLAMA_MODEL!r} for --judge ollama, "
        f"{_DEFAULT_CODEX_MODEL!r} for --judge codex",
    )
    ap.add_argument(
        "--effort",
        type=str,
        default=_DEFAULT_CODEX_EFFORT,
        help="codex reasoning effort (--judge codex only); recorded as null for --judge ollama",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_CODEX_TIMEOUT,
        help="per-call timeout in seconds (--judge codex only)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="score only the first N items of the id list"
    )
    ap.add_argument(
        "--items",
        type=Path,
        default=None,
        help="file of item ids, one per line -- replaces --calibration as the source of the id list",
    )
    args = ap.parse_args()

    if args.items is not None:
        item_ids = sorted(set(_load_item_ids_from_file(args.items)))
        print(f"{len(item_ids)} distinct item(s) named in {args.items}", flush=True)
    else:
        labels = load_calibration_jsonl(args.calibration)
        item_ids = sorted({label.item_id for label in labels})
        print(f"{len(item_ids)} distinct item(s) named in {args.calibration}", flush=True)

    if args.limit is not None:
        item_ids = item_ids[: args.limit]
        print(f"--limit {args.limit}: scoring {len(item_ids)} item(s)", flush=True)

    meta_by_id = _load_meta(args.db, item_ids)
    missing_ids = [i for i in item_ids if i not in meta_by_id]
    if missing_ids:
        print(
            f"WARNING: {len(missing_ids)}/{len(item_ids)} item(s) named in the calibration file "
            f"have no row in {args.db}'s candidate table -- recorded and skipped: {missing_ids[:10]}"
            + (" ..." if len(missing_ids) > 10 else ""),
            flush=True,
        )

    model = args.model or (_DEFAULT_OLLAMA_MODEL if args.judge == "ollama" else _DEFAULT_CODEX_MODEL)
    effort_recorded: str | None = args.effort if args.judge == "codex" else None
    _check_not_self_preferring(args.judge, model)

    ask: Ask
    if args.judge == "ollama":
        cache_path = args.cache or _DEFAULT_CACHE_OLLAMA
        client = OllamaClient(cache=ResponseCache(cache_path))
        model_ref = client.model_ref(model)
        print(
            f"judge [ollama] {model_ref.name}@{model_ref.digest[:12]} "
            f"quant={model_ref.quantization_level}",
            flush=True,
        )

        def ollama_ask(prompt: str) -> str:
            return client.generate_text(model_ref, prompt, temperature=0.0)

        ask = ollama_ask
    else:
        cache_path = args.cache or _DEFAULT_CACHE_CODEX
        # Not a real content digest -- there is no Ollama-style /api/tags digest
        # for a CLI-backed model. This is purely a cache key, and is safe as one
        # because this cache file is never shared with another judge (see the
        # module docstring's caching section).
        cache_model = ModelRef(name=model, digest=f"codex:{model}:{args.effort}")
        ask = _cached_ask(
            ResponseCache(cache_path), cache_model, _codex_ask(model, args.effort, timeout=args.timeout)
        )
        print(f"judge [codex] {model} effort={args.effort!r} timeout={args.timeout}s", flush=True)

    items = [meta_by_id[i] for i in item_ids if i in meta_by_id]
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for i, item in enumerate(items, start=1):
        try:
            raw = ask(build_prompt(item))
        except Exception as exc:  # noqa: BLE001 -- defect 39: record and continue, never abort
            errors.append({"item_id": item.item_id, "error": type(exc).__name__, "message": str(exc)})
            print(f"  [{i}/{len(items)}] {item.item_id}: ERROR ({type(exc).__name__})", flush=True)
            continue
        verdict = parse_verdict(item.item_id, raw)
        record = {**verdict.model_dump(), "judge": args.judge, "model": model, "effort": effort_recorded}
        records.append(record)
        print(f"  [{i}/{len(items)}] {item.item_id}: {verdict.verdict}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    print(f"\nwrote {len(records)} verdict(s) to {args.out}", flush=True)
    if missing_ids:
        print(f"{len(missing_ids)} item(s) skipped -- no ItemMeta in {args.db}", flush=True)
    if errors:
        counts = Counter(e["error"] for e in errors)
        breakdown = ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
        print(f"{len(errors)} item(s) skipped -- ask() failed ({breakdown}), see above", flush=True)
        errors_path = args.out.with_suffix(".errors.json")
        errors_path.write_text(json.dumps(errors, indent=1))
        print(f"error detail written to {errors_path}", flush=True)


if __name__ == "__main__":
    main()
