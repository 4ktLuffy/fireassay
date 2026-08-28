"""`ResponseCache` — a content-addressed, on-disk cache of raw LLM
responses (M3-SPEC.md §1).

Keyed on `(model.digest, prompt)`, never on `model.name`: a tag can be
re-pulled and silently point at different weights (M3-SPEC.md §1's "pin by
digest, never by tag" rule), so caching on the tag would let a re-pulled
model's cache entries be replayed as if they came from the model that is
actually pinned. Content addressing this way also makes regeneration free
(re-running `generate` against an unchanged corpus/model replays the cache
instead of re-querying), makes runs reproducible, and — the point that
matters most for this milestone — lets every test replay a fixture-recorded
response with **no live model anywhere** (M3-SPEC.md §7's `test_llm_cache.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

from fireassay.hashing import content_hash
from fireassay.llm.ollama import ModelRef

_CACHE_KEY_PREFIX = "fa.llmcache.1"


def _cache_key(model: ModelRef, prompt: str) -> str:
    return content_hash(_CACHE_KEY_PREFIX, model.digest, prompt)


class ResponseCache:
    """JSONL on disk at `path`. Loaded fully into memory on construction
    (an LLM cache for a curation run is at most a few thousand entries —
    trivial to hold in memory — and this keeps `get` O(1) with no repeated
    file scans).

    Each line is `{"key": <content_hash>, "response": <raw text>}`. `put`
    is idempotent on `key`: re-putting an identical `(model, prompt)` pair
    is a silent no-op rather than a duplicate append, mirroring `Store`'s
    idempotent-on-content-hash writes elsewhere in this codebase.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, str] = {}
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._entries[obj["key"]] = obj["response"]

    def get(self, model: ModelRef, prompt: str) -> str | None:
        """The cached raw response for `(model.digest, prompt)`, or `None`
        on a miss. **Never makes a network call** — a miss is simply
        `None`; calling the model on a miss is `OllamaClient.generate_json`'s
        job, not this class's."""
        return self._entries.get(_cache_key(model, prompt))

    def put(self, model: ModelRef, prompt: str, response: str) -> None:
        key = _cache_key(model, prompt)
        if key in self._entries:
            return
        self._entries[key] = response
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")


def populate_from_fixture(cache: ResponseCache, fixture_path: Path | str) -> None:
    """Populate `cache` from a fixture JSONL file, one object per line
    shaped `{"model_name", "model_digest", "prompt", "response"}`.

    This is the mechanism `M3-SPEC.md §1` describes: "tests use a
    `ResponseCache` pre-populated from `tests/fixtures/llm/*.jsonl`". The
    fixture format is deliberately more verbose than the cache file's own
    `{"key", "response"}` shape — `model_name`/`prompt` are kept alongside
    `model_digest` in the fixture so a reader can see *what request* a
    recorded response answers, whereas the cache file itself, once
    `put`, only needs the derived key.
    """
    with open(fixture_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            model = ModelRef(name=obj["model_name"], digest=obj["model_digest"])
            cache.put(model, obj["prompt"], obj["response"])
