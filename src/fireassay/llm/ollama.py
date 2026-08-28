"""`OllamaClient` — the one place in fireassay allowed to call a live model
(M3-SPEC.md §1). Talks to a local Ollama server over plain `urllib.request`
(no new HTTP dependency, matching `tools/fetch_govuk.py`), and every
response can be replayed from a `ResponseCache` instead of the network.

Two rules here are load-bearing (M3-SPEC.md §8 flags both explicitly):

- **Pin by digest, never by tag.** `ModelRef.digest` — not `.name` — is
  what identifies a model everywhere it matters (the cache key,
  `env_fingerprint.affects_results.generator_model`): an Ollama tag can be
  re-pulled and silently point at different weights, so anything keyed on
  the tag alone would replay/attribute results to a model that no longer
  exists.
- **Exhausting retries raises, never returns a partial or best-effort
  object.** A generator that silently degrades on bad output would quietly
  poison a golden set with data nobody chose to accept — see
  `GenerationExhaustedError`.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError


def _api_error(endpoint: str, name: str, body: Mapping[str, object], problem: str) -> str:
    """Build a diagnosis that identifies the failure WITHOUT interpolating
    the raw response.

    An earlier version interpolated the whole `/api/show` body into this
    message, which meant the actual problem arrived after several kilobytes
    of tensor metadata (`{'name': 'blk.27.ffn_down.weight', ...}` x hundreds).
    A response body is never the diagnosis: the endpoint, the model asked
    for, and the keys that came back are.
    """
    keys = ", ".join(sorted(str(k) for k in body)[:25])
    return f"ollama {endpoint} for {name!r}: {problem}. Response keys: [{keys}]"


@dataclass(frozen=True)
class ModelRef:
    """A model, identified by both its human-readable tag (`name`, e.g.
    "qwen2.5:7b") and its content digest (`digest`, from `/api/tags`) —
    the digest is the identity that actually participates in caching and
    `env_fingerprint`; `name` is carried along purely for readability in
    logs/reports.

    `quantization_level` and `parameter_size` are carried for the same
    readability reason: the digest already changes when quantisation does,
    but a human reading an `env_fingerprint` should be able to see
    "Q4_K_M" without first resolving a hash — and quantisation genuinely
    changes what the model outputs.
    """

    name: str
    digest: str
    quantization_level: str = ""
    parameter_size: str = ""


class _CacheProtocol(Protocol):
    """The subset of `llm.cache.ResponseCache`'s interface `OllamaClient`
    depends on, expressed as a `Protocol` rather than importing the
    concrete class — `cache.py` already imports `ModelRef` from this
    module, so importing `ResponseCache` back would be a cycle. Any object
    with `get`/`put` of this shape (a real cache, or a test double) works."""

    def get(self, model: ModelRef, prompt: str) -> str | None: ...

    def put(self, model: ModelRef, prompt: str, response: str) -> None: ...


class GenerationExhaustedError(Exception):
    """Raised by `OllamaClient.generate_json` when `max_retries` attempts
    all produced output that failed JSON/schema validation.

    Deliberately **not** a case where `generate_json` instead returns
    `None` or a partially-populated object: M3-SPEC.md §1 is explicit that
    exhausting retries must raise, because a generator that silently
    degrades is a generator that quietly poisons a golden set with
    candidates nobody actually validated as well-formed.
    """


class OllamaClient:
    """A thin, cache-aware wrapper over Ollama's HTTP API.

    All network I/O funnels through `_post`/`_generate_raw` — the two
    methods a test double/monkeypatch replaces to exercise retry and
    exhaustion behaviour without ever reaching a live model (M3-SPEC.md §7:
    "no test may call a live model... via recorded fixtures, never the
    network"). A cache hit in `generate_json` short-circuits before either
    method is ever called at all.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        cache: _CacheProtocol | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._cache = cache
        self._timeout = timeout

    def _get(self, path: str) -> dict[str, object]:
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body: dict[str, object] = json.loads(resp.read().decode("utf-8"))
            return body

    def _post(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body: dict[str, object] = json.loads(resp.read().decode("utf-8"))
            return body

    def model_ref(self, name: str) -> ModelRef:
        """Resolve `name` (a tag, e.g. "qwen2.5:7b") to a `ModelRef`
        carrying its content digest, via Ollama's `GET /api/tags`.

        **Not `/api/show`.** Verified against a live Ollama server: the
        `/api/show` response carries no `digest` key at all (its top-level
        keys are capabilities, details, license, model_info, modelfile,
        modified_at, system, template, tensors). Only `/api/tags` reports a
        per-model digest. Reading it from `/api/show` fails on every real
        server while passing against any fixture written from the same
        wrong assumption, so the fixture in `tests/fixtures/llm/` is a
        recorded `/api/tags` body and must stay that way.
        """
        body = self._get("/api/tags")
        models = body.get("models")
        if not isinstance(models, list):
            raise ValueError(_api_error("/api/tags", name, body, "response has no 'models' list"))
        entries = [m for m in models if isinstance(m, dict)]
        match = next((m for m in entries if m.get("name") == name), None)
        if match is None:
            match = next((m for m in entries if m.get("model") == name), None)
        if match is None:
            available = sorted(str(m.get("name", "?")) for m in entries)[:20]
            raise ValueError(
                f"ollama /api/tags has no model tagged {name!r}; available: {', '.join(available)}"
            )
        digest = match.get("digest")
        if not isinstance(digest, str) or not digest:
            raise ValueError(_api_error("/api/tags", name, match, "entry has no 'digest'"))
        details = match.get("details")
        details = details if isinstance(details, dict) else {}
        return ModelRef(
            name=name,
            digest=digest,
            quantization_level=str(details.get("quantization_level", "")),
            parameter_size=str(details.get("parameter_size", "")),
        )

    def _generate_raw(self, model: ModelRef, prompt: str, temperature: float) -> str:
        """One raw `/api/generate` call, JSON-mode, non-streaming. Returns
        the model's raw text response (not yet parsed/validated) — a
        separate method from `generate_json` specifically so tests can
        monkeypatch just this one network-touching call and drive
        `generate_json`'s retry/exhaustion logic with canned responses."""
        payload = {
            "model": model.name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        result = self._post("/api/generate", payload)
        response = result.get("response")
        if not isinstance(response, str):
            raise ValueError(
                f"ollama /api/generate response missing a 'response' string field: {result!r}"
            )
        return response

    def generate_json(
        self,
        model: ModelRef,
        prompt: str,
        schema: type[BaseModel],
        *,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> BaseModel:
        """Request JSON from `model` for `prompt`, validate it against
        `schema`, and return the validated instance.

        Cache lookup happens first and, on a hit, is the **only** thing
        that happens — no network call is made at all (this is what
        `test_llm_cache.py` pins: "a cache hit makes no client call"). A
        cached response is trusted as already-valid (it was only ever
        `put` here after passing this exact validation on a prior attempt),
        so a cache hit re-validates it but never retries it.

        On a cache miss, calls `_generate_raw` up to `max_retries` times.
        Each attempt whose output fails to parse as JSON or fails `schema`
        validation is silently retried (not raised immediately) — an LLM
        occasionally emits malformed JSON even in JSON mode, and one retry
        recovers the overwhelming majority of those. Only a **successful**
        attempt is written to the cache. **Exhausting every attempt raises
        `GenerationExhaustedError`** — see the module docstring for why
        this must never instead return something best-effort.
        """
        if self._cache is not None:
            cached = self._cache.get(model, prompt)
            if cached is not None:
                return schema.model_validate_json(cached)

        last_error: Exception | None = None
        for _attempt in range(max_retries):
            raw = self._generate_raw(model, prompt, temperature)
            try:
                validated = schema.model_validate_json(raw)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                continue
            if self._cache is not None:
                self._cache.put(model, prompt, raw)
            return validated

        raise GenerationExhaustedError(
            f"exhausted {max_retries} retries generating valid {schema.__name__} JSON "
            f"from model={model.name}@{model.digest[:12]}"
        ) from last_error
