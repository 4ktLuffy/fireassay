"""`Embed` — the embedding callable the dense retriever ranks with, and
`OllamaEmbedder`, the one implementation that reaches a live model.

Three rules here are load-bearing, and each has a negative control in
`tests/test_embedding.py` that goes red when it is removed:

- **Pin the model by digest, never by tag** (M3-SPEC.md §8, already the
  rule for every generating model in this repo). `OllamaEmbedder.resolve`
  goes through `llm.ollama.OllamaClient.model_ref`, so the `ModelRef` it
  carries is what the embedding cache keys on and what an
  `env_fingerprint` would record. An Ollama tag can be re-pulled onto
  different weights; a cache keyed on the tag would then silently replay
  one model's vectors as another's.
- **Reduce in float64, store float32.** A float32 cosine over 22M
  elements has been observed returning 1.0106 — an impossible value for a
  cosine, and the sort of thing that reaches a report as a finding. Every
  norm here is computed in float64 and only the finished, normalised
  vector is narrowed to float32.
- **A degenerate vector is an error, never a padded row.** A zero norm, a
  non-finite component, or a dimension that disagrees with the first
  vector this embedder ever produced raises `EmbeddingError`. Silently
  padding or zero-filling one row would put a vector into the matrix that
  scores against every question identically, which is indistinguishable
  downstream from a retriever that simply never finds anything.

**Batched, because the alternative is 77ms per call.** Ollama's
`/api/embed` takes an `input` *array* and returns one vector per element.
The older `/api/embeddings` takes a single `prompt`; at 12,513 chunks that
is 16 minutes of round-trip latency for one chunking. This module only
ever calls `/api/embed`.

No new dependency: `urllib.request` (the one HTTP mechanism M3-SPEC.md
permits) and numpy.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np

from fireassay.llm.ollama import ModelRef, OllamaClient

#: `(n, d) float32, L2-normalised, row i for text i`. Every consumer of an
#: embedding in this repo takes this type and nothing narrower, so a test
#: double (a constant embedder, a shuffled one) is a plain function.
Embed = Callable[[Sequence[str]], np.ndarray]

#: The local embedding model M-DENSE-SPEC.md §2 fixes: 1024 dimensions,
#: Q8_0, served by Ollama. Nothing here hard-codes the dimension — it is
#: discovered from the first batch and then enforced.
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"

DEFAULT_BASE_URL = "http://localhost:11434"

#: Guard added to every norm before dividing. A norm that is genuinely
#: zero is rejected outright (see `l2_normalise`); this exists so a
#: denormal-but-nonzero norm cannot turn into an inf.
_NORM_EPS = 1e-12


class EmbeddingError(RuntimeError):
    """An embedding response was unusable: wrong shape, wrong dimension,
    a zero norm, or a non-finite component. Raised rather than returning a
    best-effort matrix, for the reason `llm.ollama.GenerationExhaustedError`
    exists one layer over — a silently degraded vector poisons every
    comparison computed from it, with no trace."""


class _Post(Protocol):
    """The single network operation `OllamaEmbedder` performs, as a
    `Protocol` so a test can supply a recorded response without patching
    `urllib` (M3-SPEC.md §7: no test may call a live model)."""

    def __call__(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]: ...


def l2_normalise(rows: object) -> np.ndarray:
    """`(n, d)` float32, every row unit-length, reduced in float64.

    Raises `EmbeddingError` if `rows` is not rectangular and 2-D, if any
    component is non-finite, or if any row's norm is zero — see this
    module's docstring for why none of those may be repaired silently.
    """
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise EmbeddingError(f"expected a non-empty (n, d) matrix of embeddings, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        bad = int(np.argmax(~np.isfinite(arr).all(axis=1)))
        raise EmbeddingError(f"embedding row {bad} has a non-finite component")
    norms = np.sqrt((arr * arr).sum(axis=1))
    zero = np.flatnonzero(norms == 0.0)
    if zero.size:
        raise EmbeddingError(
            f"embedding row {int(zero[0])} has a zero norm; a zero vector is not a direction and "
            "must never be padded into the matrix"
        )
    return (arr / (norms[:, None] + _NORM_EPS)).astype(np.float32)


def _urllib_post(base_url: str, timeout: float) -> _Post:
    def post(path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body: Mapping[str, object] = json.loads(response.read().decode("utf-8"))
            return body

    return post


class OllamaEmbedder:
    """An `Embed` backed by a local Ollama server's batched `/api/embed`.

    `model` is a `ModelRef` (name **and** digest) — construct it with
    `OllamaEmbedder.resolve`, which reads the digest off `/api/tags`, so
    the identity that reaches `system.embedding_cache`'s key is the
    weights' content digest rather than a re-pullable tag.

    `batch_size` is how many texts go into one `input` array. 64 is a
    compromise, not a tuned figure: large enough that per-request latency
    stops dominating, small enough that one failed request costs little
    and that a long chunk batch stays inside the server's context budget.
    """

    def __init__(
        self,
        model: ModelRef,
        *,
        base_url: str = DEFAULT_BASE_URL,
        batch_size: int = 64,
        timeout: float = 600.0,
        post: _Post | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._post: _Post = post if post is not None else _urllib_post(self.base_url, timeout)
        self._dimension: int | None = None

    @classmethod
    def resolve(
        cls,
        name: str = DEFAULT_EMBED_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        batch_size: int = 64,
        timeout: float = 600.0,
    ) -> OllamaEmbedder:
        """Resolve `name` to a digest-pinned `ModelRef` via `/api/tags`
        and return an embedder for it. Raises whatever
        `OllamaClient.model_ref` raises if the server is not serving or
        does not have the model pulled — never falls back to a tag-only
        `ModelRef`, since that is precisely the identity this repo
        refuses to key anything on."""
        client = OllamaClient(base_url=base_url, timeout=timeout)
        return cls(client.model_ref(name), base_url=base_url, batch_size=batch_size, timeout=timeout)

    @property
    def dimension(self) -> int | None:
        """The embedding width, or `None` before the first batch. Set once
        from the first response and enforced on every later one."""
        return self._dimension

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        body = self._post("/api/embed", {"model": self.model.name, "input": list(texts)})
        raw = body.get("embeddings")
        if not isinstance(raw, list):
            keys = ", ".join(sorted(str(k) for k in body))
            raise EmbeddingError(f"ollama /api/embed returned no 'embeddings' list. Response keys: [{keys}]")
        if len(raw) != len(texts):
            raise EmbeddingError(
                f"ollama /api/embed returned {len(raw)} vector(s) for {len(texts)} input text(s); "
                "a batched response that does not line up with its input cannot be aligned to chunks"
            )
        return l2_normalise(raw)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Embed `texts` in batches of `batch_size`, returning one
        `(len(texts), d)` float32 matrix with rows in input order.

        An empty input returns an empty `(0, d)` matrix only once `d` is
        known; before the first successful batch there is no dimension to
        report and `EmbeddingError` is raised instead of inventing one."""
        if not texts:
            if self._dimension is None:
                raise EmbeddingError(
                    "cannot embed an empty batch before any dimension has been observed; "
                    "embed at least one text first"
                )
            return np.zeros((0, self._dimension), dtype=np.float32)

        parts: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            block = self._embed_batch(texts[start : start + self.batch_size])
            width = int(block.shape[1])
            if self._dimension is None:
                self._dimension = width
            elif width != self._dimension:
                raise EmbeddingError(
                    f"ollama /api/embed returned {width}-dimensional vectors after "
                    f"{self._dimension}-dimensional ones; a matrix of mixed width is not a vector space"
                )
            parts.append(block)
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
