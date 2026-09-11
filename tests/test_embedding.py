"""`system.embedding`: normalisation, batching, and the degeneracies that
must raise rather than pad.

No test calls a live model (M3-SPEC.md §7). `OllamaEmbedder` takes its one
network operation as an injected `post`, so every case here drives the
real batching/validation code against a recorded-shaped response.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from fireassay.llm.ollama import ModelRef
from fireassay.system.embedding import (
    DEFAULT_EMBED_MODEL,
    EmbeddingError,
    OllamaEmbedder,
    l2_normalise,
)

MODEL = ModelRef(name=DEFAULT_EMBED_MODEL, digest="ac6da0dfba84a81f", quantization_level="Q8_0")


class _Recorder:
    """A stand-in `/api/embed`: returns a deterministic vector per text and
    records the batch sizes it was asked for."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.batches: list[int] = []
        self.calls = 0

    def __call__(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        assert path == "/api/embed", "the singular /api/embeddings endpoint must never be used"
        texts = payload["input"]
        assert isinstance(texts, list), "/api/embed must be called with a batched 'input' array"
        self.calls += 1
        self.batches.append(len(texts))
        rows = []
        for text in texts:
            row = [0.0] * self.dimension
            for i, ch in enumerate(str(text)):
                row[i % self.dimension] += float(ord(ch) % 7) + 1.0
            rows.append(row)
        return {"embeddings": rows}


def test_l2_normalise_returns_unit_float32_rows() -> None:
    out = l2_normalise([[3.0, 4.0], [1.0, 0.0]])
    assert out.dtype == np.float32
    assert np.allclose(np.sqrt((out.astype(np.float64) ** 2).sum(axis=1)), 1.0)


def test_l2_normalise_reduces_in_float64() -> None:
    """A float32 reduction over a wide row loses the small components
    entirely; the float64 one keeps them. This is the 1.0106 cosine, in
    miniature."""
    row = [1.0] + [1e-4] * 4095
    out = l2_normalise([row])
    norm64 = float(np.sqrt((np.asarray(out, dtype=np.float64) ** 2).sum()))
    assert abs(norm64 - 1.0) < 1e-6


def test_l2_normalise_refuses_a_zero_vector() -> None:
    with pytest.raises(EmbeddingError, match="zero norm"):
        l2_normalise([[1.0, 1.0], [0.0, 0.0]])


def test_l2_normalise_refuses_a_non_finite_component() -> None:
    with pytest.raises(EmbeddingError, match="non-finite"):
        l2_normalise([[1.0, float("nan")]])


def test_l2_normalise_refuses_an_empty_matrix() -> None:
    with pytest.raises(EmbeddingError, match="non-empty"):
        l2_normalise(np.zeros((0, 4)))


def test_embedder_batches_at_batch_size_and_preserves_input_order() -> None:
    recorder = _Recorder()
    embedder = OllamaEmbedder(MODEL, batch_size=3, post=recorder)
    texts = [f"text-{i}" for i in range(7)]
    out = embedder(texts)
    assert out.shape == (7, 8)
    assert recorder.batches == [3, 3, 1]
    # Row i must be text i: re-embedding one text alone reproduces its row.
    assert np.allclose(out[4], embedder([texts[4]])[0])


def test_embedder_records_dimension_from_the_first_batch() -> None:
    embedder = OllamaEmbedder(MODEL, batch_size=4, post=_Recorder(dimension=16))
    assert embedder.dimension is None
    embedder(["a"])
    assert embedder.dimension == 16


def test_embedder_refuses_a_dimension_change_mid_run() -> None:
    """A model swapped underneath a long run is the realistic cause; a
    ragged matrix would be the silent result."""
    recorder = _Recorder(dimension=8)

    def post(path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        body = recorder(path, payload)
        if recorder.calls > 1:
            rows = [row + [1.0] for row in body["embeddings"]]  # type: ignore[union-attr]
            return {"embeddings": rows}
        return body

    embedder = OllamaEmbedder(MODEL, batch_size=2, post=post)
    with pytest.raises(EmbeddingError, match="mixed width"):
        embedder(["a", "b", "c", "d"])


def test_embedder_refuses_a_response_that_does_not_line_up_with_its_input() -> None:
    def short(path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        return {"embeddings": [[1.0, 0.0]]}

    embedder = OllamaEmbedder(MODEL, batch_size=4, post=short)
    with pytest.raises(EmbeddingError, match="does not line up"):
        embedder(["a", "b"])


def test_embedder_refuses_a_response_with_no_embeddings_key() -> None:
    def broken(path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        return {"error": "model not found"}

    embedder = OllamaEmbedder(MODEL, batch_size=4, post=broken)
    with pytest.raises(EmbeddingError, match="no 'embeddings' list"):
        embedder(["a"])


def test_embedder_refuses_an_empty_batch_before_any_dimension_is_known() -> None:
    embedder = OllamaEmbedder(MODEL, post=_Recorder())
    with pytest.raises(EmbeddingError, match="empty batch"):
        embedder([])
    embedder(["a"])
    assert embedder([]).shape == (0, 8)


def test_embedder_rejects_a_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        OllamaEmbedder(MODEL, batch_size=0, post=_Recorder())


def test_embedder_sends_the_model_tag_but_is_identified_by_digest() -> None:
    """The tag is what Ollama's API takes; the digest is what everything
    downstream keys on (M3-SPEC.md §8). Both must be present and they must
    not be conflated."""
    seen: list[Mapping[str, object]] = []

    def post(path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        seen.append(payload)
        return {"embeddings": [[1.0, 0.0]] * len(payload["input"])}  # type: ignore[arg-type]

    embedder = OllamaEmbedder(MODEL, post=post)
    embedder(["a"])
    assert seen[0]["model"] == DEFAULT_EMBED_MODEL
    assert embedder.model.digest == "ac6da0dfba84a81f"


def test_embed_is_a_plain_callable_over_a_sequence() -> None:
    """The `Embed` contract: anything `(Sequence[str]) -> (n, d) float32`
    is a valid embedder, which is what lets every negative control be a
    three-line function rather than a mock."""
    embedder = OllamaEmbedder(MODEL, post=_Recorder())

    def use(embed: object, texts: Sequence[str]) -> np.ndarray:
        assert callable(embed)
        return embed(texts)

    assert use(embedder, ("a", "b")).shape == (2, 8)
