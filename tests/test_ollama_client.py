"""M3-SPEC.md §7: JSON validated against schema; invalid output retries;
exhausting retries RAISES, never returns partial; all via recorded
fixtures/monkeypatches, never the network.

Every test here constructs `OllamaClient` with `base_url="http://127.0.0.1:1"`
(connection refused immediately, no DNS) and monkeypatches
`_generate_raw` — the one network-touching method — so a bug that
accidentally let a real call through would fail loudly (a connection
error), never silently pass.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from fireassay.llm.ollama import GenerationExhaustedError, ModelRef, OllamaClient

_UNREACHABLE = "http://127.0.0.1:1"


class _Answer(BaseModel):
    value: int


def _client() -> OllamaClient:
    return OllamaClient(base_url=_UNREACHABLE)


def test_generate_json_validates_against_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(client, "_generate_raw", lambda model, prompt, temperature: '{"value": 7}')
    result = client.generate_json(ModelRef(name="m", digest="d"), "prompt", _Answer)
    assert isinstance(result, _Answer)
    assert result.value == 7


def test_generate_json_retries_on_invalid_json_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    responses = iter(["not valid json at all", '{"value": 1}'])
    calls = []

    def fake_generate(model: ModelRef, prompt: str, temperature: float) -> str:
        calls.append(1)
        return next(responses)

    monkeypatch.setattr(client, "_generate_raw", fake_generate)
    result = client.generate_json(ModelRef(name="m", digest="d"), "prompt", _Answer, max_retries=3)

    assert isinstance(result, _Answer)
    assert result.value == 1
    assert len(calls) == 2


def test_generate_json_retries_on_schema_mismatch_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    # Valid JSON, but does not match the schema (missing "value") on the
    # first attempt.
    responses = iter(['{"wrong_field": true}', '{"value": 9}'])
    monkeypatch.setattr(client, "_generate_raw", lambda model, prompt, temperature: next(responses))

    result = client.generate_json(ModelRef(name="m", digest="d"), "prompt", _Answer, max_retries=3)
    assert result.value == 9


def test_generate_json_exhausting_retries_raises_never_returns_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    monkeypatch.setattr(client, "_generate_raw", lambda model, prompt, temperature: "always invalid")

    with pytest.raises(GenerationExhaustedError):
        client.generate_json(ModelRef(name="m", digest="d"), "prompt", _Answer, max_retries=3)


def test_generate_json_calls_generate_raw_exactly_max_retries_times_on_total_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    calls = []

    def fake_generate(model: ModelRef, prompt: str, temperature: float) -> str:
        calls.append(1)
        return "still invalid"

    monkeypatch.setattr(client, "_generate_raw", fake_generate)
    with pytest.raises(GenerationExhaustedError):
        client.generate_json(ModelRef(name="m", digest="d"), "prompt", _Answer, max_retries=4)
    assert len(calls) == 4


# -- generate_text: the contained-ness of the format_json change -----------
#
# generate_json must keep sending "format": "json" exactly as before;
# generate_text must send no "format" key at all. Pinned here rather than
# assumed, since a cache keyed on (model, prompt) with no mode would
# silently return the wrong shape if the two ever collided (see
# generate_text's docstring).


def test_generate_text_sends_no_format_key_while_generate_json_still_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    captured: list[dict[str, object]] = []

    def fake_post(path: str, payload: dict[str, object]) -> dict[str, object]:
        captured.append(payload)
        if "format" in payload:
            return {"response": '{"value": 1}'}
        return {"response": "free text ending VERDICT: SUPPORTED"}

    monkeypatch.setattr(client, "_post", fake_post)

    text = client.generate_text(ModelRef(name="m", digest="d"), "prompt")
    assert text == "free text ending VERDICT: SUPPORTED"
    assert "format" not in captured[-1]

    result = client.generate_json(ModelRef(name="m", digest="d"), "prompt2", _Answer)
    assert result.value == 1
    assert captured[-1]["format"] == "json"


def test_generate_text_returns_completion_verbatim_no_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client,
        "_generate_raw",
        lambda model, prompt, temperature, **kwargs: "not json at all -- VERDICT: UNCLEAR",
    )
    text = client.generate_text(ModelRef(name="m", digest="d"), "prompt")
    assert text == "not json at all -- VERDICT: UNCLEAR"


def test_generate_text_passes_format_json_false_to_generate_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_generate_raw(
        model: ModelRef, prompt: str, temperature: float, *, format_json: bool = True
    ) -> str:
        calls.append(format_json)
        return "text"

    client = _client()
    monkeypatch.setattr(client, "_generate_raw", fake_generate_raw)
    client.generate_text(ModelRef(name="m", digest="d"), "prompt")
    assert calls == [False]


def test_generate_text_cache_hit_short_circuits_generate_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubCache:
        def get(self, model: ModelRef, prompt: str) -> str | None:
            return "cached completion -- VERDICT: SUPPORTED"

        def put(self, model: ModelRef, prompt: str, response: str) -> None:
            raise AssertionError("a cache hit must never write back to the cache")

    client = OllamaClient(base_url=_UNREACHABLE, cache=_StubCache())

    def fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("a cache hit must never call _generate_raw")

    monkeypatch.setattr(client, "_generate_raw", fail)

    text = client.generate_text(ModelRef(name="m", digest="d"), "prompt")
    assert text == "cached completion -- VERDICT: SUPPORTED"


def test_generate_text_cache_miss_calls_generate_raw_and_populates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    puts: list[tuple[ModelRef, str, str]] = []

    class _StubCache:
        def get(self, model: ModelRef, prompt: str) -> str | None:
            return None

        def put(self, model: ModelRef, prompt: str, response: str) -> None:
            puts.append((model, prompt, response))

    client = OllamaClient(base_url=_UNREACHABLE, cache=_StubCache())
    monkeypatch.setattr(
        client,
        "_generate_raw",
        lambda model, prompt, temperature, **kwargs: "fresh -- VERDICT: UNCLEAR",
    )

    text = client.generate_text(ModelRef(name="m", digest="d"), "prompt")
    assert text == "fresh -- VERDICT: UNCLEAR"
    assert puts == [(ModelRef(name="m", digest="d"), "prompt", "fresh -- VERDICT: UNCLEAR")]


def test_cache_hit_short_circuits_generate_raw_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubCache:
        def get(self, model: ModelRef, prompt: str) -> str | None:
            return '{"value": 99}'

        def put(self, model: ModelRef, prompt: str, response: str) -> None:
            raise AssertionError("a cache hit must never write back to the cache")

    client = OllamaClient(base_url=_UNREACHABLE, cache=_StubCache())

    def fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("a cache hit must never call _generate_raw")

    monkeypatch.setattr(client, "_generate_raw", fail)

    result = client.generate_json(ModelRef(name="m", digest="d"), "prompt", _Answer)
    assert result.value == 99


# -- model_ref: digest comes from /api/tags, never /api/show ---------------
#
# Recorded from a live Ollama server. /api/show carries NO digest key at
# all -- its top-level keys are capabilities, details, license, model_info,
# modelfile, modified_at, system, template, tensors. Only /api/tags reports
# a per-model digest. The original implementation read /api/show and passed
# every test, because the fixture had been written from the same wrong
# assumption as the code.

_TAGS_BODY: dict[str, object] = {
    "models": [
        {
            "name": "qwen2.5:7b",
            "model": "qwen2.5:7b",
            "digest": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
            "details": {"quantization_level": "Q4_K_M", "parameter_size": "7.6B"},
        },
        {"name": "granite4:7b-a1b-h", "model": "granite4:7b-a1b-h", "digest": "566b725534ea"},
    ]
}

_SHOW_BODY: dict[str, object] = {
    "capabilities": ["completion"],
    "details": {"quantization_level": "Q4_K_M"},
    "modelfile": "FROM ...",
    "template": "{{ .Prompt }}",
    "tensors": [{"name": f"blk.{i}.ffn_down.weight", "type": "Q6_K"} for i in range(300)],
}


def test_model_ref_resolves_digest_and_details_from_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(client, "_get", lambda path: _TAGS_BODY)
    ref = client.model_ref("qwen2.5:7b")
    assert ref.digest.startswith("845dbda0ea48")
    assert ref.quantization_level == "Q4_K_M"
    assert ref.parameter_size == "7.6B"


def test_model_ref_never_consults_the_show_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for the real bug: a /api/show-shaped body has no
    digest, so resolving from it fails against every live server. If
    model_ref ever calls _post again, this fails loudly."""
    client = _client()
    monkeypatch.setattr(client, "_get", lambda path: _TAGS_BODY)

    def _forbidden(path: str, payload: object) -> dict[str, object]:
        raise AssertionError(f"model_ref must not POST to {path}; the digest is in /api/tags")

    monkeypatch.setattr(client, "_post", _forbidden)
    assert client.model_ref("qwen2.5:7b").digest.startswith("845dbda0ea48")


def test_model_ref_unknown_tag_names_what_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    monkeypatch.setattr(client, "_get", lambda path: _TAGS_BODY)
    with pytest.raises(ValueError, match="nope:404") as excinfo:
        client.model_ref("nope:404")
    assert "qwen2.5:7b" in str(excinfo.value)


def test_model_ref_error_never_contains_the_raw_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original failure printed hundreds of tensor entries before the
    actual diagnosis. A response body is never the diagnosis."""
    client = _client()
    monkeypatch.setattr(client, "_get", lambda path: _SHOW_BODY)
    with pytest.raises(ValueError) as excinfo:
        client.model_ref("qwen2.5:7b")
    message = str(excinfo.value)
    # Naming "tensors" as a KEY is good diagnosis -- it is what tells a reader
    # they hit /api/show. Reproducing its CONTENTS is what buried the original
    # failure under kilobytes of tensor metadata.
    assert "blk." not in message
    assert "Q6_K" not in message
    assert len(message) < 600, f"error message is {len(message)} chars; it should be a diagnosis"
