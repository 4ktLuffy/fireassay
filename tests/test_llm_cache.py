"""M3-SPEC.md §7: content-addressed on (digest, prompt); a cache hit makes
no client call."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from fireassay.llm.cache import ResponseCache, populate_from_fixture
from fireassay.llm.ollama import ModelRef, OllamaClient

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llm"


class _Answer(BaseModel):
    value: int


def test_get_is_none_on_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.jsonl")
    model = ModelRef(name="m", digest="d1")
    assert cache.get(model, "never put") is None


def test_content_addressed_on_digest_and_prompt(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.jsonl")
    model_a = ModelRef(name="tag-a", digest="digest-a")
    model_b = ModelRef(name="tag-b", digest="digest-b")  # different digest -> different tag OK too
    cache.put(model_a, "prompt-x", "response-for-a-x")

    assert cache.get(model_a, "prompt-x") == "response-for-a-x"
    # Same digest, different prompt -> miss.
    assert cache.get(model_a, "prompt-y") is None
    # Different digest, same prompt -> miss: keyed on digest, never on name/tag.
    assert cache.get(model_b, "prompt-x") is None


def test_same_name_different_digest_does_not_collide(tmp_path: Path) -> None:
    """Pin by digest, never by tag (M3-SPEC.md §1): a re-pulled tag must
    not replay a stale entry."""
    cache = ResponseCache(tmp_path / "cache.jsonl")
    stale = ModelRef(name="qwen2.5:7b", digest="old-digest")
    fresh = ModelRef(name="qwen2.5:7b", digest="new-digest")
    cache.put(stale, "prompt", "stale response")
    assert cache.get(fresh, "prompt") is None
    assert cache.get(stale, "prompt") == "stale response"


def test_put_is_idempotent_no_duplicate_lines(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    model = ModelRef(name="m", digest="d")
    cache.put(model, "p", "r")
    cache.put(model, "p", "r")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1


def test_cache_persists_and_reloads_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    model = ModelRef(name="m", digest="d")
    ResponseCache(path).put(model, "prompt", "the response")
    reloaded = ResponseCache(path)
    assert reloaded.get(model, "prompt") == "the response"


def test_populate_from_fixture(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.jsonl")
    populate_from_fixture(cache, FIXTURES_DIR / "sample_responses.jsonl")
    model = ModelRef(name="qwen2.5:7b", digest="digest-sha-abc123")
    assert cache.get(model, "hello world") == '{"value": 42}'
    assert cache.get(model, "second prompt") == '{"value": 7}'


def test_cache_hit_makes_no_client_call(tmp_path: Path) -> None:
    """A `ResponseCache` pre-populated from a fixture, handed to a client
    whose `base_url` would fail if actually called: `generate_json` must
    succeed purely from the cache, proving no network call was made."""
    cache = ResponseCache(tmp_path / "cache.jsonl")
    populate_from_fixture(cache, FIXTURES_DIR / "sample_responses.jsonl")
    model = ModelRef(name="qwen2.5:7b", digest="digest-sha-abc123")
    # Port 1 on localhost: connection refused immediately if ever dialed,
    # never DNS-dependent or slow -- a network call here fails loudly.
    client = OllamaClient(base_url="http://127.0.0.1:1", cache=cache)

    result = client.generate_json(model, "hello world", _Answer)

    assert isinstance(result, _Answer)
    assert result.value == 42
