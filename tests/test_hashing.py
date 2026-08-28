"""Regression-lock tests for fireassay.hashing.

Note on the "pin to literal values" requirement (M1-SPEC.md §11): each hash
scheme is locked two independent ways:

1. A literal-value test (`test_*_pinned_literal`) asserting the exact
   digest string, run and confirmed against the real implementation.
2. A from-scratch reconstruction test (`test_*_pinned_to_documented_scheme`)
   that independently rebuilds the documented byte layout
   (prefix + b'\\x1f' + UTF-8-encoded parts, hashed with sha256) using
   `hashlib` directly, without calling into `hashing.content_hash` at all.

The two catch different regressions: (1) catches ANY change to the output,
including one that accidentally kept the byte-layout logic "self-
consistent" with a rewritten `content_hash`; (2) catches ANY change to the
prefix/separator/part order/encoding even if someone changed both
`content_hash` and its only caller in a matching way. Keeping both is
deliberate, not redundant.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from fireassay.hashing import canonical_json, config_hash, content_hash, question_id, suite_hash
from fireassay.models import EvidenceSpan, Question


def _reference_content_hash(prefix: str, *parts: str | bytes) -> str:
    """Independent reimplementation of the documented byte layout in
    hashing.content_hash, used only in these tests to avoid regression-
    locking the implementation against itself."""
    encoded = [prefix.encode("utf-8")]
    for part in parts:
        encoded.append(part.encode("utf-8") if isinstance(part, str) else part)
    return hashlib.sha256(b"\x1f".join(encoded)).hexdigest()


# -- canonical_json -----------------------------------------------------


def test_canonical_json_is_key_order_independent() -> None:
    a = {"b": 1, "a": 2, "c": {"y": 1, "x": 2}}
    b = {"c": {"x": 2, "y": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_canonical_json_rejects_infinity() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_json({"x": float("-inf")})


def test_canonical_json_uses_tight_separators_and_utf8() -> None:
    payload = canonical_json({"a": 1, "b": "héllo"})
    assert b", " not in payload
    assert b": " not in payload
    assert "héllo".encode() in payload  # ensure_ascii=False


# -- content_hash regression lock ----------------------------------------


def test_content_hash_matches_documented_byte_layout() -> None:
    assert content_hash("fa.question.1", "hello", "world") == _reference_content_hash(
        "fa.question.1", "hello", "world"
    )


def test_content_hash_is_sha256_hex() -> None:
    digest = content_hash("fa.question.1", "x")
    assert len(digest) == 64
    int(digest, 16)  # does not raise: valid hex


# -- question.id (prefix fa.question.1) --------------------------------------------


def test_question_id_pinned_literal() -> None:
    q = Question(
        text="What is X?",
        qtype="factual",
        difficulty="easy",
        reference_answer="X is Y.",
        evidence_spans=(
            EvidenceSpan(doc_id="d1", chunk_id="d1#0000", char_start=0, char_end=10, quote="hello"),
        ),
        provenance="synthetic",
        generator="gen@1.0.0",
        source_doc_id="d1",
    )
    assert q.id == "55c4b935649dfd845e0cb78f4937876df5bc9dd33694d2de91cd9495686d739c"


def test_question_id_pinned_to_documented_scheme() -> None:
    expected = _reference_content_hash("fa.question.1", "What is X?", "X is Y.", "docA:chunk0:0:10")
    assert question_id("What is X?", "X is Y.", ["docA:chunk0:0:10"]) == expected


def test_question_id_sorts_evidence_span_keys_before_joining() -> None:
    forward = question_id("q", "r", ["z:1:0:1", "a:1:0:1"])
    backward = question_id("q", "r", ["a:1:0:1", "z:1:0:1"])
    assert forward == backward


def test_question_id_none_reference_answer_equals_empty_string() -> None:
    assert question_id("q", None, []) == question_id("q", "", [])


def test_question_id_changes_with_text() -> None:
    assert question_id("q1", "r", []) != question_id("q2", "r", [])


# -- suite.suite_hash (prefix fa.suite.1) ----------------------------------------


def test_suite_hash_pinned_literal() -> None:
    expected = "e33d25df7f3a21f50e855064d0c62fbd98f8f849763169cd02464cf688027330"
    assert content_hash("fa.suite.1", "\n".join(sorted(["a", "b"]))) == expected
    assert suite_hash(["b", "a"]) == expected


def test_suite_hash_pinned_to_documented_scheme() -> None:
    expected = _reference_content_hash("fa.suite.1", "q1\nq2\nq3")
    assert suite_hash(["q3", "q1", "q2"]) == expected


def test_suite_hash_is_order_independent() -> None:
    assert suite_hash(["q3", "q1", "q2"]) == suite_hash(["q1", "q2", "q3"])


def test_suite_hash_changes_with_membership() -> None:
    assert suite_hash(["q1", "q2"]) != suite_hash(["q1", "q2", "q3"])


# -- config.config_hash (prefix fa.config.1) --------------------------------------


def test_config_hash_pinned_literal() -> None:
    expected = "1a4651d812d353a31370b233a64c4b08dc0b69a3d3b7ce7f01f06984008047fa"
    assert content_hash("fa.config.1", canonical_json({"m": "x", "k": 3})) == expected
    assert config_hash({"m": "x", "k": 3}) == expected


def test_config_hash_pinned_to_documented_scheme() -> None:
    spec = {"b": 0.75, "top_k": 5}
    canonical_bytes = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    expected = _reference_content_hash("fa.config.1", canonical_bytes)
    assert config_hash(spec) == expected


def test_config_hash_is_key_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_value() -> None:
    assert config_hash({"top_k": 5}) != config_hash({"top_k": 6})


# -- EvidenceSpan.key() -----------------------------------------------------


def test_evidence_span_key_pinned_literal() -> None:
    span = EvidenceSpan(doc_id="d1", chunk_id="d1#0000", char_start=0, char_end=10, quote="hello")
    assert span.key() == "d1:d1#0000:0:10"
