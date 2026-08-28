"""Deterministic content-addressing primitives for fireassay.

fireassay identifies questions, suites and configs by content hash rather
than by an arbitrary autoincrement id, so that two people (or two runs,
weeks apart) who produce byte-identical content are guaranteed to produce
the same id, and anyone who changes the content is guaranteed to produce a
different one. That single property is what lets integrity.py refuse to
compare runs whose suites have silently drifted.

The three id definitions below (question.id, suite.suite_hash,
config.config_hash) are contractual: they are pinned by literal-value
regression tests in tests/test_hashing.py. Do not change the prefix, part
order, or separator without also bumping the prefix (which is a hash-scheme
version tag) — otherwise an old id and a new id could silently collide.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

_UNIT_SEPARATOR = b"\x1f"

# Hash-scheme version tags. Bump the numeric suffix (never change the
# letters) if the corresponding hash definition ever changes, so an old id
# and a new id can never collide even if the underlying content happens to
# match.
_QUESTION_ID_PREFIX = "fa.question.1"
_SUITE_HASH_PREFIX = "fa.suite.1"
_CONFIG_HASH_PREFIX = "fa.config.1"


def canonical_json(obj: object) -> bytes:
    """Serialise `obj` to a deterministic JSON byte string.

    Keys are sorted, separators are the tightest form (``(',', ':')``), and
    non-ASCII characters are left as UTF-8 rather than ``\\uXXXX``-escaped,
    so that the same logical object always produces the same bytes
    regardless of dict insertion order, locale, or Python version. This
    determinism is exactly what config_hash relies on.

    Raises ``ValueError`` on NaN/Infinity: Python's json module serialises
    those to the non-standard tokens ``NaN``/``Infinity``, which some
    platforms and parsers render differently. Allowing them through would
    make a hash computed on one machine unreproducible on another, which
    defeats the entire point of content addressing.
    """
    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"canonical_json: object contains NaN/Infinity, which is not "
            f"permitted in a hashed payload: {exc}"
        ) from exc
    return text.encode("utf-8")


def content_hash(prefix: str, *parts: str | bytes) -> str:
    """Compute a sha256 hex digest over `prefix` and `parts`.

    Each `str` part is UTF-8 encoded; `bytes` parts are used as-is. All
    parts, with `prefix` first, are then joined with the ASCII unit
    separator ``0x1F`` before hashing. The unit separator (rather than a
    printable character such as ``:`` or ``\\n``) is used specifically
    because question/config text is free-form and could otherwise be
    crafted to make two logically different inputs hash identically.

    `prefix` is a hash-scheme version tag (e.g. ``"fa.question.1"`` for question ids,
    scheme v1) rather than part of the semantic content: it exists so the
    hash scheme can change in a future version without silently colliding
    with ids produced by an older scheme.
    """
    encoded_parts = [prefix.encode("utf-8")]
    for part in parts:
        encoded_parts.append(part.encode("utf-8") if isinstance(part, str) else part)
    payload = _UNIT_SEPARATOR.join(encoded_parts)
    return hashlib.sha256(payload).hexdigest()


def question_id(
    text: str,
    reference_answer: str | None,
    evidence_span_keys: Iterable[str],
) -> str:
    """Content-address a question.

    ``fa.question.1`` over (text, reference_answer or "", "\\n".join(sorted(evidence
    span keys))). Evidence span keys are sorted before joining so that the
    id does not depend on the order evidence spans happen to be listed in —
    only on which spans are present.
    """
    joined_spans = "\n".join(sorted(evidence_span_keys))
    return content_hash(_QUESTION_ID_PREFIX, text, reference_answer or "", joined_spans)


def suite_hash(question_ids: Iterable[str]) -> str:
    """Content-address a suite as the sorted set of its question ids.

    Sorting before joining means a suite is defined by *which* questions it
    contains, not the order they were frozen in — reordering the same
    question set must not change suite_hash, or two functionally identical
    suites could wrongly be refused as incomparable by integrity.py.
    """
    return content_hash(_SUITE_HASH_PREFIX, "\n".join(sorted(question_ids)))


def config_hash(spec: Mapping[str, object]) -> str:
    """Content-address a config spec via its canonical JSON serialisation."""
    return content_hash(_CONFIG_HASH_PREFIX, canonical_json(dict(spec)))
