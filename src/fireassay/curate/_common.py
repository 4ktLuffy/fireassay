"""Deterministic seeding shared by `queue.py` and `honeypots.py`.

`stable_seed` is deliberately **duplicated**, not imported, from
`controls._common.stable_seed` — that module is private to `controls/`
(see its own docstring: "nothing here is part of the public API"), and
`curate/` is a fully separate package with no dependency on `controls/`.
The definition is a one-line, frozen contract (sha256 over unit-separator-
joined parts); duplicating it here is the same trade `controls._common.
overlap_chars` already made against `score.retrieval`'s private helper,
for the same reason.
"""

from __future__ import annotations

import hashlib


def stable_seed(*parts: object) -> int:
    """A stable integer seed derived from `parts` via sha256 — not
    Python's randomised-per-process builtin `hash()` (see
    `controls._common.stable_seed`'s docstring for why that matters)."""
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
