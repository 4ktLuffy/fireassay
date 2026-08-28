"""Single source of truth for text normalisation/tokenisation.

A divergent tokenizer between the retriever and any other text-matching
component (near-duplicate detection, a future curation dedup pass,
`score.retrieval`'s judged-fraction overlap logic) produces measurements
that quietly disagree for reasons unrelated to what is actually being
measured — and the divergence is invisible, because both sides still
return plausible-looking numbers. This has bitten prior projects multiple
times, in multiple files, in a single day, before being centralised.
fireassay centralises it from day one: `system.bm25.BM25System` and any
future text-matching component import `tokenize` from here. Never write a
tokenizing regex inline anywhere else in this codebase.
"""

from __future__ import annotations

import re
from typing import Final

#: Unicode-aware "word character" runs. Matches letters, digits and
#: underscore in any script, not just ASCII.
WORD_RE: Final = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """The ONLY tokenizer in this codebase.

    Lowercases `text`, then extracts every maximal run of word characters
    as a token. Equivalent to "split on runs of non-word characters, drop
    empty tokens" (M1-SPEC.md §6's original phrasing) but expressed as an
    extraction rather than a split-and-filter, which cannot produce empty
    tokens by construction.
    """
    return WORD_RE.findall(text.lower())
