"""The no-model, token-overlap baseline for "does this item's evidence
actually support its reference answer?" -- the crude proxy every
model-backed answerability judge (`items.answerability`) must beat, and
the instrument `tools/evidence.py`'s `judge.token_overlap` claim reads.

Until this module existed, `answer_unsupported`'s published precision
(handbook §9b) came from an ad-hoc script that was never committed --
`run/prereg_answer_unsupported.txt` is its FROZEN OUTPUT for one 92-item
held-out slice, but the algorithm that produced it lived only in a
scratch analysis, not in this repository. That is defect 46 recurring in
this project's own work: a load-bearing number with no re-runnable
instrument. This module is the instrument, committed, so the number can
finally be checked rather than merely quoted.

**Pure**, per `items.core`'s own discipline: imports only `items.core`
(for `ItemMeta`) and `fireassay.text` (for `tokenize` -- the one
tokenizer every text-matching component in this codebase shares, per
that module's own docstring: "Never write a tokenizing regex inline
anywhere else in this codebase"). `test_items_no_store_import.py` walks
this module's import graph and asserts it never reaches
`fireassay.store` or `fireassay.llm`.

**The `threshold=0.34` default was chosen arbitrarily, before any
labelled item was looked at, and must not be tuned against
`run/calibration.jsonl`.** That calibration set is this project's only
external referent; tuning a threshold against the very labels a claim is
later checked against would stop the result being a measurement and make
it a fit instead -- the same discipline `items.answerability`'s module
docstring states for its own prompt (`answer_unsupported`'s original
82-item held-out precision survived only because the rule was frozen
before those labels existed; see `run/prereg_answer_unsupported.txt` and
`tools/evidence.py`'s `detector.prereg_answer_unsupported` claim).
"""

from __future__ import annotations

from fireassay.items.core import ItemMeta
from fireassay.text import tokenize

#: Below this many distinct reference-answer tokens, an overlap fraction
#: is not a meaningful signal -- a one- or two-word answer's tokens are
#: either both present (100%) or not (0%, or 50%), with no room for a
#: threshold to discriminate anything. Returning `False` (not flagged)
#: below this floor is the same "cannot measure, do not fabricate"
#: reasoning `items.core.Estimate`'s `no_denominator` verdict applies
#: elsewhere in `items/`.
_MIN_ANSWER_TOKENS = 3


def answer_unsupported(meta: ItemMeta, *, threshold: float = 0.34) -> bool:
    """True when fewer than `threshold` of `meta.reference_answer`'s
    distinct tokens appear anywhere in `meta.evidence_quote` -- the
    no-model baseline every model-backed detector in this project must
    beat.

    Exact rule, as run: tokenize `reference_answer` and `evidence_quote`
    with `fireassay.text.tokenize`, take distinct token sets, return
    `False` if the answer has fewer than `_MIN_ANSWER_TOKENS` (3)
    distinct tokens, else `len(answer & evidence) / len(answer) <
    threshold`.

    `meta.reference_answer`/`meta.evidence_quote` missing (`None`, since
    both are optional on `ItemMeta`) tokenize as empty rather than
    raising -- an empty answer has 0 distinct tokens, below
    `_MIN_ANSWER_TOKENS`, so `False`; an empty evidence quote against a
    real answer has 0 overlap, so `True` (never a fabricated `False`
    from missing evidence)."""
    answer_tokens = set(tokenize(meta.reference_answer or ""))
    if len(answer_tokens) < _MIN_ANSWER_TOKENS:
        return False
    evidence_tokens = set(tokenize(meta.evidence_quote or ""))
    overlap_fraction = len(answer_tokens & evidence_tokens) / len(answer_tokens)
    return overlap_fraction < threshold
