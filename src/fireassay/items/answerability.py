"""Gold answerability -- eval-of-evals signal 1
(`~/ObitosBrain/handbook/eval-of-evals.md` §6): *with the gold evidence, can
the expected answer be derived? Require identifying the supporting span.
Failure = the triple is broken.* The crude token-overlap proxy for this
(`answer_unsupported`, <34% of answer tokens present in the evidence)
already scores 80.0% held-out precision at 5.66x lift on 92 items it had
never seen -- that is the number a model-backed check here has to beat.

**Pure, per §7b.** This module must not import `fireassay.llm`,
`fireassay.store`, or anything else outside `items/` --
`test_items_no_store_import.py` enforces this by walking the module's
static import graph. The model arrives as an injected callable, `Ask`, so a
user with their own model, their own items and no fireassay store can call
`score_answerability` directly and get verdicts -- this is what makes
signal 1 standalone.

## The prompt is pinned, not tuned

`build_prompt` is written from the handbook §6 wording above, deliberately
**before** looking at any labelled item, and is scored exactly once against
the 176 human labels in `run/calibration.jsonl` (see
`tools/measure_answerability.py`). If that number disappoints, the honest
next step is a fresh labelled sample, not editing this prompt and
re-scoring against the same 176 -- tuning against the only external
referent this project has would stop the result being a measurement at
all (`answer_unsupported`'s 80% survived only because its predictions were
frozen before the labels existed).

## Greedy last-match verdict parsing

`EVIDENCE` is gov.uk corpus text, passed into the prompt verbatim -- a
document containing the literal string `VERDICT: SUPPORTED` must not be
able to inject a verdict. `parse_verdict` finds the verdict with a
**greedy last-match** regex (the same defence Inspect AI / UK AISI uses for
`GRADE: C|P|I`, handbook §9): a leading `.*` under `DOTALL`, greedy, backs
off only as far as it must to find a match, which lands it on the
*rightmost* `VERDICT: ...` in the completion -- so the model's own final
answer always wins over anything quoted earlier from the evidence. The
same technique is applied to the `SPAN:` line for the same reason.

A completion with no parseable `VERDICT:` line is `unclear`, never a guess
and never an exception -- `raw` is recorded on every verdict regardless, so
any result can be re-derived without re-calling the model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from fireassay.items.core import ItemMeta

#: Prompt in, raw completion out -- the seam that keeps this module pure
#: (see the module docstring). A caller supplies whatever calls their model
#: of choice; this module never imports one.
Ask = Callable[[str], str]

Verdict = Literal["supported", "not_supported", "unclear"]

_VERDICT_VALUES: dict[str, Verdict] = {
    "supported": "supported",
    "not_supported": "not_supported",
    "unclear": "unclear",
}

# Greedy last-match: a leading `.*` under DOTALL is greedy by default, so it
# consumes the whole string first and backtracks only as far as it must to
# find a match -- landing on the LAST occurrence of `VERDICT: ...` in the
# completion, not the first. See the module docstring's "Greedy last-match"
# section.
_VERDICT_PATTERN = re.compile(
    r".*VERDICT:[ \t]*(SUPPORTED|NOT_SUPPORTED|UNCLEAR)", re.IGNORECASE | re.DOTALL
)

# Same greedy-last-match technique, applied to the `SPAN:` line.
_SPAN_PATTERN = re.compile(
    r".*^[ \t]*SPAN:[ \t]*(.*?)[ \t]*$", re.IGNORECASE | re.DOTALL | re.MULTILINE
)


class AnswerabilityVerdict(BaseModel):
    """One item's gold-answerability judgement. `raw` is always the
    complete, unmodified completion `ask` returned -- kept even when
    `verdict == "unclear"` because it was unparseable, so the failure is
    re-diagnosable without re-calling the model."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    verdict: Verdict
    supporting_span: str | None
    raw: str


def build_prompt(item: ItemMeta) -> str:
    """The gold-answerability judge prompt (handbook §6, signal 1) --
    pinned exactly as specified in this module's docstring, not tuned
    against any labelled item. `item.question`/`evidence_quote`/
    `reference_answer` are embedded verbatim; a `None` field renders as an
    empty string rather than raising, since `item_id` is `ItemMeta`'s only
    required field."""
    question = item.question or ""
    evidence = item.evidence_quote or ""
    reference_answer = item.reference_answer or ""
    return (
        "You are judging whether a REFERENCE ANSWER can be derived from a piece of "
        "EVIDENCE, for a given QUESTION. Judge ONLY whether the EVIDENCE supports the "
        "REFERENCE ANSWER. Do NOT judge whether the REFERENCE ANSWER is true in the real "
        "world -- that is out of scope.\n"
        "\n"
        f"QUESTION:\n{question}\n"
        "\n"
        f"EVIDENCE:\n{evidence}\n"
        "\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n"
        "\n"
        "On a line starting with `SPAN:`, quote the exact span of EVIDENCE that supports "
        "the REFERENCE ANSWER. If no such span exists, write exactly `SPAN: NONE`.\n"
        "\n"
        "Then give your verdict:\n"
        "- SUPPORTED: the REFERENCE ANSWER can be derived from the EVIDENCE.\n"
        "- NOT_SUPPORTED: the EVIDENCE does not support the REFERENCE ANSWER -- no such span "
        "exists.\n"
        "- UNCLEAR: use this when the EVIDENCE is truncated, ambiguous, or the QUESTION "
        "itself is incoherent. Do not force SUPPORTED or NOT_SUPPORTED when you genuinely "
        "cannot tell.\n"
        "\n"
        "The LAST LINE of your response must be exactly one of:\n"
        "VERDICT: SUPPORTED\n"
        "VERDICT: NOT_SUPPORTED\n"
        "VERDICT: UNCLEAR\n"
    )


def parse_verdict(item_id: str, raw: str) -> AnswerabilityVerdict:
    """Parse `raw` (a completion from `ask`) into an `AnswerabilityVerdict`
    -- see the module docstring's "Greedy last-match verdict parsing"
    section. No parseable `VERDICT:` line yields `verdict="unclear"`, never
    an exception and never a guess. `supporting_span` is the text after the
    last `SPAN:` line, or `None` when that line is missing or reads
    (case-insensitively) `NONE`."""
    verdict: Verdict = "unclear"
    verdict_match = _VERDICT_PATTERN.search(raw)
    if verdict_match is not None:
        verdict = _VERDICT_VALUES[verdict_match.group(1).lower()]

    supporting_span: str | None = None
    span_match = _SPAN_PATTERN.search(raw)
    if span_match is not None:
        span_text = span_match.group(1).strip()
        if span_text and span_text.lower() != "none":
            supporting_span = span_text

    return AnswerabilityVerdict(
        item_id=item_id, verdict=verdict, supporting_span=supporting_span, raw=raw
    )


def score_answerability(items: Sequence[ItemMeta], ask: Ask) -> list[AnswerabilityVerdict]:
    """Run the gold-answerability check over `items`, calling `ask` exactly
    once per item and preserving `items`' order in the returned list."""
    return [parse_verdict(item.item_id, ask(build_prompt(item))) for item in items]
