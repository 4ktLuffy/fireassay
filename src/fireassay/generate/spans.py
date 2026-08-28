"""Evidence-span resolution (M3-SPEC.md §2) — the failure mode the spike
found, made structural.

A candidate's `quote` must locate **exactly once** in its source document's
full text, by exact match **under whitespace normalisation**. There is no
fuzzy matching anywhere in this module, deliberately: a span that is merely "close" to the
right place is not evidence, it is a guess wearing evidence's clothing, and
at 3,000 candidates a systematic near-miss would silently seed a golden set
with wrong ground truth that looks exactly like right ground truth.

Two rejections, never approximated:

- the quote does not appear in the document at all -> `QUOTE_NOT_FOUND`
- the quote appears more than once -> `QUOTE_AMBIGUOUS` (which occurrence
  did the model mean? nothing in the candidate says, so guessing one would
  silently point ground truth at the wrong passage)

Resolution is against the **document**, not the chunk the model was shown:
the model is prompted with one chunk, but the same phrase can recur
elsewhere in the same document (or, if it does not recur, `char_start`/
`char_end` computed against the chunk's local offset would be wrong once
translated to document coordinates) — resolving against the full document
text is both the correct coordinate space and the only way `QUOTE_AMBIGUOUS`
can ever be detected at all.

Whitespace normalisation, and why it is not fuzzy matching
----------------------------------------------------------

Measured against qwen2.5:7b on the gov.uk corpus: **4 of 8 candidates were
rejected as `QUOTE_NOT_FOUND` purely because the model collapsed `\n\n` to
`\n` when copying a multi-line quote.** The characters were otherwise
identical; the divergence was always at a blank line.

That is not a paraphrase and treating it as one was costing half the yield.
Worse, the loss was **not random**: it selectively discarded multi-line
quotes -- lists, tables, structured guidance -- so the surviving candidates
would have skewed toward single-sentence facts and biased the golden set
toward easy questions.

So matching runs over both strings with runs of whitespace collapsed to a
single space, and offsets are mapped back to the original document. This is
still *exact* matching: it is exact on content, under a normalisation that
preserves every non-whitespace character in order. It is categorically
different from similarity or fuzzy matching, which would accept text the
model actually altered. Nothing here accepts a quote whose characters
differ.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

SpanRejectionReason = Literal["QUOTE_NOT_FOUND", "QUOTE_AMBIGUOUS"]


class SpanRejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: SpanRejectionReason
    detail: str


class ResolvedSpan(NamedTuple):
    char_start: int
    char_end: int


def resolve_span(quote: str, doc_text: str) -> ResolvedSpan | SpanRejection:
    """Locate `quote` in `doc_text` by exact substring match.

    Returns a `ResolvedSpan` (`char_start`, `char_end`, half-open) when
    `quote` appears **exactly once**; a `SpanRejection` otherwise. An empty
    `quote` is rejected as `QUOTE_NOT_FOUND` explicitly, rather than falling
    through to `str.count`'s degenerate "an empty string matches everywhere"
    behaviour, which would otherwise misreport it as `QUOTE_AMBIGUOUS`.
    """
    if not quote.strip():
        return SpanRejection(reason="QUOTE_NOT_FOUND", detail="quote is empty")

    norm_doc, index_map = _normalise_with_index_map(doc_text)
    norm_quote, _ = _normalise_with_index_map(quote)
    if not norm_quote:
        return SpanRejection(reason="QUOTE_NOT_FOUND", detail="quote is empty")

    count = norm_doc.count(norm_quote)
    if count == 0:
        # A model quoting from mid-document routinely capitalises the first
        # letter, because it is starting a sentence in its own output:
        # "You'll have an immigration status document" for a passage that
        # reads "you'll ...". Retry ONCE with only the first character's
        # case flipped -- still an exact lookup, and deliberately not
        # case-insensitive matching, because case carries meaning elsewhere
        # ("US" vs "us") and a blanket fold could anchor a span on a
        # genuinely different passage.
        flipped = _flip_first_char_case(norm_quote)
        if flipped != norm_quote and norm_doc.count(flipped) == 1:
            norm_quote = flipped
            count = 1
    if count == 0:
        return SpanRejection(
            reason="QUOTE_NOT_FOUND",
            detail=f"quote ({len(quote)} chars) does not appear in the source document",
        )
    if count > 1:
        return SpanRejection(
            reason="QUOTE_AMBIGUOUS",
            detail=f"quote appears {count} times in the source document; no way to tell which was meant",
        )
    start_norm = norm_doc.index(norm_quote)
    end_norm = start_norm + len(norm_quote) - 1
    return ResolvedSpan(index_map[start_norm], index_map[end_norm] + 1)


def _normalise_with_index_map(text: str) -> tuple[str, list[int]]:
    """Collapse runs of whitespace to a single space, returning the
    normalised text and a map from each normalised index back to the index
    of the character it came from in `text`.

    The index map is what keeps this honest: offsets stored on an evidence
    span always refer to the ORIGINAL document, so a curator reading the
    span sees the real passage including its real line breaks. Only the
    matching is normalised, never the stored coordinates.
    """
    out: list[str] = []
    index_map: list[int] = []
    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                out.append(" ")
                index_map.append(i)
                in_ws = True
            continue
        in_ws = False
        out.append(ch)
        index_map.append(i)
    # a leading space carries no content; strip it so a quote that began
    # mid-whitespace still anchors on its first real character
    while out and out[0] == " ":
        out.pop(0)
        index_map.pop(0)
    while out and out[-1] == " ":
        out.pop()
        index_map.pop()
    return "".join(out), index_map


def _flip_first_char_case(text: str) -> str:
    """`text` with only its first character's case inverted."""
    if not text:
        return text
    head = text[0]
    flipped = head.lower() if head.isupper() else head.upper()
    return flipped + text[1:]
