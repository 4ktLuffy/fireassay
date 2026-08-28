"""Post-schema-validation content check: does this candidate's `text` read
as a question at all, or did the model obey an imperative sentence from the
chunk instead of asking about it?

gov.uk service pages are wall-to-wall imperatives ("Start now", "You must
apply within 28 days", "Sign in to continue", "Do not send your original
documents"). A generator handed that chunk as raw material can — the
documented failure mode this guards against — produce output that *follows*
the page rather than describing it, echoing an instruction back as if it
were the generated question. `prompts.py`'s data boundary is the first line
of defence; this is the second, cheap, deterministic backstop: reject
anything that is not plausibly interrogative before it is ever offered an
evidence span, under `NOT_A_QUESTION`.

This is the same hazard class as `docs/SPEC.md` §10.4's `corpus_injection`
control, arriving one stage earlier — at generation, not at judging.
"""

from __future__ import annotations

from fireassay.text import tokenize

#: Verbs gov.uk service pages commonly issue as bare imperatives. Not an
#: attempt at real part-of-speech tagging (no new dependency for that) —
#: a deliberately generous, documented word list is sufficient here: this
#: check only needs to catch text that unambiguously reads as an
#: instruction being followed, not to classify grammar in general.
_IMPERATIVE_VERBS = frozenset(
    {
        "start", "sign", "apply", "submit", "click", "go", "visit", "call", "complete",
        "fill", "send", "bring", "contact", "check", "read", "do", "make", "keep", "use",
        "enter", "select", "choose", "download", "upload", "register", "renew", "pay",
        "confirm", "wait", "stop", "avoid", "ensure", "note", "remember", "provide",
        "attach", "return", "print", "book", "cancel", "update", "continue", "follow",
    }
)


def is_imperative(text: str) -> bool:
    """`True` when `text` has no question mark **and** its first token is a
    bare imperative verb — the two-part test M3-SPEC.md's coordinator
    addition specifies. A genuine question with no `?` (rare, but not
    impossible for a slightly malformed model response) is not flagged
    unless it *also* opens with a command verb, so this stays a narrow,
    high-precision check rather than a broad "must contain a question
    mark" rule that would reject more than intended.
    """
    stripped = text.strip()
    if "?" in stripped:
        return False
    tokens = tokenize(stripped)
    if not tokens:
        return False
    return tokens[0] in _IMPERATIVE_VERBS
