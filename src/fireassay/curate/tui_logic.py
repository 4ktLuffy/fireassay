"""Every decision the M3b curation TUI makes that is worth testing,
pulled out of the Textual widget classes into plain functions
(M3b-SPEC.md Part 2: "Pull every decision worth testing out of the widget
classes into plain functions and test those").

Nothing here talks to a terminal, a `Store`, or `curate.serve`. `tui.py`
is the only module in this package that imports `textual`; it is a thin
shell over the functions below plus `curate.serve.next_item`/
`submit_decision` — "the TUI adds no logic" (M3b-SPEC.md) means literally
that every branch a curator's keypress can take is decidable by calling
something in this file (or `curate.tui_session`) with plain arguments, in
a way a test can call directly with no live terminal.

**The fast path is one keystroke** (M3b-SPEC.md's single load-bearing
rule): `DEFAULT_ACCEPT_RUBRIC` is the exact "everything agrees" rubric `a`
submits with no further input, and `build_accept_or_edit_decision`/
`build_reject_decision` are what turn a single keypress (plus, for reject,
one more keypress picking a reason) into a fully-formed `curate.models.
Decision` ready for `curate.serve.submit_decision` — the curator is never
required to answer the six rubric questions unless they explicitly drill
in (`d`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

from fireassay.curate.models import Decision, RejectReason, RubricVerdict

#: The rubric a fast `a`ccept submits: every criterion set to its
#: "agrees" value, verbatim from M3b-SPEC.md Part 2. This is the single
#: most load-bearing constant in this module -- getting even one field
#: wrong here turns the one-keystroke fast path into a lie about what the
#: curator actually reviewed.
DEFAULT_ACCEPT_RUBRIC: Final[RubricVerdict] = RubricVerdict(
    answerable_from_kb="yes",
    self_contained=True,
    reference_answer_correct="yes",
    evidence_sufficient=True,
    difficulty_agrees=True,
    qtype_agrees=True,
)

#: Single-keystroke reason picker (M3b-SPEC.md Part 2's table). A bijection
#: onto the nine `RejectReason` codes: every code has exactly one key, and
#: every key maps to exactly one code -- `test_tui_logic.py` pins this
#: directly, since a curator must never have to remember (or guess) a
#: letter that silently does not exist, and two codes must never
#: accidentally share a key.
REASON_KEYS: Final[dict[str, RejectReason]] = {
    "d": "DUPLICATE",
    "n": "NOT_ANSWERABLE",
    "a": "AMBIGUOUS",
    "w": "WRONG_REFERENCE",
    "i": "INSUFFICIENT_EVIDENCE",
    "l": "LEADING_QUESTION",
    "t": "TRIVIAL",
    "o": "OUT_OF_SCOPE",
    "p": "PII_RISK",
}

#: The reverse of `REASON_KEYS`, for rendering "which key produces which
#: reason" on screen (M3b-SPEC.md: "a curator must never have to remember
#: them" -- the picker shows the letters, so this is what the picker
#: renders from).
REASON_LABELS: Final[dict[RejectReason, str]] = {reason: key for key, reason in REASON_KEYS.items()}


def all_reject_reasons() -> frozenset[RejectReason]:
    """Every `RejectReason` code the schema defines, read off the
    `Literal` itself rather than duplicated as a literal tuple here -- so
    if `curate.models.RejectReason` ever gains or loses a code,
    `test_tui_logic.py`'s bijection test fails loudly instead of quietly
    checking against a stale copy."""
    return frozenset(get_args(RejectReason))


def build_accept_or_edit_decision(
    candidate_id: str,
    curator_id: str,
    duration_ms: int,
    *,
    edited_text: str | None = None,
    edited_answer: str | None = None,
    rubric: RubricVerdict = DEFAULT_ACCEPT_RUBRIC,
    notes: str | None = None,
) -> Decision:
    """Build the `Decision` for the `a`ccept and `e`dit keys.

    `e`dit is `a`ccept with corrected text/answer attached (M3b-SPEC.md:
    "edit question text and/or reference answer, then accept") -- the two
    keys differ only in whether `edited_text`/`edited_answer` are given,
    which is exactly what `curate.models.Decision`'s own
    `decision: Literal["accept", "edit", "reject"]` field distinguishes,
    so this one function covers both keys rather than needing two.
    """
    decision: Literal["accept", "edit", "reject"] = (
        "edit" if (edited_text is not None or edited_answer is not None) else "accept"
    )
    return Decision(
        candidate_id=candidate_id,
        curator_id=curator_id,
        decision=decision,
        rubric=rubric,
        edited_text=edited_text,
        edited_answer=edited_answer,
        notes=notes,
        duration_ms=duration_ms,
    )


def build_reject_decision(
    candidate_id: str,
    curator_id: str,
    reason: RejectReason,
    duration_ms: int,
    *,
    rubric: RubricVerdict = DEFAULT_ACCEPT_RUBRIC,
    notes: str | None = None,
) -> Decision:
    """Build the `Decision` for the `r`eject key plus its single-keystroke
    reason.

    `rubric` defaults to `DEFAULT_ACCEPT_RUBRIC` here too, not to some
    "everything disagrees" shape: M3b-SPEC.md's fast reject path is one
    keystroke to reject plus one to name *why* (`reject_reason` carries
    that signal); it never asks the six rubric questions, so there is no
    curator-supplied rubric to attach. `curate.models.Decision` requires a
    `rubric` on every decision regardless of `decision` value (M3-SPEC.md
    §4's schema has no "not applicable" rubric state), so a fast reject
    needs *some* value there -- reusing the same all-agree baseline used
    by the fast accept path adds no fabricated signal beyond what the
    curator actually reviewed (a curator who wants their rubric answers to
    mean something on a reject must drill in via `d` first, which produces
    a genuinely curator-supplied `rubric` passed in here instead of the
    default). This specific choice is called out in the M3b delivery
    notes as one the spec itself did not resolve.
    """
    return Decision(
        candidate_id=candidate_id,
        curator_id=curator_id,
        decision="reject",
        reject_reason=reason,
        rubric=rubric,
        notes=notes,
        duration_ms=duration_ms,
    )


@dataclass(frozen=True)
class Truncated:
    """The result of `truncate_for_display`: `display` is what fits on
    screen, `full` is always the untruncated original (never dropped --
    `v` reads it back for the evidence modal), and `hidden_chars` is
    exactly how much was cut, so the on-screen marker can say precisely
    "+N chars" rather than an unquantified "...".
    """

    display: str
    full: str
    hidden_chars: int

    @property
    def is_truncated(self) -> bool:
        return self.hidden_chars > 0


def truncate_for_display(text: str, max_chars: int) -> Truncated:
    """Truncate `text` to `max_chars` for a fixed-layout display slot,
    never silently dropping anything: `Truncated.full` always carries the
    complete original string, and `Truncated.hidden_chars` always states
    exactly how many characters were cut, whether that number is zero or
    not.

    Raises `ValueError` for a non-positive `max_chars` -- a fixed-layout
    slot with no room at all is a caller bug (a screen-position constant
    set to 0), not a legitimate "hide everything" request.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if len(text) <= max_chars:
        return Truncated(display=text, full=text, hidden_chars=0)
    return Truncated(display=text[:max_chars], full=text, hidden_chars=len(text) - max_chars)


def render_truncated(t: Truncated) -> str:
    """`t.display` with an explicit "how much was hidden" marker appended
    when truncated, exactly what M3b-SPEC.md's mock layout shows
    (`"...[+312 chars, v]"`) -- never a bare ellipsis, which states *that*
    something was cut but not how much."""
    if not t.is_truncated:
        return t.display
    return f"{t.display}... [+{t.hidden_chars} chars, v]"


def format_progress(done: int, total: int) -> str:
    """`"127/1000"` -- the header's progress readout."""
    return f"{done}/{total}"


def format_elapsed_minutes(seconds: float) -> str:
    """`"23m"` -- the header's session-elapsed readout. Whole minutes
    only, floored: a session banner does not need second-level precision,
    and a curator glancing at the header should see a number that only
    changes once a minute, not one that jitters on every keypress."""
    minutes = int(seconds // 60)
    return f"{minutes}m"
