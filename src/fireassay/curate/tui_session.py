"""Live, in-session curator-quality signals for the M3b TUI: the
45-minute break banner and the "still reading?" autopilot note
(M3b-SPEC.md Part 2, "Session awareness"). Plain functions, pulled out of
`tui.py` for the same reason as `tui_logic.py` — a Textual app cannot be
exercised headlessly here, so every branch worth testing must be callable
without one.

This module is deliberately a *live*, incremental sibling of
`curate.quality` (M3's post-hoc curator-quality report), not a duplicate
of it: `curate.quality.detect_sessions`/`is_autopilot`/`is_speeding`
operate on a full, already-persisted `Decision` history after the fact,
computing the *longest* identical run anywhere in a session; the TUI needs
the *trailing* run length after every single submission, updated
incrementally as the curator works, which is a different (cheaper, O(1)
amortised) computation over the same underlying idea. Both share the same
named thresholds (`curate.quality.RECOMMENDED_MAX_SESSION_MINUTES`,
`DEFAULT_AUTOPILOT_RUN_THRESHOLD`) rather than re-defining "45" and "15"
as new magic numbers here.
"""

from __future__ import annotations

from collections.abc import Sequence

from fireassay.curate.quality import (
    DEFAULT_AUTOPILOT_RUN_THRESHOLD,
    RECOMMENDED_MAX_SESSION_MINUTES,
)

__all__ = [
    "RECOMMENDED_MAX_SESSION_MINUTES",
    "DEFAULT_AUTOPILOT_RUN_THRESHOLD",
    "should_show_break_banner",
    "trailing_identical_run",
    "autopilot_note",
]


def should_show_break_banner(
    elapsed_minutes: float, *, threshold_minutes: float = RECOMMENDED_MAX_SESSION_MINUTES
) -> bool:
    """`True` once `elapsed_minutes` reaches `threshold_minutes` (default
    45, the documented fatigue threshold `curate.quality` already uses for
    its own, post-hoc session report). Fires **at** the threshold, not
    only strictly past it, so a curator checking the header at exactly
    45:00 sees the banner rather than needing one more keypress to trip
    it. Never blocks: the caller (`tui.py`) only ever uses this to decide
    whether to show a non-blocking notification, never to refuse input."""
    return elapsed_minutes >= threshold_minutes


def trailing_identical_run(values: Sequence[str]) -> int:
    """The length of the run of identical values at the *end* of `values`
    -- "how long has the current streak been going, right now", not
    "what was the longest streak anywhere in the session"
    (`curate.quality.longest_identical_run` answers that second question,
    over a full post-hoc history; this one is the live counter the TUI
    re-checks after every submission).

    `values` is expected in submission order (oldest first); an empty
    sequence has a trailing run of `0`."""
    if not values:
        return 0
    last = values[-1]
    count = 0
    for v in reversed(values):
        if v != last:
            break
        count += 1
    return count


def autopilot_note(
    streak: int, verdict: str, *, threshold: int = DEFAULT_AUTOPILOT_RUN_THRESHOLD
) -> str | None:
    """The one-line "still reading?" note (M3b-SPEC.md: `"15 accepts in a
    row — still reading?"`), or `None` when `streak` has not reached
    `threshold` yet. `verdict` is the repeated decision value
    (`"accept"`/`"edit"`/`"reject"`) so the note names what is actually
    repeating, not a hard-coded "accepts" that would misdescribe a run of
    identical rejects."""
    if streak < threshold:
        return None
    return f"{streak} {verdict}s in a row — still reading?"
