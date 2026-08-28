"""M3b-SPEC.md Part 2: the 45-minute banner fires on time; the
identical-verdict run counter fires at 15 and resets on a different
verdict.

Plain functions only -- see `curate/tui_session.py`'s module docstring for
why this is the live, incremental sibling of `curate.quality`'s post-hoc
session report, and `test_tui_logic.py`'s for why nothing here imports
`textual`.
"""

from __future__ import annotations

from fireassay.curate.quality import DEFAULT_AUTOPILOT_RUN_THRESHOLD, RECOMMENDED_MAX_SESSION_MINUTES
from fireassay.curate.tui_session import (
    autopilot_note,
    should_show_break_banner,
    trailing_identical_run,
)

# -- should_show_break_banner: fires at the documented 45-minute threshold -


def test_break_banner_does_not_fire_before_the_threshold() -> None:
    assert should_show_break_banner(RECOMMENDED_MAX_SESSION_MINUTES - 0.1) is False


def test_break_banner_fires_exactly_at_the_threshold() -> None:
    assert should_show_break_banner(RECOMMENDED_MAX_SESSION_MINUTES) is True


def test_break_banner_fires_past_the_threshold() -> None:
    assert should_show_break_banner(RECOMMENDED_MAX_SESSION_MINUTES + 5) is True


def test_break_banner_default_threshold_is_45_minutes() -> None:
    """Pins the actual number M3b-SPEC.md names, not just "some threshold
    imported from elsewhere" -- a silent drift here would be exactly the
    kind of unmeasured claim this project exists to catch."""
    assert RECOMMENDED_MAX_SESSION_MINUTES == 45


def test_break_banner_threshold_is_overridable() -> None:
    assert should_show_break_banner(10, threshold_minutes=5) is True
    assert should_show_break_banner(4, threshold_minutes=5) is False


# -- trailing_identical_run: the live "still going right now" streak -------


def test_trailing_identical_run_empty_is_zero() -> None:
    assert trailing_identical_run([]) == 0


def test_trailing_identical_run_counts_consecutive_repeats_at_the_end() -> None:
    values = ["accept"] * 14
    assert trailing_identical_run(values) == 14


def test_trailing_identical_run_resets_on_a_different_verdict() -> None:
    values = ["accept"] * 20 + ["reject"]
    assert trailing_identical_run(values) == 1


def test_trailing_identical_run_ignores_an_earlier_run_shorter_than_the_trailing_one() -> None:
    values = ["reject"] * 3 + ["accept"] * 5
    assert trailing_identical_run(values) == 5


# -- autopilot_note: fires at 15, resets (via a fresh streak) otherwise ----


def test_autopilot_note_is_none_below_threshold() -> None:
    assert autopilot_note(DEFAULT_AUTOPILOT_RUN_THRESHOLD - 1, "accept") is None


def test_autopilot_note_fires_exactly_at_the_default_threshold_of_15() -> None:
    assert DEFAULT_AUTOPILOT_RUN_THRESHOLD == 15
    note = autopilot_note(15, "accept")
    assert note is not None
    assert note == "15 accepts in a row — still reading?"


def test_autopilot_note_names_the_repeated_verdict() -> None:
    assert autopilot_note(15, "reject") == "15 rejects in a row — still reading?"
    assert autopilot_note(15, "edit") == "15 edits in a row — still reading?"


def test_autopilot_note_resets_when_the_streak_resets() -> None:
    """The TUI recomputes `trailing_identical_run` after every submission
    and feeds the result straight in here -- a differing verdict resets
    the trailing run to 1 (see `trailing_identical_run`'s own test), which
    is well below threshold, so the note disappears on the very next
    non-matching verdict without any separate "reset" call needed."""
    values = ["accept"] * 20 + ["reject"]
    streak = trailing_identical_run(values)
    assert autopilot_note(streak, values[-1]) is None


def test_autopilot_note_threshold_is_overridable() -> None:
    assert autopilot_note(3, "accept", threshold=3) == "3 accepts in a row — still reading?"
    assert autopilot_note(2, "accept", threshold=3) is None
