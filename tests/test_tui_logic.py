"""M3b-SPEC.md Part 2: the default-accept rubric is exactly the six
"agrees" values; the reason-key map is a bijection onto the nine codes;
truncation marks how much was hidden and never silently drops text.

Everything here is a plain function -- no `textual` import anywhere in
this file, matching the spec's own instruction that a Textual app cannot
be exercised headlessly in this environment, which is exactly why the
decisions worth testing were pulled out into `curate.tui_logic` in the
first place.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fireassay.curate.tui_logic import (
    DEFAULT_ACCEPT_RUBRIC,
    REASON_KEYS,
    REASON_LABELS,
    Truncated,
    all_reject_reasons,
    build_accept_or_edit_decision,
    build_reject_decision,
    format_elapsed_minutes,
    format_progress,
    render_truncated,
    truncate_for_display,
)

# -- DEFAULT_ACCEPT_RUBRIC: the fast-path "everything agrees" rubric -------


def test_default_accept_rubric_is_exactly_the_six_agrees_values() -> None:
    """Verbatim from M3b-SPEC.md Part 2: getting even one field wrong here
    turns the one-keystroke fast path into a lie about what was reviewed."""
    assert DEFAULT_ACCEPT_RUBRIC.answerable_from_kb == "yes"
    assert DEFAULT_ACCEPT_RUBRIC.self_contained is True
    assert DEFAULT_ACCEPT_RUBRIC.reference_answer_correct == "yes"
    assert DEFAULT_ACCEPT_RUBRIC.evidence_sufficient is True
    assert DEFAULT_ACCEPT_RUBRIC.difficulty_agrees is True
    assert DEFAULT_ACCEPT_RUBRIC.qtype_agrees is True
    # And nothing reassigned -- a fast accept never overrides the
    # generator's proposed qtype/difficulty.
    assert DEFAULT_ACCEPT_RUBRIC.difficulty_reassigned is None
    assert DEFAULT_ACCEPT_RUBRIC.qtype_reassigned is None


def test_default_accept_rubric_is_frozen() -> None:
    """A single shared instance is reused as every fast accept/reject's
    rubric (see `build_accept_or_edit_decision`/`build_reject_decision`'s
    defaults) -- it must be immutable so one curation action can never
    leak a mutation into another's decision."""
    with pytest.raises(ValidationError):
        DEFAULT_ACCEPT_RUBRIC.self_contained = False  # type: ignore[misc]


# -- REASON_KEYS: bijection onto the nine RejectReason codes ---------------


def test_reason_keys_is_a_bijection_onto_all_nine_reject_reasons() -> None:
    codes = all_reject_reasons()
    assert len(codes) == 9
    assert set(REASON_KEYS.values()) == codes, "every code must have exactly one key"
    assert len(REASON_KEYS) == len(set(REASON_KEYS.values())), "no two keys may map to the same code"
    assert len(REASON_KEYS.keys()) == len(set(REASON_KEYS.keys())), "keys themselves must be unique"


def test_reason_keys_are_all_single_characters() -> None:
    for key in REASON_KEYS:
        assert len(key) == 1, f"reason picker key {key!r} is not a single keystroke"


def test_reason_labels_is_the_exact_reverse_of_reason_keys() -> None:
    for key, reason in REASON_KEYS.items():
        assert REASON_LABELS[reason] == key


# -- decision builders -------------------------------------------------------


def test_build_accept_decision_uses_the_default_rubric_and_no_edits() -> None:
    decision = build_accept_or_edit_decision("cand1", "alice", 4200)
    assert decision.decision == "accept"
    assert decision.rubric == DEFAULT_ACCEPT_RUBRIC
    assert decision.edited_text is None
    assert decision.edited_answer is None
    assert decision.reject_reason is None
    assert decision.duration_ms == 4200


def test_build_edit_decision_is_accept_shaped_with_corrections() -> None:
    decision = build_accept_or_edit_decision(
        "cand1", "alice", 9000, edited_text="corrected question?", edited_answer="corrected answer"
    )
    assert decision.decision == "edit"
    assert decision.edited_text == "corrected question?"
    assert decision.edited_answer == "corrected answer"


def test_build_edit_decision_fires_on_answer_only_edit() -> None:
    decision = build_accept_or_edit_decision("cand1", "alice", 9000, edited_answer="corrected answer only")
    assert decision.decision == "edit"
    assert decision.edited_text is None


def test_build_reject_decision_carries_the_reason_and_default_rubric() -> None:
    decision = build_reject_decision("cand1", "alice", "TRIVIAL", 3000)
    assert decision.decision == "reject"
    assert decision.reject_reason == "TRIVIAL"
    assert decision.rubric == DEFAULT_ACCEPT_RUBRIC


def test_build_reject_decision_accepts_every_reason_key_target() -> None:
    """Every reason `REASON_KEYS` can pick must round-trip through a real
    `Decision` without pydantic rejecting it -- a stale reason string in
    `tui_logic.py` would otherwise fail silently until a curator hit that
    exact key."""
    for reason in REASON_KEYS.values():
        decision = build_reject_decision("cand1", "alice", reason, 1000)
        assert decision.reject_reason == reason


# -- truncation: marks how much was hidden, never silently drops text ------


def test_truncate_for_display_short_text_is_unmodified() -> None:
    t = truncate_for_display("short text", max_chars=100)
    assert t.display == "short text"
    assert t.full == "short text"
    assert t.hidden_chars == 0
    assert t.is_truncated is False


def test_truncate_for_display_exactly_at_the_limit_is_not_truncated() -> None:
    text = "x" * 50
    t = truncate_for_display(text, max_chars=50)
    assert t.hidden_chars == 0
    assert t.is_truncated is False


def test_truncate_for_display_cuts_and_reports_the_exact_hidden_count() -> None:
    text = "a" * 312 + "b" * 50  # 362 chars total, matching the spec's own example
    t = truncate_for_display(text, max_chars=50)
    assert t.display == "a" * 50
    assert t.hidden_chars == 312
    assert t.is_truncated is True


def test_truncate_for_display_never_drops_the_full_text() -> None:
    original = "the full evidence text, quite a bit longer than the display slot allows here"
    t = truncate_for_display(original, max_chars=10)
    assert t.full == original  # the complete original is always retrievable


def test_truncate_for_display_rejects_a_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        truncate_for_display("anything", max_chars=0)
    with pytest.raises(ValueError, match="max_chars"):
        truncate_for_display("anything", max_chars=-5)


def test_render_truncated_marks_the_hidden_count_explicitly() -> None:
    t = Truncated(display="abc", full="abcdef", hidden_chars=3)
    rendered = render_truncated(t)
    assert "+3 chars" in rendered
    assert rendered.startswith("abc")


def test_render_truncated_is_bare_when_nothing_was_hidden() -> None:
    t = Truncated(display="abc", full="abc", hidden_chars=0)
    assert render_truncated(t) == "abc"
    assert "chars" not in render_truncated(t)


# -- small header-formatting helpers ----------------------------------------


def test_format_progress() -> None:
    assert format_progress(127, 1000) == "127/1000"
    assert format_progress(0, 0) == "0/0"


def test_format_elapsed_minutes_floors_to_whole_minutes() -> None:
    assert format_elapsed_minutes(0) == "0m"
    assert format_elapsed_minutes(59) == "0m"
    assert format_elapsed_minutes(60) == "1m"
    assert format_elapsed_minutes(23 * 60 + 59) == "23m"
