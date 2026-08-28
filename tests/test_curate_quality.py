"""Coordinator addition B: autopilot and speeding/fatigue detection in
`curate report`.

"a decision sequence with a 20-long identical run is flagged; a varied
sequence is not; a session whose durations halve across deciles is
flagged; a steady one is not."
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fireassay.curate.models import Decision, RubricVerdict
from fireassay.curate.quality import (
    Session,
    decile_medians,
    detect_sessions,
    is_autopilot,
    is_speeding,
    longest_identical_run,
)

_RUBRIC = RubricVerdict(
    answerable_from_kb="yes",
    self_contained=True,
    reference_answer_correct="yes",
    evidence_sufficient=True,
    difficulty_agrees=True,
    qtype_agrees=True,
)

_START = datetime(2026, 1, 1, 9, 0, 0, tzinfo=UTC)


def _ts(seconds_offset: int) -> str:
    return (_START + timedelta(seconds=seconds_offset)).isoformat()


def _decision(curator_id: str, decision: str, duration_ms: int, index: int) -> Decision:
    return Decision(
        candidate_id=f"c{index}",
        curator_id=curator_id,
        decision=decision,  # type: ignore[arg-type]
        reject_reason="TRIVIAL" if decision == "reject" else None,  # type: ignore[arg-type]
        rubric=_RUBRIC,
        duration_ms=duration_ms,
        decided_at=_ts(index * 10),
    )


# -- autopilot ------------------------------------------------------------


def test_longest_identical_run_hand_computed() -> None:
    assert longest_identical_run(["a", "a", "a", "b", "a", "a"]) == 3
    assert longest_identical_run([]) == 0
    assert longest_identical_run(["a"]) == 1


def test_autopilot_flags_a_20_long_identical_run() -> None:
    decisions = tuple(_decision("alice", "accept", 3000, i) for i in range(20))
    session = Session(curator_id="alice", decisions=decisions)
    assert is_autopilot(session, threshold=15) is True


def test_autopilot_silent_on_a_varied_sequence() -> None:
    pattern = ["accept", "reject", "edit", "accept", "reject"]
    decisions = tuple(_decision("alice", pattern[i % len(pattern)], 3000, i) for i in range(20))
    session = Session(curator_id="alice", decisions=decisions)
    assert is_autopilot(session, threshold=15) is False


def test_autopilot_boundary_run_exactly_at_threshold_is_flagged() -> None:
    decisions = tuple(_decision("alice", "accept", 3000, i) for i in range(15))
    session = Session(curator_id="alice", decisions=decisions)
    assert is_autopilot(session, threshold=15) is True


def test_autopilot_run_one_below_threshold_is_silent() -> None:
    decisions = tuple(_decision("alice", "accept", 3000, i) for i in range(14))
    session = Session(curator_id="alice", decisions=decisions)
    assert is_autopilot(session, threshold=15) is False


# -- speeding ---------------------------------------------------------------


def test_decile_medians_hand_computed() -> None:
    durations = list(range(1, 11))  # deciles of 1 item each
    medians = decile_medians(durations, n_deciles=10)
    assert medians == [float(x) for x in durations]


def test_speeding_flags_a_session_whose_durations_halve_across_deciles() -> None:
    n = 100
    decisions = tuple(
        _decision("alice", "accept", 10000 if i < n // 2 else 5000, i) for i in range(n)
    )
    session = Session(curator_id="alice", decisions=decisions)
    assert is_speeding(session) is True


def test_speeding_silent_on_a_steady_session() -> None:
    n = 100
    decisions = tuple(_decision("alice", "accept", 5000, i) for i in range(n))
    session = Session(curator_id="alice", decisions=decisions)
    assert is_speeding(session) is False


def test_speeding_silent_with_too_few_decisions_for_a_trend() -> None:
    decisions = tuple(_decision("alice", "accept", 5000, i) for i in range(1))
    session = Session(curator_id="alice", decisions=decisions)
    assert is_speeding(session) is False


# -- session detection --------------------------------------------------


def test_detect_sessions_splits_on_a_large_gap() -> None:
    early = Decision(
        candidate_id="c1", curator_id="alice", decision="accept", rubric=_RUBRIC,
        duration_ms=3000, decided_at=_START.isoformat(),
    )
    late = Decision(
        candidate_id="c2", curator_id="alice", decision="accept", rubric=_RUBRIC,
        duration_ms=3000, decided_at=(_START + timedelta(hours=2)).isoformat(),
    )
    sessions = detect_sessions([early, late], session_gap_minutes=30.0)
    assert len(sessions) == 2
    assert sessions[0].decisions == (early,)
    assert sessions[1].decisions == (late,)


def test_detect_sessions_keeps_close_decisions_in_one_session() -> None:
    decisions = [
        Decision(
            candidate_id=f"c{i}", curator_id="alice", decision="accept", rubric=_RUBRIC,
            duration_ms=3000, decided_at=_ts(i * 60),
        )
        for i in range(5)
    ]
    sessions = detect_sessions(decisions, session_gap_minutes=30.0)
    assert len(sessions) == 1
    assert len(sessions[0].decisions) == 5


def test_detect_sessions_groups_by_curator_independently() -> None:
    alice = Decision(
        candidate_id="c1", curator_id="alice", decision="accept", rubric=_RUBRIC,
        duration_ms=3000, decided_at=_START.isoformat(),
    )
    bob = Decision(
        candidate_id="c2", curator_id="bob", decision="accept", rubric=_RUBRIC,
        duration_ms=3000, decided_at=_START.isoformat(),
    )
    sessions = detect_sessions([alice, bob], session_gap_minutes=30.0)
    curators = {s.curator_id for s in sessions}
    assert curators == {"alice", "bob"}
