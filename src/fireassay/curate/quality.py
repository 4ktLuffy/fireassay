"""Curator-quality monitoring, built entirely from fields already recorded
on `Decision`: no new schema, no new instrument, just two documented
degradation signatures made visible.

- **Autopilot**: long uninterrupted runs of identical `decision` values in
  timestamp order — a curator clicking the same button repeatedly without
  actually looking.
- **Speeding**: a downward trend in `duration_ms` over a session, which
  correlates with declining attention.

Both are session-scoped: `detect_sessions` splits one curator's decisions,
in `decided_at` order, wherever the gap since the previous decision exceeds
`session_gap_minutes` — decisions with no large gap between them are
treated as one continuous sitting. Sessions are recommended to stay under
**45 minutes**: the documented threshold beyond which mental fatigue
measurably degrades annotation quality. `curate report` states this
alongside the numbers; nothing here enforces it (there is no gate in M3).

Together with honeypot accuracy per curator/decile (`honeypots.py`),
autopilot runs and the pace trend are a curator-quality monitor built
entirely from data this milestone already records.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from fireassay.curate.models import Decision

DEFAULT_SESSION_GAP_MINUTES = 30.0
DEFAULT_AUTOPILOT_RUN_THRESHOLD = 15
DEFAULT_SPEEDUP_FLAG_PCT = 0.6
RECOMMENDED_MAX_SESSION_MINUTES = 45


@dataclass(frozen=True)
class Session:
    curator_id: str
    decisions: tuple[Decision, ...]  # in decided_at order


def detect_sessions(
    decisions: Sequence[Decision], *, session_gap_minutes: float = DEFAULT_SESSION_GAP_MINUTES
) -> list[Session]:
    """Split `decisions` into per-curator sessions.

    Decisions are grouped by `curator_id`, sorted by `decided_at`, and a
    new session starts whenever the gap since the previous decision
    exceeds `session_gap_minutes` — there is no explicit session id
    anywhere in the schema (M3-SPEC.md's `decision` table has none), so a
    time-gap heuristic over the timestamp fireassay already records
    (`decided_at`) is the only signal available, and the one the spec's
    own "session decile" language implies exists.
    """
    by_curator: dict[str, list[Decision]] = {}
    for d in decisions:
        by_curator.setdefault(d.curator_id, []).append(d)

    sessions: list[Session] = []
    for curator_id, curator_decisions in by_curator.items():
        ordered = sorted(curator_decisions, key=lambda d: d.decided_at)
        current: list[Decision] = []
        prev_time: datetime | None = None
        for d in ordered:
            t = datetime.fromisoformat(d.decided_at)
            if prev_time is not None and (t - prev_time).total_seconds() / 60.0 > session_gap_minutes:
                sessions.append(Session(curator_id, tuple(current)))
                current = []
            current.append(d)
            prev_time = t
        if current:
            sessions.append(Session(curator_id, tuple(current)))
    return sessions


def longest_identical_run(values: Sequence[str]) -> int:
    """Longest run of consecutive identical values in `values`, taken in
    the order given (callers pass an already-time-ordered sequence)."""
    best = 0
    current = 0
    prev: str | None = None
    for v in values:
        current = current + 1 if v == prev else 1
        best = max(best, current)
        prev = v
    return best


def is_autopilot(
    session: Session, *, threshold: int = DEFAULT_AUTOPILOT_RUN_THRESHOLD
) -> bool:
    """`True` when `session` contains a run of `threshold` or more
    consecutive identical `decision` values, in `decided_at` order."""
    return longest_identical_run([d.decision for d in session.decisions]) >= threshold


def decile_medians(durations_ms: Sequence[int], n_deciles: int = 10) -> list[float]:
    """Median `duration_ms` within each of `n_deciles` equal-sized,
    time-ordered chunks of `durations_ms`. Empty deciles (fewer items than
    `n_deciles`) are simply omitted rather than padded — a session with 4
    decisions has 4 non-empty "deciles", not 10 mostly-empty ones."""
    n = len(durations_ms)
    if n == 0:
        return []
    medians: list[float] = []
    for i in range(n_deciles):
        lo = (i * n) // n_deciles
        hi = ((i + 1) * n) // n_deciles
        chunk = durations_ms[lo:hi]
        if chunk:
            medians.append(statistics.median(chunk))
    return medians


def is_speeding(
    session: Session, *, n_deciles: int = 10, speedup_flag_pct: float = DEFAULT_SPEEDUP_FLAG_PCT
) -> bool:
    """`True` when the session's last decile's median `duration_ms` falls
    below `speedup_flag_pct` of its first decile's median — a downward
    pace trend, the documented correlate of declining attention."""
    medians = decile_medians([d.duration_ms for d in session.decisions], n_deciles)
    if len(medians) < 2 or medians[0] <= 0:
        return False
    return medians[-1] < speedup_flag_pct * medians[0]
