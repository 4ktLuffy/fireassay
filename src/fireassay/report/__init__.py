"""Rendering fireassay results (`rich` terminal tables in M1).

Deliberately separate from compare.py: aggregation (what the numbers are)
and rendering (how they are displayed) are different concerns, so a future
HTML/JSON renderer can reuse `Leaderboard`/`ComparabilityReport` without
importing `rich`.
"""

from fireassay.report.text import render_leaderboard, render_refusal

__all__ = ["render_leaderboard", "render_refusal"]
