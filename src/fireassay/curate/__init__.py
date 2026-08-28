"""Human curation core (M3-SPEC.md §4) — deterministic, LLM-free.

`curate next`/`curate submit` are the non-interactive core M3b's TUI will
drive: everything here operates on plain data (a queue item in, a decision
out), with no terminal I/O or interactivity of its own, so it is
independently testable now and the TUI will add keystrokes, never logic.

**Honeypots and agreement are different instruments and neither substitutes
for the other**: `agreement.py` (Krippendorff's α) says whether two
curators agree with each other; `honeypots.py` says whether a curator is
right, against a known-bad item planted invisibly in the queue. A report
built on only one of the two could show perfect numbers from two curators
confidently agreeing on the same wrong answer.
"""
