"""The `System` protocol: the interface every system under test implements.

Kept deliberately tiny so that adding a new system (a different retriever,
later an LLM-backed one) never requires touching runner.py or the scorers.
`answer` is intentionally synchronous: M1 runs sequentially (concurrency is
M5), so a `System` need not be thread-safe or async-aware.
"""

from __future__ import annotations

from typing import Protocol

from fireassay.models import Question, SystemOutput


class System(Protocol):
    name: str

    #: Whether this system ever generates a natural-language answer, as
    #: opposed to retrieval-only. `runner.run_matrix` copies this onto
    #: `ScoringContext.system_generates_answers` for every question, so
    #: `score.abstention.AbstentionScorer` can tell "this system chose to
    #: abstain" (a real signal) apart from "this system structurally never
    #: answers anything" (not a signal at all, and not something an
    #: abstention metric should report on). `BM25System` sets this `False`.
    generates_answers: bool

    def answer(self, question: Question) -> SystemOutput: ...
