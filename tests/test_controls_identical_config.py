from __future__ import annotations

from collections.abc import Mapping, Sequence

from fireassay.controls.identical_config import IdenticalConfigControl
from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.corpus import Doc
from helpers import make_control_context


def test_identical_config_zero_deltas_on_deterministic_system(store: Store) -> None:
    ctx = make_control_context(store)
    outcome = IdenticalConfigControl().run(ctx)
    assert outcome.kind == "identical_config"
    assert outcome.status == "PASSED"
    assert outcome.twin_ok is True
    assert outcome.cause_assertions["metric_sets_match"] is True
    assert outcome.cause_assertions["all_deltas_zero"] is True
    assert outcome.observed["max_abs_delta"] == 0.0


class _NondeterministicSystem:
    """A deliberately nondeterministic stub, made reliably (not
    probabilistically) nondeterministic for test purposes: the *first*
    system built for "the same config" always retrieves `docs[0]` for
    every question; the *second* always retrieves `docs[1]`. Two
    constructions of the same config therefore disagree on every
    gold-bearing question's hit/miss, deterministically -- standing in for
    a real nondeterminism bug (e.g. an unseeded tie-break, a race against
    an external index) without relying on `random` and the flakiness that
    would come with it. Deliberately does not rely on `latency.total_ms`
    varying, since `identical_config` excludes wall-clock latency from its
    comparison by design (see controls/identical_config.py's docstring).
    """

    _construction_count = 0

    def __init__(self, docs: Sequence[Doc]) -> None:
        self._variant = _NondeterministicSystem._construction_count % 2
        _NondeterministicSystem._construction_count += 1
        self.name = "nondeterministic"
        self.generates_answers = False
        self._docs = list(docs)

    def answer(self, question: Question) -> SystemOutput:
        doc = self._docs[self._variant]
        retrieved = (
            RetrievedChunk(doc_id=doc.doc_id, chunk_id=f"{doc.doc_id}#0000", score=1.0, rank=1,
                            char_start=0, char_end=min(50, len(doc.text))),
        )
        return SystemOutput(answer=None, abstained=True, retrieved=retrieved, latency_ms={"total": 1.0})


def _nondeterministic_factory(config: Mapping[str, object], docs: Sequence[Doc]) -> System:
    return _NondeterministicSystem(docs)


def test_identical_config_fails_on_nondeterministic_stub(store: Store) -> None:
    ctx = make_control_context(store, system_factory=_nondeterministic_factory)
    outcome = IdenticalConfigControl().run(ctx)
    assert outcome.status == "FAILED"
    assert outcome.cause_assertions["all_deltas_zero"] is False
    assert outcome.observed["max_abs_delta"] > 0.0
