"""Shared M2 test helpers: fixture loading and a reusable `ControlContext`
builder, so every `test_controls_*.py` file doesn't reimplement the same
BM25-over-fixtures wiring `test_runner_e2e.py` already established for M1.

Not itself a `test_*.py` module — pytest does not collect it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from fireassay.controls.base import ControlContext, build_control_context
from fireassay.controls.expected import Band, load_expected_bands
from fireassay.models import EvidenceSpan, Question, Suite
from fireassay.score.abstention import AbstentionScorer
from fireassay.score.base import Scorer, ScoringContext
from fireassay.score.cost import CostScorer
from fireassay.score.latency import LatencyScorer
from fireassay.score.policy import PolicyScorer
from fireassay.score.retrieval import RetrievalScorer
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Doc, chunk_corpus, load_corpus

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_PATH = Path(__file__).parent.parent / "src" / "fireassay" / "controls" / "expected.yaml"


def load_fixture_docs() -> list[Doc]:
    return load_corpus(FIXTURES_DIR / "corpus.jsonl")


def load_fixture_questions() -> list[Question]:
    questions: list[Question] = []
    with open(FIXTURES_DIR / "questions.jsonl", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            spans = tuple(EvidenceSpan(**span) for span in obj.get("evidence_spans", []))
            questions.append(
                Question(
                    text=obj["text"],
                    qtype=obj["qtype"],
                    difficulty=obj["difficulty"],
                    reference_answer=obj.get("reference_answer"),
                    evidence_spans=spans,
                    provenance=obj["provenance"],
                    generator=obj.get("generator"),
                    source_doc_id=obj.get("source_doc_id"),
                )
            )
    return questions


def bm25_system_factory(config: Mapping[str, object], docs: Sequence[Doc]) -> System:
    chunks = list(chunk_corpus(docs, size=800, overlap=100))
    top_k = int(config.get("top_k", 5))  # type: ignore[arg-type]
    return BM25System(chunks, top_k=top_k)


def scoring_ctx_factory(config: Mapping[str, object]) -> ScoringContext:
    top_k = int(config.get("top_k", 5))  # type: ignore[arg-type]
    return ScoringContext(top_k=top_k)


def standard_scorers() -> list[Scorer]:
    return [RetrievalScorer(), LatencyScorer(), CostScorer(), PolicyScorer(), AbstentionScorer()]


def load_expected() -> dict[str, dict[str, Band]]:
    return load_expected_bands(EXPECTED_PATH)


def setup_suite(store: Store, name: str = "kb", version: str = "1.0.0") -> tuple[Suite, list[Question]]:
    questions = load_fixture_questions()
    store.put_questions(questions)
    suite = store.freeze_suite(name, version, [q.id for q in questions])
    return suite, questions


def make_control_context(
    store: Store,
    *,
    top_k: int = 5,
    system_factory: Callable[[Mapping[str, object], Sequence[Doc]], System] | None = None,
    scorers: list[Scorer] | None = None,
) -> ControlContext:
    """Build a `ControlContext` over the shared fixture corpus/questions,
    with the real BM25 system factory and all five M1 scorers unless
    overridden — override either to construct a deliberately broken
    harness for a "must fire on broken" test.
    """
    suite, questions = setup_suite(store)
    docs = load_fixture_docs()
    base_config = store.put_config({"top_k": top_k})
    expected = load_expected()
    factory = system_factory if system_factory is not None else bm25_system_factory
    return build_control_context(
        store,
        suite,
        base_config,
        questions,
        docs,
        factory,
        scoring_ctx_factory,
        scorers if scorers is not None else standard_scorers(),
        expected,
    )
