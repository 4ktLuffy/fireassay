from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from fireassay.compare import leaderboard
from fireassay.config import MatrixSpec
from fireassay.models import EvidenceSpan, Question
from fireassay.runner import run_matrix
from fireassay.score.abstention import AbstentionScorer
from fireassay.score.base import Scorer, ScoringContext
from fireassay.score.cost import CostScorer
from fireassay.score.latency import LatencyScorer
from fireassay.score.policy import PolicyScorer
from fireassay.score.retrieval import RetrievalScorer
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, chunk_corpus, corpus_hash, load_corpus

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    with open(path, encoding="utf-8") as f:
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


def test_full_pipeline_import_freeze_run_two_configs_compare(tmp_path: Path) -> None:
    store = Store(tmp_path / "e2e.db")
    store.migrate()

    questions = _load_questions(FIXTURES_DIR / "questions.jsonl")
    assert len(questions) == 20
    store.put_questions(questions)
    store.freeze_suite("kb", "1.0.0", [q.id for q in questions])

    docs = load_corpus(FIXTURES_DIR / "corpus.jsonl")

    def _chunks_for(config: Mapping[str, object]) -> list[Chunk]:
        return list(chunk_corpus(docs, size=800, overlap=100))

    def system_factory(config: Mapping[str, object]) -> System:
        return BM25System(_chunks_for(config), top_k=int(config["top_k"]))  # type: ignore[arg-type]

    def ctx_factory(config: Mapping[str, object]) -> ScoringContext:
        return ScoringContext(top_k=int(config["top_k"]))  # type: ignore[arg-type]

    def corpus_hash_factory() -> str:
        # Zero-arg: corpus_hash no longer depends on chunking config.
        return corpus_hash(docs)

    spec = MatrixSpec(suite="kb@1.0.0", axes={"top_k": ["3", "5"]})
    scorers: list[Scorer] = [
        RetrievalScorer(),
        LatencyScorer(),
        CostScorer(),
        PolicyScorer(),
        AbstentionScorer(),
    ]

    runs = run_matrix(
        store, spec, system_factory, scorers, ctx_factory, corpus_hash_factory=corpus_hash_factory
    )
    assert len(runs) == 2
    assert all(r.status == "complete" for r in runs)
    assert all(r.admissible for r in runs)  # a correct scorer implementation must never violate invariants
    assert len({r.config_hash for r in runs}) == 2  # two genuinely distinct configs

    # Both runs used the same (unchanged) fixture corpus -> same corpus_hash
    # -> no ENV_MISMATCH -> comparable.
    corpus_hashes = {r.env_json["affects_results"]["corpus_hash"] for r in runs}  # type: ignore[index]
    assert len(corpus_hashes) == 1

    board = leaderboard(store, runs)
    assert board.report.comparable is True
    assert len(board.entries) == 2
    assert {e.run_id for e in board.entries} == {r.id for r in runs}

    # Every entry should have at least the always-emitted metrics (latency,
    # cost, retrieval for the 14 answerable questions with gold spans).
    # BM25System is retrieval-only (generates_answers=False), so
    # AbstentionScorer must emit NOTHING for it -- not 0.0/1.0 -- see
    # test_abstention_scorer.py for the dedicated coverage of that rule.
    for entry in board.entries:
        metric_names = {m.metric for m in entry.metrics}
        assert "latency.total_ms" in metric_names
        assert "cost.usd" in metric_names
        assert "abstention.correct" not in metric_names
        assert "abstention.wrongly_abstained" not in metric_names

    # Re-running without rerun=True must be a no-op that returns the same runs.
    runs_again = run_matrix(
        store, spec, system_factory, scorers, ctx_factory, corpus_hash_factory=corpus_hash_factory
    )
    assert {r.id for r in runs_again} == {r.id for r in runs}

    store.close()
