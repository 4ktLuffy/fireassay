"""`tools/panel_dense.py`: the recovered `correct` definition, and the
twin check that proves it.

The frozen panel's generator is lost. This suite pins the two claims
`--verify-bm25` rests on, against the real scorer rather than against
prose:

1. `is_correct` is exactly `RetrievalScorer`'s `retrieval.recall@k > 0` at
   `overlap_min_chars=1` — checked by running the actual scorer;
2. scoring one ranking at several `k`s is what building one system per `k`
   does — the optimisation that turns the frozen panel's 172 minutes into
   minutes, and a silent source of wrong cells if it were not true.

Plus: `verify_bm25` must fail on a planted mismatch. A twin check that has
never fired is not a check.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import ModuleType

import pytest

from fireassay.score.base import ScoringContext
from fireassay.score.retrieval import RetrievalScorer
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import chunk_corpus
from helpers import load_fixture_docs, load_fixture_questions, load_tool_module

_TOP_KS = (3, 5, 10)


@pytest.fixture(scope="module")
def panel_dense() -> ModuleType:
    return load_tool_module("panel_dense")


def _items(panel_dense: ModuleType) -> list[object]:
    """One `Item` per fixture question that carries gold spans, built from
    that question's *first* span — the panel's own shape, where each item
    has exactly one evidence span."""
    return [
        panel_dense.Item(
            item_id=q.id,
            question=q.text,
            doc_id=q.evidence_spans[0].doc_id,
            char_start=q.evidence_spans[0].char_start,
            char_end=q.evidence_spans[0].char_end,
        )
        for q in load_fixture_questions()
        if q.evidence_spans
    ]


def test_correct_matches_retrieval_scorer_recall(panel_dense: ModuleType) -> None:
    """The recovered definition, checked against the shipped scorer.

    This is the claim the whole milestone stands on: `correct` is not a
    third notion of relevance invented by a lost script, it is
    `retrieval.recall@k > 0` under the scorer every other number in this
    repo already uses.
    """
    chunks = list(chunk_corpus(load_fixture_docs(), size=800, overlap=100))
    scorer = RetrievalScorer()
    questions = {q.id: q for q in load_fixture_questions() if q.evidence_spans}
    checked = 0
    for k in _TOP_KS:
        system = BM25System(chunks, top_k=k)
        ctx = ScoringContext(top_k=k)
        for item in _items(panel_dense):
            question = questions[item.item_id]
            # One span per item, so recall is 0.0 or 1.0 and the
            # comparison is exact rather than thresholded.
            single_span = question.model_copy(update={"evidence_spans": question.evidence_spans[:1]})
            output = system.answer(single_span)
            recall = next(
                s.value for s in scorer.score(single_span, output, ctx) if s.metric == f"retrieval.recall@{k}"
            )
            assert panel_dense.is_correct(output.retrieved, item, k) is (recall > 0.0)
            checked += 1
    assert checked == len(_TOP_KS) * len(_items(panel_dense))


def test_truncating_one_ranking_equals_separate_systems(panel_dense: ModuleType) -> None:
    """`columns_for_system` scores each item once at `max(top_ks)` and
    slices. If that were not identical to building one system per `k`, the
    regenerated columns would differ from the frozen ones for a reason
    having nothing to do with the definition being recovered."""
    chunks = list(chunk_corpus(load_fixture_docs(), size=800, overlap=100))
    items = _items(panel_dense)
    sliced = panel_dense.columns_for_system(BM25System(chunks, top_k=max(_TOP_KS)), items, _TOP_KS)
    for k in _TOP_KS:
        per_system = panel_dense.columns_for_system(BM25System(chunks, top_k=k), items, (k,))[k]
        assert sliced[k] == per_system


def test_is_correct_requires_overlap_in_the_same_document(panel_dense: ModuleType) -> None:
    """Touching ranges in *different* documents are not a hit, and ranges
    that merely abut (`[0, 10)` and `[10, 20)`) share no character."""
    from fireassay.models import RetrievedChunk

    item = panel_dense.Item(item_id="i", question="q", doc_id="doc01", char_start=10, char_end=20)
    same_doc_abutting = RetrievedChunk(
        doc_id="doc01", chunk_id="doc01#0000", score=1.0, rank=1, char_start=0, char_end=10
    )
    other_doc_overlapping = RetrievedChunk(
        doc_id="doc02", chunk_id="doc02#0000", score=1.0, rank=1, char_start=10, char_end=20
    )
    one_char = RetrievedChunk(
        doc_id="doc01", chunk_id="doc01#0001", score=1.0, rank=1, char_start=19, char_end=30
    )
    assert panel_dense.is_correct([same_doc_abutting], item, 1) is False
    assert panel_dense.is_correct([other_doc_overlapping], item, 1) is False
    assert panel_dense.is_correct([one_char], item, 1) is True


def test_is_correct_truncates_by_rank_not_list_position(panel_dense: ModuleType) -> None:
    from fireassay.models import RetrievedChunk

    item = panel_dense.Item(item_id="i", question="q", doc_id="doc01", char_start=0, char_end=10)
    hit_at_rank_4 = RetrievedChunk(
        doc_id="doc01", chunk_id="doc01#0000", score=0.1, rank=4, char_start=0, char_end=10
    )
    assert panel_dense.is_correct([hit_at_rank_4], item, 3) is False
    assert panel_dense.is_correct([hit_at_rank_4], item, 5) is True


def _write_frozen(path: Path, items: list[object], columns: dict[str, list[bool]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "system_id", "correct"])
        for index, item in enumerate(items):
            for sid, column in columns.items():
                writer.writerow([item.item_id, sid, 1 if column[index] else 0])


def test_verify_bm25_fires_on_a_planted_mismatch(panel_dense: ModuleType, tmp_path: Path) -> None:
    """**Planted defect: flip one cell of the frozen CSV.** A twin check
    that cannot fail is not evidence that the definition was recovered."""
    chunks = list(chunk_corpus(load_fixture_docs(), size=800, overlap=100))
    items = _items(panel_dense)
    per_k = panel_dense.columns_for_system(BM25System(chunks, top_k=max(_TOP_KS)), items, _TOP_KS)
    columns = {panel_dense.system_id("bm25", (800, 100), k): per_k[k] for k in _TOP_KS}

    frozen = tmp_path / "frozen.csv"
    _write_frozen(frozen, items, columns)
    compared, mismatched, _report = panel_dense.verify_bm25(items, [(800, 100)], _TOP_KS, columns, frozen)
    assert compared == len(items) * len(_TOP_KS)
    assert mismatched == 0

    flipped = dict(columns)
    first = panel_dense.system_id("bm25", (800, 100), _TOP_KS[0])
    flipped[first] = [not columns[first][0], *columns[first][1:]]
    _write_frozen(frozen, items, flipped)
    _compared, mismatched, report = panel_dense.verify_bm25(items, [(800, 100)], _TOP_KS, columns, frozen)
    assert mismatched == 1
    assert report["examples"], "a mismatch must name the offending cell, not only count it"


def test_twin_report_records_the_definition_it_verified(panel_dense: ModuleType, tmp_path: Path) -> None:
    """The artifact has to carry the recovered definition in words, since
    the CSV it verifies carries only zeroes and ones."""
    chunks = list(chunk_corpus(load_fixture_docs(), size=800, overlap=100))
    items = _items(panel_dense)
    per_k = panel_dense.columns_for_system(BM25System(chunks, top_k=max(_TOP_KS)), items, _TOP_KS)
    columns = {panel_dense.system_id("bm25", (800, 100), k): per_k[k] for k in _TOP_KS}
    frozen = tmp_path / "frozen.csv"
    _write_frozen(frozen, items, columns)
    _compared, _mismatched, report = panel_dense.verify_bm25(items, [(800, 100)], _TOP_KS, columns, frozen)
    assert "recall@k > 0" in str(report["definition"])
    assert json.loads(json.dumps(report))["n_cells_mismatched"] == 0


def test_system_ids_match_the_frozen_panels_scheme(panel_dense: ModuleType) -> None:
    """A regenerated column has to be comparable to `run/panel54_matrix.csv`
    by name, so the id scheme is `system.panel.build_panel`'s, unchanged."""
    assert panel_dense.system_id("bm25", (1024, 128), 5) == "bm25/1024-128/k5"
    assert panel_dense.system_id("dense", (512, 128), 10) == "dense/512-128/k10"
