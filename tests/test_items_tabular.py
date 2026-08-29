"""`test_items_tabular.py` (M-ITEMS-SPEC.md §6): CSV round-trip; malformed
rows rejected with a clear message."""

from __future__ import annotations

from pathlib import Path

import pytest

from fireassay.items.adapters.tabular import (
    load_meta_jsonl,
    load_responses_csv,
    load_responses_jsonl,
    write_responses_csv,
)
from fireassay.items.core import ItemMeta, ItemResponses


def test_csv_round_trip(tmp_path: Path) -> None:
    responses = [
        ItemResponses(item_id="q1", responses=(True, False, True)),
        ItemResponses(item_id="q2", responses=(False, False, True)),
    ]
    system_ids = ["bm25-k3", "bm25-k5", "dense-k5"]
    path = tmp_path / "matrix.csv"
    write_responses_csv(responses, system_ids, path)

    loaded = load_responses_csv(path)
    assert loaded == responses


def test_csv_load_matches_the_documented_shape(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text(
        "item_id,system_id,correct\n"
        "q1,bm25-k3,1\n"
        "q1,bm25-k5,0\n"
        "q2,bm25-k3,1\n"
        "q2,bm25-k5,1\n",
        encoding="utf-8",
    )
    responses = load_responses_csv(path)
    by_id = {r.item_id: r for r in responses}
    assert by_id["q1"].responses == (True, False)
    assert by_id["q2"].responses == (True, True)


def test_csv_accepts_true_false_and_1_0(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text(
        "item_id,system_id,correct\n"
        "q1,s1,true\n"
        "q1,s2,false\n"
        "q1,s3,1\n"
        "q1,s4,0\n",
        encoding="utf-8",
    )
    responses = load_responses_csv(path)
    assert responses == [ItemResponses(item_id="q1", responses=(True, False, True, False))]


def test_csv_wrong_header_rejected(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text("id,system,ok\nq1,s1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_responses_csv(path)


def test_csv_malformed_correct_value_rejected_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text("item_id,system_id,correct\nq1,s1,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_responses_csv(path)


def test_csv_missing_system_for_an_item_rejected(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text(
        "item_id,system_id,correct\nq1,s1,1\nq1,s2,1\nq2,s1,1\n",  # q2 missing s2
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing a row"):
        load_responses_csv(path)


def test_csv_duplicate_row_rejected(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text(
        "item_id,system_id,correct\nq1,s1,1\nq1,s1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_responses_csv(path)


def test_csv_no_data_rows_rejected(tmp_path: Path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text("item_id,system_id,correct\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        load_responses_csv(path)


def test_jsonl_load_matches_the_csv_shape(tmp_path: Path) -> None:
    path = tmp_path / "matrix.jsonl"
    path.write_text(
        '{"item_id": "q1", "system_id": "bm25-k3", "correct": true}\n'
        '{"item_id": "q1", "system_id": "bm25-k5", "correct": false}\n',
        encoding="utf-8",
    )
    responses = load_responses_jsonl(path)
    assert responses == [ItemResponses(item_id="q1", responses=(True, False))]


def test_jsonl_invalid_json_rejected_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "matrix.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_responses_jsonl(path)


def test_jsonl_non_boolean_correct_rejected(tmp_path: Path) -> None:
    path = tmp_path / "matrix.jsonl"
    path.write_text('{"item_id": "q1", "system_id": "s1", "correct": "yes"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON boolean"):
        load_responses_jsonl(path)


def test_meta_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "meta.jsonl"
    path.write_text(
        '{"item_id": "q1", "question": "what?", "claimed_difficulty": "easy"}\n'
        '{"item_id": "q2", "reference_answer": "42"}\n',
        encoding="utf-8",
    )
    meta = load_meta_jsonl(path)
    assert meta["q1"] == ItemMeta(item_id="q1", question="what?", claimed_difficulty="easy")
    assert meta["q2"] == ItemMeta(item_id="q2", reference_answer="42")


def test_meta_jsonl_duplicate_item_id_rejected(tmp_path: Path) -> None:
    path = tmp_path / "meta.jsonl"
    path.write_text(
        '{"item_id": "q1", "question": "a"}\n{"item_id": "q1", "question": "b"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate item_id"):
        load_meta_jsonl(path)


def test_write_responses_csv_rejects_mismatched_system_count(tmp_path: Path) -> None:
    responses = [ItemResponses(item_id="q1", responses=(True, False))]
    with pytest.raises(ValueError, match="system_ids"):
        write_responses_csv(responses, ["only-one-system"], tmp_path / "out.csv")
