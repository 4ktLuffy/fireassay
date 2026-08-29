"""`fireassay items review build`/`score` at the CLI (M-ITEMS-SPEC.md
§7b): the standalone `--matrix`/`--meta` path and the in-project `--db`
path must produce byte-identical batch/key output for equivalent input at
the same `--seed`, and the `--worksheet` CSV must round-trip through
`items review score --labels` exactly like a hand-authored JSONL would.

Exercised via `typer.testing.CliRunner`, matching `test_curate_cli.py`'s/
`test_mutate_cli.py`'s established pattern -- `items review build`/`score`
are new commands with no existing CLI test file of their own.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fireassay.cli import _load_labels_file, app
from fireassay.items.adapters.store import load_matrix as items_load_matrix
from fireassay.items.adapters.store import load_meta as items_load_meta
from fireassay.items.adapters.tabular import write_responses_csv
from fireassay.items.calibration import evaluate_detector, load_calibration_jsonl
from fireassay.items.core import ItemResponses
from fireassay.items.review import ReviewKeyEntry, score_review
from fireassay.store.db import Store

FIXTURES_DIR = Path(__file__).parent / "fixtures"

runner = CliRunner()


def _invoke(*args: str) -> object:
    return runner.invoke(app, list(args))


# The fixed 7-item x 4-system matrix `test_items_core.py` hand-verifies:
# one mislabel_suspect, one dead_all_pass, one dead_all_fail, four live.
_FIXED_MATRIX = [
    ItemResponses(item_id="ramp1", responses=(True, False, False, False)),
    ItemResponses(item_id="ramp2", responses=(True, True, False, False)),
    ItemResponses(item_id="ramp3", responses=(True, True, True, False)),
    ItemResponses(item_id="dead_pass", responses=(True, True, True, True)),
    ItemResponses(item_id="dead_fail", responses=(False, False, False, False)),
    ItemResponses(item_id="live_pos", responses=(True, True, False, False)),
    ItemResponses(item_id="mislabel", responses=(False, False, True, True)),
]
_FIXED_SYSTEM_IDS = ["s0", "s1", "s2", "s3"]


def _write_standalone_fixture(tmp_path: Path) -> tuple[Path, Path]:
    matrix_csv = tmp_path / "matrix.csv"
    write_responses_csv(_FIXED_MATRIX, _FIXED_SYSTEM_IDS, matrix_csv)
    meta_jsonl = tmp_path / "meta.jsonl"
    with open(meta_jsonl, "w", encoding="utf-8") as f:
        for item in _FIXED_MATRIX:
            f.write(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "question": f"Question text for {item.item_id}, long enough to be real?",
                        "reference_answer": f"Reference answer for {item.item_id}.",
                        "evidence_quote": f"Evidence quote for {item.item_id}.",
                    }
                )
            )
            f.write("\n")
    return matrix_csv, meta_jsonl


# -- Change 4: standalone --matrix/--meta and in-project --db must agree -----


def _setup_suite_with_four_systems(tmp_path: Path) -> Path:
    """init -> import -> freeze -> run, with 4 distinct `top_k` systems so
    `items review build --db` has enough systems for `analyse`."""
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _invoke("questions", "import", str(FIXTURES_DIR / "questions.jsonl"), "--db", str(db))
    _invoke("suite", "freeze", "--name", "kb", "--version", "1.0.0", "--db", str(db))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text('suite: kb@1.0.0\naxes:\n  top_k: ["3", "5", "10", "20"]\n', encoding="utf-8")
    result = _invoke(
        "run", "--matrix", str(matrix), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"), "--db", str(db)
    )
    assert result.exit_code == 0
    return db


def test_standalone_and_db_paths_produce_identical_batch_and_key(tmp_path: Path) -> None:
    db = _setup_suite_with_four_systems(tmp_path)

    db_out, db_key = tmp_path / "db_batch.jsonl", tmp_path / "db_key.json"
    result = _invoke(
        "items", "review", "build",
        "--db", str(db), "--suite", "kb@1.0.0",
        "--metric", "retrieval.recall@5", "--threshold", "0.0",
        "--out", str(db_out), "--key", str(db_key),
        "--n-flagged", "3", "--n-unflagged", "3", "--n-calibration", "2", "--seed", "7",
    )
    assert result.exit_code == 0, result.stdout

    # Build the equivalent standalone matrix/meta files from the exact
    # same store, via the same adapters `items review build --db` itself
    # uses internally -- the standalone contract's whole point is that
    # this round trip is lossless.
    store = Store(db)
    suite_row = store.get_suite("kb", "1.0.0")
    runs = store.runs_for_suite(suite_row.id)
    responses = items_load_matrix(store, runs, metric="retrieval.recall@5", threshold=0.0)
    meta = items_load_meta(store, suite_row.id)
    store.close()

    matrix_csv = tmp_path / "standalone_matrix.csv"
    write_responses_csv(responses, [r.id for r in runs], matrix_csv)
    meta_jsonl = tmp_path / "standalone_meta.jsonl"
    with open(meta_jsonl, "w", encoding="utf-8") as f:
        for m in meta.values():
            f.write(m.model_dump_json())
            f.write("\n")

    mm_out, mm_key = tmp_path / "mm_batch.jsonl", tmp_path / "mm_key.json"
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv), "--meta", str(meta_jsonl),
        "--out", str(mm_out), "--key", str(mm_key),
        "--n-flagged", "3", "--n-unflagged", "3", "--n-calibration", "2", "--seed", "7",
    )
    assert result.exit_code == 0, result.stdout

    db_batch_lines = db_out.read_text(encoding="utf-8").splitlines()
    mm_batch_lines = mm_out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in db_batch_lines] == [json.loads(line) for line in mm_batch_lines]

    assert json.loads(db_key.read_text(encoding="utf-8")) == json.loads(mm_key.read_text(encoding="utf-8"))


def test_matrix_and_db_together_is_an_error(tmp_path: Path) -> None:
    matrix_csv, meta_jsonl = _write_standalone_fixture(tmp_path)
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv), "--meta", str(meta_jsonl), "--db", str(tmp_path / "x.db"),
        "--out", str(tmp_path / "out.jsonl"), "--key", str(tmp_path / "key.json"),
        "--n-flagged", "1", "--n-unflagged", "1",
    )
    assert result.exit_code == 1
    # `typer.echo(..., err=True)` writes to stderr, not stdout -- see
    # `items analyse`'s identical convention.
    assert "pass either --matrix or --db, not both" in result.stderr


def test_matrix_without_meta_is_an_error(tmp_path: Path) -> None:
    matrix_csv, _meta_jsonl = _write_standalone_fixture(tmp_path)
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv),
        "--out", str(tmp_path / "out.jsonl"), "--key", str(tmp_path / "key.json"),
        "--n-flagged", "1", "--n-unflagged", "1",
    )
    assert result.exit_code == 1
    assert "--matrix requires --meta" in result.stderr


def test_neither_matrix_nor_db_is_an_error(tmp_path: Path) -> None:
    result = _invoke(
        "items", "review", "build",
        "--out", str(tmp_path / "out.jsonl"), "--key", str(tmp_path / "key.json"),
        "--n-flagged", "1", "--n-unflagged", "1",
    )
    assert result.exit_code == 1
    assert "one of --matrix or --db is required" in result.stderr


# -- Change 5: worksheet round trip -------------------------------------------


def test_worksheet_round_trip(tmp_path: Path) -> None:
    matrix_csv, meta_jsonl = _write_standalone_fixture(tmp_path)
    out, key, worksheet = tmp_path / "batch.jsonl", tmp_path / "key.json", tmp_path / "worksheet.csv"
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv), "--meta", str(meta_jsonl),
        "--out", str(out), "--key", str(key), "--worksheet", str(worksheet),
        "--n-flagged", "1", "--n-unflagged", "2", "--seed", "0",
    )
    assert result.exit_code == 0, result.stdout

    with open(worksheet, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ["review_id", "verdict", "question", "reference_answer", "evidence_quote"]
    assert len(rows) == 3  # 1 flagged + 2 unflagged
    assert all(row[1] == "" for row in rows)  # verdict left empty

    # Fill in every row but the last, alternating verdicts; leave the
    # last row's verdict blank -- it must not become a label.
    filled_rows = []
    for i, row in enumerate(rows):
        new_row = list(row)
        if i < len(rows) - 1:
            new_row[1] = "purge" if i % 2 == 0 else "keep"
        filled_rows.append(new_row)
    with open(worksheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filled_rows)

    blank_review_id = rows[-1][0]

    labels = _load_labels_file(worksheet)
    label_by_id = {label.review_id: label for label in labels}
    assert blank_review_id not in label_by_id
    assert len(labels) == len(rows) - 1

    with open(key, encoding="utf-8") as f:
        key_entries = [ReviewKeyEntry(**entry) for entry in json.load(f)]
    key_by_id = {entry.review_id: entry for entry in key_entries}

    score = score_review(labels, key_entries)
    expected_flagged_n = sum(1 for rid in label_by_id if key_by_id[rid].source == "flagged")
    expected_flagged_bad = sum(
        1 for rid, label in label_by_id.items()
        if key_by_id[rid].source == "flagged" and label.verdict == "purge"
    )
    assert score.precision_n == expected_flagged_n
    if expected_flagged_n > 0:
        assert score.precision == pytest.approx(expected_flagged_bad / expected_flagged_n)
    else:
        assert score.precision is None

    # the CLI itself must also read the filled-in worksheet without error
    result = _invoke("items", "review", "score", "--labels", str(worksheet), "--key", str(key))
    assert result.exit_code == 0, result.stdout


# -- --exclude: multi-pass calibration -----------------------------------------


def test_exclude_accepts_flags_format_and_reports_matching_count(tmp_path: Path) -> None:
    """`--exclude` reuses `_load_flags_file` (the same file `--flags`
    reads for `items detector-score`) -- comments and blank lines must be
    ignored, and the reported count must match what was actually loaded,
    not the number of lines in the file."""
    matrix_csv, meta_jsonl = _write_standalone_fixture(tmp_path)
    out, key = tmp_path / "batch.jsonl", tmp_path / "key.json"
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv), "--meta", str(meta_jsonl),
        "--out", str(out), "--key", str(key),
        "--n-flagged", "1", "--n-unflagged", "3", "--n-calibration", "2", "--seed", "5",
    )
    assert result.exit_code == 0, result.stdout

    with open(key, encoding="utf-8") as f:
        key_entries = [ReviewKeyEntry(**entry) for entry in json.load(f)]
    already_reviewed = sorted({e.item_id for e in key_entries})
    # the fixture must actually produce something to exclude for this
    # test to mean anything -- fail loudly, not vacuously, if it does not
    assert already_reviewed

    exclude_path = tmp_path / "exclude.txt"
    with open(exclude_path, "w", encoding="utf-8") as f:
        f.write("# already reviewed in pass one\n\n")
        for item_id in already_reviewed:
            f.write(item_id + "\n")
        f.write("\n")

    out2, key2 = tmp_path / "batch2.jsonl", tmp_path / "key2.json"
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv), "--meta", str(meta_jsonl),
        "--out", str(out2), "--key", str(key2),
        "--n-flagged", "1", "--n-unflagged", "3", "--n-calibration", "2", "--seed", "5",
        "--exclude", str(exclude_path),
    )
    assert result.exit_code == 0, result.stdout
    assert f"excluded {len(already_reviewed)} already-reviewed item(s)" in result.stdout

    with open(key2, encoding="utf-8") as f:
        key_entries2 = [ReviewKeyEntry(**entry) for entry in json.load(f)]
    assert set(already_reviewed).isdisjoint({e.item_id for e in key_entries2})


# -- calibration export: seeded items are dropped, detector-score round trip -


def test_export_calibration_drops_seeded_entries(tmp_path: Path) -> None:
    matrix_csv, meta_jsonl = _write_standalone_fixture(tmp_path)
    out, key, worksheet = tmp_path / "batch.jsonl", tmp_path / "key.json", tmp_path / "worksheet.csv"
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv), "--meta", str(meta_jsonl),
        "--out", str(out), "--key", str(key), "--worksheet", str(worksheet),
        "--n-flagged", "1", "--n-unflagged", "2", "--n-calibration", "1",
        "--n-seeded-per-kind", "2", "--seed", "9",
    )
    assert result.exit_code == 0, result.stdout

    with open(key, encoding="utf-8") as f:
        key_entries = [ReviewKeyEntry(**entry) for entry in json.load(f)]
    seeded_entries = [e for e in key_entries if e.is_seeded]
    # the fixture must actually exercise seeding for this test to mean
    # anything -- fail loudly, not vacuously, if it does not
    assert seeded_entries, "test fixture produced no seeded entries -- adjust --seed/--n-seeded-per-kind"

    # fill every row with a verdict -- a blind reviewer cannot tell a
    # seeded row from a natural one, so they all get filled the same way
    with open(worksheet, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    verdict_cycle = ["purge", "keep", "rewrite"]
    filled_rows = [[row[0], verdict_cycle[i % 3], row[2], row[3], row[4]] for i, row in enumerate(rows)]
    with open(worksheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filled_rows)

    calib_out = tmp_path / "calib.jsonl"
    result = _invoke(
        "items", "review", "export-calibration",
        "--labels", str(worksheet), "--key", str(key),
        "--reviewer", "alice", "--batch-id", "b1",
        "--out", str(calib_out),
    )
    assert result.exit_code == 0, result.stdout
    assert f"skipped {len(seeded_entries)} seeded" in result.stdout

    exported = load_calibration_jsonl(calib_out)
    exported_item_ids = {label.item_id for label in exported}
    seeded_item_ids = {e.item_id for e in seeded_entries}
    assert exported_item_ids.isdisjoint(seeded_item_ids)

    non_seeded_entries = [e for e in key_entries if not e.is_seeded]
    assert len(exported) == len(non_seeded_entries)


def test_full_round_trip_export_calibration_then_detector_score(tmp_path: Path) -> None:
    matrix_csv, meta_jsonl = _write_standalone_fixture(tmp_path)
    out, key, worksheet = tmp_path / "batch.jsonl", tmp_path / "key.json", tmp_path / "worksheet.csv"
    result = _invoke(
        "items", "review", "build",
        "--matrix", str(matrix_csv), "--meta", str(meta_jsonl),
        "--out", str(out), "--key", str(key), "--worksheet", str(worksheet),
        "--n-flagged", "1", "--n-unflagged", "3", "--n-calibration", "2", "--seed", "5",
    )
    assert result.exit_code == 0, result.stdout

    with open(key, encoding="utf-8") as f:
        key_entries = [ReviewKeyEntry(**entry) for entry in json.load(f)]
    assert not any(e.is_seeded for e in key_entries)  # --n-seeded-per-kind defaults to 0

    with open(worksheet, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert len(rows) == 6  # 1 flagged + 3 unflagged + 2 calibration, out of a 7-item pool

    # fill every row but the last with alternating verdicts; leave the
    # last blank -- it must not become a calibration label at all
    filled_rows = []
    for i, row in enumerate(rows):
        new_row = list(row)
        if i < len(rows) - 1:
            new_row[1] = "purge" if i % 2 == 0 else "keep"
        filled_rows.append(new_row)
    with open(worksheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filled_rows)
    blank_review_id = rows[-1][0]

    calib_out = tmp_path / "calib.jsonl"
    result = _invoke(
        "items", "review", "export-calibration",
        "--labels", str(worksheet), "--key", str(key),
        "--reviewer", "alice", "--batch-id", "batch1",
        "--out", str(calib_out),
    )
    assert result.exit_code == 0, result.stdout

    exported = load_calibration_jsonl(calib_out)
    key_by_id = {e.review_id: e for e in key_entries}
    blank_item_id = key_by_id[blank_review_id].item_id
    assert all(label.item_id != blank_item_id for label in exported)
    assert len(exported) == len(rows) - 1

    # a "detector" that flags every item sampled under the flagged or
    # calibration strata -- a plain-text list, as any external detector,
    # in any language, could produce
    flagged_ids = {e.item_id for e in key_entries if e.source in ("flagged", "calibration")}
    flags_path = tmp_path / "flags.txt"
    with open(flags_path, "w", encoding="utf-8") as f:
        f.write("# detector output\n\n")
        for item_id in sorted(flagged_ids):
            f.write(item_id + "\n")

    # ground truth, recomputed independently from the files this round
    # trip actually produced -- "the reported counts match what was put
    # in" means these must agree with evaluate_detector's own answer.
    expected_eval = evaluate_detector(flagged_ids, exported, reviewer="alice")

    precision_labels = [label for label in exported if label.item_id in flagged_ids]
    expected_precision_n = len(precision_labels)
    expected_precision_bad = sum(1 for label in precision_labels if label.verdict == "purge")
    assert expected_eval.precision.n == expected_precision_n
    if expected_precision_n > 0:
        assert expected_eval.precision.value == pytest.approx(expected_precision_bad / expected_precision_n)
    else:
        assert expected_eval.precision.value is None

    calibration_exported = [label for label in exported if label.stratum == "calibration"]
    expected_base_den = len(calibration_exported)
    expected_base_num = sum(1 for label in calibration_exported if label.verdict == "purge")
    assert expected_eval.base_rate.n == expected_base_den
    if expected_base_den > 0:
        assert expected_eval.base_rate.value == pytest.approx(expected_base_num / expected_base_den)
    else:
        assert expected_eval.base_rate.value is None

    labelled_flagged_item_ids = {label.item_id for label in exported} & flagged_ids
    assert expected_eval.n_flagged_total == len(flagged_ids)
    assert expected_eval.n_flagged_labelled == len(labelled_flagged_item_ids)
    assert expected_eval.n_flagged_unlabelled == len(flagged_ids) - len(labelled_flagged_item_ids)

    # the CLI itself must run this same computation without error
    result = _invoke(
        "items", "detector-score",
        "--flags", str(flags_path), "--calibration", str(calib_out), "--reviewer", "alice",
    )
    assert result.exit_code == 0, result.stdout
    assert "precision" in result.stdout
    assert "base_rate" in result.stdout
    assert f"n_flagged_total={expected_eval.n_flagged_total}" in result.stdout
