from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fireassay.cli import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"

runner = CliRunner()


def _invoke(*args: str) -> object:
    return runner.invoke(app, list(args))


def test_init_creates_a_usable_store(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    result = _invoke("init", "--db", str(db))
    assert result.exit_code == 0
    assert db.exists()


def test_questions_import_and_suite_freeze_show(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))

    result = _invoke("questions", "import", str(FIXTURES_DIR / "questions.jsonl"), "--db", str(db))
    assert result.exit_code == 0
    assert "imported 20" in result.stdout

    result = _invoke("suite", "freeze", "--name", "kb", "--version", "1.0.0", "--db", str(db))
    assert result.exit_code == 0

    result = _invoke("suite", "show", "kb@1.0.0", "--db", str(db))
    assert result.exit_code == 0
    assert "kb@1.0.0" in result.stdout


def test_run_and_compare_smoke(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _invoke("questions", "import", str(FIXTURES_DIR / "questions.jsonl"), "--db", str(db))
    _invoke("suite", "freeze", "--name", "kb", "--version", "1.0.0", "--db", str(db))

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text('suite: kb@1.0.0\naxes:\n  top_k: ["5"]\n', encoding="utf-8")

    result = _invoke(
        "run", "--matrix", str(matrix), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"), "--db", str(db)
    )
    assert result.exit_code == 0
    assert "status=complete" in result.stdout

    result = _invoke("compare", "--suite", "kb@1.0.0", "--db", str(db))
    assert result.exit_code == 0


def test_diff_across_two_different_suites_exits_2(tmp_path: Path) -> None:
    """M1-SPEC.md §11 requires exit code 2 on a ComparisonRefusedError ("compare
    across two different suites exits 2"), distinct from exit code 1 for a
    usage error.

    The `compare` subcommand's CLI signature (M1-SPEC.md §10) only ever
    takes a single `--suite name@version` and compares runs *within* that
    one resolved suite_id, so it structurally cannot be handed two runs
    from two different suites. `diff --base <run> --head <run>` is the CLI
    command that can, and it reaches the exact same refusal machinery
    (integrity.assert_comparable -> SUITE_MISMATCH -> exit 2) — this test
    exercises the cross-suite refusal through `diff` for that reason.
    """
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _invoke("questions", "import", str(FIXTURES_DIR / "questions.jsonl"), "--db", str(db))

    # Suite A: all 20 fixture questions.
    result = _invoke("suite", "freeze", "--name", "kb-a", "--version", "1.0.0", "--db", str(db))
    assert result.exit_code == 0

    # Import one more question exclusive to suite B, so suite B's
    # question set (and therefore suite_hash) genuinely differs from A's.
    extra = tmp_path / "extra.jsonl"
    extra.write_text(
        json.dumps(
            {
                "text": "An extra question that only exists in suite B.",
                "qtype": "factual",
                "difficulty": "easy",
                "provenance": "synthetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = _invoke("questions", "import", str(extra), "--db", str(db))
    assert result.exit_code == 0

    result = _invoke("suite", "freeze", "--name", "kb-b", "--version", "1.0.0", "--db", str(db))
    assert result.exit_code == 0

    matrix_a = tmp_path / "matrix_a.yaml"
    matrix_a.write_text('suite: kb-a@1.0.0\naxes:\n  top_k: ["3"]\n', encoding="utf-8")
    matrix_b = tmp_path / "matrix_b.yaml"
    matrix_b.write_text('suite: kb-b@1.0.0\naxes:\n  top_k: ["3"]\n', encoding="utf-8")

    result = _invoke(
        "run", "--matrix", str(matrix_a), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"), "--db", str(db)
    )
    assert result.exit_code == 0
    run_a_id = result.stdout.strip().splitlines()[0].split()[0]

    result = _invoke(
        "run", "--matrix", str(matrix_b), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"), "--db", str(db)
    )
    assert result.exit_code == 0
    run_b_id = result.stdout.strip().splitlines()[0].split()[0]

    result = _invoke("diff", "--base", run_a_id, "--head", run_b_id, "--db", str(db))
    assert result.exit_code == 2
    assert "SUITE_MISMATCH" in result.stdout
