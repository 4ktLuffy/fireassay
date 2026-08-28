from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fireassay.cli import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"

runner = CliRunner()

#: `judge_calibration` and `null_questions` always report NOT_RUN (both
#: need something M2 does not have -- see controls/judge_calibration.py,
#: controls/null_questions.py) -- so every CLI invocation below that
#: expects a clean run needs both named.
_ALLOW_NOT_RUN = ["--allow-not-run", "judge_calibration", "--allow-not-run", "null_questions"]


def _invoke(*args: str) -> object:
    return runner.invoke(app, list(args))


def _standard_matrix(tmp_path: Path, suite_ref: str) -> Path:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(f'suite: {suite_ref}\naxes:\n  top_k: ["5"]\n', encoding="utf-8")
    return matrix


def test_controls_run_exits_3_on_disallowed_not_run(tmp_path: Path) -> None:
    """On the healthy fixture suite, the four controls with a real
    PASSED/FAILED implementation are all expected to PASS (see the
    per-control unit tests); `judge_calibration` and `null_questions`
    always report NOT_RUN, and with neither allowed here, that alone must
    produce exit 3 -- distinct from 2 (a genuine FAILED).

    Note: `shuffled_gold` self-calibrates its chance band from `n_seeds`
    independent permutations of this fixture and compares one further,
    independent permutation against them (controls/shuffled_gold.py); on
    a small fixture this is a genuine statistical comparison, not a
    tautology, so it carries a small (single-digit percent, with
    `n_seeds=20`) chance of landing FAILED here through ordinary sampling
    variance rather than a bug. If this test starts failing with exit
    code 2 and `shuffled_gold` FAILED in the output rather than 3, that is
    the likely explanation -- re-run is not expected to change a
    deterministic outcome; increasing `n_seeds` in `controls/expected.yaml`
    would.
    """
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _invoke("questions", "import", str(FIXTURES_DIR / "questions.jsonl"), "--db", str(db))
    _invoke("suite", "freeze", "--name", "kb", "--version", "1.0.0", "--db", str(db))
    matrix = _standard_matrix(tmp_path, "kb@1.0.0")

    result = _invoke(
        "controls", "run", "--matrix", str(matrix), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"),
        "--db", str(db),
    )
    assert result.exit_code == 3
    assert "judge_calibration" in result.stdout
    assert "null_questions" in result.stdout
    assert "NOT_RUN" in result.stdout


def test_controls_run_exits_0_when_not_run_is_allowed(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _invoke("questions", "import", str(FIXTURES_DIR / "questions.jsonl"), "--db", str(db))
    _invoke("suite", "freeze", "--name", "kb", "--version", "1.0.0", "--db", str(db))
    matrix = _standard_matrix(tmp_path, "kb@1.0.0")

    result = _invoke(
        "controls", "run", "--matrix", str(matrix), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"),
        "--db", str(db), *_ALLOW_NOT_RUN,
    )
    assert result.exit_code == 0


def test_controls_run_exits_2_on_failed(tmp_path: Path) -> None:
    """A suite with no gold-bearing questions at all makes `no_retrieval`'s
    writable twin impossible to satisfy (there is nothing for the
    unmodified config to score `recall@k > 0` on) -- FAILED, exit 2, which
    must take priority over the simultaneous disallowed NOT_RUN from
    `judge_calibration` (and, here, `null_questions` and `shuffled_gold`/
    `corpus_ablation`, both of which report NOT_RUN on a suite with no
    gold spans at all)."""
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))

    no_gold = tmp_path / "no_gold_questions.jsonl"
    with open(no_gold, "w", encoding="utf-8") as f:
        for text in ["What is question one?", "What is question two?", "What is question three?"]:
            f.write(
                json.dumps(
                    {"text": text, "qtype": "factual", "difficulty": "easy", "provenance": "synthetic"}
                )
                + "\n"
            )

    _invoke("questions", "import", str(no_gold), "--db", str(db))
    _invoke("suite", "freeze", "--name", "nogold", "--version", "1.0.0", "--db", str(db))
    matrix = _standard_matrix(tmp_path, "nogold@1.0.0")

    result = _invoke(
        "controls", "run", "--matrix", str(matrix), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"),
        "--db", str(db), *_ALLOW_NOT_RUN,
    )
    assert result.exit_code == 2
    assert "no_retrieval" in result.stdout
    assert "FAILED" in result.stdout


def test_controls_show_after_controls_run(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _invoke("questions", "import", str(FIXTURES_DIR / "questions.jsonl"), "--db", str(db))
    _invoke("suite", "freeze", "--name", "kb", "--version", "1.0.0", "--db", str(db))
    matrix = _standard_matrix(tmp_path, "kb@1.0.0")

    result = _invoke(
        "controls", "run", "--matrix", str(matrix), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"),
        "--db", str(db), *_ALLOW_NOT_RUN,
    )
    assert result.exit_code == 0

    result = _invoke("run", "--matrix", str(matrix), "--corpus", str(FIXTURES_DIR / "corpus.jsonl"),
                      "--db", str(db))
    run_id = result.stdout.strip().splitlines()[0].split()[0]

    result = _invoke("controls", "show", run_id, "--db", str(db))
    assert result.exit_code == 0
    assert "no_retrieval" in result.stdout
