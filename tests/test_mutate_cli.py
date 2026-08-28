from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from fireassay.cli import app
from fireassay.hashing import config_hash as compute_config_hash

FIXTURES_DIR = Path(__file__).parent / "fixtures"

runner = CliRunner()


def _invoke(*args: str) -> object:
    return runner.invoke(app, list(args))


def _setup_suite_and_config(tmp_path: Path) -> tuple[Path, str]:
    """init -> import -> freeze -> run, returning (db, config_id) so a
    test can call `mutate --config config_id`."""
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
    # The matrix expands to exactly one config, {"top_k": "5"} (the axis
    # value is the quoted YAML string "5", not the int 5) -- config_hash
    # is a pure function of that dict, so this is the same id `run` just
    # stored, without needing to parse it back out of stdout (which only
    # prints a 12-char truncated prefix, not the full config id `mutate
    # --config` needs).
    config_id = compute_config_hash({"top_k": "5"})
    return db, config_id


def _mutants_yaml(tmp_path: Path, *, metric: str, max_drop: float = 0.05) -> Path:
    path = tmp_path / "mutants.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "detector": {"metric": metric, "max_drop": max_drop},
                "operators": [{"kind": "swap_ranking"}],
            }
        )
    )
    return path


def test_mutate_cli_uses_yaml_detector_metric_when_no_flag_is_passed(tmp_path: Path) -> None:
    """Regression test for the bug: `detector:` in the YAML must actually
    drive the detector used, with no `--detector-*` flag passed at all."""
    db, config_id = _setup_suite_and_config(tmp_path)
    mutants_yaml = _mutants_yaml(tmp_path, metric="retrieval.mrr")

    result = _invoke(
        "mutate", "--suite", "kb@1.0.0", "--config", config_id, "--operators", str(mutants_yaml),
        "--corpus", str(FIXTURES_DIR / "corpus.jsonl"), "--db", str(db),
    )
    assert result.exit_code == 0
    assert "retrieval.mrr" in result.stdout
    assert "retrieval.recall@5" not in result.stdout


def test_mutate_cli_explicit_flag_overrides_yaml_detector(tmp_path: Path) -> None:
    db, config_id = _setup_suite_and_config(tmp_path)
    mutants_yaml = _mutants_yaml(tmp_path, metric="retrieval.mrr")

    result = _invoke(
        "mutate", "--suite", "kb@1.0.0", "--config", config_id, "--operators", str(mutants_yaml),
        "--corpus", str(FIXTURES_DIR / "corpus.jsonl"), "--db", str(db),
        "--detector-metric", "retrieval.recall@5",
    )
    assert result.exit_code == 0
    assert "retrieval.recall@5" in result.stdout
    assert "retrieval.mrr" not in result.stdout


def test_mutate_cli_falls_back_to_default_when_neither_yaml_nor_flag_set_a_detector(
    tmp_path: Path,
) -> None:
    """A mutants YAML with no `detector:` block at all, and no
    `--detector-*` flags, must still work -- falling back to this
    command's original default (`retrieval.recall@5`), not crashing on
    "no detector configured"."""
    db, config_id = _setup_suite_and_config(tmp_path)
    mutants_yaml = tmp_path / "mutants.yaml"
    mutants_yaml.write_text(yaml.safe_dump({"operators": [{"kind": "swap_ranking"}]}))

    result = _invoke(
        "mutate", "--suite", "kb@1.0.0", "--config", config_id, "--operators", str(mutants_yaml),
        "--corpus", str(FIXTURES_DIR / "corpus.jsonl"), "--db", str(db),
    )
    assert result.exit_code == 0
    assert "retrieval.recall@5" in result.stdout
