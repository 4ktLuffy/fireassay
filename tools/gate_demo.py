#!/usr/bin/env python3
"""The gate, shown going red on a planted regression -- and refusing when
it cannot resolve one -- on the fixture suite, through the real CLI.

Pre-registration criterion 7 for M5 (`~/ObitosBrain/notes/2026-08-28-
fireassay-eval-integrity.md`, Goal 1) asked for "a planted regression and
a block". A gate that has never blocked anything is the first failure
condition that note lists, so this script exists to make the block a
committed, re-runnable artifact rather than a sentence in a commit
message (defect 46's shape).

What it does, every time, from nothing:

1. builds a fresh database in a temporary directory via `fireassay init /
   questions import / suite freeze / run` over `tests/fixtures/` -- 20
   questions, 14 with gold spans, a two-config matrix (`top_k` 5 and 3);
2. plants two regressions into the `top_k=5` config with `fireassay
   mutate`: `drop_results n=4` (total: recall@5 1.0 -> 0.0) and
   `corrupt_query pct=0.9` (real but subtle: recall@5 drops ~0.39);
3. runs `fireassay gate` three times and records the exit code each time:
   - **benign**: `top_k=5` vs `top_k=3`, threshold 0.5 -> exit 0, `pass`;
   - **planted_total**: vs the `drop_results` mutant, threshold 0.5 on
     `retrieval.recall@5` and `retrieval.mrr` -> exit 4, `block` on both;
   - **planted_subtle_refused**: vs the `corrupt_query` mutant, threshold
     0.3 -> exit 5. The regression is real (~0.39) and larger than the
     threshold, and the gate still refuses: 14 paired items resolve a
     minimum detectable effect of ~0.35, so 0.3 is a threshold this suite
     cannot honestly enforce (defect 9: "a 1-point threshold on a suite
     that resolves 2.7 points"). Refusing is not passing, and it is not
     blocking either.
4. writes `run/gate_demo.json`: every per-item value each gate call was
   computed over, the gate parameters, the verdicts and the exit codes --
   enough for `tools/evidence.py --claim gate.*` to recompute every
   verdict from the committed file alone, with no database, no corpus and
   no model.

Usage:

    .venv/bin/python tools/gate_demo.py [--out run/gate_demo.json] [--b 10000]

Exit status is non-zero unless all three cases came out exactly as
described above -- so `make gate-demo` in CI is itself a gate on the gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fireassay.cli import GATE_EXIT_BELOW_MDE, GATE_EXIT_BLOCKED
from fireassay.store.db import Store

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CORPUS = FIXTURES / "corpus.jsonl"

BENIGN_THRESHOLD = 0.5
TOTAL_THRESHOLD = 0.5
SUBTLE_THRESHOLD = 0.3


class DemoError(RuntimeError):
    pass


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fireassay.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )


def _must(*args: str) -> str:
    proc = _cli(*args)
    if proc.returncode != 0:
        raise DemoError(f"fireassay {' '.join(args)} exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def _per_item(
    store: Store, base_run_id: str, head_run_id: str, metrics: list[str]
) -> dict[str, dict[str, list[float] | list[str]]]:
    """Same alignment rule `cli._aligned_per_item` applies -- rebuilt here
    from `Store.run_scores` so the artifact records exactly what the gate
    paired, keyed by metric with parallel `base`/`head` lists in question
    id order."""
    base_by: dict[str, dict[str, float]] = {}
    for q, m, v in store.run_scores(base_run_id):
        base_by.setdefault(m, {})[q] = v
    head_by: dict[str, dict[str, float]] = {}
    for q, m, v in store.run_scores(head_run_id):
        head_by.setdefault(m, {})[q] = v
    out: dict[str, dict[str, list[float] | list[str]]] = {}
    for metric in metrics:
        shared = sorted(set(base_by.get(metric, {})) & set(head_by.get(metric, {})))
        out[metric] = {
            "question_ids": shared,
            "base": [base_by[metric][q] for q in shared],
            "head": [head_by[metric][q] for q in shared],
        }
    return out


def _gate(db: Path, base: str, head: str, specs: dict[str, float], b: int, seed: int) -> tuple[int, str]:
    args = ["gate", "--db", str(db), "--base", base, "--head", head, "--b", str(b), "--seed", str(seed)]
    for metric, threshold in specs.items():
        args += ["--metric", f"{metric}={threshold}"]
    proc = _cli(*args)
    return proc.returncode, proc.stdout


def run_demo(out: Path, b: int, seed: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="fireassay-gate-demo-") as tmp:
        tmp_path = Path(tmp)
        db = tmp_path / "gate-demo.db"

        _must("init", "--db", str(db))
        _must("questions", "import", str(FIXTURES / "questions.jsonl"), "--db", str(db))
        _must("suite", "freeze", "--name", "demo", "--version", "1.0.0", "--db", str(db))
        matrix = tmp_path / "matrix.yaml"
        matrix.write_text('suite: demo@1.0.0\naxes:\n  top_k: ["5", "3"]\n', encoding="utf-8")
        _must("run", "--matrix", str(matrix), "--corpus", str(CORPUS), "--db", str(db))

        store = Store(db)
        suite = store.get_suite("demo", "1.0.0")
        by_top_k = {store.get_config(r.config_id).spec["top_k"]: r for r in store.runs_for_suite(suite.id)}
        base_run, other_run = by_top_k["5"], by_top_k["3"]
        store.close()

        operators = tmp_path / "mutants.yaml"
        operators.write_text(
            "detector:\n  metric: retrieval.recall@5\n  max_drop: 0.05\n"
            "operators:\n  - kind: drop_results\n    n: 4\n  - kind: corrupt_query\n    pct: 0.9\n",
            encoding="utf-8",
        )
        stdout = _must(
            "mutate", "--suite", "demo@1.0.0", "--config", base_run.config_id,
            "--operators", str(operators), "--corpus", str(CORPUS), "--db", str(db),
        )
        mutation_run_id = stdout.split("mutation_run_id=")[1].split()[0]

        store = Store(db)
        mutants = {m.operator: m.mutant_run_id for m in store.get_mutants(mutation_run_id)}
        corpus_hash = base_run.env_json["affects_results"]["corpus_hash"]  # type: ignore[index]

        cases: dict[str, dict[str, object]] = {}

        def record(name: str, head_id: str, specs: dict[str, float], head_label: str) -> tuple[int, str]:
            code, text = _gate(db, base_run.id, head_id, specs, b, seed)
            cases[name] = {
                "base": {"run_id": base_run.id, "config": store.get_config(base_run.config_id).spec},
                "head": {
                    "run_id": head_id,
                    "config": store.get_config(store.get_run(head_id).config_id).spec,
                    "label": head_label,
                },
                "thresholds": specs,
                "per_item": _per_item(store, base_run.id, head_id, list(specs)),
                "exit_code": code,
                "gate_checks": [
                    {"metric": r.metric, "verdict": r.verdict, "delta": r.delta, "mde": r.mde,
                     "p_adjusted": r.p_adjusted, "n_items": r.n_items}
                    for r in store.get_gate_checks(base_run.id, head_id)
                ],
            }
            return code, text

        code, text = record("benign", other_run.id, {"retrieval.recall@5": BENIGN_THRESHOLD}, "top_k=3")
        if code != 0:
            raise DemoError(f"benign case: expected exit 0, got {code}\n{text}")

        code, text = record(
            "planted_total", mutants["drop_results"],
            {"retrieval.recall@5": TOTAL_THRESHOLD, "retrieval.mrr": TOTAL_THRESHOLD}, "drop_results n=4",
        )
        if code != GATE_EXIT_BLOCKED:
            raise DemoError(
                f"planted_total: expected exit {GATE_EXIT_BLOCKED} (blocked), got {code}\n{text}"
            )

        code, text = record(
            "planted_subtle_refused", mutants["corrupt_query"],
            {"retrieval.recall@5": SUBTLE_THRESHOLD}, "corrupt_query pct=0.9",
        )
        if code != GATE_EXIT_BELOW_MDE:
            raise DemoError(
                f"planted_subtle_refused: expected exit {GATE_EXIT_BELOW_MDE} (refused), got {code}\n{text}"
            )
        store.close()

    artifact: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "tools/gate_demo.py",
        "suite": "demo@1.0.0 (tests/fixtures/questions.jsonl)",
        "suite_hash": suite.suite_hash,
        "corpus_hash": corpus_hash,
        "gate": {"alpha": 0.05, "power": 0.8, "b": b, "seed": seed},
        "cases": cases,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "run" / "gate_demo.json")
    parser.add_argument("--b", type=int, default=10000, help="bootstrap replicates per metric")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        artifact = run_demo(args.out, args.b, args.seed)
    except DemoError as exc:
        print(f"GATE DEMO FAILED: {exc}", file=sys.stderr)
        return 1
    cases = artifact["cases"]
    assert isinstance(cases, dict)
    for name, case in cases.items():
        checks = case["gate_checks"]
        verdicts = ", ".join(
            f"{c['metric']} {c['verdict']} (delta {c['delta']:+.3f}, mde {c['mde']:.3f})" for c in checks
        )
        print(f"{name:24s} exit={case['exit_code']}  {verdicts or 'no verdict recorded (refused)'}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
