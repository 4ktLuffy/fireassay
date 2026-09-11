#!/usr/bin/env python3
"""The head-to-head: dense against BM25, through the gate that already
refuses what it cannot resolve.

Reads `run/panel_dense_matrix.csv` (2,364 items x 12 systems) and, for
each matched `(chunking, top_k)` pair, feeds BM25's per-item `correct`
vector as `base` and dense's as `head` into `gate.evaluate_gate`. All six
pairs go into **one** call, so Holm-Bonferroni corrects across the six
comparisons actually being made rather than pretending each was the only
one — and the MDE each is checked against is computed at `alpha / 6`, the
level the correction actually costs.

**The threshold is pre-registered, not tuned.** `--threshold` defaults to
0.05: five points of retrieval success, the smallest difference that
would change which retriever ships. The number is fixed before the data
is read and is not moved afterwards; if the gate refuses it, the refusal
is the finding (M-DENSE-SPEC.md §3.6). `--refusal-probe` additionally
runs the same comparison at a deliberately small threshold to show the
refusal boundary — the threshold below which this suite, at this size,
cannot honestly enforce anything.

    .venv/bin/python tools/dense_vs_bm25.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from fireassay.gate import MetricSpec, ThresholdBelowMDEError, evaluate_gate, paired_bootstrap

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN = _REPO_ROOT / "run"

#: Pre-registered. Five points of retrieval success. Stated here, in the
#: source, rather than passed on a command line, so the value a report
#: quotes and the value the code used cannot drift apart.
DEFAULT_THRESHOLD = 0.05

#: The deliberately-too-small threshold `--refusal-probe` uses to
#: demonstrate the gate's own detection floor.
PROBE_THRESHOLD = 0.01


def load_columns(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """`(item_ids, {system_id: [0.0/1.0 per item]})` from a response
    matrix, item order as first seen."""
    item_ids: list[str] = []
    seen: set[str] = set()
    columns: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = row["item_id"]
            if item_id not in seen:
                seen.add(item_id)
                item_ids.append(item_id)
            columns.setdefault(row["system_id"], {})[item_id] = 1.0 if row["correct"] == "1" else 0.0
    ordered = {sid: [col[i] for i in item_ids] for sid, col in columns.items()}
    return item_ids, ordered


def pair_name(chunking: str, k: str) -> str:
    """The metric name one `(chunking, top_k)` comparison is carried under
    inside the gate call."""
    return f"dense_vs_bm25.{chunking}.{k}"


def build_pairs(columns: dict[str, list[float]]) -> dict[str, tuple[list[float], list[float]]]:
    """One `(base, head)` per `(chunking, top_k)` that has both a `bm25/`
    and a `dense/` column. A pair missing one side is skipped loudly
    rather than compared against nothing."""
    per_item: dict[str, tuple[list[float], list[float]]] = {}
    for sid in columns:
        retriever, _, rest = sid.partition("/")
        if retriever != "bm25":
            continue
        head_id = f"dense/{rest}"
        if head_id not in columns:
            continue
        chunking, _, k = rest.partition("/")
        per_item[pair_name(chunking, k)] = (columns[sid], columns[head_id])
    if not per_item:
        raise SystemExit("dense_vs_bm25: no (chunking, top_k) has both a bm25 and a dense column")
    return dict(sorted(per_item.items()))


def run_gate(
    per_item: dict[str, tuple[list[float], list[float]]], threshold: float, *, alpha: float, seed: int
) -> dict[str, object]:
    """One `evaluate_gate` call over every pair. A refusal is reported as
    a refusal — never converted into a pass, and never worked around by
    lowering `threshold`."""
    specs = [MetricSpec(metric=name, threshold=threshold, higher_is_better=True) for name in per_item]
    try:
        report = evaluate_gate(
            base_run_id="bm25",
            head_run_id="dense",
            suite_id="panel_dense",
            per_item=per_item,
            specs=specs,
            alpha=alpha,
            seed=seed,
        )
    except ThresholdBelowMDEError as exc:
        boot = paired_bootstrap(*per_item[exc.metric], alpha=alpha / len(specs), seed=seed)
        return {
            "outcome": "refused",
            "threshold": threshold,
            "refused_metric": exc.metric,
            "mde": exc.mde,
            "delta": boot.delta,
            "n": boot.n,
            "detail": (
                f"threshold {threshold} < mde {exc.mde:.4f} on n={boot.n}; no GateReport was "
                "produced. A refusal is not a pass."
            ),
        }
    return {
        "outcome": "block" if report.blocked else "pass",
        "threshold": threshold,
        "alpha": report.alpha,
        "power": report.power,
        "b": report.b,
        "seed": report.seed,
        "n_metrics": len(report.results),
        "results": [
            {
                "pair": r.metric,
                "n": r.n,
                "bm25_mean": r.base_mean,
                "dense_mean": r.head_mean,
                "delta": r.delta,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "p_value": r.p_value,
                "p_adjusted": r.p_adjusted,
                "mde": r.mde,
                "verdict": r.verdict,
            }
            for r in report.results
        ],
    }


def print_table(result: dict[str, object]) -> None:
    if result["outcome"] == "refused":
        print(f"REFUSED at threshold {result['threshold']}: {result['detail']}")
        return
    header = (
        f"{'pair':34} {'n':>5} {'bm25':>7} {'dense':>7} {'delta':>8} "
        f"{'ci':>19} {'p_holm':>8} {'mde':>7} verdict"
    )
    print(header)
    print("-" * len(header))
    for r in result["results"]:  # type: ignore[union-attr]
        ci = f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
        print(
            f"{r['pair']:34} {r['n']:>5} {r['bm25_mean']:>7.4f} {r['dense_mean']:>7.4f} "
            f"{r['delta']:>+8.4f} {ci:>19} {r['p_adjusted']:>8.4f} {r['mde']:>7.4f} {r['verdict']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matrix", type=Path, default=_RUN / "panel_dense_matrix.csv")
    parser.add_argument("--out", type=Path, default=_RUN / "dense_vs_bm25.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refusal-probe", type=float, default=PROBE_THRESHOLD)
    args = parser.parse_args(argv)

    item_ids, columns = load_columns(args.matrix)
    per_item = build_pairs(columns)
    print(f"{len(item_ids)} items; {len(per_item)} matched (chunking, top_k) pair(s)")

    main_result = run_gate(per_item, args.threshold, alpha=args.alpha, seed=args.seed)
    print_table(main_result)

    probe = run_gate(per_item, args.refusal_probe, alpha=args.alpha, seed=args.seed)
    print(
        f"\nrefusal probe at threshold {args.refusal_probe}: {probe['outcome']}"
        + (f" ({probe['detail']})" if probe["outcome"] == "refused" else "")
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "matrix": str(args.matrix.name),
                "n_items": len(item_ids),
                "preregistered_threshold": DEFAULT_THRESHOLD,
                "gate": main_result,
                "refusal_probe": probe,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
