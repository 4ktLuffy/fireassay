#!/usr/bin/env python3
"""Report the minimum detectable effect (MDE) curve for a response-matrix
suite, at a range of candidate suite sizes -- the pre-registration check
docs/SPEC.md §8 describes ("`fireassay power` computes required n from
desired MDE, alpha and metric count; the gate checks its own thresholds
against the suite's MDE"), run as a standalone report rather than inside
the gate itself.

**MDE is a distribution over system pairs, not a single number, and this
tool reports it as one.** `fireassay.gate.mde` takes one `se` -- the
paired bootstrap standard error of one (base system, head system, metric)
comparison -- and `se` itself depends on how correlated that particular
pair's per-item correctness is (`items.core`'s "a hard question is hard
for both" fact, restated here one level up: how strongly correlated *two
systems'* errors are, not just whether one item is hard). Two systems
that agree on almost everything give a small `se` and a small MDE; two
systems whose failures are nearly uncorrelated give a larger one. Quoting
a single MDE for a suite therefore silently picks one system pair (usually
without saying so) and presents its answer as if it were universal. This
tool instead draws a sample of system pairs at each suite size and reports
the **median** MDE across them plus the 10th/90th percentiles -- the
honest form of the number, and the one that shows how much it moves with
the pair chosen, not just its typical value.

Usage:

    .venv/bin/python tools/mde.py --matrix run/panel54_matrix.csv \\
        --sizes 150,200,300,600,1000 --alpha 0.05 --power 0.80 \\
        --b 2000 --seed 20260830

`--matrix` is a response-matrix CSV, header exactly `item_id,system_id,
correct` (see `fireassay.items.adapters.tabular`'s module docstring for
the full contract). `--sizes` is a comma-separated list of suite sizes to
report; a size larger than the matrix's own item count is skipped with a
warning rather than silently truncated. `--pairs` (default 200) caps how
many system pairs are sampled per suite size, without replacement, from
all `C(n_systems, 2)` possible pairs -- capped further, with a warning, if
fewer pairs exist than requested. `--b` is the bootstrap replicate count
passed to `fireassay.gate.paired_bootstrap` for every pair (kept modest
by default since this tool calls it `len(sizes) * pairs` times).

This module imports `paired_bootstrap`/`mde` from `fireassay.gate` rather
than reimplementing either: the statistics have exactly one implementation
in this repository, and this tool is a report over it, not a second one.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fireassay.gate import mde as compute_mde
from fireassay.gate import paired_bootstrap
from fireassay.items.adapters.tabular import load_responses_csv
from fireassay.items.core import ItemResponses


def _read_system_order(path: Path) -> list[str]:
    """First-seen `system_id` order in a response-matrix CSV, mirroring
    `fireassay.items.adapters.tabular`'s own ordering rule (its module
    docstring: "System order ... is first-seen order across the file").
    Duplicated minimally here for the same reason `tools/evidence.py`
    duplicates it: `load_responses_csv` validates and returns
    `ItemResponses` but does not expose the column order it used
    internally, and this tool needs to map each `ItemResponses.responses[i]`
    back to the system id it came from."""
    order: list[str] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system_id = str(row["system_id"])
            if system_id not in seen:
                seen.add(system_id)
                order.append(system_id)
    return order


def _parse_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            n = int(chunk)
        except ValueError as exc:
            raise ValueError(f"--sizes: {chunk!r} is not an integer") from exc
        if n <= 0:
            raise ValueError(f"--sizes: suite size must be positive, got {n}")
        sizes.append(n)
    if not sizes:
        raise ValueError("--sizes: must name at least one suite size")
    return sizes


def mde_curve(
    responses: Sequence[ItemResponses],
    n_systems: int,
    sizes: Sequence[int],
    *,
    alpha: float,
    power: float,
    b: int,
    pairs: int,
    seed: int,
) -> dict[int, tuple[float, float, float]]:
    """For each suite size in `sizes`: draw that many items without
    replacement, sample up to `pairs` system pairs without replacement
    from all `C(n_systems, 2)` possible pairs, and compute each pair's
    `paired_bootstrap(...).se` -> `mde(se, alpha=alpha, power=power)` on
    the `correct` column of the two systems restricted to the drawn
    items. Returns `{n: (median_mde, p10_mde, p90_mde)}`.

    A single `numpy.random.default_rng(seed)` drives every random choice
    below (which items, which pairs, and each pair's own bootstrap
    resampling) so the whole curve is reproducible from one integer, per
    this tool's module docstring.
    """
    if n_systems < 2:
        raise ValueError(f"mde_curve: need at least 2 systems to form a pair, got {n_systems}")
    all_pairs = list(itertools.combinations(range(n_systems), 2))
    rng = np.random.default_rng(seed)

    curve: dict[int, tuple[float, float, float]] = {}
    for n in sizes:
        if n > len(responses):
            print(
                f"warning: --sizes {n} exceeds the matrix's {len(responses)} available items; skipped",
                file=sys.stderr,
            )
            continue
        item_positions = rng.choice(len(responses), size=n, replace=False)
        sampled = [responses[i] for i in item_positions]

        n_pairs = min(pairs, len(all_pairs))
        if n_pairs < pairs:
            print(
                f"warning: --pairs {pairs} exceeds the {len(all_pairs)} possible system pairs "
                f"for n={n}; using {n_pairs}",
                file=sys.stderr,
            )
        pair_positions = rng.choice(len(all_pairs), size=n_pairs, replace=False)

        pair_mdes: list[float] = []
        for pos in pair_positions:
            sys_a, sys_b = all_pairs[int(pos)]
            base = [1.0 if item.responses[sys_a] else 0.0 for item in sampled]
            head = [1.0 if item.responses[sys_b] else 0.0 for item in sampled]
            boot_seed = int(rng.integers(0, 2**31 - 1))
            se = paired_bootstrap(base, head, b=b, alpha=alpha, seed=boot_seed).se
            pair_mdes.append(compute_mde(se, alpha=alpha, power=power))

        arr = np.asarray(pair_mdes, dtype=float)
        median = float(np.median(arr))
        p10 = float(np.percentile(arr, 10))
        p90 = float(np.percentile(arr, 90))
        curve[n] = (median, p10, p90)
    return curve


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matrix", required=True, type=Path, help="response-matrix CSV")
    parser.add_argument("--sizes", required=True, help="comma-separated suite sizes, e.g. 150,200,300")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--b", type=int, default=2000, help="bootstrap replicates per system pair")
    parser.add_argument("--pairs", type=int, default=200, help="system pairs sampled per suite size")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        sizes = _parse_sizes(args.sizes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    responses = load_responses_csv(args.matrix)
    system_order = _read_system_order(args.matrix)

    print(f"{len(responses)} items x {len(system_order)} systems; alpha={args.alpha} power={args.power}")
    print("MDE is a distribution over system pairs (see module docstring) -- median with 10th/90th pct")
    print(f"{'n':>6}  {'median_mde':>12}  {'p10':>10}  {'p90':>10}")

    curve = mde_curve(
        responses,
        len(system_order),
        sizes,
        alpha=args.alpha,
        power=args.power,
        b=args.b,
        pairs=args.pairs,
        seed=args.seed,
    )
    for n in sizes:
        if n not in curve:
            continue
        median, p10, p90 = curve[n]
        print(f"{n:>6}  {median:>12.4f}  {p10:>10.4f}  {p90:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
