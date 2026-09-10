#!/usr/bin/env python3
"""EVIDENCE.md's generator and verifier -- every claim mapped to a command
and a committed artifact (Goal 1, criterion 8,
`~/ObitosBrain/notes/2026-08-28-fireassay-eval-integrity.md`).

This exists because of two defects in that same log. **Defect 46:** a
load-bearing number (24 `mislabel_suspect` items) came from an ad-hoc
script that was never committed, reached the handbook, the plan and a
spec, and turned out to be 32 when re-run on byte-identical data.
**Defect 43:** a `0%` recall was quoted as a finding when it was an
artifact of the experiment, not a property of the detector. A commit
message is prose. Prose cannot be re-run.

## The design -- generated, never hand-written

`EVIDENCE.md` is not typed by hand: a hand-maintained table of numbers
rots away from the code that produced it, which is the same failure one
level up. Every claim below is one `Claim` record -- id, one-line
description, the committed artifact(s) it reads, a `recompute` function,
and the expected value(s) with a tolerance -- and this file offers three
commands over that list:

    .venv/bin/python tools/evidence.py --check       recompute every claim,
                                                       compare to expected,
                                                       print PASS/FAIL per
                                                       claim, exit non-zero
                                                       on any FAIL or ERROR
    .venv/bin/python tools/evidence.py --markdown     emit EVIDENCE.md to
                                                       stdout (regenerate with
                                                       `--markdown > EVIDENCE.md`)
    .venv/bin/python tools/evidence.py --claim ID     recompute one claim and
                                                       print it in detail

**A claim whose recompute function disagrees with its expected value is a
FAIL, not a silently updated number.** `--check` never rewrites a `Claim`
record to match whatever it just computed -- that would reproduce exactly
the failure this file exists to prevent, one layer further in. A FAIL is
the correct, intended outcome when reality has moved and the code has
not; it is a finding, not a bug in this file.

`--markdown` does **not** recompute anything -- it formats the claim
records (description, artifacts, expected value, tolerance, and the exact
command to verify each one) that are already registered below. Only
`--check` and `--claim` call a claim's `recompute`. This keeps
`--markdown` cheap and keeps the split honest: this file is where the
numbers are asserted, `--check` is where they are verified, and the
distinction between "documented" and "verified" is never allowed to blur.

## Every recompute function reads only committed artifacts under `run/`

If a claim needed `run/golden.db` (gitignored -- see `.gitignore`), it
would not belong here: shipping a command a stranger cannot run is the
defect-46 shape with different variable names. No claim below touches
`run/golden.db`. `run/items_meta.jsonl` (committed, written by
`tools/export_items_meta.py`) is where item text/reference answers/
evidence quotes live for claims that would otherwise need the database.

`--check` must not require `ppi_py`: it is a dev-only optional dependency
(`pyproject.toml` `[project.optional-dependencies].dev`), cross-checked
against the hand-rolled PPI estimator only in `tests/test_items_ppi.py`
(`pytest.importorskip("ppi_py")`). This module never imports it.

`fisher_exact_two_sided` below hand-rolls a two-sided Fisher exact test
from `math.comb` for the same reason PPI is hand-rolled in
`fireassay.items.ppi` (Decision 5 in the project notes): `scipy` is not a
project dependency (see `pyproject.toml`), and a stranger running
`--check` should not need to install one just to verify a p-value.

## Claims deliberately left out (documented, not silently dropped)

- **`judge.nli`'s Fisher exact p-value** (handbook §9b quotes p=0.169,
  comparing the NLI judge's flagged-precision against a random-draw
  comparison group): unlike `detector.mislabel_suspect`, whose comparison
  group's exact counts are a committed, tested constant
  (`fireassay.items.core.MISLABEL_SUSPECT_VALIDATION`), the NLI
  comparison group's sampling design (which batch, what `n`) is not
  recorded in any committed artifact. `judge.nli`'s precision is still
  checked below (`judge.nli`); the Fisher p-value is not.

## Judge precision is a stratum choice, not a universal constant

Every `judge.*` claim (`judge.granite4`, `judge.gpt54`, `judge.nli`,
`judge.token_overlap`) computes precision over the 132 uniform-random
calibration-stratum labels only, never over all 176 labelled items --
see `_judge_precision_recall`'s docstring for the full reasoning, echoed
briefly in each claim's own recompute docstring. In short: the other 44
labels are a census hand-selected *by* the old `mislabel_suspect`
detector (`items.review._plan_batch` claims flagged items first) plus a
batch of unflagged controls, not a random sample -- a judge whose
verdicts correlate with that detector's own selection would silently
inherit its selection bias if scored against that enriched set. The
unbiased 132-item random draw is the only stratum a judge-vs-judge
precision *comparison* can trust.

This is a different question from what `detector.mislabel_suspect`/
`detector.prereg_answer_unsupported`/`detector.prereg_answer_echoes_evidence`
below ask, where `fireassay.items.calibration.evaluate_detector`'s own
all-strata precision rule is correct and deliberately unchanged: each of
those scores *one* detector's own flagged set against every label it
has, and stratum enrichment does not bias one detector's own precision
(`items.calibration`'s module docstring's stratum table: "enrichment
does not bias `precision`"). Comparing several *different* judges against
each other, however, needs every judge measured on the same, unbiased
sample, or one judge's correlation with the old detector's selection
would leak into how every other judge compares against it.

Both readings shipped without anyone noticing they disagreed until
`--check` caught it: `run/validations.jsonl`'s single persisted record
for `answerability.not_supported_or_unclear` used the all-176 rule
(66.7%), while the eval-integrity notes' judges table used the 132-random
rule (65.2%) for the same judge -- two different, individually
defensible computations, silently presented as the same number.

## One number below disagrees with the prose it was copied from

`closedbook.forced`: `run/closedbook_forced.log`'s own printed `RESULT`
line reads `2/149 = 1.3%`, not the `1.4%` (`2/147`) figure quoted in
`~/ObitosBrain/notes/2026-08-28-fireassay-eval-integrity.md`. Reported
here as what the committed artifact itself says -- a `--check` FAIL here
would be correct, not a bug, if the notes file's prose is what a reader
trusts instead.

Not itself a `test_*.py` module -- `--check` is a verification command
that reads large run artifacts (`run/panel54_matrix.csv` alone is
2,364 items x 54 systems) and is deliberately excluded from
`tests/test_evidence.py` (see that module's docstring for the same
reasoning `tests/helpers.py` documents for keeping heavy fixtures out of
the regular suite).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fireassay.evidence import Claim, ClaimResult, EvidenceError, MarkdownStyle, Metric
from fireassay.evidence import main as _evidence_main
from fireassay.evidence import render_markdown as _render_markdown
from fireassay.gate import MetricSpec, ThresholdBelowMDEError, evaluate_gate, paired_bootstrap
from fireassay.items.adapters.tabular import load_meta_jsonl, load_responses_csv
from fireassay.items.baseline import answer_unsupported
from fireassay.items.calibration import CalibrationLabel, evaluate_detector, load_calibration_jsonl
from fireassay.items.core import MISLABEL_SUSPECT_VALIDATION, ItemResponses, analyse, wilson_ci
from fireassay.items.ppi import ppi_mean_ci

_RUN = Path("run")

#: Kelley's classical top/bottom split fraction, mirrored from
#: `fireassay.items.core._EXTREME_GROUP_FRACTION` (private there) --
#: `panel.strength_is_breadth` needs the same "top/bottom N configs"
#: split `analyse` itself uses internally, but `analyse` never returns
#: which *systems* landed in the extreme groups, only the resulting
#: per-item statistics -- so this claim recomputes the split directly
#: over per-system totals instead of over per-item point-biserials.
_EXTREME_GROUP_FRACTION = 0.27


# ---------------------------------------------------------------------------
# Fisher exact test, hand-rolled (no scipy -- see module docstring)
# ---------------------------------------------------------------------------


def _hypergeom_pmf(a: int, row1: int, row2: int, col1: int) -> float:
    """P(X = a) for the hypergeometric distribution implied by a 2x2
    table's fixed margins (`row1`, `row2`, `col1`; `col2` and `n` follow).
    Shared by `fisher_exact_two_sided` for every table sharing those
    margins."""
    n = row1 + row2
    return math.comb(row1, a) * math.comb(row2, col1 - a) / math.comb(n, col1)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for the 2x2 table
    `[[a, b], [c, d]]`, hand-rolled from `math.comb` (stdlib) -- see the
    module docstring for why this project does not depend on `scipy`.

    Standard definition: sum the hypergeometric probability of every table
    with the same margins whose probability is no greater than the
    observed table's own probability (times `1 + eps`, so the observed
    table itself is never excluded by floating-point rounding)."""
    row1, row2 = a + b, c + d
    col1 = a + c
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    observed = _hypergeom_pmf(a, row1, row2, col1)
    eps = 1e-7
    threshold = observed * (1 + eps)
    return sum(
        p
        for k in range(lo, hi + 1)
        if (p := _hypergeom_pmf(k, row1, row2, col1)) <= threshold
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_system_order(path: Path) -> list[str]:
    """First-seen `system_id` order in a response-matrix CSV, mirroring
    `fireassay.items.adapters.tabular`'s own ordering rule (its module
    docstring: "System order ... is first-seen order across the file").
    Duplicated minimally here because `load_responses_csv` validates and
    returns `ItemResponses` but does not expose the column order it used
    internally -- `panel.monotonic`/`panel.strength_is_breadth` need to
    map each `ItemResponses.responses[i]` back to the system id it came
    from."""
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


def _extreme_group_size(n_systems: int) -> int:
    """Mirrors `fireassay.items.core._extreme_group_size` (private
    there) -- see `_EXTREME_GROUP_FRACTION` above."""
    n = max(1, round(n_systems * _EXTREME_GROUP_FRACTION))
    return min(n, n_systems // 2)


def _load_verdicts_jsonl(path: Path) -> dict[str, str]:
    """`item_id -> verdict` from any answerability-shaped JSONL
    (`fireassay.items.answerability.AnswerabilityVerdict` plus whatever
    extra fields a given judge run adds, e.g. `judge`/`model`/`effort`).
    Iterates the open file handle, never `read_text().splitlines()` --
    the gov.uk-corpus U+2028 trap documented in
    `fireassay.items.adapters.tabular`'s module docstring applies to any
    JSONL this project reads, this file included."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            item_id = obj["item_id"]
            verdict = obj["verdict"]
            out[str(item_id)] = str(verdict)
    return out


def _flagged_not_supported_or_unclear(verdicts: dict[str, str]) -> set[str]:
    """The shared "flagged" rule every answerability-style judge in this
    file uses: anything that is not a clean `"supported"` verdict --
    `"not_supported"` or `"unclear"` both count, matching
    `run/flags_answerability.txt`'s own header comment."""
    return {item_id for item_id, verdict in verdicts.items() if verdict != "supported"}


def _load_id_list(path: Path) -> set[str]:
    """One item id per non-blank, non-`#`-comment line -- the shape of
    `run/flags_mislabel_suspect.txt`, `run/prereg_answer_unsupported.txt`
    and `run/prereg_answer_echoes_evidence.txt`."""
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line)
    return ids


def _estimate_value(estimate_value: float | None, *, where: str) -> float:
    """Unwrap an `Estimate.value` for a claim that assumes it is never
    `None` (i.e. the claim's own denominator is known, from the numbers
    above, to be positive) -- raises `EvidenceError` naming the claim
    rather than silently substituting a fabricated `0.0` if that
    assumption is ever violated by a future run's data."""
    if estimate_value is None:
        raise EvidenceError(f"{where}: Estimate.value is None (no_denominator) -- cannot check a number")
    return estimate_value


def _calibration_random_stratum(calibration_path: Path) -> list[CalibrationLabel]:
    """The 132 uniform-random `stratum == "calibration"` labels only, out
    of `run/calibration.jsonl`'s full 176 -- the unbiased sample every
    `judge.*` precision claim below is measured against. Excludes the
    other 44 labels, which are a census hand-selected *by* the old
    `mislabel_suspect` detector (`items.review._plan_batch` claims
    flagged items first) plus a batch of unflagged controls, not a random
    draw -- see `_judge_precision_recall`'s docstring for why mixing them
    in would let that detector's own selection bias leak into every
    judge's precision figure."""
    labels = load_calibration_jsonl(calibration_path)
    return [label for label in labels if label.stratum == "calibration"]


def _judge_precision_recall(verdicts_path: Path, calibration_path: Path) -> ClaimResult:
    """Shared recompute for every "flag anything not `supported`, score
    against the human calibration labels" judge claim
    (`judge.granite4`/`judge.gpt54`/`judge.nli`) -- differs claim to claim
    only in which JSONL of judge verdicts it reads.

    **Precision (like recall) is computed over the 132 uniform-random
    calibration-stratum labels only, never all 176.** The other 44 are a
    census hand-selected *by* the old `mislabel_suspect` detector
    (`items.review._plan_batch` claims flagged items first) plus a batch
    of unflagged controls -- not a random sample of the suite. Any judge
    whose verdicts correlate at all with that detector's own selection
    rule would inherit its selection bias if scored against that enriched
    44, silently moving precision for a reason that has nothing to do
    with the judge's actual accuracy. `fireassay.items.calibration.
    evaluate_detector`'s own all-strata precision rule is correct for its
    own purpose -- scoring *one* detector's flagged set, where enrichment
    does not bias precision (`items.calibration`'s own module docstring)
    -- but a *comparison* across several different judges needs every one
    of them measured on the same unbiased draw, which is why this
    function restricts `labels` to the calibration stratum with
    `_calibration_random_stratum` before `evaluate_detector` ever sees
    them, rather than calling `evaluate_detector` on the full label set
    the way the single-detector claims below correctly still do. This
    project shipped both readings without noticing they disagreed until
    `--check` caught it -- see the module docstring's "Judge precision is
    a stratum choice" section."""
    verdicts = _load_verdicts_jsonl(verdicts_path)
    flagged = _flagged_not_supported_or_unclear(verdicts)
    random_stratum_labels = _calibration_random_stratum(calibration_path)
    evaluation = evaluate_detector(flagged, random_stratum_labels)
    precision = _estimate_value(evaluation.precision.value, where=f"{verdicts_path}: precision")
    recall = _estimate_value(evaluation.recall.value, where=f"{verdicts_path}: recall")
    detail = (
        f"{len(flagged)} flagged of {len(verdicts)} judged; "
        f"precision={precision:.3f} (n={evaluation.precision.n}), "
        f"recall={recall:.3f} (n={evaluation.recall.n}), "
        f"stratum_estimator={evaluation.stratum_estimator}"
    )
    return ClaimResult(values={"precision": precision, "recall": recall}, detail=detail)


def _rate_over_calibration_stratum(
    labels: Sequence[CalibrationLabel], bad_verdicts: frozenset[str]
) -> tuple[float, tuple[float, float], int, int]:
    """`(rate, ci, bad_count, n)` of `label.verdict in bad_verdicts` over
    `labels` restricted to `stratum == "calibration"` -- the suite's only
    unbiased random sample (see `fireassay.items.calibration`'s module
    docstring's stratum table). Deliberately not routed through
    `evaluate_detector` (whose `base_rate` requires a non-empty
    `flagged_ids` to avoid the disjoint-stratum trap firing on a
    trivially-empty flagged set -- see that module's docstring): this is
    a plain proportion over one stratum, nothing scaled or stratified."""
    calibration = [label for label in labels if label.stratum == "calibration"]
    n = len(calibration)
    bad = sum(1 for label in calibration if label.verdict in bad_verdicts)
    rate = bad / n if n else 0.0
    return rate, wilson_ci(bad, n), bad, n


# ---------------------------------------------------------------------------
# Panel (run/panel54_matrix.csv) and old panel (run/panel_matrix.csv)
# ---------------------------------------------------------------------------


def _load_panel(path: Path) -> tuple[list[ItemResponses], list[str]]:
    return load_responses_csv(path), _read_system_order(path)


def _recompute_panel_shape() -> ClaimResult:
    responses, system_order = _load_panel(_RUN / "panel54_matrix.csv")
    return ClaimResult(
        values={"n_items": float(len(responses)), "n_systems": float(len(system_order))},
        detail=f"{len(responses)} items x {len(system_order)} systems",
    )


def _recompute_panel_classes() -> ClaimResult:
    responses, _system_order = _load_panel(_RUN / "panel54_matrix.csv")
    _item_stats, panel_stats = analyse(responses)
    values: dict[str, float | str] = {k: float(v) for k, v in panel_stats.class_counts.items()}
    return ClaimResult(values=values, detail=str(panel_stats.class_counts))


def _recompute_panel_reliability() -> ClaimResult:
    responses, _system_order = _load_panel(_RUN / "panel54_matrix.csv")
    _item_stats, panel_stats = analyse(responses)
    return ClaimResult(
        values={
            "reliability": panel_stats.split_half_reliability,
            "verdict": panel_stats.reliability_verdict,
        },
        detail=f"n_items_used_for_reliability={panel_stats.n_items_used_for_reliability}",
    )


def _recompute_panel_monotonic() -> ClaimResult:
    """Within one `(retriever, chunking)` family -- e.g. every `bm25/512-32/k*`
    system -- a `top_k=3` retrieval set is a subset of that same family's
    `top_k=10` set by construction (truncating one fixed ranking further
    down can only add hits, never remove one; the same fact
    `fireassay.score.invariants._check_recall_monotonic` checks per-question
    for `retrieval.recall@k`). This claim checks the panel-level
    consequence: no `(item, family)` pair should ever see a `top_k=3`
    correct/`top_k=5` incorrect (or `top_k=5` correct/`top_k=10` incorrect)
    disagreement.

    **Scope note:** defect 49 (`~/ObitosBrain/notes/2026-08-28-fireassay-eval-
    integrity.md`) and `fireassay.system.panel`'s module docstring both quote
    this measurement as "14,184 (item, chunking) pairs" -- but that number
    was measured on the single-retriever predecessor panel
    (`run/panel_matrix.csv`, defect 49's own subject, built *before*
    `panel54_matrix.csv` existed). That panel's systems are labelled
    `sys00..sys17` positionally (`fireassay.system.panel`'s own module
    docstring: "which is exactly why the top_k ladder went unnoticed"), so
    there is no way to recover which system is which `(retriever,
    chunking, top_k)` from `run/panel_matrix.csv` alone -- the 14,184
    figure cannot be reproduced from a committed artifact with a
    recoverable grouping. `run/panel54_matrix.csv`'s system ids are
    self-describing (`bm25/512-32/k5`), so this claim instead checks the
    full, larger, self-describing set: 2,364 items x 18 `(retriever,
    chunking)` families = 42,552 pairs. The historical 14,184-pair figure
    is a subset of what is checked here, not a different claim.
    """
    responses, system_order = _load_panel(_RUN / "panel54_matrix.csv")
    families: dict[str, dict[str, int]] = {}
    for index, system_id in enumerate(system_order):
        family, _sep, k = system_id.rpartition("/")
        families.setdefault(family, {})[k] = index

    k_order = ("k3", "k5", "k10")
    n_pairs = 0
    n_violations = 0
    n_families_checked = 0
    for _family, by_k in families.items():
        if not all(k in by_k for k in k_order):
            continue
        n_families_checked += 1
        idx3, idx5, idx10 = by_k["k3"], by_k["k5"], by_k["k10"]
        for item in responses:
            n_pairs += 1
            r3, r5, r10 = item.responses[idx3], item.responses[idx5], item.responses[idx10]
            if (r3 and not r5) or (r5 and not r10):
                n_violations += 1

    detail = (
        f"{n_violations} violation(s) in {n_pairs} (item, chunking-family) pairs "
        f"across {n_families_checked} families (retriever x size-overlap); "
        "historical figure (defect 49): 14,184 pairs on the single-retriever "
        "predecessor panel, not independently reproducible -- see this claim's docstring"
    )
    return ClaimResult(values={"violations": float(n_violations)}, detail=detail)


def _recompute_panel_strength_is_breadth() -> ClaimResult:
    """Of the top/bottom `_extreme_group_size(54) = 15` configs by total
    score across all 2,364 items, how many are `k=10` vs `k=3` --
    defect 49's mechanism: within a `top_k`-only ladder, the widest
    configs always score highest and the narrowest always score lowest,
    so "strength" and "breadth" (retrieval window size) become
    indistinguishable. Checked as two directional facts (top skews `k=10`
    over `k=3`; bottom skews `k=3` over `k=10`), not exact counts -- the
    spec this claim is drawn from reports counts without pinning specific
    values for the full 2,364-item panel (only a 400-item probe run,
    `run/panel54_probe.log`, has committed exact figures, and those are a
    different, smaller sample). The full breakdown is in `detail`."""
    responses, system_order = _load_panel(_RUN / "panel54_matrix.csv")
    n_systems = len(system_order)
    totals = [0.0] * n_systems
    for item in responses:
        for index, correct in enumerate(item.responses):
            if correct:
                totals[index] += 1.0

    n_extreme = _extreme_group_size(n_systems)
    order = sorted(range(n_systems), key=lambda i: -totals[i])
    top = order[:n_extreme]
    bottom = order[-n_extreme:]

    def k_of(system_id: str) -> str:
        return system_id.rpartition("/")[2]

    top_k = Counter(k_of(system_order[i]) for i in top)
    bottom_k = Counter(k_of(system_order[i]) for i in bottom)
    top_favors_k10 = 1.0 if top_k.get("k10", 0) > top_k.get("k3", 0) else 0.0
    bottom_favors_k3 = 1.0 if bottom_k.get("k3", 0) > bottom_k.get("k10", 0) else 0.0

    detail = f"n_extreme={n_extreme}; top={dict(top_k)}; bottom={dict(bottom_k)}"
    return ClaimResult(
        values={"top_favors_k10": top_favors_k10, "bottom_favors_k3": bottom_favors_k3},
        detail=detail,
    )


def _recompute_panel18_classes() -> ClaimResult:
    """`run/panel_matrix.csv` re-analysed fresh via the shipped
    `fireassay.items.core.analyse` -- deliberately **not** read from
    `run/full_panel.log`, whose printed classification (24 `mislabel_suspect`,
    1.0%) is exactly defect 46's stale, pre-fix number. `analyse` on this
    same, byte-identical, committed CSV gives 32 (1.4%) -- see
    `fireassay.items.core.MISLABEL_SUSPECT_VALIDATION` and
    `run/flags_mislabel_suspect.txt`. Cross-checked independently against
    `run/panel54_probe.log`'s own printed baseline line ("OLD 18-config
    panel on 2,364: dead_all_pass 53.4% live 40.4% mislabel 1.4%
    reliability 0.286"), which matches these expected values exactly."""
    responses, _system_order = _load_panel(_RUN / "panel_matrix.csv")
    _item_stats, panel_stats = analyse(responses)
    values: dict[str, float | str] = {k: float(v) for k, v in panel_stats.class_counts.items()}
    return ClaimResult(values=values, detail=str(panel_stats.class_counts))


def _recompute_panel18_reliability() -> ClaimResult:
    responses, _system_order = _load_panel(_RUN / "panel_matrix.csv")
    _item_stats, panel_stats = analyse(responses)
    return ClaimResult(
        values={
            "reliability": panel_stats.split_half_reliability,
            "verdict": panel_stats.reliability_verdict,
        },
        detail=f"n_items_used_for_reliability={panel_stats.n_items_used_for_reliability}",
    )


# ---------------------------------------------------------------------------
# Human labels (run/calibration.jsonl)
# ---------------------------------------------------------------------------


def _recompute_labels_counts() -> ClaimResult:
    labels = load_calibration_jsonl(_RUN / "calibration.jsonl")
    n_calibration = sum(1 for label in labels if label.stratum == "calibration")
    return ClaimResult(
        values={"n_labels": float(len(labels)), "n_calibration": float(n_calibration)},
        detail=f"{len(labels)} total, {n_calibration} in the calibration (uniform-random) stratum",
    )


def _recompute_labels_base_rate() -> ClaimResult:
    labels = load_calibration_jsonl(_RUN / "calibration.jsonl")
    rate, ci, bad, n = _rate_over_calibration_stratum(labels, frozenset({"purge"}))
    return ClaimResult(
        values={"rate": rate, "ci_lo": ci[0], "ci_hi": ci[1]},
        detail=f"{bad}/{n} purge, wilson_ci={ci}",
    )


def _recompute_labels_lenient_rate() -> ClaimResult:
    labels = load_calibration_jsonl(_RUN / "calibration.jsonl")
    rate, ci, bad, n = _rate_over_calibration_stratum(labels, frozenset({"purge", "rewrite"}))
    return ClaimResult(values={"rate": rate}, detail=f"{bad}/{n} purge-or-rewrite, wilson_ci={ci}")


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


def _recompute_judge_granite4() -> ClaimResult:
    """Precision and recall both computed over the 132 uniform-random
    calibration-stratum labels, not all 176 -- see
    `_judge_precision_recall`'s docstring for why (the other 44 are a
    census selected by the old `mislabel_suspect` detector, whose
    selection a correlated judge would otherwise inherit)."""
    return _judge_precision_recall(_RUN / "answerability.jsonl", _RUN / "calibration.jsonl")


def _recompute_judge_gpt54() -> ClaimResult:
    """Uses `run/answerability_codex_all.jsonl` (gpt-5.4/low over all
    2,364 items), not the narrower `run/answerability_codex.jsonl` (the
    same judge/model/effort, but only the 176 calibration-labelled
    items) -- either file gives the same precision/recall here, since
    both are restricted to the 132 uniform-random calibration-stratum
    labels before scoring (see `_judge_precision_recall`'s docstring);
    `answerability_codex_all.jsonl` is used because `ppi.defect_rate`
    needs its full 2,364-item coverage anyway, so one file serves both
    claims. **This is the pair the eval-integrity notes' judges table
    and `run/validations.jsonl` disagreed about** (65.2% vs 66.7%) until
    `--check` caught it: `run/validations.jsonl`'s persisted record used
    the all-176 rule; this claim, like every other `judge.*` claim, uses
    the 132-random rule, which matches the notes' 65.2% exactly."""
    return _judge_precision_recall(_RUN / "answerability_codex_all.jsonl", _RUN / "calibration.jsonl")


def _recompute_judge_nli() -> ClaimResult:
    """Precision only, over the 132 uniform-random calibration stratum
    (see `_judge_precision_recall`'s docstring) -- see the module
    docstring's "Claims deliberately left out" section for why the
    handbook's Fisher p=0.169 for this judge is not checked here."""
    return _judge_precision_recall(_RUN / "answerability_nli.jsonl", _RUN / "calibration.jsonl")


def _recompute_judge_token_overlap() -> ClaimResult:
    """The no-model token-overlap baseline
    (`fireassay.items.baseline.answer_unsupported`, defect 46's fix --
    see that module's docstring) scored against the 132 uniform-random
    calibration-stratum labels, reading item text from
    `run/items_meta.jsonl` (committed; `run/golden.db`, where this text
    otherwise lives, is not -- see this module's own docstring).
    Precision only, over the same 132-random stratum every other
    `judge.*` claim uses (see `_judge_precision_recall`'s docstring) --
    `answer_unsupported` has no published recall figure to check
    against, only precision (handbook §9b)."""
    meta_by_id = load_meta_jsonl(_RUN / "items_meta.jsonl")
    flagged = {item_id for item_id, meta in meta_by_id.items() if answer_unsupported(meta)}
    random_stratum_labels = _calibration_random_stratum(_RUN / "calibration.jsonl")
    evaluation = evaluate_detector(flagged, random_stratum_labels)
    precision = _estimate_value(evaluation.precision.value, where="judge.token_overlap: precision")
    detail = (
        f"{len(flagged)} flagged of {len(meta_by_id)} items; "
        f"precision={precision:.3f} (n={evaluation.precision.n})"
    )
    return ClaimResult(values={"precision": precision}, detail=detail)


def _recompute_judge_cache_consistency() -> ClaimResult:
    """`run/answerability_codex.jsonl` (176 items) and
    `run/answerability_codex_all.jsonl` (2,364 items, a superset) were
    both produced by the identical judge/model/effort
    (`gpt-5.4`/`low`) against the same on-disk cache
    (`.cache/judge-codex/responses.jsonl`) -- every item id present in
    both files is therefore a cache hit on the second run, and a cache
    that is doing its job returns the identical verdict both times.
    Checks the full intersection, not a hand-picked 176 -- if the two
    files' overlap were ever smaller or larger than 176, that would
    itself be worth seeing in `detail`."""
    small = _load_verdicts_jsonl(_RUN / "answerability_codex.jsonl")
    full = _load_verdicts_jsonl(_RUN / "answerability_codex_all.jsonl")
    shared = sorted(set(small) & set(full))
    if not shared:
        raise EvidenceError(
            "judge.cache_consistency: no item_id is shared between "
            "run/answerability_codex.jsonl and run/answerability_codex_all.jsonl"
        )
    agree = sum(1 for item_id in shared if small[item_id] == full[item_id])
    rate = agree / len(shared)
    return ClaimResult(
        values={"agree_rate": rate, "n_shared": float(len(shared))},
        detail=f"{agree}/{len(shared)} identical verdicts across the shared item ids",
    )


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _recompute_detector_mislabel_suspect() -> ClaimResult:
    """Reads `fireassay.items.core.MISLABEL_SUSPECT_VALIDATION` -- itself
    committed, tested source, and the module docstring's own citation for
    these exact numbers (defect 50) -- for the flagged/random-draw counts,
    then independently recomputes the two-sided Fisher exact p-value from
    those counts via `fisher_exact_two_sided` (hand-rolled, no scipy) and
    checks it against the constant's own stored `fisher_p`. This is a
    genuine recompute of the *statistic*, not a re-assertion of the
    constant: it would catch a hand-arithmetic mistake in
    `MISLABEL_SUSPECT_VALIDATION` itself. It does not re-derive the
    sampling design (which items were flagged, which 40 were the random
    draw) -- that provenance is `MISLABEL_SUSPECT_VALIDATION`'s own, not
    re-read from `run/calibration.jsonl` here."""
    v = MISLABEL_SUSPECT_VALIDATION
    a = round(v.precision * v.n_flagged_labelled)
    b = v.n_flagged_labelled - a
    c = round(v.random_draw_precision * v.random_draw_n)
    d = v.random_draw_n - c
    p = fisher_exact_two_sided(a, b, c, d)
    detail = (
        f"flagged {a}/{v.n_flagged_labelled}={v.precision:.1%} vs random draw "
        f"{c}/{v.random_draw_n}={v.random_draw_precision:.1%}, base_rate={v.base_rate:.1%}, "
        f"table=[[{a},{b}],[{c},{d}]], stored verdict={v.verdict!r}"
    )
    return ClaimResult(values={"precision": v.precision, "fisher_p": p}, detail=detail)


def _recompute_prereg(flags_path: Path) -> ClaimResult:
    flagged = _load_id_list(flags_path)
    labels = load_calibration_jsonl(_RUN / "calibration.jsonl")
    evaluation = evaluate_detector(flagged, labels)
    precision = _estimate_value(evaluation.precision.value, where=f"{flags_path}: precision")
    detail = f"{len(flagged)} pre-registered flagged id(s), precision n={evaluation.precision.n}"
    return ClaimResult(
        values={"precision": precision, "n": float(evaluation.precision.n)}, detail=detail
    )


def _recompute_detector_prereg_answer_unsupported() -> ClaimResult:
    return _recompute_prereg(_RUN / "prereg_answer_unsupported.txt")


def _recompute_detector_prereg_answer_echoes_evidence() -> ClaimResult:
    return _recompute_prereg(_RUN / "prereg_answer_echoes_evidence.txt")


# ---------------------------------------------------------------------------
# PPI (run/calibration.jsonl + run/answerability_codex_all.jsonl)
# ---------------------------------------------------------------------------


def _ppi_inputs() -> tuple[list[float], list[float], list[float]]:
    """`(y_labeled, f_labeled, f_unlabeled)` for the gpt-5.4/low judge
    against the 132 calibration-stratum human labels -- `y`/`f` are both
    "is this a defect" bools-as-floats (`verdict == "purge"` for the
    human label, `verdict != "supported"` for the judge, matching every
    other claim's flagging rule in this file), `f_unlabeled` the same
    judge rule applied to every other item `run/answerability_codex_all.jsonl`
    covers. Raises `EvidenceError` if any calibration-stratum item id is
    missing from the judge file -- `ppi_mean_ci` requires `y_labeled` and
    `f_labeled` aligned item-for-item, so a silent skip here would
    silently misalign the two."""
    labels = load_calibration_jsonl(_RUN / "calibration.jsonl")
    calibration = {label.item_id: label for label in labels if label.stratum == "calibration"}
    verdicts = _load_verdicts_jsonl(_RUN / "answerability_codex_all.jsonl")

    missing = [item_id for item_id in calibration if item_id not in verdicts]
    if missing:
        raise EvidenceError(
            f"ppi: {len(missing)} calibration-stratum item(s) missing from "
            f"run/answerability_codex_all.jsonl: {missing[:5]}"
        )

    labelled_ids = sorted(calibration)
    y = [1.0 if calibration[item_id].verdict == "purge" else 0.0 for item_id in labelled_ids]
    f_labeled = [1.0 if verdicts[item_id] != "supported" else 0.0 for item_id in labelled_ids]
    labelled_set = set(labelled_ids)
    f_unlabeled = [
        1.0 if verdict != "supported" else 0.0
        for item_id, verdict in verdicts.items()
        if item_id not in labelled_set
    ]
    return y, f_labeled, f_unlabeled


def _recompute_ppi_defect_rate() -> ClaimResult:
    y, f_labeled, f_unlabeled = _ppi_inputs()
    estimate = ppi_mean_ci(y, f_labeled, f_unlabeled)
    value = _estimate_value(estimate.value, where="ppi.defect_rate")
    if estimate.ci is None:
        raise EvidenceError("ppi.defect_rate: Estimate.ci is None")
    detail = f"n_labeled={estimate.n}, n_unlabeled={len(f_unlabeled)}, ci={estimate.ci}"
    return ClaimResult(
        values={"value": value, "ci_lo": estimate.ci[0], "ci_hi": estimate.ci[1]}, detail=detail
    )


def _recompute_ppi_classical_matches_reference() -> ClaimResult:
    """With `lam=1.0`, `ppi_mean_ci` must equal the hand-computed
    classical PPI formula `mean(f_unlabeled) + mean(y - f_labeled)`
    exactly (`fireassay.items.ppi`'s own module docstring: "`lam = 1`
    recovers classical PPI exactly")."""
    y, f_labeled, f_unlabeled = _ppi_inputs()
    ppi_classical = ppi_mean_ci(y, f_labeled, f_unlabeled, lam=1.0)
    ppi_value = _estimate_value(ppi_classical.value, where="ppi.classical_matches_reference")

    y_arr = np.asarray(y, dtype=float)
    f_labeled_arr = np.asarray(f_labeled, dtype=float)
    f_unlabeled_arr = np.asarray(f_unlabeled, dtype=float)
    classical_formula = float(f_unlabeled_arr.mean() + np.mean(y_arr - f_labeled_arr))

    diff = abs(ppi_value - classical_formula)
    detail = f"ppi_mean_ci(lam=1.0)={ppi_value!r} vs classical formula={classical_formula!r}"
    return ClaimResult(values={"abs_diff": diff}, detail=detail)


# ---------------------------------------------------------------------------
# Handbook cross-checks (run/item_analysis.json)
# ---------------------------------------------------------------------------


def _load_item_analysis() -> list[dict[str, float]]:
    raw_text = (_RUN / "item_analysis.json").read_text(encoding="utf-8")
    raw_rows: list[dict[str, Any]] = json.loads(raw_text)
    return [{"p": float(row["p"]), "D": float(row["D"]), "rpb": float(row["rpb"])} for row in raw_rows]


def _recompute_handbook_corr_d_rpbis() -> ClaimResult:
    rows = _load_item_analysis()
    d = np.array([row["D"] for row in rows], dtype=float)
    rpb = np.array([row["rpb"] for row in rows], dtype=float)
    corr = float(np.corrcoef(d, rpb)[0, 1])
    return ClaimResult(values={"corr": corr}, detail=f"n={len(rows)}")


def _recompute_handbook_negative_discrimination() -> ClaimResult:
    rows = _load_item_analysis()
    n_negative = sum(1 for row in rows if row["rpb"] < 0)
    n_total = len(rows)
    rate = n_negative / n_total if n_total else 0.0
    return ClaimResult(
        values={"n_negative": float(n_negative), "n_total": float(n_total), "rate": rate},
        detail=f"{n_negative}/{n_total} items with point_biserial < 0",
    )


# ---------------------------------------------------------------------------
# Closed-book (run/closedbook_forced.log)
# ---------------------------------------------------------------------------

_CLOSEDBOOK_RESULT_RE = re.compile(r"RESULT:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%")


def _recompute_closedbook_forced() -> ClaimResult:
    """Parses `run/closedbook_forced.log`'s own printed `RESULT: n/m = x%`
    line and recomputes the rate from `n`/`m` directly, rather than
    trusting the log's own pre-rounded `x%` -- see the module docstring's
    "One number below disagrees with the prose it was copied from"."""
    text = (_RUN / "closedbook_forced.log").read_text(encoding="utf-8")
    match = _CLOSEDBOOK_RESULT_RE.search(text)
    if match is None:
        raise EvidenceError(
            "closedbook.forced: no 'RESULT: n/m = x%' line found in run/closedbook_forced.log"
        )
    hits, scored = int(match.group(1)), int(match.group(2))
    rate = hits / scored if scored else 0.0
    detail = f"{hits}/{scored} = {rate:.1%} (log's own printed figure: {match.group(3)}%)"
    return ClaimResult(values={"rate": rate, "hits": float(hits), "scored": float(scored)}, detail=detail)


# ---------------------------------------------------------------------------
# Gate (run/gate_demo.json)
# ---------------------------------------------------------------------------


def _load_gate_demo() -> dict[str, Any]:
    data = json.loads((_RUN / "gate_demo.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise EvidenceError("run/gate_demo.json: not a gate_demo artifact (no 'cases' key)")
    return data


def _gate_case_inputs(
    case: dict[str, Any],
) -> tuple[dict[str, tuple[list[float], list[float]]], list[MetricSpec]]:
    per_item = {
        metric: (list(map(float, values["base"])), list(map(float, values["head"])))
        for metric, values in case["per_item"].items()
    }
    specs = [MetricSpec(metric=m, threshold=float(t)) for m, t in case["thresholds"].items()]
    return per_item, specs


def _recompute_gate_case(name: str) -> ClaimResult:
    """Re-run `fireassay.gate.evaluate_gate` on the per-item values the
    artifact records for one case, with the artifact's own alpha/power/b/
    seed. The verdict is recomputed, never read back from the artifact's
    `gate_checks` (those are the CLI's record of the same call, kept for
    comparison in `detail`)."""
    data = _load_gate_demo()
    case = data["cases"][name]
    per_item, specs = _gate_case_inputs(case)
    params = data["gate"]
    try:
        report = evaluate_gate(
            base_run_id="base", head_run_id="head", suite_id="suite",
            per_item=per_item, specs=specs,
            alpha=float(params["alpha"]), power=float(params["power"]),
            b=int(params["b"]), seed=int(params["seed"]),
        )
    except ThresholdBelowMDEError as exc:
        # Refusal is a distinct, reportable outcome -- not a pass, not a
        # block. Report the MDE and the observed regression it refused to
        # judge, computed the way the gate computes them.
        boot = paired_bootstrap(
            *per_item[exc.metric], b=int(params["b"]), alpha=float(params["alpha"]) / len(specs),
            seed=int(params["seed"]),
        )
        recorded = case["exit_code"]
        return ClaimResult(
            values={"outcome": "refused", "mde": exc.mde, "delta": boot.delta, "n": float(boot.n),
                    "exit_code": float(recorded)},
            detail=(f"threshold {exc.threshold} < mde {exc.mde:.4f} on n={boot.n}; observed delta "
                    f"{boot.delta:+.4f}; CLI recorded exit {recorded}"),
        )
    outcome = "block" if report.blocked else "pass"
    recorded = case["exit_code"]
    detail = "; ".join(
        f"{r.metric}: delta {r.delta:+.4f}, p_holm {r.p_adjusted:.4f}, mde {r.mde:.4f} -> {r.verdict}"
        for r in report.results
    )
    # Deltas are keyed by metric name, never by position: the artifact is
    # written with sorted keys, so "the first metric" is whichever sorts
    # first, and a claim that assumed otherwise failed its own first check.
    values: dict[str, float | str] = {
        "outcome": outcome,
        "n": float(report.results[0].n),
        "exit_code": float(recorded),
        "n_metrics": float(len(report.results)),
        "n_block": float(sum(r.verdict == "block" for r in report.results)),
    }
    for r in report.results:
        values[f"delta:{r.metric}"] = r.delta
    return ClaimResult(values=values, detail=f"{detail}; CLI recorded exit {recorded}")


def _recompute_gate_benign() -> ClaimResult:
    return _recompute_gate_case("benign")


def _recompute_gate_planted_total() -> ClaimResult:
    return _recompute_gate_case("planted_total")


def _recompute_gate_planted_subtle_refused() -> ClaimResult:
    return _recompute_gate_case("planted_subtle_refused")


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------

CLAIMS: tuple[Claim, ...] = (
    Claim(
        id="gate.benign_passes",
        group="Gate (run/gate_demo.json)",
        description=(
            "The M5 gate passes a benign config change (top_k 5 vs 3 on the fixture suite, "
            "threshold 0.5 on retrieval.recall@5): recomputed from the per-item values "
            "tools/gate_demo.py recorded, with its alpha/power/b/seed."
        ),
        artifacts=("run/gate_demo.json",),
        metrics=(
            Metric("outcome", "pass"), Metric("delta:retrieval.recall@5", -1 / 14, 1e-9), Metric("n", 14.0),
            Metric("exit_code", 0.0), Metric("n_block", 0.0),
        ),
        recompute=_recompute_gate_benign,
    ),
    Claim(
        id="gate.planted_regression_blocked",
        group="Gate (run/gate_demo.json)",
        description=(
            "The gate blocks a planted regression (drop_results n=4 on the top_k=5 config: "
            "retrieval.recall@5 1.0 -> 0.0 on every paired item, retrieval.mrr likewise) on "
            "both metrics after Holm correction, and the CLI exited 4. The first failure "
            "condition of the M5 pre-registration -- a gate that has never blocked anything -- "
            "is what this claim exists to refute."
        ),
        artifacts=("run/gate_demo.json",),
        metrics=(
            Metric("outcome", "block"), Metric("delta:retrieval.recall@5", -1.0, 1e-9),
            Metric("delta:retrieval.mrr", -0.9107, 0.001), Metric("n", 14.0),
            Metric("n_metrics", 2.0), Metric("n_block", 2.0), Metric("exit_code", 4.0),
        ),
        recompute=_recompute_gate_planted_total,
    ),
    Claim(
        id="gate.subtle_regression_refused",
        group="Gate (run/gate_demo.json)",
        description=(
            "A real regression the suite cannot resolve is refused, not passed and not blocked: "
            "corrupt_query pct=0.9 drops retrieval.recall@5 by about 0.39 on 14 paired items, "
            "larger than the 0.3 threshold asked for, but the minimum detectable effect at this "
            "n is about 0.35, so the gate raises ThresholdBelowMDEError and the CLI exits 5. "
            "Defect 9 (a threshold below what the suite resolves), caught by the gate itself."
        ),
        artifacts=("run/gate_demo.json",),
        metrics=(
            Metric("outcome", "refused"), Metric("delta", -0.3929, 0.001), Metric("mde", 0.35, 0.02),
            Metric("n", 14.0), Metric("exit_code", 5.0),
        ),
        recompute=_recompute_gate_planted_subtle_refused,
    ),
    Claim(
        id="panel.shape",
        group="Panel (run/panel54_matrix.csv)",
        description=(
            "The 54-config panel's response matrix covers every kept item against every config."
        ),
        artifacts=("run/panel54_matrix.csv",),
        metrics=(Metric("n_items", 2364.0), Metric("n_systems", 54.0)),
        recompute=_recompute_panel_shape,
    ),
    Claim(
        id="panel.classes",
        group="Panel (run/panel54_matrix.csv)",
        description=(
            "fireassay.items.core.analyse's item classification over the full 54-config panel."
        ),
        artifacts=("run/panel54_matrix.csv",),
        metrics=(
            Metric("live", 1731.0),
            Metric("dead_all_pass", 487.0),
            Metric("dead_all_fail", 74.0),
            Metric("mislabel_suspect", 72.0),
        ),
        recompute=_recompute_panel_classes,
    ),
    Claim(
        id="panel.reliability",
        group="Panel (run/panel54_matrix.csv)",
        description="Split-half reliability over the full 54-config panel, and its usability verdict.",
        artifacts=("run/panel54_matrix.csv",),
        metrics=(Metric("reliability", 0.5031, 0.0015), Metric("verdict", "usable")),
        recompute=_recompute_panel_reliability,
    ),
    Claim(
        id="panel.monotonic",
        group="Panel (run/panel54_matrix.csv)",
        description=(
            "Within one (retriever, chunking) family, a top_k=3 success implies a top_k=10 "
            "success -- defect 49's mechanism. See this claim's recompute docstring for why the "
            "checked scope (42,552 pairs) differs from the historically-quoted 14,184."
        ),
        artifacts=("run/panel54_matrix.csv",),
        metrics=(Metric("violations", 0.0),),
        recompute=_recompute_panel_monotonic,
    ),
    Claim(
        id="panel.strength_is_breadth",
        group="Panel (run/panel54_matrix.csv)",
        description=(
            "Of the top/bottom 15 configs by total score, whether the top group skews k=10 "
            "and the bottom group skews k=3 (defect 49: strength and retrieval-window breadth "
            "are confounded within a top_k-only ladder)."
        ),
        artifacts=("run/panel54_matrix.csv",),
        metrics=(Metric("top_favors_k10", 1.0), Metric("bottom_favors_k3", 1.0)),
        recompute=_recompute_panel_strength_is_breadth,
    ),
    Claim(
        id="panel18.classes",
        group="Old panel (run/panel_matrix.csv)",
        description=(
            "fireassay.items.core.analyse's item classification over the single-retriever "
            "18-config predecessor panel -- kept so the panel54 improvement is checkable. "
            "Recomputed fresh, not read from run/full_panel.log (defect 46's stale figure)."
        ),
        artifacts=("run/panel_matrix.csv",),
        metrics=(
            Metric("dead_all_pass", 1263.0),
            Metric("live", 955.0),
            Metric("dead_all_fail", 114.0),
            Metric("mislabel_suspect", 32.0),
        ),
        recompute=_recompute_panel18_classes,
    ),
    Claim(
        id="panel18.reliability",
        group="Old panel (run/panel_matrix.csv)",
        description="Split-half reliability over the old 18-config panel, and its usability verdict.",
        artifacts=("run/panel_matrix.csv",),
        metrics=(Metric("reliability", 0.286, 0.0015), Metric("verdict", "too_few_systems")),
        recompute=_recompute_panel18_reliability,
    ),
    Claim(
        id="labels.counts",
        group="Human labels (run/calibration.jsonl)",
        description=(
            "Total human labels collected, and how many are in the uniform-random "
            "calibration stratum."
        ),
        artifacts=("run/calibration.jsonl",),
        metrics=(Metric("n_labels", 176.0), Metric("n_calibration", 132.0)),
        recompute=_recompute_labels_counts,
    ),
    Claim(
        id="labels.base_rate",
        group="Human labels (run/calibration.jsonl)",
        description=(
            "The suite's defect base rate: purge-verdict fraction over the 132 "
            "uniform-random labels."
        ),
        artifacts=("run/calibration.jsonl",),
        metrics=(
            Metric("rate", 0.174, 0.003),
            Metric("ci_lo", 0.119, 0.01),
            Metric("ci_hi", 0.248, 0.01),
        ),
        recompute=_recompute_labels_base_rate,
    ),
    Claim(
        id="labels.lenient_rate",
        group="Human labels (run/calibration.jsonl)",
        description="The lenient defect rate: purge-or-rewrite fraction over the same 132 labels.",
        artifacts=("run/calibration.jsonl",),
        metrics=(Metric("rate", 0.409, 0.01),),
        recompute=_recompute_labels_lenient_rate,
    ),
    Claim(
        id="judge.granite4",
        group="Judges",
        description=(
            "granite4:7b-a1b-h (local Ollama) gold-answerability judge vs the 132 "
            "uniform-random human labels (not all 176 -- see the module docstring's "
            "\"Judge precision is a stratum choice\" section)."
        ),
        artifacts=("run/answerability.jsonl", "run/calibration.jsonl"),
        metrics=(Metric("precision", 0.571, 0.01), Metric("recall", 0.174, 0.01)),
        recompute=_recompute_judge_granite4,
    ),
    Claim(
        id="judge.gpt54",
        group="Judges",
        description=(
            "gpt-5.4/low (via codex) gold-answerability judge, all 2,364 items, vs the 132 "
            "uniform-random human labels. See this claim's recompute docstring for the "
            "run/validations.jsonl vs eval-integrity-notes discrepancy this scope choice "
            "resolves."
        ),
        artifacts=("run/answerability_codex_all.jsonl", "run/calibration.jsonl"),
        metrics=(Metric("precision", 0.652, 0.01), Metric("recall", 0.6522, 0.01)),
        recompute=_recompute_judge_gpt54,
    ),
    Claim(
        id="judge.nli",
        group="Judges",
        description=(
            "DeBERTa-v3-large NLI gold-answerability judge vs the 132 uniform-random human "
            "labels. Precision only -- see the module docstring for why the Fisher exact "
            "comparison is excluded."
        ),
        artifacts=("run/answerability_nli.jsonl", "run/calibration.jsonl"),
        metrics=(Metric("precision", 0.224, 0.01),),
        recompute=_recompute_judge_nli,
    ),
    Claim(
        id="judge.token_overlap",
        group="Judges",
        description=(
            "The no-model token-overlap baseline (answer_unsupported, "
            "fireassay.items.baseline) vs the 132 uniform-random human labels -- the "
            "no-model floor every model-backed judge above must beat. "
            "Verified 2026-08-30: the committed instrument reproduces the uncommitted "
        "scratch analysis exactly (8 flagged of the 132, 6 judged broken). Until "
        "this change the detector had no committed implementation at all -- "
        "defect 46 recurring, caught by writing this file."
        ),
        artifacts=("run/items_meta.jsonl", "run/calibration.jsonl", "src/fireassay/items/baseline.py"),
        metrics=(Metric("precision", 0.75, 0.01),),
        recompute=_recompute_judge_token_overlap,
    ),
    Claim(
        id="judge.cache_consistency",
        group="Judges",
        description="The gpt-5.4/low judge's cache returns identical verdicts on every re-scored item.",
        artifacts=("run/answerability_codex.jsonl", "run/answerability_codex_all.jsonl"),
        metrics=(Metric("agree_rate", 1.0), Metric("n_shared", 176.0)),
        recompute=_recompute_judge_cache_consistency,
    ),
    Claim(
        id="detector.mislabel_suspect",
        group="Detectors",
        description=(
            "mislabel_suspect (discrimination_d < 0) is no better than a random draw of the "
            "same size -- defect 50, MISLABEL_SUSPECT_VALIDATION's own stored counts, Fisher "
            "exact p independently recomputed here."
        ),
        artifacts=("src/fireassay/items/core.py",),
        metrics=(Metric("precision", 7 / 32, 0.001), Metric("fisher_p", 0.79, 0.02)),
        recompute=_recompute_detector_mislabel_suspect,
    ),
    Claim(
        id="detector.prereg_answer_unsupported",
        group="Detectors",
        description=(
            "answer_unsupported's pre-registered, held-out 5-item flagged list, frozen before "
            "the pass-2 labels existed, scored against them."
        ),
        artifacts=("run/prereg_answer_unsupported.txt", "run/calibration.jsonl"),
        metrics=(Metric("precision", 0.8, 0.01), Metric("n", 5.0)),
        recompute=_recompute_detector_prereg_answer_unsupported,
    ),
    Claim(
        id="detector.prereg_answer_echoes_evidence",
        group="Detectors",
        description=(
            "answer_echoes_evidence's pre-registered 49-item flagged list -- the control "
            "that stayed at zero."
        ),
        artifacts=("run/prereg_answer_echoes_evidence.txt", "run/calibration.jsonl"),
        metrics=(Metric("precision", 0.0, 0.01), Metric("n", 49.0)),
        recompute=_recompute_detector_prereg_answer_echoes_evidence,
    ),
    Claim(
        id="ppi.defect_rate",
        group="PPI",
        description=(
            "PPI defect-rate estimate combining the gpt-5.4/low judge over all 2,364 "
            "items with the 132 human labels."
        ),
        artifacts=("run/calibration.jsonl", "run/answerability_codex_all.jsonl"),
        metrics=(
            Metric("value", 0.207, 0.01),
            Metric("ci_lo", 0.153, 0.015),
            Metric("ci_hi", 0.261, 0.015),
        ),
        recompute=_recompute_ppi_defect_rate,
    ),
    Claim(
        id="ppi.classical_matches_reference",
        group="PPI",
        description="With lam=1.0, ppi_mean_ci equals the hand-computed classical PPI formula exactly.",
        artifacts=(
            "run/calibration.jsonl",
            "run/answerability_codex_all.jsonl",
            "src/fireassay/items/ppi.py",
        ),
        metrics=(Metric("abs_diff", 0.0, 1e-9),),
        recompute=_recompute_ppi_classical_matches_reference,
    ),
    Claim(
        id="handbook.corr_d_rpbis",
        group="Handbook cross-checks (run/item_analysis.json)",
        description=(
            "Correlation between discrimination D and point-biserial r, over the "
            "400-question item analysis."
        ),
        artifacts=("run/item_analysis.json",),
        metrics=(Metric("corr", 0.938, 0.002),),
        recompute=_recompute_handbook_corr_d_rpbis,
    ),
    Claim(
        id="handbook.negative_discrimination",
        group="Handbook cross-checks (run/item_analysis.json)",
        description="How many of the 400 items have a negative point-biserial correlation.",
        artifacts=("run/item_analysis.json",),
        metrics=(Metric("n_negative", 11.0), Metric("n_total", 400.0)),
        recompute=_recompute_handbook_negative_discrimination,
    ),
    Claim(
        id="closedbook.forced",
        group="Closed-book (run/closedbook_forced.log)",
        description=(
            "Forced-guess closed-book screen: fraction of sampled items answerable with no "
            "retrieval at all. See the module docstring for why this claim's expected rate "
            "(1.3%) differs from the eval-integrity notes' 1.4%."
        ),
        artifacts=("run/closedbook_forced.log",),
        metrics=(Metric("hits", 2.0), Metric("scored", 149.0), Metric("rate", 2 / 149, 0.0005)),
        recompute=_recompute_closedbook_forced,
    ),
)

_EXCLUDED_CLAIMS_NOTE = """\
- **`judge.nli`'s Fisher exact p-value** (handbook: p=0.169) -- the
  random-draw comparison group's sampling design is not recorded in any
  committed artifact. `judge.nli`'s precision alone is checked above.
"""


STYLE = MarkdownStyle(
    regenerate_command=".venv/bin/python tools/evidence.py --markdown > EVIDENCE.md",
    check_command=".venv/bin/python tools/evidence.py --check",
    verify_command=".venv/bin/python tools/evidence.py --claim {claim_id}",
    preamble=(
        "Every claim below names the committed artifact(s) it reads and the command that "
        "recomputes and verifies it. `tools/evidence.py --check` recomputes every claim "
        "fresh from those artifacts and compares it against the expected value encoded "
        "there; a mismatch prints FAIL and the run exits non-zero -- it is never silently "
        "rewritten to match whatever the recompute produced (defects 43 and 46, "
        "`~/ObitosBrain/notes/2026-08-28-fireassay-eval-integrity.md`)."
    ),
    excluded_note=_EXCLUDED_CLAIMS_NOTE,
)


def render_markdown(claims: Sequence[Claim]) -> str:
    """`fireassay.evidence.render_markdown` with this repo's `STYLE`."""
    return _render_markdown(claims, STYLE)


def main(argv: Sequence[str] | None = None) -> int:
    """`fireassay.evidence.main` over this repo's `CLAIMS`. The same
    claims file also runs through the CLI, with the generic style:
    `fireassay evidence --claims tools/evidence.py --check`."""
    return _evidence_main(CLAIMS, argv, style=STYLE, description=__doc__)


if __name__ == "__main__":
    sys.exit(main())
