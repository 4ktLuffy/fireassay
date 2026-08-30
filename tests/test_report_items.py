"""`test_report_items.py`: `report.text.render_items_analysis`'s two
integrity-layer obligations (Goal 2, criteria 3/4 in
`~/ObitosBrain/notes/2026-08-28-fireassay-eval-integrity.md`), each
checked by capturing the actual rendered console output rather than
inspecting the data structures that feed it -- "labelled unvalidated in
its own output" means the words have to be in the output.
"""

from __future__ import annotations

from rich.console import Console

from fireassay.items.core import Estimate, ItemStats, PanelStats
from fireassay.items.validation import DetectorValidation
from fireassay.report.text import render_items_analysis


def _panel_stats(class_counts: dict[str, int]) -> PanelStats:
    return PanelStats(
        n_items=sum(class_counts.values()),
        n_systems=8,
        split_half_reliability=0.6,
        n_items_used_for_reliability=sum(class_counts.values()),
        reliability_verdict="usable",
        class_counts=class_counts,
        claimed_vs_measured_difficulty_r=None,
    )


def _item_stats(class_counts: dict[str, int]) -> list[ItemStats]:
    stats: list[ItemStats] = []
    i = 0
    for classification, count in class_counts.items():
        for _ in range(count):
            d = -1.0 if classification == "mislabel_suspect" else 1.0
            p = {"dead_all_pass": 1.0, "dead_all_fail": 0.0}.get(classification, 0.5)
            stats.append(
                ItemStats(
                    item_id=f"item{i}",
                    p=p,
                    discrimination_d=d,
                    point_biserial=0.0,
                    classification=classification,  # type: ignore[arg-type]
                )
            )
            i += 1
    return stats


def _rendered(
    item_stats: list[ItemStats],
    panel_stats: PanelStats,
    validations: dict[str, DetectorValidation] | None = None,
) -> str:
    console = Console(record=True, width=200)
    render_items_analysis(item_stats, panel_stats, validations, console=console)
    return console.export_text()


# -- criterion 4: a non-zero mislabel_suspect count is always beside UNVALIDATED --


def test_nonzero_mislabel_suspect_count_output_contains_unvalidated() -> None:
    class_counts = {"live": 5, "dead_all_pass": 1, "dead_all_fail": 1, "mislabel_suspect": 2}
    output = _rendered(_item_stats(class_counts), _panel_stats(class_counts))
    assert "mislabel_suspect" in output
    assert "UNVALIDATED" in output


def test_zero_mislabel_suspect_count_output_has_no_unvalidated_line() -> None:
    """The UNVALIDATED line is conditional on a non-zero count -- a panel
    with none flagged must not print an unearned warning about a
    classification that never fired."""
    class_counts = {"live": 5, "dead_all_pass": 1, "dead_all_fail": 1, "mislabel_suspect": 0}
    output = _rendered(_item_stats(class_counts), _panel_stats(class_counts))
    assert "UNVALIDATED" not in output


# -- criterion 3: --validations prints measured precision/recall and verdict -----


def test_validations_table_contains_precision_and_verdict() -> None:
    class_counts = {"live": 4, "dead_all_pass": 1, "dead_all_fail": 1, "mislabel_suspect": 0}
    validations = {
        "mislabel_suspect": DetectorValidation(
            detector="mislabel_suspect",
            measured_on="2026-08-29",
            labels="132 uniform-random human labels, run/calibration.jsonl",
            base_rate=0.174,
            precision=Estimate(value=0.219, ci=(0.10, 0.38), n=32, verdict="measured"),
            recall=None,
            note="indistinguishable from chance, Fisher p=0.79",
            verdict="not_validated",
        ),
        "gpt-5.4 low effort": DetectorValidation(
            detector="gpt-5.4 low effort",
            measured_on="2026-08-29",
            labels="132 uniform-random human labels, run/calibration.jsonl",
            base_rate=0.174,
            precision=Estimate(value=0.652, ci=(0.55, 0.75), n=46, verdict="measured"),
            recall=Estimate(value=0.652, ci=(0.50, 0.79), n=46, verdict="measured"),
            note=None,
            verdict="validated",
        ),
    }
    output = _rendered(_item_stats(class_counts), _panel_stats(class_counts), validations)
    assert "mislabel_suspect" in output
    assert "0.219" in output
    assert "not_validated" in output
    assert "gpt-5.4 low effort" in output
    assert "0.652" in output
    # "validated" appears once as a standalone verdict (the second
    # detector's) beyond the one occurrence embedded inside "not_validated"
    # -- distinguishes an actual "validated" verdict being printed from
    # merely being a substring of "not_validated".
    assert output.count("validated") == output.count("not_validated") + 1


def test_no_validations_argument_omits_the_table() -> None:
    class_counts = {"live": 4, "dead_all_pass": 1, "dead_all_fail": 1, "mislabel_suspect": 0}
    output = _rendered(_item_stats(class_counts), _panel_stats(class_counts))
    assert "detector validation status" not in output
