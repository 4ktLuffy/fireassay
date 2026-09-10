"""Render `ComparabilityReport`/`Leaderboard` objects, (M2) control and
mutation results, (M3) curation funnel reports, and (items) item-analysis
and blind-review reports, to the terminal via `rich`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.console import Console
from rich.table import Table

from fireassay.compare import Leaderboard
from fireassay.controls.base import ControlOutcome
from fireassay.curate.report import CurateReport, SessionFlag
from fireassay.gate import GateReport
from fireassay.integrity import ComparabilityReport, RefusalReason
from fireassay.items.calibration import DetectorEvaluation
from fireassay.items.core import MISLABEL_SUSPECT_VALIDATION, Estimate, ItemStats, PanelStats
from fireassay.items.review import DetectorScore
from fireassay.items.validation import DetectorValidation
from fireassay.mutation.score import MutationScoreResult
from fireassay.store.db import ControlCheckRow, MutantRow, MutationRunRow


def render_refusal(reasons: Sequence[RefusalReason], console: Console | None = None) -> None:
    """Render a list of refusal reasons (from a caught `ComparisonRefusedError`)
    as a table."""
    console = console or Console()
    table = Table(title="Comparison refused")
    table.add_column("code")
    table.add_column("detail")
    for reason in reasons:
        table.add_row(reason.code, reason.detail)
    console.print(table)


def render_gate_report(
    report: GateReport,
    comparability: ComparabilityReport,
    dropped: Mapping[str, int],
    console: Console | None = None,
) -> None:
    """Render a `gate.GateReport` (from `fireassay gate`): one row per
    metric with the paired-bootstrap delta, its family-wise interval, the
    raw and Holm-adjusted p-values, the MDE the threshold was checked
    against, and the verdict; then the single line a CI log is read for.

    Like every renderer, prefixes 'UNSOUND COMPARISON' when
    `comparability.forced` (the caller passed --force past a refusal).
    `dropped[metric]` is the number of questions scored on one run only
    and so excluded from the pairing -- printed, never hidden, because a
    gate whose paired population silently shrank is measuring a different
    suite from the one it names.
    """
    console = console or Console()
    if comparability.forced:
        console.print("[bold red]UNSOUND COMPARISON[/bold red]")
        for reason in comparability.reasons:
            console.print(f"  {reason.code}: {reason.detail}")

    table = Table(title=f"fireassay gate  base={report.base_run_id[:12]}  head={report.head_run_id[:12]}")
    table.add_column("metric")
    table.add_column("n", justify="right")
    table.add_column("base", justify="right")
    table.add_column("head", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("CI", justify="right")
    table.add_column("p", justify="right")
    table.add_column("p Holm", justify="right")
    table.add_column("MDE", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("verdict")
    for result in report.results:
        verdict = "[bold red]BLOCK[/bold red]" if result.verdict == "block" else "[green]pass[/green]"
        n_text = str(result.n)
        if dropped.get(result.metric, 0):
            n_text += f" (-{dropped[result.metric]} unpaired)"
        table.add_row(
            result.metric,
            n_text,
            f"{result.base_mean:.4f}",
            f"{result.head_mean:.4f}",
            f"{result.delta:+.4f}",
            f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]",
            f"{result.p_value:.4f}",
            f"{result.p_adjusted:.4f}",
            f"{result.mde:.4f}",
            f"{result.threshold:.4f}",
            verdict,
        )
    console.print(table)
    console.print(
        f"alpha={report.alpha} power={report.power} b={report.b} seed={report.seed} "
        f"metrics={len(report.results)}  CI is family-wise at "
        f"{1 - report.alpha / len(report.results):.2%}"
    )
    if report.blocked:
        blocked = ", ".join(r.metric for r in report.results if r.verdict == "block")
        console.print(f"[bold red]GATE BLOCKED[/bold red] on {blocked}")
    else:
        console.print("[green]GATE PASSED[/green] on every metric checked")


def render_leaderboard(board: Leaderboard, console: Console | None = None) -> None:
    """Render a `Leaderboard` as a table.

    Every renderer MUST check `report.forced` and, if True, prefix its
    output with 'UNSOUND COMPARISON' (M1-SPEC.md §4/§8) — this function is
    where that rule is enforced for terminal output. A future HTML/JSON
    renderer must repeat this check independently; it is not inherited
    from `Leaderboard` itself, because forgetting the prefix would silently
    turn a flagged-unsound comparison back into what looks like a normal
    one.
    """
    console = console or Console()
    title = "fireassay leaderboard"
    if board.report.forced:
        console.print("[bold red]UNSOUND COMPARISON[/bold red]")
        title = f"UNSOUND COMPARISON — {title}"

    table = Table(title=title)
    table.add_column("run_id")
    table.add_column("config_hash")
    table.add_column("slice")
    table.add_column("metric")
    table.add_column("mean (n/applicable_n)", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")

    for entry in board.entries:
        slice_str = ", ".join(f"{k}={v}" for k, v in sorted(entry.slice.items()))
        for i, metric_summary in enumerate(entry.metrics):
            # n < applicable_n MUST be visible here, not just in the mean:
            # a metric silently measured on a shrinking subset of its
            # population is the single most common way an eval flatters a
            # config (see MetricSummary.applicable_n).
            mean_str = f"{metric_summary.mean:.4f} (n={metric_summary.n}/{metric_summary.applicable_n})"
            table.add_row(
                entry.run_id if i == 0 else "",
                entry.config_hash[:12] if i == 0 else "",
                slice_str if i == 0 else "",
                metric_summary.metric,
                mean_str,
                f"{metric_summary.p50:.2f}" if metric_summary.p50 is not None else "",
                f"{metric_summary.p95:.2f}" if metric_summary.p95 is not None else "",
            )

    console.print(table)


def _control_row(
    table: Table, kind: str, status: str, twin_ok: bool, cause_ok: bool, detail: str
) -> None:
    status_style = {"PASSED": "green", "FAILED": "bold red", "NOT_RUN": "bold yellow"}.get(status, "")
    table.add_row(
        kind,
        f"[{status_style}]{status}[/{status_style}]" if status_style else status,
        str(twin_ok),
        str(cause_ok),
        detail,
    )


def render_control_outcomes(outcomes: Sequence[ControlOutcome], console: Console | None = None) -> None:
    """Render freshly-computed `ControlOutcome`s (from `fireassay controls
    run`, before persistence) as a table. `NOT_RUN` is styled distinctly
    from `FAILED` for readability, but both are equally inadmissible unless
    explicitly allowed (M2-SPEC.md §1) — this renderer draws no other
    distinction between them."""
    console = console or Console()
    table = Table(title="fireassay controls")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("twin_ok")
    table.add_column("cause_assertions all true")
    table.add_column("detail")
    for outcome in outcomes:
        cause_ok = all(outcome.cause_assertions.values()) if outcome.cause_assertions else False
        _control_row(table, outcome.kind, outcome.status, outcome.twin_ok, cause_ok, outcome.detail)
    console.print(table)


def render_control_checks(checks: Sequence[ControlCheckRow], console: Console | None = None) -> None:
    """Render `control_check` rows read back from the store (`fireassay
    controls show --run <run_id>`)."""
    console = console or Console()
    table = Table(title="fireassay controls (persisted)")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("twin_ok")
    table.add_column("cause_assertions all true")
    table.add_column("checked_at")
    table.add_column("detail")
    for check in checks:
        cause_ok = all(check.cause_assertions.values()) if check.cause_assertions else False
        status_style = {"PASSED": "green", "FAILED": "bold red", "NOT_RUN": "bold yellow"}.get(
            check.status, ""
        )
        status_text = f"[{status_style}]{check.status}[/{status_style}]" if status_style else check.status
        table.add_row(
            check.kind, status_text, str(check.twin_ok), str(cause_ok), check.checked_at, check.detail
        )
    console.print(table)


_MutantRow = tuple[str, dict[str, object], bool, bool, str | None, str]


def _mutant_rows_sorted(rows: Sequence[_MutantRow]) -> list[_MutantRow]:
    """Survivors (not equivalent, not killed) first, then killed, then
    equivalent/excluded — M2-SPEC.md §6: "the report lists survivors
    first."""
    survivors = [r for r in rows if not r[3] and not r[2]]
    killed = [r for r in rows if not r[3] and r[2]]
    equivalent = [r for r in rows if r[3]]
    return survivors + killed + equivalent


def _render_mutation_table(
    title: str,
    detector: str,
    killed: int,
    total: int,
    equivalent: int,
    score: float,
    rows: Sequence[_MutantRow],
    console: Console,
) -> None:
    console.print(
        f"[bold]{title}[/bold]  detector={detector}  "
        f"gate_mutation_score = {killed}/{total - equivalent} = {score:.4f}  "
        f"(killed={killed}, total={total}, equivalent={equivalent})"
    )
    table = Table(title="mutants (survivors first)")
    table.add_column("operator")
    table.add_column("params")
    table.add_column("result")
    table.add_column("equivalent_reason")
    table.add_column("detail")
    for operator, params, killed_flag, equivalent_flag, equivalent_reason, detail in _mutant_rows_sorted(
        rows
    ):
        if equivalent_flag:
            result = "[dim]EXCLUDED (equivalent)[/dim]"
        elif killed_flag:
            result = "[green]KILLED[/green]"
        else:
            result = "[bold red]SURVIVED[/bold red]"
        table.add_row(operator, str(params), result, equivalent_reason or "", detail)
    console.print(table)


def render_mutation_score(result: MutationScoreResult, console: Console | None = None) -> None:
    """Render a freshly-computed `MutationScoreResult` (from `fireassay
    mutate`, before persistence)."""
    console = console or Console()
    rows: list[_MutantRow] = [
        (m.operator, m.params, m.killed, m.equivalent, m.equivalent_reason, m.detail) for m in result.mutants
    ]
    _render_mutation_table(
        "fireassay mutate",
        result.detector,
        result.killed,
        result.total,
        result.equivalent,
        result.score,
        rows,
        console,
    )


def render_mutation_run(
    run_row: MutationRunRow, mutants: Sequence[MutantRow], console: Console | None = None
) -> None:
    """Render a `mutation_run` + its `mutant` rows read back from the store
    (`fireassay mutation score --mutation-run <id>`)."""
    console = console or Console()
    rows: list[_MutantRow] = [
        (m.operator, m.params, m.killed, m.equivalent, m.equivalent_reason, m.detail) for m in mutants
    ]
    _render_mutation_table(
        f"mutation_run {run_row.id}",
        run_row.detector,
        run_row.killed,
        run_row.total,
        run_row.equivalent,
        run_row.score,
        rows,
        console,
    )


def render_curate_report(report: CurateReport, console: Console | None = None) -> None:
    """Render `fireassay curate report`'s output (M3-SPEC.md §4/§6)."""
    console = console or Console()

    funnel_table = Table(title="curation funnel")
    funnel_table.add_column("stage")
    funnel_table.add_column("kept", justify="right")
    funnel_table.add_column("rejected", justify="right")
    funnel_table.add_column("reasons")
    for stage in report.funnel.stages:
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(stage.rejected_by_reason.items()))
        funnel_table.add_row(stage.stage, str(stage.kept), str(stage.rejected), reasons)
    console.print(funnel_table)
    reconciled = report.funnel.reconciles()
    reconcile_style = "green" if reconciled else "bold red"
    console.print(
        f"generated={report.funnel.generated}  filter_kept={report.funnel.filter_kept}  "
        f"[{reconcile_style}]reconciles={reconciled}[/{reconcile_style}]"
    )
    console.print(
        f"curated: accepted={report.funnel.accepted}  edited={report.funnel.edited}  "
        f"rejected={report.funnel.rejected}  reject_reasons={report.funnel.reject_reason_counts}"
    )
    console.print(
        f"reviewer_hours={report.reviewer_hours:.2f}  "
        f"median_seconds_per_item={report.median_seconds_per_item}"
    )
    console.print(
        "[dim]unretrievable: a floor (top-N of the whole corpus, applied identically to every "
        "config), not a selection criterion -- but the resulting set will under-represent "
        "questions that require semantic rather than lexical matching. See filter/stages.py "
        "and the README.[/dim]"
    )

    agreement_statuses = {r.status for r in report.agreement_by_criterion.values()}
    if report.agreement_by_criterion and len(agreement_statuses) == 1 and agreement_statuses != {"OK"}:
        # Every criterion shares one non-OK status -- the six-row table
        # would otherwise repeat the identical boilerplate detail sentence
        # once per criterion (~40 lines for one fact). Collapsed to a
        # single line; an "OK" status is never collapsed this way, since
        # each OK row carries a distinct, meaningful alpha value.
        only_status = next(iter(agreement_statuses))
        sample_detail = next(iter(report.agreement_by_criterion.values())).detail
        criteria_list = ", ".join(sorted(report.agreement_by_criterion))
        console.print(
            f"Krippendorff's alpha: all {len(report.agreement_by_criterion)} criteria "
            f"({criteria_list}) are [bold red]{only_status}[/bold red] -- {sample_detail}"
        )
    else:
        agreement_table = Table(title="Krippendorff's alpha (agreement, not accuracy)")
        agreement_table.add_column("criterion")
        agreement_table.add_column("status")
        agreement_table.add_column("alpha", justify="right")
        agreement_table.add_column("flag")
        agreement_table.add_column("detail")
        for criterion, result in report.agreement_by_criterion.items():
            # "OK but below threshold" and "not OK at all" are reported as
            # distinct flags -- a NaN/raised alpha must never be
            # indistinguishable from "we measured it and agreement is
            # fine" (see agreement.py).
            if criterion in report.unmeasurable_criteria:
                flag = f"[bold red]{result.status}[/bold red]"
            elif criterion in report.low_agreement_criteria:
                flag = "[bold yellow]LOW AGREEMENT[/bold yellow]"
            else:
                flag = ""
            alpha_text = f"{result.alpha:.4f}" if result.alpha is not None else "n/a"
            agreement_table.add_row(criterion, result.status, alpha_text, flag, result.detail)
        console.print(agreement_table)

    if not report.honeypot_accuracy_by_curator:
        # `None` is not `0`: an empty table cannot tell a reader "no
        # honeypots were scheduled" apart from "the curator scored zero"
        # (M1's missing-score rule / M2's NOT_RUN, applied to a queue).
        console.print(
            f"[dim]honeypot accuracy: no honeypots were scheduled this session "
            f"({report.total_honeypot_items}/{report.total_queue_items} queue items are "
            "honeypots) -- nothing to measure curator accuracy against yet.[/dim]"
        )
    else:
        honeypot_table = Table(title="honeypot accuracy (correctness, not agreement)")
        honeypot_table.add_column("curator")
        honeypot_table.add_column("accuracy", justify="right")
        for curator, accuracy in sorted(report.honeypot_accuracy_by_curator.items()):
            honeypot_table.add_row(curator, f"{accuracy:.4f}" if accuracy is not None else "n/a")
        console.print(honeypot_table)

    if any(v is not None for v in report.honeypot_accuracy_by_session_decile.values()):
        decile_table = Table(title="honeypot accuracy by session decile")
        decile_table.add_column("decile")
        decile_table.add_column("accuracy", justify="right")
        for decile, accuracy in sorted(report.honeypot_accuracy_by_session_decile.items()):
            decile_table.add_row(str(decile), f"{accuracy:.4f}" if accuracy is not None else "n/a")
        console.print(decile_table)
    # else: suppressed entirely -- ten "n/a" rows say nothing an already-
    # printed "no honeypots were scheduled" line above has not said.

    def _flag_text(flag: SessionFlag) -> str:
        return f"{flag.curator_id} session {flag.session_index} (n={flag.n_decisions})"

    if report.autopilot_flags:
        console.print(
            "[bold yellow]AUTOPILOT[/bold yellow]: "
            + ", ".join(_flag_text(f) for f in report.autopilot_flags)
        )
    if report.speeding_flags:
        console.print(
            "[bold yellow]SPEEDING[/bold yellow]: "
            + ", ".join(_flag_text(f) for f in report.speeding_flags)
        )
    console.print(
        f"[dim]sessions are recommended to stay under {report.recommended_max_session_minutes} minutes — "
        "the documented threshold beyond which mental fatigue measurably degrades annotation quality.[/dim]"
    )

    if report.coverage is not None:
        cov = report.coverage
        console.print(
            f"content coverage: {cov.documents_with_questions}/{cov.total_documents} documents "
            f"({cov.doc_coverage_fraction:.2%})"
        )
        cell_table = Table(title="(qtype x difficulty) cell occupancy")
        cell_table.add_column("qtype")
        cell_table.add_column("difficulty")
        cell_table.add_column("count", justify="right")
        for (qtype, difficulty), count in sorted(cov.cell_occupancy.items()):
            cell_table.add_row(qtype, difficulty, str(count))
        console.print(cell_table)
    else:
        console.print("[dim]content coverage: no --corpus given, skipped[/dim]")

    if report.difficulty_correlation:
        corr_table = Table(title="proposed difficulty vs. measured features, incl. gold_doc_rank (Pearson r)")
        corr_table.add_column("feature")
        corr_table.add_column("r", justify="right")
        for feature, r in sorted(report.difficulty_correlation.items()):
            corr_table.add_row(feature, f"{r:.4f}")
        console.print(corr_table)
    else:
        console.print("[dim]difficulty/feature correlation: not enough variation to compute[/dim]")

    obedience = report.generator_obedience
    if obedience.n_candidates == 0:
        console.print("[dim]generator obedience: no candidates to measure[/dim]")
    else:
        assert obedience.qtype_match_rate is not None
        assert obedience.difficulty_match_rate is not None
        assert obedience.cell_match_rate is not None
        console.print(
            f"generator obedience (proposed matches the requested target cell), "
            f"n={obedience.n_candidates}: "
            f"qtype={obedience.qtype_match_rate:.1%}  "
            f"difficulty={obedience.difficulty_match_rate:.1%}  "
            f"cell={obedience.cell_match_rate:.1%}"
        )


_CLASSIFICATION_STYLE = {
    "mislabel_suspect": "bold red",
    "dead_all_pass": "dim",
    "dead_all_fail": "dim",
    "live": "",
}

#: Style for `DetectorValidation.verdict` in the `--validations` table
#: (Change 5) -- `not_validated` gets the same prominence as any other
#: WARNING-register verdict in this file (`reliability_verdict ==
#: "too_few_systems"`, `Estimate.verdict == "needs_sampling_design"`).
_DETECTOR_VALIDATION_STYLE = {
    "validated": "green",
    "not_validated": "bold red",
    "unmeasured": "dim",
}


def _estimate_text(estimate: Estimate | None) -> str:
    """`value [ci_lo, ci_hi]`, or `n/a` when there is nothing to report
    (`estimate is None`, or `Estimate.value is None` -- `verdict ==
    "no_denominator"`) -- shared by the `--validations` table below."""
    if estimate is None or estimate.value is None:
        return "n/a"
    ci_text = f" [{estimate.ci[0]:.3f}, {estimate.ci[1]:.3f}]" if estimate.ci is not None else ""
    return f"{estimate.value:.3f}{ci_text}"


def render_items_analysis(
    item_stats: Sequence[ItemStats],
    panel_stats: PanelStats,
    validations: Mapping[str, DetectorValidation] | None = None,
    console: Console | None = None,
) -> None:
    """Render `fireassay items analyse`'s output (M-ITEMS-SPEC.md §5).

    Leads with `split_half_reliability` and its verdict, ahead of any
    per-item number -- "a reader must see how much to trust the per-item
    numbers before they see the numbers" -- and prints an explicit warning
    when the panel cannot support per-item ranking (`reliability_verdict ==
    "too_few_systems"`); classification still stands regardless (see
    `items.core`'s module docstring). Never prints a composite score --
    `class_counts` plus the per-item table are the disposition, not a
    single number.

    A non-zero `mislabel_suspect` count in `class_counts` is followed
    immediately by an UNVALIDATED line (`items.core.MISLABEL_SUSPECT_VALIDATION`)
    -- it must be impossible to read the count without also reading that
    the classification failed measurement as a detector (defect 50; Goal
    2 criterion 4).

    `validations`, if supplied (`--validations`, loaded via
    `items.validation.load_validations`), prints a short table of every
    measured detector's precision/recall and derived verdict before the
    item scorecard (Goal 2 criterion 3) -- a `not_validated` record prints
    in the same warning register as everything else in this function.
    """
    console = console or Console()
    verdict_style = "green" if panel_stats.reliability_verdict == "usable" else "bold red"
    console.print(
        f"split_half_reliability = {panel_stats.split_half_reliability:.4f}  "
        f"(measured over {panel_stats.n_items_used_for_reliability} of {panel_stats.n_items} "
        "items -- zero-variance items are excluded, see items.core's module docstring)  "
        f"[{verdict_style}]{panel_stats.reliability_verdict}[/{verdict_style}]"
    )
    if panel_stats.reliability_verdict == "too_few_systems":
        console.print(
            "[bold red]WARNING[/bold red]: too few systems to trust per-item "
            "discrimination_d / point_biserial -- classification (live / dead_all_pass / "
            "dead_all_fail / mislabel_suspect) still stands, per-item ranking does not."
        )

    console.print(
        f"n_items={panel_stats.n_items}  n_systems={panel_stats.n_systems}  "
        f"class_counts={panel_stats.class_counts}"
    )
    mislabel_count = panel_stats.class_counts.get("mislabel_suspect", 0)
    if mislabel_count > 0:
        console.print(
            f"[bold red]UNVALIDATED[/bold red]: {mislabel_count} item(s) flagged "
            f"{MISLABEL_SUSPECT_VALIDATION.detector} -- {MISLABEL_SUSPECT_VALIDATION.note}"
        )
    if panel_stats.claimed_vs_measured_difficulty_r is not None:
        console.print(
            "claimed-vs-measured difficulty correlation r = "
            f"{panel_stats.claimed_vs_measured_difficulty_r:.4f} "
            "-- a diagnostic only; never trust a claimed difficulty label on its own"
        )
    else:
        console.print("[dim]claimed-vs-measured difficulty correlation: no claimed labels supplied[/dim]")

    if validations is not None:
        val_table = Table(title="detector validation status (measured precision/recall, items.validation)")
        val_table.add_column("detector")
        val_table.add_column("precision", justify="right")
        val_table.add_column("recall", justify="right")
        val_table.add_column("base_rate", justify="right")
        val_table.add_column("verdict")
        val_table.add_column("measured_on")
        for name in sorted(validations):
            record = validations[name]
            style = _DETECTOR_VALIDATION_STYLE.get(record.verdict, "")
            verdict_text = f"[{style}]{record.verdict}[/{style}]" if style else record.verdict
            val_table.add_row(
                record.detector,
                _estimate_text(record.precision),
                _estimate_text(record.recall),
                f"{record.base_rate:.3f}",
                verdict_text,
                record.measured_on,
            )
        console.print(val_table)

    table = Table(title="item scorecard (a disposition, not a composite score)")
    table.add_column("item_id")
    table.add_column("p", justify="right")
    table.add_column("discrimination_d", justify="right")
    table.add_column("point_biserial", justify="right")
    table.add_column("classification")
    for s in item_stats:
        style = _CLASSIFICATION_STYLE.get(s.classification, "")
        classification_text = f"[{style}]{s.classification}[/{style}]" if style else s.classification
        table.add_row(
            s.item_id,
            f"{s.p:.3f}",
            f"{s.discrimination_d:+.3f}",
            f"{s.point_biserial:+.3f}",
            classification_text,
        )
    console.print(table)


def render_items_score(score: DetectorScore, console: Console | None = None) -> None:
    """Render `fireassay items review score`'s output (M-ITEMS-SPEC.md
    §3)."""
    console = console or Console()
    console.print(
        "[dim]seeded recall is an UPPER BOUND: a seeded flaw may be easier to detect than "
        "a naturally occurring one.[/dim]"
    )
    table = Table(title="blind review: detector score")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("n", justify="right")

    precision_ci = score.precision_ci
    precision_text = f"{score.precision:.3f}" if score.precision is not None else "n/a"
    precision_ci_text = (
        f"[{precision_ci[0]:.3f}, {precision_ci[1]:.3f}]" if precision_ci is not None else "n/a"
    )
    table.add_row("precision (of flagged items)", precision_text, precision_ci_text, str(score.precision_n))

    recall_ci = score.recall_ci
    recall_text = f"{score.recall:.3f}" if score.recall is not None else "n/a"
    recall_ci_text = f"[{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]" if recall_ci is not None else "n/a"
    table.add_row("recall (of seeded items, UPPER BOUND)", recall_text, recall_ci_text, str(score.recall_n))

    console.print(table)


#: How each `DetectorEvaluation.stratum_estimator` value is described in
#: the "computed over" column -- printed alongside every row so the choice
#: of formula (`items.calibration.evaluate_detector`'s module docstring)
#: is visible at the point of use, not only to a reader who opens that
#: docstring.
_STRATUM_ESTIMATOR_TEXT = {
    "random_stratum_only": "random stratum only",
    "stratified": "stratified (census + sample)",
}

_ESTIMATE_VERDICT_STYLE = {
    "measured": "green",
    "too_few_labels": "bold yellow",
    "no_denominator": "dim",
    "needs_sampling_design": "bold red",
}


def render_detector_evaluation(evaluation: DetectorEvaluation, console: Console | None = None) -> None:
    """Render `fireassay items detector-score`'s output: every statistic
    with its value, 95% CI, `n`, and `verdict` -- never a bare value --
    and which formula computed `recall`/`fpr`/`base_rate` (`"random
    stratum only"` vs `"stratified (census + sample)"`), so
    `evaluate_detector`'s stratum rule (pooling the flagged census with
    the random calibration draw for anything but `precision` over-weights
    the census and misreports the suite's real defect rate) is visible at
    the point of use. When `verdict == "needs_sampling_design"`, an
    explicit warning follows the table -- a reader must never mistake a
    structurally-unmeasurable recall/fpr for a detector with zero false
    positives (see `evaluate_detector`'s docstring on the disjoint-stratum
    trap)."""
    console = console or Console()
    table = Table(title="detector evaluation (calibration labels)")
    table.add_column("statistic")
    table.add_column("value", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("n", justify="right")
    table.add_column("verdict")
    table.add_column("computed over")

    stratum_text = _STRATUM_ESTIMATOR_TEXT[evaluation.stratum_estimator]
    for name in ("precision", "recall", "fpr", "base_rate"):
        estimate: Estimate = getattr(evaluation, name)
        value_text = f"{estimate.value:.4f}" if estimate.value is not None else "n/a"
        ci_text = f"[{estimate.ci[0]:.4f}, {estimate.ci[1]:.4f}]" if estimate.ci is not None else "n/a"
        style = _ESTIMATE_VERDICT_STYLE.get(estimate.verdict, "")
        verdict_text = f"[{style}]{estimate.verdict}[/{style}]" if style else estimate.verdict
        computed_over = "all labelled flagged items, any stratum" if name == "precision" else stratum_text
        table.add_row(name, value_text, ci_text, str(estimate.n), verdict_text, computed_over)
    console.print(table)

    if evaluation.recall.verdict == "needs_sampling_design":
        console.print(
            "[bold red]needs_sampling_design[/bold red]: the random (calibration) stratum "
            "contains none of this detector's flagged items, so recall and FPR cannot be "
            "computed from it directly -- this is NOT a detector with zero false positives. "
            "Supply --pool-size and --flagged-size to enable the stratified "
            "(census + sample) estimator."
        )

    console.print(
        f"n_labels={evaluation.n_labels}  n_random_stratum={evaluation.n_random_stratum}  "
        f"n_flagged_total={evaluation.n_flagged_total}  "
        f"n_flagged_labelled={evaluation.n_flagged_labelled}  "
        f"n_flagged_unlabelled={evaluation.n_flagged_unlabelled}"
    )
