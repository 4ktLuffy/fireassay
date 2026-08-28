"""Render `ComparabilityReport`/`Leaderboard` objects, (M2) control and
mutation results, and (M3) curation funnel reports, to the terminal via
`rich`."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from fireassay.compare import Leaderboard
from fireassay.controls.base import ControlOutcome
from fireassay.curate.report import CurateReport, SessionFlag
from fireassay.integrity import RefusalReason
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
