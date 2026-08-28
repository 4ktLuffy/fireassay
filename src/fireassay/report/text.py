"""Render `ComparabilityReport`/`Leaderboard` objects, and (M2) control and
mutation results, to the terminal via `rich`."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from fireassay.compare import Leaderboard
from fireassay.controls.base import ControlOutcome
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
