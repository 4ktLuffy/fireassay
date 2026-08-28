"""Render `ComparabilityReport`/`Leaderboard` objects to the terminal via
`rich`."""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from fireassay.compare import Leaderboard
from fireassay.integrity import RefusalReason


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
