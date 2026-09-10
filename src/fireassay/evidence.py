"""The evidence discipline as a library: claims that name a committed
artifact, a recompute function and an expected value, checked by
recomputing and never by rewriting.

This is the generic half of what `tools/evidence.py` did for fireassay's
own EVIDENCE.md, extracted so a second repository can have the same
guarantee with a claims file and nothing else. A `Claim` is: an id, a
group heading, a one-line description, the artifact path(s) it reads, a
`recompute()` that returns a `ClaimResult`, and the `Metric`s -- expected
value plus tolerance -- that result is checked against.

Three operations, from a claims module or from the CLI
(`fireassay evidence --claims path/to/claims.py ...`):

- `--check`     recompute every claim, print PASS/FAIL/ERROR per claim
                with its detail, exit non-zero on any failure;
- `--markdown`  render the registered claims as an EVIDENCE.md -- the
                *expected* values, never a live recompute, so the file
                documents what is asserted and `--check` is the only
                place anything is verified;
- `--claim ID`  recompute and print one claim in detail.

**A claim whose recompute disagrees with its expected value is a FAIL,
never a silently updated number.** Nothing here writes to a claims file.
That asymmetry is the whole point: the two defects that motivated it
(a load-bearing number whose instrument was never committed; a `0%`
quoted as a finding that was an artifact of the experiment) both came
from prose that could not be re-run. A claims file is the instrument,
stored.

Claims are ordinary Python so a recompute can use whatever the project's
own code offers; the only contract is that it reads committed artifacts
(never a database or a model), returns every metric name it is checked
against, and raises `EvidenceError` when the artifacts are internally
inconsistent in a way a tolerance check cannot express.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class EvidenceError(RuntimeError):
    """A claim's recompute function found the committed artifacts it read
    to be internally inconsistent in a way no `Metric` tolerance check
    could express (e.g. a missing item id) -- raised instead of silently
    dropping or zero-filling the offending data."""


@dataclass(frozen=True)
class Metric:
    """One number (or verdict string) a `Claim`'s recompute is checked
    against. `expected` a `str` means exact-match (a verdict like
    `"usable"`); a `float` means `abs(actual - expected) <= tolerance`."""

    name: str
    expected: float | str
    tolerance: float = 0.0

    def check(self, actual: float | str) -> tuple[bool, str]:
        if isinstance(self.expected, str):
            ok = actual == self.expected
            return ok, f"{self.name}={actual!r} (expected {self.expected!r})"
        if isinstance(actual, bool) or not isinstance(actual, int | float):
            return False, f"{self.name}: expected a number, got {actual!r}"
        ok = abs(float(actual) - self.expected) <= self.tolerance
        return ok, f"{self.name}={actual!r} (expected {self.expected!r} +/- {self.tolerance!r})"


@dataclass(frozen=True)
class ClaimResult:
    """One `recompute()` call's output: every `Metric.name` this claim
    declares must have a matching key in `values`. `detail` is free text
    for `--claim`/`--check` to print alongside the pass/fail line -- the
    full counts a `Metric`'s single number cannot carry on its own."""

    values: dict[str, float | str]
    detail: str = ""


@dataclass(frozen=True)
class Claim:
    id: str
    group: str
    description: str
    artifacts: tuple[str, ...]
    metrics: tuple[Metric, ...]
    recompute: Callable[[], ClaimResult]


@dataclass(frozen=True)
class MarkdownStyle:
    """Everything project-specific about the rendered EVIDENCE.md: the
    commands a reader is told to run, the opening paragraph, and an
    optional closing section for claims the project deliberately does not
    make. `verify_command` is formatted with `{claim_id}`."""

    regenerate_command: str
    check_command: str
    verify_command: str
    preamble: str
    excluded_note: str | None = None


DEFAULT_PREAMBLE = (
    "Every claim below names the committed artifact(s) it reads and the command that "
    "recomputes and verifies it. The check command recomputes every claim fresh from those "
    "artifacts and compares it against the expected value encoded in the claims file; a "
    "mismatch prints FAIL and the run exits non-zero -- it is never silently rewritten to "
    "match whatever the recompute produced."
)


def default_style(claims_path: str) -> MarkdownStyle:
    """The style `fireassay evidence --claims <path>` renders with."""
    base = f"fireassay evidence --claims {claims_path}"
    return MarkdownStyle(
        regenerate_command=f"{base} --markdown > EVIDENCE.md",
        check_command=f"{base} --check",
        verify_command=f"{base} --claim {{claim_id}}",
        preamble=DEFAULT_PREAMBLE,
    )


def render_markdown(claims: Sequence[Claim], style: MarkdownStyle) -> str:
    """Format `claims` as an EVIDENCE.md -- the registered `expected`
    value/tolerance for every metric, never a live recompute."""
    lines: list[str] = [
        "<!--",
        "GENERATED. Do not hand-edit.",
        f"Regenerate: {style.regenerate_command}",
        f"Verify:     {style.check_command}",
        "-->",
        "",
        "# EVIDENCE.md",
        "",
        style.preamble,
        "",
    ]

    groups: list[str] = []
    for claim in claims:
        if claim.group not in groups:
            groups.append(claim.group)

    for group in groups:
        lines.append(f"## {group}")
        lines.append("")
        for claim in claims:
            if claim.group != group:
                continue
            lines.append(f"### `{claim.id}`")
            lines.append("")
            lines.append(claim.description)
            lines.append("")
            artifact_list = ", ".join(f"`{artifact}`" for artifact in claim.artifacts)
            lines.append(f"- artifact(s): {artifact_list}")
            lines.append(f"- verify: `{style.verify_command.format(claim_id=claim.id)}`")
            for metric in claim.metrics:
                if isinstance(metric.expected, str):
                    lines.append(f"- expected `{metric.name}`: `{metric.expected}`")
                else:
                    lines.append(
                        f"- expected `{metric.name}`: `{metric.expected}` "
                        f"(tolerance `{metric.tolerance}`)"
                    )
            lines.append("")

    if style.excluded_note:
        lines.append("## Claims deliberately excluded")
        lines.append("")
        lines.append(style.excluded_note)
    return "\n".join(lines)


def check_one(claim: Claim) -> bool:
    """Recompute `claim`, print a PASS/FAIL (or ERROR) line and its
    detail, and return whether it passed. A recompute that raises is an
    ERROR and counts as a failure -- never skipped."""
    try:
        result = claim.recompute()
    except Exception as exc:  # noqa: BLE001 -- a failed recompute is itself a FAIL, never skipped
        print(f"ERROR {claim.id}: {type(exc).__name__}: {exc}")
        return False

    ok = True
    messages: list[str] = []
    for metric in claim.metrics:
        if metric.name not in result.values:
            print(f"ERROR {claim.id}: recompute did not report metric {metric.name!r}")
            return False
        passed, message = metric.check(result.values[metric.name])
        ok = ok and passed
        messages.append(message)

    print(f"{'PASS' if ok else 'FAIL'} {claim.id}: {'; '.join(messages)}")
    if result.detail:
        print(f"     {result.detail}")
    return ok


def run_check(claims: Sequence[Claim]) -> int:
    """Check every claim; 0 iff all passed. Duplicate claim ids are an
    error before anything is recomputed -- two claims with one name would
    let a FAIL hide behind a PASS in a log."""
    seen: set[str] = set()
    for claim in claims:
        if claim.id in seen:
            print(f"ERROR duplicate claim id {claim.id!r}")
            return 2
        seen.add(claim.id)
    any_failed = False
    for claim in claims:
        if not check_one(claim):
            any_failed = True
    return 1 if any_failed else 0


def run_one_claim(claims: Sequence[Claim], claim_id: str) -> int:
    by_id = {claim.id: claim for claim in claims}
    claim = by_id.get(claim_id)
    if claim is None:
        known = ", ".join(sorted(by_id))
        print(f"unknown claim id {claim_id!r}. Known ids: {known}", file=sys.stderr)
        return 2
    print(f"{claim.id}: {claim.description}")
    print(f"artifacts: {', '.join(claim.artifacts)}")
    return 0 if check_one(claim) else 1


def load_claims(path: Path) -> tuple[Claim, ...]:
    """Import a claims file by path and return its `CLAIMS` tuple.

    The module is registered in `sys.modules` under a name derived from
    the file *before* it executes -- `dataclasses` resolves a class's own
    module through `sys.modules` while building it, so a claims file that
    defines a dataclass would otherwise fail to import at all.
    """
    path = path.resolve()
    if not path.is_file():
        raise EvidenceError(f"claims file not found: {path}")
    module_name = f"_fireassay_claims_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"cannot import claims file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    claims = getattr(module, "CLAIMS", None)
    if not isinstance(claims, tuple | list) or not all(isinstance(c, Claim) for c in claims):
        raise EvidenceError(f"{path} must define CLAIMS: a tuple of fireassay.evidence.Claim")
    return tuple(claims)


def main(
    claims: Sequence[Claim],
    argv: Sequence[str] | None = None,
    *,
    style: MarkdownStyle,
    description: str | None = None,
) -> int:
    """The `--check` / `--markdown` / `--claim ID` entry point a claims
    file's `if __name__ == "__main__":` block calls with its own
    `CLAIMS` and `MarkdownStyle`."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="recompute every claim and verify it")
    mode.add_argument("--markdown", action="store_true", help="emit EVIDENCE.md to stdout")
    mode.add_argument("--claim", metavar="ID", help="recompute and print one claim in detail")
    args = parser.parse_args(argv)

    if args.markdown:
        print(render_markdown(claims, style))
        return 0
    if args.claim:
        return run_one_claim(claims, args.claim)
    return run_check(claims)
