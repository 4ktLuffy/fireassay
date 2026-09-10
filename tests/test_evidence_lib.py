"""`fireassay.evidence` -- the claims runner as a library and as
`fireassay evidence --claims <file>`.

`tests/test_evidence.py` checks this repo's own claims file structurally;
this file checks the runner itself, on a throwaway claims file written
into `tmp_path`, including the two negative controls that matter: a
claim whose recompute disagrees with its expected value FAILS (and the
exit code says so), and a claims file with a duplicate id is refused
before anything is recomputed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from fireassay.cli import app
from fireassay.evidence import (
    EvidenceError,
    Metric,
    default_style,
    load_claims,
    render_markdown,
    run_check,
    run_one_claim,
)

runner = CliRunner()

_CLAIMS_FILE = '''
import json
from pathlib import Path
from fireassay.evidence import Claim, ClaimResult, Metric

ARTIFACT = Path(__file__).parent / "numbers.json"


def _recompute_good() -> ClaimResult:
    data = json.loads(ARTIFACT.read_text())
    return ClaimResult(values={"n": float(data["n"]), "verdict": data["verdict"]}, detail="read numbers.json")


def _recompute_bad() -> ClaimResult:
    data = json.loads(ARTIFACT.read_text())
    return ClaimResult(values={"n": float(data["n"])})


def _recompute_raises() -> ClaimResult:
    raise RuntimeError("artifact missing a field")


CLAIMS = (
    Claim("good.n", "Group A", "n is 14 and the verdict is usable", ("numbers.json",),
          (Metric("n", 14.0), Metric("verdict", "usable")), _recompute_good),
    Claim("bad.n", "Group A", "n is asserted to be 15, which it is not", ("numbers.json",),
          (Metric("n", 15.0, 0.5),), _recompute_bad),
    Claim("broken.recompute", "Group B", "the recompute raises", ("numbers.json",),
          (Metric("n", 14.0),), _recompute_raises),
)
'''


@pytest.fixture
def claims_dir(tmp_path: Path) -> Path:
    (tmp_path / "numbers.json").write_text('{"n": 14, "verdict": "usable"}', encoding="utf-8")
    (tmp_path / "claims.py").write_text(_CLAIMS_FILE, encoding="utf-8")
    return tmp_path


def test_metric_check_semantics() -> None:
    assert Metric("x", 1.0, 0.1).check(1.05)[0]
    assert not Metric("x", 1.0, 0.01).check(1.05)[0]
    assert Metric("v", "usable").check("usable")[0]
    assert not Metric("v", "usable").check("too_few_systems")[0]
    # A bool is not a number: True == 1 would otherwise pass a numeric check.
    assert not Metric("x", 1.0).check(True)[0]
    assert not Metric("x", 1.0).check("1.0")[0]


def test_load_claims_and_check(claims_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    claims = load_claims(claims_dir / "claims.py")
    assert [c.id for c in claims] == ["good.n", "bad.n", "broken.recompute"]

    assert run_check(claims) == 1
    out = capsys.readouterr().out
    assert "PASS good.n" in out
    assert "FAIL bad.n" in out
    assert "ERROR broken.recompute: RuntimeError" in out

    assert run_check(claims[:1]) == 0
    assert run_one_claim(claims, "good.n") == 0
    assert run_one_claim(claims, "bad.n") == 1
    assert run_one_claim(claims, "no.such") == 2


def test_duplicate_claim_ids_are_refused_before_recompute(claims_dir: Path) -> None:
    claims = load_claims(claims_dir / "claims.py")
    assert run_check([claims[0], claims[0]]) == 2


def test_load_claims_errors_are_typed(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError):
        load_claims(tmp_path / "missing.py")
    (tmp_path / "noclaims.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(EvidenceError):
        load_claims(tmp_path / "noclaims.py")


def test_markdown_lists_every_claim_with_its_expected_values(claims_dir: Path) -> None:
    claims = load_claims(claims_dir / "claims.py")
    text = render_markdown(claims, default_style("tools/claims.py"))
    for claim in claims:
        assert f"### `{claim.id}`" in text
    assert "## Group A" in text and "## Group B" in text
    assert "- expected `n`: `14.0` (tolerance `0.0`)" in text
    assert "- expected `verdict`: `usable`" in text
    assert "fireassay evidence --claims tools/claims.py --claim good.n" in text
    assert "Claims deliberately excluded" not in text


def test_cli_check_markdown_and_claim(claims_dir: Path) -> None:
    claims_path = str(claims_dir / "claims.py")
    result = runner.invoke(app, ["evidence", "--claims", claims_path, "--check"])
    assert result.exit_code == 1, result.output
    assert "FAIL bad.n" in result.output

    result = runner.invoke(app, ["evidence", "--claims", claims_path, "--claim", "good.n"])
    assert result.exit_code == 0, result.output
    assert "PASS good.n" in result.output

    result = runner.invoke(app, ["evidence", "--claims", claims_path, "--markdown"])
    assert result.exit_code == 0, result.output
    assert "# EVIDENCE.md" in result.output and "### `bad.n`" in result.output

    result = runner.invoke(app, ["evidence", "--claims", claims_path])
    assert result.exit_code == 1
    result = runner.invoke(app, ["evidence", "--claims", str(claims_dir / "nope.py"), "--check"])
    assert result.exit_code == 1 and "claims file not found" in result.output


def test_repo_claims_file_runs_through_the_cli() -> None:
    """The repo's own tools/evidence.py is a valid claims file for the
    generic runner -- the extraction did not fork the two."""
    repo = Path(__file__).resolve().parents[1]
    result = runner.invoke(app, ["evidence", "--claims", str(repo / "tools" / "evidence.py"), "--markdown"])
    assert result.exit_code == 0, result.output
    assert "### `gate.planted_regression_blocked`" in result.output
