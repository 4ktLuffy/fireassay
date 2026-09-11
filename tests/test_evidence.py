"""`test_evidence.py`: structural tests for `tools/evidence.py`'s claim
registry -- every claim names a real, git-tracked artifact and a
non-empty id/description/expected value; ids are unique; `--markdown`
mentions every one.

**Deliberately does NOT test `--check`.** `--check` recomputes every
claim fresh against real `run/` artifacts (`run/panel54_matrix.csv`
alone is 2,364 items x 54 systems) and is a verification command, not a
unit test -- see `tools/evidence.py`'s own module docstring for the same
reasoning. Run it directly instead:

    .venv/bin/python tools/evidence.py --check

`test_every_artifact_exists_and_is_not_gitignored` is the defect-46
regression test: a claim pointing at an artifact that does not exist, or
exists but is gitignored (`run/golden.db`, for instance -- see
`.gitignore`), must fail this suite rather than ship silently.

`test_every_artifact_is_tracked_by_git` is its companion, and **skips
rather than passes** while a new artifact is still awaiting commit -- see
its docstring for why a skip, and not a pass, is the honest outcome
there.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Cache for `_load_evidence_module` -- one module object, loaded once
#: per test session and reused, mirroring `tests/helpers.py.
#: load_measure_detector_recall`'s own caching for the same reason.
_evidence_module: ModuleType | None = None


def _load_evidence_module() -> ModuleType:
    """`tools/` is a scripts directory, not an importable package (no
    `tools/__init__.py`, deliberately -- see `tests/helpers.py.
    load_measure_detector_recall`'s docstring) -- load
    `tools/evidence.py` by file path so these tests exercise the actual
    module, never a copy of its logic.

    **Registers the module in `sys.modules` under its spec name BEFORE
    `exec_module` runs.** Required, not a formality: `evidence.py`
    defines `@dataclass` classes, and `dataclasses` resolves a class's
    own module via `sys.modules.get(cls.__module__)` to read its
    annotations -- with the module absent from `sys.modules` that lookup
    returns `None` and dataclass construction fails. See
    `tests/helpers.py.load_measure_detector_recall`'s docstring for the
    same fix applied to a different `tools/` module.
    """
    global _evidence_module
    if _evidence_module is not None:
        return _evidence_module

    spec = importlib.util.spec_from_file_location("evidence", _REPO_ROOT / "tools" / "evidence.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # must precede exec_module -- see docstring above
    spec.loader.exec_module(module)
    _evidence_module = module
    return module


def _is_git_tracked(artifact: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", artifact],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _is_git_ignored(artifact: str) -> bool:
    """Whether `.gitignore` excludes `artifact` — the defect-46 case
    itself (`run/golden.db` is ignored, so a claim reading it would ship a
    command a stranger cannot run)."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", artifact],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def test_every_claim_has_id_description_artifacts_and_expected_value() -> None:
    evidence = _load_evidence_module()
    for claim in evidence.CLAIMS:
        assert claim.id, "a claim has an empty id"
        assert claim.description.strip(), f"{claim.id}: empty description"
        assert claim.artifacts, f"{claim.id}: no artifact(s) named"
        assert claim.metrics, f"{claim.id}: no expected value(s) (metrics)"
        for metric in claim.metrics:
            assert metric.name, f"{claim.id}: a metric has an empty name"
            assert metric.expected is not None, f"{claim.id}: metric {metric.name!r} has no expected value"


def test_every_artifact_exists_and_is_not_gitignored() -> None:
    """The defect-46 regression test proper: a claim must never read an
    artifact a stranger who clones this repo cannot get.

    `run/golden.db` is the case this exists for — it is gitignored, so a
    claim pointing at it would ship a `--check` command that fails
    everywhere but on the machine that generated it. Missing and ignored
    are both failures here; *untracked but committable* is a separate
    question, checked by the test below."""
    evidence = _load_evidence_module()
    failures: list[str] = []
    for claim in evidence.CLAIMS:
        for artifact in claim.artifacts:
            if not (_REPO_ROOT / artifact).exists():
                failures.append(f"{claim.id}: {artifact!r} does not exist")
            elif _is_git_ignored(artifact):
                failures.append(f"{claim.id}: {artifact!r} exists but is GITIGNORED")
    assert not failures, "\n".join(failures)


def test_every_artifact_is_tracked_by_git() -> None:
    """Every claim's artifact is in the repository, not just on this
    machine.

    **Skipped, never passed, while an artifact is still uncommitted.** A
    builder is not allowed to commit (they leave work in the tree for
    review), so a brand-new artifact is legitimately untracked for the
    length of one review cycle — but "not checked yet" must not read as
    "checked and fine", which is this project's governing rule for
    controls (M2-SPEC.md §1) applied to its own test suite. The skip names
    the files, so the reason is on the report rather than in someone's
    memory. On any committed tree — CI included, which tests exactly what
    was pushed — nothing is pending and this runs for real.
    """
    evidence = _load_evidence_module()
    artifacts = sorted({a for claim in evidence.CLAIMS for a in claim.artifacts})
    present = [a for a in artifacts if (_REPO_ROOT / a).exists() and not _is_git_ignored(a)]
    pending = [a for a in present if not _is_git_tracked(a)]
    if pending:
        pytest.skip(
            "NOT CHECKED: claim artifact(s) exist but are not yet committed, so this test could "
            f"not run: {pending}. Commit them and it will."
        )
    untracked = [a for a in artifacts if not _is_git_tracked(a)]
    assert not untracked, f"claim artifact(s) not tracked by git: {untracked}"


def test_markdown_contains_every_claim_id() -> None:
    evidence = _load_evidence_module()
    markdown = evidence.render_markdown(evidence.CLAIMS)
    missing = [claim.id for claim in evidence.CLAIMS if claim.id not in markdown]
    assert not missing, f"claim id(s) missing from --markdown output: {missing}"


def test_claim_ids_are_unique() -> None:
    evidence = _load_evidence_module()
    ids = [claim.id for claim in evidence.CLAIMS]
    duplicates = sorted({claim_id for claim_id in ids if ids.count(claim_id) > 1})
    assert not duplicates, f"duplicate claim id(s): {duplicates}"
