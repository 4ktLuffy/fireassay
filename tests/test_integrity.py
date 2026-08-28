from __future__ import annotations

import pytest

from fireassay.integrity import ComparisonRefusedError, assert_comparable
from fireassay.models import Run


def _run(**overrides: object) -> Run:
    defaults: dict[str, object] = {
        "id": "run1",
        "suite_id": "suite1",
        "suite_hash": "hashA",
        "config_id": "cfg1",
        "config_hash": "cfghashA",
        "env_json": {"python": "3.12", "fireassay": "0.1.0", "affects_results": {}},
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "status": "complete",
        "result_count": 10,
    }
    defaults.update(overrides)
    return Run(**defaults)  # type: ignore[arg-type]


def test_suite_mismatch_fires_alone() -> None:
    a = _run(id="a", suite_hash="hashA")
    b = _run(id="b", suite_hash="hashB")
    scorers = {"a": frozenset({"s@1.0.0"}), "b": frozenset({"s@1.0.0"})}
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([a, b], scorers_by_run=scorers)
    codes = {r.code for r in exc_info.value.reasons}
    assert codes == {"SUITE_MISMATCH"}


def test_scorer_mismatch_fires_alone() -> None:
    a = _run(id="a")
    b = _run(id="b")
    scorers = {"a": frozenset({"retrieval@1.0.0"}), "b": frozenset({"retrieval@1.1.0"})}
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([a, b], scorers_by_run=scorers)
    codes = {r.code for r in exc_info.value.reasons}
    assert codes == {"SCORER_MISMATCH"}


def test_env_mismatch_fires_alone() -> None:
    a = _run(id="a", env_json={"python": "3.12", "fireassay": "0.1.0", "affects_results": {"embed": "x"}})
    b = _run(id="b", env_json={"python": "3.12", "fireassay": "0.1.0", "affects_results": {"embed": "y"}})
    scorers = {"a": frozenset({"s@1.0.0"}), "b": frozenset({"s@1.0.0"})}
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([a, b], scorers_by_run=scorers)
    codes = {r.code for r in exc_info.value.reasons}
    assert codes == {"ENV_MISMATCH"}


def test_env_mismatch_ignores_python_and_fireassay_version_fields() -> None:
    a = _run(id="a", env_json={"python": "3.12.0", "fireassay": "0.1.0", "affects_results": {}})
    b = _run(id="b", env_json={"python": "3.12.9", "fireassay": "0.1.1", "affects_results": {}})
    scorers = {"a": frozenset({"s@1.0.0"}), "b": frozenset({"s@1.0.0"})}
    report = assert_comparable([a, b], scorers_by_run=scorers)
    assert report.comparable is True


def test_run_incomplete_fires_alone() -> None:
    a = _run(id="a", status="running")
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([a], scorers_by_run={"a": frozenset({"s@1.0.0"})})
    codes = {r.code for r in exc_info.value.reasons}
    assert codes == {"RUN_INCOMPLETE"}


def test_empty_run_fires_alone() -> None:
    a = _run(id="a", result_count=0)
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([a], scorers_by_run={"a": frozenset({"s@1.0.0"})})
    codes = {r.code for r in exc_info.value.reasons}
    assert codes == {"EMPTY_RUN"}


def test_run_inadmissible_fires_alone() -> None:
    a = _run(id="a", admissible=False)
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([a], scorers_by_run={"a": frozenset({"s@1.0.0"})})
    codes = {r.code for r in exc_info.value.reasons}
    assert codes == {"RUN_INADMISSIBLE"}


def test_admissible_true_does_not_fire() -> None:
    a = _run(id="a", admissible=True)
    report = assert_comparable([a], scorers_by_run={"a": frozenset({"s@1.0.0"})})
    assert report.comparable is True


def test_force_returns_forced_report_without_raising() -> None:
    a = _run(id="a", suite_hash="hashA")
    b = _run(id="b", suite_hash="hashB")
    scorers = {"a": frozenset({"s@1.0.0"}), "b": frozenset({"s@1.0.0"})}
    report = assert_comparable([a, b], scorers_by_run=scorers, force=True)
    assert report.forced is True
    assert report.comparable is False
    assert any(r.code == "SUITE_MISMATCH" for r in report.reasons)


def test_genuinely_comparable_pair_passes() -> None:
    a = _run(id="a")
    b = _run(id="b")
    scorers = {"a": frozenset({"s@1.0.0"}), "b": frozenset({"s@1.0.0"})}
    report = assert_comparable([a, b], scorers_by_run=scorers)
    assert report.comparable is True
    assert report.reasons == ()
    assert report.forced is False


def test_multiple_reasons_all_collected_together() -> None:
    a = _run(id="a", suite_hash="hashA", status="running")
    b = _run(id="b", suite_hash="hashB", result_count=0)
    scorers = {"a": frozenset({"x@1.0.0"}), "b": frozenset({"y@1.0.0"})}
    with pytest.raises(ComparisonRefusedError) as exc_info:
        assert_comparable([a, b], scorers_by_run=scorers)
    codes = {r.code for r in exc_info.value.reasons}
    assert {"SUITE_MISMATCH", "SCORER_MISMATCH", "RUN_INCOMPLETE", "EMPTY_RUN"} <= codes
