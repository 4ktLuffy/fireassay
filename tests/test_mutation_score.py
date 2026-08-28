from __future__ import annotations

import pytest

from fireassay.mutation.score import MutantResult, MutationScoreResult, compute_score


def _mutant(operator: str, killed: bool, equivalent: bool, reason: str | None = None) -> MutantResult:
    return MutantResult(
        operator=operator,
        params={},
        mutant_run_id=f"run-{operator}",
        killed=killed,
        equivalent=equivalent,
        equivalent_reason=reason if equivalent else None,
        detail="",
    )


def test_score_arithmetic() -> None:
    mutants = [
        _mutant("drop_results", killed=True, equivalent=False),
        _mutant("truncate_topk", killed=False, equivalent=False),
        _mutant("shuffle_topk", killed=True, equivalent=False),
        _mutant("corrupt_query", killed=True, equivalent=True, reason="pct<=0"),
    ]
    killed, total, equivalent, score = compute_score(mutants)
    assert total == 4
    assert equivalent == 1
    assert killed == 2  # the equivalent mutant's `killed=True` does not count
    assert score == pytest.approx(2 / 3)  # killed / (total - equivalent) = 2 / (4 - 1)


def test_all_equivalent_gives_zero_score_not_a_crash() -> None:
    mutants = [_mutant("drop_results", killed=False, equivalent=True, reason="n<=0")]
    killed, total, equivalent, score = compute_score(mutants)
    assert equivalent == 1
    assert total == 1
    assert score == 0.0


def test_equivalent_mutant_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="equivalent_reason"):
        MutantResult(
            operator="drop_results",
            params={},
            mutant_run_id="r1",
            killed=False,
            equivalent=True,
            equivalent_reason=None,
            detail="",
        )


def test_survivors_are_not_equivalent_and_not_killed() -> None:
    survivor = _mutant("truncate_topk", killed=False, equivalent=False)
    killed = _mutant("drop_results", killed=True, equivalent=False)
    excluded = _mutant("corrupt_query", killed=True, equivalent=True, reason="pct<=0")

    result = MutationScoreResult(
        suite_id="s1", suite_hash="h1", base_config_id="c1", detector="threshold@1.0.0",
        killed=1, total=3, equivalent=1, score=0.5,
        mutants=(killed, survivor, excluded),
    )
    assert result.survivors() == (survivor,)


def test_sorted_for_report_lists_survivors_first() -> None:
    survivor = _mutant("truncate_topk", killed=False, equivalent=False)
    killed_mutant = _mutant("drop_results", killed=True, equivalent=False)
    excluded = _mutant("corrupt_query", killed=True, equivalent=True, reason="pct<=0")
    result = MutationScoreResult(
        suite_id="s1", suite_hash="h1", base_config_id="c1", detector="threshold@1.0.0",
        killed=1, total=3, equivalent=1, score=0.5,
        mutants=(killed_mutant, excluded, survivor),
    )
    ordered = result.sorted_for_report()
    assert ordered[0] is survivor
    assert ordered[1] is killed_mutant
    assert ordered[2] is excluded
