"""`curate/coverage.py`: content coverage, and difficulty/feature
correlation including `gold_doc_rank` as a nullable, retrieval-based
feature alongside the always-present lexical ones."""

from __future__ import annotations

from typing import Literal

import pytest

from fireassay.curate.coverage import content_coverage, difficulty_feature_correlation, generator_obedience
from fireassay.generate.models import CandidateFeatures, ResolvedCandidate
from fireassay.system.corpus import Doc


def _candidate(
    candidate_id: str,
    difficulty: Literal["easy", "medium", "hard"],
    *,
    qtype: Literal["factual", "procedural", "comparative", "policy_sensitive"] = "factual",
    target_qtype: Literal["factual", "procedural", "comparative", "policy_sensitive"] | None = None,
    target_difficulty: Literal["easy", "medium", "hard"] | None = None,
    title_overlap: float = 0.0,
    quote_overlap: float = 0.0,
    question_len_tokens: int = 5,
    gold_doc_rank: int | None = None,
    source_doc_id: str = "doc1",
) -> ResolvedCandidate:
    return ResolvedCandidate(
        id=candidate_id,
        batch_id="b1",
        text="a question?",
        qtype=qtype,
        difficulty=difficulty,
        target_qtype=target_qtype if target_qtype is not None else qtype,
        target_difficulty=target_difficulty if target_difficulty is not None else difficulty,
        reference_answer="an answer",
        quote="a quote",
        source_doc_id=source_doc_id,
        char_start=0,
        char_end=1,
        features=CandidateFeatures(
            title_overlap=title_overlap,
            quote_overlap=quote_overlap,
            question_len_tokens=question_len_tokens,
            gold_doc_rank=gold_doc_rank,
        ),
        model_digest="digest",
        prompt_hash="prompt-hash",
        created_at="2026-01-01T00:00:00",
    )


# -- difficulty_feature_correlation: gold_doc_rank -------------------------


def test_gold_doc_rank_is_recorded_and_appears_in_the_correlation() -> None:
    """Rank increases with proposed difficulty (easy=findable, hard=deep) --
    a strong positive correlation, and gold_doc_rank must show up in the
    result exactly like the lexical features do."""
    candidates = [
        _candidate("c1", "easy", gold_doc_rank=1),
        _candidate("c2", "easy", gold_doc_rank=2),
        _candidate("c3", "medium", gold_doc_rank=15),
        _candidate("c4", "hard", gold_doc_rank=40),
    ]
    result = difficulty_feature_correlation(candidates)
    assert "gold_doc_rank" in result
    assert result["gold_doc_rank"] > 0.8


def test_gold_doc_rank_all_none_is_excluded_not_treated_as_zero() -> None:
    candidates = [
        _candidate("c1", "easy", gold_doc_rank=None),
        _candidate("c2", "hard", gold_doc_rank=None),
    ]
    result = difficulty_feature_correlation(candidates)
    assert "gold_doc_rank" not in result


def test_gold_doc_rank_partial_availability_does_not_suppress_other_features() -> None:
    """gold_doc_rank has data for only one candidate here (insufficient to
    correlate) -- title_overlap, available for both, must still be
    reported. Each feature is filtered independently."""
    candidates = [
        _candidate("c1", "easy", title_overlap=0.1, gold_doc_rank=1),
        _candidate("c2", "hard", title_overlap=0.9, gold_doc_rank=None),
    ]
    result = difficulty_feature_correlation(candidates)
    assert "gold_doc_rank" not in result
    assert "title_overlap" in result


def test_correlation_empty_with_fewer_than_two_candidates() -> None:
    assert difficulty_feature_correlation([_candidate("c1", "easy")]) == {}
    assert difficulty_feature_correlation([]) == {}


def test_correlation_omits_a_zero_variance_feature() -> None:
    candidates = [
        _candidate("c1", "easy", title_overlap=0.5),
        _candidate("c2", "hard", title_overlap=0.5),  # constant -> no variance
    ]
    result = difficulty_feature_correlation(candidates)
    assert "title_overlap" not in result


# -- content_coverage --------------------------------------------------


def test_content_coverage_hand_computed() -> None:
    docs = [
        Doc(doc_id="d1", title="t1", text="x"),
        Doc(doc_id="d2", title="t2", text="y"),
        Doc(doc_id="d3", title="t3", text="z"),
    ]
    candidates = [
        _candidate("c1", "easy", source_doc_id="d1"),
        _candidate("c2", "medium", source_doc_id="d1"),
    ]
    report = content_coverage(candidates, docs)
    assert report.documents_with_questions == 1
    assert report.total_documents == 3
    assert report.doc_coverage_fraction == pytest.approx(1 / 3)
    assert report.cell_occupancy[("factual", "easy")] == 1
    assert report.cell_occupancy[("factual", "medium")] == 1


def test_content_coverage_empty_docs_is_zero_not_a_crash() -> None:
    report = content_coverage([], [])
    assert report.total_documents == 0
    assert report.doc_coverage_fraction == 0.0


# -- generator_obedience -------------------------------------------------


def test_generator_obedience_hand_computed() -> None:
    candidates = [
        # obeys both qtype and difficulty
        _candidate("c1", "easy", qtype="factual", target_qtype="factual", target_difficulty="easy"),
        # obeys qtype only
        _candidate("c2", "hard", qtype="factual", target_qtype="factual", target_difficulty="easy"),
        # obeys difficulty only
        _candidate(
            "c3", "easy", qtype="procedural", target_qtype="factual", target_difficulty="easy"
        ),
        # obeys neither
        _candidate(
            "c4", "hard", qtype="procedural", target_qtype="factual", target_difficulty="easy"
        ),
    ]
    result = generator_obedience(candidates)
    assert result.n_candidates == 4
    assert result.qtype_match_rate == pytest.approx(0.5)  # c1, c2
    assert result.difficulty_match_rate == pytest.approx(0.5)  # c1, c3
    assert result.cell_match_rate == pytest.approx(0.25)  # c1 only


def test_generator_obedience_empty_is_none_not_zero() -> None:
    result = generator_obedience([])
    assert result.n_candidates == 0
    assert result.qtype_match_rate is None
    assert result.difficulty_match_rate is None
    assert result.cell_match_rate is None


def test_generator_obedience_perfect() -> None:
    candidates = [
        _candidate("c1", "easy", qtype="factual", target_qtype="factual", target_difficulty="easy"),
        _candidate(
            "c2", "hard", qtype="comparative", target_qtype="comparative", target_difficulty="hard"
        ),
    ]
    result = generator_obedience(candidates)
    assert result.qtype_match_rate == pytest.approx(1.0)
    assert result.difficulty_match_rate == pytest.approx(1.0)
    assert result.cell_match_rate == pytest.approx(1.0)
