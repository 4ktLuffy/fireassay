"""M3-SPEC.md §7 (as revised): each filter stage fires on its own case and
stays silent on a clean one; the near-duplicate Jaccard threshold
boundary; the spike control question is rejected as UNRETRIEVABLE; a
question whose gold document ranks 2 is kept; the filter needs a corpus
and fails clearly without one.

`TOO_GENERIC` (per-word document frequency) is gone, not adapted: measured
on the real 1,181-document corpus it had a ~50% false-positive rate,
rejecting specific, well-formed questions because ordinary words are
common on a topically narrow corpus. `UNRETRIEVABLE` (BM25 answerability)
replaces it -- see `filter/stages.py`'s docstring for the full case.
"""

from __future__ import annotations

import pytest

from fireassay.filter.config import FilterConfig
from fireassay.filter.pipeline import run_filter
from fireassay.filter.stages import (
    check_degeneracy,
    check_near_duplicate,
    check_self_containment,
    check_unretrievable,
)
from fireassay.generate.models import CandidateFeatures, ResolvedCandidate
from fireassay.models import Question
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, Doc

CONFIG = FilterConfig()


def _candidate(
    candidate_id: str,
    text: str,
    *,
    qtype: str = "factual",
    difficulty: str = "easy",
    source_doc_id: str = "doc1",
    created_at: str = "2026-01-01T00:00:00",
) -> ResolvedCandidate:
    return ResolvedCandidate(
        id=candidate_id,
        batch_id="b1",
        text=text,
        qtype=qtype,  # type: ignore[arg-type]
        difficulty=difficulty,  # type: ignore[arg-type]
        target_qtype=qtype,  # type: ignore[arg-type]
        target_difficulty=difficulty,  # type: ignore[arg-type]
        reference_answer="an answer",
        quote="a quote",
        source_doc_id=source_doc_id,
        char_start=0,
        char_end=8,
        features=CandidateFeatures(
            title_overlap=0.0, quote_overlap=0.0, question_len_tokens=len(text.split())
        ),
        model_digest="digest",
        prompt_hash="prompt-hash",
        created_at=created_at,
    )


# -- degeneracy ---------------------------------------------------------


def test_degeneracy_fires_on_too_short_question() -> None:
    ok, reason = check_degeneracy(_candidate("c1", "What?"), CONFIG)
    assert not ok
    assert reason == "DEGENERATE"


def test_degeneracy_fires_on_no_content_word() -> None:
    ok, reason = check_degeneracy(_candidate("c1", "What is the this that of it?"), CONFIG)
    assert not ok
    assert reason == "DEGENERATE"


def test_degeneracy_silent_on_clean_question() -> None:
    ok, reason = check_degeneracy(
        _candidate("c1", "How do I reset my password using the settings page?"), CONFIG
    )
    assert ok
    assert reason is None


# -- self-containment -----------------------------------------------------


def test_self_containment_fires_on_dangling_referent() -> None:
    ok, reason = check_self_containment(
        _candidate("c1", "What is this used for in the account settings page?")
    )
    assert not ok
    assert reason == "NOT_SELF_CONTAINED"


def test_self_containment_fires_on_the_above() -> None:
    ok, reason = check_self_containment(_candidate("c1", "What does the above section explain?"))
    assert not ok
    assert reason == "NOT_SELF_CONTAINED"


def test_self_containment_silent_on_clean_question() -> None:
    ok, reason = check_self_containment(
        _candidate("c1", "What is two factor authentication used for?")
    )
    assert ok
    assert reason is None


# -- near-duplicate: Jaccard threshold boundary --------------------------


def test_near_duplicate_at_exact_threshold_is_rejected() -> None:
    words = [f"tok{i}" for i in range(20)]
    kept_token_sets = [frozenset(words)]
    candidate = _candidate("c1", " ".join(words[:17]))  # 17-token subset -> jaccard = 17/20 = 0.85
    ok, reason = check_near_duplicate(candidate, kept_token_sets, threshold=0.85)
    assert not ok
    assert reason == "NEAR_DUPLICATE"


def test_near_duplicate_just_below_threshold_is_silent() -> None:
    words = [f"tok{i}" for i in range(20)]
    kept_token_sets = [frozenset(words)]
    candidate = _candidate("c1", " ".join(words[:16]))  # 16/20 = 0.80 < 0.85
    ok, reason = check_near_duplicate(candidate, kept_token_sets, threshold=0.85)
    assert ok
    assert reason is None


def test_near_duplicate_silent_with_no_kept_candidates_yet() -> None:
    ok, reason = check_near_duplicate(_candidate("c1", "anything at all here"), [], threshold=0.85)
    assert ok
    assert reason is None


# -- unretrievable: BM25 answerability -------------------------------------


def test_check_unretrievable_rejects_the_spike_control_question() -> None:
    """*"What does the guidance say about revenue and customs?"* -- the
    spike's own failure case. `d1` (its claimed source) shares zero query
    terms; three decoys do share them, so `d1` cannot make the top 2."""
    chunks = [
        Chunk(doc_id="decoy1", chunk_id="decoy1#0000", char_start=0, char_end=10,
              text="guidance revenue customs apply here for businesses today"),
        Chunk(doc_id="decoy2", chunk_id="decoy2#0000", char_start=0, char_end=10,
              text="guidance revenue customs information for importers"),
        Chunk(doc_id="decoy3", chunk_id="decoy3#0000", char_start=0, char_end=10,
              text="guidance revenue customs rules for exporters"),
        Chunk(doc_id="d1", chunk_id="d1#0000", char_start=0, char_end=10,
              text="renew your passport application form plus fees payable online now"),
    ]
    retriever = BM25System(chunks, top_k=2)
    candidate = _candidate(
        "c1", "What does the guidance say about revenue and customs?", source_doc_id="d1"
    )

    ok, reason = check_unretrievable(candidate, retriever, top_n=2)

    assert not ok
    assert reason == "UNRETRIEVABLE"


def test_check_unretrievable_keeps_a_gold_document_that_is_not_ranked_first() -> None:
    """A question is not required to retrieve its own source at rank 1 to
    survive -- only somewhere in the top `top_n`. The decoy here (the
    query text repeated three times) is constructed to outscore the true
    source under BM25's own term-frequency weighting, which the test
    verifies directly rather than assuming."""
    query_text = "What are the two ways to sign in to HMRC online services?"
    chunks = [
        Chunk(
            doc_id="decoy", chunk_id="decoy#0000", char_start=0, char_end=10,
            text=" ".join([query_text] * 3),
        ),
        Chunk(doc_id="d1", chunk_id="d1#0000", char_start=0, char_end=10,
              text="There are two ways to sign in to HMRC online services."),
        Chunk(doc_id="filler", chunk_id="filler#0000", char_start=0, char_end=10,
              text="Completely unrelated content about gardening and cooking recipes today."),
    ]
    retriever = BM25System(chunks, top_k=5)
    candidate = _candidate("c1", query_text, source_doc_id="d1")

    # The premise this test relies on: the true source is NOT rank 1.
    probe = retriever.answer(
        Question(text=candidate.text, qtype="factual", difficulty="easy", provenance="synthetic")
    )
    d1_rank = next(rc.rank for rc in probe.retrieved if rc.doc_id == "d1")
    assert d1_rank > 1

    ok, reason = check_unretrievable(candidate, retriever, top_n=5)

    assert ok
    assert reason is None


def test_check_unretrievable_rejects_when_source_doc_is_absent_from_the_index() -> None:
    """A candidate whose `source_doc_id` is not in the retriever's corpus
    at all (a stale/mismatched corpus) is UNRETRIEVABLE, not a crash."""
    chunks = [
        Chunk(
            doc_id="other", chunk_id="other#0000", char_start=0, char_end=10, text="something else entirely"
        )
    ]
    retriever = BM25System(chunks, top_k=5)
    candidate = _candidate("c1", "What are the opening hours?", source_doc_id="missing-doc")

    ok, reason = check_unretrievable(candidate, retriever, top_n=5)

    assert not ok
    assert reason == "UNRETRIEVABLE"


# -- full pipeline: order + balance ----------------------------------------


def test_run_filter_keeps_earliest_and_enforces_cell_cap() -> None:
    docs = [
        Doc(
            doc_id="d1",
            title="Password Reset",
            text="To reset your password, click Forgot Password on the sign in page.",
        )
    ]
    # unretrievable_top_n set far above the corpus's own chunk count so
    # the new stage cannot interfere with this test's actual subject
    # (balance/ordering) -- mirrors near_dup_jaccard_threshold=1.1 below,
    # which disables near-duplicate the same way.
    config = FilterConfig(cell_cap=1, unretrievable_top_n=10_000, near_dup_jaccard_threshold=1.1)
    earliest = _candidate(
        "c1", "How do I reset my forgotten password on the sign in page?", source_doc_id="d1",
        created_at="2026-01-01T00:00:00",
    )
    later = _candidate(
        "c2", "What steps do I follow to change my account password today?", source_doc_id="d1",
        created_at="2026-01-01T00:00:01",
    )
    # Handed in reverse order deliberately -- run_filter must sort by
    # (created_at, id), not trust caller order, for "keeping the earliest"
    # to mean anything.
    result = run_filter([later, earliest], docs, config)

    assert [c.id for c in result.kept] == ["c1"]
    reasons = {(r.candidate_id, r.stage): r.reason for r in result.stage_results}
    assert reasons[("c1", "balance")] is None
    assert reasons[("c2", "balance")] == "CELL_FULL"


def test_run_filter_stops_at_first_rejection_no_later_stage_rows() -> None:
    docs = [Doc(doc_id="d1", title="t", text="irrelevant corpus text")]
    config = FilterConfig()
    rejected = _candidate("c1", "What?")  # DEGENERATE: too few tokens
    result = run_filter([rejected], docs, config)

    assert result.kept == ()
    stages_seen = [r.stage for r in result.stage_results if r.candidate_id == "c1"]
    assert stages_seen == ["degeneracy"]


def test_run_filter_raises_clearly_with_no_corpus() -> None:
    """The `unretrievable` stage builds a BM25 index over `docs`; an empty
    corpus must fail loudly, not silently reject every candidate against
    a degenerate empty index (M2-SPEC.md §1's rule, applied to a corpus
    precondition instead of a control)."""
    candidate = _candidate("c1", "How do I reset my forgotten password on the sign in page?")
    with pytest.raises(ValueError, match="non-empty corpus"):
        run_filter([candidate], [], FilterConfig())
