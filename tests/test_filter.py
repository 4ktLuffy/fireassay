"""M3-SPEC.md §7: each filter stage fires on its own case and stays silent
on a clean one; the near-duplicate Jaccard threshold boundary; TOO_GENERIC
catches a spike-style generic question."""

from __future__ import annotations

from fireassay.filter.config import FilterConfig
from fireassay.filter.pipeline import run_filter
from fireassay.filter.stages import (
    check_degeneracy,
    check_generic,
    check_near_duplicate,
    check_self_containment,
)
from fireassay.generate.models import LexicalFeatures, ResolvedCandidate
from fireassay.system.corpus import Doc

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
        reference_answer="an answer",
        quote="a quote",
        source_doc_id=source_doc_id,
        char_start=0,
        char_end=8,
        features=LexicalFeatures(title_overlap=0.0, quote_overlap=0.0, question_len_tokens=len(text.split())),
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


# -- generic: spike-style question ----------------------------------------


def test_generic_fires_when_every_content_word_is_common() -> None:
    """The spike's failure mode: "What does the guidance say about revenue
    and customs?" matched thousands of gov.uk pages."""
    doc_freq = {"guidance": 19, "revenue": 19, "customs": 19}
    n_docs = 20
    candidate = _candidate("c1", "What does the guidance say about revenue and customs?")
    ok, reason = check_generic(candidate, doc_freq, n_docs, generic_df_pct=0.05)
    assert not ok
    assert reason == "TOO_GENERIC"


def test_generic_silent_when_one_content_word_is_rare() -> None:
    doc_freq = {"guidance": 19, "revenue": 19, "customs": 1}
    n_docs = 20
    candidate = _candidate("c1", "What does the guidance say about revenue and customs?")
    ok, reason = check_generic(candidate, doc_freq, n_docs, generic_df_pct=0.05)
    assert ok
    assert reason is None


def test_generic_fires_when_the_only_non_common_word_is_absent_from_the_corpus() -> None:
    """A word absent from the corpus (df == 0) discriminates nothing -- it
    must not count as "rare and therefore specific" and rescue a question
    from TOO_GENERIC. "say" has no entry at all here (the spike's own
    example: filler the generator introduced, not corpus-attested
    evidence of specificity), and every other content word is common --
    this must still fire.
    """
    doc_freq = {"guidance": 19, "revenue": 19, "customs": 19}  # "say" deliberately absent
    n_docs = 20
    candidate = _candidate("c1", "What does the guidance say about revenue and customs?")
    ok, reason = check_generic(candidate, doc_freq, n_docs, generic_df_pct=0.05)
    assert not ok
    assert reason == "TOO_GENERIC"


def test_generic_silent_when_every_content_word_is_absent_from_the_corpus() -> None:
    """Nothing left to judge genericity against -- this is an
    answerability problem (a term with no corpus support at all), not a
    genericity verdict for this stage to make."""
    doc_freq: dict[str, int] = {}
    n_docs = 20
    candidate = _candidate("c1", "What does the guidance say about revenue and customs?")
    ok, reason = check_generic(candidate, doc_freq, n_docs, generic_df_pct=0.05)
    assert ok
    assert reason is None


# -- full pipeline: order + balance ----------------------------------------


def test_run_filter_keeps_earliest_and_enforces_cell_cap() -> None:
    docs = [
        Doc(
            doc_id="d1",
            title="Password Reset",
            text="To reset your password, click Forgot Password on the sign in page.",
        )
    ]
    config = FilterConfig(cell_cap=1, generic_df_pct=1.1, near_dup_jaccard_threshold=1.1)
    earliest = _candidate(
        "c1", "How do I reset my forgotten password on the sign in page?", created_at="2026-01-01T00:00:00"
    )
    later = _candidate(
        "c2", "What steps do I follow to change my account password today?", created_at="2026-01-01T00:00:01"
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
