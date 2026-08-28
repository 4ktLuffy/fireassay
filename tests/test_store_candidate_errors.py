"""`Store._row_to_candidate` must raise a clear, named error for a
pre-migration-0004 candidate row (`target_qtype`/`target_difficulty ==
''`), never a raw pydantic `literal_error` traceback. The migration's own
comment already explains why `''` is correct-but-unusable; this is about
the *message* a reader gets when they hit it, not the schema decision --
the same "never let a raw validation error be the diagnosis" rule already
applied to `llm/ollama.py`'s API error handling.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fireassay.generate.models import CandidateFeatures, ResolvedCandidate
from fireassay.store.db import PreStratificationCandidateError, Store


def _insert_pre_stratification_row(db_path: Path, candidate_id: str) -> None:
    """Simulate a `candidate` row written before migration 0004: raw SQL,
    bypassing `Store.put_candidate` entirely, since `ResolvedCandidate`
    itself refuses to construct with an empty `target_qtype`/
    `target_difficulty` -- exactly the state this test exists to cover,
    and the reason a raw SQL insert is the only way to reach it."""
    conn = sqlite3.connect(str(db_path))
    try:
        features_json = json.dumps(
            {"title_overlap": 0.0, "quote_overlap": 0.0, "question_len_tokens": 1, "gold_doc_rank": None}
        )
        conn.execute(
            "INSERT INTO candidate (id, batch_id, text, qtype, difficulty, target_qtype, "
            "target_difficulty, reference_answer, quote, source_doc_id, char_start, char_end, "
            "features_json, model_digest, prompt_hash, created_at) "
            "VALUES (?, 'b1', 'a question?', 'factual', 'easy', '', '', 'ans', 'a quote', 'doc1', "
            "0, 1, ?, 'digest', 'prompt-hash', '2026-01-01T00:00:00')",
            (candidate_id, features_json),
        )
        conn.commit()
    finally:
        conn.close()


def test_pre_stratification_row_raises_a_short_named_error(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(db_path)
    store.migrate()
    _insert_pre_stratification_row(db_path, "cand-old-1")

    with pytest.raises(PreStratificationCandidateError) as exc_info:
        store.get_candidate("cand-old-1")

    message = str(exc_info.value)
    assert "cand-old-1" in message
    assert "migration 0004" in message
    assert len(message) < 400, "diagnosis must stay short, not traceback-shaped"
    # A raw pydantic ValidationError for this exact failure would contain
    # these phrases; neither must leak through the named error's message.
    assert "Input should be" not in message
    assert "For further information visit" not in message

    store.close()


def test_pre_stratification_row_raises_via_get_candidates_by_batch(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(db_path)
    store.migrate()
    _insert_pre_stratification_row(db_path, "cand-old-2")

    with pytest.raises(PreStratificationCandidateError):
        store.get_candidates_by_batch("b1")

    store.close()


def test_pre_stratification_row_raises_via_get_all_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = Store(db_path)
    store.migrate()
    _insert_pre_stratification_row(db_path, "cand-old-3")

    with pytest.raises(PreStratificationCandidateError):
        store.get_all_candidates()

    store.close()


def test_ordinary_candidate_row_is_unaffected(tmp_path: Path) -> None:
    """A normal, post-migration candidate (real target cell) must read
    back fine -- this check must fire only on the pre-stratification
    state, never on a well-formed row."""
    db_path = tmp_path / "test.db"
    store = Store(db_path)
    store.migrate()
    store.put_candidate(
        ResolvedCandidate(
            id="cand-new-1",
            batch_id="b1",
            text="a question?",
            qtype="factual",
            difficulty="easy",
            target_qtype="factual",
            target_difficulty="easy",
            reference_answer="an answer",
            quote="a quote",
            source_doc_id="doc1",
            char_start=0,
            char_end=1,
            features=CandidateFeatures(title_overlap=0.0, quote_overlap=0.0, question_len_tokens=1),
            model_digest="digest",
            prompt_hash="prompt-hash",
            created_at="2026-01-01T00:00:00",
        )
    )

    candidate = store.get_candidate("cand-new-1")
    assert candidate.target_qtype == "factual"

    store.close()
