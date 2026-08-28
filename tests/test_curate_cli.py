"""M3-SPEC.md §7: `curate next`/`curate submit` round-trip; a reject
without a reason is refused.

Candidates are seeded directly via `Store` (not through `generate`/
`filter`, which need a live model / real corpus) -- `curate next`/
`curate submit` are exercised exactly as the CLI: `typer.testing.
CliRunner`, matching `test_mutate_cli.py`'s established pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fireassay.cli import app
from fireassay.generate.models import LexicalFeatures, ResolvedCandidate
from fireassay.store.db import Store

runner = CliRunner()

_ALL_STAGES = (
    "not_a_question",
    "span_resolution",
    "degeneracy",
    "self_containment",
    "near_duplicate",
    "generic",
    "balance",
)

_RUBRIC_JSON = {
    "answerable_from_kb": "yes",
    "self_contained": True,
    "reference_answer_correct": "yes",
    "evidence_sufficient": True,
    "difficulty_agrees": True,
    "qtype_agrees": True,
}


def _invoke(*args: str) -> object:
    return runner.invoke(app, list(args))


def _seed_kept_candidate(db: Path, candidate_id: str = "cand1") -> None:
    store = Store(db)
    store.migrate()
    candidate = ResolvedCandidate(
        id=candidate_id,
        batch_id="b1",
        text="What is the refund window?",
        qtype="factual",
        difficulty="easy",
        reference_answer="14 days",
        quote="Refunds are issued within 14 days",
        source_doc_id="doc1",
        char_start=0,
        char_end=10,
        features=LexicalFeatures(title_overlap=0.1, quote_overlap=0.2, question_len_tokens=5),
        model_digest="digest",
        prompt_hash="prompt",
        created_at="2026-01-01T00:00:00",
    )
    store.put_candidate(candidate)
    for stage in _ALL_STAGES:
        store.put_filter_result(candidate_id, stage, True, None)
    store.close()


def test_curate_next_and_submit_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _seed_kept_candidate(db)

    next_result = _invoke(
        "curate", "next", "--curator", "alice", "--db", str(db),
        "--honeypot-rate", "0", "--double-review-rate", "0",
    )
    assert next_result.exit_code == 0
    payload = json.loads(next_result.stdout)
    assert payload["candidate_id"] == "cand1"
    assert payload["text"] == "What is the refund window?"
    assert payload["curator_id"] == "alice"
    # Honeypot/double-review flags must never be exposed to the curator.
    assert "is_honeypot" not in payload
    assert "is_double_review" not in payload
    assert "honeypot_expected_reason" not in payload

    verdict = {
        "candidate_id": payload["candidate_id"],
        "curator_id": "alice",
        "decision": "accept",
        "rubric": _RUBRIC_JSON,
        "duration_ms": 4200,
    }
    verdict_file = tmp_path / "verdict.json"
    verdict_file.write_text(json.dumps(verdict), encoding="utf-8")

    submit_result = _invoke("curate", "submit", str(verdict_file), "--db", str(db))
    assert submit_result.exit_code == 0
    assert "cand1" in submit_result.stdout

    store = Store(db)
    decisions = store.get_all_decisions()
    store.close()
    assert len(decisions) == 1
    assert decisions[0].candidate_id == "cand1"
    assert decisions[0].decision == "accept"
    assert decisions[0].id  # minted by the store, not the caller
    assert decisions[0].decided_at

    # The queue is now exhausted for alice: exactly one item existed, and
    # it has a decision recorded.
    done_result = _invoke("curate", "next", "--curator", "alice", "--db", str(db))
    assert done_result.exit_code == 0
    assert json.loads(done_result.stdout) == {"done": True}


def test_curate_submit_refuses_a_reject_with_no_reason(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))

    verdict = {
        "candidate_id": "cand1",
        "curator_id": "alice",
        "decision": "reject",
        # reject_reason deliberately omitted.
        "rubric": _RUBRIC_JSON,
        "duration_ms": 3000,
    }
    verdict_file = tmp_path / "verdict.json"
    verdict_file.write_text(json.dumps(verdict), encoding="utf-8")

    result = _invoke("curate", "submit", str(verdict_file), "--db", str(db))
    assert result.exit_code == 1

    store = Store(db)
    store.migrate()
    assert store.get_all_decisions() == []
    store.close()


def test_curate_submit_refuses_a_non_reject_with_a_reason(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))

    verdict = {
        "candidate_id": "cand1",
        "curator_id": "alice",
        "decision": "accept",
        "reject_reason": "TRIVIAL",  # invalid: only valid on decision == "reject"
        "rubric": _RUBRIC_JSON,
        "duration_ms": 3000,
    }
    verdict_file = tmp_path / "verdict.json"
    verdict_file.write_text(json.dumps(verdict), encoding="utf-8")

    result = _invoke("curate", "submit", str(verdict_file), "--db", str(db))
    assert result.exit_code == 1


def test_curate_report_smoke(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    _invoke("init", "--db", str(db))
    _seed_kept_candidate(db)
    _invoke("curate", "next", "--curator", "alice", "--db", str(db))

    verdict = {
        "candidate_id": "cand1",
        "curator_id": "alice",
        "decision": "accept",
        "rubric": _RUBRIC_JSON,
        "duration_ms": 4200,
    }
    verdict_file = tmp_path / "verdict.json"
    verdict_file.write_text(json.dumps(verdict), encoding="utf-8")
    _invoke("curate", "submit", str(verdict_file), "--db", str(db))

    report_result = _invoke("curate", "report", "--db", str(db))
    assert report_result.exit_code == 0
