from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fireassay.models import Question, Score, SystemOutput
from fireassay.score.invariants import InvariantViolation
from fireassay.store.db import AdmissibilityAlreadySetError, RunSealedError, Store, SuiteExistsError


def _question(text: str = "q", **kwargs: object) -> Question:
    defaults: dict[str, object] = {"qtype": "factual", "difficulty": "easy", "provenance": "synthetic"}
    defaults.update(kwargs)
    return Question(text=text, **defaults)  # type: ignore[arg-type]


def test_migrate_is_idempotent(store: Store) -> None:
    store.migrate()
    store.migrate()
    assert store.put_questions([_question()]) == 1


def test_put_questions_idempotent_on_id(store: Store) -> None:
    q = _question()
    assert store.put_questions([q]) == 1
    assert store.put_questions([q]) == 0


def test_freeze_suite_reuses_identical_suite(store: Store) -> None:
    q = _question()
    store.put_questions([q])
    first = store.freeze_suite("kb", "1.0.0", [q.id])
    second = store.freeze_suite("kb", "1.0.0", [q.id])
    assert first.id == second.id
    assert first.suite_hash == second.suite_hash
    assert first.frozen_at == second.frozen_at


def test_freeze_suite_conflicting_content_raises(store: Store) -> None:
    q1 = _question(text="q1")
    q2 = _question(text="q2")
    store.put_questions([q1, q2])
    store.freeze_suite("kb", "1.0.0", [q1.id])
    with pytest.raises(SuiteExistsError):
        store.freeze_suite("kb", "1.0.0", [q1.id, q2.id])


def test_put_config_idempotent_on_hash(store: Store) -> None:
    a = store.put_config({"top_k": 5}, label="first")
    b = store.put_config({"top_k": 5}, label="second")
    assert a.id == b.id
    assert a.label == "first"  # first label wins


def _start_open_run(store: Store) -> tuple[str, str]:
    q = _question()
    store.put_questions([q])
    suite = store.freeze_suite("kb", "1.0.0", [q.id])
    config = store.put_config({"top_k": 5})
    run = store.start_run(suite, config, {"python": "3.12", "fireassay": "0.1.0", "affects_results": {}})
    return run.id, q.id


def test_sealed_run_rejects_writes_in_python(store: Store) -> None:
    run_id, question_id = _start_open_run(store)
    output = SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0})
    store.put_result(run_id, question_id, output, 0.0)
    store.finish_run(run_id, "complete")

    with pytest.raises(RunSealedError):
        store.put_result(run_id, question_id, output, 0.0)
    with pytest.raises(RunSealedError):
        store.put_scores(run_id, question_id, [Score(metric="m", value=1.0, scorer="s@1.0.0")])


def test_sealed_run_rejects_writes_via_sql_trigger(store: Store, tmp_path: Path) -> None:
    run_id, question_id = _start_open_run(store)
    store.finish_run(run_id, "complete")

    # Bypass the Store class entirely: a raw connection on the same file
    # must still be blocked by the trigger.
    raw = sqlite3.connect(str(tmp_path / "test.db"))
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            "INSERT INTO result (run_id, question_id, answer, abstained, retrieved_json, "
            "latency_json, tokens_in, tokens_out, cost_usd) "
            "VALUES (?, ?, NULL, 0, '[]', '{}', 0, 0, 0.0)",
            (run_id, question_id),
        )
    raw.close()


def test_finish_run_twice_raises(store: Store) -> None:
    run_id, _question_id = _start_open_run(store)
    store.finish_run(run_id, "complete")
    with pytest.raises(RunSealedError):
        store.finish_run(run_id, "complete")


def test_iter_questions_returns_evidence_spans(store: Store) -> None:
    from fireassay.models import EvidenceSpan

    span = EvidenceSpan(doc_id="d", chunk_id="d#0000", char_start=0, char_end=5, quote="hello")
    q = _question(evidence_spans=(span,))
    store.put_questions([q])
    suite = store.freeze_suite("kb", "1.0.0", [q.id])
    loaded = list(store.iter_questions(suite.id))
    assert len(loaded) == 1
    assert loaded[0].evidence_spans[0].key() == span.key()


def test_get_run_reports_result_count(store: Store) -> None:
    run_id, question_id = _start_open_run(store)
    run_before = store.get_run(run_id)
    assert run_before.result_count == 0
    store.put_result(
        run_id, question_id, SystemOutput(answer=None, abstained=True, latency_ms={"total": 1.0}), 0.0
    )
    run_after = store.get_run(run_id)
    assert run_after.result_count == 1


def test_run_admissible_defaults_true(store: Store) -> None:
    run_id, _question_id = _start_open_run(store)
    store.finish_run(run_id, "complete")
    run = store.get_run(run_id)
    assert run.admissible is True
    assert store.get_run_invariant_violations(run_id) == []


def test_run_admissible_false_with_recorded_violations(store: Store) -> None:
    run_id, _question_id = _start_open_run(store)
    violations = [InvariantViolation(rule="range_0_1", question_id="q1", detail="bad value")]
    store.finish_run(run_id, "complete", admissible=False, invariant_violations=violations)

    run = store.get_run(run_id)
    assert run.admissible is False

    stored = store.get_run_invariant_violations(run_id)
    assert len(stored) == 1
    assert stored[0].rule == "range_0_1"
    assert stored[0].question_id == "q1"


def test_set_admissibility_is_idempotent_on_identical_verdict(store: Store) -> None:
    """Admissibility is a derived verdict, not run data: recomputing the
    identical verdict twice (e.g. a second `controls run` against a
    matrix whose run was reused by `run_matrix`) must be a silent no-op,
    not an error -- otherwise the command is unusable in normal
    iteration."""
    run_id, _question_id = _start_open_run(store)
    store.finish_run(run_id, "complete")

    store.set_admissibility(run_id, True, {"admissible": True, "detail": "admissible"})
    store.set_admissibility(run_id, True, {"admissible": True, "detail": "admissible"})  # no-op, no raise

    run = store.get_run(run_id)
    assert run.admissible is True
    assert run.admissibility_json == {"admissible": True, "detail": "admissible"}


def test_set_admissibility_raises_on_conflicting_verdict(store: Store) -> None:
    """A *different* verdict landing on a run that already has one must
    raise -- a run's recorded admissibility must not silently change
    underneath a reader who already looked at it."""
    run_id, _question_id = _start_open_run(store)
    store.finish_run(run_id, "complete")

    store.set_admissibility(run_id, True, {"admissible": True, "detail": "admissible"})
    with pytest.raises(AdmissibilityAlreadySetError):
        store.set_admissibility(run_id, False, {"admissible": False, "detail": "failed control(s): x"})

    # The original verdict must survive the rejected conflicting write.
    run = store.get_run(run_id)
    assert run.admissible is True
    assert run.admissibility_json == {"admissible": True, "detail": "admissible"}
