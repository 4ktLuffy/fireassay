"""Headless drive of the real Textual app (M3b-SPEC.md Part 2).

`test_tui_logic.py` and `test_tui_session.py` cover the decisions, and they
deliberately do not import `curate.tui` at all -- that separation is what
keeps the logic strictly testable. But it also means the widget layer had
**zero** coverage, and that gap hid a real bug: every action awaiting
`push_screen_wait` (reject, edit, rubric, evidence, help -- five of seven)
raised `NoActiveWorker` on the first keypress, because Textual requires
`wait_for_dismiss=True` to run inside a worker. The app imported fine, the
whole suite passed, and pressing `r` crashed.

So this file exists to press the keys. It asserts behaviour a user would
notice -- a decision row appears, a modal does not crash -- not rendering.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fireassay.curate.funnel import FUNNEL_STAGE_ORDER
from fireassay.curate.tui import CurateApp
from fireassay.generate.models import CandidateFeatures, ResolvedCandidate
from fireassay.store import Store


def _seed(db: Path, n: int) -> None:
    store = Store(db)
    store.migrate()
    for i in range(n):
        store.put_candidate(
            ResolvedCandidate(
                id=f"cand{i}",
                batch_id="b1",
                text=f"What is rule {i}?",
                qtype="factual",
                difficulty="easy",
                target_qtype="factual",
                target_difficulty="easy",
                reference_answer=f"Answer {i}",
                quote=f"Rule {i} says so",
                source_doc_id=f"doc{i}",
                chunk_id=f"doc{i}#0000",
                char_start=0,
                char_end=10,
                features=CandidateFeatures(
                    title_overlap=0.1, quote_overlap=0.2, question_len_tokens=5
                ),
                model_digest="digest",
                prompt_hash=f"prompt{i}",
                created_at="2026-01-01T00:00:00",
            )
        )
        for stage in FUNNEL_STAGE_ORDER:
            store.put_filter_result(f"cand{i}", stage, True, None)
    store.close()


def _decisions(store: Store) -> int:
    row = store._conn.execute("select count(*) from decision").fetchone()
    return int(row[0])


def test_fast_path_accept_records_one_decision(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _seed(db, 3)

    async def drive() -> tuple[int, int]:
        store = Store(db)
        store.migrate()
        app = CurateApp(store=store, curator_id="c")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            before = _decisions(store)
            await pilot.press("a")
            await pilot.pause()
            await pilot.pause()
            return before, _decisions(store)

    before, after = asyncio.run(drive())
    assert after == before + 1


def test_every_modal_action_runs_without_crashing(tmp_path: Path) -> None:
    """The regression test for NoActiveWorker. Each of these keys opens a
    modal; before `@work` was applied, every one of them raised."""
    db = tmp_path / "t.db"
    _seed(db, 4)

    async def drive() -> tuple[int, str | None]:
        store = Store(db)
        store.migrate()
        app = CurateApp(store=store, curator_id="c")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("v")  # evidence modal
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("?")  # help
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("r")  # reject -> reason picker
            await pilot.pause()
            await pilot.press("w")  # WRONG_REFERENCE
            await pilot.pause()
            await pilot.pause()
            row = store._conn.execute(
                "select reject_reason from decision order by rowid desc limit 1"
            ).fetchone()
            return _decisions(store), (row[0] if row else None)

    count, reason = asyncio.run(drive())
    assert count == 1, "the reject should have recorded exactly one decision"
    assert reason == "WRONG_REFERENCE"
