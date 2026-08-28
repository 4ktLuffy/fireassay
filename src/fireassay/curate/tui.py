"""The M3b curation TUI (M3b-SPEC.md Part 2) — `fireassay curate tui`.

**This file is the one place in `curate/` this milestone could not
directly test.** A Textual app needs a live terminal/event loop to drive
key presses and read back what was rendered; nothing in this repo's test
harness can exercise that headlessly (M3b-SPEC.md says so explicitly).
Every decision worth testing has therefore already been pulled out into
plain functions in `curate.tui_logic` and `curate.tui_session`, both
fully covered by `test_tui_logic.py`/`test_tui_session.py` with no
`textual` import anywhere in either test file. What remains here is
wiring: render a `curate.serve.ServedItem`'s fields into fixed screen
positions, read a keypress, call a plain function, call
`curate.serve.submit_decision`, repeat. **No rule about what a keypress
means lives in this file** — see the docstring on each `action_*` method
for exactly which `tui_logic`/`tui_session` function it defers to.

Layout is fixed and never scrolls (M3b-SPEC.md: "No scrolling, ever"):
every field is truncated to a known character budget via
`tui_logic.truncate_for_display` *before* it is handed to a `Static`
widget, so nothing ever overflows its slot in the first place — this
file does not rely on any Textual scroll/overflow behaviour to make that
true.

Honeypots are rendered through the exact same `_render_item` path as any
other item: `ServedItem.view` (from `curate.serve.next_item`) never
carries `is_honeypot`/`is_double_review`/`honeypot_expected_reason` (see
that module's own docstring for why), so there is no code path here that
could even branch on them — "honeypots render identically to real items"
is true by construction, not by care taken in this file.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Static

from fireassay.curate.models import Decision, RejectReason, RubricVerdict
from fireassay.curate.serve import (
    DEFAULT_DOUBLE_REVIEW_RATE,
    DEFAULT_HONEYPOT_RATE,
    ServedItem,
    next_item,
    submit_decision,
)
from fireassay.curate.tui_logic import (
    REASON_KEYS,
    build_accept_or_edit_decision,
    build_reject_decision,
    format_elapsed_minutes,
    format_progress,
    render_truncated,
    truncate_for_display,
)
from fireassay.curate.tui_session import (
    autopilot_note,
    should_show_break_banner,
    trailing_identical_run,
)
from fireassay.models import Difficulty, QType
from fireassay.store.db import Store

#: Per-field character budgets for the fixed layout (M3b-SPEC.md's mock
#: shows a question spanning ~2 lines, a one-line reference answer, and a
#: ~2-line evidence excerpt at 80 columns -- these are sized to roughly
#: match that at a typical terminal width, not tied to any one exact
#: width, since `truncate_for_display` degrades gracefully either way.
_QUESTION_MAX_CHARS = 220
_REFERENCE_MAX_CHARS = 160
_EVIDENCE_MAX_CHARS = 220

_DIFFICULTY_KEYS: dict[str, Difficulty] = {"e": "easy", "m": "medium", "h": "hard"}
_QTYPE_KEYS: dict[str, QType] = {
    "f": "factual", "p": "procedural", "c": "comparative", "s": "policy_sensitive",
}


@dataclass
class _LastSubmission:
    """What `u`ndo needs to re-open the previous item: enough to re-render
    it and to build a corrected `Decision` for the *same* candidate/queue
    item. Deliberately carries no way to delete the earlier row —
    `curate.serve.submit_decision` only ever inserts (M3-SPEC.md §5:
    decisions are append-only), and undo's own contract (M3b-SPEC.md) is
    "writes a new decision row, never deletes one". Undo depth of 1: this
    is overwritten on every submission, so only the immediately previous
    item can ever be corrected."""

    queue_item_id: str
    candidate_id: str
    view: dict[str, object]


# -- modal screens ------------------------------------------------------


class ReasonPickerScreen(ModalScreen[str | None]):
    """`r`eject's single-keystroke reason picker — every key shown on
    screen from `tui_logic.REASON_KEYS`, so a curator never has to
    remember one (M3b-SPEC.md Part 2). `escape` cancels the reject
    entirely, dismissing with `None`."""

    def compose(self) -> ComposeResult:
        lines = "\n".join(f"  {key}  {reason}" for key, reason in REASON_KEYS.items())
        yield Static(f"reject reason — pick one, or esc to cancel\n\n{lines}")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            return
        reason = REASON_KEYS.get(event.key)
        if reason is not None:
            self.dismiss(reason)


class EvidenceModal(ModalScreen[None]):
    """`v`iew: the full, untruncated evidence quote — the rare-case escape
    hatch from the fixed layout's truncation (M3b-SPEC.md: "for the rare
    case it is needed"). Any key closes it."""

    def __init__(self, full_text: str) -> None:
        super().__init__()
        self._full_text = full_text

    def compose(self) -> ComposeResult:
        yield Static(f"EVIDENCE (full)\n\n{self._full_text}\n\n[any key to close]")

    def on_key(self, event: events.Key) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    """`?`: the full key reference, including every reject-reason key —
    "a curator must never have to remember them" (M3b-SPEC.md Part 2)."""

    def compose(self) -> ComposeResult:
        reason_lines = "\n".join(f"  {key}  {reason}" for key, reason in REASON_KEYS.items())
        text = (
            "a accept   r reject   e edit   d rubric   v evidence   "
            "u undo   ? help   q quit\n\n"
            f"reject reason keys:\n{reason_lines}\n\n[any key to close]"
        )
        yield Static(text)

    def on_key(self, event: events.Key) -> None:
        self.dismiss(None)


class EditScreen(ModalScreen[tuple[str | None, str | None] | None]):
    """`e`dit: correct the question text and/or reference answer, then
    accept (M3b-SPEC.md Part 2). Pre-filled with the candidate's current
    text/answer. Dismisses with `(edited_text, edited_answer)`, each
    `None` iff the curator left that field byte-identical to what was
    shown — `tui_logic.build_accept_or_edit_decision` treats a `None`
    field as "unchanged", never as "edited to the same text"."""

    BINDINGS = [Binding("escape", "cancel", "cancel"), Binding("ctrl+s", "submit", "submit")]

    def __init__(self, text: str, reference_answer: str) -> None:
        super().__init__()
        self._original_text = text
        self._original_answer = reference_answer

    def compose(self) -> ComposeResult:
        yield Static("edit question text (top) / reference answer (bottom) — ctrl+s submits, esc cancels")
        yield Input(value=self._original_text, id="edit_text")
        yield Input(value=self._original_answer, id="edit_answer")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        text = self.query_one("#edit_text", Input).value
        answer = self.query_one("#edit_answer", Input).value
        edited_text = text if text != self._original_text else None
        edited_answer = answer if answer != self._original_answer else None
        self.dismiss((edited_text, edited_answer))


class DrillInScreen(ModalScreen[RubricVerdict | None]):
    """`d`rill: the six rubric criteria, one at a time — the ONLY path
    that ever asks a curator all six questions (M3b-SPEC.md Part 2: "the
    six criteria are only touched when something is actually wrong").
    `escape` at any point cancels the whole drill-in with no partial
    rubric submitted (dismisses `None`), never a half-answered
    `RubricVerdict`."""

    _STEPS = (
        "answerable_from_kb",
        "self_contained",
        "reference_answer_correct",
        "evidence_sufficient",
        "difficulty_agrees",
        "qtype_agrees",
    )

    def __init__(self) -> None:
        super().__init__()
        self._step = 0
        self._answers: dict[str, object] = {}
        self._awaiting_reassignment: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="drill_prompt")

    def on_mount(self) -> None:
        self._render_step()

    def _render_step(self) -> None:
        prompt = self.query_one("#drill_prompt", Static)
        if self._awaiting_reassignment == "difficulty":
            prompt.update("difficulty disagrees — reassign: e easy / m medium / h hard")
            return
        if self._awaiting_reassignment == "qtype":
            prompt.update(
                "qtype disagrees — reassign: f factual / p procedural / c comparative / s policy_sensitive"
            )
            return
        field = self._STEPS[self._step]
        if field == "answerable_from_kb":
            prompt.update("answerable from KB? y yes / n no / p partially")
        elif field == "reference_answer_correct":
            prompt.update("reference answer correct? y yes / n no / i incomplete")
        else:
            prompt.update(f"{field.replace('_', ' ')}? y yes / n no")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            return
        if self._awaiting_reassignment == "difficulty":
            value = _DIFFICULTY_KEYS.get(event.key)
            if value is not None:
                self._answers["difficulty_reassigned"] = value
                self._awaiting_reassignment = None
                self._advance()
            return
        if self._awaiting_reassignment == "qtype":
            qtype_value = _QTYPE_KEYS.get(event.key)
            if qtype_value is not None:
                self._answers["qtype_reassigned"] = qtype_value
                self._awaiting_reassignment = None
                self._advance()
            return

        field = self._STEPS[self._step]
        if field == "answerable_from_kb":
            mapping = {"y": "yes", "n": "no", "p": "partially"}
            if event.key in mapping:
                self._answers[field] = mapping[event.key]
                self._advance()
        elif field == "reference_answer_correct":
            mapping = {"y": "yes", "n": "no", "i": "incomplete"}
            if event.key in mapping:
                self._answers[field] = mapping[event.key]
                self._advance()
        elif event.key in ("y", "n"):
            agrees = event.key == "y"
            self._answers[field] = agrees
            if field == "difficulty_agrees" and not agrees:
                self._awaiting_reassignment = "difficulty"
                self._render_step()
                return
            if field == "qtype_agrees" and not agrees:
                self._awaiting_reassignment = "qtype"
                self._render_step()
                return
            self._advance()

    def _advance(self) -> None:
        self._step += 1
        if self._step >= len(self._STEPS):
            self.dismiss(_rubric_from_answers(self._answers))
            return
        self._render_step()


def _rubric_from_answers(answers: Mapping[str, object]) -> RubricVerdict:
    """Build a `RubricVerdict` from the drill-in screen's collected answers.

    Explicit rather than `RubricVerdict(**answers)`: the answers dict is
    heterogeneous (`str` verdicts, `bool` flags, optional reassignments), so
    unpacking it defeats type checking on exactly the structure that decides
    what lands in a curated golden set.
    """
    return RubricVerdict(
        answerable_from_kb=cast(Literal["yes", "no", "partially"], answers["answerable_from_kb"]),
        self_contained=bool(answers["self_contained"]),
        reference_answer_correct=cast(
            Literal["yes", "no", "incomplete"], answers["reference_answer_correct"]
        ),
        evidence_sufficient=bool(answers["evidence_sufficient"]),
        difficulty_agrees=bool(answers["difficulty_agrees"]),
        difficulty_reassigned=cast(Difficulty | None, answers.get("difficulty_reassigned")),
        qtype_agrees=bool(answers["qtype_agrees"]),
        qtype_reassigned=cast(QType | None, answers.get("qtype_reassigned")),
    )


# -- the main app ---------------------------------------------------------


class CurateApp(App[None]):
    """Fixed 80x24-friendly layout, identical element positions on every
    item (M3b-SPEC.md Part 2). Every widget id below is written to
    exactly once per item by `_render_item`/`_render_done`; nothing here
    ever appends to a scrollable log."""

    CSS = """
    Screen { layout: vertical; }
    #status_bar { height: 1; background: $boost; content-align: left middle; }
    #question_label, #reference_label, #evidence_meta, #proposed_body {
        height: 1; text-style: bold;
    }
    #question_body { height: 3; }
    #reference_body { height: 2; }
    #evidence_body { height: 3; }
    #banner { height: 1; color: $warning; }
    """

    BINDINGS = [
        Binding("a", "accept", "accept"),
        Binding("r", "reject", "reject"),
        Binding("e", "edit", "edit"),
        Binding("d", "drill", "rubric"),
        Binding("v", "view_evidence", "evidence"),
        Binding("u", "undo", "undo"),
        Binding("question_mark", "help", "help", key_display="?"),
        Binding("q", "quit_and_save", "quit"),
    ]

    def __init__(
        self,
        store: Store,
        curator_id: str,
        *,
        honeypot_rate: float = DEFAULT_HONEYPOT_RATE,
        double_review_rate: float = DEFAULT_DOUBLE_REVIEW_RATE,
    ) -> None:
        super().__init__()
        self._store = store
        self._curator_id = curator_id
        self._honeypot_rate = honeypot_rate
        self._double_review_rate = double_review_rate
        self._session_started = time.monotonic()
        self._item_started = time.monotonic()
        self._current: ServedItem | None = None
        self._verdict_history: list[str] = []
        self._last_submission: _LastSubmission | None = None
        self._done = 0
        self._total = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id="status_bar")
        with Vertical():
            yield Static("QUESTION", id="question_label")
            yield Static(id="question_body")
            yield Static("REFERENCE ANSWER", id="reference_label")
            yield Static(id="reference_body")
            yield Static(id="evidence_meta")
            yield Static(id="evidence_body")
            yield Static(id="proposed_body")
            yield Static("", id="banner")
        yield Footer()

    def on_mount(self) -> None:
        self._recompute_progress()
        self.load_next()

    # -- state / rendering --------------------------------------------------

    def _recompute_progress(self) -> None:
        """Read once, on mount: `_done` is thereafter tracked
        incrementally by `_submit_and_advance` rather than re-queried
        after every item, since this curator is the only writer for their
        own queue during one TUI session."""
        queue_items = [qi for qi in self._store.get_all_queue_items() if qi.curator_id == self._curator_id]
        self._total = len(queue_items)
        decided_candidates = {
            d.candidate_id for d in self._store.get_all_decisions() if d.curator_id == self._curator_id
        }
        self._done = len({qi.candidate_id for qi in queue_items if qi.candidate_id in decided_candidates})

    def load_next(self) -> None:
        item = next_item(
            self._store,
            self._curator_id,
            honeypot_rate=self._honeypot_rate,
            double_review_rate=self._double_review_rate,
        )
        self._current = item
        self._item_started = time.monotonic()
        if item is None:
            self._render_done()
            return
        self._render_item(item.view)
        self._update_status_bar()

    def _elapsed_seconds(self) -> float:
        return time.monotonic() - self._session_started

    def _update_status_bar(self) -> None:
        bar = self.query_one("#status_bar", Static)
        progress = format_progress(self._done, self._total)
        elapsed = format_elapsed_minutes(self._elapsed_seconds())
        bar.update(f"{self._curator_id} · {progress} · session {elapsed}")

        notes: list[str] = []
        if should_show_break_banner(self._elapsed_seconds() / 60.0):
            notes.append("45+ minutes — consider a break")
        if self._verdict_history:
            streak = trailing_identical_run(self._verdict_history)
            note = autopilot_note(streak, self._verdict_history[-1])
            if note is not None:
                notes.append(note)
        # Neither note ever blocks input (M3b-SPEC.md Part 2) -- this is
        # a plain label update, not a modal, not a confirmation prompt.
        self.query_one("#banner", Static).update("  ".join(notes))

    def _render_item(self, view: dict[str, object]) -> None:
        question = truncate_for_display(str(view["text"]), _QUESTION_MAX_CHARS)
        reference = truncate_for_display(str(view["reference_answer"]), _REFERENCE_MAX_CHARS)
        quote = truncate_for_display(str(view["quote"]), _EVIDENCE_MAX_CHARS)

        self.query_one("#question_body", Static).update(render_truncated(question))
        self.query_one("#reference_body", Static).update(render_truncated(reference))
        self.query_one("#evidence_meta", Static).update(
            f"EVIDENCE   {view['source_doc_id']} · chars {view['char_start']}–{view['char_end']}"
        )
        self.query_one("#evidence_body", Static).update(render_truncated(quote))
        gold_rank = view["gold_doc_rank"]
        gold_rank_text = "n/a" if gold_rank is None else str(gold_rank)
        self.query_one("#proposed_body", Static).update(
            f"PROPOSED   {view['qtype']} / {view['difficulty']}   gold rank {gold_rank_text}"
        )

    def _render_done(self) -> None:
        self.query_one("#status_bar", Static).update(f"{self._curator_id} · queue complete")
        self.query_one("#question_body", Static).update("(queue exhausted — nothing left to review)")
        for widget_id in ("#reference_body", "#evidence_meta", "#evidence_body", "#proposed_body", "#banner"):
            self.query_one(widget_id, Static).update("")

    def _duration_ms(self) -> int:
        return int((time.monotonic() - self._item_started) * 1000)

    def _submit_and_advance(self, decision: Decision) -> None:
        if self._current is None:
            return
        stored = submit_decision(self._store, decision)
        self._last_submission = _LastSubmission(
            queue_item_id=self._current.queue_item_id,
            candidate_id=self._current.candidate_id,
            view=self._current.view,
        )
        self._verdict_history.append(stored.decision)
        self._done += 1
        self.load_next()

    # -- actions --------------------------------------------------------
    #
    # Every action below defers the actual decision (what the default
    # rubric is, which reason a key means, what "truncated" displays) to
    # `curate.tui_logic`/`curate.tui_session` — nothing here decides any
    # of that itself.

    def action_accept(self) -> None:
        """`a`: the one-keystroke fast path — see
        `tui_logic.build_accept_or_edit_decision`'s default rubric."""
        if self._current is None:
            return
        decision = build_accept_or_edit_decision(
            self._current.candidate_id, self._curator_id, self._duration_ms()
        )
        self._submit_and_advance(decision)

    # `@work` on every action that awaits `push_screen_wait`: Textual requires
    # `wait_for_dismiss=True` to run inside a worker, and raises `NoActiveWorker`
    # otherwise. Without it, reject/edit/rubric/evidence/help all crash on the
    # first keypress -- five of the seven interactive actions. Caught only by
    # driving the app headlessly; no logic test imports this module.
    @work
    async def action_reject(self) -> None:
        """`r` then one more keystroke — see `ReasonPickerScreen` and
        `tui_logic.build_reject_decision`."""
        if self._current is None:
            return
        picked = await self.push_screen_wait(ReasonPickerScreen())
        # the picker only ever dismisses with a RejectReason or None; the cast
        # keeps that guarantee visible to the type checker across the modal boundary
        reason = cast("RejectReason | None", picked)
        if reason is None:
            return
        decision = build_reject_decision(
            self._current.candidate_id, self._curator_id, reason, self._duration_ms()
        )
        self._submit_and_advance(decision)

    @work
    async def action_edit(self) -> None:
        """`e`: correct the text/answer, then accept — see `EditScreen`."""
        if self._current is None:
            return
        result = await self.push_screen_wait(
            EditScreen(str(self._current.view["text"]), str(self._current.view["reference_answer"]))
        )
        if result is None:
            return
        edited_text, edited_answer = result
        decision = build_accept_or_edit_decision(
            self._current.candidate_id,
            self._curator_id,
            self._duration_ms(),
            edited_text=edited_text,
            edited_answer=edited_answer,
        )
        self._submit_and_advance(decision)

    @work
    async def action_drill(self) -> None:
        """`d`: the only path that asks all six rubric questions — see
        `DrillInScreen`. A drilled-in rubric still results in an accept
        (with the curator's own answers attached, not the default) —
        a curator who wants to reject after drilling in still uses `r`."""
        if self._current is None:
            return
        rubric = await self.push_screen_wait(DrillInScreen())
        if rubric is None:
            return
        decision = build_accept_or_edit_decision(
            self._current.candidate_id, self._curator_id, self._duration_ms(), rubric=rubric
        )
        self._submit_and_advance(decision)

    @work
    async def action_view_evidence(self) -> None:
        """`v`: the full, untruncated evidence — see `EvidenceModal`."""
        if self._current is None:
            return
        await self.push_screen_wait(EvidenceModal(str(self._current.view["quote"])))

    @work
    async def action_help(self) -> None:
        """`?`: see `HelpScreen`."""
        await self.push_screen_wait(HelpScreen())

    def action_undo(self) -> None:
        """`u`: re-open the previous item for a corrected verdict.
        Undo depth of 1 (M3b-SPEC.md Part 2): re-submitting after this
        writes a **new** `decision` row via the normal action_* path above
        (`submit_decision` never updates or deletes) — this method only
        re-renders the item and restores `_current` so the next accept/
        reject/edit/drill targets it again, rather than the queue's next
        undecided item."""
        if self._last_submission is None:
            return
        self._current = ServedItem(
            queue_item_id=self._last_submission.queue_item_id,
            candidate_id=self._last_submission.candidate_id,
            curator_id=self._curator_id,
            view=self._last_submission.view,
        )
        self._render_item(self._last_submission.view)
        self._item_started = time.monotonic()
        self._last_submission = None

    def action_quit_and_save(self) -> None:
        """`q`: every decision is already persisted synchronously at
        submit time (`Store.put_decision` commits immediately) — there is
        no separate save step to perform here beyond exiting."""
        self.exit()


def run_curate_tui(
    store: Store,
    curator_id: str,
    *,
    honeypot_rate: float = DEFAULT_HONEYPOT_RATE,
    double_review_rate: float = DEFAULT_DOUBLE_REVIEW_RATE,
) -> None:
    """`cli.py`'s `curate tui` entry point — kept as its own function
    (rather than inlining `CurateApp(...).run()` at the call site) so a
    future test harness with a real terminal, or a script, can drive the
    app without going through the CLI layer."""
    CurateApp(
        store, curator_id, honeypot_rate=honeypot_rate, double_review_rate=double_review_rate
    ).run()
