"""Read/write the plain-text contract for standalone use of `items.core`
-- no fireassay dependency beyond `items.core` itself.

**Response matrix, CSV** -- one row per (item, system) response, header
required exactly as shown:

    item_id,system_id,correct
    q1,bm25-k3,1
    q1,bm25-k5,0
    q2,bm25-k3,1
    q2,bm25-k5,1

`correct` accepts `1`/`0` or `true`/`false` (case-insensitive). Every
`item_id` must have exactly one row for every `system_id` that appears
anywhere in the file -- a missing (item_id, system_id) pair is rejected
with a clear error rather than silently treated as `False`, since "not
measured" and "measured incorrect" are different facts (the same
omit-don't-zero rule `items.core`/fireassay's scorers apply throughout).
System order in the resulting `ItemResponses.responses` is first-seen
order across the file, not re-sorted -- a human-authored CSV's column
order is meaningful.

**Response matrix, JSONL** -- the line-oriented equivalent, one JSON
object per line, same three fields, `correct` a JSON boolean:

    {"item_id": "q1", "system_id": "bm25-k3", "correct": true}
    {"item_id": "q1", "system_id": "bm25-k5", "correct": false}

**Metadata, JSONL** -- optional, one `ItemMeta`-shaped JSON object per
line, keyed on `item_id`; every field beyond `item_id` is optional:

    {"item_id": "q1", "question": "...", "reference_answer": "...",
     "evidence_quote": "...", "source_doc_id": "doc1",
     "claimed_difficulty": "easy", "claimed_qtype": "factual"}

These three shapes are the public contract for standalone use: any tool
that can emit a CSV or JSONL in this shape can be analysed by
`fireassay items analyse --matrix ...` without ever touching a fireassay
store.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from fireassay.items.core import ItemMeta, ItemResponses

_TRUE_VALUES = frozenset({"1", "true", "True", "TRUE"})
_FALSE_VALUES = frozenset({"0", "false", "False", "FALSE"})
_REQUIRED_CSV_FIELDS = frozenset({"item_id", "system_id", "correct"})


def _parse_correct(raw: str, *, source: str, line_no: int) -> bool:
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(
        f"tabular: {source}: line {line_no}: 'correct' must be one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}"
    )


def _rows_to_responses(rows: Sequence[tuple[str, str, bool]], *, source: str) -> list[ItemResponses]:
    """`rows` is `(item_id, system_id, correct)` in file order."""
    system_order: list[str] = []
    seen_systems: set[str] = set()
    item_order: list[str] = []
    by_item: dict[str, dict[str, bool]] = {}

    for item_id, system_id, correct in rows:
        if system_id not in seen_systems:
            seen_systems.add(system_id)
            system_order.append(system_id)
        if item_id not in by_item:
            by_item[item_id] = {}
            item_order.append(item_id)
        if system_id in by_item[item_id]:
            raise ValueError(
                f"tabular: {source}: duplicate row for item_id={item_id!r}, system_id={system_id!r}"
            )
        by_item[item_id][system_id] = correct

    responses: list[ItemResponses] = []
    for item_id in item_order:
        row = by_item[item_id]
        missing = [s for s in system_order if s not in row]
        if missing:
            raise ValueError(
                f"tabular: {source}: item_id={item_id!r} is missing a row for system_id(s) "
                f"{missing} -- every item must report a response for every system_id seen "
                "anywhere in the file"
            )
        responses.append(ItemResponses(item_id=item_id, responses=tuple(row[s] for s in system_order)))
    return responses


def load_responses_csv(path: Path | str) -> list[ItemResponses]:
    """Load a response-matrix CSV (see module docstring). Raises
    `ValueError` with the offending line number for a malformed header,
    row, or ragged item."""
    source = str(path)
    rows: list[tuple[str, str, bool]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or set(reader.fieldnames) != _REQUIRED_CSV_FIELDS:
            raise ValueError(
                f"tabular: {source}: header must be exactly 'item_id,system_id,correct', "
                f"got {reader.fieldnames!r}"
            )
        for line_no, raw_row in enumerate(reader, start=2):  # header occupies line 1
            item_id = raw_row["item_id"]
            system_id = raw_row["system_id"]
            correct_raw = raw_row["correct"]
            if not item_id or not system_id or correct_raw is None:
                raise ValueError(f"tabular: {source}: line {line_no}: missing item_id/system_id/correct")
            correct = _parse_correct(correct_raw, source=source, line_no=line_no)
            rows.append((item_id, system_id, correct))
    if not rows:
        raise ValueError(f"tabular: {source}: no data rows")
    return _rows_to_responses(rows, source=source)


def load_responses_jsonl(path: Path | str) -> list[ItemResponses]:
    """Load a response-matrix JSONL (see module docstring). Raises
    `ValueError` with the offending line number for invalid JSON, a
    missing/mistyped field, or a ragged item."""
    source = str(path)
    rows: list[tuple[str, str, bool]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"tabular: {source}: line {line_no}: invalid JSON: {exc}") from exc
            try:
                item_id = obj["item_id"]
                system_id = obj["system_id"]
                correct = obj["correct"]
            except KeyError as exc:
                raise ValueError(f"tabular: {source}: line {line_no}: missing field {exc}") from exc
            if not isinstance(item_id, str) or not isinstance(system_id, str):
                raise ValueError(f"tabular: {source}: line {line_no}: item_id and system_id must be strings")
            if not isinstance(correct, bool):
                raise ValueError(f"tabular: {source}: line {line_no}: 'correct' must be a JSON boolean")
            rows.append((item_id, system_id, correct))
    if not rows:
        raise ValueError(f"tabular: {source}: no data rows")
    return _rows_to_responses(rows, source=source)


def write_responses_csv(
    responses: Sequence[ItemResponses], system_ids: Sequence[str], path: Path | str
) -> None:
    """Write `responses` out as a response-matrix CSV -- the inverse of
    `load_responses_csv`. `system_ids` must be supplied explicitly (rather
    than re-derived) since `ItemResponses.responses` is a bare positional
    tuple with no embedded system labels; `system_ids[i]` must name the
    system `responses[j].responses[i]` refers to, for every item `j`."""
    for r in responses:
        if len(r.responses) != len(system_ids):
            raise ValueError(
                f"write_responses_csv: item {r.item_id!r} has {len(r.responses)} response(s), "
                f"expected {len(system_ids)} to match system_ids"
            )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "system_id", "correct"])
        for r in responses:
            for system_id, correct in zip(system_ids, r.responses, strict=True):
                writer.writerow([r.item_id, system_id, 1 if correct else 0])


def load_meta_jsonl(path: Path | str) -> dict[str, ItemMeta]:
    """Load an optional metadata JSONL (see module docstring), keyed on
    `item_id`. Raises `ValueError` for invalid JSON, a field that fails
    `ItemMeta` validation, or a duplicate `item_id` (silently letting a
    later row overwrite an earlier one would hide a data-authoring
    mistake)."""
    source = str(path)
    out: dict[str, ItemMeta] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"tabular: {source}: line {line_no}: invalid JSON: {exc}") from exc
            try:
                meta = ItemMeta(**obj)
            except ValidationError as exc:
                raise ValueError(f"tabular: {source}: line {line_no}: invalid ItemMeta: {exc}") from exc
            if meta.item_id in out:
                raise ValueError(f"tabular: {source}: duplicate item_id {meta.item_id!r} at line {line_no}")
            out[meta.item_id] = meta
    return out
