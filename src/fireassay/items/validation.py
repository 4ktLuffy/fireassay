"""The durable record of a shipped detector's measured validation status --
what closes Goal 2, criteria 3 and 4
(`~/ObitosBrain/notes/2026-08-28-fireassay-eval-integrity.md`): *"every
shipped detector carries measured precision AND recall, with CIs -- the
report prints them"* and *"any detector that cannot be measured is
removed, or labelled unvalidated in its own output"*.

**Store-free**, same discipline as `items.core`/`items.calibration`:
nothing in this module imports `fireassay.store` or `fireassay.llm`, or
anything that transitively does (`test_items_no_store_import.py` enforces
this for this module too). Imports only `items.core` (for `Estimate`)
plus pydantic/stdlib.

## `verdict` is derived, never asserted

The entire point of `DetectorValidation.verdict` existing as a field a
caller can *pass* is undermined the moment it can be trusted as written:
a caller could hand-mark a detector that measurably failed (like
`mislabel_suspect`, defect 50 -- 21.9% precision against a 17.4% base
rate, Fisher p=0.79, indistinguishable from chance) as `"validated"` and
nothing downstream would know better. `derive_verdict` computes the real
answer from `precision`/`base_rate`, and `write_validations` -- the one
function that puts a record on disk -- calls it on every record before
writing, discarding whatever `verdict` the caller supplied. A record
constructed in memory with an inconsistent `verdict` is not intercepted
(pydantic has no hook for "recompute one field from two others" that
survives `model_copy`), but nothing in this module, or the CLI wired to
it (`items detector-score --write-validation`), ever reads a
`DetectorValidation` except by loading one back from a file
`write_validations` produced -- so the derived verdict is the only one
that is ever actually consumed.

`unmeasured`: no precision estimate to judge at all -- `precision is
None`, or `Estimate.value is None` (`Estimate.verdict ==
"no_denominator"`: nothing was ever flagged and labelled).

`validated`: precision's 95% CI lies entirely above `base_rate` --
measured to do reliably better than a random draw of the same size, not
merely a higher point estimate.

`not_validated`: every other measured case -- `base_rate` inside the CI
(indistinguishable from chance, the `mislabel_suspect` case) or the CI at
or below `base_rate` (measured *worse* than a random draw). A detector
measured significantly worse than chance is a different failure from one
indistinguishable from it, but neither is a working detector, so neither
keeps the `validated` label.

## Persistence -- append, never rewrite

`load_validations`/`write_validations` mirror `items.calibration`'s JSONL
persistence functions: one JSON object per line, the file handle iterated
(never `read_text().splitlines()` -- the gov.uk-corpus U+2028 trap
documented in `items.calibration`'s module docstring applies identically
to any JSONL this project reads). `write_validations` **appends**: a
detector may be re-measured on a later calibration pass, and each
measurement is its own dated record, not an overwrite of the last.
`load_validations` returns a `dict` keyed by `detector` -- if a file
accumulated by repeated `--write-validation` runs holds more than one
record for the same detector, the last one read wins, so a caller always
sees that detector's most recent measurement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from fireassay.items.core import Estimate


class DetectorValidation(BaseModel):
    """One detector's measured validation status -- the unit
    `write_validations` persists and `render_items_analysis`'s
    `--validations` table reads back (Change 5, M-ITEMS-SPEC.md /
    eval-of-evals Goal 2 criterion 3).

    `detector` is the name being scored, e.g. `"mislabel_suspect"` or
    `"answerability.not_supported"` -- not restricted to fireassay's own
    detectors, the same "any detector, including one written by someone
    who has never heard of fireassay" discipline as
    `items.calibration.evaluate_detector`. `labels` describes what it was
    measured against in prose, e.g. `"132 uniform-random human labels,
    run/calibration.jsonl"` -- there is no structured field for this
    because the labels source is whatever a caller can name, not a
    fireassay-internal identifier. `note` is free text for context a
    number alone cannot carry, e.g. the structural reason a detector
    failed (`mislabel_suspect`'s nested-retrieval-sets story, defect 49) --
    `None` when there is nothing more to say than the numbers themselves.

    **`verdict` is never trusted as written** -- see the module docstring.
    """

    model_config = ConfigDict(frozen=True)

    detector: str
    measured_on: str
    labels: str
    base_rate: float
    precision: Estimate | None
    recall: Estimate | None
    note: str | None
    verdict: Literal["validated", "not_validated", "unmeasured"]


def derive_verdict(
    precision: Estimate | None, base_rate: float
) -> Literal["validated", "not_validated", "unmeasured"]:
    """Compute a `DetectorValidation.verdict` from the numbers -- never
    from what a caller asserts. See the module docstring for what each of
    the three outcomes means and why a CI below `base_rate` is
    `not_validated`, not `validated`-in-the-opposite-direction."""
    if precision is None or precision.value is None or precision.ci is None:
        return "unmeasured"
    lo, _hi = precision.ci
    return "validated" if lo > base_rate else "not_validated"


def load_validations(path: Path | str) -> dict[str, DetectorValidation]:
    """Load a `DetectorValidation` JSONL file into a `dict` keyed by
    `detector` (see the module docstring for the last-record-wins rule on
    a repeated detector name). Raises `ValueError` with the offending line
    number for invalid JSON or a field that fails `DetectorValidation`
    validation."""
    source = str(path)
    out: dict[str, DetectorValidation] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"validation: {source}: line {line_no}: invalid JSON: {exc}") from exc
            try:
                record = DetectorValidation(**obj)
            except ValidationError as exc:
                raise ValueError(
                    f"validation: {source}: line {line_no}: invalid DetectorValidation: {exc}"
                ) from exc
            out[record.detector] = record
    return out


def write_validations(records: Sequence[DetectorValidation], path: Path | str) -> int:
    """Append `records` to `path`, creating it if absent -- **append,
    never rewrite** (see the module docstring). Returns the number of
    records written.

    **Recomputes `verdict` via `derive_verdict` before writing, for every
    record, discarding whatever `verdict` the caller supplied** -- this is
    the actual enforcement point for the module docstring's "verdict is
    derived, never asserted" rule: `DetectorValidation` is frozen, so the
    caller's in-memory object is untouched, but nothing that lands on disk
    through this function ever carries a hand-asserted verdict that
    disagrees with its own `precision`/`base_rate`."""
    p = Path(path)
    corrected = [r.model_copy(update={"verdict": derive_verdict(r.precision, r.base_rate)}) for r in records]
    with open(p, "a", encoding="utf-8") as f:
        for record in corrected:
            f.write(record.model_dump_json())
            f.write("\n")
    return len(corrected)
