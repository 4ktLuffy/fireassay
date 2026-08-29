"""The persistent record of what a human judged about real items --
`items/review.py` builds one blind review batch and scores it against its
own key; this module is what makes those human labels **detector-agnostic
and durable**: any detector, now or later -- including one written by
someone who has never heard of fireassay -- can be scored against the same
accumulated labels with no further human time (`handbook/eval-of-
evals.md` §8).

**Store-free**, same discipline as `items.core`: nothing in this module
imports `fireassay.store`, or anything that transitively does
(`test_items_no_store_import.py` enforces this for the new module too).

**Seeded items must never become `CalibrationLabel`s.** A seeded item is a
manufactured defect (`items.seed`), not an observation of the real suite --
one in this file silently corrupts every base rate `evaluate_detector` will
ever produce from it, and nothing downstream could tell the difference.
`CalibrationLabel` has no field to carry a seeded flag at all: the schema
itself cannot represent one. The enforcement point is the export boundary
-- `fireassay items review export-calibration` (the CLI wired to this
module) skips every `is_seeded` entry from a review key before a
`CalibrationLabel` is ever constructed, and reports how many it skipped.
This module trusts that boundary; it does not re-check it, since by the
time a `CalibrationLabel` reaches here it has already been stripped of the
only signal (`ReviewKeyEntry.is_seeded`) that could identify it.

## `Estimate` -- no statistic is ever a bare number

`Estimate` itself is defined in `items.core` (shared with `items.ppi`) --
imported here, not redefined. `value`/`ci` are `None` **only** when `n == 0` (`verdict ==
"no_denominator"`) -- never a fabricated `0.0`, the same discipline
`DetectorScore` already applies in `items.review` and fireassay's scorers
apply throughout. When `0 < n < min_denominator`, `value`/`ci` are still
reported -- they are not wrong, only wide -- but `verdict ==
"too_few_labels"` so no reader can quote the point estimate as settled.
`verdict == "needs_sampling_design"` is a third way a value can be missing,
distinct from both -- see "The disjoint-stratum trap" below.

`_MIN_DENOMINATOR = 10` is a documented **convention**, not a measured
threshold -- mirror how `items.core._MIN_LIVE_ITEMS_PER_SPLIT = 10` is
justified: a round number picked so a reader is warned well before a
single-digit sample masquerades as a rate, not a value derived from any
experiment. A caller may override it via `evaluate_detector`'s
`min_denominator` argument.

## The estimator -- read this twice, it is the whole point

**A review batch deliberately over-samples the detector's own positives.**
A real batch might take every `mislabel_suspect` item the detector flagged
-- a census of that stratum -- alongside a modest random draw from the
whole suite (`stratum == "calibration"`). Pooling every label into one
denominator would over-weight the flagged stratum by an order of magnitude
or more and report a wildly inflated defect rate for the suite. So the
strata are **not interchangeable**, and each statistic must come from the
one that is unbiased for it:

| statistic | computed over |
|---|---|
| `precision` | **all** labelled flagged items, any stratum |
| `recall` | `stratum == "calibration"` **only** |
| `fpr` | `stratum == "calibration"` **only** |
| `base_rate` | `stratum == "calibration"` **only** |

**Why:** enrichment does not bias `precision` -- of what a detector
flagged, how many were bad is the same question at any sample rate.
`recall`/`fpr` each need an unbiased denominator (of bad items, and of
not-bad items, respectively); the over-sampled flagged census would
inflate `recall` toward 1.0 and misstate `fpr`. `base_rate` is a property
of the suite, estimable only from a random draw, never from a set the
detector itself selected.

`flagged_ids` is supplied by the caller, independently of whatever stratum
a label happened to be sampled under when it was originally reviewed --
this is what lets a detector nobody has heard of yet be scored against
labels collected for a different one entirely: `precision` asks only
"is this item's id in `flagged_ids`?", never "was this label drawn from
the *flagged* stratum?".

`n_flagged_unlabelled` matters: a detector may flag items nobody has
labelled. Those are excluded from precision's denominator -- you cannot
judge what was not judged -- and reporting the count is how a reader sees
the coverage, the same "omit, never zero-fill" rule `items.core` and
fireassay's scorers apply throughout.

## The disjoint-stratum trap, and the stratified fix

**`recall`/`fpr` computed directly over `stratum == "calibration"` can be
zero *by construction*, with no bearing on the detector at all.**
`items.review._plan_batch` claims flagged items first and draws the
calibration sample from whatever remains -- so on a deck built that way,
`flagged_ids INTERSECT` the calibration stratum's item ids is **always
empty**, regardless of any label. `recall = (bad AND flagged in random) /
(bad in random)` and `fpr = (flagged AND not-bad in random) / (not-bad in
random)` both then read `0/n` no matter what the detector actually did --
a detector with real false positives can print `fpr = 0.000`. This was
shipped once; it must never happen again silently.

`evaluate_detector` therefore checks whether the calibration stratum's
item ids intersect `flagged_ids` before computing `recall`/`fpr`/
`base_rate`:

- **Disjoint, no `design` supplied** -- the direct computation is
  structurally meaningless (see above). `value`/`ci` are `None` and
  `verdict == "needs_sampling_design"`, never a computed `0.0`.
- **Disjoint, `design` supplied** -- use the stratified population
  estimator below, which *is* valid for exactly this deck shape.
- **Not disjoint** -- the calibration stratum genuinely contains some of
  `flagged_ids` (a deck built without the claim-flagged-first invariant,
  or a detector scored against someone else's deck), so it is already a
  valid sample of the whole pool. Use the original direct computation --
  `design`, if supplied, is ignored, not applied on top of it.

`precision` is never affected by any of this -- it needs no scaling (see
above) and does not depend on the calibration stratum at all.

### `SamplingDesign` and the stratified estimator

    class SamplingDesign(BaseModel):    # frozen
        pool_size: int        # items in the suite the labels were drawn from
        flagged_size: int     # size of the flagged stratum in the SUITE

Neither is inferable from the labels file alone -- it has no idea how big
the suite is -- so a caller who wants recall/fpr/base_rate on a disjoint
deck must state both explicitly. Given `N_pool = design.pool_size`,
`N_flag = design.flagged_size`, `N_rest = N_pool - N_flag`, and the
observed counts (`TP`/`FP` = bad/not-bad among labelled flagged items,
i.e. exactly `precision`'s own numerator and denominator; `FN`/`TN` =
bad/not-bad among labelled calibration-stratum items):

    census_scale = N_flag / n_flag_labelled      (n_flag_labelled = TP + FP)
    rand_scale   = N_rest / n_rand_labelled       (n_rand_labelled = FN + TN)

    TP_pop = TP * census_scale     FP_pop = FP * census_scale
    FN_pop = FN * rand_scale       TN_pop = TN * rand_scale

    recall    = TP_pop / (TP_pop + FN_pop)
    fpr       = FP_pop / (FP_pop + TN_pop)
    base_rate = (TP_pop + FN_pop) / N_pool

Both strata are scaled up to the population they were drawn from -- the
census scaled by how much of the (already-enumerated) flagged stratum got
labelled, the random draw scaled by how much of the (much larger) rest of
the pool it represents. Mixing a raw sample count from one stratum against
a population estimate from the other (e.g. dividing an unscaled `TP` by a
scaled `FN_pop`) reproduces the exact same stratum error this whole fix
exists to correct, one level down -- both sides of every ratio above are
population estimates, or none are.

**This assumes the flagged items a reviewer skipped resemble the ones they
labelled.** `census_scale` scales `TP`/`FP` up as if the `N_flag -
n_flag_labelled` unlabelled census items would split the same way -- that
is not guaranteed: a reviewer may have skipped precisely the items they
found hardest or most ambiguous to judge, which would make the unlabelled
remainder systematically different from the labelled one. This module does
not attempt to correct for that -- there is no signal in a `CalibrationLabel`
file that could -- it only states the assumption here. The assumption is
moot, not merely small, when the census is complete: `n_flag_labelled ==
N_flag` makes `census_scale` exactly `1.0`, and the stratified figures
coincide exactly with the raw observed ones.

**Confidence intervals are an approximation, stated as one.** The interval
re-evaluates the formulas above at the endpoints of `wilson_ci(FN,
n_rand_labelled)` -- the random stratum's bad-proportion, the dominant
source of uncertainty here -- while holding `TP_pop`/`FP_pop` fixed at
their point estimate. It therefore propagates the random stratum's
sampling error only: it ignores any covariance between the two strata's
sampling errors, and it treats the (near-complete) census as exact. State
this, do not paper over it with a falsely tight or falsely symmetric
interval.

## Reviewer agreement

**An item labelled by two reviewers appears twice.** `evaluate_detector`
does not silently deduplicate: pass `reviewer=None` (the default) and, if
every `item_id` in `labels` carries only one verdict (whether from one
reviewer or several reviewers who happened to agree), every label counts.
If two reviewers *disagree* on the same `item_id`, `evaluate_detector`
raises `ValueError` naming the item and the disagreeing reviewers --
adjudicating a disagreement is a human act, and guessing at it here (e.g.
by majority vote, or by silently preferring one label) would manufacture
an agreement that was never measured. Pass `reviewer="alice"` to restrict
the computation to one reviewer's labels and sidestep the conflict.

## Persistence -- append-only

`load_calibration_jsonl`/`append_calibration_jsonl` read/write one JSON
object per line, `CalibrationLabel`-shaped. **Append, never rewrite**:
this file is meant to accumulate across many review passes, toward the
200-500 items §8 asks for -- a single 40-item pass is a seed sample, not a
sufficient one, and overwriting it on a second pass would throw away every
label collected so far. `append_calibration_jsonl` rejects a duplicate
`(item_id, reviewer)` (against both the file's existing contents and
within the batch being appended) with a `ValueError` naming both, the same
reasoning as `adapters.tabular.load_meta_jsonl`'s duplicate-`item_id`
rejection: letting a later row silently overwrite an earlier one hides a
data-authoring mistake. The same `item_id` from *different* reviewers is
legitimate and must be accepted -- that is how inter-rater agreement gets
measured later.

**Iterate the file handle; never `read_text().splitlines()`.** The gov.uk
corpus contains a U+2028 LINE SEPARATOR character, which `str.splitlines()`
treats as a line break even though JSON permits it raw inside a string --
splitting on it truncates one JSONL record into two unparseable halves.
Iterating the open file object instead only splits on the newline
conventions Python's text-mode line iteration recognises (`\\n`, `\\r\\n`,
`\\r`), leaving a U+2028 inside a JSON string value alone. See the same
warning in `adapters/tabular.py`'s module docstring, where the JSONL
contract this module's persistence functions also follow is defined.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from fireassay.items.core import Estimate, wilson_ci

Stratum = Literal["flagged", "unflagged", "calibration"]


class CalibrationLabel(BaseModel):
    """One reviewer's verdict on one real item, tagged with the stratum it
    was sampled from -- the durable, detector-agnostic unit `evaluate_detector`
    consumes. Never constructed from a seeded item (see the module
    docstring)."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    verdict: Literal["keep", "rewrite", "purge"]
    stratum: Stratum
    reviewer: str
    batch_id: str


class SamplingDesign(BaseModel):
    """The suite-level facts the stratified estimator needs (see the
    module docstring) -- neither is inferable from a labels file alone,
    since it has no idea how big the suite is. `pool_size` is every item
    in the suite the labels were drawn from; `flagged_size` is the size of
    the flagged stratum **in the suite**, not merely how many of it made
    it into the review deck."""

    model_config = ConfigDict(frozen=True)

    pool_size: int
    flagged_size: int

    @model_validator(mode="after")
    def _check_sizes(self) -> SamplingDesign:
        if self.flagged_size < 0:
            raise ValueError(f"SamplingDesign: flagged_size must be >= 0, got {self.flagged_size}")
        if self.pool_size <= self.flagged_size:
            raise ValueError(
                f"SamplingDesign: pool_size ({self.pool_size}) must be greater than "
                f"flagged_size ({self.flagged_size})"
            )
        return self


class DetectorEvaluation(BaseModel):
    """`evaluate_detector`'s output: `precision`/`recall`/`fpr`/`base_rate`
    as `Estimate`s (see the module docstring's stratum table for what each
    is computed over), plus the counts a reader needs to audit them.
    `stratum_estimator` says which formula produced `recall`/`fpr`/
    `base_rate` -- `"random_stratum_only"` (the direct computation) or
    `"stratified"` (the census-plus-sample population estimator) -- so a
    renderer or reader is never left guessing which one is behind a given
    number (`precision` is unaffected either way)."""

    model_config = ConfigDict(frozen=True)

    precision: Estimate
    recall: Estimate
    fpr: Estimate
    base_rate: Estimate
    stratum_estimator: Literal["random_stratum_only", "stratified"]
    n_labels: int
    n_random_stratum: int
    n_flagged_total: int
    n_flagged_labelled: int
    n_flagged_unlabelled: int


#: Documented convention, not a measured threshold -- see the module
#: docstring's `Estimate` section, mirroring
#: `items.core._MIN_LIVE_ITEMS_PER_SPLIT`.
_MIN_DENOMINATOR = 10


def _estimate(successes: int, n: int, *, min_denominator: int) -> Estimate:
    if n == 0:
        return Estimate(value=None, ci=None, n=0, verdict="no_denominator")
    verdict: Literal["measured", "too_few_labels"] = (
        "measured" if n >= min_denominator else "too_few_labels"
    )
    return Estimate(value=successes / n, ci=wilson_ci(successes, n), n=n, verdict=verdict)


def _stratified_point(
    tp_pop: float, fp_pop: float, fn_pop: float, tn_pop: float, pool_size: int
) -> tuple[float, float, float]:
    """One evaluation of the stratified formulas (see the module
    docstring) at a given set of population-scale TP/FP/FN/TN. Called once
    at the point estimate and twice more (at the random stratum's Wilson
    endpoints) to build the approximate CI in `_stratified_estimate`."""
    recall = tp_pop / (tp_pop + fn_pop) if (tp_pop + fn_pop) > 0 else 0.0
    fpr = fp_pop / (fp_pop + tn_pop) if (fp_pop + tn_pop) > 0 else 0.0
    base_rate = (tp_pop + fn_pop) / pool_size
    return recall, fpr, base_rate


def _stratified_estimate(
    *,
    tp_obs: int,
    fp_obs: int,
    fn_obs: int,
    tn_obs: int,
    design: SamplingDesign,
    n_flag_labelled: int,
    n_rand_labelled: int,
    min_denominator: int,
) -> tuple[Estimate, Estimate, Estimate]:
    """The stratified population estimator (see the module docstring) --
    returns `(recall, fpr, base_rate)`. Caller guarantees `n_flag_labelled
    > 0` and `n_rand_labelled > 0`; a zero on either side is handled by
    `evaluate_detector` before this is called, since there is nothing here
    to scale by a zero denominator."""
    census_scale = design.flagged_size / n_flag_labelled
    n_rest = design.pool_size - design.flagged_size
    rand_scale = n_rest / n_rand_labelled

    tp_pop = tp_obs * census_scale
    fp_pop = fp_obs * census_scale
    fn_pop = fn_obs * rand_scale
    tn_pop = tn_obs * rand_scale

    recall_point, fpr_point, base_rate_point = _stratified_point(
        tp_pop, fp_pop, fn_pop, tn_pop, design.pool_size
    )

    # Approximate CI: re-evaluate at the random stratum's Wilson endpoints
    # for its bad-proportion, holding the census (TP_pop/FP_pop) fixed at
    # its point estimate -- see the module docstring's caveats.
    p_lo, p_hi = wilson_ci(fn_obs, n_rand_labelled)
    recall_lo, fpr_lo, base_lo = _stratified_point(
        tp_pop, fp_pop, p_lo * n_rest, (1 - p_lo) * n_rest, design.pool_size
    )
    recall_hi, fpr_hi, base_hi = _stratified_point(
        tp_pop, fp_pop, p_hi * n_rest, (1 - p_hi) * n_rest, design.pool_size
    )

    verdict: Literal["measured", "too_few_labels"] = (
        "measured" if n_rand_labelled >= min_denominator else "too_few_labels"
    )
    return (
        Estimate(
            value=recall_point,
            ci=(min(recall_lo, recall_hi), max(recall_lo, recall_hi)),
            n=n_rand_labelled,
            verdict=verdict,
        ),
        Estimate(
            value=fpr_point,
            ci=(min(fpr_lo, fpr_hi), max(fpr_lo, fpr_hi)),
            n=n_rand_labelled,
            verdict=verdict,
        ),
        Estimate(
            value=base_rate_point,
            ci=(min(base_lo, base_hi), max(base_lo, base_hi)),
            n=n_rand_labelled,
            verdict=verdict,
        ),
    )


def evaluate_detector(
    flagged_ids: Collection[str],
    labels: Sequence[CalibrationLabel],
    *,
    bad_verdicts: frozenset[str] = frozenset({"purge"}),
    min_denominator: int = _MIN_DENOMINATOR,
    reviewer: str | None = None,
    design: SamplingDesign | None = None,
) -> DetectorEvaluation:
    """Score any detector's `flagged_ids` against `labels` -- see the
    module docstring for the stratum rule this function exists to enforce
    (pooling `flagged` and `calibration` labels for `recall`/`fpr`/
    `base_rate` would over-weight the flagged census and misreport the
    suite's real defect rate), and for the disjoint-stratum trap: a deck
    built by claiming flagged items first (`items.review._plan_batch`)
    makes the calibration stratum structurally disjoint from `flagged_ids`,
    which makes a *direct* recall/fpr computation read `0/n` regardless of
    the labels. `design` (a `SamplingDesign`) enables the stratified
    population estimator that is valid on exactly that deck shape; without
    it, a disjoint stratum reports `verdict == "needs_sampling_design"`
    rather than a misleading computed zero.

    **The stratified estimator assumes the flagged census's unlabelled
    remainder resembles its labelled part** -- see the module docstring's
    "This assumes..." paragraph. This is not corrected for, only stated;
    it disappears (the scale factor becomes exactly `1.0`) when every
    flagged item got a label.

    `reviewer=None` (the default) uses every label in `labels`, after
    checking that no `item_id` carries conflicting verdicts from different
    reviewers -- raises `ValueError` naming the item and the disagreeing
    reviewers if it does (see the module docstring). Passing a `reviewer`
    name restricts the computation to that reviewer's own labels, which by
    construction cannot conflict with itself.

    An item counts toward a numerator iff `label.verdict in bad_verdicts`
    (default the strict reading, `{"purge"}`; pass `{"purge", "rewrite"}`
    for the lenient one -- no default here is "the" threshold, the same
    refusal to pick one for the reader as `score_review`'s own
    `bad_verdicts`)."""
    if reviewer is not None:
        effective = [label for label in labels if label.reviewer == reviewer]
    else:
        by_item: dict[str, dict[str, str]] = {}
        for label in labels:
            by_item.setdefault(label.item_id, {})[label.reviewer] = label.verdict
        for item_id in sorted(by_item):
            verdicts_by_reviewer = by_item[item_id]
            if len(set(verdicts_by_reviewer.values())) > 1:
                detail = ", ".join(f"{r}={v}" for r, v in sorted(verdicts_by_reviewer.items()))
                raise ValueError(
                    f"evaluate_detector: item_id {item_id!r} has conflicting verdicts from "
                    f"different reviewers ({detail}) -- pass reviewer= to pick one; "
                    "adjudicating a disagreement is a human act, not something this function "
                    "may guess at"
                )
        effective = list(labels)

    flagged_set = set(flagged_ids)

    precision_num = precision_den = 0
    recall_num = recall_den = 0
    fpr_num = fpr_den = 0
    base_num = base_den = 0
    flagged_ids_labelled: set[str] = set()
    calibration_item_ids: set[str] = set()

    for label in effective:
        is_bad = label.verdict in bad_verdicts
        if label.item_id in flagged_set:
            precision_den += 1
            flagged_ids_labelled.add(label.item_id)
            if is_bad:
                precision_num += 1
        if label.stratum == "calibration":
            calibration_item_ids.add(label.item_id)
            base_den += 1
            if is_bad:
                base_num += 1
                recall_den += 1
                if label.item_id in flagged_set:
                    recall_num += 1
            else:
                fpr_den += 1
                if label.item_id in flagged_set:
                    fpr_num += 1

    n_flagged_total = len(flagged_set)
    n_flagged_labelled = len(flagged_ids_labelled)

    is_disjoint = flagged_set.isdisjoint(calibration_item_ids)

    recall: Estimate
    fpr: Estimate
    base_rate: Estimate
    stratum_estimator: Literal["random_stratum_only", "stratified"]

    if not is_disjoint:
        # The calibration stratum genuinely contains some of flagged_ids --
        # a valid sample of the whole pool. `design`, if supplied, is
        # deliberately ignored here; see the module docstring.
        stratum_estimator = "random_stratum_only"
        recall = _estimate(recall_num, recall_den, min_denominator=min_denominator)
        fpr = _estimate(fpr_num, fpr_den, min_denominator=min_denominator)
        base_rate = _estimate(base_num, base_den, min_denominator=min_denominator)
    else:
        stratum_estimator = "stratified"
        if base_den == 0:
            # No random-stratum labels at all -- the same "nothing to
            # compute" case as any other zero denominator, regardless of
            # disjointness or a supplied design.
            recall = Estimate(value=None, ci=None, n=0, verdict="no_denominator")
            fpr = Estimate(value=None, ci=None, n=0, verdict="no_denominator")
            base_rate = Estimate(value=None, ci=None, n=0, verdict="no_denominator")
        elif design is None:
            recall = Estimate(value=None, ci=None, n=base_den, verdict="needs_sampling_design")
            fpr = Estimate(value=None, ci=None, n=base_den, verdict="needs_sampling_design")
            base_rate = Estimate(value=None, ci=None, n=base_den, verdict="needs_sampling_design")
        elif precision_den == 0:
            # No labelled flagged/census items at all -- there is nothing
            # to scale the census by, the same discipline as above.
            recall = Estimate(value=None, ci=None, n=0, verdict="no_denominator")
            fpr = Estimate(value=None, ci=None, n=0, verdict="no_denominator")
            base_rate = Estimate(value=None, ci=None, n=0, verdict="no_denominator")
        else:
            recall, fpr, base_rate = _stratified_estimate(
                tp_obs=precision_num,
                fp_obs=precision_den - precision_num,
                fn_obs=base_num,
                tn_obs=fpr_den,
                design=design,
                n_flag_labelled=precision_den,
                n_rand_labelled=base_den,
                min_denominator=min_denominator,
            )

    return DetectorEvaluation(
        precision=_estimate(precision_num, precision_den, min_denominator=min_denominator),
        recall=recall,
        fpr=fpr,
        base_rate=base_rate,
        stratum_estimator=stratum_estimator,
        n_labels=len(effective),
        n_random_stratum=base_den,
        n_flagged_total=n_flagged_total,
        n_flagged_labelled=n_flagged_labelled,
        n_flagged_unlabelled=n_flagged_total - n_flagged_labelled,
    )


def load_calibration_jsonl(path: Path | str) -> list[CalibrationLabel]:
    """Load a `CalibrationLabel` JSONL file (see the module docstring for
    the format and the U+2028 trap). Raises `ValueError` with the offending
    line number for invalid JSON, a field that fails `CalibrationLabel`
    validation, or a duplicate `(item_id, reviewer)` within the file."""
    source = str(path)
    out: list[CalibrationLabel] = []
    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"calibration: {source}: line {line_no}: invalid JSON: {exc}") from exc
            try:
                label = CalibrationLabel(**obj)
            except ValidationError as exc:
                raise ValueError(
                    f"calibration: {source}: line {line_no}: invalid CalibrationLabel: {exc}"
                ) from exc
            key = (label.item_id, label.reviewer)
            if key in seen:
                raise ValueError(
                    f"calibration: {source}: line {line_no}: duplicate (item_id, reviewer) "
                    f"{key!r} -- already loaded earlier in this file"
                )
            seen.add(key)
            out.append(label)
    return out


def append_calibration_jsonl(labels: Sequence[CalibrationLabel], path: Path | str) -> int:
    """Append `labels` to `path`, creating it if absent -- **append, never
    rewrite** (see the module docstring). Raises `ValueError` naming both
    the `item_id` and `reviewer` for any duplicate `(item_id, reviewer)`,
    checked against both `path`'s existing contents and the rest of
    `labels` itself, before writing anything -- a duplicate anywhere in the
    batch aborts the whole append rather than leaving a partially-written
    file. Returns the number of labels written."""
    p = Path(path)
    existing = load_calibration_jsonl(p) if p.exists() else []
    seen = {(label.item_id, label.reviewer) for label in existing}
    for label in labels:
        key = (label.item_id, label.reviewer)
        if key in seen:
            raise ValueError(
                f"calibration: {p}: duplicate (item_id, reviewer) {key!r} -- already present"
            )
        seen.add(key)
    with open(p, "a", encoding="utf-8") as f:
        for label in labels:
            f.write(label.model_dump_json())
            f.write("\n")
    return len(labels)
