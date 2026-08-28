"""`shuffled_gold` — permutes `evidence_spans` across gold-bearing
questions (deterministic seeds derived from `suite_hash`), leaving the
system and every question's `text`/`id` untouched, and checks that
`recall@k` collapses to a **self-calibrated** chance level.

**Replaces v2's `shuffled_reference`** (fireassay-SPEC.md §6):
permuting reference *answers* only moves a judge-based correctness metric,
which does not exist before M4. Permuting gold *spans* collapses
retrieval, which is measurable today — same idea, deterministic form
(M2-SPEC.md §3).

The system is never wrapped: it answers the *original* question text
exactly as it would for an ordinary run (so its actual retrieval behaviour
is unaffected), but each question's evidence spans used for *scoring* are
swapped for another gold-bearing question's — via `Question.model_copy`,
which does not recompute `id` (unlike constructing a fresh `Question`),
so results still key against the suite's real, already-frozen question
ids.

**Why the chance band is measured, not assumed (correction #1):**
chance-level recall under a random permutation of gold spans depends
entirely on corpus size and `top_k` — a BM25 system that returns
`top_k=5` out of a 12-document corpus retrieves ~40% of the corpus on
every query, so permuted (i.e. meaningless) gold still overlaps
*something* roughly half the time. `0.5` genuinely *is* chance on this
fixture; it is nowhere near chance on the intended ~1,181-document
production corpus (~5/1181 ≈ 0.004). A fixed band silently encodes one
corpus size and is wrong for every other. So this control *measures* its
own chance level, per suite/config, from `n_seeds` independent
permutations, the same way a paired-bootstrap gate measures its own noise
floor rather than assuming one.

**Why the pass criterion compares an estimate to an estimate, not one
draw to a band (correction #2, then #3):** a first attempt required
`primary_recall <= max(calibration_recalls)`. One fresh, independent draw
exceeds the maximum of `n` previous independent draws with probability
`1/(n+1)` *even when everything works correctly* — a ~5% built-in
false-alarm rate that never reaches zero as `n_seeds` grows. A second
attempt replaced the maximum with a normal-theory tolerance interval,
`chance_level + k * chance_sd` at `k=3`. That undershot too: on this
fixture's real, measured single-draw distribution (300 draws sampled;
`mean≈0.3763`, `sd≈0.1048`, only **17 distinct values** — `recall@5` over
14 gold-bearing questions is a coarse, discrete statistic, not a smooth
normal one) `k=3` still produced a measured **1.0% false-alarm rate**
across 200 independently-seeded suites, because a normal-theory tail
bound underperforms against a coarse, discrete, slightly-skewed empirical
tail.

The actual fix: stop comparing a single draw to the calibration spread at
all. Average `m_primary` independent permutations into `primary_recall`
(still disjoint from the calibration set) and compare *that estimate* to
the calibration mean using the **standard error of the difference between
two means**, `se_diff = sd * sqrt(1/m_primary + 1/n_seeds)`, rather than
the raw per-draw `sd`. Averaging shrinks the noise in what is being
compared without shrinking the signal a genuinely broken metric produces
(a broken metric still reports ~twin-level recall, ~1.0, regardless of
how many draws are averaged) — simulating 40,000 trials per candidate
against the measured empirical distribution found this criterion
(`n_seeds=50`, `m_primary=5`, `k=5`) at a **0.0000%** simulated
false-alarm rate, comfortably below the single-digit-percent rates every
single-draw variant produced regardless of `k`.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Literal

from fireassay.controls._common import (
    band_map,
    base_scoring_ctx,
    canonical_k,
    execute,
    fixed_k_ctx,
    mean,
    mean_metric,
    or_nan,
    stable_seed,
)
from fireassay.controls.base import ControlContext, ControlOutcome
from fireassay.models import Question


def _seeded_derangement(n: int, seed: int) -> list[int]:
    """A permutation of `range(n)` with no fixed points (when `n > 1`),
    derived from `seed`. A true fixed point would mean a gold-bearing
    question keeps its own gold spans, defeating the point of the
    control for that one question."""
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    if n > 1:
        for i in range(n):
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
    return perm


def _permute_gold_spans(questions: list[Question], seed: int) -> list[Question]:
    """Return a copy of `questions` where every gold-bearing question's
    `evidence_spans` has been swapped for a *different* gold-bearing
    question's, via a seeded derangement. Questions with no gold spans
    (e.g. `unanswerable`) are returned unchanged — permuting an empty
    tuple onto another empty tuple is a no-op that would only add noise.
    """
    gold_indices = [i for i, q in enumerate(questions) if q.evidence_spans]
    perm = _seeded_derangement(len(gold_indices), seed)
    donor_spans = {
        gold_indices[i]: questions[gold_indices[perm[i]]].evidence_spans for i in range(len(gold_indices))
    }
    return [
        q.model_copy(update={"evidence_spans": donor_spans[i]}) if i in donor_spans else q
        for i, q in enumerate(questions)
    ]


class ShuffledGoldControl:
    kind = "shuffled_gold"
    version = "1.0.0"

    def run(self, ctx: ControlContext) -> ControlOutcome:
        bands = band_map(ctx, self.kind)
        min_margin = bands["min_margin"].eq
        n_seeds_raw = bands["n_seeds"].eq
        m_primary_raw = bands["m_primary"].eq
        k_raw = bands["k"].eq
        sd_floor = bands["sd_floor"].eq
        if (
            min_margin is None
            or n_seeds_raw is None
            or m_primary_raw is None
            or k_raw is None
            or sd_floor is None
        ):
            raise KeyError(
                "controls/expected.yaml: shuffled_gold requires min_margin.eq, n_seeds.eq, "
                "m_primary.eq, k.eq and sd_floor.eq"
            )
        n_seeds = max(1, int(n_seeds_raw))
        m_primary = max(1, int(m_primary_raw))
        tolerance_k = k_raw

        k = canonical_k(ctx)
        scoring_ctx = fixed_k_ctx(base_scoring_ctx(ctx), k)
        questions = list(ctx.questions)

        gold_indices = [i for i, q in enumerate(questions) if q.evidence_spans]
        if not gold_indices:
            return ControlOutcome(
                kind=self.kind,
                status="NOT_RUN",
                observed={},
                expected={name: b.model_dump(exclude_none=True) for name, b in bands.items()},
                twin_ok=False,
                cause_assertions={},
                detail="suite has no gold-bearing questions to permute evidence spans across",
            )

        # Writable twin: the unmodified config's own recall, measured once.
        twin_system = ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs)
        twin_run = execute(
            ctx,
            config_spec=dict(ctx.base_config.spec, _control=f"{self.kind}_twin"),
            system=twin_system,
            questions=questions,
            scoring_ctx=scoring_ctx,
        )
        twin_recall = mean_metric(ctx, twin_run.id, f"retrieval.recall@{k}")

        # The primary, canonical, reported measurement -- m_primary
        # permutations, seeded purely from suite_hash so they reproduce
        # across runs of the same suite (see test_controls_shuffled_gold.py's
        # determinism test), averaged into one estimate. Kept disjoint
        # from the calibration seeds below so that estimate can genuinely
        # be compared against them, rather than being trivially a member
        # of the set it is judged against. Averaging several draws (rather
        # than reporting one) is what makes the comparison a
        # standard-error-of-the-difference test instead of a single-draw
        # tail bound -- see the module docstring for why a single draw
        # was not good enough on this fixture's coarse, discrete recall
        # distribution.
        primary_recalls: list[float] = []
        primary_spans_permuted = True
        primary_ids_preserved = True
        for i in range(m_primary):
            seed = stable_seed(ctx.suite.suite_hash, "shuffled_gold_primary", i)
            primary_questions = _permute_gold_spans(questions, seed)
            primary_spans_permuted = primary_spans_permuted and all(
                primary_questions[j].evidence_spans != questions[j].evidence_spans for j in gold_indices
            )
            primary_ids_preserved = primary_ids_preserved and all(
                primary_questions[j].id == questions[j].id for j in range(len(questions))
            )
            primary_system = ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs)
            primary_run = execute(
                ctx,
                config_spec=dict(ctx.base_config.spec, _control=f"{self.kind}_primary_{i}"),
                system=primary_system,
                questions=primary_questions,
                scoring_ctx=scoring_ctx,
            )
            recall = mean_metric(ctx, primary_run.id, f"retrieval.recall@{k}")
            if recall is not None:
                primary_recalls.append(recall)
        primary_recall = mean(primary_recalls)

        # Calibration: n_seeds INDEPENDENT permutations (never overlapping
        # the primary seeds above) purely to measure this suite/config's
        # own chance level and spread -- "measure the baseline, never
        # assume it", the same principle a paired-bootstrap gate applies
        # to its own noise floor.
        calibration_recalls: list[float] = []
        for i in range(n_seeds):
            seed = stable_seed(ctx.suite.suite_hash, "shuffled_gold_calibration", i)
            calib_questions = _permute_gold_spans(questions, seed)
            calib_system = ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs)
            calib_run = execute(
                ctx,
                config_spec=dict(ctx.base_config.spec, _control=f"{self.kind}_calibration_{i}"),
                system=calib_system,
                questions=calib_questions,
                scoring_ctx=scoring_ctx,
            )
            recall = mean_metric(ctx, calib_run.id, f"retrieval.recall@{k}")
            if recall is not None:
                calibration_recalls.append(recall)

        chance_level = mean(calibration_recalls)
        # Population standard deviation (not sample/Bessel-corrected): we
        # are describing the spread of exactly the n_seeds draws we took,
        # not estimating a larger population's spread from a subsample.
        chance_sd = statistics.pstdev(calibration_recalls) if calibration_recalls else None
        se_diff = (
            max(chance_sd, sd_floor) * math.sqrt(1 / m_primary + 1 / n_seeds)
            if chance_sd is not None
            else None
        )
        upper_tolerance = (
            chance_level + tolerance_k * se_diff
            if chance_level is not None and se_diff is not None
            else None
        )

        twin_ok = (
            twin_recall is not None
            and chance_level is not None
            and twin_recall >= (chance_level + min_margin)
        )

        observed = {
            "recall_at_k": or_nan(primary_recall),
            "chance_level": or_nan(chance_level),
            "chance_sd": or_nan(chance_sd),
            "se_diff": or_nan(se_diff),
            "upper_tolerance": or_nan(upper_tolerance),
            "m_primary": float(m_primary),
            "n_seeds": float(n_seeds),
        }
        cause_assertions = {
            "gold_spans_were_permuted": primary_spans_permuted,
            "question_ids_preserved": primary_ids_preserved,
            "permuted_recall_within_tolerance": (
                primary_recall is not None
                and upper_tolerance is not None
                and primary_recall <= upper_tolerance
            ),
        }
        status: Literal["PASSED", "FAILED"] = (
            "PASSED" if (twin_ok and all(cause_assertions.values())) else "FAILED"
        )
        detail = (
            f"twin retrieval.recall@{k}={twin_recall!r}; chance_level={chance_level!r} "
            f"(n_seeds={n_seeds}), chance_sd={chance_sd!r}, se_diff={se_diff!r}, "
            f"upper_tolerance={upper_tolerance!r} (k={tolerance_k!r}, sd_floor={sd_floor!r}), "
            f"min_margin={min_margin!r} (twin_ok={twin_ok}); "
            f"primary (mean of {m_primary}) retrieval.recall@{k}={primary_recall!r}"
        )
        return ControlOutcome(
            kind=self.kind,
            status=status,
            observed=observed,
            expected={name: b.model_dump(exclude_none=True) for name, b in bands.items()},
            twin_ok=twin_ok,
            cause_assertions=cause_assertions,
            detail=detail,
        )
