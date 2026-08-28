"""`corpus_ablation` — drops 50% of the documents that carry gold spans
(seeded from `suite_hash`), rebuilds the system over the ablated corpus,
and checks that `recall@k` drops by at least `min_drop` relative to the
unmodified corpus.
"""

from __future__ import annotations

import random
from typing import Literal

from fireassay.controls._common import (
    band_map,
    base_scoring_ctx,
    canonical_k,
    execute,
    fixed_k_ctx,
    mean_metric,
    or_nan,
    seed_from_suite_hash,
)
from fireassay.controls.base import ControlContext, ControlOutcome


class CorpusAblationControl:
    kind = "corpus_ablation"
    version = "1.0.0"

    def run(self, ctx: ControlContext) -> ControlOutcome:
        bands = band_map(ctx, self.kind)
        drop_band = bands["recall_at_k_drop"]
        k = canonical_k(ctx)
        scoring_ctx = fixed_k_ctx(base_scoring_ctx(ctx), k)

        gold_doc_ids = sorted({span.doc_id for q in ctx.questions for span in q.evidence_spans})
        if not gold_doc_ids:
            return ControlOutcome(
                kind=self.kind,
                status="NOT_RUN",
                observed={},
                expected={name: b.model_dump(exclude_none=True) for name, b in bands.items()},
                twin_ok=False,
                cause_assertions={},
                detail="suite has no gold-bearing documents to ablate",
            )

        rng = random.Random(seed_from_suite_hash(ctx.suite.suite_hash))
        shuffled_doc_ids = gold_doc_ids[:]
        rng.shuffle(shuffled_doc_ids)
        n_drop = max(1, len(shuffled_doc_ids) // 2)
        dropped = set(shuffled_doc_ids[:n_drop])
        ablated_docs = [d for d in ctx.corpus_docs if d.doc_id not in dropped]

        # Writable twin: the unmodified corpus must itself be the higher
        # scorer, or a "drop" here proves nothing about the ablation.
        twin_system = ctx.system_factory(dict(ctx.base_config.spec), ctx.corpus_docs)
        twin_run = execute(
            ctx,
            config_spec=dict(ctx.base_config.spec, _control=f"{self.kind}_twin"),
            system=twin_system,
            questions=ctx.questions,
            scoring_ctx=scoring_ctx,
        )
        twin_recall = mean_metric(ctx, twin_run.id, f"retrieval.recall@{k}")

        mutant_system = ctx.system_factory(dict(ctx.base_config.spec), ablated_docs)
        mutant_run = execute(
            ctx,
            config_spec=dict(ctx.base_config.spec, _control=self.kind, _ablated_doc_ids=sorted(dropped)),
            system=mutant_system,
            questions=ctx.questions,
            scoring_ctx=scoring_ctx,
        )
        mutant_recall = mean_metric(ctx, mutant_run.id, f"retrieval.recall@{k}")

        drop = (
            (twin_recall - mutant_recall) if (twin_recall is not None and mutant_recall is not None) else None
        )
        twin_ok = twin_recall is not None and mutant_recall is not None and twin_recall > mutant_recall
        observed = {"recall_at_k_drop": or_nan(drop)}
        cause_assertions = {
            "documents_were_dropped": len(dropped) > 0,
            "recall_dropped": drop is not None and drop > 0.0,
            "drop_meets_min": drop is not None and drop_band.check(drop),
        }
        status: Literal["PASSED", "FAILED"] = (
            "PASSED" if (twin_ok and all(cause_assertions.values())) else "FAILED"
        )
        detail = (
            f"dropped {len(dropped)}/{len(gold_doc_ids)} gold-bearing doc(s); "
            f"twin retrieval.recall@{k}={twin_recall!r}, mutant retrieval.recall@{k}={mutant_recall!r}, "
            f"drop={drop!r} (min required {drop_band.min!r})"
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
