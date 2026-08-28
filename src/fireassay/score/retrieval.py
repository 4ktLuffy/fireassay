"""`RetrievalScorer` (`retrieval@1.0.0`) — deterministic retrieval metrics
against gold evidence spans, judged by character-range overlap.

**Why overlap, not chunk_id equality (M1-SPEC.md correction 1):** chunking
is a config axis (M1-SPEC.md §5 / fireassay-SPEC.md §5 — fixed/512,
recursive/1024, semantic, ...). `chunk_id` is `f"{doc_id}#{index:04d}"`, so
two configs with different chunk sizes produce entirely disjoint chunk_id
spaces even when they retrieve the exact same underlying text. Matching on
`chunk_id` equality meant any config not using the exact chunking that
produced a question's gold spans scored `recall@k == 0.0` — silently, and
indistinguishable from a genuinely bad retriever. Matching on character-range
overlap within the same document makes recall/nDCG/MRR comparable *across*
chunking strategies, which is the entire point of chunking being a
sweepable config axis in the first place.

**Why `judged_fraction` and `bpref` exist:** our gold evidence spans are
*incomplete* — they mark passages we know are relevant, not every passage
that is. A chunk that genuinely answers a question but was never annotated
as a gold span is counted as a miss by recall/nDCG/MRR, which systematically
penalises any config that retrieves a correct-but-different passage than the
one a question happened to be written against — which is most of the point
of comparing configs at all. We cannot fix that incompleteness in M1, but we
can make it visible (`judged_fraction`) and use a metric designed for
incomplete judgments in the first place (`bpref`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from fireassay.models import EvidenceSpan, Question, RetrievedChunk, Score, SystemOutput
from fireassay.score.base import ScoringContext

_NAME = "retrieval"
_VERSION = "1.0.0"


def _overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of the overlap between two half-open character ranges
    `[a_start, a_end)` and `[b_start, b_end)`; 0 if they do not overlap."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _overlaps_any(chunk: RetrievedChunk, spans: Sequence[EvidenceSpan], overlap_min_chars: int) -> bool:
    """Whether `chunk` overlaps at least one of `spans` by
    `overlap_min_chars` characters, restricted to spans in the same
    document as the chunk."""
    return any(
        span.doc_id == chunk.doc_id
        and _overlap_chars(chunk.char_start, chunk.char_end, span.char_start, span.char_end)
        >= overlap_min_chars
        for span in spans
    )


class RetrievalScorer:
    name = _NAME
    version = _VERSION
    requires_llm = False

    def score(self, question: Question, output: SystemOutput, ctx: ScoringContext) -> list[Score]:
        """Score one question's retrieval against its gold evidence spans.

        **If the question has no evidence spans, this returns an empty
        list** — no Score objects at all, not a Score of 0.0 (this
        applicability rule applies to every metric this scorer emits,
        including `judged_fraction` and `bpref`). A question with no gold
        evidence carries zero information about whether retrieval worked,
        in either direction; emitting 0.0 would silently assert "retrieval
        failed here," which is not something we know, and would corrupt
        every mean computed over `retrieval.*` (`compare.leaderboard`
        averages only over questions that *have* the metric — omitting is
        what makes that average trustworthy).

        For a question that does have gold spans, `retrieval.recall@k`,
        `retrieval.ndcg@k` and `retrieval.judged_fraction@k` are computed
        for **every k in `ctx.eval_ks`** (default `(1, 3, 5, 10)`), not
        only a single cutoff — this is what lets
        `score.invariants.check_invariants` verify recall/nDCG are
        monotonically non-decreasing in k for a given question, a sanity
        property that only a multi-k sweep can check. Each k is truncated
        to `len(output.retrieved)`: a system that only returned 3 chunks
        cannot be meaningfully evaluated at k=10, so `retrieval.recall@10`'s
        *name* still says 10 (comparable across runs with different
        retrieval breadths) while its *window* is
        `min(10, len(output.retrieved))`.

        - `retrieval.recall@k` = the fraction of **gold spans** (not gold
          chunks) covered by at least one relevant chunk in the top k. One
          large retrieved chunk covering three gold spans scores 3/3 —
          recall counts spans covered, not chunks retrieved. Two different
          retrieved chunks both covering the same one gold span still only
          count that span once (1/1, not 2/1).
        - `retrieval.ndcg@k` uses binary relevance per retrieved chunk
          ("covers >= 1 gold span"): `DCG = sum(1 / log2(rank + 1))` over
          1-based ranks of relevant chunks in the top k; `IDCG` is the same
          sum over the best-case `1..ideal_hits` ranks, where
          `ideal_hits = min(effective_k, max(len(gold_spans),
          relevant_chunks_actually_found))`; `ndcg = DCG / IDCG`, or `0.0`
          if `IDCG == 0`. The `max(...)` with the actually-observed
          relevant-chunk count (not just `len(gold_spans)`) is required
          for correctness: because relevance is per-chunk overlap rather
          than per-span chunk_id membership, a chunking strategy can
          legitimately retrieve *more* distinct relevant chunks than
          there are gold spans (e.g. two different chunks each partially
          overlapping the same wide gold span). Using `len(gold_spans)`
          alone as the ideal hit count would let `DCG` exceed `IDCG` and
          push `ndcg` above `1.0` in that case.
        - `retrieval.judged_fraction@k` = `(number of chunks in the top k
          that overlap ANY evidence span of ANY question in the suite,
          matched by doc_id) / effective_k` — using `ctx.judged_spans_by_doc`
          (the suite-wide judged pool, populated once per suite by
          `runner.run_matrix`), not just this question's own gold spans.
          This is a **coverage-of-judgment** signal, not a quality signal:
          a low `judged_fraction` means most of what was retrieved has
          simply never been looked at by anyone, so `recall`/`ndcg` for
          that question are measuring against a small, possibly
          unrepresentative sample of "actually relevant" — a leaderboard
          MUST show `judged_fraction` next to `recall`/`ndcg`, not bury it,
          because a 100% recall next to a 20% judged_fraction means much
          less than a 100% recall next to a 90% one.
        - `retrieval.mrr` = `1 / rank` of the first chunk in
          `output.retrieved` (the system's own full retrieved list, not
          re-truncated by `eval_ks`) that covers >= 1 gold span, else
          `0.0`. Unlike recall/nDCG, MRR's metric name does not
          interpolate a cutoff (matching M1-SPEC.md §7 exactly).
        - `retrieval.bpref` — a single value (also not `@k`-suffixed, like
          MRR): the standard IR metric for **incomplete relevance
          judgments** (Buckley & Voorhees 2004). Unlike recall/nDCG, it
          does not treat every unjudged chunk as irrelevant — it is
          computed only from chunks the run has an explicit judgment for,
          positive (this question's gold spans, R) or negative
          (`ctx.judged_nonrelevant_spans_by_question[question.id]`, N):

              bpref = (1/|R|) * sum_{r in R} (1 - min(|n above r|, min(|R|,|N|)) / min(|R|,|N|))

          where `r` ranges over retrieved judged-relevant chunks and `n`
          over retrieved judged-nonrelevant chunks, "above" meaning
          earlier rank. `0.0` if `R` is empty (nothing relevant was
          retrieved to compute a ratio over). `1.0` whenever `R` is
          non-empty and `N` is empty — **this is the expected, correct M1
          behaviour, not a bug**: M1 has no data source for negative
          judgments (nothing marks a passage irrelevant, only
          `evidence_spans` marks passages relevant), so
          `judged_nonrelevant_spans_by_question` is always empty in a real
          M1 run. `bpref` is implemented and tested against
          hand-supplied negative judgments now specifically so that a
          later milestone's pooling work (judging a sample of each
          config's unique top results) is a drop-in — populate that ctx
          field — rather than a scorer rewrite.
        """
        gold_spans = question.evidence_spans
        if not gold_spans:
            return []

        overlap_min = ctx.overlap_min_chars
        scorer_tag = f"{self.name}@{self.version}"
        retrieved_by_rank = sorted(output.retrieved, key=lambda rc: rc.rank)

        def is_relevant(chunk: RetrievedChunk) -> bool:
            return _overlaps_any(chunk, gold_spans, overlap_min)

        scores: list[Score] = []
        for k in ctx.eval_ks:
            effective_k = min(k, len(retrieved_by_rank))
            window = [rc for rc in retrieved_by_rank if rc.rank <= effective_k]

            covered_spans = sum(
                1 for span in gold_spans if any(_overlaps_any(rc, [span], overlap_min) for rc in window)
            )
            recall = covered_spans / len(gold_spans)

            relevant_in_window = sum(1 for rc in window if is_relevant(rc))
            dcg = sum(1.0 / math.log2(rc.rank + 1) for rc in window if is_relevant(rc))
            # ideal_hits must be >= the number of relevant chunks actually
            # observed, not just len(gold_spans): because relevance is
            # per-chunk overlap rather than per-span chunk_id membership, a
            # chunking strategy can retrieve *more* distinct relevant
            # chunks than there are gold spans (e.g. two different chunks
            # each partially overlapping the same wide gold span). Capping
            # ideal_hits at len(gold_spans) alone would let `dcg` exceed
            # `idcg` and push ndcg above 1.0 -- caught by
            # score.invariants.check_invariants' range check, which is
            # exactly why that check exists. max(...) keeps IDCG's
            # per-rank terms (1/log2(rank+1), strictly decreasing) as the
            # provable upper bound of DCG for however many relevant chunks
            # were actually found.
            ideal_hits = min(effective_k, max(len(gold_spans), relevant_in_window))
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            judged_in_window = sum(
                1
                for rc in window
                if _overlaps_any(rc, ctx.judged_spans_by_doc.get(rc.doc_id, ()), overlap_min)
            )
            judged_fraction = (judged_in_window / effective_k) if effective_k > 0 else 0.0

            scores.append(Score(metric=f"retrieval.recall@{k}", value=recall, scorer=scorer_tag))
            scores.append(Score(metric=f"retrieval.ndcg@{k}", value=ndcg, scorer=scorer_tag))
            scores.append(
                Score(metric=f"retrieval.judged_fraction@{k}", value=judged_fraction, scorer=scorer_tag)
            )

        mrr = 0.0
        for rc in retrieved_by_rank:
            if is_relevant(rc):
                mrr = 1.0 / rc.rank
                break
        scores.append(Score(metric="retrieval.mrr", value=mrr, scorer=scorer_tag))

        nonrelevant_spans = ctx.judged_nonrelevant_spans_by_question.get(question.id, ())
        bpref = _bpref(retrieved_by_rank, is_relevant, nonrelevant_spans, overlap_min)
        scores.append(Score(metric="retrieval.bpref", value=bpref, scorer=scorer_tag))

        return scores


def _bpref(
    retrieved_by_rank: Sequence[RetrievedChunk],
    is_relevant: Callable[[RetrievedChunk], bool],
    nonrelevant_spans: Sequence[EvidenceSpan],
    overlap_min_chars: int,
) -> float:
    """Buckley & Voorhees (2004) bpref, over `retrieved_by_rank`.

    `is_relevant` is `RetrievalScorer.score`'s closure over this
    question's gold spans. A chunk that is somehow both relevant (overlaps
    a gold span) and judged-nonrelevant (overlaps a hand-supplied negative
    judgment) is treated as relevant only — R and N are computed as
    disjoint sets by construction (nonrelevant is checked only among
    chunks not already counted as relevant), since a genuinely
    contradictory judgment is a data-quality problem to flag upstream
    (e.g. in a future curation agreement report), not something this
    scorer should silently resolve one way or the other without a trace.
    """
    relevant_ranked = [rc for rc in retrieved_by_rank if is_relevant(rc)]
    nonrelevant_ranked = [
        rc
        for rc in retrieved_by_rank
        if not is_relevant(rc) and _overlaps_any(rc, nonrelevant_spans, overlap_min_chars)
    ]

    r_count = len(relevant_ranked)
    n_count = len(nonrelevant_ranked)
    if r_count == 0:
        return 0.0
    if n_count == 0:
        return 1.0

    denom = min(r_count, n_count)
    total = 0.0
    for r in relevant_ranked:
        n_above = sum(1 for n in nonrelevant_ranked if n.rank < r.rank)
        n_above = min(n_above, denom)
        total += 1.0 - (n_above / denom)
    return total / r_count
