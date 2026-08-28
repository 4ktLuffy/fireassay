"""End-to-end candidate generation: target-cell assignment -> LLM call ->
content validation -> span resolution -> feature measurement -> persistence.

This is the only pipeline in fireassay that calls an LLM. Every rejection
along the way (`GENERATION_FAILED`, `NOT_A_QUESTION`, `QUOTE_NOT_FOUND`,
`QUOTE_AMBIGUOUS`) is persisted as a `filter_result` row exactly like a
later filter-pipeline stage would (`filter.pipeline.run_filter`) — from
the funnel's point of view these are simply the first three stages,
computed here because they depend on the raw LLM call, its output, and
the source document text, all of which only `generate/` has in hand. This
is what lets `curate report`'s funnel reconcile exactly:
`generated == kept + every rejection reason`, with no stage's losses
uncounted.

**Generation is stratified across the taxonomy, not left to the model's
choice.** An unstratified prompt ("produce N questions of whatever type
you like") was measured to collapse to ~2 of 21 `(qtype, difficulty)`
cells (`factual`/`easy` and `procedural`/`medium` dominating) — a model
left free defaults to easy factual questions. At 3,000 candidates that
would have meant the `balance` stage capping `factual`/`easy` at
`cell_cap` and discarding thousands of near-duplicate questions, a
monotonous surviving set, and `multi_hop` (the type retrieval quality
actually differentiates configs on) entirely absent. So every request now
names an explicit target cell; see `_TARGET_CELLS` for which 12 and why
only those 12.

**`gold_doc_rank` is measured here, once, at insert time — not
recomputed later.** `candidate` rows are append-only (migration 0003's
triggers forbid `UPDATE`), so a retrieval-based feature can only ever be
stored if it is known before the row is first written. A BM25 index over
the whole corpus is built once per `generate` run (not once per
candidate) and every candidate's own source document is looked up against
it; see `CandidateFeatures.gold_doc_rank`'s docstring for what the number
means and `filter.stages.check_unretrievable` for the independent,
filter-time recomputation of the same kind of measurement (deliberately
independent, not read from here — see that function's docstring for why).

**Known, unaddressed hazard:** `generate` and `filter` can be pointed at
different corpora (via independent `--corpus` flags) and nothing here
notices — a `gold_doc_rank` measured against one corpus and a later
`UNRETRIEVABLE` verdict measured against a different one would silently
disagree about the same candidate for a reason that has nothing to do
with retrieval quality. The real fix (recording the `corpus_hash` a rank
was measured against, and refusing a mismatch) belongs with the M5 gate
work, where corpus identity is already enforced end to end; noted here so
it is not lost before then. In the meantime: use one corpus file
throughout a run.

**A single generation call exhausting its retries must not abort the
whole run.** `OllamaClient.generate_json` is right to raise
`GenerationExhaustedError` rather than return a partial result — that
rule is about *one call*, which genuinely failed and must not pretend
otherwise. But some `(chunk, target cell)` pairs are simply impossible:
asking for `comparative`/`hard` from a passage with nothing to compare
gives the model very little to work with, and it flails on every retry.
Measured live: intermittent, roughly one bad pairing in several dozen,
and it is not a bug in the prompt or the model each time it happens — a
non-zero `GENERATION_FAILED` count in a funnel is expected, not
necessarily a sign anything is broken. What made this catastrophic
before the fix below is that the cache faithfully replays every prompt
already attempted, so a restart after an uncaught abort walked straight
back to the same impossible pairing and died again — a wall a run could
never get past, not a recoverable crash. `generate_candidates` now
catches `GenerationExhaustedError` **per attempt**, records it as a
`GENERATION_FAILED` discard exactly like a failed span resolution, and
continues — one impossible pairing costs one candidate, never the run.
`max_generation_failure_rate` is the backstop for the genuinely different
case: a bad model, a bad prompt, or a dead server failing on *most*
attempts, where grinding through thousands of doomed calls would be its
own kind of silent failure.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from fireassay.generate.lexical import compute_candidate_features
from fireassay.generate.models import CandidateBatch, ResolvedCandidate
from fireassay.generate.prompts import build_prompt
from fireassay.generate.spans import SpanRejection, resolve_span
from fireassay.generate.validate import is_imperative
from fireassay.hashing import content_hash
from fireassay.llm.ollama import GenerationExhaustedError, ModelRef, OllamaClient
from fireassay.models import Difficulty, QType, Question
from fireassay.store.db import Store
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk, Doc, chunk_corpus

_PROMPT_HASH_PREFIX = "fa.prompt.1"

#: The three stage names generate/ itself is responsible for recording
#: filter_result rows against — see filter.pipeline.py for the remaining
#: five stages, computed later against already-persisted candidates.
STAGE_GENERATION = "generation"
STAGE_NOT_A_QUESTION = "not_a_question"
STAGE_SPAN_RESOLUTION = "span_resolution"

#: Default fraction of generation attempts allowed to exhaust their
#: retries before `generate_candidates` aborts the whole run rather than
#: grinding through the rest. A few impossible `(chunk, target cell)`
#: pairs are normal; this many failing means something is actually
#: wrong -- a bad model, a bad prompt, or a dead server.
DEFAULT_MAX_GENERATION_FAILURE_RATE = 0.2
#: How many attempts to make before the failure-rate check applies at
#: all -- otherwise a single unlucky early failure (100% of 1 attempt)
#: would abort a run that was never actually in trouble. Not exposed on
#: the CLI: it protects the check's own statistics, not a tuning knob a
#: user needs, though tests override it to exercise the check cheaply.
DEFAULT_MIN_ATTEMPTS_BEFORE_FAILURE_RATE_CHECK = 20


class GenerationFailureRateExceededError(Exception):
    """Raised by `generate_candidates` when more than
    `max_generation_failure_rate` of generation attempts exhaust their
    retries (`GenerationExhaustedError`) — the backstop for "a bad model,
    a bad prompt, or a dead server", distinct from the individually
    expected, individually harmless `GENERATION_FAILED` discards a few
    genuinely impossible `(chunk, target cell)` pairs produce. Aborts
    rather than spending the rest of a run grinding through calls that
    are, on the evidence so far, mostly doomed."""

#: Chunking used to build the one-off BM25 index `gold_doc_rank` is
#: measured against — matches the defaults `cli.py` already uses
#: elsewhere for the same corpus (`run`, `controls run`); not (yet)
#: configurable here, same trade `filter.pipeline` makes for its own
#: retrieval index.
_GOLD_RANK_CHUNK_SIZE = 800
_GOLD_RANK_CHUNK_OVERLAP = 100
#: How deep to search for the candidate's own document. Generous on
#: purpose: this is a *reported* difficulty signal, not a pass/fail gate
#: (that is `filter.stages.check_unretrievable`'s job, at its own,
#: separately-configured depth), so erring toward "still findable, just
#: deep" rather than truncating early is the right default.
_GOLD_RANK_SEARCH_DEPTH = 100

#: The taxonomy cells generation is stratified over in M3 — a deliberate
#: 12 of the schema's full 7 x 3 = 21, not all of them:
#:
#: - `multi_hop` is excluded: it needs two or more chunks by definition
#:   (a question that requires combining information from several
#:   passages), and this pipeline prompts with exactly one chunk at a
#:   time. Multi-chunk prompting is a coherent later milestone, not a
#:   difficulty knob on top of the current single-chunk design.
#: - `unanswerable` is excluded: it cannot be generated from a passage
#:   that contains the answer, almost by definition. It needs a dedicated
#:   generation mode over topics *adjacent to but absent from* the corpus
#:   -- a different prompt shape, not a cell in this grid.
#:   **M4 needs these for abstention evaluation; this dependency must not
#:   be forgotten when M4 is scoped.**
#: - `ambiguous` is excluded: deliberately under-specifying a question is
#:   a distinct generation task in its own right (the model must be asked
#:   to remove information, not just told to write an "ambiguous" one),
#:   not a difficulty setting on an otherwise normal question.
_TARGET_QTYPES: tuple[QType, ...] = ("factual", "procedural", "comparative", "policy_sensitive")
_TARGET_DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
_TARGET_CELLS: tuple[tuple[QType, Difficulty], ...] = tuple(
    (qtype, difficulty) for qtype in _TARGET_QTYPES for difficulty in _TARGET_DIFFICULTIES
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_seed(*parts: object) -> int:
    """A stable integer seed derived from `parts` via sha256 -- not
    Python's randomised-per-process builtin `hash()`. Deliberately
    duplicated (not imported) from `controls._common.stable_seed`/
    `curate._common.stable_seed`: each of those is private to its own
    package, and this one-line, frozen contract is cheaper to repeat than
    to import across a package boundary that otherwise has no reason to
    exist (see `curate._common`'s own docstring for the same trade)."""
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def target_cell_for(chunk_id: str, index_in_chunk: int) -> tuple[QType, Difficulty]:
    """The `(qtype, difficulty)` cell to request for the `index_in_chunk`-th
    candidate drawn from `chunk_id`.

    Deterministic and **seeded from the chunk id**, not from list
    position: re-running generation over the same chunk always requests
    the same cell for it regardless of what else is in the batch (a
    different `--n`/subset of chunks elsewhere in the run does not
    silently reshuffle this chunk's own assignment). `index_in_chunk`
    (0, 1, 2, ... for the 1st, 2nd, 3rd candidate requested from the same
    chunk when `n_per_chunk > 1`) advances the cycle from that per-chunk
    starting point, so multiple candidates from one chunk spread across
    different cells rather than repeating the same one.

    Across many chunks (thousands, at the scale this pipeline runs at),
    `sha256`-derived starting points land close to uniformly across the
    12 cells — deterministic and bias-free "by construction", in the
    sense of never depending on what the model feels like proposing,
    though not a mathematically exact round-robin the way a raw position
    counter would be. A raw counter was deliberately not used: it would
    make every chunk's target depend on which other chunks happen to be
    selected alongside it in this run, which is a worse property for
    reproducibility than the small amount of exactness this trades away.
    """
    offset = _stable_seed(chunk_id) % len(_TARGET_CELLS)
    return _TARGET_CELLS[(offset + index_in_chunk) % len(_TARGET_CELLS)]


@dataclass(frozen=True)
class GenerationStats:
    """Reconciles exactly: `generated == kept + generation_failed +
    not_a_question + quote_not_found + quote_ambiguous`.

    `resumed` (M3b-SPEC.md Part 1) is deliberately **not** part of that
    reconciliation: a resumed chunk was skipped before any attempt was
    made against it this run — it contributes no `filter_result` row this
    run at all (it already has one from whichever earlier run actually
    generated it), so counting it into `generated` would double-count work
    done in a previous run. It is reported purely so a resumed run can say
    what it skipped (`resumed=N chunks already present`)."""

    generated: int
    kept: int
    generation_failed: int
    not_a_question: int
    quote_not_found: int
    quote_ambiguous: int
    resumed: int = 0


_CHUNK_ORDER_PREFIX = "fa.chunkorder.1"


def _chunk_order_key(chunk: Chunk) -> str:
    return content_hash(_CHUNK_ORDER_PREFIX, chunk.chunk_id)


def select_chunks_for_target(
    chunks: Sequence[Chunk], target_candidates: int, n_per_chunk: int
) -> list[Chunk]:
    """Deterministically select a subset of `chunks` sized to produce
    approximately `target_candidates` raw candidates at `n_per_chunk`
    each.

    **The selection for a larger `target_candidates` is a strict prefix of
    the selection for a smaller one, for the same `chunks`/`n_per_chunk`.**
    All chunks are ranked once by `content_hash(chunk.chunk_id)` — a fixed,
    corpus-file-order-independent shuffle — sorted, and the first
    `num_chunks` are taken. This is what makes topping up a batch cheap: a
    generation run over `--n 4000` that came up short and gets re-run at
    `--n 5000` reuses every prompt the first run already paid for and
    cached, instead of re-striding the whole corpus and missing cache on
    almost everything.

    An earlier version instead strode evenly across the chunk list
    (`step = len(chunks) / num_chunks`) — deterministic for a *fixed* `n`,
    but the stride itself depends on `n`, so changing `n` changes almost
    every selected index and therefore almost every prompt. Measured on a
    real run: `--n 12` then `--n 16` against the same warm cache reused
    only 4 of 16 prompts (95s instead of ~35s) — for an 8-hour run that
    difference is the entire point of caching. The hash-order selection
    keeps the "spread across the whole corpus, not just its first few
    documents" property the even-stride approach was for — a good hash
    distributes chunk_ids across the corpus regardless of which document
    they came from — without coupling the selection to `n`.
    """
    if n_per_chunk <= 0:
        raise ValueError(f"n_per_chunk must be positive, got {n_per_chunk}")
    if not chunks:
        return []
    num_chunks = max(1, -(-target_candidates // n_per_chunk))  # ceil division
    ordered = sorted(chunks, key=_chunk_order_key)
    return ordered[:num_chunks]


def _build_gold_rank_retriever(docs: Sequence[Doc]) -> BM25System | None:
    """One BM25 index over the whole corpus, or `None` if `docs` is empty
    (nothing to measure `gold_doc_rank` against — every candidate then
    simply gets `gold_doc_rank=None`, never a crash)."""
    if not docs:
        return None
    retrieval_chunks = list(chunk_corpus(docs, size=_GOLD_RANK_CHUNK_SIZE, overlap=_GOLD_RANK_CHUNK_OVERLAP))
    return BM25System(retrieval_chunks, top_k=_GOLD_RANK_SEARCH_DEPTH)


def _gold_doc_rank(
    retriever: BM25System | None, query_text: str, source_doc_id: str, qtype: QType, difficulty: Difficulty
) -> int | None:
    """The 1-based rank at which `source_doc_id` first appears when
    `retriever` is queried with `query_text`, or `None` if it never
    appears within `retriever`'s own `top_k` (i.e. `_GOLD_RANK_SEARCH_DEPTH`)
    or `retriever` is `None`.

    Deliberately duplicated (not imported) from `filter.stages`'s own
    rank lookup: `generate/` and `filter/` are independent packages by
    design (`filter/` never imports from `generate/pipeline`, and this
    module must not start a dependency the other direction either), and
    the two computations are independent measurements on purpose — see
    `filter.stages.check_unretrievable`'s docstring.
    """
    if retriever is None:
        return None
    question = Question(text=query_text, qtype=qtype, difficulty=difficulty, provenance="synthetic")
    output = retriever.answer(question)
    for chunk in output.retrieved:
        if chunk.doc_id == source_doc_id:
            return chunk.rank
    return None


def generate_candidates(
    store: Store,
    client: OllamaClient,
    model: ModelRef,
    docs: Sequence[Doc],
    chunks: Sequence[Chunk],
    *,
    n_per_chunk: int,
    batch_id: str,
    max_generation_failure_rate: float = DEFAULT_MAX_GENERATION_FAILURE_RATE,
    min_attempts_before_failure_rate_check: int = DEFAULT_MIN_ATTEMPTS_BEFORE_FAILURE_RATE_CHECK,
) -> GenerationStats:
    """Generate candidates for every chunk in `chunks`, persisting kept
    candidates (`Store.put_candidate`) and every rejection
    (`Store.put_filter_result`) as it goes, and return the batch's
    aggregate `GenerationStats`.

    Per chunk: `n_per_chunk` separate `generate_json` calls, one per
    target cell (`target_cell_for`), each requesting exactly one
    `generate.models.Candidate` for that specific `(qtype, difficulty)` —
    stratified, not "produce `n_per_chunk` questions of whatever type you
    like" (see this module's docstring for why that collapsed generation
    to ~2 of 21 taxonomy cells). Each attempt is checked, in order,
    against:

    0. The call itself: `client.generate_json` can raise
       `GenerationExhaustedError` (every retry produced invalid JSON) —
       caught here and recorded as `GENERATION_FAILED`, never left to
       propagate and abort the run (see this module's docstring for why
       that distinction matters). If more than
       `max_generation_failure_rate` of attempts fail once at least
       `min_attempts_before_failure_rate_check` have been made, raises
       `GenerationFailureRateExceededError` instead of continuing.
    1. `validate.is_imperative` — reject `NOT_A_QUESTION` before ever
       attempting span resolution, since an imperative-shaped `text` is
       unusable regardless of whether its `quote` happens to resolve.
    2. `spans.resolve_span`, against the **document** `chunk.doc_id`
       belongs to (not the chunk's own text — see `spans.py`) — reject
       `QUOTE_NOT_FOUND`/`QUOTE_AMBIGUOUS`.

    A candidate that survives all three gets its features measured
    (lexical overlap plus `gold_doc_rank`, via one BM25 index built once
    over `docs` — see `_build_gold_rank_retriever`) and is persisted as a
    `ResolvedCandidate`, carrying both the target cell it was asked for
    and the model's own proposed `qtype`/`difficulty` (which may disagree
    — that disagreement rate is `curate.coverage.generator_obedience`).

    **Incremental resume (M3b-SPEC.md Part 1):** before touching any
    chunk, `store.candidate_source_chunks()` is read once to get the set
    of chunk ids `store` already has at least one `candidate` row for.
    Any chunk in that set is skipped entirely — no `generate_json` call,
    no cache lookup, no `filter_result` row — and counted in
    `GenerationStats.resumed` instead. This is the fix for two problems
    measured on a real run: re-running `generate` into an existing
    database used to mint a fresh uuid4 candidate id per chunk every time,
    duplicating candidates (5 became 10 across two batches); and replaying
    an already-cached chunk into a *fresh* database still cost ~1s/chunk
    with zero model calls (an hour at 4,000 chunks) — a cost this skip
    removes outright for any chunk already present, rather than trying to
    make the replay itself faster (that would need profiling this
    environment could not do; see M3b-SPEC.md Part 1 and the delivery
    notes for what was and was not measured).
    """
    docs_by_id = {doc.doc_id: doc for doc in docs}
    retriever = _build_gold_rank_retriever(docs)
    already_present = store.candidate_source_chunks()
    generation_failed = 0
    not_a_question = 0
    quote_not_found = 0
    quote_ambiguous = 0
    kept = 0
    resumed = 0
    total_attempts = 0

    for chunk in chunks:
        if chunk.chunk_id in already_present:
            resumed += 1
            continue

        doc = docs_by_id.get(chunk.doc_id)
        if doc is None:
            raise KeyError(f"chunk {chunk.chunk_id} references unknown doc_id {chunk.doc_id!r}")

        for index_in_chunk in range(n_per_chunk):
            target_qtype, target_difficulty = target_cell_for(chunk.chunk_id, index_in_chunk)
            prompt = build_prompt(chunk.text, doc.title, target_qtype, target_difficulty)
            prompt_hash = content_hash(_PROMPT_HASH_PREFIX, prompt)
            total_attempts += 1

            try:
                batch = cast(CandidateBatch, client.generate_json(model, prompt, CandidateBatch))
            except GenerationExhaustedError as exc:
                generation_failed += 1
                store.put_filter_result(uuid.uuid4().hex, STAGE_GENERATION, False, "GENERATION_FAILED")
                if total_attempts >= min_attempts_before_failure_rate_check:
                    failure_rate = generation_failed / total_attempts
                    if failure_rate > max_generation_failure_rate:
                        raise GenerationFailureRateExceededError(
                            f"generation failure rate {failure_rate:.1%} "
                            f"({generation_failed}/{total_attempts} attempts) exceeds "
                            f"max_generation_failure_rate={max_generation_failure_rate:.1%}; "
                            "aborting rather than grinding through likely-doomed calls. "
                            f"Most recent failure: {exc}"
                        ) from exc
                continue

            for raw in batch.candidates:
                candidate_id = uuid.uuid4().hex
                store.put_filter_result(candidate_id, STAGE_GENERATION, True, None)

                if is_imperative(raw.text):
                    not_a_question += 1
                    store.put_filter_result(candidate_id, STAGE_NOT_A_QUESTION, False, "NOT_A_QUESTION")
                    continue
                store.put_filter_result(candidate_id, STAGE_NOT_A_QUESTION, True, None)

                resolution = resolve_span(raw.quote, doc.text)
                if isinstance(resolution, SpanRejection):
                    if resolution.reason == "QUOTE_NOT_FOUND":
                        quote_not_found += 1
                    else:
                        quote_ambiguous += 1
                    store.put_filter_result(
                        candidate_id, STAGE_SPAN_RESOLUTION, False, resolution.reason
                    )
                    continue
                store.put_filter_result(candidate_id, STAGE_SPAN_RESOLUTION, True, None)

                rank = _gold_doc_rank(retriever, raw.text, chunk.doc_id, raw.qtype, raw.difficulty)
                features = compute_candidate_features(raw.text, doc.title, raw.quote, gold_doc_rank=rank)
                resolved = ResolvedCandidate(
                    id=candidate_id,
                    batch_id=batch_id,
                    text=raw.text,
                    qtype=raw.qtype,
                    difficulty=raw.difficulty,
                    target_qtype=target_qtype,
                    target_difficulty=target_difficulty,
                    reference_answer=raw.reference_answer,
                    quote=raw.quote,
                    source_doc_id=chunk.doc_id,
                    char_start=resolution.char_start,
                    char_end=resolution.char_end,
                    features=features,
                    model_digest=model.digest,
                    prompt_hash=prompt_hash,
                    chunk_id=chunk.chunk_id,
                    created_at=_now(),
                )
                store.put_candidate(resolved)
                kept += 1

    generated = kept + generation_failed + not_a_question + quote_not_found + quote_ambiguous
    return GenerationStats(
        generated=generated,
        kept=kept,
        generation_failed=generation_failed,
        not_a_question=not_a_question,
        quote_not_found=quote_not_found,
        quote_ambiguous=quote_ambiguous,
        resumed=resumed,
    )
