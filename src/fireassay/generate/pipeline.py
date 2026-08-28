"""End-to-end candidate generation: chunk selection -> LLM call -> content
validation -> span resolution -> lexical features -> persistence.

This is the only pipeline in fireassay that calls an LLM. Every rejection
along the way (`NOT_A_QUESTION`, `QUOTE_NOT_FOUND`, `QUOTE_AMBIGUOUS`) is
persisted as a `filter_result` row exactly like a later filter-pipeline
stage would (`filter.pipeline.run_filter`) — from the funnel's point of
view these are simply the first two stages, computed here because they
depend on the raw LLM output and the source document text, both of which
only `generate/` has in hand. This is what lets `curate report`'s funnel
reconcile exactly: `generated == kept + every rejection reason`, with no
stage's losses uncounted.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from fireassay.generate.lexical import compute_lexical_features
from fireassay.generate.models import CandidateBatch, ResolvedCandidate
from fireassay.generate.prompts import build_prompt
from fireassay.generate.spans import SpanRejection, resolve_span
from fireassay.generate.validate import is_imperative
from fireassay.hashing import content_hash
from fireassay.llm.ollama import ModelRef, OllamaClient
from fireassay.store.db import Store
from fireassay.system.corpus import Chunk, Doc

_PROMPT_HASH_PREFIX = "fa.prompt.1"

#: The two stage names generate/ itself is responsible for recording
#: filter_result rows against — see filter.pipeline.py for the remaining
#: five stages, computed later against already-persisted candidates.
STAGE_NOT_A_QUESTION = "not_a_question"
STAGE_SPAN_RESOLUTION = "span_resolution"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class GenerationStats:
    """Reconciles exactly: `generated == kept + not_a_question +
    quote_not_found + quote_ambiguous`."""

    generated: int
    kept: int
    not_a_question: int
    quote_not_found: int
    quote_ambiguous: int


def select_chunks_for_target(
    chunks: Sequence[Chunk], target_candidates: int, n_per_chunk: int
) -> list[Chunk]:
    """Deterministically select an evenly-spaced subset of `chunks` sized
    to produce approximately `target_candidates` raw candidates at
    `n_per_chunk` each.

    Spread evenly across the **whole** chunk list (`step = len(chunks) /
    num_chunks`, not `chunks[:num_chunks]`) rather than concentrated in the
    first few documents — a `--n 3000` generation run over a ~21k-chunk
    corpus must sample across the corpus for the content-coverage report
    to mean anything, not just describe however many documents happen to
    sort first in the corpus file.
    """
    if n_per_chunk <= 0:
        raise ValueError(f"n_per_chunk must be positive, got {n_per_chunk}")
    if not chunks:
        return []
    num_chunks = max(1, -(-target_candidates // n_per_chunk))  # ceil division
    if num_chunks >= len(chunks):
        return list(chunks)
    step = len(chunks) / num_chunks
    indices = sorted({int(i * step) for i in range(num_chunks)})
    return [chunks[i] for i in indices]


def generate_candidates(
    store: Store,
    client: OllamaClient,
    model: ModelRef,
    docs: Sequence[Doc],
    chunks: Sequence[Chunk],
    *,
    n_per_chunk: int,
    batch_id: str,
) -> GenerationStats:
    """Generate candidates for every chunk in `chunks`, persisting kept
    candidates (`Store.put_candidate`) and every rejection
    (`Store.put_filter_result`) as it goes, and return the batch's
    aggregate `GenerationStats`.

    Per chunk: one `generate_json` call proposes up to `n_per_chunk`
    `generate.models.Candidate`s. Each is checked, in order, against:

    1. `validate.is_imperative` — reject `NOT_A_QUESTION` before ever
       attempting span resolution, since an imperative-shaped `text` is
       unusable regardless of whether its `quote` happens to resolve.
    2. `spans.resolve_span`, against the **document** `chunk.doc_id`
       belongs to (not the chunk's own text — see `spans.py`) — reject
       `QUOTE_NOT_FOUND`/`QUOTE_AMBIGUOUS`.

    A candidate that survives both gets its lexical features measured and
    is persisted as a `ResolvedCandidate`.
    """
    docs_by_id = {doc.doc_id: doc for doc in docs}
    not_a_question = 0
    quote_not_found = 0
    quote_ambiguous = 0
    kept = 0

    for chunk in chunks:
        doc = docs_by_id.get(chunk.doc_id)
        if doc is None:
            raise KeyError(f"chunk {chunk.chunk_id} references unknown doc_id {chunk.doc_id!r}")

        prompt = build_prompt(chunk.text, doc.title, n_per_chunk)
        prompt_hash = content_hash(_PROMPT_HASH_PREFIX, prompt)
        batch = cast(CandidateBatch, client.generate_json(model, prompt, CandidateBatch))

        for raw in batch.candidates:
            candidate_id = uuid.uuid4().hex

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

            features = compute_lexical_features(raw.text, doc.title, raw.quote)
            resolved = ResolvedCandidate(
                id=candidate_id,
                batch_id=batch_id,
                text=raw.text,
                qtype=raw.qtype,
                difficulty=raw.difficulty,
                reference_answer=raw.reference_answer,
                quote=raw.quote,
                source_doc_id=chunk.doc_id,
                char_start=resolution.char_start,
                char_end=resolution.char_end,
                features=features,
                model_digest=model.digest,
                prompt_hash=prompt_hash,
                created_at=_now(),
            )
            store.put_candidate(resolved)
            kept += 1

    generated = kept + not_a_question + quote_not_found + quote_ambiguous
    return GenerationStats(
        generated=generated,
        kept=kept,
        not_a_question=not_a_question,
        quote_not_found=quote_not_found,
        quote_ambiguous=quote_ambiguous,
    )
