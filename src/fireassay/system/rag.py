"""`RagSystem` -- the first system in the project for which
`generates_answers` is `True`.

Every system before this one (`BM25System`, `CoverageSystem`, `TfidfSystem`)
retrieves only: `generates_answers = False`, and `answer=None`/
`abstained=True` on every call (see e.g. `bm25.BM25System.answer`'s
docstring). That makes `score.abstention.AbstentionScorer` structurally
inert project-wide -- see its module docstring: both its metrics "are
suppressed entirely -- not applicable, not zeroed -- when
`ctx.system_generates_answers` is `False`." `RagSystem` is what turns it
on: it wraps any retriever `System` with an injected model call that
produces an actual natural-language answer (or a deliberate abstention),
so a judge finally has something to score against a reference answer.

**The model is injected, never imported.** `Generate = Callable[[str], str]`
mirrors `items.answerability.Ask` exactly: a prompt in, a completion out.
This module must not import `fireassay.llm` or wire up Ollama itself --
`system.adapters.ollama_generate` is the thin factory that does that.
Keeping the boundary here is what lets both the retriever (bm25/coverage/
tfidf, or anything else satisfying `System`) and the generator be swapped
-- including for a fake in a hermetic test -- without ever touching this
file.

**Chunk text.** `RetrievedChunk` (models.py) deliberately carries no `text`
field -- only `doc_id`/`chunk_id`/`char_start`/`char_end`, the identity a
scorer needs, not the payload a prompt needs. Widening the `System`
protocol so every retriever carried a corpus accessor was rejected:
`system.base.System` is kept deliberately tiny so adding a system never
requires touching the runner or the scorers, and only `RagSystem` needs
chunk text. So `RagSystem` takes the `chunks` its retriever was built from
directly, and builds a `{(doc_id, chunk_id): text}` lookup once, in
`__init__` -- the caller already has them, since it needed them to
construct the retriever in the first place (`system.panel.
build_chunks_by_config` hands them out exactly this way).

**A retriever and a `chunks` sequence that disagree fail loudly, not
silently.** Nothing stops a caller passing chunks from a different
chunking than the retriever was actually built on (e.g. a 512/32 retriever
paired with 1024/128 chunks): retrieval would still succeed, but every
`chunk_id` would then either miss the lookup or -- worse -- collide with an
unrelated chunk, and the model would be handed text that was never
actually retrieved, while every downstream number kept looking healthy.
`_resolve_text` raises `ValueError` naming the offending `(doc_id,
chunk_id)` the moment that happens, rather than silently skipping the
chunk or substituting empty text.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence

from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.system.base import System
from fireassay.system.corpus import Chunk

#: Prompt in, raw completion out -- the seam that keeps this module free of
#: any LLM client import (see the module docstring), mirroring
#: `items.answerability.Ask`.
Generate = Callable[[str], str]

#: The model must reply with exactly this string to abstain (Change 2's
#: prompt instructs it to). See `_SENTINEL_PATTERN` for how a chunk that
#: quotes this string verbatim in CONTEXT is prevented from forcing an
#: abstention the model did not itself choose.
_SENTINEL = "INSUFFICIENT_CONTEXT"

# Greedy last-match -- the same defence `items.answerability._VERDICT_PATTERN`
# uses, and for the same reason: CONTEXT is gov.uk corpus text injected into
# the prompt verbatim, so a chunk that happens to *contain* the literal
# string `INSUFFICIENT_CONTEXT` must not be able to force an abstention on
# its own. A leading `.*` under DOTALL is greedy, so it consumes the whole
# completion first and backtracks only as far as it must -- landing this
# check on whether the sentinel is the model's own *last* word (nothing but
# trailing whitespace after it), not merely present somewhere earlier. If
# the completion instead ends with real answer text, this does not match,
# and the whole completion becomes the answer (Change 3, "otherwise").
_SENTINEL_PATTERN = re.compile(r".*" + re.escape(_SENTINEL) + r"\s*\Z", re.DOTALL)


def build_prompt(question: Question, chunk_texts: Sequence[str]) -> str:
    """The RAG answer prompt (Change 2) -- pinned in one place so it can be
    reviewed and reused, since it produces every answer the project will
    ever judge.

    Numbers each chunk (1-based) so CONTEXT is unambiguous about how many
    chunks were retrieved, then presents QUESTION last, closest to the
    instruction that follows it.
    """
    context_block = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(chunk_texts, start=1))
    return (
        "Answer the QUESTION using ONLY the CONTEXT below. Do not use any outside "
        "knowledge, and do not guess.\n"
        "\n"
        f"CONTEXT:\n{context_block}\n"
        "\n"
        f"QUESTION:\n{question.text}\n"
        "\n"
        "Give a short, direct answer -- a few words or a short sentence, not a "
        "paragraph.\n"
        "\n"
        f"If the CONTEXT does not contain the answer, reply with exactly "
        f"`{_SENTINEL}` and nothing else. Do not hedge and do not write "
        f"\"not sure\" or similar -- reply with that exact sentinel, or with an "
        "answer.\n"
    )


def _parse_completion(raw: str) -> tuple[str | None, bool]:
    """Parse a raw completion into `(answer, abstained)` (Change 3).

    - Empty or whitespace-only -> `(None, True)`: a scorer cannot tell an
      empty string apart from a non-answer, and would grade `""` as simply
      wrong rather than as a non-answer -- the entire reason
      `score.abstention` exists.
    - The sentinel, matched by greedy last-match (see `_SENTINEL_PATTERN`)
      -> `(None, True)`.
    - Otherwise -> `(raw.strip(), False)`.
    """
    stripped = raw.strip()
    if not stripped:
        return None, True
    if _SENTINEL_PATTERN.search(stripped) is not None:
        return None, True
    return stripped, False


class RagSystem:
    """Wraps any retriever `System` with an injected `Generate` call to
    produce an actual answer -- the first system in the project for which
    `generates_answers` is `True` (see the module docstring).

    `chunks` must be the same chunk collection `retriever` was built from;
    see the module docstring's "fail loudly" section and `_resolve_text`.
    """

    def __init__(
        self,
        retriever: System,
        generate: Generate,
        chunks: Sequence[Chunk],
        *,
        name: str = "rag",
    ) -> None:
        self.name = name
        # The first system in the project for which this is True -- see
        # the module docstring for what that activates.
        self.generates_answers = True
        self.retriever = retriever
        self.generate = generate
        self._chunk_text: dict[tuple[str, str], str] = {
            (chunk.doc_id, chunk.chunk_id): chunk.text for chunk in chunks
        }

    def _resolve_text(self, chunk: RetrievedChunk) -> str:
        """Look up the text of a retrieved chunk in the `chunks` collection
        this `RagSystem` was constructed with, raising `ValueError` --
        never skipping the chunk, never substituting empty text -- if it is
        absent (see the module docstring's "fail loudly" section)."""
        key = (chunk.doc_id, chunk.chunk_id)
        if key not in self._chunk_text:
            raise ValueError(
                f"RagSystem {self.name!r}: retriever {self.retriever.name!r} returned chunk "
                f"(doc_id={chunk.doc_id!r}, chunk_id={chunk.chunk_id!r}) that is not present in "
                "the chunks this RagSystem was constructed with -- the retriever and the "
                "supplied chunks disagree (e.g. built from different chunking configs), so the "
                "model would be handed text that was never actually retrieved. Pass the exact "
                "chunks the retriever was built from."
            )
        return self._chunk_text[key]

    def answer(self, question: Question) -> SystemOutput:
        """Retrieve, then generate.

        Delegates retrieval to `self.retriever.answer(question)` and keeps
        its `retrieved` tuple on the returned `SystemOutput` unchanged, so
        every retrieval metric (`RetrievalScorer` etc.) still measures the
        retriever exactly as it would standalone -- `RagSystem` adds a
        generation stage on top of it, it does not change what was
        retrieved.

        `latency_ms` carries `total` (required by `SystemOutput`, see that
        model's validator) plus `retrieval` and `generation` stage keys, so
        the two are separable. **All three are closed-loop wall time under
        this sequential runner** (M1 runs one question at a time -- see
        `system.base.System`'s docstring): they measure how long this one
        call took end-to-end on this machine, not a service's latency
        distribution under concurrent load. Per defect 13 (coordinated
        omission), a p99 read off a series of these numbers is wrong by up
        to 25x -- a closed-loop client never samples the tail an open-loop
        (Poisson-arrival) client would see, because it only ever issues its
        next request after the previous one has already returned. Report
        these as closed-loop wall-clock numbers; never as a production
        latency percentile.
        """
        start_total = time.perf_counter()

        start_retrieval = time.perf_counter()
        retrieval_output = self.retriever.answer(question)
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

        chunk_texts = [self._resolve_text(chunk) for chunk in retrieval_output.retrieved]
        prompt = build_prompt(question, chunk_texts)

        start_generation = time.perf_counter()
        completion = self.generate(prompt)
        generation_ms = (time.perf_counter() - start_generation) * 1000

        answer, abstained = _parse_completion(completion)

        total_ms = (time.perf_counter() - start_total) * 1000

        return SystemOutput(
            answer=answer,
            abstained=abstained,
            retrieved=retrieval_output.retrieved,
            latency_ms={"total": total_ms, "retrieval": retrieval_ms, "generation": generation_ms},
        )
