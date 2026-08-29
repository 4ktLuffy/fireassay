"""Build `lexical_decoy` `SeededItem`s (`items.seed.SeedKind`) -- the one
seed kind that needs a retriever, and therefore lives here rather than in
`items/seed.py` (M-ITEMS-SPEC.md's "no fireassay dependency inside
`items/`" rule; see `items.core`'s module docstring and
`adapters/__init__.py`). This is the sanctioned place `items/` meets the
rest of fireassay: this module may import `fireassay.system.*`,
`fireassay.models`, `fireassay.text` -- it must **not** import
`fireassay.store`, which stays `adapters/store.py`'s alone.

**Why relocate evidence at all:** the four corruptions in `items.seed`
cannot produce the ranking inversion (`discrimination_d < 0`) the
mislabel detector looks for -- `swapped_reference`/`negated_reference`
never touch a retrieval-scored matrix cell at all, `foreign_evidence`
makes every retrieval config fail uniformly, and `truncated_question`
degrades every config roughly evenly. A **lexical decoy** relocates an
item's gold evidence into whichever corpus document a BM25 retriever
ranks highest for that item's *question*, excluding the item's own
source document -- broad retrieval configs (large chunks, high `top_k`)
are more likely to reach that decoy; precise configs (small chunks, low
`top_k`) are more likely to return the true topical match instead and
miss it. Whether that asymmetry actually produces the inversion is
exactly what `tools/measure_detector_recall.py` measures -- it is not
assumed here.

**THE TRAP (read before changing anything below):** the detector's rule
*is* `discrimination_d < 0`. Nothing in this module may filter, rank, or
discard a candidate item based on whether its decoy would produce that
inversion, a negative point-biserial, or a `mislabel_suspect`
classification -- doing so would make measured recall 100% by
construction and would measure nothing. `build_lexical_decoy_seeds`
therefore builds each seed by one fixed rule (highest-ranked non-source
chunk, deterministic in `seed`) and emits it unconditionally; a seed is
skipped only when the rule itself cannot be applied at all (no non-source
chunk retrieved, or `items.seed.seeded_item`'s never-emit-unchanged
invariant refuses it) -- never because of how the seed later scores.

**Deliberately unavailable standalone**, for the same reason
counterfactual stability is (`~/ObitosBrain/handbook/eval-of-evals.md`
§7b): picking a decoy means re-running retrieval, which needs the corpus
and the retriever the system under test actually uses, not just a bare
response matrix. `items.core`/`items.seed` work for a stranger who has
never heard of fireassay; this module does not, and does not pretend to.
State the limit rather than hiding it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from fireassay.items.core import ItemMeta
from fireassay.items.seed import SeededItem, seeded_item, stable_seed
from fireassay.models import Question
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Chunk

#: How deep into a question's BM25 ranking `select_decoy_chunk` searches
#: for the highest-ranked non-source chunk before giving up. Every panel
#: config `tools/measure_detector_recall.py` scores against tops out at
#: `top_k=10`; searching to 20 gives the decoy some room to sit below the
#: very top of the ranking (so it is not trivially found by every config)
#: while still being within reach of the broadest ones -- not a claim
#: about any specific config's actual recall, which is exactly the thing
#: left to be measured, not assumed here.
DEFAULT_TOP_K = 20


@dataclass(frozen=True)
class DecoyChunk:
    """The chunk `select_decoy_chunk` picked for one item, plus how it
    ranked. `items.seed.SeededItem` -- a fixed shape shared with the
    other four corruption kinds -- has no char-offset fields, so a caller
    that needs the decoy's exact character range for scoring (e.g.
    `tools/measure_detector_recall.py`, which must know exactly where in
    the decoy document the relocated evidence sits, not merely which
    document) calls `select_decoy_chunk` again rather than parsing it back
    out of a `SeededItem`. Selection is a pure function of `(item, chunks,
    top_k)`, so a caller that passes the same three values back gets
    byte-identical results to whatever `build_lexical_decoy_seeds` picked."""

    chunk: Chunk
    rank: int
    score: float


def select_decoy_chunk(item: ItemMeta, chunks: Sequence[Chunk], *, top_k: int = DEFAULT_TOP_K) -> DecoyChunk:
    """Retrieve `chunks` against `item.question` with the project's BM25
    (`fireassay.system.bm25.BM25System` -- no new scorer, no tokenizing
    regex of our own; `fireassay.text.tokenize` is the only tokenizer, and
    `BM25System` already uses it) and return the highest-ranked chunk
    among the top `top_k` whose `doc_id` differs from `item.source_doc_id`
    -- the decoy.

    Raises `ValueError` if `item.question` or `item.source_doc_id` is
    missing (there is no question to retrieve against, or no source
    document to exclude), or if every one of the top `top_k` retrieved
    chunks belongs to `item.source_doc_id` -- i.e. no non-source chunk was
    retrieved at all. Never pads and never substitutes a chunk chosen some
    other way: that would turn this into `foreign_evidence` wearing a
    different name."""
    if not item.question:
        raise ValueError(f"select_decoy_chunk: item {item.item_id!r} has no question to retrieve against")
    if item.source_doc_id is None:
        raise ValueError(f"select_decoy_chunk: item {item.item_id!r} has no source_doc_id to exclude")

    system = BM25System(chunks, top_k=top_k)
    question = Question(
        text=item.question,
        qtype="factual",
        difficulty="easy",
        reference_answer=None,
        provenance="synthetic",
        generator="lexical_decoy@1",
    )
    output = system.answer(question)
    chunk_by_key = {(c.doc_id, c.chunk_id): c for c in chunks}
    for rc in output.retrieved:
        if rc.doc_id != item.source_doc_id:
            return DecoyChunk(chunk=chunk_by_key[(rc.doc_id, rc.chunk_id)], rank=rc.rank, score=rc.score)
    raise ValueError(
        f"select_decoy_chunk: item {item.item_id!r}: no non-source chunk among the top "
        f"{top_k} retrieved for its question -- every candidate belonged to its own "
        f"source_doc_id {item.source_doc_id!r}"
    )


def build_lexical_decoy_seeds(
    items: Sequence[ItemMeta],
    chunks: Sequence[Chunk],
    *,
    n: int,
    seed: int,
    top_k: int = DEFAULT_TOP_K,
) -> list[SeededItem]:
    """Deterministically choose up to `n` items from `items` (order
    derived from `seed`, no item statistic consulted -- see THE TRAP in
    the module docstring) and build a `lexical_decoy` `SeededItem` for
    each: `evidence_quote`/`source_doc_id` come from
    `select_decoy_chunk`'s decoy; `question`/`reference_answer` are left
    exactly as `item`'s own, so the question stays answerable-looking and
    only the evidence moves.

    An item is skipped -- not substituted, not padded -- when
    `select_decoy_chunk` cannot find a non-source chunk for it, or when
    `items.seed.seeded_item`'s never-emit-unchanged invariant refuses the
    result (mirrors `items.seed.seed_batch`'s per-item skip-and-continue
    contract for the other four kinds). This is the *only* reason a seed
    is ever left out: no seed is filtered because of how it later scores
    against any detector.

    `top_k` is passed straight through to `select_decoy_chunk` for every
    item tried."""
    candidates = list(items)
    random.Random(stable_seed(seed, "lexical_decoy")).shuffle(candidates)

    out: list[SeededItem] = []
    for item in candidates:
        if len(out) >= n:
            break
        try:
            decoy = select_decoy_chunk(item, chunks, top_k=top_k)
        except ValueError:
            continue
        detail = (
            f"evidence relocated to doc {decoy.chunk.doc_id!r} (BM25 rank {decoy.rank} of "
            f"the top {top_k} retrieved for this item's own question, score "
            f"{decoy.score:.4f}) -- the highest-ranked chunk from a document other than "
            f"this item's own source_doc_id {item.source_doc_id!r}"
        )
        try:
            out.append(
                seeded_item(
                    item,
                    kind="lexical_decoy",
                    question=item.question,
                    reference_answer=item.reference_answer,
                    evidence_quote=decoy.chunk.text,
                    source_doc_id=decoy.chunk.doc_id,
                    detail=detail,
                )
            )
        except ValueError:
            continue
    return out
