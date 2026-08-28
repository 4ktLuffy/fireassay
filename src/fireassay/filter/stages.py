"""The per-candidate checks behind four of `filter.pipeline.run_filter`'s
five stages (`balance` is inherently cross-candidate state, and lives in
`pipeline.py` itself).

Each `check_*` function returns `(ok, reason)`: `ok=True, reason=None` on a
clean pass, `ok=False, reason=<CODE>` on rejection — mirroring
`mutation.operators.MutationOperator.check_equivalence`'s `(bool, str)`
shape elsewhere in this codebase.
"""

from __future__ import annotations

import re

from fireassay.filter.config import FilterConfig
from fireassay.generate.models import ResolvedCandidate
from fireassay.models import Question
from fireassay.system.bm25 import BM25System
from fireassay.text import tokenize

#: A minimal function-word list used only to detect a degenerate,
#: content-free question (e.g. one that is nothing but "what is the this
#: that") — not a linguistic stopword list for retrieval, and never
#: shared with BM25's own tokenizer path (which does no stopword removal
#: at all; that is a deliberate, separate design choice recorded in
#: system/bm25.py).
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "this", "that", "these", "those", "it", "its", "of", "in", "on", "at", "to",
        "for", "and", "or", "but", "if", "so", "do", "does", "did", "can", "could",
        "should", "would", "will", "shall", "may", "might", "must", "not", "no",
        "with", "as", "by", "from", "about", "into", "than", "then", "there", "here",
    }
)

_DANGLING_PATTERNS = (
    re.compile(
        r"^(what|how|why|when|where|which|who)\s+"
        r"(is|are|was|were|does|do|did|can|could|should|would|will)\s+"
        r"(this|it|that|these|those)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe above\b", re.IGNORECASE),
)


def check_degeneracy(candidate: ResolvedCandidate, config: FilterConfig) -> tuple[bool, str | None]:
    """`DEGENERATE`: question token count outside
    `[min_question_tokens, max_question_tokens]`, or no content word at
    all (every token is a function word / too short to carry meaning)."""
    tokens = tokenize(candidate.text)
    if not (config.min_question_tokens <= len(tokens) <= config.max_question_tokens):
        return False, "DEGENERATE"
    if not any(t not in _STOPWORDS and len(t) >= 3 for t in tokens):
        return False, "DEGENERATE"
    return True, None


def check_self_containment(candidate: ResolvedCandidate) -> tuple[bool, str | None]:
    """`NOT_SELF_CONTAINED`: the question's only subject is a dangling
    referent ("this", "it", "the above", ...) with nothing in the question
    text itself to resolve it against."""
    if any(p.search(candidate.text) for p in _DANGLING_PATTERNS):
        return False, "NOT_SELF_CONTAINED"
    return True, None


def check_near_duplicate(
    candidate: ResolvedCandidate, kept_token_sets: list[frozenset[str]], threshold: float
) -> tuple[bool, str | None]:
    """`NEAR_DUPLICATE`: token-set Jaccard >= `threshold` against any
    already-kept candidate (processed in order, so "kept" means "kept
    earlier in this same filter run" — the earliest of a near-duplicate
    cluster survives, per M3-SPEC.md §3's "keeping the earliest")."""
    this_set = frozenset(tokenize(candidate.text))
    for other_set in kept_token_sets:
        union = this_set | other_set
        if not union:
            continue
        jaccard = len(this_set & other_set) / len(union)
        if jaccard >= threshold:
            return False, "NEAR_DUPLICATE"
    return True, None


def check_unretrievable(
    candidate: ResolvedCandidate, retriever: BM25System, top_n: int
) -> tuple[bool, str | None]:
    """`UNRETRIEVABLE`: the candidate's own source document does not
    appear anywhere in `retriever`'s top `top_n` BM25 results for the
    candidate's own question text, queried over the whole corpus.

    **Replaces `TOO_GENERIC`** (a per-word document-frequency check),
    which measured on the real 1,181-document gov.uk corpus had a ~50%
    false-positive rate: on a topically narrow corpus, ordinary content
    words ("hmrc", "sign", "online", "payment", "cancel") are common
    simply because the *whole corpus* is about government services, so
    "is every content word common" rejects specific, well-formed
    questions. It also misdiagnosed the case it was built for — the
    spike's *"What does the guidance say about revenue and customs?"*
    fails not because its words are individually common, but because
    **nothing in it identifies which document could answer it.**
    Answerability, not per-word rarity, is the actual property that
    matters, and an answerability filter is established practice in
    published synthetic-data pipelines; per-word document frequency is
    not a technique anyone uses. `check_unretrievable` asks the question
    directly, with machinery already in this repo: can the retriever
    that will eventually have to answer this find its own source
    document at all?

    **Two things make filtering with BM25 defensible even though later
    evaluation also uses BM25 configs, and both must hold, not be
    assumed:**

    1. This is a **floor, not a selection criterion** — `top_n` defaults
       to 50 of ~24,584 chunks on the real corpus (~0.2%). It discards
       only candidates BM25 cannot locate at all within a wide margin,
       never candidates it merely ranks imperfectly.
    2. It is applied **identically to every config** at evaluation time,
       so it cannot differentially favour one over another. The bias
       this leaves is on the *curated set's composition*, never on the
       *comparison* between configs.

    **The composition bias is real and must be stated, not hidden: the
    resulting set will under-represent questions that require semantic
    rather than lexical matching** — a paraphrase-heavy question whose
    source document shares little vocabulary with it can fail this check
    even though a semantic retriever would find it easily. When a second
    retriever family (e.g. a dense/embedding retriever) exists in this
    repo, filtering with a *different* one than whichever is under
    evaluation is the proper fix; until then this is a known, documented
    limitation, not a solved problem. See `curate.report`'s rendered
    output and the README for the same disclosure — it belongs wherever
    someone might read the resulting numbers, not only here.

    `retriever` must already be constructed with `top_k == top_n`
    (`filter.pipeline.run_filter` does this once per run, not once per
    candidate) so `retriever.answer(...)` naturally returns exactly the
    window this check needs.
    """
    question = Question(
        text=candidate.text, qtype=candidate.qtype, difficulty=candidate.difficulty, provenance="synthetic"
    )
    output = retriever.answer(question)
    for chunk in output.retrieved:
        if chunk.doc_id == candidate.source_doc_id:
            return True, None
    return False, "UNRETRIEVABLE"
