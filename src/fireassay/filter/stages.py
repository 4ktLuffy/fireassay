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
from collections.abc import Mapping

from fireassay.filter.config import FilterConfig
from fireassay.generate.models import ResolvedCandidate
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


def check_generic(
    candidate: ResolvedCandidate, doc_freq: Mapping[str, int], n_docs: int, generic_df_pct: float
) -> tuple[bool, str | None]:
    """`TOO_GENERIC`: every content word **that actually appears somewhere
    in the corpus** shows up in more than `generic_df_pct` of documents —
    i.e. not a single corpus-attested word in the question narrows down
    which document(s) it could be about. This is the spike's *"What does
    the guidance say about revenue and customs?"* failure mode: a question
    with no recoverable ground truth because nothing in it discriminates
    the corpus at all.

    **Genericity is a claim about words that exist in the corpus and fail
    to discriminate it.** A content word with `doc_freq.get(w, 0) == 0` is
    excluded from that judgement entirely, never treated as "rare and
    therefore specific" — a word absent from every document discriminates
    nothing, because there is nothing for it to point at. Scoring it as
    maximally specific has it backwards: it would make a question look
    well-targeted precisely when one of its terms doesn't occur anywhere
    in the corpus. Concretely, in *"What does the guidance say about
    revenue and customs?"*, "say" is filler that happens not to appear in
    the fixture corpus at all — it must not single-handedly save the
    question from `TOO_GENERIC` just because it has no document-frequency
    entry to be common *in*.

    A candidate whose content words are all either common-in-corpus or
    absent-from-corpus (no word is both present and rare) still has no
    recoverable ground truth and must be flagged — the same failure this
    check exists for. Only when every content word is genuinely absent
    from the corpus (nothing left to judge genericity against at all) does
    this stage pass by default: that case is an answerability problem
    (the generator introduced a term with no corpus support), which span
    resolution's document-level quote match already guards against
    separately, not a genericity problem for this stage to adjudicate.
    """
    if n_docs <= 0:
        return True, None
    content_words = [t for t in tokenize(candidate.text) if t not in _STOPWORDS and len(t) >= 3]
    if not content_words:
        return True, None
    considered = [w for w in content_words if doc_freq.get(w, 0) > 0]
    if not considered:
        return True, None
    fractions = [doc_freq[w] / n_docs for w in considered]
    if all(f > generic_df_pct for f in fractions):
        return False, "TOO_GENERIC"
    return True, None
