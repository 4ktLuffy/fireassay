"""Candidate feature measurement (M3-SPEC.md §2) — the lexical features via
`fireassay.text.tokenize` and nothing else (per M1 §6's "one tokenizer"
rule: a dedup/feature pass using a second, subtly different tokenizer would
disagree with the retriever and the filter's own near-duplicate stage for
reasons that have nothing to do with what is actually being measured), plus
one retrieval-based feature (`gold_doc_rank`) computed by the caller via
BM25 and passed straight through — this module does not build a retriever
itself, since `generate.pipeline` already builds one once per run rather
than once per candidate.
"""

from __future__ import annotations

from fireassay.generate.models import CandidateFeatures
from fireassay.text import tokenize


def compute_candidate_features(
    question_text: str, title: str, quote: str, *, gold_doc_rank: int | None
) -> CandidateFeatures:
    """`title_overlap = |tokens(question) & tokens(title)| / |tokens(title)|`,
    `quote_overlap = |tokens(question) & tokens(quote)| / |tokens(question)|`,
    `question_len_tokens = len(tokens(question))` (M3-SPEC.md §2), plus
    `gold_doc_rank` passed through unchanged — see `CandidateFeatures`'s
    docstring for what it means and how the caller computes it.

    Both overlap ratios are `0.0`, not undefined, when their denominator's
    token set is empty (a blank title, or a zero-token question) — an
    empty denominator carries no signal either way, and `0.0` is the
    inert value for a ratio that measures "how much overlap", not a claim
    that overlap was measured and found to be exactly zero.
    """
    question_tokens = tokenize(question_text)
    question_token_set = set(question_tokens)
    title_token_set = set(tokenize(title))
    quote_token_set = set(tokenize(quote))

    title_overlap = (
        len(question_token_set & title_token_set) / len(title_token_set) if title_token_set else 0.0
    )
    quote_overlap = (
        len(question_token_set & quote_token_set) / len(question_token_set) if question_token_set else 0.0
    )

    return CandidateFeatures(
        title_overlap=title_overlap,
        quote_overlap=quote_overlap,
        question_len_tokens=len(question_tokens),
        gold_doc_rank=gold_doc_rank,
    )
