"""Deterministic, LLM-free candidate filtering (M3-SPEC.md §3).

Operates only on already-persisted `ResolvedCandidate` rows (i.e.
candidates whose span already resolved — `generate/` handles
`QUOTE_NOT_FOUND`/`QUOTE_AMBIGUOUS`/`NOT_A_QUESTION` itself, since those
depend on the raw LLM output and source document that only `generate/` has
in hand). Every stage here is a pure function of already-known data: token
counts, a fixed stopword-style content-word check, token-set Jaccard
(`text.tokenize`, never a second tokenizer), and corpus-wide document
frequency. No model call anywhere in this package.
"""
