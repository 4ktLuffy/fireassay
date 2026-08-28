"""Question generation (M3-SPEC.md §2) — the only package permitted to call
`llm.OllamaClient` anywhere in fireassay.

For each chunk, the model is asked for `n_per_chunk` questions, each
carrying a verbatim quote. Every candidate's evidence span is then resolved
by locating that quote in the **source document** (never approximated —
see `spans.py`), and its lexical features are measured, never assumed (see
`lexical.py`) — the spike (`spike/RESULTS.md`) found a generator's
`difficulty` label can be *inverted* relative to what actually drives
retrieval, so the proposed label is recorded as a proposal, not a fact.

**Prompt-injection hazard, one stage earlier than `docs/SPEC.md` §10.4's
`corpus_injection`:** gov.uk service pages are wall-to-wall imperatives
("Start now", "You must apply within 28 days", "Sign in to continue") —
text a generator can obey instead of writing a question *about*.
`prompts.py` places the chunk inside an explicit data boundary and states
its contents are material to ask about, never instructions to follow;
`validate.py` additionally rejects any candidate whose own `text` reads as
an imperative that leaked through, under `NOT_A_QUESTION`.

Also worth knowing when reading `curate report`'s content-coverage numbers:
generators are documented to over-select locally salient spans (headings,
bolded text, the first sentence), so raw generation coverage skews toward
whatever is visually prominent in a chunk rather than being uniform. M3
does not correct for this — the content-coverage report exists partly to
make that skew visible, not to hide it.
"""
