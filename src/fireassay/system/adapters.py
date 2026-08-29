"""Thin adapters wiring a concrete model client into the `Generate`
callable `system.rag.RagSystem` expects.

`system/rag.py` must not import `fireassay.llm` -- see that module's
docstring for why the boundary matters: it is what lets the generator be
swapped, or replaced with a fake in a hermetic test, without ever touching
the system. This module is the other side of that seam: the one place that
does know about `OllamaClient`, so a caller can still get a working
`RagSystem` wired to a real model.
"""

from __future__ import annotations

from fireassay.llm.ollama import ModelRef, OllamaClient
from fireassay.system.rag import Generate


def ollama_generate(client: OllamaClient, model: ModelRef, *, temperature: float = 0.0) -> Generate:
    """Build a `Generate` for `RagSystem` that calls
    `client.generate_text(model, prompt, temperature=temperature)`.

    `generate_text` (not `generate_json`) is the right call here for the
    same reason it is for `items.answerability.Ask`: `RagSystem`'s
    abstention contract depends on a free-text completion ending in the
    model's own final word (the greedy last-match sentinel check in
    `system.rag._parse_completion`), not a value nested inside a JSON
    string field -- see `OllamaClient.generate_text`'s docstring.
    """

    def generate(prompt: str) -> str:
        return client.generate_text(model, prompt, temperature=temperature)

    return generate
