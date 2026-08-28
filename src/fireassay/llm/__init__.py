"""fireassay's only model boundary (M3-SPEC.md §1).

`generate/` is the only package permitted to import from here — filter and
curate are deterministic and LLM-free by construction (M3-SPEC.md
"Constraints"). `OllamaClient` talks to a local Ollama server via
`urllib.request` only (no new HTTP dependency, matching `tools/
fetch_govuk.py`'s existing pattern), and every call it makes can be
short-circuited by a `ResponseCache` so tests never reach a live model.
"""
