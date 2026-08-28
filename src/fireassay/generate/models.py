"""Pydantic models for `generate/`.

`Candidate` matches M3-SPEC.md §2 exactly (`text`, `qtype`, `difficulty`,
`reference_answer`, `quote`) — it is the schema `OllamaClient.generate_json`
validates the model's JSON response against, so it carries **only** what
the model itself produces. `ResolvedCandidate` is the richer, persisted
shape (`candidate` table row): a `Candidate` plus the evidence span
resolved against the source document (`spans.py`), measured lexical
features (`lexical.py`), and generation provenance.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from fireassay.models import Difficulty, QType


class Candidate(BaseModel):
    """One question the generator proposed for a chunk, before span
    resolution. `difficulty`/`qtype` here are the generator's *proposal*,
    validated (or overturned) later by a human via the curation rubric's
    `difficulty_agrees`/`qtype_agrees` — never trusted as ground truth on
    their own (M3-SPEC.md §2)."""

    model_config = ConfigDict(frozen=True)

    text: str
    qtype: QType
    difficulty: Difficulty
    reference_answer: str
    quote: str


class CandidateBatch(BaseModel):
    """The full JSON shape one `generate_json` call returns: `n_per_chunk`
    candidates for one chunk, wrapped in an object because
    `OllamaClient.generate_json` validates a single top-level schema, not a
    bare JSON array."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[Candidate, ...]


class LexicalFeatures(BaseModel):
    """Measured, not assumed (M3-SPEC.md §2) — recorded on every candidate
    alongside its proposed `difficulty` so `curate report` can report
    whether the label correlates with anything measurable at all (the
    spike found it can be inverted: probes labelled `easy` scored 0.176
    while `hard` scored 0.588, because what drove retrieval was
    document-title overlap, not passage overlap)."""

    model_config = ConfigDict(frozen=True)

    title_overlap: float
    quote_overlap: float
    question_len_tokens: int


class ResolvedCandidate(BaseModel):
    """A `Candidate` whose evidence span has been resolved against its
    source document and whose lexical features have been measured — the
    unit `generate/` persists to the `candidate` table
    (`Store.put_candidates`) and everything downstream (`filter/`,
    `curate/`) operates on.

    `id` is a random identifier, not a content hash (mirrors `Run.id`,
    `control_check.id`, `mutant.id`): a candidate is the outcome of an LLM
    generation event, which — even at temperature 0 — is not guaranteed
    reproducible byte-for-byte the way a `Question` derived purely from
    already-known text is, so treating it as content-addressed would be a
    false promise.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    batch_id: str
    text: str
    qtype: QType
    difficulty: Difficulty
    reference_answer: str
    quote: str
    source_doc_id: str
    char_start: int
    char_end: int
    features: LexicalFeatures
    model_digest: str
    prompt_hash: str
    created_at: str
