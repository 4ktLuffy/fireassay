"""Pydantic models for `generate/`.

`Candidate` matches M3-SPEC.md §2 exactly (`text`, `qtype`, `difficulty`,
`reference_answer`, `quote`) — it is the schema `OllamaClient.generate_json`
validates the model's JSON response against, so it carries **only** what
the model itself produces. `ResolvedCandidate` is the richer, persisted
shape (`candidate` table row): a `Candidate` plus the evidence span
resolved against the source document (`spans.py`), measured features
(`lexical.py`) including a retrieval-based one, and generation provenance.
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


class CandidateFeatures(BaseModel):
    """Measured, not assumed (M3-SPEC.md §2) — recorded on every candidate
    alongside its proposed `difficulty` so `curate report` can report
    whether the label correlates with anything measurable at all (the
    spike found it can be inverted: probes labelled `easy` scored 0.176
    while `hard` scored 0.588, because what drove retrieval was
    document-title overlap, not passage overlap).

    Renamed from `LexicalFeatures`: `gold_doc_rank` is a retrieval-based
    feature, not a lexical one, so the old name stopped fitting once it
    was added.

    `gold_doc_rank` — the 1-based BM25 rank at which the candidate's own
    source document was first found when queried with the candidate's own
    `text`, against the whole corpus, at **generation time** (`None` if
    not found within the search depth `generate.pipeline` uses). This is
    an *empirical* difficulty signal, deliberately alongside the
    generator's *proposed* `difficulty` label: the spike showed the
    proposed label can be inverted relative to what actually drives
    retrieval, and a measured rank (1 = easy to find, 30+ = hard) is
    checked against reality rather than trusted (`curate.coverage.
    difficulty_feature_correlation`).

    This is a **separate measurement** from `filter.stages.
    check_unretrievable`'s own, filter-time BM25 lookup — the filter does
    not read this stored value back (`candidate` rows are append-only, so
    a value computed once at generation could go stale against a corpus
    that has since drifted; the filter re-measures against whatever
    corpus it is actually run with). Both share the same BM25-vs-BM25
    tautology risk when the *same* retriever family is later evaluated —
    see `check_unretrievable`'s docstring for why that is defensible for
    filtering and not assumed away.
    """

    model_config = ConfigDict(frozen=True)

    title_overlap: float
    quote_overlap: float
    question_len_tokens: int
    gold_doc_rank: int | None = None


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

    `target_qtype`/`target_difficulty` are the cell `generate.pipeline`
    explicitly asked the model for (see its module docstring for why an
    unstratified prompt collapsed generation to ~2 of 21 taxonomy cells).
    `qtype`/`difficulty` remain the model's own *proposal* and may
    disagree with the target — `curate.coverage.generator_obedience`
    reports how often they agree, a free measure of whether the model is
    actually listening to what it is asked for.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    batch_id: str
    text: str
    qtype: QType
    difficulty: Difficulty
    target_qtype: QType
    target_difficulty: Difficulty
    reference_answer: str
    quote: str
    source_doc_id: str
    char_start: int
    char_end: int
    features: CandidateFeatures
    model_digest: str
    prompt_hash: str
    created_at: str
