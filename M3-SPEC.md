# fireassay M3 — generation, machine filter, curation core

Repo root: `/Users/Twinkle/AI Engineering/fireassay`
Read `M1-SPEC.md`, `M2-SPEC.md`, `docs/SPEC.md` §7, and **`spike/RESULTS.md`** — the spike
changed several decisions below and its findings are load-bearing.

M1 and M2 are green: 199 tests, 95% coverage, ruff + mypy --strict clean. Do not regress them.

## M3 goal

Turn a corpus into ~3,000 candidate questions, filter them deterministically, and provide a
fully-tested **non-interactive** curation core that takes a human's verdicts and reports the
funnel, agreement and coverage.

Target funnel: **~3,000 generated → ~1,800 after filter → 1,000 curated**.

### In scope
Ollama client · generation with evidence spans · deterministic machine filter · rubric and
decision model · queue with honeypots · Krippendorff's α · funnel report · content coverage · CLI

### Out of scope (do not build)
The interactive TUI (**M3b**) · LLM judges · CJE · pooling · the statistical gate · HTML report ·
Dawid-Skene / crowd-kit (pointless with one curator; revisit when there are several)

## Constraints

- **Only `generate/` may call the LLM.** Filter and curation core are deterministic and LLM-free.
- **No test may call a live model.** All LLM interaction goes through a cache (§1) that tests
  populate from fixtures.
- New runtime dependency: **`krippendorff`** only. No pandas, no scikit-learn, no HTTP library —
  use `urllib.request` for Ollama, as `tools/fetch_govuk.py` already does.
- `mypy --strict` and `ruff` clean. No git, no shell.

---

## 1. `src/fireassay/llm/` — the only model boundary

`ollama.py`:

```python
@dataclass(frozen=True)
class ModelRef:
    name: str          # "qwen2.5:7b"
    digest: str        # from /api/show — the identity that goes in env_fingerprint

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434",
                 cache: ResponseCache | None = None, timeout: float = 120.0) -> None: ...
    def model_ref(self, name: str) -> ModelRef: ...
    def generate_json(self, model: ModelRef, prompt: str, schema: type[BaseModel],
                      *, temperature: float = 0.0, max_retries: int = 3) -> BaseModel: ...
```

- **Pin by digest, never by tag.** An Ollama tag can be re-pulled and silently change. `ModelRef`
  carries the digest and it is what lands in `env_fingerprint.affects_results.generator_model`.
- `generate_json` requests JSON, validates against `schema`, and retries on invalid output up to
  `max_retries`. **Exhausting retries raises** — it never returns a partial or best-effort object.
  A generator that silently degrades is a generator that quietly poisons a golden set.

`cache.py`:

```python
class ResponseCache:
    """Content-addressed on (model_digest, prompt). Makes regeneration free,
    makes runs reproducible, and lets tests replay recorded responses with no
    live model."""
    def get(self, model: ModelRef, prompt: str) -> str | None: ...
    def put(self, model: ModelRef, prompt: str, response: str) -> None: ...
```

JSONL on disk under `--cache-dir`. Key is `content_hash("fa.llmcache.1", digest, prompt)`.

**Tests use a `ResponseCache` pre-populated from `tests/fixtures/llm/*.jsonl` and a client
constructed with a base_url that would fail if called.** A test that reaches the network must
fail loudly, not silently pass.

## 2. `src/fireassay/generate/`

For each chunk, ask the model for `n_per_chunk` questions. Each candidate carries a question,
a reference answer, and **a verbatim quote from the chunk** — from which we compute the evidence
span by locating the quote in the source document.

```python
class Candidate(BaseModel):
    text: str
    qtype: QType
    difficulty: Difficulty      # the generator's PROPOSAL, validated in curation
    reference_answer: str
    quote: str                  # must appear verbatim in the source chunk
```

Span resolution rules — these matter, the spike found the failure mode:

- The quote must be found in the source **document** text. If it is not found, **discard the
  candidate** and count it under `QUOTE_NOT_FOUND`. Never approximate or fuzzy-match a span.
- If the quote appears **more than once** in the document, discard it and count it under
  `QUOTE_AMBIGUOUS`. The spike produced one such probe; at 3,000 candidates it will be many, and
  a span pointing at the wrong occurrence is a silently wrong ground truth.

### Lexical features, recorded not assumed

The spike proved a generator's difficulty labels can be **inverted**: probes labelled `easy`
scored 0.176 while `hard` scored 0.588, because what actually drove retrieval was whether the
question contained document-title terms, not passage overlap.

So every candidate records measured features alongside the proposed label:

```python
class LexicalFeatures(BaseModel):
    title_overlap: float      # |tokens(question) & tokens(title)| / |tokens(title)|
    quote_overlap: float      # |tokens(question) & tokens(quote)| / |tokens(question)|
    question_len_tokens: int
```

Stored on the question row. `fireassay curate report` reports the correlation between the
proposed difficulty and each feature. **A difficulty label that does not correlate with anything
measurable is a label, not a difficulty** — and we will be able to say so with a number.

## 3. `src/fireassay/filter/` — deterministic, no model

Ordered pipeline; each stage records a reason code and a count.

| Stage | Rule | Reason code |
|---|---|---|
| span resolution | quote missing / ambiguous (§2) | `QUOTE_NOT_FOUND`, `QUOTE_AMBIGUOUS` |
| degeneracy | question < 5 or > 60 tokens; no content word | `DEGENERATE` |
| self-containment | question contains a dangling referent (`this`, `it`, `the above`) as its only subject | `NOT_SELF_CONTAINED` |
| near-duplicate | **token-set Jaccard ≥ 0.85** against any kept candidate | `NEAR_DUPLICATE` |
| generic | question's content words appear in > `generic_df_pct` (default 5%) of corpus documents | `TOO_GENERIC` |
| balance | per `(qtype × difficulty)` cell cap, keeping the earliest | `CELL_FULL` |

**Near-duplicate uses token-set Jaccard, not embeddings** — deterministic, testable, no model,
and it uses the single `text.tokenize` (never a new tokenizer; see M1 §6).

`TOO_GENERIC` exists because of the spike: *"What does the guidance say about revenue and
customs?"* matched thousands of gov.uk pages and never retrieved its own source document in the
top 50. Such a question has no recoverable ground truth and must not reach a human.

## 4. `src/fireassay/curate/` — core only, no UI

```python
class RubricVerdict(BaseModel):
    answerable_from_kb: Literal["yes", "no", "partially"]
    self_contained: bool
    reference_answer_correct: Literal["yes", "no", "incomplete"]
    evidence_sufficient: bool
    difficulty_agrees: bool
    difficulty_reassigned: Difficulty | None = None
    qtype_agrees: bool
    qtype_reassigned: QType | None = None

class Decision(BaseModel):
    candidate_id: str
    curator_id: str
    decision: Literal["accept", "edit", "reject"]
    reject_reason: RejectReason | None = None      # required iff decision == "reject"
    rubric: RubricVerdict
    edited_text: str | None = None
    edited_answer: str | None = None
    notes: str | None = None
    duration_ms: int
```

`RejectReason`: `DUPLICATE` `NOT_ANSWERABLE` `AMBIGUOUS` `WRONG_REFERENCE`
`INSUFFICIENT_EVIDENCE` `LEADING_QUESTION` `TRIVIAL` `OUT_OF_SCOPE` `PII_RISK`

### Queue

```python
def build_queue(candidates, *, curator_id, honeypot_rate=0.05,
                double_review_rate=0.10, seed) -> list[QueueItem]
```

- honeypots and double-reviews are interleaved **indistinguishably** from real items
- deterministic given `seed`; the seed derives from the candidate-set hash

### Honeypots — known-bad first

A honeypot is an item whose correct verdict is known in advance.

- **known-bad** (available immediately, no bootstrap): take a real candidate and corrupt it —
  swap in another candidate's reference answer (`WRONG_REFERENCE`), or attach an evidence span
  from an unrelated document (`INSUFFICIENT_EVIDENCE`). The curator **must reject** it.
- **known-good**: real candidates verified by a second curator in an earlier session. Only
  available after a bootstrap round, so M3 ships known-bad and leaves a hook for known-good.

`curator_accuracy = correct honeypot verdicts / honeypots seen`, reported per curator and per
session decile so drift is visible.

**Agreement is not accuracy.** Krippendorff's α says whether two curators agree; honeypots say
whether a curator is right. Both are reported and neither substitutes for the other.

### Agreement

`agreement.py` wraps the `krippendorff` package. **Ordinal** metric for
`answerable_from_kb` and `reference_answer_correct`; **nominal** for the booleans. Missing
observations are expected and must be passed through, not imputed — that is why α and not κ.

### Reports

`fireassay curate report` emits:

- the funnel: generated → each filter stage with reason counts → curated, with accept/edit/reject
  and reject-reason breakdown
- reviewer hours, and median seconds per item
- Krippendorff's α per rubric criterion, with the α < 0.6 flag written to `suite.agreement_json`
- honeypot accuracy per curator, and per session decile
- **content coverage**: fraction of corpus documents with ≥1 question, and the
  `(qtype × difficulty)` cell occupancy matrix
- **difficulty validation**: correlation of proposed difficulty against each lexical feature

## 5. Schema — `migrations/0003_generation_curation.sql`

```sql
candidate(id, batch_id, text, qtype, difficulty, reference_answer, quote,
          source_doc_id, char_start, char_end, features_json,
          model_digest, prompt_hash, created_at)

filter_result(candidate_id, stage, kept, reason, checked_at)

queue_item(id, queue_id, curator_id, candidate_id, position,
           is_honeypot, honeypot_expected_reason, is_double_review)

decision(id, candidate_id, curator_id, decision, reject_reason,
         rubric_json, edited_text, edited_answer, notes, duration_ms, decided_at)
```

Append-only, sealed as in M1/M2. A `decision` may not be overwritten — a changed mind is a new
row, and the report uses the latest per `(candidate_id, curator_id)`.

## 6. CLI

```
fireassay generate  --corpus data/corpus.jsonl --model qwen2.5:7b --n 3000 \
                    --cache-dir .cache/llm --db fa.db
fireassay filter    --batch <id> --config configs/filter.yaml --db fa.db
fireassay curate next    --curator henos --db fa.db          # emits one item as JSON
fireassay curate submit  --file verdict.json --db fa.db       # records one decision
fireassay curate report  --db fa.db
fireassay suite freeze --name govuk --version 1.0.0 --from-curated --db fa.db
```

`curate next` / `curate submit` are the **non-interactive core the M3b TUI will drive**. They must
be independently usable and fully tested — the TUI adds no logic, only keystrokes.

## 7. Tests

| File | Must prove |
|---|---|
| `test_llm_cache.py` | content-addressed on (digest, prompt); a cache hit makes no client call |
| `test_ollama_client.py` | JSON validated against schema; invalid output retries; **exhausting retries RAISES, never returns partial**; all via recorded fixtures, never the network |
| `test_generate_spans.py` | quote not in document → `QUOTE_NOT_FOUND`; quote appearing twice → `QUOTE_AMBIGUOUS`; **no fuzzy matching anywhere** |
| `test_lexical_features.py` | title/quote overlap computed via `text.tokenize` only |
| `test_filter.py` | each stage fires on its own case and stays silent on a clean one; Jaccard threshold boundary; `TOO_GENERIC` catches a spike-style generic question |
| `test_queue.py` | deterministic under a seed; honeypots and double-reviews are indistinguishable in the emitted item |
| `test_honeypots.py` | a corrupted candidate's expected reason is recorded; accuracy computed correctly; a curator who accepts a known-bad item scores 0 for it |
| `test_agreement.py` | α on a hand-computed example; ordinal vs nominal handled distinctly; missing observations pass through un-imputed |
| `test_curate_report.py` | funnel counts reconcile — **generated == kept + every rejection reason**, with no silent losses |
| `test_curate_cli.py` | next/submit round-trip; a reject without a reason is refused |

The funnel reconciliation test is the important one: a funnel that loses candidates without
naming a reason is the reporting equivalent of a check that cannot run reading as a pass.

## 8. If anything is ambiguous

Stop and ask. Especially: the retry-exhaustion rule (§1), discarding rather than approximating
ambiguous spans (§2), and honeypots-vs-agreement being different instruments (§4). A plausible
guess in those three is worse than a question.
