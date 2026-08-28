# fireassay M1 — implementation spec

Repo root: `/Users/Twinkle/AI Engineering/fireassay`
Read `../fireassay-SPEC.md` for context. This document governs M1 and overrides it on detail.

## M1 goal

The deterministic spine. **No LLM call anywhere in M1.** A user can import a golden set,
freeze it into a content-addressed suite, run a real retrieval system against it, score it
on five deterministic metrics, and produce a leaderboard — and the tool refuses to compare
runs that are not comparable.

### In scope
store · suite hashing & freeze · integrity/refusal · config matrix · runner · BM25 system ·
5 deterministic scorers · leaderboard · CLI subset · tests

### Out of scope (later milestones — do not build)
question generation · curation TUI · negative controls · LLM judges · CJE · release gate ·
HTML report · corpus scraping

## Environment & constraints

- Python 3.12+. `src/` layout. Package name `fireassay`.
- Dependencies — **these only**: `pydantic>=2`, `typer`, `pyyaml`, `numpy`, `rich`.
  Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`.
  SQLite via stdlib `sqlite3`. **No SQLAlchemy, no ORM, no network libraries.**
- `mypy --strict` clean. `ruff` clean. Full type annotations, no bare `Any` in public APIs.
- No git commands, no shell, no network, no LLM calls.
- Docstrings on every public function: what it does and *why it is defined that way* where
  the definition is a judgement call (e.g. why recall is omitted rather than zeroed).

---

## 1. `src/fireassay/hashing.py`

```python
def canonical_json(obj: object) -> bytes:
    """Deterministic JSON: sorted keys, (',', ':') separators, ensure_ascii=False,
    UTF-8 encoded. Raises ValueError on NaN/Infinity — they would make a hash
    unstable across platforms."""

def content_hash(prefix: str, *parts: str | bytes) -> str:
    """sha256 hex. Parts are joined with b'\\x1f' (unit separator) after UTF-8
    encoding, prefixed by `prefix` + b'\\x1f'. The prefix is a scheme version tag
    so the hash scheme can change without silently colliding with old ids."""
```

Exact hash definitions — these are contractual, tests must pin them:

| Id | Prefix | Input parts, in order |
|---|---|---|
| `question.id` | `fa.question.1` | `text`, `reference_answer or ""`, `"\n".join(sorted(evidence_span_keys))` |
| `suite.suite_hash` | `fa.suite.1` | `"\n".join(sorted(question_ids))` |
| `config.config_hash` | `fa.config.1` | `canonical_json(spec)` |

`evidence_span_key` = `f"{doc_id}:{chunk_id}:{char_start}:{char_end}"`.

## 2. `src/fireassay/models.py`

Pydantic v2 models, all frozen (`model_config = ConfigDict(frozen=True)`).

```python
class EvidenceSpan(BaseModel):
    doc_id: str; chunk_id: str; page: int | None = None
    char_start: int; char_end: int; quote: str
    def key(self) -> str: ...

QType = Literal["factual","procedural","comparative","multi_hop",
                "unanswerable","ambiguous","policy_sensitive"]
Difficulty = Literal["easy","medium","hard"]
Provenance = Literal["synthetic","real_traffic"]

class Question(BaseModel):
    id: str = ""                      # computed in model_post_init if empty
    text: str; qtype: QType; difficulty: Difficulty
    reference_answer: str | None = None
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    provenance: Provenance
    generator: str | None = None      # name@version; must be None when provenance=="real_traffic"
    source_doc_id: str | None = None

class RetrievedChunk(BaseModel):
    doc_id: str; chunk_id: str; score: float; rank: int   # rank is 1-based

class SystemOutput(BaseModel):
    answer: str | None; abstained: bool
    retrieved: tuple[RetrievedChunk, ...] = ()
    latency_ms: dict[str, float]      # keys: any stage names + required "total"
    tokens_in: int = 0; tokens_out: int = 0

class Score(BaseModel):
    metric: str; value: float; scorer: str      # scorer is "name@version"
    rationale: str | None = None
```

Validation: `Question` rejects `generator is not None` when `provenance == "real_traffic"`.
`SystemOutput` rejects `latency_ms` missing the `"total"` key.

## 3. `src/fireassay/store/`

`schema.sql` — the DDL from `../fireassay-SPEC.md` §5, **restricted to M1 tables**:
`question`, `evidence_span`, `suite`, `suite_question`, `config`, `run`, `result`, `score`.
Omit `curation`, `control_check`, `gate_check` (later milestones).

`migrations/0001_initial.sql` holds that DDL. `store/db.py`:

```python
class Store:
    def __init__(self, path: Path | str) -> None: ...
    def migrate(self) -> None:
        """Apply un-applied migrations in filename order, tracked in a
        schema_migrations table. Idempotent."""
    # writes
    def put_questions(self, questions: Iterable[Question]) -> int: ...
    def freeze_suite(self, name: str, version: str, question_ids: Sequence[str]) -> Suite: ...
    def put_config(self, spec: Mapping[str, object], label: str | None = None) -> Config: ...
    def start_run(self, suite: Suite, config: Config, env: Mapping[str, object]) -> Run: ...
    def put_result(self, run_id: str, question_id: str, output: SystemOutput, cost_usd: float) -> None: ...
    def put_scores(self, run_id: str, question_id: str, scores: Sequence[Score]) -> None: ...
    def finish_run(self, run_id: str, status: Literal["complete","failed"]) -> None: ...
    # reads
    def get_run(self, run_id: str) -> Run: ...
    def get_suite(self, name: str, version: str) -> Suite: ...
    def iter_questions(self, suite_id: str) -> Iterator[Question]: ...
    def run_scores(self, run_id: str) -> list[tuple[str, str, float]]: ...  # (question_id, metric, value)
```

Rules:
- `PRAGMA foreign_keys = ON` and `journal_mode = WAL` on every connection.
- **Append-only**: after `finish_run`, any write touching that `run_id` raises
  `RunSealedError`. Enforce in Python *and* with SQL triggers on `result`/`score`
  that reject writes when the parent run's status is `complete`.
- `freeze_suite` raises `SuiteExistsError` if `(name, version)` exists with a different
  `suite_hash`; returns the existing row if the hash matches (idempotent re-freeze).
- `put_config` is idempotent on `config_hash`.

## 4. `src/fireassay/integrity.py`

The core of the repo.

```python
class ComparisonRefused(Exception):
    def __init__(self, reasons: Sequence[RefusalReason]) -> None: ...

class RefusalReason(BaseModel):
    code: Literal["SUITE_MISMATCH","SCORER_MISMATCH","ENV_MISMATCH",
                  "RUN_INCOMPLETE","EMPTY_RUN"]
    detail: str

class ComparabilityReport(BaseModel):
    comparable: bool
    reasons: tuple[RefusalReason, ...]
    forced: bool = False

def assert_comparable(runs: Sequence[Run], *, scorers_by_run: Mapping[str, frozenset[str]],
                      force: bool = False) -> ComparabilityReport:
    """Refuse to compare runs that are not soundly comparable.

    Refuses when: suite_hash differs between any two runs; the set of scorer
    name@version differs; env fields marked affects_results differ; any run has
    status != 'complete'; any run has zero results.

    With force=True, returns a report with forced=True instead of raising. Every
    renderer MUST check `.forced` and prefix its output with 'UNSOUND COMPARISON'.
    """
```

`env_fingerprint`: `run.env_json` is `{"python": ..., "fireassay": ..., "affects_results": {...}}`.
Only the `affects_results` sub-dict participates in `ENV_MISMATCH`.

## 5. `src/fireassay/config.py`

```python
class MatrixSpec(BaseModel):
    suite: str                              # "name@version"
    axes: dict[str, list[dict[str, object]] | list[str]]
    include: list[dict[str, object]] = []
    exclude: list[dict[str, object]] = []

def expand(spec: MatrixSpec) -> list[dict[str, object]]:
    """Cartesian product over axes, minus configs matching any `exclude` pattern
    (a pattern matches when every key it names equals the config's value for that
    key), plus `include` entries. Deterministic order: sorted by config_hash."""
```

## 6. `src/fireassay/system/`

`base.py`:
```python
class System(Protocol):
    name: str
    def answer(self, question: Question) -> SystemOutput: ...
```

`bm25.py` — **implement BM25 in pure Python**, no new dependency (numpy is allowed for the
scoring loop). `k1=1.5`, `b=0.75`, defaults documented. Tokeniser: lowercase, split on
`\W+`, drop empties. This is the M1 system-under-test: real retrieval, fully deterministic,
no API key.

```python
class BM25System:
    def __init__(self, chunks: Sequence[Chunk], top_k: int = 5,
                 k1: float = 1.5, b: float = 0.75) -> None: ...
    def answer(self, question: Question) -> SystemOutput:
        """Returns retrieved chunks only; answer=None, abstained=True, tokens=0.
        M1 measures retrieval, not generation."""
```

`corpus.py`: load a corpus from JSONL (`{"doc_id","title","text"}` per line), chunk with a
fixed-size character chunker (`size`, `overlap`), yielding `Chunk(doc_id, chunk_id, text,
char_start, char_end)`. `chunk_id` = `f"{doc_id}#{index:04d}"`.

## 7. `src/fireassay/score/`

`base.py`:
```python
class Scorer(Protocol):
    name: str; version: str; requires_llm: bool
    def score(self, question: Question, output: SystemOutput,
              ctx: ScoringContext) -> list[Score]: ...
```
`ctx` carries `price_in_per_mtok`, `price_out_per_mtok`, `top_k`, `policy_rules`.

All five M1 scorers have `requires_llm = False`.

### `retrieval.py` — `RetrievalScorer` (`retrieval@1.0.0`)

Gold set = `{span.chunk_id for span in question.evidence_spans}`.

- **If the question has no evidence spans, emit no scores at all.** Do not emit 0.0 — a
  question with no gold evidence carries no information about retrieval, and zeroing it
  would silently drag every average down.
- `retrieval.recall@k` = `|gold ∩ top_k| / |gold|`
- `retrieval.ndcg@k` = binary relevance; `DCG = Σ rel_i / log2(i+1)` over 1-based ranks;
  `IDCG` over `min(|gold|, k)` ideal hits; 0.0 when `IDCG == 0`
- `retrieval.mrr` = `1/rank` of the first gold chunk within top-k, else 0.0

`k` comes from `ctx.top_k`; metric names interpolate it (`retrieval.recall@5`).

### `latency.py` — `LatencyScorer` (`latency@1.0.0`)
Emits `latency.total_ms` per question from `output.latency_ms["total"]`, plus
`latency.{stage}_ms` for each other stage key. Percentiles are computed at aggregation
time (§8), **not** here.

### `cost.py` — `CostScorer` (`cost@1.0.0`)
`cost.usd = tokens_in / 1e6 * price_in_per_mtok + tokens_out / 1e6 * price_out_per_mtok`.
Prices are **per million tokens** — state this in the docstring; a units error here
silently corrupts every cost number in the repo.

### `policy.py` — `PolicyScorer` (`policy@1.0.0`)
Rule pack loaded from YAML:
```yaml
rules:
  - id: no_email_egress
    pattern: '[\w.+-]+@[\w-]+\.[\w.]+'
    applies_to: answer          # answer | retrieved
    severity: high
```
Emits `policy.violations` = count of matching rules (not matches). Emits 0.0 when the
answer is None. `rationale` lists the ids of rules that fired.

### `abstention.py` — `AbstentionScorer` (`abstention@1.0.0`)
Only for `question.qtype == "unanswerable"`; emits nothing otherwise.
`abstention.correct` = 1.0 if `output.abstained` else 0.0.

## 8. `src/fireassay/compare.py`

```python
class MetricSummary(BaseModel):
    metric: str; n: int; mean: float
    p50: float | None = None; p95: float | None = None   # latency.* only

def leaderboard(store: Store, runs: Sequence[Run], *, force: bool = False,
                split_by_provenance: bool = False) -> Leaderboard:
    """Calls assert_comparable first. Aggregates per run per metric.

    - mean over questions that HAVE the metric; `n` reports that count, which will
      differ between metrics (see RetrievalScorer). Never impute a missing score.
    - latency.* additionally get p50/p95 by the nearest-rank method on sorted values.
    - split_by_provenance emits a separate summary per provenance value.
    """
```

Rendering to `rich` tables lives in `report/text.py`, not here. Any render of a
`Leaderboard` whose `report.forced` is True MUST print `UNSOUND COMPARISON` as the first
line and in the table title.

## 9. `src/fireassay/runner.py`

```python
def run_matrix(store: Store, spec: MatrixSpec, system_factory: Callable[[Mapping[str, object]], System],
               scorers: Sequence[Scorer], ctx_factory: Callable[[Mapping[str, object]], ScoringContext],
               *, rerun: bool = False) -> list[Run]:
    """For each expanded config: skip if a complete run exists for
    (suite_hash, config_hash) unless rerun=True. Otherwise start_run, iterate the
    suite's questions, call system.answer, time it, score it, persist, finish_run.

    A question that raises is recorded as a failed result (answer=None,
    abstained=False) and does not abort the run; the run's status becomes 'failed'
    if more than 1% of questions raised.
    """
```

Sequential execution in M1. No threads, no async — concurrency is M5.

## 10. `src/fireassay/cli.py` (typer)

```
fireassay init                --db bench.db
fireassay questions import    questions.jsonl --db bench.db
fireassay suite freeze        --name support-kb --version 1.0.0 --db bench.db
fireassay suite show          support-kb@1.0.0 --db bench.db
fireassay run                 --matrix configs/matrix.yaml --corpus corpus.jsonl --db bench.db
fireassay compare             --suite support-kb@1.0.0 --db bench.db [--force] [--by-provenance]
fireassay diff                --base <run_id> --head <run_id> --db bench.db [--force]
```

`compare`/`diff` exit **2** on `ComparisonRefused` (distinct from 1 = usage error), and
print the refusal reasons as a table.

## 11. Tests — `tests/`

Fixtures in `tests/fixtures/`: `corpus.jsonl` (12 short docs) and `questions.jsonl`
(20 questions: 14 answerable with gold spans, 3 `unanswerable`, 3 with no evidence spans).

Required tests:

| File | Must prove |
|---|---|
| `test_hashing.py` | canonical_json is key-order independent; rejects NaN; the three hash ids match pinned literal values (regression-locks the scheme) |
| `test_models.py` | `generator` rejected for real_traffic; `latency_ms` without "total" rejected; question id auto-computed |
| `test_store.py` | migrate is idempotent; re-freezing an identical suite returns the same row; re-freezing a changed suite raises `SuiteExistsError`; writing to a sealed run raises `RunSealedError` **and** the SQL trigger blocks it |
| `test_integrity.py` | each of the five refusal codes fires on its own; `force=True` returns `forced=True` and does not raise; a genuinely comparable pair passes |
| `test_retrieval_scorer.py` | recall/nDCG/MRR against **hand-computed** values on a fixed ranking; **a question with no evidence spans yields zero Score objects** |
| `test_cost_scorer.py` | per-million-token arithmetic on a worked example |
| `test_policy_scorer.py` | rule fires, counts rules not matches, 0.0 on None answer |
| `test_abstention_scorer.py` | emits only for unanswerable; 1.0/0.0 correctly |
| `test_bm25.py` | ranks an obviously-relevant doc first on the fixture corpus; deterministic across two runs |
| `test_compare.py` | means skip missing scores rather than imputing; `n` differs per metric; forced leaderboard is flagged |
| `test_runner_e2e.py` | full path: import → freeze → run 2 configs → compare produces a leaderboard with both runs |
| `test_cli.py` | `compare` across two different suites exits 2 |

Aim for >90% line coverage on `integrity.py`, `hashing.py` and `score/`.

## 12. Also produce

- `pyproject.toml` (uv-compatible, ruff + mypy strict config, pytest config)
- `README.md` — **M1 scope only**, honest that judges/gate/curation are not built yet
- `Makefile` — `install`, `test`, `lint`, `typecheck`, `demo` (runs the fixture end-to-end)
- `configs/matrix.example.yaml`, `configs/policy.example.yaml`
- `.gitignore`

## 13. If anything here is ambiguous

Stop and ask. Do not fill a gap with an assumption — particularly around the hash
definitions, the refusal codes, or the "omit rather than zero" rule in RetrievalScorer.
Those three are load-bearing and a plausible-looking guess is worse than a question.
