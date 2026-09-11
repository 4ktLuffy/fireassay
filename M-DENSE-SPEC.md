# M-DENSE — a dense retriever, and BM25 measured against it

`decided: 2026-09-11` · status: **spec, nothing built yet** · scope: one milestone, no judge.

## 0. What this is, in one paragraph

Add a **dense (embedding) retriever** as a fourth `System` alongside `bm25`, `tfidf` and
`coverage`, and answer one question honestly: **does dense retrieval beat BM25 on this
corpus, and can this suite even resolve the difference?** The comparison runs through
`gate.evaluate_gate`, which already ships paired bootstrap, Holm-Bonferroni and a
minimum-detectable-effect refusal, so a difference smaller than the suite can resolve is
**refused rather than reported**. Everything is deterministic: no judge, no generated
answers, no new metric. The four existing negative controls must pass against the dense
retriever unchanged.

**Not in this milestone:** any answer-correctness or groundedness scorer, any Groq
integration, RAG Gain, and widening the frozen 54-config panel. Section 7 says why.

## 1. Rules

1. **No git commits, ever, by a builder.** Leave the work in the tree; Henos reviews the
   diff and commits. **No AI attribution trailers anywhere** — this repo has zero and keeps
   zero.
2. **The tree is green and must stay green.** At HEAD `b412c93`: `ruff check src tests tools`
   clean, `mypy src` clean on 90 files, **644 tests passing**, **91% coverage**, and
   `python tools/evidence.py --check` reporting **all 27 claims PASS**. Every one of those
   must still hold at the end. CI order is lint → mypy → pytest → `evidence --check` →
   `gate-demo`.
3. **No new dependencies.** `M1-SPEC.md` restricts runtime deps to `pydantic>=2`, `typer`,
   `pyyaml`, `numpy`, `rich`; `M3-SPEC.md` added `krippendorff` and permitted `urllib.request`
   for Ollama, explicitly forbidding any new HTTP library. So the dense retriever is
   **numpy + urllib.request**. No chromadb, no qdrant, no faiss, no sentence-transformers,
   no torch. At 1,181 documents an exact dot product is the correct engineering choice
   anyway; an approximate index would add recall loss that then has to be measured.
4. **Zero paid API.** Embeddings come from local Ollama (`qwen3-embedding:0.6b`, 1024
   dimensions, confirmed responding). No key exists on this machine and none is needed.
5. **Every new scorer, control or invariant ships with a negative control** proving it goes
   red on a planted defect. A checker that has never fired is untested.
6. **Do not touch `run/panel54_matrix.csv`, `run/item_analysis.json`, or any existing
   artifact under `run/`.** They are frozen evidence behind passing claims. New artifacts get
   new filenames.
7. **Do not change any existing scorer's `version`.** Bumping one makes every stored run
   incomparable through `integrity.assert_comparable`'s `SCORER_MISMATCH`.
8. Use `.venv` (Python 3.12.13). Installs via `.venv/bin/python -m pip`, never `uv pip`.
9. Write `NOTES/` progress as you go if the repo has that convention; otherwise record
   measurements in the spec's closing-check section of your final report. Every number is
   preceded by the command that produced it.

## 2. Facts verified 2026-09-11 (do not re-derive)

**The `System` contract is tiny** (`system/base.py`): `name: str`,
`generates_answers: bool`, `answer(question) -> SystemOutput`. No version field. A system's
identity in a run is carried entirely by the config dict via `config_hash`.

**`SystemOutput`** requires `latency_ms["total"]`; `RetrievedChunk` carries
`doc_id, chunk_id, score, rank (1-based), char_start, char_end` and **no text**.
`RetrievalScorer` decides relevance by **character-range overlap** against evidence spans,
not `chunk_id` equality, which is what makes scores comparable across chunkings. So
`char_start`/`char_end` must be copied verbatim from the indexed `Chunk`.

**`BM25System` is the template**: constructor `(chunks, top_k=5, k1=1.5, b=0.75)`, takes
**already-chunked** `Sequence[Chunk]`, precomputes once in `__init__`, and breaks ties
deterministically by `(-score, doc_id, chunk_id)`. Retrieval-only systems return
`answer=None, abstained=True, tokens_in=0, tokens_out=0`.

**`panel.py` has no config dispatch.** `RANKERS` is a hardcoded tuple of
`(name, Callable[[Sequence[Chunk], int], System])`; `build_panel` is 3 rankers × 6 chunkings
× 3 top_ks = 54. **`cli.py run` is hardcoded to `BM25System`** and no config key selects a
retriever anywhere in the repo.

**Corpus and cost**, measured with fireassay's own chunker:

| chunking | chunks | serial embed @77ms | fp32 size |
|---|---|---|---|
| (1024, 128) | 12,513 | 16 min | 51 MB |
| (512, 128) | 28,408 | 36 min | 116 MB |
| all six | 208,258 | 4.5 h | 853 MB |

1,181 docs, 10,829,674 characters, median 4,814. **Ollama's `/api/embed` accepts a batched
`input` array** (verified: 3 texts in, 3 vectors out), so the serial figures above are an
upper bound. The older `/api/embeddings` endpoint takes one `prompt` at a time and must not
be used.

**Controls, and what a dense retriever does to each:**

| control | dense impact |
|---|---|
| `no_retrieval` | safe, operates on the `System` interface |
| `shuffled_gold` | safe, measures its own chance level per suite and config |
| `corpus_ablation` | safe, rebuilds through `ctx.system_factory` |
| **`identical_config`** | **at risk**: asserts `max_abs_delta == 0.0` across every metric except `latency.*`. A dense retriever is only bit-deterministic if its vectors come from a cache rather than a live model call. This is why the cache is load-bearing, not a speed optimisation. |
| `judge_calibration`, `null_questions` | stay `NOT_RUN`; both wait on a generating system and a judge |

**`check_invariants`** are properties of the scorer's arithmetic, not of the system, so a
correct dense retriever cannot violate them. The unchecked expectation it *could* violate is
panel-level top-k nesting (a narrower top_k's hits being a subset of a wider one's), which
holds by construction for exact search and fails for approximate search.

## 3. What to build

### 3.1 `src/fireassay/system/embedding.py`

```python
Embed = Callable[[Sequence[str]], np.ndarray]   # returns (n, d) float32, L2-normalised
```

- `OllamaEmbedder(model: ModelRef, *, base_url, batch_size: int = 64) -> Embed` using
  `urllib.request` against `/api/embed`. **Pin the model by digest** via
  `OllamaClient.model_ref()`, never by tag, matching how every other model in this repo is
  pinned and how `env_fingerprint` records identity.
- L2-normalise every vector on the way in, so similarity is a plain dot product. Guard the
  norm with `+ 1e-12`.
- **Reduce in float64 and store float32.** The embeddings handbook records a float32 cosine
  over 22M elements returning 1.0106, an impossible value for a cosine.
- A vector whose norm is 0, or whose dimension differs from the first vector seen, is an
  error, never a silently-padded row.

### 3.2 `src/fireassay/system/embedding_cache.py`

Content-addressed, resumable, and guarded. Cache key is
`(model_digest, chunk_size, chunk_overlap, corpus_hash)`; `corpus_hash` already exists in
`system/corpus.py`. On disk under `.cache/embeddings/<key>/`:

- `metadata.json` — model name and digest, dimension, chunking, `corpus_hash`, chunk count,
  and a sha256 over the ordered `chunk_id` list.
- `shards/<start>-<stop>.npy` — float32 shards written **atomically** (temp file then
  `os.replace`), so a kill costs at most one shard.

**Refuse a stale cache rather than use it.** If `metadata.json` disagrees with the current
run on any field, raise; do not silently re-embed or, worse, silently reuse. This is the
pattern already proven in `~/Other/embed-bench`.

`.cache/embeddings/` is gitignored. Reproducing dense numbers needs Ollama and roughly an
hour, exactly as the generation run needed an hour of model time; say so in the docs.

### 3.3 `src/fireassay/system/dense.py`

```python
class DenseSystem:
    def __init__(self, chunks: Sequence[Chunk], top_k: int = 5, *,
                 vectors: np.ndarray, query_embed: Embed) -> None: ...
```

- `name = "dense"`, `generates_answers = False`.
- `vectors` is the `(n_chunks, d)` matrix **already loaded from the cache**, aligned
  index-for-index with `chunks`. The constructor asserts that alignment and raises otherwise.
- `answer()` embeds the question, computes `scores = vectors @ q`, takes the top-k with
  `np.argpartition`, and sorts by **`(-score, doc_id, chunk_id)`** — the same tie-break
  BM25 uses. A constant-scoring embedder must not be able to produce a perfect ranking; that
  bug (`mteb#5092`) is exactly what deterministic tie-breaking prevents.
- Returns the retrieval-only `SystemOutput` contract, with `latency_ms` carrying `total` and
  a `query_embed` sub-stage.

### 3.4 The retriever dispatch

Add a `retriever` config key, read in `cli.py`'s `system_factory` / `_docs_system_factory`,
accepting `bm25` (**the default when the key is absent**), `tfidf`, `coverage`, `dense`.
Absence of the key must reproduce today's behaviour exactly, so every existing config hash,
stored run and evidence claim is unchanged.

### 3.5 The comparison, and its writable twin

Produce `run/panel_dense_matrix.csv` in the established
`item_id,system_id,correct` contract (`items/adapters/tabular.py`), covering the same 2,364
items as `run/panel54_matrix.csv`, over **2 chunkings × 3 top_ks × {bm25, dense}** = 12
systems.

**The twin comes first.** The definition of `correct` for a retrieval system is not written
down anywhere and the script that produced `panel54_matrix.csv` was never committed and is
lost; only its log survives. So:

1. Determine the definition by reproducing it. Regenerate the **six BM25 columns** that
   already exist in `panel54_matrix.csv` for the chunkings you pick, using your new
   generator.
2. If they do not match the committed CSV cell for cell, your generator is wrong. Fix it
   before embedding anything. **Do not proceed on a mismatch, and do not adjust the
   definition to make it match without saying exactly what you changed and why.**
3. Only once the BM25 columns reproduce may the dense columns be trusted.

**Commit the generator this time** as `tools/panel_dense.py`, with `--chunkings`,
`--top-ks`, `--limit`, `--resume` and a `--verify-bm25` mode that performs step 1 and exits
non-zero on any mismatch. The lost panel script is the reason this milestone exists in the
form it does.

### 3.6 The verdict, via the gate that already exists

Feed the per-item `correct` vectors into `gate.evaluate_gate` with base `bm25` and head
`dense` at matched chunking and top_k. Report per pair: `n`, both means, delta, the
confidence interval, the Holm-adjusted p-value, and the minimum detectable effect. If the
threshold sits below the MDE the gate raises `ThresholdBelowMDEError` and **no report is
produced** — that refusal is a finding and must be reported as one, not worked around by
lowering the threshold.

### 3.7 Negative controls, from the embeddings handbook

Each is a test that must fail on a planted defect:

- **Constant embedder.** Every vector identical. Recall must land at chance, not 1.0. This is
  the `mteb#5092` tie-break bug.
- **Score agreement is not ranking agreement.** Perturb vectors by float16-scale noise and
  assert the harness reports the top-1 flip count rather than only a mean score delta.
- **Shuffled vectors.** Permute the vector matrix against its chunks; recall must collapse.
  Proves the alignment assertion in 3.3 is real.
- **Stale cache.** Change the chunking with the cache in place; the loader must raise.
- **Determinism.** Two `DenseSystem` runs over the same cache produce byte-identical scores,
  which is what lets `identical_config` pass.

Run all four existing controls against a dense config and report each verdict.

### 3.8 Evidence

Add claims to `tools/evidence.py`'s `CLAIMS` tuple, each recomputing from committed
artifacts under `run/` only, never from the gitignored `run/golden.db`:
`dense.panel_shape`, `dense.vs_bm25` (the deltas and intervals), and `dense.bm25_twin` (the
step-1 reproduction). `make evidence` must pass with them.

Also fix the drift noticed in passing: `Makefile`'s `lint` target checks `src tests` while
CI checks `src tests tools`.

## 4. Closing checks

Report each with the command that produced it.

1. `ruff check src tests tools`, `mypy src`, `pytest` all green; test count and coverage at
   or above 644 and 91%.
2. `python tools/evidence.py --check` — all claims pass, including the new ones.
3. `python tools/panel_dense.py --verify-bm25` reproduces the existing BM25 columns cell for
   cell, or the milestone stops there with the discrepancy explained.
4. The four existing controls, each with its verdict against a dense config, and
   `identical_config` specifically passing at `max_abs_delta == 0.0`.
5. The five negative controls in 3.7, each shown going red on its planted defect.
6. The head-to-head table: per chunking and top_k, BM25 and dense means, delta, interval,
   adjusted p, MDE, and the verdict. **If dense loses, that is the finding and it leads the
   report.** The lexical baseline is what makes this a finding rather than a table. Carry no
   expectation in either direction into the measurement: this corpus, this chunker, this
   embedding model and this suite have never been measured together, and a result from any
   other corpus is not evidence about this one.
7. Embedding cost actually incurred: chunks embedded, wall-clock, cache size, and the
   speedup from batching against the 77ms serial baseline.

## 5. What would make this wrong

- If the BM25 twin cannot be reproduced, the definition of `correct` in the frozen panel is
  unknown, and every comparison against it is unsound. Stop and report.
- If `identical_config` fails on dense, the cache is not actually deterministic; find out why
  before reporting any comparison.
- If the MDE refusal fires on every pair, this suite cannot resolve retriever differences at
  2,364 items, and the honest output of this milestone is that fact plus the suite size that
  would be needed. `tools/mde.py` already computes that curve.

## 6. Budget

Two chunkings, 40,921 chunks, plus 2,364 question embeddings. Roughly an hour serially and
substantially less batched. Resumable, so a stop costs one shard.

## 7. Why the answer judge is not in this milestone

The user's three questions were: did retrieval find the right information, is the answer
correct and grounded, and does RAG beat no-RAG. The first is what this milestone measures.

The second **does not exist in this repo**. Every scorer in `score/` is
`requires_llm = False` by deliberate design, and the only judge,
`items/answerability.py`, judges whether an *item* is internally coherent, never whether a
*generated answer* is correct. `score/__init__.py` states plainly that correctness,
groundedness and judge calibration are out of scope until M4.

The third is **blocked on the second**: RAG Gain is accuracy with retrieval minus accuracy
closed book, and there is no accuracy metric over generated answers to difference. The
closed-book floor is already measured at 2 of 149, or 1.3%, under forced guessing.

Building that judge properly means calibrating it against human labels and implementing
`SPEC.md` §10.3's three countermeasures: position-swap augmentation, a verbosity-versus-score
correlation, and the self-preference refusal when judge and generator models match. It also
forces `no_retrieval`'s expected band to be re-measured, since correctness under no
retrieval will not be zero. That is a milestone of its own, and doing it badly is precisely
what this repository exists to prevent.
