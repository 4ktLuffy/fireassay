# Spike: first end-to-end run on the real corpus

`2026-08-28` · 1,181 gov.uk documents · 51 probe questions · M1 + M2 only, no LLM anywhere

Purpose: turn assumptions into measurements before building M3. Everything until now had
only ever run on a 12-document fixture.

## Headline: no single metric catches everything

`gate_mutation_score` for one config, five mutation operators, varying only the detector metric:

| detector metric | mutation score | blind to |
|---|---|---|
| `retrieval.recall@5` | **0.60** | `shuffle_topk`, `swap_ranking` |
| `retrieval.ndcg@5` | **0.80** | `truncate_topk` |
| `retrieval.mrr` | **0.80** | `truncate_topk` |
| `retrieval.bpref` | **0.60** | `shuffle_topk`, `swap_ranking` |

Gate on recall and you cannot detect a system that returns exactly the right documents in the
worst possible order — recall is order-insensitive by definition. Gate on nDCG or MRR and you
cannot detect truncation from top-5 to top-3, because it only costs you the hits that were
ranked 4th and 5th. The union of `recall@5` and `ndcg@5` catches all five.

**This is what the mutation score is for.** Every one of these setups reports respectable
numbers. Only the mutation score says which regressions each would miss.

## Performance — nothing is a bottleneck yet

| Stage | Time |
|---|---|
| load + chunk 1,181 docs | 0.05s → 24,584 chunks @512 |
| BM25 index build | 0.44s |
| single query | 23 ms |
| 6 configs × 51 questions | 9.2s |
| full control suite | 103s |
| mutation run (5 operators) | 9.3s |

**Flag:** controls take 103s at 51 questions, projecting to ~33 min at 1,000. `n_seeds: 50` was
tuned for a high-variance fixture; on a real corpus the calibration variance is exactly zero, so
most of that work is wasted. `n_seeds` should scale down as question count rises — each draw is
a mean over more questions, so its variance falls.

## σ_d measured: 0.269 (spec assumed 0.30)

Paired per-question deltas on `recall@5` between configs that genuinely differ.

At σ_d = 0.269, detecting a 1-point regression at α=0.05 / power 0.8 needs **~5,700 questions**.
A production set of 5,000 is the right order of magnitude — that claim now rests on a
measurement rather than an assumption.

**Also found: `k1` is inert.** Every pair of configs differing only in `k1` (1.2 vs 1.5) produced
**bit-identical** per-question scores. A swept axis that changes nothing should be reported as
such, not silently occupy a leaderboard row.

## `sd_floor` governs on a real corpus

| | fixture (12 docs) | real (1,181 docs) |
|---|---|---|
| `chance_level` | 0.376 | **0.0** |
| `chance_sd` | 0.105 | **0.0** |
| effective tolerance | measured | `5 × sd_floor × 0.469 = 0.047` |

Permuting gold spans across 24,584 chunks never retrieves anything relevant, so calibration
variance is exactly zero and `sd_floor: 0.02` does all the work. The control passed with a large
margin. Self-calibration matters on small corpora; on large ones the floor governs — that is
acceptable, but the floor is now a documented choice rather than an incidental default.

## Slicing caught a mislabelled taxonomy

`recall@5` by difficulty on the real corpus:

| difficulty | recall@5 |
|---|---|
| easy | **0.176** |
| medium | 0.647 |
| hard | 0.588 |

Inverted. The probe generator labelled difficulty by lexical overlap with the source passage,
but what actually drives BM25 recall here is whether the question contains the **document
title**. The overall mean of 0.47 hides a 3.7× spread.

Difficulty must be *validated*, not assigned — which is exactly what the curation rubric's
`difficulty_agrees` field exists for. Slice, do not average.

## Pooling bias, visible

`retrieval.judged_fraction@5 = 0.098`. Only ~10% of retrieved chunks carry any judgment, because
51 questions cannot judge a 1,181-document corpus. Recall numbers here are optimistic in exactly
the way the metric was added to expose.

## Bug found that unit tests could not

The `detector:` block in the mutants YAML was **documented, accepted without error, and silently
discarded** — `load_operators` parsed only the operators list, and the CLI's
`--detector-metric` default always won. Every mutation score was computed against `recall@5`
regardless of configuration.

This is the project's own thesis turned on itself: a confident number that was not measuring
what it claimed. Fixed by parsing the detector block, resolving precedence explicitly
(explicit flag → YAML → fallback), and **rejecting unrecognised config keys**.

## Caveat on the probes

The 51 questions are mechanical probes with real character offsets, not curated golden-set
questions. They exist to exercise the machinery at scale. Absolute recall values are optimistic
and must not be read as retrieval quality.

## Reproduce

    .venv/bin/python spike/make_questions.py
    .venv/bin/fireassay init --db spike/spike.db
    .venv/bin/fireassay questions import spike/questions.jsonl --db spike/spike.db
    .venv/bin/fireassay suite freeze --name govuk-spike --version 1.0.0 --db spike/spike.db
    .venv/bin/fireassay run --matrix spike/matrix.yaml --corpus data/corpus.jsonl --db spike/spike.db
    .venv/bin/fireassay mutate --suite govuk-spike@1.0.0 --config <hash> \
        --operators configs/mutants.example.yaml --corpus data/corpus.jsonl --db spike/spike.db
