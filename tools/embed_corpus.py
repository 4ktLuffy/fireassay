#!/usr/bin/env python3
"""Fill the embedding cache: one corpus matrix per chunking, plus the
panel's question vectors.

Resumable and cheap to interrupt (`system.embedding_cache` writes shards
atomically), so a kill costs one shard rather than the run. Nothing here
is reachable from a clean clone: it needs a local Ollama serving
`qwen3-embedding:0.6b` and roughly an hour of model time, exactly as
reproducing the golden set needs the generation run.

    ollama serve &
    .venv/bin/python tools/embed_corpus.py --chunkings 1024-128,512-128
    .venv/bin/python tools/embed_corpus.py --probe-serial 20

`--probe-serial N` measures this machine's own serial baseline through the
deprecated single-`prompt` `/api/embeddings` endpoint and prints the
speedup batching actually bought, rather than quoting a remembered
figure.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from fireassay.system.corpus import Doc, chunk_corpus, corpus_hash, load_corpus
from fireassay.system.embedding import (
    DEFAULT_BASE_URL,
    DEFAULT_EMBED_MODEL,
    OllamaEmbedder,
)
from fireassay.system.embedding_cache import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_SHARD_SIZE,
    CacheKey,
    EmbeddingCache,
    QueryCache,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_chunkings(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        size, _, overlap = part.strip().partition("-")
        out.append((int(size), int(overlap)))
    return out


def question_texts(db: Path) -> list[str]:
    """The panel's 2,364 question texts, in the same order and from the
    same filter stage `tools/panel_dense.py` reads them."""
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "select text from candidate where id in (select candidate_id from filter_result "
            "where stage='balance' and kept=1)"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def embed_chunking(
    docs: Sequence[Doc],
    chunking: tuple[int, int],
    embedder: OllamaEmbedder,
    *,
    root: Path,
    shard_size: int,
) -> tuple[EmbeddingCache, int, float]:
    """Build (or resume) one chunking's matrix. Returns the cache, the
    number of chunks actually embedded this run, and the seconds spent."""
    size, overlap = chunking
    chunks = list(chunk_corpus(docs, size=size, overlap=overlap))
    key = CacheKey(
        model_name=embedder.model.name,
        model_digest=embedder.model.digest,
        chunk_size=size,
        chunk_overlap=overlap,
        corpus_hash=corpus_hash(docs),
        chunk_ids=tuple(c.chunk_id for c in chunks),
    )
    cache = EmbeddingCache.open(key, root=root)
    print(f"{size}-{overlap}: {len(chunks)} chunks -> {cache.path}", flush=True)

    started = time.perf_counter()
    embedded = 0

    def progress(done: int, total: int) -> None:
        nonlocal embedded
        embedded = done
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        print(f"  {done}/{total}  {elapsed:.0f}s  {rate:.0f} chunk/s", flush=True)

    cache.build([c.text for c in chunks], embedder, shard_size=shard_size, progress=progress)
    return cache, embedded, time.perf_counter() - started


def probe_serial(texts: Sequence[str], model: str, base_url: str, timeout: float = 120.0) -> float:
    """Mean seconds per text through the **deprecated** single-`prompt`
    `/api/embeddings` endpoint — the baseline the batched figures are a
    speedup over, measured on this machine rather than quoted."""
    started = time.perf_counter()
    for text in texts:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/embeddings",
            data=json.dumps({"model": model, "prompt": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        if "embedding" not in body:
            raise SystemExit(f"ollama /api/embeddings returned no 'embedding': keys {sorted(body)}")
    return (time.perf_counter() - started) / len(texts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=_REPO_ROOT / "data" / "corpus.jsonl")
    parser.add_argument("--db", type=Path, default=_REPO_ROOT / "run" / "golden.db")
    parser.add_argument("--chunkings", type=str, default="1024-128,512-128")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--cache", type=Path, default=_REPO_ROOT / DEFAULT_CACHE_ROOT)
    parser.add_argument("--skip-questions", action="store_true")
    parser.add_argument(
        "--probe-serial",
        type=int,
        default=0,
        help="measure the /api/embeddings (singular) baseline over N texts and exit",
    )
    args = parser.parse_args(argv)

    docs = load_corpus(args.corpus)
    if args.probe_serial:
        sample = [c.text for c in chunk_corpus(docs, size=1024, overlap=128)][: args.probe_serial]
        per_text = probe_serial(sample, args.model, args.base_url)
        print(f"serial /api/embeddings baseline: {per_text * 1000:.1f} ms/text over {len(sample)} text(s)")
        return 0

    embedder = OllamaEmbedder.resolve(args.model, base_url=args.base_url, batch_size=args.batch_size)
    print(f"model {embedder.model.name}@{embedder.model.digest[:16]} batch_size={args.batch_size}")

    total_embedded = 0
    total_seconds = 0.0
    total_bytes = 0
    for chunking in _parse_chunkings(args.chunkings):
        cache, embedded, seconds = embed_chunking(
            docs, chunking, embedder, root=args.cache, shard_size=args.shard_size
        )
        total_embedded += embedded
        total_seconds += seconds
        total_bytes += cache.size_bytes()
        print(f"  done: {embedded} embedded this run in {seconds:.0f}s, {cache.size_bytes() / 1e6:.1f} MB")

    if not args.skip_questions:
        texts = question_texts(args.db)
        query_cache = QueryCache.open(embedder.model, root=args.cache)
        cached_embed = query_cache.wrap(embedder)
        started = time.perf_counter()
        for start in range(0, len(texts), args.batch_size):
            cached_embed(texts[start : start + args.batch_size])
            if (start // args.batch_size) % 10 == 0:
                print(f"  questions {start}/{len(texts)}  {time.perf_counter() - started:.0f}s", flush=True)
        seconds = time.perf_counter() - started
        total_seconds += seconds
        total_bytes += query_cache.size_bytes()
        print(f"questions: {len(texts)} in {seconds:.0f}s, {query_cache.size_bytes() / 1e6:.1f} MB")

    print(
        f"TOTAL: {total_embedded} chunk(s) embedded this run, {total_seconds:.0f}s wall clock, "
        f"{total_bytes / 1e6:.1f} MB on disk"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
