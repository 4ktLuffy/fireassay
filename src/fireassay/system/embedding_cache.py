"""The embedding cache: content-addressed, resumable, and guarded —
**the reason `identical_config` can pass against a dense retriever at
all.**

`controls.identical_config` asserts `max_abs_delta == 0.0` across every
non-latency metric between two independently built systems. A dense
retriever that calls a live model for its vectors on each build is not
bit-deterministic and cannot satisfy that. So this cache is not a speed
optimisation that happens to also help; it is the mechanism that makes a
dense system a legitimate subject for the control at all. Both the corpus
matrix (`EmbeddingCache`) and the per-question vectors (`QueryCache`)
come from disk for exactly that reason.

**A disagreeing cache is refused, never reused and never silently
rebuilt.** `EmbeddingCache.load` compares every field of the stored
`metadata.json` against the run asking for it — model digest, chunking,
`corpus_hash`, chunk count, and a sha256 over the ordered `chunk_id`
list — and raises `StaleCacheError` on the first mismatch. Silently
re-embedding would burn an hour without saying why; silently reusing
would rank one chunking's questions against another chunking's vectors,
which is a wrong answer that looks exactly like a right one.

**Shards are written atomically.** `shards/<start>-<stop>.npy` goes to a
temp file and then `os.replace`, so a kill mid-run costs at most the
shard in flight and `build` picks up from the next missing one. An hour
of local model time deserves the same checkpoint discipline
`.cache/llm/` already gets for the generation run.

`.cache/embeddings/` is gitignored. Reproducing any dense number in this
repo therefore needs a running Ollama and roughly an hour of model time,
exactly as reproducing the golden set needs the generation run.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fireassay.llm.ollama import ModelRef
from fireassay.system.embedding import Embed, EmbeddingError

#: Default cache root. Gitignored (`.cache/`), like every other on-disk
#: model checkpoint in this repo.
DEFAULT_CACHE_ROOT = Path(".cache/embeddings")

#: Chunks per shard. 512 x 1024 float32 is 2 MB — small enough that a
#: kill costs seconds, large enough that a 28,408-chunk corpus is 56
#: files rather than thousands.
DEFAULT_SHARD_SIZE = 512


class StaleCacheError(RuntimeError):
    """The cache on disk disagrees with the run asking for it. Carries
    the field that disagreed, so a caller reports *what* moved rather
    than only that something did."""


def chunk_ids_digest(chunk_ids: Iterable[str]) -> str:
    """sha256 over the ordered `chunk_id` list, newline-joined.

    Ordered, not sorted: the cache's rows are positional — row `i` is
    `chunks[i]` — so a corpus that re-orders without changing membership
    is a different cache, not the same one. `DenseSystem`'s alignment
    assertion checks the count; this checks the identity."""
    h = hashlib.sha256()
    for chunk_id in chunk_ids:
        h.update(chunk_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """Everything that must agree for a stored matrix to be the matrix
    this run wants.

    `digest()` folds only the four fields M-DENSE-SPEC.md §3.2 names
    (`model_digest`, `chunk_size`, `chunk_overlap`, `corpus_hash`) into
    the directory name. `n_chunks`/`chunk_ids_sha256` are *also* checked
    on load but are deliberately not in the path: a chunk-id list that
    disagrees at the same key is a corruption to report, not a second
    directory to quietly start filling.
    """

    model_name: str
    model_digest: str
    chunk_size: int
    chunk_overlap: int
    corpus_hash: str
    chunk_ids: tuple[str, ...]

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_ids)

    @property
    def chunk_ids_sha256(self) -> str:
        return chunk_ids_digest(self.chunk_ids)

    def digest(self) -> str:
        payload = "\x1f".join(
            [self.model_digest, str(self.chunk_size), str(self.chunk_overlap), self.corpus_hash]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class CacheMetadata:
    """`metadata.json`: the key, plus the dimension actually observed and
    the shard layout the matrix was written in.

    `shard_size` is stored rather than assumed. It is not part of the
    key — a cache written in 5-row shards holds the same vectors as one
    written in 512-row shards — but `load` has to know it to name the
    files, and a loader that assumed a default would read a cache built
    at any other size as *missing*, then silently re-embed it. That was a
    real bug here, caught by `test_build_then_load_round_trips_in_chunk_order`.
    """

    model_name: str
    model_digest: str
    chunk_size: int
    chunk_overlap: int
    corpus_hash: str
    n_chunks: int
    chunk_ids_sha256: str
    dimension: int
    shard_size: int = DEFAULT_SHARD_SIZE

    def matches(self, key: CacheKey) -> str | None:
        """The name of the first field of `key` this metadata disagrees
        with, or `None` if every one agrees."""
        for field, want, got in (
            ("model_digest", key.model_digest, self.model_digest),
            ("model_name", key.model_name, self.model_name),
            ("chunk_size", key.chunk_size, self.chunk_size),
            ("chunk_overlap", key.chunk_overlap, self.chunk_overlap),
            ("corpus_hash", key.corpus_hash, self.corpus_hash),
            ("n_chunks", key.n_chunks, self.n_chunks),
            ("chunk_ids_sha256", key.chunk_ids_sha256, self.chunk_ids_sha256),
        ):
            if want != got:
                return f"{field}={got!r} but this run asks for {field}={want!r}"
        return None


def _shard_bounds(n: int, shard_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + shard_size, n)) for start in range(0, n, shard_size)]


class EmbeddingCache:
    """One `(model, chunking, corpus)` matrix on disk.

    Construct with `open` (which derives the directory from `key.digest()`)
    or directly with an explicit `path` — the latter is what lets a test
    point a *mismatched* key at an existing directory and prove `load`
    refuses it, which is the whole point of the guard.
    """

    def __init__(self, path: Path, key: CacheKey) -> None:
        self.path = path
        self.key = key

    @classmethod
    def open(cls, key: CacheKey, *, root: Path = DEFAULT_CACHE_ROOT) -> EmbeddingCache:
        return cls(Path(root) / key.digest(), key)

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def shards_path(self) -> Path:
        return self.path / "shards"

    def read_metadata(self) -> CacheMetadata | None:
        if not self.metadata_path.exists():
            return None
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return CacheMetadata(**raw)

    def _write_metadata(self, metadata: CacheMetadata) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.metadata_path, json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n")

    @property
    def chunk_ids_path(self) -> Path:
        """The ordered `chunk_id` list this matrix's rows are, written
        beside `metadata.json` (which stores only its sha256). Needed by
        `resolve_vectors` to select a subset corpus's rows out of a
        superset corpus's matrix -- see that function."""
        return self.path / "chunk_ids.json"

    def read_chunk_ids(self) -> list[str] | None:
        if not self.chunk_ids_path.exists():
            return None
        ids: list[str] = json.loads(self.chunk_ids_path.read_text(encoding="utf-8"))
        return ids

    def _check(self, metadata: CacheMetadata) -> None:
        disagreement = metadata.matches(self.key)
        if disagreement is not None:
            raise StaleCacheError(
                f"embedding cache {self.path} was built with {disagreement}; refusing to reuse it. "
                "Delete the directory to rebuild."
            )

    def build(
        self,
        texts: Sequence[str],
        embed: Embed,
        *,
        shard_size: int = DEFAULT_SHARD_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> CacheMetadata:
        """Fill every missing shard, in order, and write `metadata.json`.

        Resumable: a shard file that already exists and has the expected
        row count and width is skipped, so re-running after a kill costs
        only the shard that was in flight. `progress(done, total)` is
        called after each shard actually embedded (never for a skipped
        one), so a caller's ETA reflects work, not files.

        Raises `StaleCacheError` if a `metadata.json` is already present
        and disagrees — resuming into somebody else's matrix is exactly
        the failure this class exists to prevent.
        """
        if len(texts) != self.key.n_chunks:
            raise ValueError(
                f"build: given {len(texts)} text(s) but the cache key names {self.key.n_chunks} chunk(s)"
            )
        existing = self.read_metadata()
        if existing is not None:
            self._check(existing)
            if existing.shard_size != shard_size:
                raise StaleCacheError(
                    f"embedding cache {self.path} was written in {existing.shard_size}-row shards but "
                    f"this build asks for {shard_size}; resuming across layouts would mix two shard "
                    "grids. Delete the directory to rebuild."
                )
        self.shards_path.mkdir(parents=True, exist_ok=True)

        dimension = existing.dimension if existing is not None else 0
        done = 0
        for start, stop in _shard_bounds(len(texts), shard_size):
            shard = self.shards_path / f"{start}-{stop}.npy"
            block = _read_shard(shard, stop - start)
            if block is None:
                block = embed(texts[start:stop])
                if block.shape[0] != stop - start:
                    raise EmbeddingError(
                        f"embedder returned {block.shape[0]} vector(s) for {stop - start} text(s)"
                    )
                _atomic_save(shard, np.ascontiguousarray(block, dtype=np.float32))
                done += stop - start
                if progress is not None:
                    progress(done, len(texts))
            width = int(block.shape[1])
            if dimension == 0:
                dimension = width
            elif width != dimension:
                raise EmbeddingError(
                    f"shard {shard.name} is {width}-dimensional but the cache is {dimension}-dimensional"
                )

        metadata = CacheMetadata(
            model_name=self.key.model_name,
            model_digest=self.key.model_digest,
            chunk_size=self.key.chunk_size,
            chunk_overlap=self.key.chunk_overlap,
            corpus_hash=self.key.corpus_hash,
            n_chunks=self.key.n_chunks,
            chunk_ids_sha256=self.key.chunk_ids_sha256,
            dimension=dimension,
            shard_size=shard_size,
        )
        _atomic_write(self.chunk_ids_path, json.dumps(list(self.key.chunk_ids)))
        self._write_metadata(metadata)
        return metadata

    def load(self) -> np.ndarray:
        """The full `(n_chunks, dimension)` float32 matrix, in chunk
        order. Raises `StaleCacheError` if the cache is absent, disagrees
        with this run's key, or is missing a shard."""
        metadata = self.read_metadata()
        if metadata is None:
            raise StaleCacheError(
                f"no embedding cache at {self.path}; build it first "
                "(.venv/bin/python tools/embed_corpus.py)"
            )
        self._check(metadata)
        matrix = self._load_raw(metadata)
        if matrix.shape != (metadata.n_chunks, metadata.dimension):
            raise StaleCacheError(
                f"embedding cache {self.path} holds a {matrix.shape} matrix but its metadata says "
                f"{(metadata.n_chunks, metadata.dimension)}"
            )
        return matrix

    def load_subset(self, chunk_ids: Sequence[str]) -> np.ndarray | None:
        """This cache's rows for `chunk_ids`, in that order, or `None` if
        it does not hold every one of them.

        Used only by `resolve_vectors`; see there for why selecting rows
        out of a larger corpus's matrix is exact rather than approximate.
        Deliberately does **not** call `_check`: the whole point is that
        the stored `corpus_hash`/`chunk_ids_sha256` differ. What is
        checked instead is the one thing that must hold — every requested
        chunk id is present — plus the model and chunking, which
        `resolve_vectors` establishes before calling this.
        """
        stored = self.read_chunk_ids()
        metadata = self.read_metadata()
        if stored is None or metadata is None:
            return None
        index = {chunk_id: i for i, chunk_id in enumerate(stored)}
        try:
            rows = [index[chunk_id] for chunk_id in chunk_ids]
        except KeyError:
            return None
        matrix = self._load_raw(metadata)
        return np.ascontiguousarray(matrix[rows], dtype=np.float32)

    def _load_raw(self, metadata: CacheMetadata) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start, stop in _shard_bounds(metadata.n_chunks, metadata.shard_size):
            shard = self.shards_path / f"{start}-{stop}.npy"
            block = _read_shard(shard, stop - start)
            if block is None:
                raise StaleCacheError(
                    f"embedding cache {self.path} is missing or truncated at shard {shard.name}; "
                    "re-run the build to fill it"
                )
            rows.append(block)
        return np.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]

    def size_bytes(self) -> int:
        """Total bytes on disk, for a cost report."""
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())


class QueryCache:
    """Per-question vectors, one atomic `.npy` per `(model, text)`.

    Separate from `EmbeddingCache` because questions are not chunks: they
    are keyed on the text alone, shared across every chunking (the same
    2,364 questions are embedded once, not once per corpus matrix), and
    arrive one at a time from `DenseSystem.answer` rather than in a
    planned sweep.

    `wrap` returns an `Embed` that serves hits from disk and calls through
    only for misses, so two `DenseSystem` instances over the same cache
    score byte-identically — which is what `identical_config` checks.
    """

    def __init__(self, path: Path, model: ModelRef) -> None:
        self.path = Path(path) / model.digest.replace(":", "-")
        self.model = model
        self._memo: dict[str, np.ndarray] = {}

    @classmethod
    def open(cls, model: ModelRef, *, root: Path = DEFAULT_CACHE_ROOT) -> QueryCache:
        return cls(Path(root) / "queries", model)

    def _vector_path(self, text: str) -> Path:
        return self.path / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.npy"

    def wrap(self, embed: Embed) -> Embed:
        def cached(texts: Sequence[str]) -> np.ndarray:
            missing = [t for t in texts if t not in self._memo and not self._vector_path(t).exists()]
            if missing:
                fresh = embed(missing)
                self.path.mkdir(parents=True, exist_ok=True)
                for text, row in zip(missing, fresh, strict=True):
                    vector = np.ascontiguousarray(row, dtype=np.float32)
                    _atomic_save(self._vector_path(text), vector)
                    self._memo[text] = vector
            out: list[np.ndarray] = []
            for text in texts:
                memoised = self._memo.get(text)
                if memoised is None:
                    memoised = np.ascontiguousarray(np.load(self._vector_path(text)), dtype=np.float32)
                    self._memo[text] = memoised
                out.append(memoised)
            return np.stack(out, axis=0)

        return cached

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    """Write a `.npy` via a temp file and `os.replace`, so a kill can
    never leave a half-written shard that a later run would read as
    complete."""
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def _read_shard(path: Path, expected_rows: int) -> np.ndarray | None:
    """A complete shard, or `None` if it is absent or unreadable. A shard
    present but with the wrong row count is a corruption, not a miss, and
    raises."""
    if not path.exists():
        return None
    block = np.load(path)
    if block.ndim != 2 or block.shape[0] != expected_rows:
        raise StaleCacheError(
            f"embedding shard {path} holds shape {block.shape}, expected {expected_rows} row(s)"
        )
    return np.ascontiguousarray(block, dtype=np.float32)


def resolve_vectors(key: CacheKey, *, root: Path = DEFAULT_CACHE_ROOT) -> tuple[np.ndarray, str]:
    """The `(n_chunks, d)` matrix for `key`, and how it was obtained
    (`"exact"` or `"derived from <dir>"`).

    **Why a derivation exists at all.** `controls.corpus_ablation` drops
    documents and rebuilds the system over the remaining ones. That corpus
    has a different `corpus_hash`, so it has no cache of its own — and a
    dense retriever that could only load an exact cache would either fail
    the control or have to embed inside the run, which is an hour of model
    time nobody asked for and, worse, a live model call in the middle of a
    determinism-sensitive control.

    **Why the derivation is exact, not an approximation.**
    `corpus.chunk_corpus` windows each document independently: a chunk's
    text, `chunk_id` and offsets depend only on that one document and the
    `(size, overlap)` pair. Removing other documents therefore leaves every
    surviving chunk byte-identical, so its embedding is byte-identical
    too. Selecting rows from the full corpus's matrix by `chunk_id` gives
    exactly the vectors a fresh embedding run would have produced —
    `tests/test_embedding_cache.py::test_derived_subset_equals_a_freshly_built_subset`
    checks that against a real rebuild rather than asserting it.

    The search is scoped to caches with the **same model digest and the
    same chunking**, and every requested `chunk_id`
    must be present; anything else raises rather than returning a partial
    matrix.
    """
    cache = EmbeddingCache.open(key, root=root)
    if cache.read_metadata() is not None:
        return cache.load(), "exact"

    for candidate_dir in sorted(Path(root).glob("*")):
        if not (candidate_dir / "metadata.json").is_file() or candidate_dir == cache.path:
            continue
        metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
        same_namespace = (
            metadata.get("model_digest") == key.model_digest
            and metadata.get("chunk_size") == key.chunk_size
            and metadata.get("chunk_overlap") == key.chunk_overlap
        )
        if not same_namespace:
            continue
        subset = EmbeddingCache(candidate_dir, key).load_subset(key.chunk_ids)
        if subset is not None:
            return subset, f"derived from {candidate_dir.name}"

    raise StaleCacheError(
        f"no embedding cache at {cache.path}, and no cache under {root} for model "
        f"{key.model_digest[:12]} at chunking {key.chunk_size}-{key.chunk_overlap} holds every one of "
        f"this corpus's {key.n_chunks} chunk id(s). Build it first: "
        ".venv/bin/python tools/embed_corpus.py"
    )
