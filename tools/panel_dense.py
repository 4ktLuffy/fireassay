#!/usr/bin/env python3
"""Build the dense-vs-BM25 response matrix -- and, first, prove this
generator reproduces the frozen one.

`run/panel54_matrix.csv` is the frozen 2,364-item x 54-system panel every
`panel.*` evidence claim reads. The script that produced it was never
committed; only `run/panel54_full.log` survives. So the *definition* of
its `correct` column -- the single number this whole milestone compares
against -- was not written down anywhere.

`--verify-bm25` recovers that definition the only honest way: regenerate
the BM25 columns for the chunkings and top_ks asked for, and compare them
against the committed CSV cell for cell. A single disagreeing cell exits
non-zero and no dense column is trusted (M-DENSE-SPEC.md §3.5).

**The definition, recovered and stated.** A cell is `1` iff at least one
of the system's top-`k` retrieved chunks overlaps the item's evidence span
by >= 1 character within the same document:

    any(ch.doc_id == doc_id and ch.char_start < char_end and ch.char_end > char_start
        for ch in retrieved[:k])

That is `score.retrieval.RetrievalScorer`'s `retrieval.recall@k > 0` at
`ScoringContext.overlap_min_chars == 1` -- every panel item has exactly
one evidence span, so its recall is 0.0 or 1.0 and nothing else.
`tests/test_panel_dense.py::test_correct_matches_retrieval_scorer_recall`
pins that equivalence against the real scorer rather than asserting it
here in prose. The same expression is already committed, independently, in
`tools/measure_detector_recall.py`'s `responses_for`, which is where it
was recovered from.

**Where the 2,364 items come from.** `run/golden.db`'s `question` table is
empty and `candidate` holds 3,054 rows; the panel's items are the
candidates kept by every filter stage, which `filter_result` records as
`stage='balance' AND kept=1` -- exactly 2,364 rows, in `candidate` rowid
order, which is byte-for-byte the item order of `run/panel54_matrix.csv`,
`run/full_matrix.json` and the committed `run/all_kept_ids.txt`. Each
item's evidence span is the candidate's own `(source_doc_id, char_start,
char_end)`; there is no `evidence_span` row for any of them.

    .venv/bin/python tools/panel_dense.py --verify-bm25
    .venv/bin/python tools/panel_dense.py --retrievers bm25,dense --resume

`run/golden.db` is gitignored, so this script is not reproducible from a
clean clone -- which is precisely why its *output* (`run/panel_dense_
matrix.csv`, `run/panel_dense_twin.json`) is committed and why every
`dense.*` claim in `tools/evidence.py` reads those files instead of
re-running this one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from fireassay.models import Question, RetrievedChunk
from fireassay.system.base import System
from fireassay.system.corpus import Chunk, Doc, chunk_corpus, load_corpus

#: The two chunkings M-DENSE-SPEC.md §6 budgets for: 12,513 and 28,408
#: chunks. Both are `system.panel.CHUNKINGS` members, so their BM25
#: columns exist in `run/panel54_matrix.csv` and can be verified.
DEFAULT_CHUNKINGS: tuple[tuple[int, int], ...] = ((1024, 128), (512, 128))

#: Unchanged from `system.panel.TOP_KS`, for the same reason.
DEFAULT_TOP_KS: tuple[int, ...] = (3, 5, 10)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN = _REPO_ROOT / "run"
_PARTS = _REPO_ROOT / ".cache" / "panel_dense"


@dataclass(frozen=True)
class Item:
    """One panel item: the question text a retriever sees, and the single
    evidence span a cell is scored against."""

    item_id: str
    question: str
    doc_id: str
    char_start: int
    char_end: int


def load_items(db: Path) -> list[Item]:
    """The 2,364 kept candidates, in the order the frozen panel used --
    see this module's docstring."""
    conn = sqlite3.connect(db)
    try:
        rows = list(
            conn.execute(
                "select id, text, source_doc_id, char_start, char_end from candidate "
                "where id in (select candidate_id from filter_result "
                "where stage='balance' and kept=1)"
            )
        )
    finally:
        conn.close()
    return [Item(item_id=r[0], question=r[1], doc_id=r[2], char_start=r[3], char_end=r[4]) for r in rows]


def as_question(item: Item) -> Question:
    """The `Question` a retriever is handed. Only `text` can affect any
    retriever's ranking; the rest are the required `Question` fields,
    carried at the same values `tools/measure_detector_recall.py` uses so
    two generators of the same cell cannot silently differ."""
    return Question(
        text=item.question,
        qtype="factual",
        difficulty="easy",
        reference_answer=None,
        provenance="synthetic",
        generator="probe@1",
    )


def is_correct(retrieved: Sequence[RetrievedChunk], item: Item, k: int) -> bool:
    """The recovered `correct` definition -- see this module's docstring.

    `k` truncates by `RetrievedChunk.rank` (1-based), never by list
    position, so a caller that hands over a differently-ordered sequence
    still scores the same top-k the ranking actually named."""
    return any(
        ch.doc_id == item.doc_id and ch.char_start < item.char_end and ch.char_end > item.char_start
        for ch in retrieved
        if ch.rank <= k
    )


def columns_for_system(
    system: System, items: Sequence[Item], top_ks: Sequence[int], *, progress_every: int = 200
) -> dict[int, list[bool]]:
    """Score every item once and derive one column per `k` in `top_ks`.

    `system` must have been built at `top_k >= max(top_ks)`. Truncating
    one ranking to several `k`s is exactly what building a separate system
    per `k` does -- `BM25System.answer` sorts the full corpus and slices
    `ranked[:top_k]`, so the top-3 of a top-10 system is the top-3 system's
    whole output -- and that equivalence is pinned by
    `tests/test_panel_dense.py::test_truncating_one_ranking_equals_separate_systems`
    rather than assumed here. It is what makes this script minutes rather
    than the frozen panel's 172."""
    out: dict[int, list[bool]] = {k: [] for k in top_ks}
    start = time.perf_counter()
    for index, item in enumerate(items, start=1):
        retrieved = system.answer(as_question(item)).retrieved
        for k in top_ks:
            out[k].append(is_correct(retrieved, item, k))
        if progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"    {index}/{len(items)}  {elapsed:.0f}s", flush=True)
    return out


def system_id(retriever: str, chunking: tuple[int, int], k: int) -> str:
    """`system.panel.build_panel`'s own id scheme, unchanged, so a column
    here lines up with the same-named column in `run/panel54_matrix.csv`."""
    size, overlap = chunking
    return f"{retriever}/{size}-{overlap}/k{k}"


def _part_path(retriever: str, chunking: tuple[int, int], n_items: int) -> Path:
    size, overlap = chunking
    return _PARTS / f"{retriever}_{size}-{overlap}_n{n_items}.json"


def _load_part(path: Path, items: Sequence[Item], top_ks: Sequence[int]) -> dict[int, list[bool]] | None:
    """A resumable part file, or `None` if absent. Raises if it is present
    but describes different items or top_ks -- a part that does not match
    the run asking for it is a stale artifact to refuse, never one to
    silently reuse (the same rule `system.embedding_cache` applies)."""
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob["item_ids"] != [i.item_id for i in items]:
        raise SystemExit(f"panel_dense: {path} was built over different items; delete it to rebuild")
    stored = {int(k): [bool(x) for x in v] for k, v in blob["columns"].items()}
    missing = [k for k in top_ks if k not in stored]
    if missing:
        raise SystemExit(f"panel_dense: {path} has no column for top_k {missing}; delete it to rebuild")
    return stored


def _save_part(path: Path, items: Sequence[Item], columns: dict[int, list[bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "item_ids": [i.item_id for i in items],
                "columns": {str(k): [int(x) for x in v] for k, v in columns.items()},
            }
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def build_system(
    retriever: str, chunks: Sequence[Chunk], top_k: int, chunking: tuple[int, int], docs: Sequence[Doc]
) -> System:
    """Build one panel system **through the shipped config dispatch**
    (`cli._build_retriever`), not through a constructor called directly
    here.

    That is the point: `run/panel_dense_matrix.csv`'s columns must be
    produced by the same code path a `fireassay run` produces them with,
    or the matrix measures this script rather than the system. It is also
    what makes the `--verify-bm25` twin meaningful — the BM25 columns it
    reproduces come out of the dispatch, so a regression in the dispatch
    would break the reproduction rather than hide behind a direct call.

    `cli` is imported lazily so `--verify-bm25` can run, and fail, without
    Ollama, numpy vectors or a populated embedding cache existing at all.
    Note that an absent `retriever` key means `bm25`, so the bm25 path
    here passes a config dict that is byte-identical to the ones every
    pre-M-DENSE run was recorded under.
    """
    from fireassay.cli import _build_retriever

    size, overlap = chunking
    config: dict[str, object] = {"top_k": top_k, "chunk_size": size, "chunk_overlap": overlap}
    if retriever != "bm25":
        config["retriever"] = retriever
    return _build_retriever(config, chunks, docs)


def read_frozen_columns(path: Path, wanted: Iterable[str]) -> dict[str, dict[str, bool]]:
    """`{system_id: {item_id: correct}}` for `wanted` columns of a
    response-matrix CSV, read directly rather than through
    `items.adapters.tabular.load_responses_csv` so one column can be
    pulled from a 127k-row file without materialising all 54."""
    want = set(wanted)
    out: dict[str, dict[str, bool]] = {s: {} for s in want}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["system_id"]
            if sid in want:
                out[sid][row["item_id"]] = row["correct"] == "1"
    return out


def write_matrix(
    path: Path, items: Sequence[Item], system_ids: Sequence[str], columns: dict[str, list[bool]]
) -> None:
    """Write the `item_id,system_id,correct` contract
    (`items.adapters.tabular`), item-major, system order as given."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "system_id", "correct"])
        for index, item in enumerate(items):
            for sid in system_ids:
                writer.writerow([item.item_id, sid, 1 if columns[sid][index] else 0])


def _parse_chunkings(raw: str) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        size, _, overlap = part.strip().partition("-")
        out.append((int(size), int(overlap)))
    return tuple(out)


def _parse_top_ks(raw: str) -> tuple[int, ...]:
    return tuple(int(p) for p in raw.split(","))


def _relative_if_inside(path: Path) -> Path:
    """`path` relative to the repo root when it is under it, else `path`
    itself -- the committed artifact should say `run/panel54_matrix.csv`,
    while a test pointing at a tmp_path should still get a report."""
    resolved = path.resolve()
    return resolved.relative_to(_REPO_ROOT) if resolved.is_relative_to(_REPO_ROOT) else resolved


def verify_bm25(
    items: Sequence[Item],
    chunkings: Sequence[tuple[int, int]],
    top_ks: Sequence[int],
    columns: dict[str, list[bool]],
    frozen_path: Path,
) -> tuple[int, int, dict[str, object]]:
    """Compare freshly generated BM25 columns against the frozen panel,
    cell for cell. Returns `(n_compared, n_mismatch, report)`."""
    wanted = [system_id("bm25", c, k) for c in chunkings for k in top_ks]
    frozen = read_frozen_columns(frozen_path, wanted)
    per_column: dict[str, dict[str, int]] = {}
    n_compared = 0
    n_mismatch = 0
    examples: list[str] = []
    for sid in wanted:
        frozen_column = frozen[sid]
        if not frozen_column:
            raise SystemExit(f"panel_dense: {frozen_path} has no column {sid!r} to verify against")
        mismatches = 0
        for index, item in enumerate(items):
            if item.item_id not in frozen_column:
                raise SystemExit(f"panel_dense: {frozen_path} has no row for item {item.item_id!r}")
            n_compared += 1
            if columns[sid][index] != frozen_column[item.item_id]:
                mismatches += 1
                if len(examples) < 10:
                    examples.append(
                        f"{sid} {item.item_id}: regenerated={int(columns[sid][index])} "
                        f"frozen={int(frozen_column[item.item_id])}"
                    )
        per_column[sid] = {"compared": len(items), "mismatch": mismatches}
        n_mismatch += mismatches
    report: dict[str, object] = {
        "frozen_matrix": str(_relative_if_inside(frozen_path)),
        "n_items": len(items),
        "n_columns": len(wanted),
        "n_cells_compared": n_compared,
        "n_cells_mismatched": n_mismatch,
        "per_column": per_column,
        "examples": examples,
        "definition": (
            "correct = any(ch.doc_id == doc_id and ch.char_start < char_end and "
            "ch.char_end > char_start for ch in retrieved[:k]) -- i.e. "
            "retrieval.recall@k > 0 at overlap_min_chars=1"
        ),
    }
    return n_compared, n_mismatch, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", type=Path, default=_RUN / "golden.db")
    parser.add_argument("--corpus", type=Path, default=_REPO_ROOT / "data" / "corpus.jsonl")
    parser.add_argument("--out", type=Path, default=_RUN / "panel_dense_matrix.csv")
    parser.add_argument("--twin-out", type=Path, default=_RUN / "panel_dense_twin.json")
    parser.add_argument("--frozen", type=Path, default=_RUN / "panel54_matrix.csv")
    parser.add_argument(
        "--chunkings", type=str, default=",".join(f"{a}-{b}" for a, b in DEFAULT_CHUNKINGS)
    )
    parser.add_argument("--top-ks", type=str, default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--retrievers", type=str, default="bm25,dense")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items (smoke runs)")
    parser.add_argument("--resume", action="store_true", help="reuse per-(retriever, chunking) part files")
    parser.add_argument(
        "--verify-bm25",
        action="store_true",
        help="regenerate the BM25 columns only, compare them to --frozen cell for cell, and exit "
        "non-zero on any mismatch",
    )
    args = parser.parse_args(argv)

    chunkings = _parse_chunkings(args.chunkings)
    top_ks = _parse_top_ks(args.top_ks)
    retrievers = ("bm25",) if args.verify_bm25 else tuple(r.strip() for r in args.retrievers.split(","))
    for r in retrievers:
        if r not in {"bm25", "dense"}:
            parser.error(f"unknown retriever {r!r}; expected bm25 or dense")

    items = load_items(args.db)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items; {len(retrievers)} retriever(s) x {len(chunkings)} chunking(s)", flush=True)

    docs = load_corpus(args.corpus)
    max_k = max(top_ks)
    columns: dict[str, list[bool]] = {}
    system_ids: list[str] = []

    for chunking in chunkings:
        size, overlap = chunking
        chunks = list(chunk_corpus(docs, size=size, overlap=overlap))
        print(f"  chunking {size}-{overlap}: {len(chunks)} chunks", flush=True)
        for retriever in retrievers:
            part = _part_path(retriever, chunking, len(items))
            per_k = _load_part(part, items, top_ks) if args.resume else None
            if per_k is not None:
                print(f"  {retriever}/{size}-{overlap}: resumed from {part.name}", flush=True)
            else:
                built = time.perf_counter()
                system = build_system(retriever, chunks, max_k, chunking, docs)
                print(
                    f"  {retriever}/{size}-{overlap}: index built in "
                    f"{time.perf_counter() - built:.0f}s",
                    flush=True,
                )
                per_k = columns_for_system(system, items, top_ks)
                _save_part(part, items, per_k)
            for k in top_ks:
                sid = system_id(retriever, chunking, k)
                columns[sid] = per_k[k]
                system_ids.append(sid)

    if args.verify_bm25:
        n_compared, n_mismatch, report = verify_bm25(items, chunkings, top_ks, columns, args.frozen)
        args.twin_out.parent.mkdir(parents=True, exist_ok=True)
        args.twin_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for sid, counts in report["per_column"].items():  # type: ignore[union-attr]
            print(f"  {sid}: {counts['mismatch']}/{counts['compared']} mismatched")
        print(f"{n_compared} cells compared, {n_mismatch} mismatched -> {args.twin_out}")
        if n_mismatch:
            for line in report["examples"]:  # type: ignore[union-attr]
                print(f"  {line}", file=sys.stderr)
            print("BM25 TWIN FAILED: this generator does not reproduce the frozen panel", file=sys.stderr)
            return 1
        print("BM25 TWIN OK: the frozen panel's `correct` definition is reproduced cell for cell")
        return 0

    write_matrix(args.out, items, system_ids, columns)
    print(f"{len(items)} items x {len(system_ids)} systems -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
