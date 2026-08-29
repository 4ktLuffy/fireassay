from __future__ import annotations

import re

from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Doc
from fireassay.system.coverage import CoverageSystem
from fireassay.system.panel import CHUNKINGS, TOP_KS, build_chunks_by_config, build_panel
from fireassay.system.tfidf import TfidfSystem

_DOCS = [
    Doc(doc_id="doc0", title="", text="cats like fish and dogs like bones and birds fly high"),
    Doc(doc_id="doc1", title="", text="a second document with some different words about weather"),
]

_SYSTEM_ID_RE = re.compile(r"^(bm25|coverage|tfidf)/(\d+)-(\d+)/k(\d+)$")

_EXPECTED_RANKER_CLASSES: dict[str, type] = {
    "bm25": BM25System,
    "coverage": CoverageSystem,
    "tfidf": TfidfSystem,
}


def test_build_panel_returns_54_distinct_entries_with_parseable_ids() -> None:
    chunks_by_config = build_chunks_by_config(_DOCS)
    panel = build_panel(chunks_by_config)
    assert len(panel) == 54

    ids = [system_id for system_id, _ in panel]
    assert len(set(ids)) == 54  # every id unique

    for system_id, system in panel:
        m = _SYSTEM_ID_RE.match(system_id)
        assert m is not None, f"system_id {system_id!r} does not parse"
        ranker, size, overlap, top_k = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        assert (size, overlap) in CHUNKINGS
        assert top_k in TOP_KS
        assert system.name == ranker
        assert isinstance(system, _EXPECTED_RANKER_CLASSES[ranker])


def test_old_18_bm25_configs_are_present_and_exact() -> None:
    """The old 18-config panel was the cross product of the same 6
    chunkings and 3 top_ks, using only `BM25System`. `build_panel`'s
    `bm25/*` entries must correspond exactly to those 18 configs -- the
    same set of (chunk_size, chunk_overlap, top_k) triples, no more, no
    fewer -- so the old 18 remain identifiable within the new 54-config
    panel."""
    chunks_by_config = build_chunks_by_config(_DOCS)
    panel = build_panel(chunks_by_config)

    bm25_configs = set()
    for system_id, _ in panel:
        if not system_id.startswith("bm25/"):
            continue
        m = _SYSTEM_ID_RE.match(system_id)
        assert m is not None
        bm25_configs.add((int(m.group(2)), int(m.group(3)), int(m.group(4))))

    expected = {(size, overlap, top_k) for size, overlap in CHUNKINGS for top_k in TOP_KS}
    assert bm25_configs == expected
    assert len(bm25_configs) == 18


def test_build_chunks_by_config_keys_match_chunkings() -> None:
    chunks_by_config = build_chunks_by_config(_DOCS)
    assert set(chunks_by_config.keys()) == set(CHUNKINGS)


def test_build_panel_respects_custom_top_ks() -> None:
    chunks_by_config = build_chunks_by_config(_DOCS, chunkings=((256, 32),))
    panel = build_panel(chunks_by_config, top_ks=(1, 2))
    assert len(panel) == 3 * 1 * 2  # 3 rankers x 1 chunking x 2 top_ks
    for system_id, _ in panel:
        assert system_id.endswith("/k1") or system_id.endswith("/k2")
