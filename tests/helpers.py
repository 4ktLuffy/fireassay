"""Shared M2 test helpers: fixture loading and a reusable `ControlContext`
builder, so every `test_controls_*.py` file doesn't reimplement the same
BM25-over-fixtures wiring `test_runner_e2e.py` already established for M1.

Not itself a `test_*.py` module — pytest does not collect it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np

from fireassay.controls.base import ControlContext, build_control_context
from fireassay.controls.expected import Band, load_expected_bands
from fireassay.models import EvidenceSpan, Question, Suite
from fireassay.score.abstention import AbstentionScorer
from fireassay.score.base import Scorer, ScoringContext
from fireassay.score.cost import CostScorer
from fireassay.score.latency import LatencyScorer
from fireassay.score.policy import PolicyScorer
from fireassay.score.retrieval import RetrievalScorer
from fireassay.store.db import Store
from fireassay.system.base import System
from fireassay.system.bm25 import BM25System
from fireassay.system.corpus import Doc, chunk_corpus, load_corpus
from fireassay.system.embedding import l2_normalise
from fireassay.text import tokenize

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_PATH = Path(__file__).parent.parent / "src" / "fireassay" / "controls" / "expected.yaml"
_TOOLS_DIR = Path(__file__).parent.parent / "tools"

#: Cache for `load_tool_module` -- one module object per tool, loaded
#: once per test session and reused, rather than a fresh `exec_module`
#: (and a fresh `sys.modules` registration under the same name) per
#: caller. See `load_tool_module`'s docstring for why registration itself
#: is required at all.
_tool_modules: dict[str, ModuleType] = {}


def load_tool_module(name: str) -> ModuleType:
    """Load `tools/<name>.py` by file path and return the module.

    `tools/` is a scripts directory, not an importable package (no
    `tools/__init__.py`, deliberately) -- so a test that wants to exercise
    a tool's *actual* code, rather than a copy of its logic, has to load
    it this way. One loader, here, rather than one per test file: two
    different module names for the same file is exactly what makes
    `mypy --strict` see it as reachable under two identities and fail
    asking for `--explicit-package-bases`.

    **Registers the module in `sys.modules` under its spec name BEFORE
    `exec_module` runs.** This is the actual fix, not a formality --
    without it, a `@dataclass` in the loaded module fails to import at
    all: `dataclasses` resolves a class's own module via
    `sys.modules.get(cls.__module__)` to read its (possibly
    string/forward-referenced) annotations, and with the module absent
    from `sys.modules` that lookup returns `None`, then `AttributeError:
    'NoneType' object has no attribute '__dict__'`. Do not "simplify"
    this back out -- it was silently fine only because nothing loaded
    this way had used `@dataclass` yet.
    """
    cached = _tool_modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, _TOOLS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # must precede exec_module -- see docstring above
    spec.loader.exec_module(module)
    _tool_modules[name] = module
    return module


def load_measure_detector_recall() -> ModuleType:
    """`tools/` is a scripts directory, not an importable package (no
    `tools/__init__.py`, deliberately -- see `tools/measure_detector_
    recall.py`'s own docstring) -- load the real module by file path so
    tests exercise the actual `tools/measure_detector_recall.py` code
    path (`instrument_inputs`/`reached_instrument`/`count_reached_
    instrument`/`verdict_for_kind`/`lexical_decoy_report`/...), never a
    copy of its logic. A thin, named wrapper over `load_tool_module` (see
    that function for the loading rules), kept because the name is what
    every existing caller asks for."""
    return load_tool_module("measure_detector_recall")


def load_fixture_docs() -> list[Doc]:
    return load_corpus(FIXTURES_DIR / "corpus.jsonl")


def load_fixture_questions() -> list[Question]:
    questions: list[Question] = []
    with open(FIXTURES_DIR / "questions.jsonl", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            spans = tuple(EvidenceSpan(**span) for span in obj.get("evidence_spans", []))
            questions.append(
                Question(
                    text=obj["text"],
                    qtype=obj["qtype"],
                    difficulty=obj["difficulty"],
                    reference_answer=obj.get("reference_answer"),
                    evidence_spans=spans,
                    provenance=obj["provenance"],
                    generator=obj.get("generator"),
                    source_doc_id=obj.get("source_doc_id"),
                )
            )
    return questions


#: Width of `toy_embed`'s vector space. Small enough that a test's
#: matrices print, wide enough that the fixture corpus's twelve documents
#: land in distinguishable directions.
TOY_DIM = 64


def toy_embed(texts: Sequence[str]) -> np.ndarray:
    """A deterministic, offline stand-in for a real embedder: hashed
    bag-of-words, L2-normalised through the production `l2_normalise`.

    Deliberately **not** a mock returning canned vectors. A dense
    retriever's tests have to distinguish "retrieval works" from
    "retrieval is broken", and a canned matrix cannot do that — it has no
    relationship to the text. This one does real lexical matching, so
    `test_shuffled_vectors_collapse_recall` has a genuine signal to
    destroy. It never touches the network (M3-SPEC.md §7).
    """
    rows = np.zeros((len(texts), TOY_DIM), dtype=np.float64)
    for i, text in enumerate(texts):
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            rows[i, int.from_bytes(digest, "big") % TOY_DIM] += 1.0
        # Never hand a zero row to `l2_normalise` (it rejects them, as it
        # should): an empty/stopword-only text still needs a direction.
        rows[i, 0] += 1e-3
    return l2_normalise(rows)


def bm25_system_factory(config: Mapping[str, object], docs: Sequence[Doc]) -> System:
    chunks = list(chunk_corpus(docs, size=800, overlap=100))
    top_k = int(config.get("top_k", 5))  # type: ignore[arg-type]
    return BM25System(chunks, top_k=top_k)


def scoring_ctx_factory(config: Mapping[str, object]) -> ScoringContext:
    top_k = int(config.get("top_k", 5))  # type: ignore[arg-type]
    return ScoringContext(top_k=top_k)


def standard_scorers() -> list[Scorer]:
    return [RetrievalScorer(), LatencyScorer(), CostScorer(), PolicyScorer(), AbstentionScorer()]


def load_expected() -> dict[str, dict[str, Band]]:
    return load_expected_bands(EXPECTED_PATH)


def setup_suite(store: Store, name: str = "kb", version: str = "1.0.0") -> tuple[Suite, list[Question]]:
    questions = load_fixture_questions()
    store.put_questions(questions)
    suite = store.freeze_suite(name, version, [q.id for q in questions])
    return suite, questions


def make_control_context(
    store: Store,
    *,
    top_k: int = 5,
    system_factory: Callable[[Mapping[str, object], Sequence[Doc]], System] | None = None,
    scorers: list[Scorer] | None = None,
) -> ControlContext:
    """Build a `ControlContext` over the shared fixture corpus/questions,
    with the real BM25 system factory and all five M1 scorers unless
    overridden — override either to construct a deliberately broken
    harness for a "must fire on broken" test.
    """
    suite, questions = setup_suite(store)
    docs = load_fixture_docs()
    base_config = store.put_config({"top_k": top_k})
    expected = load_expected()
    factory = system_factory if system_factory is not None else bm25_system_factory
    return build_control_context(
        store,
        suite,
        base_config,
        questions,
        docs,
        factory,
        scoring_ctx_factory,
        scorers if scorers is not None else standard_scorers(),
        expected,
    )
