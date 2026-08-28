from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from fireassay.controls._common import stable_seed
from fireassay.models import Question, RetrievedChunk, SystemOutput
from fireassay.mutation.operators import (
    CorruptQueryOperator,
    DropResultsOperator,
    ShuffleTopkOperator,
    SwapRankingOperator,
    TruncateTopkOperator,
    load_mutation_config,
    load_operators,
)


def _question(text: str = "how do i reset my password") -> Question:
    return Question(text=text, qtype="factual", difficulty="easy", provenance="synthetic")


class _FixedSystem:
    """Always returns the same, known five-chunk ranking, and records the
    question it was actually called with (so `corrupt_query` can be
    checked)."""

    def __init__(self) -> None:
        self.name = "fixed"
        self.generates_answers = False
        self.last_question: Question | None = None

    def answer(self, question: Question) -> SystemOutput:
        self.last_question = question
        retrieved = tuple(
            RetrievedChunk(doc_id=f"doc{i:02d}", chunk_id=f"doc{i:02d}#0000", score=float(5 - i), rank=i,
                            char_start=0, char_end=100)
            for i in range(1, 6)
        )
        return SystemOutput(answer=None, abstained=True, retrieved=retrieved, latency_ms={"total": 1.0})


def test_drop_results_deletes_from_the_top() -> None:
    inner = _FixedSystem()
    system = DropResultsOperator(n=2).wrap(inner, seed=1)
    out = system.answer(_question())
    assert [rc.doc_id for rc in out.retrieved] == ["doc03", "doc04", "doc05"]
    assert [rc.rank for rc in out.retrieved] == [1, 2, 3]


def test_drop_results_zero_is_a_no_op() -> None:
    inner = _FixedSystem()
    system = DropResultsOperator(n=0).wrap(inner, seed=1)
    out = system.answer(_question())
    assert [rc.doc_id for rc in out.retrieved] == ["doc01", "doc02", "doc03", "doc04", "doc05"]


def test_truncate_topk_keeps_only_first_k() -> None:
    inner = _FixedSystem()
    system = TruncateTopkOperator(k=2).wrap(inner, seed=1)
    out = system.answer(_question())
    assert [rc.doc_id for rc in out.retrieved] == ["doc01", "doc02"]


def test_shuffle_topk_reorders_but_keeps_same_members_and_is_deterministic() -> None:
    system_a = ShuffleTopkOperator().wrap(_FixedSystem(), seed=42)
    system_b = ShuffleTopkOperator().wrap(_FixedSystem(), seed=42)
    out_a = system_a.answer(_question())
    out_b = system_b.answer(_question())
    assert {rc.doc_id for rc in out_a.retrieved} == {"doc01", "doc02", "doc03", "doc04", "doc05"}
    assert [rc.doc_id for rc in out_a.retrieved] == [rc.doc_id for rc in out_b.retrieved]
    assert [rc.rank for rc in out_a.retrieved] == [1, 2, 3, 4, 5]


def test_stable_seed_matches_a_pinned_sha256_test_vector() -> None:
    """`stable_seed` must be a pure, cross-interpreter-stable function of
    its inputs -- it is explicitly NOT allowed to use Python's builtin
    `hash()`, which is randomised per process for `str` via
    `PYTHONHASHSEED` (on by default since Python 3.3). Pinned against the
    standard NIST test vector for SHA-256("abc") (single positional arg,
    so no `\\x1f` separator is inserted). The first assertion independently
    confirms the quoted vector is the real SHA-256("abc") digest, so a
    typo here would fail loudly rather than silently validate a wrong
    `stable_seed` implementation.
    """
    assert hashlib.sha256(b"abc").hexdigest() == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    expected = int.from_bytes(bytes.fromhex("ba7816bf8f01cfea"), "big")
    assert stable_seed("abc") == expected


def test_stable_seed_is_identical_across_interpreters_with_different_hash_seeds() -> None:
    """The regression this guards against: builtin `hash()` is randomised
    per-process for `str` via `PYTHONHASHSEED`, so two interpreters
    started with different hash seeds would silently disagree on a
    `hash()`-derived seed while each still looking internally consistent
    -- a determinism test that only compares two calls inside *one*
    process would not catch that (this is exactly why the original
    `random.Random((seed, question.id, ...))` bug reached this point: it
    was not even a hash-seed issue, but the same class of "only tested
    within one process" gap). This spawns two genuinely separate
    interpreters, forced to different `PYTHONHASHSEED` values, and checks
    they compute the identical `stable_seed`.
    """

    def _seed_in_subprocess(hash_seed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from fireassay.controls._common import stable_seed\n"
                "print(stable_seed(1, 'question-id-123', 'shuffle_topk'))",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    first = _seed_in_subprocess("0")
    second = _seed_in_subprocess("4242")
    assert first == second
    assert first == str(stable_seed(1, "question-id-123", "shuffle_topk"))


def test_shuffle_topk_is_a_genuine_permutation_under_a_different_seed() -> None:
    out = ShuffleTopkOperator().wrap(_FixedSystem(), seed=2).answer(_question())
    order = [rc.doc_id for rc in out.retrieved]
    assert sorted(order) == ["doc01", "doc02", "doc03", "doc04", "doc05"]


def test_swap_ranking_reverses_the_list() -> None:
    inner = _FixedSystem()
    system = SwapRankingOperator().wrap(inner, seed=1)
    out = system.answer(_question())
    assert [rc.doc_id for rc in out.retrieved] == ["doc05", "doc04", "doc03", "doc02", "doc01"]
    assert [rc.rank for rc in out.retrieved] == [1, 2, 3, 4, 5]


def test_corrupt_query_drops_tokens_but_preserves_question_id() -> None:
    inner = _FixedSystem()
    question = _question("how do i reset my forgotten password today")
    system = CorruptQueryOperator(pct=0.5).wrap(inner, seed=7)
    system.answer(question)
    assert inner.last_question is not None
    assert inner.last_question.id == question.id  # id preserved (pure hash of ORIGINAL text)
    assert inner.last_question.text != question.text
    assert len(inner.last_question.text.split()) < len(question.text.split())


def test_corrupt_query_zero_pct_is_a_no_op() -> None:
    inner = _FixedSystem()
    question = _question()
    system = CorruptQueryOperator(pct=0.0).wrap(inner, seed=7)
    system.answer(question)
    assert inner.last_question is not None
    assert inner.last_question.text == question.text


def test_corrupt_query_is_deterministic_for_a_fixed_seed() -> None:
    question = _question("how do i reset my forgotten password today please")
    inner_a = _FixedSystem()
    CorruptQueryOperator(pct=0.5).wrap(inner_a, seed=99).answer(question)
    inner_b = _FixedSystem()
    CorruptQueryOperator(pct=0.5).wrap(inner_b, seed=99).answer(question)
    assert inner_a.last_question is not None and inner_b.last_question is not None
    assert inner_a.last_question.text == inner_b.last_question.text


def test_truncate_topk_equivalence_when_k_exceeds_every_retrieved_length() -> None:
    equivalent, reason = TruncateTopkOperator(k=10).check_equivalence([5, 3, 5])
    assert equivalent is True
    assert reason

    not_equivalent, _ = TruncateTopkOperator(k=2).check_equivalence([5, 3, 5])
    assert not_equivalent is False


def test_drop_results_equivalence_when_n_is_zero() -> None:
    equivalent, reason = DropResultsOperator(n=0).check_equivalence([5, 5])
    assert equivalent is True
    assert reason
    not_equivalent, _ = DropResultsOperator(n=1).check_equivalence([5, 5])
    assert not_equivalent is False


def test_shuffle_and_swap_equivalence_when_never_more_than_one_retrieved() -> None:
    for operator in (ShuffleTopkOperator(), SwapRankingOperator()):
        equivalent, reason = operator.check_equivalence([1, 0, 1])
        assert equivalent is True
        assert reason
        not_equivalent, _ = operator.check_equivalence([1, 2, 1])
        assert not_equivalent is False


def test_corrupt_query_equivalence_when_pct_is_zero() -> None:
    equivalent, reason = CorruptQueryOperator(pct=0.0).check_equivalence([5])
    assert equivalent is True
    assert reason
    not_equivalent, _ = CorruptQueryOperator(pct=0.1).check_equivalence([5])
    assert not_equivalent is False


def test_load_operators_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "mutants.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "operators": [
                    {"kind": "drop_results", "n": 2},
                    {"kind": "truncate_topk", "k": 3},
                    {"kind": "shuffle_topk"},
                    {"kind": "corrupt_query", "pct": 0.5},
                    {"kind": "swap_ranking"},
                ]
            }
        )
    )
    operators = load_operators(path)
    assert [op.name for op in operators] == [
        "drop_results",
        "truncate_topk",
        "shuffle_topk",
        "corrupt_query",
        "swap_ranking",
    ]


def test_load_mutation_config_parses_the_detector_block(tmp_path: Path) -> None:
    """Regression test for the bug: `load_operators`/an earlier
    `load_mutation_config` parsed `operators:` and silently discarded
    `detector:` entirely, so a YAML asking for `metric: retrieval.mrr`
    had no effect at all -- `cli.py`'s `mutate` always built a
    `ThresholdDetector` from its own hardcoded default instead. This
    asserts the parsed `DetectorSpec` actually reflects the file."""
    path = tmp_path / "mutants.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "detector": {"metric": "retrieval.mrr", "max_drop": 0.05},
                "operators": [{"kind": "swap_ranking"}],
            }
        )
    )
    config = load_mutation_config(path)
    assert config.detector is not None
    assert config.detector.metric == "retrieval.mrr"
    assert config.detector.max_drop == 0.05
    assert config.detector.max_rise is None
    assert [op.name for op in config.operators] == ["swap_ranking"]


def test_load_mutation_config_detector_is_none_when_the_file_has_no_detector_block(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mutants.yaml"
    path.write_text(yaml.safe_dump({"operators": [{"kind": "swap_ranking"}]}))
    config = load_mutation_config(path)
    assert config.detector is None


def test_load_mutation_config_parses_max_rise(tmp_path: Path) -> None:
    path = tmp_path / "mutants.yaml"
    path.write_text(
        yaml.safe_dump({"detector": {"metric": "cost.usd", "max_rise": 0.1}, "operators": []})
    )
    config = load_mutation_config(path)
    assert config.detector is not None
    assert config.detector.max_drop is None
    assert config.detector.max_rise == 0.1


def test_load_mutation_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "mutants.yaml"
    path.write_text(yaml.safe_dump({"operators": [], "bogus_key": 1}))
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_mutation_config(path)


def test_load_mutation_config_rejects_unknown_detector_key(tmp_path: Path) -> None:
    path = tmp_path / "mutants.yaml"
    path.write_text(
        yaml.safe_dump({"detector": {"metric": "retrieval.mrr", "bogus_key": 1}, "operators": []})
    )
    with pytest.raises(ValueError, match="unknown key.*detector"):
        load_mutation_config(path)


def test_load_mutation_config_detector_requires_metric(tmp_path: Path) -> None:
    path = tmp_path / "mutants.yaml"
    path.write_text(yaml.safe_dump({"detector": {"max_drop": 0.05}, "operators": []}))
    with pytest.raises(ValueError, match="requires a 'metric'"):
        load_mutation_config(path)
