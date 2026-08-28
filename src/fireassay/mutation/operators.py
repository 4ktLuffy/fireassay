"""System-mutation operators (M2-SPEC.md §6): each wraps a `System` and
degrades its retrieval deterministically, seeded from `suite_hash` (per
mutation run) plus the operator's own parameters.

`MutationOperator` is a `Protocol` (not a base class) specifically so a
later milestone can add generation-side operators (weakened model,
truncated context, planted prompt regression — M5) without touching this
module or anything importing it.

Every operator also computes its own **equivalence** check
(`check_equivalence`): whether, given the *baseline* run's actual
per-question retrieved-list lengths, this operator's parameters
provably could not have degraded anything (M2-SPEC.md §6 — "equivalent
mutants must be computed and justified, never assumed"). This is why
`check_equivalence` takes the baseline's real retrieved lengths rather
than asserting equivalence from parameters alone: a `truncate_topk(k=3)`
mutant is only equivalent if the baseline genuinely never retrieved more
than 3 chunks for *any* question — a fact about this suite/config, not
something the operator can know a priori.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import yaml

from fireassay.controls._common import stable_seed
from fireassay.models import Question, SystemOutput
from fireassay.system.base import System


class MutationOperator(Protocol):
    name: str
    params: Mapping[str, object]

    def wrap(self, system: System, seed: int) -> System:
        """Return a new `System` that degrades `system`'s retrieval
        deterministically. `seed` is combined with `self.params` by each
        operator's own wrapper (see e.g. `_CorruptQuerySystem`) so that two
        operators with different parameters never coincidentally produce
        the same degradation from the same seed."""
        ...

    def check_equivalence(self, baseline_retrieved_lengths: Sequence[int]) -> tuple[bool, str]:
        """Whether this operator, with its current parameters, provably
        cannot have degraded quality against a run whose per-question
        retrieved-list lengths were `baseline_retrieved_lengths`. Returns
        `(is_equivalent, reason)` — `reason` is always populated (M2's
        mutation score prints every exclusion with its reason;
        `mutation.score.MutantResult.equivalent_reason` is `None` only
        when `is_equivalent` is `False`)."""
        ...


class DropResultsOperator:
    """Delete the top `n` chunks from the ranking, shifting the rest up.
    Equivalent when `n <= 0` (drops nothing)."""

    name = "drop_results"

    def __init__(self, n: int) -> None:
        self.n = n
        self.params: Mapping[str, object] = {"n": n}

    def wrap(self, system: System, seed: int) -> System:
        return _DropResultsSystem(system, self.n)

    def check_equivalence(self, baseline_retrieved_lengths: Sequence[int]) -> tuple[bool, str]:
        if self.n <= 0:
            return True, f"drop_results(n={self.n}): n <= 0 drops nothing from any ranking"
        return False, ""


class _DropResultsSystem:
    def __init__(self, inner: System, n: int) -> None:
        self._inner = inner
        self._n = n
        self.name = f"drop_results(n={n})({inner.name})"
        self.generates_answers = inner.generates_answers

    def answer(self, question: Question) -> SystemOutput:
        out = self._inner.answer(question)
        by_rank = sorted(out.retrieved, key=lambda rc: rc.rank)
        kept = by_rank[self._n :]
        reranked = tuple(rc.model_copy(update={"rank": i}) for i, rc in enumerate(kept, start=1))
        return out.model_copy(update={"retrieved": reranked})


class TruncateTopkOperator:
    """Keep only the first `k` results. Equivalent when the baseline never
    retrieved more than `k` chunks for any question — truncation never
    actually binds."""

    name = "truncate_topk"

    def __init__(self, k: int) -> None:
        self.k = k
        self.params: Mapping[str, object] = {"k": k}

    def wrap(self, system: System, seed: int) -> System:
        return _TruncateTopkSystem(system, self.k)

    def check_equivalence(self, baseline_retrieved_lengths: Sequence[int]) -> tuple[bool, str]:
        if baseline_retrieved_lengths and max(baseline_retrieved_lengths) <= self.k:
            return True, (
                f"truncate_topk(k={self.k}): baseline never retrieved more than "
                f"{max(baseline_retrieved_lengths)} chunks for any question, so truncation never binds"
            )
        return False, ""


class _TruncateTopkSystem:
    def __init__(self, inner: System, k: int) -> None:
        self._inner = inner
        self._k = k
        self.name = f"truncate_topk(k={k})({inner.name})"
        self.generates_answers = inner.generates_answers

    def answer(self, question: Question) -> SystemOutput:
        out = self._inner.answer(question)
        by_rank = sorted(out.retrieved, key=lambda rc: rc.rank)[: self._k]
        return out.model_copy(update={"retrieved": tuple(by_rank)})


class ShuffleTopkOperator:
    """Randomise the order of the retrieved list, deterministically per
    question (seed + question id). Equivalent when the baseline never
    retrieved more than 1 chunk for any question — shuffling 0 or 1 items
    is a no-op."""

    name = "shuffle_topk"

    def __init__(self) -> None:
        self.params: Mapping[str, object] = {}

    def wrap(self, system: System, seed: int) -> System:
        return _ShuffleTopkSystem(system, seed)

    def check_equivalence(self, baseline_retrieved_lengths: Sequence[int]) -> tuple[bool, str]:
        if not baseline_retrieved_lengths or max(baseline_retrieved_lengths) <= 1:
            return True, "shuffle_topk: baseline never retrieved more than 1 chunk for any question"
        return False, ""


class _ShuffleTopkSystem:
    def __init__(self, inner: System, seed: int) -> None:
        self._inner = inner
        self._seed = seed
        self.name = f"shuffle_topk({inner.name})"
        self.generates_answers = inner.generates_answers

    def answer(self, question: Question) -> SystemOutput:
        out = self._inner.answer(question)
        by_rank = list(sorted(out.retrieved, key=lambda rc: rc.rank))
        rng = random.Random(stable_seed(self._seed, question.id, "shuffle_topk"))
        rng.shuffle(by_rank)
        reranked = tuple(rc.model_copy(update={"rank": i}) for i, rc in enumerate(by_rank, start=1))
        return out.model_copy(update={"retrieved": reranked})


class CorruptQueryOperator:
    """Drop `pct` of the query's whitespace-delimited tokens before
    retrieval, deterministically per question. Equivalent when `pct <= 0`
    (drops nothing)."""

    name = "corrupt_query"

    def __init__(self, pct: float) -> None:
        self.pct = pct
        self.params: Mapping[str, object] = {"pct": pct}

    def wrap(self, system: System, seed: int) -> System:
        return _CorruptQuerySystem(system, self.pct, seed)

    def check_equivalence(self, baseline_retrieved_lengths: Sequence[int]) -> tuple[bool, str]:
        if self.pct <= 0.0:
            return True, f"corrupt_query(pct={self.pct}): pct <= 0 drops no query tokens"
        return False, ""


class _CorruptQuerySystem:
    def __init__(self, inner: System, pct: float, seed: int) -> None:
        self._inner = inner
        self._pct = pct
        self._seed = seed
        self.name = f"corrupt_query(pct={pct})({inner.name})"
        self.generates_answers = inner.generates_answers

    def answer(self, question: Question) -> SystemOutput:
        words = question.text.split()
        if words and self._pct > 0.0:
            rng = random.Random(stable_seed(self._seed, question.id, "corrupt_query"))
            n_drop = min(len(words), int(round(len(words) * self._pct)))
            drop_indices = set(rng.sample(range(len(words)), n_drop)) if n_drop > 0 else set()
            corrupted_text = " ".join(w for i, w in enumerate(words) if i not in drop_indices)
        else:
            corrupted_text = question.text
        # model_copy: preserves `id` (a pure hash of the *original* text),
        # so results still key against the suite's real question id, while
        # the system actually retrieves against the corrupted text.
        corrupted_question = question.model_copy(update={"text": corrupted_text})
        return self._inner.answer(corrupted_question)


class SwapRankingOperator:
    """Reverse the retrieved list. Equivalent when the baseline never
    retrieved more than 1 chunk for any question — reversing 0 or 1 items
    is a no-op."""

    name = "swap_ranking"

    def __init__(self) -> None:
        self.params: Mapping[str, object] = {}

    def wrap(self, system: System, seed: int) -> System:
        return _SwapRankingSystem(system)

    def check_equivalence(self, baseline_retrieved_lengths: Sequence[int]) -> tuple[bool, str]:
        if not baseline_retrieved_lengths or max(baseline_retrieved_lengths) <= 1:
            return True, "swap_ranking: baseline never retrieved more than 1 chunk for any question"
        return False, ""


class _SwapRankingSystem:
    def __init__(self, inner: System) -> None:
        self._inner = inner
        self.name = f"swap_ranking({inner.name})"
        self.generates_answers = inner.generates_answers

    def answer(self, question: Question) -> SystemOutput:
        out = self._inner.answer(question)
        by_rank = list(sorted(out.retrieved, key=lambda rc: rc.rank))
        reversed_chunks = list(reversed(by_rank))
        reranked = tuple(rc.model_copy(update={"rank": i}) for i, rc in enumerate(reversed_chunks, start=1))
        return out.model_copy(update={"retrieved": reranked})


ALL_OPERATOR_KINDS = frozenset(
    {"drop_results", "truncate_topk", "shuffle_topk", "corrupt_query", "swap_ranking"}
)


def load_operators(path: Path | str) -> list[MutationOperator]:
    """Load a list of `MutationOperator` instances from a YAML file shaped:

        operators:
          - kind: drop_results
            n: 2
          - kind: truncate_topk
            k: 3
          - kind: shuffle_topk
          - kind: corrupt_query
            pct: 0.5
          - kind: swap_ranking

    Raises `ValueError` for an unknown `kind`.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    operators: list[MutationOperator] = []
    for spec in raw.get("operators", []):
        kind = spec["kind"]
        if kind == "drop_results":
            operators.append(DropResultsOperator(n=int(spec["n"])))
        elif kind == "truncate_topk":
            operators.append(TruncateTopkOperator(k=int(spec["k"])))
        elif kind == "shuffle_topk":
            operators.append(ShuffleTopkOperator())
        elif kind == "corrupt_query":
            operators.append(CorruptQueryOperator(pct=float(spec["pct"])))
        elif kind == "swap_ranking":
            operators.append(SwapRankingOperator())
        else:
            raise ValueError(
                f"mutants.yaml: unknown operator kind {kind!r}; must be one of {sorted(ALL_OPERATOR_KINDS)}"
            )
    return operators
