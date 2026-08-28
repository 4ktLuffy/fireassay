"""M3-SPEC.md §7: deterministic under a seed; honeypots and double-reviews
are indistinguishable in the emitted item."""

from __future__ import annotations

from fireassay.curate.queue import build_queue, seed_from_candidates
from fireassay.generate.models import CandidateFeatures, ResolvedCandidate


def _candidates(n: int) -> list[ResolvedCandidate]:
    return [
        ResolvedCandidate(
            id=f"c{i:04d}",
            batch_id="b1",
            text=f"question number {i} about something specific?",
            qtype="factual",
            difficulty="easy",
            target_qtype="factual",
            target_difficulty="easy",
            reference_answer="an answer",
            quote="a quote",
            source_doc_id=f"doc{i % 5}",
            char_start=0,
            char_end=8,
            features=CandidateFeatures(title_overlap=0.0, quote_overlap=0.0, question_len_tokens=6),
            model_digest="digest",
            prompt_hash="prompt-hash",
            created_at=f"2026-01-01T00:{i:02d}:00",
        )
        for i in range(n)
    ]


def _summary(items: list) -> list[tuple[str, int, bool, bool]]:
    return [(i.candidate_id, i.position, i.is_honeypot, i.is_double_review) for i in items]


def test_build_queue_deterministic_given_seed() -> None:
    candidates = _candidates(50)
    first = build_queue(candidates, curator_id="alice", seed=123)
    second = build_queue(candidates, curator_id="alice", seed=123)
    assert _summary(first) == _summary(second)


def test_build_queue_different_seed_produces_different_order() -> None:
    candidates = _candidates(50)
    a = build_queue(candidates, curator_id="alice", seed=1)
    b = build_queue(candidates, curator_id="alice", seed=2)
    assert [i.candidate_id for i in a] != [i.candidate_id for i in b]


def test_honeypot_selection_is_curator_independent() -> None:
    """The same candidates flagged as honeypots for every curator sharing
    a seed -- required so honeypot/double-review analysis compares the
    same items across curators."""
    candidates = _candidates(200)
    alice = build_queue(candidates, curator_id="alice", seed=123)
    bob = build_queue(candidates, curator_id="bob", seed=123)
    alice_honeypots = {i.candidate_id for i in alice if i.is_honeypot}
    bob_honeypots = {i.candidate_id for i in bob if i.is_honeypot}
    assert alice_honeypots == bob_honeypots
    assert alice_honeypots  # nonempty at 5% of 200 candidates


def test_position_order_differs_across_curators() -> None:
    candidates = _candidates(50)
    alice = build_queue(candidates, curator_id="alice", seed=123)
    bob = build_queue(candidates, curator_id="bob", seed=123)
    assert [i.candidate_id for i in alice] != [i.candidate_id for i in bob]


def test_honeypot_and_double_review_rates_are_respected() -> None:
    candidates = _candidates(200)
    items = build_queue(candidates, curator_id="alice", honeypot_rate=0.05, double_review_rate=0.10, seed=7)
    assert sum(1 for i in items if i.is_honeypot) == 10
    assert sum(1 for i in items if i.is_double_review) == 20


def test_honeypot_and_double_review_sets_never_overlap() -> None:
    candidates = _candidates(200)
    items = build_queue(candidates, curator_id="alice", honeypot_rate=0.05, double_review_rate=0.10, seed=7)
    honeypot_ids = {i.candidate_id for i in items if i.is_honeypot}
    double_review_ids = {i.candidate_id for i in items if i.is_double_review}
    assert not (honeypot_ids & double_review_ids)


def test_every_item_carries_the_same_fields_regardless_of_flags() -> None:
    """Honeypots and double-reviews are structurally indistinguishable
    from a real item at the `QueueItem` level -- there is no extra/missing
    field, no special candidate_id namespace, nothing that would let a
    consumer single them out without reading the boolean flags themselves
    (which `curate.serve.next_item` deliberately never exposes to the
    curator)."""
    candidates = _candidates(100)
    items = build_queue(candidates, curator_id="alice", seed=42)
    for item in items:
        assert item.candidate_id.startswith("c")
        assert 1 <= item.position <= len(candidates)
        if item.is_honeypot:
            assert item.honeypot_expected_reason in ("WRONG_REFERENCE", "INSUFFICIENT_EVIDENCE")
        else:
            assert item.honeypot_expected_reason is None


def test_honeypot_reason_is_deterministic_given_candidate_and_seed() -> None:
    candidates = _candidates(100)
    first = build_queue(candidates, curator_id="alice", seed=99)
    second = build_queue(candidates, curator_id="bob", seed=99)
    first_reasons = {i.candidate_id: i.honeypot_expected_reason for i in first if i.is_honeypot}
    second_reasons = {i.candidate_id: i.honeypot_expected_reason for i in second if i.is_honeypot}
    assert first_reasons == second_reasons


def test_seed_from_candidates_is_deterministic_and_content_dependent() -> None:
    ids_a = ["a", "b", "c"]
    ids_b = ["a", "b", "d"]
    assert seed_from_candidates(ids_a) == seed_from_candidates(ids_a)
    assert seed_from_candidates(ids_a) != seed_from_candidates(ids_b)
    # Order-independent -- a queue seed should not depend on how the
    # caller happened to list the same candidate set.
    assert seed_from_candidates(ids_a) == seed_from_candidates(list(reversed(ids_a)))
