"""`test_items_baseline.py`: `items.baseline.answer_unsupported`, the
no-model token-overlap detector `tools/evidence.py`'s `judge.token_overlap`
claim reads -- see that module's docstring for why this function exists
at all (defect 46: the number it reproduces had no committed instrument
until now).
"""

from __future__ import annotations

from fireassay.items.baseline import answer_unsupported
from fireassay.items.core import ItemMeta


def _meta(reference_answer: str | None, evidence_quote: str | None) -> ItemMeta:
    return ItemMeta(item_id="q1", reference_answer=reference_answer, evidence_quote=evidence_quote)


def test_hand_worked_example_supported() -> None:
    # answer tokens: {the, fee, is, twenty, one, pounds} -- 6 distinct.
    # evidence covers 4 of them (the, fee, twenty, one): overlap 4/6 =
    # 0.667, well above the default 0.34 threshold -- NOT flagged.
    meta = _meta(
        reference_answer="The fee is twenty one pounds",
        evidence_quote="The application fee must be paid within twenty one days",
    )
    assert answer_unsupported(meta) is False


def test_hand_worked_example_unsupported() -> None:
    # same answer, evidence sharing only "the": overlap 1/6 = 0.167,
    # below the default 0.34 threshold -- flagged.
    meta = _meta(
        reference_answer="The fee is twenty one pounds",
        evidence_quote="Contact the office for more information",
    )
    assert answer_unsupported(meta) is True


def test_boundary_is_strictly_less_than_not_less_than_or_equal() -> None:
    # 5 distinct answer tokens, 2 present in evidence: 2/5 = 0.4, exactly
    # at threshold=0.4 -- the rule is "< threshold", not "<=", so this is
    # NOT flagged.
    at_boundary = _meta(reference_answer="alpha beta gamma delta epsilon", evidence_quote="alpha beta")
    assert answer_unsupported(at_boundary, threshold=0.4) is False

    # one fewer token present: 1/5 = 0.2 < 0.4 -- flagged.
    below_boundary = _meta(reference_answer="alpha beta gamma delta epsilon", evidence_quote="alpha")
    assert answer_unsupported(below_boundary, threshold=0.4) is True


def test_fewer_than_three_distinct_answer_tokens_never_flagged() -> None:
    # a one-word reference answer has zero overlap with its evidence, but
    # there are too few tokens for an overlap fraction to mean anything --
    # never flagged, regardless of the evidence.
    meta = _meta(reference_answer="Yes", evidence_quote="completely unrelated text")
    assert answer_unsupported(meta) is False


def test_missing_reference_answer_or_evidence_never_raises_or_fabricates() -> None:
    # missing answer: 0 distinct tokens, below the floor -- False, not an
    # exception.
    assert answer_unsupported(_meta(None, "some evidence")) is False
    # missing evidence against a real answer: 0 overlap -- True, not a
    # fabricated False from absent text.
    assert answer_unsupported(_meta("The fee is twenty one pounds", None)) is True
