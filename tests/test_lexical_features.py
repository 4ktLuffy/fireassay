"""M3-SPEC.md §7: title/quote overlap computed via `text.tokenize` only."""

from __future__ import annotations

import pytest

from fireassay.generate.lexical import compute_lexical_features


def test_title_and_quote_overlap_hand_computed() -> None:
    features = compute_lexical_features(
        question_text="How do I reset my Password?",
        title="Password Reset",
        quote="click Forgot Password",
    )
    # tokens(question) = {how, do, i, reset, my, password} (6)
    # tokens(title) = {password, reset} (2); intersection = {reset, password} (2)
    assert features.title_overlap == pytest.approx(1.0)
    # tokens(quote) = {click, forgot, password} (3); intersection with
    # question tokens = {password} (1); |question tokens| = 6
    assert features.quote_overlap == pytest.approx(1 / 6)
    assert features.question_len_tokens == 6


def test_title_overlap_is_zero_not_undefined_for_empty_title() -> None:
    features = compute_lexical_features("What is this?", title="", quote="some quote")
    assert features.title_overlap == 0.0


def test_quote_overlap_is_zero_not_undefined_for_empty_question() -> None:
    features = compute_lexical_features("", title="Some Title", quote="a quote")
    assert features.quote_overlap == 0.0
    assert features.question_len_tokens == 0


def test_no_overlap_at_all_is_zero() -> None:
    features = compute_lexical_features(
        question_text="How long does shipping take internationally?",
        title="Refund Policy",
        quote="refunds are issued within fourteen days",
    )
    assert features.title_overlap == 0.0
    assert features.quote_overlap == 0.0
