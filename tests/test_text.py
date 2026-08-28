from __future__ import annotations

from fireassay.system import bm25 as bm25_module
from fireassay.text import tokenize


def test_tokenize_lowercases() -> None:
    assert tokenize("HELLO World") == ["hello", "world"]


def test_tokenize_splits_on_non_word_characters_and_drops_empties() -> None:
    assert tokenize("hello, world!! foo-bar") == ["hello", "world", "foo", "bar"]


def test_tokenize_empty_string_yields_no_tokens() -> None:
    assert tokenize("") == []
    assert tokenize("   ...   ") == []


def test_bm25_module_imports_the_shared_tokenizer_not_a_local_definition() -> None:
    """Guards against a future regression where someone adds a second,
    divergent tokenizing regex directly in bm25.py instead of importing
    fireassay.text.tokenize."""
    assert bm25_module.tokenize is tokenize
