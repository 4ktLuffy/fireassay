"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from fireassay.store.db import Store

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "test.db")
    s.migrate()
    yield s
    s.close()
