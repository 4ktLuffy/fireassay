"""`test_items_no_store_import.py` (M-ITEMS-SPEC.md §6): the guard on the
whole `items/` design. `items.core` must not import `fireassay.store`, or
anything that transitively does.

Implemented as a **static** walk of `import`/`from ... import` statements
(via `ast`, stdlib only -- no new dependency), rather than importing
`fireassay.items.core` and inspecting `sys.modules`: an earlier test in
the same pytest session may have already imported `fireassay.store` for
unrelated reasons (e.g. the `store` fixture in `conftest.py`), which would
make a post-hoc `sys.modules` check report a false positive that has
nothing to do with what `items.core` itself imports. Parsing the source
files' own import statements has no such cross-test contamination risk.
"""

from __future__ import annotations

import ast
from pathlib import Path

import fireassay

_PACKAGE_ROOT = Path(fireassay.__file__).parent


def _module_file(module_name: str) -> Path | None:
    """The source file for a dotted `fireassay.*` module name, or `None`
    if it does not resolve to a file inside this package (e.g. a
    third-party module such as `numpy` or `pydantic`)."""
    if module_name != "fireassay" and not module_name.startswith("fireassay."):
        return None
    parts = module_name.split(".")[1:]
    if not parts:
        return _PACKAGE_ROOT / "__init__.py"
    candidate_dir = _PACKAGE_ROOT.joinpath(*parts)
    candidate_pkg = candidate_dir / "__init__.py"
    if candidate_pkg.exists():
        return candidate_pkg
    candidate_module = candidate_dir.with_suffix(".py")
    return candidate_module if candidate_module.exists() else None


def _direct_imports(module_name: str) -> set[str]:
    """Every `fireassay.*` module name imported directly (one hop) by
    `module_name`'s own source, resolving relative imports (`from . import
    x` / `from .. import y`) against `module_name`'s own package."""
    path = _module_file(module_name)
    if path is None:
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    module_parts = module_name.split(".")
    # A module file's own package is itself for a package __init__.py,
    # else its parent -- needed to resolve `from . import sibling` etc.
    package_parts = module_parts if path.name == "__init__.py" else module_parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "fireassay" or alias.name.startswith("fireassay."):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and (node.module == "fireassay" or node.module.startswith("fireassay.")):
                    imports.add(node.module)
            else:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                if node.module:
                    base = base + node.module.split(".")
                resolved = ".".join(base)
                if resolved == "fireassay" or resolved.startswith("fireassay."):
                    imports.add(resolved)
    return imports


def _reachable_modules(start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for imported in _direct_imports(current):
            if imported not in seen:
                stack.append(imported)
    return seen


def test_items_core_module_graph_excludes_store() -> None:
    reachable = _reachable_modules("fireassay.items.core")
    leaked = {m for m in reachable if m == "fireassay.store" or m.startswith("fireassay.store.")}
    assert not leaked, (
        f"fireassay.items.core's module graph reaches {leaked} -- items/core.py "
        "(or something it imports) must not import fireassay.store, see its module docstring"
    )


def test_sanity_the_walker_follows_a_known_cross_module_import() -> None:
    """`items.review` imports `items.core` and `items.seed` -- confirms the
    walker actually traverses real cross-module imports rather than only
    ever returning the start node (which would make the main test above
    pass vacuously for any input, including a broken walker)."""
    reachable = _reachable_modules("fireassay.items.review")
    assert "fireassay.items.core" in reachable
    assert "fireassay.items.seed" in reachable


def test_sanity_the_walker_detects_a_real_store_import() -> None:
    """`items.adapters.store` DOES import the store -- confirms the walker
    is capable of detecting the exact thing `items.core` must avoid,
    rather than this whole test file passing vacuously."""
    reachable = _reachable_modules("fireassay.items.adapters.store")
    leaked = {m for m in reachable if m == "fireassay.store" or m.startswith("fireassay.store.")}
    assert leaked


def test_items_calibration_module_graph_excludes_store() -> None:
    """`items.calibration` (the calibration-consumer module) is bound by
    the same store-free rule as `items.core` -- see its module
    docstring."""
    reachable = _reachable_modules("fireassay.items.calibration")
    leaked = {m for m in reachable if m == "fireassay.store" or m.startswith("fireassay.store.")}
    assert not leaked, (
        f"fireassay.items.calibration's module graph reaches {leaked} -- items/calibration.py "
        "(or something it imports) must not import fireassay.store, see its module docstring"
    )


def test_sanity_the_walker_follows_items_calibrations_own_cross_module_import() -> None:
    """`items.calibration` imports `items.core` (for `wilson_ci`) -- the
    same non-vacuous-pass check as this file's other sanity tests, applied
    to the new module: confirms the walker traverses `items.calibration`'s
    real cross-module import rather than only ever returning the start
    node."""
    reachable = _reachable_modules("fireassay.items.calibration")
    assert "fireassay.items.core" in reachable


def test_items_answerability_module_graph_excludes_store_and_llm() -> None:
    """`items.answerability` (gold answerability, handbook §6/§9) is bound
    by the same store-free rule as `items.core`/`items.calibration`, **and**
    additionally must not reach `fireassay.llm` at all -- the model arrives
    as an injected `Ask` callable, never an import (see its module
    docstring)."""
    reachable = _reachable_modules("fireassay.items.answerability")
    leaked_store = {m for m in reachable if m == "fireassay.store" or m.startswith("fireassay.store.")}
    leaked_llm = {m for m in reachable if m == "fireassay.llm" or m.startswith("fireassay.llm.")}
    assert not leaked_store, (
        f"fireassay.items.answerability's module graph reaches {leaked_store} -- "
        "answerability.py (or something it imports) must not import fireassay.store, see its "
        "module docstring"
    )
    assert not leaked_llm, (
        f"fireassay.items.answerability's module graph reaches {leaked_llm} -- "
        "answerability.py (or something it imports) must not import fireassay.llm, see its "
        "module docstring"
    )


def test_sanity_the_walker_follows_items_answerabilitys_own_cross_module_import() -> None:
    """`items.answerability` imports `items.core` (for `ItemMeta`) -- the
    same non-vacuous-pass check as this file's other sanity tests, applied
    to the new module: confirms the walker traverses `items.answerability`'s
    real cross-module import rather than only ever returning the start
    node."""
    reachable = _reachable_modules("fireassay.items.answerability")
    assert "fireassay.items.core" in reachable


def test_items_adapters_retrieval_module_graph_excludes_store() -> None:
    """`items.adapters.retrieval` (the `lexical_decoy` seed builder) is
    the second module -- alongside `items.adapters.store` -- allowed to
    reach outside `items/` at all, but it must not be the second module
    allowed to reach `fireassay.store`: that stays `adapters/store.py`'s
    alone (see `items.adapters.retrieval`'s module docstring)."""
    reachable = _reachable_modules("fireassay.items.adapters.retrieval")
    leaked = {m for m in reachable if m == "fireassay.store" or m.startswith("fireassay.store.")}
    assert not leaked, (
        f"fireassay.items.adapters.retrieval's module graph reaches {leaked} -- "
        "retrieval.py (or something it imports) must not import fireassay.store"
    )


def test_sanity_the_walker_follows_items_adapters_retrievals_own_cross_module_import() -> None:
    """`items.adapters.retrieval` imports `items.core`/`items.seed` (for
    `ItemMeta`/`SeededItem`/`seeded_item`) and `fireassay.system.bm25` --
    the same non-vacuous-pass check as this file's other sanity tests,
    applied to the new module."""
    reachable = _reachable_modules("fireassay.items.adapters.retrieval")
    assert "fireassay.items.core" in reachable
    assert "fireassay.items.seed" in reachable
    assert "fireassay.system.bm25" in reachable
