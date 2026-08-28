"""Config matrix expansion.

A `MatrixSpec` describes a grid of system configurations to run and score
against one suite (e.g. "top_k in {5, 10}" x "chunk_size in {200, 400}").
`expand` turns that grid into the concrete list of config dicts that
`runner.run_matrix` will actually execute, one per cell.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from fireassay.hashing import config_hash


class MatrixSpec(BaseModel):
    """A config matrix.

    `axes` maps an axis name to its list of possible values. A value may be
    a plain string (e.g. a model name) or a dict of several related
    parameters bundled together as one option (e.g.
    `{"size": 200, "overlap": 20}` for a "chunking" axis) — the two are
    interchangeable from `expand`'s point of view: whatever the option is,
    it becomes the value stored under that axis's key in the resulting
    config dict.
    """

    model_config = ConfigDict(frozen=True)

    suite: str  # "name@version"
    axes: dict[str, list[dict[str, object]] | list[str]]
    include: list[dict[str, object]] = Field(default_factory=list)
    exclude: list[dict[str, object]] = Field(default_factory=list)


def _matches(pattern: Mapping[str, object], config: Mapping[str, object]) -> bool:
    """A pattern matches a config when every key the pattern names equals
    the config's value for that key. Keys the pattern does not mention are
    unconstrained, so a one-key pattern can exclude an entire axis value
    across every combination of the other axes."""
    return all(config.get(key) == value for key, value in pattern.items())


def expand(spec: MatrixSpec) -> list[dict[str, object]]:
    """Expand a `MatrixSpec` into the concrete list of config dicts to run.

    Algorithm:

    1. Cartesian product over `spec.axes` (axes visited in sorted-name
       order, purely so the intermediate product is itself deterministic
       before the final sort).
    2. Drop any config matching at least one `spec.exclude` pattern.
    3. Append `spec.include` entries verbatim — these are full config
       dicts added outside the grid (e.g. a baseline config that does not
       fit the axes), so they are not subject to `exclude` filtering.
    4. De-duplicate by `config_hash` (an `include` entry that happens to
       coincide with a grid cell is not run twice), keeping first
       occurrence.
    5. Sort by `config_hash`.

    The final sort-by-hash (rather than, say, grid order) is deliberate:
    grid order depends on axis insertion order in the YAML file, which is
    incidental. Sorting by hash means `expand` returns the same list, in
    the same order, for the same logical set of configs regardless of how
    the spec happened to be written — which matters because `runner.py`
    reports progress and results in this order.
    """
    axis_names = sorted(spec.axes)
    product: list[dict[str, object]] = [{}]
    for name in axis_names:
        options = spec.axes[name]
        product = [dict(base, **{name: option}) for base in product for option in options]

    filtered = [cfg for cfg in product if not any(_matches(pattern, cfg) for pattern in spec.exclude)]

    combined: list[dict[str, object]] = [*filtered, *spec.include]

    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for cfg in combined:
        h = config_hash(cfg)
        if h in seen:
            continue
        seen.add(h)
        unique.append(cfg)

    unique.sort(key=config_hash)
    return unique
