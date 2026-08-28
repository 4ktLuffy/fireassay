from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fireassay.controls.expected import Band, load_expected_bands
from helpers import EXPECTED_PATH


def test_packaged_expected_yaml_loads_cleanly() -> None:
    """`null_questions` has no entry (like `judge_calibration`): both
    always report NOT_RUN and so have nothing to check a band against —
    see controls/null_questions.py."""
    bands = load_expected_bands(EXPECTED_PATH)
    assert set(bands) == {
        "no_retrieval",
        "shuffled_gold",
        "corpus_ablation",
        "identical_config",
    }
    assert bands["no_retrieval"]["recall_at_k"].eq == 0.0
    # shuffled_gold's chance band is self-calibrated at runtime, not a
    # fixed threshold (controls/shuffled_gold.py) -- expected.yaml carries
    # the calibration's own parameters instead.
    assert bands["shuffled_gold"]["min_margin"].eq == 0.05
    assert bands["shuffled_gold"]["n_seeds"].eq == 50
    assert bands["shuffled_gold"]["m_primary"].eq == 5
    assert bands["shuffled_gold"]["k"].eq == 5


def test_unknown_control_kind_rejected(tmp_path: Path) -> None:
    path = tmp_path / "expected.yaml"
    path.write_text(yaml.safe_dump({"not_a_real_control": {"recall_at_k": {"eq": 0.0}}}))
    with pytest.raises(ValueError, match="unknown control kind"):
        load_expected_bands(path)


def test_empty_band_at_metric_level_rejected(tmp_path: Path) -> None:
    path = tmp_path / "expected.yaml"
    path.write_text(yaml.safe_dump({"no_retrieval": {"recall_at_k": {}}}))
    with pytest.raises(ValueError, match="no constraint keys"):
        load_expected_bands(path)


def test_empty_band_at_control_level_rejected(tmp_path: Path) -> None:
    path = tmp_path / "expected.yaml"
    path.write_text(yaml.safe_dump({"no_retrieval": {}}))
    with pytest.raises(ValueError, match="no metric bands"):
        load_expected_bands(path)


def test_band_model_rejects_empty_constraints_directly() -> None:
    with pytest.raises(ValueError):
        Band()


def test_widened_band_is_visible_in_a_diff(tmp_path: Path) -> None:
    """A band cannot be quietly widened to make a run pass: this is a
    textual, version-controlled file, so widening `max: 0.05` to
    `max: 0.50` is a one-line, reviewable diff -- proven here by loading
    both versions and asserting the `check()` behaviour genuinely changed,
    which is exactly what a reviewer would see in the diff."""
    original = tmp_path / "expected.yaml"
    original.write_text(yaml.safe_dump({"shuffled_gold": {"recall_at_k": {"max": 0.05}}}))
    widened = tmp_path / "expected_widened.yaml"
    widened.write_text(yaml.safe_dump({"shuffled_gold": {"recall_at_k": {"max": 0.50}}}))

    original_band = load_expected_bands(original)["shuffled_gold"]["recall_at_k"]
    widened_band = load_expected_bands(widened)["shuffled_gold"]["recall_at_k"]

    assert original_band.check(0.20) is False
    assert widened_band.check(0.20) is True
    assert original_band.max != widened_band.max
