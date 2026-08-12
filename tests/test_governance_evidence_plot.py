"""Tests for the promotion-governance evidence figure (SF-S5-SL-MR5).

The figure makes a claim about the inference machinery's operating
characteristics, so the script that renders it is held to the same bar as the
machinery: it must be deterministic, and it must refuse to publish a figure
whose premise the study did not actually establish.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_governance_evidence.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot_governance_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evidence = _load_script()


def _point(**overrides: object) -> object:
    base: dict[str, object] = {
        "dependence": 0.0,
        "block_false_positive_rate": 0.05,
        "naive_false_positive_rate": 0.05,
        "block_median_width": 0.002,
        "naive_median_width": 0.002,
    }
    base.update(overrides)
    return evidence.CalibrationPoint(**base)  # type: ignore[arg-type]


def _sweep(block_rate: float, naive_rate: float) -> list[object]:
    """A minimal sweep spanning independence to full sharing."""
    return [
        _point(dependence=0.0),
        _point(
            dependence=1.0,
            block_false_positive_rate=block_rate,
            naive_false_positive_rate=naive_rate,
        ),
    ]


def test_the_study_refuses_to_publish_when_the_block_bootstrap_lost_its_level() -> None:
    """The figure asserts calibration; it must not render when that is false."""
    with pytest.raises(evidence.EvidenceError, match="did not hold its nominal level"):
        evidence._validate(_sweep(block_rate=0.40, naive_rate=0.45))


def test_the_study_refuses_when_the_central_contrast_is_absent() -> None:
    """If the naive estimator was not anticonservative, there is nothing to show."""
    with pytest.raises(evidence.EvidenceError, match="not anticonservative"):
        evidence._validate(_sweep(block_rate=0.06, naive_rate=0.05))


def test_an_empty_study_is_refused() -> None:
    with pytest.raises(evidence.EvidenceError, match="no points"):
        evidence._validate([])


def test_a_sweep_that_does_not_span_the_dependence_range_is_refused() -> None:
    """Panel 1's claim is about the range, so a partial sweep cannot support it."""
    partial = [_point(dependence=0.25), _point(dependence=0.5)]
    with pytest.raises(evidence.EvidenceError, match="must span"):
        evidence._validate(partial)


def test_a_calibrated_study_with_the_expected_contrast_passes() -> None:
    evidence._validate(_sweep(block_rate=0.06, naive_rate=0.40))


def test_holm_family_shows_correction_changing_conclusions() -> None:
    """The panel is only worth plotting if correction actually changes something."""
    frame = evidence._holm_family()
    raw = frame[frame["stage"] == "raw"].set_index("test")["p_value"]
    adjusted = frame[frame["stage"] == "Holm-adjusted"].set_index("test")["p_value"]
    assert (adjusted >= raw).all(), "Holm can only increase a p-value"
    crossed = [name for name in raw.index if raw[name] <= 0.05 < adjusted[name]]
    assert crossed, "the family must contain a test that correction moves past alpha"


def test_the_study_is_deterministic_under_a_fixed_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decision that cannot be reproduced is not evidence."""
    monkeypatch.setattr(evidence, "DEPENDENCE_LEVELS", (0.0, 1.0))
    monkeypatch.setattr(evidence, "COHORTS_PER_LEVEL", 8)
    monkeypatch.setattr(evidence, "STUDY_REPLICATES", 200)
    first = evidence.run_calibration_study(seed=7)
    second = evidence.run_calibration_study(seed=7)
    assert first == second


def test_render_writes_a_figure_and_a_summary_carrying_its_seed(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "evidence.png"
    summary = evidence.render(_sweep(block_rate=0.06, naive_rate=0.40), destination, seed=99)
    assert destination.is_file() and destination.stat().st_size > 0
    assert summary["seed"] == 99
    assert summary["figure_bytes"] == destination.stat().st_size
    # The summary must round-trip: it is the machine-readable record behind the plot.
    assert json.loads(json.dumps(summary, sort_keys=True))["calibration"][-1]["dependence"] == 1.0


def test_main_reports_a_refusal_with_a_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed premise must fail the command, not quietly emit a figure."""
    monkeypatch.setattr(evidence, "_validate", _raise_evidence_error)
    destination = tmp_path / "refused.png"
    assert evidence.main(["--output", str(destination)]) == 2
    assert not destination.exists()


def _raise_evidence_error(_points: object) -> None:
    raise evidence.EvidenceError("the calibration study produced no points")
