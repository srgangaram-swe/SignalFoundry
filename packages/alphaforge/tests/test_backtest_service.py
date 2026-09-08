"""Tests for the backtest orchestration service (SF-S2-MR10a)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import alphaforge.service.backtest_service as service_module
from alphaforge.data.synthetic import SyntheticMarketConfig, generate_synthetic_market
from alphaforge.service import (
    BacktestRequest,
    BacktestResourceNotFoundError,
    BacktestResult,
    BacktestServiceError,
    available_baselines,
    available_strategy_models,
    discover_bundles,
    run_backtest_service,
)

# A small, fast configuration using cheap baseline "models" (no heavy training).
FAST: dict[str, Any] = dict(
    model="momentum_baseline",
    baselines=("zero_baseline", "historical_mean"),
    n_symbols=5,
    n_days=440,
    min_train_days=180,
    test_days=60,
    step_days=60,
    seed=7,
)


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return run_backtest_service(BacktestRequest(**FAST))


# --- Catalog helpers ---------------------------------------------------------


def test_catalog_helpers() -> None:
    models = available_strategy_models()
    assert "random_forest" in models and "equal_probability" not in models
    assert set(available_baselines()) <= set(models) | set(available_baselines())
    assert "zero_baseline" in available_baselines()
    assert discover_bundles("does/not/exist") == []


# --- Request validation (no pipeline run) ------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "nope"},
        {"baselines": ("zero_baseline", "nope")},
        {"data_source": "mars"},
        {"data_source": "signal_foundry"},  # missing bundle_dir
        {"strategy": "bogus"},
        {"strategy": "confidence"},
        {"horizon": 0},
        {"horizon": 11, "embargo_days": 10},
        {"cost_bps": -1.0},
        {"cost_bps": float("nan")},
        {"seed": -1},
        {"seed": True},
        {"n_symbols": 1},
        {"n_symbols": True},
        {"n_days": 5001},
        {"min_train_days": 0},
        {"model": "equal_probability"},
        {"baselines": ("equal_probability",)},
        {"bundle_dir": "not-allowed-for-synthetic"},
    ],
)
def test_invalid_requests_fail_closed(overrides: dict) -> None:
    with pytest.raises(BacktestServiceError):
        BacktestRequest(**{**FAST, **overrides})


def test_baselines_dedupe_and_drop_headline() -> None:
    request = BacktestRequest(
        model="zero_baseline",
        baselines=("zero_baseline", "momentum_baseline", "momentum_baseline"),
    )
    assert request.baselines == ("momentum_baseline",)  # headline removed, deduped


def test_request_bounds_json_model_parameters() -> None:
    with pytest.raises(BacktestServiceError, match="finite"):
        BacktestRequest(**{**FAST, "model_params": {"alpha": np.inf}})
    with pytest.raises(BacktestServiceError, match="service limit"):
        BacktestRequest(**{**FAST, "model_params": {"n_estimators": 2_001}})
    with pytest.raises(BacktestServiceError, match="JSON-compatible"):
        BacktestRequest(**{**FAST, "model_params": {"callback": object()}})


def test_request_copies_and_deeply_freezes_model_parameters() -> None:
    source: dict[str, Any] = {"members": [{"name": "ridge", "params": {"alpha": 2.0}}]}
    request = BacktestRequest(**{**FAST, "model_params": source})
    source["members"][0]["params"]["alpha"] = 999.0

    assert request.to_dict()["model_params"]["members"][0]["params"]["alpha"] == 2.0
    with pytest.raises(TypeError):
        request.model_params["members"] = ()  # type: ignore[index]


def test_config_hash_is_stable_and_sensitive() -> None:
    a = BacktestRequest(**FAST)
    b = BacktestRequest(**FAST)
    c = BacktestRequest(**{**FAST, "seed": 999})
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


# --- Result structure and correctness ----------------------------------------


def test_headline_is_the_chosen_model(result: BacktestResult) -> None:
    assert result.headline.name == FAST["model"]
    assert result.headline.is_headline is True


def test_comparison_covers_model_and_all_baselines(result: BacktestResult) -> None:
    names = {row["name"] for row in result.comparison}
    assert names == {"momentum_baseline", "zero_baseline", "historical_mean"}
    assert result.headline.name == "momentum_baseline"
    # Here the headline model is itself a baseline, so all rows are flagged.
    assert all(row["is_baseline"] for row in result.comparison)


def test_non_baseline_headline_is_not_flagged_baseline() -> None:
    # A real model (linear) as headline is not flagged a baseline; the requested
    # baselines are. Linear trains cheaply, so this stays fast.
    request = BacktestRequest(
        model="linear",
        baselines=("zero_baseline",),
        n_symbols=4,
        n_days=380,
        min_train_days=160,
        test_days=60,
        step_days=60,
    )
    result = run_backtest_service(request)
    assert result.headline.name == "linear" and result.headline.is_baseline is False
    flags = {row["name"]: row["is_baseline"] for row in result.comparison}
    assert flags == {"linear": False, "zero_baseline": True}


def test_result_has_evidence(result: BacktestResult) -> None:
    assert result.n_windows >= 1
    assert result.n_observations > 0
    assert len(result.headline.equity_curve) > 0
    point = result.headline.equity_curve[0]
    assert set(point) == {"date", "strategy_cum", "benchmark_cum", "drawdown"}
    assert "sharpe" in result.headline.metrics


def test_reproducibility_fields(result: BacktestResult) -> None:
    repro = result.reproducibility
    assert repro["seed"] == FAST["seed"]
    assert repro["data_id"] == result.data_id
    assert result.data_id.startswith("synthetic:")
    assert len(repro["config_hash"]) == 16


def test_result_is_json_serializable(result: BacktestResult) -> None:
    text = json.dumps(result.to_dict())
    assert "Not financial advice" in text
    # no NaN/Infinity tokens leaked into the JSON
    assert "NaN" not in text and "Infinity" not in text


def test_deterministic_replay() -> None:
    a = run_backtest_service(BacktestRequest(**FAST))
    b = run_backtest_service(BacktestRequest(**FAST))
    assert a.headline.metrics == b.headline.metrics
    assert a.comparison == b.comparison
    assert a.reproducibility["config_hash"] == b.reproducibility["config_hash"]


def test_signal_foundry_bundle_source_runs_identical_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the bundle service path at the verified loader seam.

    The data loader has its own adversarial manifest/parquet integration suite;
    here a producer-shaped dataset proves the service composes that boundary
    through features, walk-forward training, and the backtester.
    """
    bundle = tmp_path / ("a" * 64)
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    panel = generate_synthetic_market(SyntheticMarketConfig(n_symbols=5, n_days=440, seed=13))
    monkeypatch.setattr(
        service_module,
        "load_signal_foundry_dataset",
        lambda path: SimpleNamespace(panel=panel, bundle_id="a" * 64),
    )

    request = BacktestRequest(
        **{
            **FAST,
            "data_source": "signal_foundry",
            "bundle_dir": str(bundle),
            "benchmark_symbol": "BENCH",
        }
    )
    bundle_result = run_backtest_service(request)

    assert bundle_result.data_id == f"bundle:{'a' * 64}"
    assert bundle_result.n_observations == len(panel)
    assert bundle_result.headline.name == FAST["model"]
    assert bundle_result.n_windows >= 1


def test_missing_signal_foundry_bundle_is_typed_resource_error(tmp_path: Path) -> None:
    request = BacktestRequest(
        **{
            **FAST,
            "data_source": "signal_foundry",
            "bundle_dir": str(tmp_path / "missing"),
            "benchmark_symbol": "BENCH",
        }
    )
    with pytest.raises(BacktestResourceNotFoundError, match="not found"):
        run_backtest_service(request)


def test_model_configuration_failure_is_mapped_to_service_error() -> None:
    request = BacktestRequest(
        **{**FAST, "model": "ridge", "model_params": {"not_a_ridge_parameter": 1}}
    )
    with pytest.raises(BacktestServiceError, match="could not be completed") as exc:
        run_backtest_service(request)
    assert isinstance(exc.value.__cause__, TypeError)


def test_confidence_weighted_strategy_is_wired_to_signal_contract() -> None:
    request = BacktestRequest(**{**FAST, "strategy": "confidence_weighted"})
    confidence_result = run_backtest_service(request)
    assert confidence_result.headline.equity_curve
