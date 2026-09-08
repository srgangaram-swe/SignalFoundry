"""Tests for the backtest API and dashboard presentation (SF-S2-MR10b/c).

The endpoint functions are exercised directly (no HTTP server / httpx needed);
the dashboard's pure presentation helpers are tested without a Streamlit runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("plotly")

from fastapi import HTTPException  # noqa: E402

import alphaforge  # noqa: E402
import apps.api as api_module  # noqa: E402
from alphaforge.service import BacktestRequest, BacktestResult, run_backtest_service  # noqa: E402
from apps.api import BacktestSpec, app, catalog, create_backtest  # noqa: E402
from apps.dashboard import (  # noqa: E402
    build_comparison_table,
    build_drawdown_figure,
    build_equity_figure,
    format_metric_tiles,
)

FAST: dict[str, Any] = dict(
    model="momentum_baseline",
    baselines=["zero_baseline"],
    n_symbols=5,
    n_days=440,
    min_train_days=180,
    test_days=60,
    step_days=60,
)


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return run_backtest_service(BacktestRequest(**{**FAST, "baselines": tuple(FAST["baselines"])}))


# --- API ---------------------------------------------------------------------


def test_api_version_matches_package_version() -> None:
    assert app.version == alphaforge.__version__


def test_catalog_lists_options() -> None:
    body = catalog()
    assert "random_forest" in body["models"]
    assert "zero_baseline" in body["baselines"]
    assert "long_short" in body["strategies"]
    assert body["data_sources"] == ["synthetic", "signal_foundry"]
    assert "confidence_weighted" in body["strategies"]
    assert "confidence" not in body["strategies"]
    assert "Not financial advice" in body["disclaimer"]


def test_create_backtest_returns_result() -> None:
    body = create_backtest(BacktestSpec(**FAST))
    assert body["headline"]["name"] == "momentum_baseline"
    assert {row["name"] for row in body["comparison"]} == {"momentum_baseline", "zero_baseline"}
    assert "Not financial advice" in body["disclaimer"]
    assert body["reproducibility"]["data_id"].startswith("synthetic:")


def test_create_backtest_rejects_unknown_model() -> None:
    with pytest.raises(HTTPException) as exc:
        create_backtest(BacktestSpec(model="does_not_exist"))
    assert exc.value.status_code == 404


def test_spec_rejects_out_of_range_values() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BacktestSpec(n_symbols=1)  # below ge=2
    with pytest.raises(ValidationError):
        BacktestSpec(cost_bps=-1.0)
    with pytest.raises(ValidationError):
        BacktestSpec(cost_bps=float("nan"))
    with pytest.raises(ValidationError):
        BacktestSpec(strategy="confidence")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BacktestSpec(unexpected=True)  # type: ignore[call-arg]


def test_openapi_has_typed_catalog_and_backtest_schemas() -> None:
    schema = app.openapi()
    assert schema["paths"]["/catalog"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/CatalogResponse")
    assert schema["paths"]["/backtests"]["post"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/BacktestResponse")


def test_api_bundle_selection_cannot_escape_catalog_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundles"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(api_module, "_BUNDLES_ROOT", root)

    with pytest.raises(HTTPException) as exc:
        create_backtest(
            BacktestSpec(
                data_source="signal_foundry",
                bundle_dir=str(outside),
                benchmark_symbol="SPY",
            )
        )
    assert exc.value.status_code == 404
    assert "not found" in str(exc.value.detail)


# --- Dashboard presentation helpers ------------------------------------------


def test_metric_tiles(result: BacktestResult) -> None:
    tiles = format_metric_tiles(result)
    labels = [label for label, _ in tiles]
    assert labels == [
        "Total return",
        "Sharpe",
        "Sortino",
        "Max drawdown",
        "Annual vol",
        "Avg turnover",
    ]
    assert all(isinstance(value, str) for _, value in tiles)


def test_comparison_table(result: BacktestResult) -> None:
    table = build_comparison_table(result)
    assert "name" in table.columns and "sharpe" in table.columns
    assert set(table["name"]) == {"momentum_baseline", "zero_baseline"}


def test_equity_and_drawdown_figures(result: BacktestResult) -> None:
    equity = build_equity_figure(result)
    drawdown = build_drawdown_figure(result)
    assert len(equity.data) == 2  # strategy + benchmark
    assert equity.data[0].name == "momentum_baseline"
    assert len(drawdown.data) == 1
    # colorblind-safe palette wired through
    assert equity.data[0].line.color == "#0072B2"
