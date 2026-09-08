"""Compose existing AlphaForge mathematics behind the bounded worker contract.

This module is imported only inside the isolated AlphaForge environment. It adds
orchestration and safe aggregate projections, not alternative model/backtest math.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_foundry.boundary import (
    FoundryError,
    code_identity,
    encode,
    package_identity,
)
from signal_foundry.contracts import (
    Column,
    EvidenceTable,
    ResearchEvidence,
    ResearchRequest,
    Scalar,
    SeedAssignment,
    Validation,
)
from signal_foundry.worker_catalog import specifications
from signal_foundry.worker_data import bundle_path

CAPITAL = 1_000_000.0  # simulated accounting units; no actual capital authorization


@dataclass(frozen=True)
class Prepared:
    panel: pd.DataFrame
    validation: Validation
    specs: list[dict[str, Any]]
    features: pd.DataFrame
    labels: pd.DataFrame


def fold_config(request: ResearchRequest) -> Any:
    from alphaforge.training import WalkForwardConfig

    return WalkForwardConfig(
        scheme=request.folds.scheme,
        min_train_days=request.folds.train_days,
        test_days=request.folds.test_days,
        step_days=request.folds.test_days,
        embargo_days=request.folds.embargo_days,
    )


def decision_features(
    panel: pd.DataFrame, features: pd.DataFrame, request: ResearchRequest
) -> pd.DataFrame:
    """Reserve calendar space for decisions and lagged terminal liquidation."""
    calendar = pd.Index(sorted(panel["date"].unique()))
    reserve = request.costs.rebalance_days + request.costs.execution_lag + 1
    return features.loc[features["date"].lt(calendar[-reserve])]


def model_inputs(
    panel: pd.DataFrame, request: ResearchRequest
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preflight actual transforms on training folds, before queue admission.

    No test target or distribution determines preprocessing or feature selection.
    Execution uses the identical registered technical-feature configuration.
    """
    from alphaforge.features import (
        FittedFeatureTransformer,
        FittedTransformSpec,
        build_features,
    )
    from alphaforge.labels.labels import build_labels
    from alphaforge.training.walk_forward import (
        make_walk_forward_splits,
        supervised_frame,
    )

    try:
        features = build_features(
            panel, benchmark_symbol=request.data.benchmark, config={"hmm_regime": False}
        )
        labels = build_labels(
            panel,
            benchmark_symbol=request.data.benchmark,
            horizons=[request.folds.horizon],
        )
        target = f"fwd_ret_{request.folds.horizon}"
        supervised, columns = supervised_frame(
            decision_features(panel, features, request), labels, target
        )
        windows = make_walk_forward_splits(
            supervised["date"], fold_config(request), max_horizon=request.folds.horizon
        )
        if not windows:
            raise ValueError("no chronological windows")
        for window in windows:
            train = supervised.loc[
                supervised["date"].between(window.train_start, window.train_end)
            ].dropna(subset=[target])
            if request.folds.standardize:
                FittedFeatureTransformer(FittedTransformSpec(enabled=True)).fit(
                    train[columns], train["date"]
                )
        return features, labels
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise FoundryError(
            "feature_incompatible",
            "Data, history and feature profile do not satisfy training-fold contracts.",
        ) from exc


def prepare(request: ResearchRequest, bundles: Path | None) -> Prepared:
    from alphaforge.data.signal_foundry import load_signal_foundry_dataset
    from alphaforge.data.synthetic import (
        SyntheticMarketConfig,
        generate_synthetic_market,
    )

    specs = specifications(request)
    limitations = [
        (
            "Development walk-forward evidence; interactive selection is not"
            " final-holdout qualification."
        ),
        "Daily-bar costs and fills are assumptions, not observed broker executions.",
    ]
    if request.data.kind == "synthetic":
        panel = generate_synthetic_market(
            SyntheticMarketConfig(
                n_symbols=request.data.symbols,
                n_days=request.data.days,
                seed=request.seed,
            )
        )
        identity = "synthetic:" + request.data.digest() + f":{request.seed}"
        limitations.append(
            "Synthetic market mechanics are not evidence of an investable edge."
        )
    else:
        if bundles is None or request.data.bundle_id is None:
            raise FoundryError(
                "dataset_unavailable", "No approved bundle directory is configured."
            )
        try:
            dataset = load_signal_foundry_dataset(
                bundle_path(bundles, request.data.bundle_id)
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise FoundryError(
                "invalid_dataset", "AlphaForge rejected the selected producer bundle."
            ) from exc
        if dataset.bundle_id != request.data.bundle_id:
            raise FoundryError(
                "dataset_identity", "Consumer and requested bundle identities disagree."
            )
        panel = dataset.panel
        identity = "bundle:" + dataset.bundle_id
        limitations.append(
            "Historical/current-vintage prices do not establish current-market"
            " freshness."
        )
        diagnostic = dataset.point_in_time_diagnostics
        if diagnostic is None:
            raise FoundryError(
                "data_provenance", "The consumer did not return temporal diagnostics."
            )
        limitations.extend(diagnostic.warnings)
    symbols = int(panel["symbol"].nunique())
    sessions = int(panel["date"].nunique())
    if len(panel) > 40_000 or symbols > 32 or sessions > 2000:
        raise FoundryError(
            "dataset_limit", "The decoded dataset exceeds interactive limits."
        )
    if request.data.benchmark not in set(panel["symbol"]):
        raise FoundryError(
            "benchmark_unavailable",
            "Choose a benchmark present in the selected dataset.",
        )
    if (
        sessions
        <= request.folds.train_days
        + request.folds.embargo_days
        + request.folds.horizon
        + request.costs.rebalance_days
        + request.costs.execution_lag
        + 20
    ):
        raise FoundryError(
            "insufficient_history", "Dataset leaves no out-of-sample test interval."
        )
    features, labels = model_inputs(panel, request)
    return Prepared(
        panel,
        Validation(
            request_hash=request.digest(),
            data_identity=identity,
            observations=len(panel),
            symbols=symbols,
            sessions=sessions,
            limitations=tuple(limitations),
        ),
        specs,
        features,
        labels,
    )


def scalar(value: Any) -> Scalar:
    """Project a scalar without object repr, raw buffers, or non-finite JSON."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    raise FoundryError(
        "unsupported_evidence", "A diagnostic contains an unsupported scalar.", 500
    )


def table(
    name: str,
    frame: pd.DataFrame,
    fields: dict[str, str],
    description: str,
) -> EvidenceTable:
    """Allowlisted aggregate projection; explicit total count exposes truncation."""
    columns = [name for name in fields if name in frame.columns]
    if not columns:
        # Empty source diagnostics still have a declared schema, never a made-up
        # zero. Consumers render the empty state and the scientific limitation.
        if not frame.empty:
            raise FoundryError(
                "evidence_schema", "Source diagnostic columns changed.", 500
            )
        columns = list(fields)
        frame = pd.DataFrame(columns=columns)
    rows = tuple(
        tuple(scalar(value) for value in row)
        for row in frame[columns].head(2048).itertuples(index=False, name=None)
    )
    return EvidenceTable(
        name=name,
        description=description,
        columns=tuple(Column(name=column, unit=fields[column]) for column in columns),
        rows=rows,
        total_rows=len(frame),
    )


def _backtest(
    panel: pd.DataFrame, weights: pd.DataFrame, request: ResearchRequest
) -> Any:
    from alphaforge.backtesting import run_backtest

    return run_backtest(
        panel=panel,
        target_weights=weights,
        benchmark_symbol=request.data.benchmark,
        initial_capital=CAPITAL,
        execution_lag=request.costs.execution_lag,
        rebalance_frequency=request.costs.rebalance_days,
        costs={
            "commission_bps": request.costs.commission_bps,
            "half_spread_bps": request.costs.half_spread_bps,
            "slippage_bps": request.costs.slippage_bps,
        },
        risk={
            "max_leverage": 1.0,
            "drawdown_deleverage": request.risk.drawdown_deleverage,
            "drawdown_cut": 0.5,
        },
        execution={
            "price_field": "open",
            "max_participation_rate": request.risk.participation_rate,
        },
        carry={
            "short_borrow_bps_annual": request.costs.short_borrow_bps_annual,
            "cash_financing_bps_annual": request.costs.cash_financing_bps_annual,
        },
        liquidate_at_end=True,
    )


def _diagnostics(
    result: Any,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    request: ResearchRequest,
    curve: pd.DataFrame,
) -> list[EvidenceTable]:
    from alphaforge.evaluation import (
        CapacityColumns,
        CapacityConfig,
        estimate_capacity,
        information_coefficient_by_date,
        quantile_return_table,
    )
    from alphaforge.risk import (
        exposure_summary,
        regime_performance,
        stress_test_summary,
    )

    regimes = features.groupby("date")["high_vol_regime"].first()
    active = result.weights.groupby("date")["weight"].apply(
        lambda values: values.abs().sum()
    )
    active_dates = active.loc[active.gt(0)].index
    snapshot_date = (
        active_dates.max() if len(active_dates) else result.weights["date"].max()
    )
    latest = result.weights.loc[result.weights["date"].eq(snapshot_date)]
    attribution = result.pnl_attribution.groupby("symbol", as_index=False)[
        ["market_pnl", "trading_cost", "net_pnl"]
    ].sum()
    diagnostics = [
        table(
            "signal_ic",
            information_coefficient_by_date(predictions),
            {
                "date": "market date",
                "n_obs": "observations",
                "ic": "correlation",
                "rank_ic": "correlation",
            },
            "Out-of-sample cross-sectional correlation; undefined correlations remain"
            " missing.",
        ),
        table(
            "prediction_quantiles",
            quantile_return_table(predictions),
            {
                "quantile": "bin",
                "mean_return": "target return fraction",
                "median_return": "target return fraction",
                "count": "observations",
            },
            "Same-date prediction ranks versus realized targets, aggregated across"
            " development folds.",
        ),
        table(
            "regimes",
            regime_performance(curve, regimes, "high_vol_regime"),
            {
                "high_vol_regime": "regime label",
                "n_days": "sessions",
                "annual_return": "return fraction/year",
                "sharpe": "annualized ratio",
                "max_drawdown": "return fraction",
            },
            "Causal volatility regimes; small or missing regimes do not establish"
            " stability.",
        ),
        table(
            "holdings",
            latest,
            {
                "date": "market date",
                "symbol": "instrument",
                "weight": "capital fraction",
            },
            "Last nonzero marked portfolio snapshot, before terminal liquidation; not"
            " current positions.",
        ),
        table(
            "concentration",
            pd.DataFrame([exposure_summary(latest)]),
            {
                "gross_exposure": "capital fraction",
                "net_exposure": "capital fraction",
                "n_positions": "instruments",
                "max_abs_weight": "capital fraction",
                "hhi_concentration": "gross-normalized Herfindahl index",
            },
            "Last nonzero marked snapshot; inverse HHI is an effective position count,"
            " not independent bets.",
        ),
        table(
            "attribution",
            attribution,
            {
                "symbol": "instrument",
                "market_pnl": "simulated USD",
                "trading_cost": "simulated USD",
                "net_pnl": "simulated USD",
            },
            "Whole-run simulated P&L attribution; no underlying price bars are"
            " returned.",
        ),
        table(
            "stress",
            stress_test_summary(
                latest,
                scenarios=[
                    {"name": "market_down_10", "market_shock": -0.1},
                    {"name": "market_down_20", "market_shock": -0.2},
                    {"name": "market_up_10", "market_shock": 0.1},
                ],
            ),
            {
                "scenario": "scenario",
                "market_shock": "return fraction",
                "portfolio_beta": "assumed beta",
                "estimated_portfolio_return": "return fraction",
            },
            "Last nonzero snapshot beta-one sensitivity, not an execution replay or"
            " crash-loss probability.",
        ),
    ]
    if not result.fills.empty:
        capacity = estimate_capacity(
            result.fills,
            CapacityConfig(
                reference_aum=CAPITAL,
                aum_values=(CAPITAL / 2, CAPITAL, CAPITAL * 2, CAPITAL * 5),
                max_participation_rate=request.risk.participation_rate,
                columns=CapacityColumns.for_fill_records(),
            ),
        )
        diagnostics.append(
            table(
                "capacity",
                capacity.curve,
                {
                    "aum_multiple": "relative simulated AUM",
                    "fill_ratio": "fraction",
                    "participation_p95": "fraction",
                    "modeled_cost_bps_per_traded_notional": "bps",
                },
                "Daily-bar capacity sensitivity from lagged liquidity assumptions, not"
                " an AUM forecast.",
            )
        )
        counts = (
            result.fills.groupby("fill_date", as_index=False)
            .size()
            .rename(columns={"size": "fill_rows", "fill_date": "date"})
        )
        diagnostics.append(
            table(
                "execution",
                counts,
                {"date": "market date", "fill_rows": "simulated fill records"},
                "Aggregate next-open execution record counts, not live orders or raw"
                " fill prices.",
            )
        )
    return diagnostics


def research(
    request: ResearchRequest, bundles: Path | None, root: Path
) -> ResearchEvidence:
    from alphaforge.evaluation import BlockBootstrapConfig, block_bootstrap_interval
    from alphaforge.features import FittedTransformSpec
    from alphaforge.portfolio import construct_portfolio
    from alphaforge.risk import drawdown_series, performance_summary
    from alphaforge.signals import apply_regime_filter, build_signals
    from alphaforge.training import run_walk_forward

    prepared = prepare(request, bundles)
    features, labels = prepared.features, prepared.labels
    learned = run_walk_forward(
        decision_features(prepared.panel, features, request),
        labels,
        prepared.specs,
        target=f"fwd_ret_{request.folds.horizon}",
        config=fold_config(request),
        max_horizon=request.folds.horizon,
        transform_config=FittedTransformSpec(enabled=request.folds.standardize),
    )
    if learned.predictions.empty:
        raise FoundryError(
            "no_predictions", "No out-of-sample predictions were produced."
        )
    start = learned.predictions["date"].min()
    tables = [
        table(
            "folds",
            learned.windows,
            {
                "window_id": "fold",
                "train_start": "market date",
                "train_end": "market date",
                "test_start": "market date",
                "test_end": "market date",
                "embargo_days": "sessions",
                "train_rows": "observations",
                "test_rows": "observations",
            },
            "Chronological folds; test windows never overlap and embargo covers the"
            " label horizon.",
        ),
        table(
            "learning",
            learned.metrics,
            {
                "model": "registry identity",
                "window_id": "fold",
                "mae": "target return fraction",
                "mse": "squared target return fraction",
                "ic": "correlation",
                "rank_ic": "correlation",
                "training_status": "termination state",
                "training_iterations": "iterations",
                "training_warning_count": "warnings",
            },
            "Out-of-sample fold errors and actual fit termination diagnostics; warnings"
            " are not convergence.",
        ),
    ]
    comparisons: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(f"{request.seed}:return-bootstrap".encode()).digest()[:4], "big"
    )
    for spec in prepared.specs:
        name = spec["name"]
        predictions = learned.predictions.loc[learned.predictions["model"].eq(name)]
        if predictions.empty:
            raise FoundryError(
                "missing_model_evidence", "A requested model produced no evidence."
            )
        signals = build_signals(predictions, strategy=request.strategy)
        if request.regime == "causal_volatility":
            signals = apply_regime_filter(signals, features)
        weights = construct_portfolio(
            signals,
            features=features,
            config={
                "max_weight": request.risk.max_weight,
                "max_gross_exposure": request.risk.max_gross,
                "max_net_exposure": request.risk.max_net,
                "turnover_cap": request.risk.turnover_cap,
                "inverse_vol_scaling": request.risk.inverse_volatility,
            },
        )
        result = _backtest(prepared.panel, weights, request)
        curve = result.equity_curve.loc[result.equity_curve["date"].ge(start)].copy()
        curve["cumulative_return"] = curve["equity"] / CAPITAL - 1.0
        curve["drawdown"] = drawdown_series(curve["equity"])
        comparisons.append(
            {"model": name, **performance_summary(curve, trim_inactive=False)}
        )
        tables.append(
            table(
                f"equity_{name}",
                curve,
                {
                    "date": "market date",
                    "cumulative_return": "net return fraction",
                    "drawdown": "return fraction",
                    "gross_exposure": "capital fraction",
                    "net_exposure": "capital fraction",
                    "turnover": "capital fraction/session",
                    "transaction_cost": "capital fraction/session",
                },
                "Costed strategy over the common out-of-sample calendar; initial"
                " accounting capital is simulated.",
            )
        )
        interval = block_bootstrap_interval(
            curve["return"].to_numpy(),
            BlockBootstrapConfig(
                n_resamples=200,
                block_length=min(20, len(curve)),
                seed=bootstrap_seed,
            ),
        )
        intervals.append(
            {
                "model": name,
                "mean": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "observations": interval.n_observations,
                "resamples": interval.n_resamples,
            }
        )
        if name == request.model.name:
            tables.extend(_diagnostics(result, predictions, features, request, curve))
    tables.extend(
        [
            table(
                "comparison",
                pd.DataFrame(comparisons),
                {
                    "model": "registry identity",
                    "total_return": "net return fraction",
                    "annual_return": "net return fraction/year",
                    "annual_volatility": "fraction/year^0.5",
                    "sharpe": "annualized ratio",
                    "max_drawdown": "return fraction",
                    "average_turnover": "fraction/session",
                    "transaction_cost_impact": "sum of session cost fractions",
                    "average_gross_exposure": "capital fraction",
                },
                "Forecast models/baselines passed through identical strategy, folds,"
                " risk and cost policies; not a model ranking or qualification.",
            ),
            table(
                "uncertainty",
                pd.DataFrame(intervals),
                {
                    "model": "registry identity",
                    "mean": "net return fraction/session",
                    "lower": "95% lower bound",
                    "upper": "95% upper bound",
                    "observations": "sessions",
                    "resamples": "bootstrap draws",
                },
                "Moving-block mean-return intervals assume local stationarity; 200"
                " resamples are exploratory and do not correct model selection.",
            ),
        ]
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": sorted(
            (item.metadata["Name"], item.version)
            for item in importlib.metadata.distributions()
        ),
    }
    return ResearchEvidence(
        request=request,
        request_hash=request.digest(),
        data_identity=prepared.validation.data_identity,
        code_hash=code_identity(root),
        source_code_hash=package_identity(root, "alphaforge"),
        environment_hash=hashlib.sha256(encode(environment)).hexdigest(),
        seed_map=(
            SeedAssignment(name="data_and_model_seed", value=request.seed),
            SeedAssignment(name="return_bootstrap", value=bootstrap_seed),
        ),
        tables=tuple(tables),
        limitations=(
            *prepared.validation.limitations,
            (
                "Technical feature profile excludes HMM; trailing sessions are reserved"
                " for lagged terminal liquidation."
            ),
            (
                "Bootstrap intervals assume local stationarity and do not adjust for"
                " repeated interactive trials."
            ),
            (
                "Forecast baselines use the selected signal policy; their names do not"
                " imply a separate passive portfolio."
            ),
            (
                "Risk caps constrain targets; market movement can cause realized"
                " weights to drift."
            ),
            (
                "No intraday order-book, queue-position, current borrow, tax or"
                " live-latency evidence is established."
            ),
        ),
    )
