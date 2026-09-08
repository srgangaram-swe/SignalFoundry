"""Governed development-selection and frozen final-holdout research.

The workflow consumes one verified Signal Foundry bundle, selects a candidate
using development-only walk-forward evidence, purges an embargo before the
pre-registered holdout, evaluates that holdout once per immutable run identity,
and emits an auditable paper-readiness dossier. It has no live-order path.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.backtesting import BacktestResult, run_backtest
from alphaforge.data.signal_foundry import SignalFoundryDataset
from alphaforge.evaluation import (
    CapacityColumns,
    CapacityConfig,
    ReadinessThresholds,
    assess_paper_readiness,
    estimate_capacity,
    information_coefficient_by_date,
    probability_of_backtest_overfitting,
)
from alphaforge.execution import ExecutionStressProfile, standard_stress_profiles
from alphaforge.features import (
    FittedFeatureTransformer,
    FittedTransformSpec,
    build_features,
)
from alphaforge.labels.labels import build_labels
from alphaforge.models.registry import create_model, seed_model_specs
from alphaforge.paper import audit_offline_paper_controls
from alphaforge.portfolio import construct_portfolio
from alphaforge.research.manifest import (
    ExperimentManifest,
    capture_environment,
    capture_git_context,
    inventory_artifacts,
    redact_cli_arguments,
)
from alphaforge.risk import (
    drawdown_series,
    exposure_summary,
    performance_summary,
    regime_performance,
)
from alphaforge.signals import build_signals
from alphaforge.training import run_walk_forward
from alphaforge.training.walk_forward import supervised_frame
from alphaforge.utils import set_seed


@dataclass(frozen=True)
class GovernedResearchConfig:
    """Pre-registered final-holdout and model-selection policy."""

    holdout_start: str
    benchmark_symbol: str = "SPY"
    target: str = "fwd_ret_5"
    horizons: tuple[int, ...] = (1, 5, 20)
    selection_metric: str = "rank_ic"
    seed: int = 42

    def __post_init__(self) -> None:
        holdout = pd.Timestamp(self.holdout_start)
        if holdout.tzinfo is not None:
            raise ValueError("holdout_start must be a timezone-naive market date")
        if not self.benchmark_symbol.strip():
            raise ValueError("benchmark_symbol must be non-empty")
        if not self.target.strip():
            raise ValueError("target must be non-empty")
        if not self.horizons or any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.selection_metric not in {"rank_ic", "ic"}:
            raise ValueError("selection_metric must be rank_ic or ic")


@dataclass(frozen=True)
class GovernedResearchResult:
    """Identity and artifacts for one immutable final-holdout evaluation."""

    run_id: str
    run_dir: Path
    candidate_model: str
    dossier: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot determine producer Git SHA for governed run")
    value = result.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("governed run requires a full Git SHA")
    return value


def _model_matrix(frame: pd.DataFrame, columns: list[str], model: Any) -> pd.DataFrame:
    matrix = frame[columns].copy()
    if getattr(model, "needs_sequence_index", False):
        matrix.index = pd.MultiIndex.from_frame(frame[["date", "symbol"]])
    return matrix


def _transformed_model_matrix(
    matrix: pd.DataFrame, frame: pd.DataFrame, model: Any
) -> pd.DataFrame:
    result = matrix.copy()
    if getattr(model, "needs_sequence_index", False):
        result.index = pd.MultiIndex.from_frame(frame[["date", "symbol"]])
    return result


def _validate_model_specs(model_specs: list[dict[str, Any]]) -> None:
    if not model_specs:
        raise ValueError("at least one pre-registered model is required")
    names: list[str] = []
    for spec in model_specs:
        if set(spec) - {"name", "params"}:
            raise ValueError(
                f"unknown model specification fields: {sorted(set(spec) - {'name', 'params'})}"
            )
        name = spec.get("name")
        params = spec.get("params", {})
        if not isinstance(name, str) or not name.strip() or not isinstance(params, dict):
            raise ValueError("each model specification requires a name and parameter mapping")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("model names must be unique within a governed trial set")


def _development_cutoff(
    dates: pd.Series,
    holdout_start: pd.Timestamp,
    embargo_sessions: int,
) -> pd.Timestamp:
    unique = pd.DatetimeIndex(pd.to_datetime(dates).drop_duplicates().sort_values())
    holdout_position = int(unique.searchsorted(holdout_start, side="left"))
    cutoff_position = holdout_position - embargo_sessions - 1
    if holdout_position >= len(unique) or cutoff_position < 0:
        raise ValueError("holdout_start leaves insufficient pre-holdout embargo history")
    return pd.Timestamp(unique[cutoff_position])


def _select_candidate(
    metrics: pd.DataFrame,
    *,
    selection_metric: str,
) -> tuple[str, pd.DataFrame]:
    if metrics.empty or selection_metric not in metrics:
        raise ValueError("development walk-forward produced no selection evidence")
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            windows=("window_id", "nunique"),
            rank_ic=("rank_ic", "mean"),
            ic=("ic", "mean"),
            mae=("mae", "mean"),
        )
        .sort_values(
            [selection_metric, "mae", "model"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    eligible = summary.loc[np.isfinite(summary[selection_metric])].copy()
    if eligible.empty:
        raise ValueError("no model produced finite development selection evidence")
    return str(eligible.iloc[0]["model"]), summary


def _trial_ledger(
    model_specs: list[dict[str, Any]],
    development_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    evidence = development_summary.set_index("model").to_dict(orient="index")
    previous_hash = "0" * 64
    records: list[dict[str, Any]] = []
    for sequence, spec in enumerate(model_specs, start=1):
        payload = {
            "sequence": sequence,
            "model": spec["name"],
            "params": spec.get("params", {}),
            "development_evidence": evidence.get(spec["name"], {}),
            "previous_hash": previous_hash,
        }
        record_hash = _sha256_bytes(_canonical_json(payload))
        records.append({**payload, "record_hash": record_hash})
        previous_hash = record_hash
    return records


def _daily_ic_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for model_name, block in predictions.groupby("model", sort=True):
        daily = information_coefficient_by_date(block)
        columns[str(model_name)] = daily.set_index("date")["rank_ic"]
    return pd.DataFrame(columns).sort_index()


def _pbo(predictions: pd.DataFrame, seed: int) -> dict[str, Any]:
    matrix = _daily_ic_matrix(predictions)
    max_blocks = min(16, len(matrix) // 2)
    n_blocks = max_blocks - max_blocks % 2
    if n_blocks < 2:
        return {"pbo": np.nan, "n_combinations": 0, "n_obs": len(matrix)}
    result = probability_of_backtest_overfitting(
        matrix,
        n_blocks=n_blocks,
        seed=seed,
    )
    return {key: value for key, value in result.items() if key != "logits"}


def _backtest(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_symbol: str,
    backtest_config: dict[str, Any],
    stress_profile: ExecutionStressProfile | None = None,
) -> BacktestResult:
    signal_config = backtest_config.get("strategy_params", {})
    signals = build_signals(
        predictions,
        strategy=str(backtest_config.get("strategy", "long_short")),
        params=signal_config,
    )
    weights = construct_portfolio(
        signals,
        features=features,
        config=backtest_config.get("portfolio", {}),
    )
    return run_backtest(
        panel=panel,
        target_weights=weights,
        benchmark_symbol=benchmark_symbol,
        initial_capital=float(backtest_config.get("initial_capital", 1_000_000.0)),
        execution_lag=int(backtest_config.get("execution_lag", 1)),
        rebalance_frequency=int(backtest_config.get("rebalance_frequency", 1)),
        costs=backtest_config.get("costs", {}),
        risk=backtest_config.get("risk", {}),
        execution=backtest_config.get("execution", {}),
        latency=backtest_config.get("latency", {}),
        carry=backtest_config.get("borrow_financing", {}),
        stress_profile=stress_profile,
        liquidate_at_end=bool(backtest_config.get("liquidate_at_end", True)),
    )


def _gross_performance(equity_curve: pd.DataFrame) -> dict[str, float]:
    """Reconstruct the pre-cost curve from the reconciled ledger output."""

    gross_curve = equity_curve.copy()
    first_net_return = float(gross_curve["return"].iloc[0])
    initial_equity = float(gross_curve["equity"].iloc[0]) / (1.0 + first_net_return)
    gross_curve["return"] = gross_curve["gross_return"].astype(float)
    gross_curve["equity"] = initial_equity * (1.0 + gross_curve["return"]).cumprod()
    return performance_summary(gross_curve)


def _development_economic_summary(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_symbol: str,
    backtest_config: dict[str, Any],
) -> pd.DataFrame:
    """Cost every candidate's identical development-only OOS prediction panel.

    The returned table is aggregate evidence. Row-level orders, fills, weights,
    and predictions remain inside the ignored governed-run directory.
    """

    records: list[dict[str, Any]] = []
    for model_name, candidate_predictions in predictions.groupby("model", sort=True):
        result = _backtest(
            panel=panel,
            predictions=candidate_predictions,
            features=features,
            benchmark_symbol=benchmark_symbol,
            backtest_config=backtest_config,
        )
        net = performance_summary(result.equity_curve)
        gross = _gross_performance(result.equity_curve)
        records.append(
            {
                "model": str(model_name),
                "prediction_rows": int(len(candidate_predictions)),
                "trading_sessions": int(len(result.equity_curve)),
                "order_count": int(len(result.orders)),
                "fill_count": int(len(result.fills)),
                "gross_annual_return": float(gross["annual_return"]),
                "net_annual_return": float(net["annual_return"]),
                "annual_cost_drag": float(gross["annual_return"] - net["annual_return"]),
                "sharpe": float(net["sharpe"]),
                "max_drawdown": float(net["max_drawdown"]),
                "average_turnover": float(net["average_turnover"]),
                "average_gross_exposure": float(net["average_gross_exposure"]),
                "average_net_exposure": float(net["average_net_exposure"]),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("model", kind="stable")


def _stress_scenarios(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_symbol: str,
    backtest_config: dict[str, Any],
    maximum_drawdown: float,
    seed: int,
) -> tuple[list[dict[str, Any]], bool]:
    execution_profiles = standard_stress_profiles()[1:]
    selection_scenarios: list[tuple[str, dict[str, Any]]] = []
    base_strategy = dict(backtest_config.get("strategy_params", {}))
    base_quantile = float(base_strategy.get("quantile", 0.20))
    for name, quantile in (
        ("narrower_selection", base_quantile * 0.75),
        ("wider_selection", min(base_quantile * 1.25, 0.49)),
    ):
        perturbed = dict(backtest_config)
        perturbed["strategy_params"] = {**base_strategy, "quantile": quantile}
        selection_scenarios.append((name, perturbed))

    summaries: list[dict[str, Any]] = []
    passed = True
    for profile in execution_profiles:
        result = _backtest(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=benchmark_symbol,
            backtest_config=backtest_config,
            stress_profile=profile,
        )
        summary = performance_summary(result.equity_curve)
        scenario_passed = bool(
            np.isfinite(summary["max_drawdown"]) and summary["max_drawdown"] >= -maximum_drawdown
        )
        passed = passed and scenario_passed
        summaries.append(
            {
                "scenario": profile.name,
                "passed": scenario_passed,
                "accounting_reconciled": True,
                "stress_profile_digest": profile.digest,
                **summary,
            }
        )

    for name, scenario_config in selection_scenarios:
        result = _backtest(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=benchmark_symbol,
            backtest_config=scenario_config,
        )
        summary = performance_summary(result.equity_curve)
        scenario_passed = bool(
            np.isfinite(summary["max_drawdown"]) and summary["max_drawdown"] >= -maximum_drawdown
        )
        passed = passed and scenario_passed
        summaries.append(
            {
                "scenario": name,
                "passed": scenario_passed,
                "accounting_reconciled": True,
                **summary,
            }
        )

    placebo = predictions.copy()
    rng = np.random.default_rng(seed)
    placebo["prediction"] = placebo.groupby("date", sort=True)["prediction"].transform(
        lambda values: rng.permutation(values.to_numpy())
    )
    placebo_result = _backtest(
        panel=panel,
        predictions=placebo,
        features=features,
        benchmark_symbol=benchmark_symbol,
        backtest_config=backtest_config,
    )
    placebo_summary = performance_summary(placebo_result.equity_curve)
    summaries.append(
        {
            "scenario": "permuted_signal_placebo",
            "accounting_reconciled": True,
            **placebo_summary,
        }
    )
    return summaries, passed


def _bootstrap_uncertainty(
    returns: pd.Series,
    *,
    seed: int,
    n_resamples: int = 500,
    block_size: int = 20,
) -> dict[str, Any]:
    """Circular moving-block intervals for serially dependent daily returns."""
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(clean) < block_size * 2:
        return {
            "method": "circular_moving_block_bootstrap",
            "n_resamples": n_resamples,
            "block_size": block_size,
            "available": False,
        }
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(clean) / block_size))
    annual_returns: list[float] = []
    sharpes: list[float] = []
    for _ in range(n_resamples):
        starts = rng.integers(0, len(clean), size=n_blocks)
        sample = np.concatenate(
            [np.take(clean, np.arange(start, start + block_size), mode="wrap") for start in starts]
        )[: len(clean)]
        years = len(sample) / 252.0
        annual_returns.append(float(np.prod(1.0 + sample) ** (1.0 / years) - 1.0))
        volatility = float(np.std(sample, ddof=1))
        sharpes.append(
            np.nan if volatility == 0.0 else float(np.mean(sample) / volatility * np.sqrt(252.0))
        )
    return {
        "method": "circular_moving_block_bootstrap",
        "n_resamples": n_resamples,
        "block_size": block_size,
        "seed": seed,
        "available": True,
        "annual_return_ci_95": np.quantile(annual_returns, [0.025, 0.975]).tolist(),
        "sharpe_ci_95": np.nanquantile(sharpes, [0.025, 0.975]).tolist(),
    }


def _drawdown_diagnostics(equity_curve: pd.DataFrame) -> dict[str, Any]:
    drawdown = drawdown_series(equity_curve["equity"].astype(float))
    underwater = drawdown < 0.0
    longest = 0
    current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "time_under_drawdown_fraction": float(underwater.mean()),
        "maximum_drawdown_duration_sessions": longest,
    }


def _year_stability(equity_curve: pd.DataFrame) -> list[dict[str, Any]]:
    frame = equity_curve.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    records: list[dict[str, Any]] = []
    for year, block in frame.groupby("year", sort=True):
        records.append({"year": int(year), **performance_summary(block, trim_inactive=False)})
    return records


def _native_borrow_financing_summary(
    result: BacktestResult,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize carry already reconciled inside the event ledger.

    This deliberately does not post-process the return series: financing and
    borrow must have exactly one accounting path through ``CashChargeAccrued``.
    """

    expected = {
        "short_borrow_bps_annual",
        "cash_financing_bps_annual",
        "sessions_per_year",
        "calibration_provenance",
    }
    required = {"short_borrow_bps_annual", "cash_financing_bps_annual"}
    missing = required - set(config)
    unknown = set(config) - expected
    if missing or unknown:
        raise ValueError(
            f"borrow_financing fields mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    latest = result.accounting.iloc[-1]
    reconciled = bool(
        (
            result.accounting["reconciliation_error"]
            <= result.accounting["reconciliation_tolerance"]
        ).all()
    )
    return {
        "scenario": "native_event_ledger_borrow_and_financing",
        "accounting_reconciled": reconciled,
        "short_borrow_bps_annual": float(config["short_borrow_bps_annual"]),
        "cash_financing_bps_annual": float(config["cash_financing_bps_annual"]),
        "sessions_per_year": int(config.get("sessions_per_year", 252)),
        "financing_charges_usd": float(latest["financing"]),
        "borrow_charges_usd": float(latest["borrow"]),
        "method": (
            "native event-ledger accrual after DAY-order termination and before close; "
            "not a locate or borrow-availability model"
        ),
        **performance_summary(result.equity_curve),
    }


def _missing_price_halts(
    *,
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    benchmark_symbol: str,
    backtest_config: dict[str, Any],
) -> bool:
    corrupted = panel.copy()
    corrupted["open"] = np.nan
    try:
        _backtest(
            panel=corrupted,
            predictions=predictions,
            features=features,
            benchmark_symbol=benchmark_symbol,
            backtest_config=backtest_config,
        )
    except (ValueError, RuntimeError):
        return True
    return False


def _capacity_config(backtest_config: dict[str, Any]) -> tuple[CapacityConfig, float]:
    """Translate the strict backtest capacity policy into evaluator inputs."""
    settings = dict(backtest_config.get("capacity", {}))
    required = {"aum_multiples", "max_participation_rate", "minimum_fill_ratio"}
    supported = required | {"enabled", "impact_exponent", "variable_cost_fraction"}
    missing = required - set(settings)
    unknown = set(settings) - supported
    if missing or unknown:
        raise ValueError(
            "capacity fields mismatch; " f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if settings.get("enabled", True) is not True:
        raise ValueError("governed research requires capacity evaluation to be enabled")
    reference_aum = float(backtest_config.get("initial_capital", 1_000_000.0))
    aum_values = tuple(reference_aum * float(multiple) for multiple in settings["aum_multiples"])
    return (
        CapacityConfig(
            reference_aum=reference_aum,
            aum_values=aum_values,
            max_participation_rate=float(settings["max_participation_rate"]),
            impact_exponent=float(settings.get("impact_exponent", 0.5)),
            variable_cost_fraction=float(settings.get("variable_cost_fraction", 0.5)),
            columns=CapacityColumns.for_fill_records(),
        ),
        float(settings["minimum_fill_ratio"]),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _write_markdown_dossier(path: Path, dossier: dict[str, Any]) -> None:
    metrics = dossier["metrics"]
    lines = [
        "# Signal Foundry Final-Holdout Dossier",
        "",
        f"Decision: **{dossier['decision']}**",
        "",
        dossier["scope"],
        "",
        "## Gate results",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in sorted(dossier["gates"].items())
        ],
        "",
        "## Selected metrics",
        "",
        *[
            f"- {name}: {value}"
            for name, value in sorted(metrics.items())
            if isinstance(value, (bool, int, float))
        ],
        "",
        "## Interpretation",
        "",
        "A backtest is historical research evidence, not money and not a promise of future profit. "
        "READY_FOR_PAPER permits only a time-bounded, zero-capital shadow evaluation under the "
        "documented controls. NOT_READY is the mandatory outcome whenever any gate fails.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_governed_signal_foundry_research(
    *,
    dataset: SignalFoundryDataset,
    model_specs: list[dict[str, Any]],
    feature_config: dict[str, Any],
    walk_forward_config: dict[str, Any],
    backtest_config: dict[str, Any],
    research_config: GovernedResearchConfig,
    readiness_thresholds: ReadinessThresholds,
    output_root: str | Path = "runs/signal-foundry",
    code_sha: str | None = None,
    invocation: Mapping[str, Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> GovernedResearchResult:
    """Execute one immutable, governed development/final-holdout evaluation.

    ``clock`` is injectable so reference tests can prove deterministic semantic
    evidence while production runs retain honest start and finish timestamps.
    """
    _validate_model_specs(model_specs)
    set_seed(research_config.seed)
    model_specs = seed_model_specs(model_specs, research_config.seed)
    if research_config.benchmark_symbol not in set(dataset.panel["symbol"]):
        raise ValueError("pre-registered benchmark is absent from the verified bundle")
    if (
        not dataset.manifest["license"]["bundle_must_remain_local"]
        and not dataset.manifest["license"]["observations_redistributable"]
    ):
        raise ValueError("bundle license policy is inconsistent")

    now = clock or (lambda: datetime.now(UTC))
    started_at = now().astimezone(UTC).isoformat().replace("+00:00", "Z")
    git_context = capture_git_context()
    git_context["sha"] = code_sha or _git_sha()
    runtime_environment = capture_environment()
    invocation_record = dict(invocation or {})
    invocation_record.setdefault("entrypoint", Path(sys.argv[0]).name)
    invocation_record["arguments"] = redact_cli_arguments(
        [str(argument) for argument in invocation_record.get("arguments", sys.argv[1:])]
    )
    dataset_record = {
        "bundle_id": dataset.bundle_id,
        "schema_version": dataset.manifest.get("schema_version", "unknown"),
        "license": dataset.manifest["license"],
        "point_in_time_limits": dataset.manifest["point_in_time_limits"],
    }
    universe = sorted(dataset.panel["symbol"].unique().tolist())
    date_range = {
        "start": str(pd.Timestamp(dataset.panel["date"].min()).date()),
        "end": str(pd.Timestamp(dataset.panel["date"].max()).date()),
    }
    label_record = asdict(research_config)
    validation_record = {
        "walk_forward": walk_forward_config,
        "readiness": asdict(readiness_thresholds),
    }
    planned_manifest = ExperimentManifest.build(
        code=git_context,
        dataset=dataset_record,
        universe=universe,
        date_range=date_range,
        features=feature_config,
        label=label_record,
        models=model_specs,
        validation=validation_record,
        transaction_costs=backtest_config,
        root_seed=research_config.seed,
        environment=runtime_environment,
        invocation=invocation_record,
        execution={"started_at": started_at, "finished_at": started_at},
        artifacts=(),
    )
    run_id = planned_manifest.experiment_id
    root = Path(output_root)
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(
            f"final holdout run {run_id} already exists; immutable runs cannot be repeated"
        )
    staging = root / f".publishing-{run_id}"
    if staging.exists():
        raise FileExistsError(f"stale governed-run staging exists: {staging}")
    staging.mkdir(parents=True)

    try:
        panel = dataset.panel
        decision_panel = dataset.decision_panel
        if decision_panel is None:
            if not dataset.source_panel.empty:
                raise ValueError("verified source data lacks a decision-eligible panel")
            decision_panel = panel
        features = build_features(
            decision_panel,
            benchmark_symbol=research_config.benchmark_symbol,
            config=feature_config,
        )
        labels = build_labels(
            panel,
            benchmark_symbol=research_config.benchmark_symbol,
            horizons=list(research_config.horizons),
        )
        max_horizon = max(research_config.horizons)
        holdout_start = pd.Timestamp(research_config.holdout_start)
        development_end = _development_cutoff(
            features["date"],
            holdout_start,
            embargo_sessions=max_horizon,
        )
        development_features = features.loc[features["date"].le(development_end)].copy()
        development_labels = labels.loc[labels["date"].le(development_end)].copy()
        development = run_walk_forward(
            features=development_features,
            labels=development_labels,
            model_specs=model_specs,
            target=research_config.target,
            config=walk_forward_config,
            max_horizon=max_horizon,
            transform_config=feature_config.get("fitted_transform"),
        )
        candidate_name, development_summary = _select_candidate(
            development.metrics,
            selection_metric=research_config.selection_metric,
        )
        selected_spec = next(spec for spec in model_specs if spec["name"] == candidate_name)
        ledger = _trial_ledger(model_specs, development_summary)
        development_economics = _development_economic_summary(
            panel=panel,
            predictions=development.predictions,
            features=features,
            benchmark_symbol=research_config.benchmark_symbol,
            backtest_config=backtest_config,
        )

        supervised, columns = supervised_frame(features, labels, research_config.target)
        train = supervised.loc[supervised["date"].le(development_end)].dropna(
            subset=[research_config.target]
        )
        holdout = supervised.loc[supervised["date"].ge(holdout_start)].dropna(
            subset=[research_config.target]
        )
        if train.empty or holdout.empty:
            raise ValueError("pre-registered final holdout has no eligible train/test rows")
        model = create_model(candidate_name, **selected_spec.get("params", {}))
        transform_spec = FittedTransformSpec.from_config(feature_config.get("fitted_transform"))
        final_transform_state: dict[str, Any] | None = None
        if transform_spec.enabled and not getattr(model, "requires_raw_features", False):
            transformer = FittedFeatureTransformer(transform_spec)
            transformed_train = transformer.fit_transform(train[columns], train["date"])
            transformed_holdout = transformer.transform(holdout[columns])
            train_matrix = _transformed_model_matrix(transformed_train, train, model)
            holdout_matrix = _transformed_model_matrix(transformed_holdout, holdout, model)
            if transformer.state_ is None:  # pragma: no cover - fit_transform guarantees state
                raise RuntimeError("final-holdout transformer did not publish fitted state")
            final_transform_state = asdict(transformer.state_)
        else:
            train_matrix = _model_matrix(train, columns, model)
            holdout_matrix = _model_matrix(holdout, columns, model)
        model.fit(train_matrix, train[research_config.target].astype(float))
        predictions = holdout[["date", "symbol", research_config.target]].rename(
            columns={research_config.target: "target"}
        )
        predictions["prediction"] = model.predict(holdout_matrix)
        predictions["model"] = candidate_name
        predictions["window_id"] = "final_holdout"

        primary = _backtest(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=research_config.benchmark_symbol,
            backtest_config=backtest_config,
        )
        pbo = _pbo(development.predictions, research_config.seed)
        scenario_summaries, scenarios_passed = _stress_scenarios(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=research_config.benchmark_symbol,
            backtest_config=backtest_config,
            maximum_drawdown=readiness_thresholds.maximum_drawdown,
            seed=research_config.seed,
        )
        borrow_sensitivity = _native_borrow_financing_summary(
            primary,
            dict(backtest_config.get("borrow_financing", {})),
        )
        scenario_summaries.append(borrow_sensitivity)
        missing_price_halt = _missing_price_halts(
            panel=panel,
            predictions=predictions,
            features=features,
            benchmark_symbol=research_config.benchmark_symbol,
            backtest_config=backtest_config,
        )
        capacity_config, minimum_fill_ratio = _capacity_config(backtest_config)
        capacity = estimate_capacity(primary.fills, capacity_config)
        capacity_passed = bool(capacity.curve["fill_ratio"].min() >= minimum_fill_ratio)
        primary_summary = performance_summary(primary.equity_curve)
        gross_summary = _gross_performance(primary.equity_curve)
        concentration = exposure_summary(primary.weights)
        if dataset.source_panel.empty:
            paper_anchor = pd.Timestamp(panel["date"].max()).tz_localize(UTC)
        else:
            paper_anchor = pd.Timestamp(dataset.source_panel["available_at"].max())
            if paper_anchor.tzinfo is None:
                raise ValueError("paper-control audit requires timezone-aware availability")
            paper_anchor = paper_anchor.tz_convert(UTC)
        paper_controls = audit_offline_paper_controls(
            decision_time=paper_anchor.to_pydatetime(),
            maximum_notional=float(backtest_config.get("initial_capital", 1_000_000.0)),
        )
        placebo_summary = next(
            item for item in scenario_summaries if item["scenario"] == "permuted_signal_placebo"
        )
        placebo_passed = bool(
            np.isfinite(primary_summary["annual_return"])
            and np.isfinite(placebo_summary["annual_return"])
            and primary_summary["annual_return"] >= placebo_summary["annual_return"]
        )
        dossier = assess_paper_readiness(
            equity_curve=primary.equity_curve,
            benchmark_returns=primary.equity_curve["benchmark_return"],
            n_trials=len(model_specs),
            probability_of_backtest_overfitting=float(pbo.get("pbo", np.nan)),
            point_in_time_limits=dataset.manifest["point_in_time_limits"],
            accounting_reconciled=True,
            stress_scenarios_passed=scenarios_passed and placebo_passed,
            thresholds=readiness_thresholds,
            additional_gates={
                "capacity_liquidity": capacity_passed,
                "missing_price_halt": missing_price_halt,
                "paper_controls": bool(paper_controls["all_controls_passed"]),
            },
        )
        dossier["metrics"]["gross_annual_return"] = gross_summary["annual_return"]
        dossier["metrics"]["gross_total_return"] = gross_summary["total_return"]
        dossier["candidate_model"] = candidate_name
        dossier["bundle_id"] = dataset.bundle_id
        dossier["run_id"] = run_id
        dossier["development_end"] = str(development_end.date())
        dossier["holdout_start"] = research_config.holdout_start
        dossier["overfitting"] = pbo
        dossier["scenarios"] = scenario_summaries
        dossier["placebo_outperformed"] = placebo_passed
        dossier["concentration"] = concentration
        dossier["paper_controls"] = paper_controls
        dossier["uncertainty"] = _bootstrap_uncertainty(
            primary.equity_curve["return"],
            seed=research_config.seed,
        )
        dossier["drawdown_diagnostics"] = _drawdown_diagnostics(primary.equity_curve)
        dossier["year_stability"] = _year_stability(primary.equity_curve)
        regime = (
            features[["date", "high_vol_regime"]]
            .drop_duplicates("date")
            .set_index("date")["high_vol_regime"]
        )
        dossier["regime_stability"] = regime_performance(
            primary.equity_curve,
            regime,
            regime_name="high_vol_regime",
        ).to_dict(orient="records")
        dossier["limitations"] = [
            "Daily bars do not establish intraday queue position or institutional execution quality.",
            "Borrow availability and locate failures are not modeled; configured borrow and "
            "financing are native event-ledger sensitivities, not executable terms.",
            "Point-in-time completeness is limited to the producer's explicit declarations.",
            "A readiness decision is not evidence that an edge will persist or be profitable.",
        ]

        development_summary.to_csv(staging / "development_model_selection.csv", index=False)
        development.metrics.to_csv(staging / "development_windows.csv", index=False)
        development.predictions.to_csv(staging / "development_predictions.csv", index=False)
        development_economics.to_csv(staging / "development_economic_metrics.csv", index=False)
        if not development.transformations.empty:
            development.transformations.to_csv(
                staging / "development_fitted_transformations.csv", index=False
            )
        if final_transform_state is not None:
            _write_json(staging / "final_holdout_fitted_transformation.json", final_transform_state)
        predictions.to_csv(staging / "final_holdout_predictions.csv", index=False)
        primary.equity_curve.to_csv(staging / "final_holdout_equity.csv", index=False)
        primary.orders.to_csv(staging / "orders.csv", index=False)
        primary.fills.to_csv(staging / "fills.csv", index=False)
        primary.pnl_attribution.to_csv(staging / "pnl_attribution.csv", index=False)
        primary.events.to_csv(staging / "execution_events.csv", index=False)
        primary.accounting.to_csv(staging / "accounting.csv", index=False)
        primary.friction_model_manifest.to_csv(staging / "friction_model_manifest.csv", index=False)
        primary.friction_attribution.to_csv(staging / "friction_attribution.csv", index=False)
        primary.latency_schedule.to_csv(staging / "latency_schedule.csv", index=False)
        capacity.curve.to_csv(staging / "capacity_curve.csv", index=False)
        capacity.scenario_trades.to_csv(staging / "capacity_scenario_trades.csv", index=False)
        _write_json(staging / "capacity_diagnostics.json", asdict(capacity.diagnostics))
        _write_json(staging / "paper_control_evidence.json", paper_controls)
        (staging / "trial_ledger.jsonl").write_text(
            "".join(_canonical_json(record).decode("utf-8") + "\n" for record in ledger),
            encoding="utf-8",
        )
        _write_json(staging / "dossier.json", dossier)
        _write_markdown_dossier(staging / "dossier.md", dossier)

        artifacts = inventory_artifacts(staging)
        finished_at = now().astimezone(UTC).isoformat().replace("+00:00", "Z")
        experiment_manifest = ExperimentManifest.build(
            code=git_context,
            dataset=dataset_record,
            universe=universe,
            date_range=date_range,
            features=feature_config,
            label=label_record,
            models=model_specs,
            validation=validation_record,
            transaction_costs=backtest_config,
            root_seed=research_config.seed,
            environment=runtime_environment,
            invocation=invocation_record,
            execution={"started_at": started_at, "finished_at": finished_at},
            artifacts=artifacts,
        )
        if experiment_manifest.experiment_id != run_id:
            raise RuntimeError("experiment identity changed while publishing evidence")
        _write_json(
            staging / "run_manifest.json",
            {
                "run_manifest_version": "2.0.0",
                "experiment": experiment_manifest.to_dict(),
                "result": {
                    "run_id": run_id,
                    "candidate_model": candidate_name,
                    "development_end": str(development_end.date()),
                    "holdout_start": research_config.holdout_start,
                    "trial_ledger_head": ledger[-1]["record_hash"],
                },
            },
        )
        staging.replace(destination)
        return GovernedResearchResult(
            run_id=run_id,
            run_dir=destination,
            candidate_model=candidate_name,
            dossier=dossier,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
