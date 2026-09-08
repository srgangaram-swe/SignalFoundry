"""Governed predictive and trading metrics with time-series uncertainty.

This module is the canonical metric boundary for research comparisons.  Every
published scalar is paired with a contract that declares units, annualization,
benchmark semantics, missing-data behavior, minimum sample size, aggregation,
and invalid-state behavior.  Ordered observations are resampled only through
contiguous moving blocks; no IID bootstrap is exposed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from alphaforge.evaluation.calibration import brier_score, expected_calibration_error
from alphaforge.evaluation.uncertainty import BlockBootstrapConfig

_CALENDAR_DAYS_PER_YEAR = 365.2425
_MAX_METRIC_ROWS = 10_000_000

MetricStatus = Literal["ok", "undefined"]


class MetricError(ValueError):
    """Raised when a metric input or predeclared policy is invalid."""


@dataclass(frozen=True)
class MetricContract:
    """Public interpretation and failure contract for one metric."""

    name: str
    unit: str
    annualization: str
    benchmark: str
    missingness: str
    minimum_samples: int
    aggregation: str
    invalid_state: str

    def __post_init__(self) -> None:
        for name in (
            "name",
            "unit",
            "annualization",
            "benchmark",
            "missingness",
            "aggregation",
            "invalid_state",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MetricError(f"metric contract {name} must be a non-empty string")
        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or self.minimum_samples < 2
        ):
            raise MetricError("metric contract minimum_samples must be an integer >= 2")


@dataclass(frozen=True)
class TimeSeriesDistribution:
    """Moving-block bootstrap evidence for one scalar statistic."""

    estimate: float
    lower: float
    upper: float
    variance: float
    standard_error: float
    n_observations: int
    n_resamples: int
    block_length: int
    confidence_level: float
    seed: int
    circular: bool
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        scalar_values = (
            self.estimate,
            self.lower,
            self.upper,
            self.variance,
            self.standard_error,
            self.confidence_level,
        )
        if not np.isfinite(scalar_values).all():
            raise MetricError("time-series distribution scalars must be finite")
        if self.lower > self.upper:
            raise MetricError("time-series distribution lower must not exceed upper")
        if self.variance < 0.0 or self.standard_error < 0.0:
            raise MetricError("time-series distribution dispersion must be non-negative")
        if not np.isclose(self.standard_error**2, self.variance, rtol=1e-10, atol=1e-15):
            raise MetricError("time-series distribution variance and standard error disagree")
        for name, value, minimum in (
            ("n_observations", self.n_observations, 2),
            ("n_resamples", self.n_resamples, 100),
            ("block_length", self.block_length, 2),
            ("seed", self.seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise MetricError(
                    f"time-series distribution {name} must be an integer >= {minimum}"
                )
        if self.block_length > self.n_observations:
            raise MetricError("time-series distribution block length exceeds observations")
        if not 0.0 < self.confidence_level < 1.0:
            raise MetricError("time-series distribution confidence level must be in (0, 1)")
        if not isinstance(self.circular, bool):
            raise TypeError("time-series distribution circular must be a boolean")
        if not self.assumptions or any(
            not isinstance(value, str) or not value.strip() for value in self.assumptions
        ):
            raise MetricError("time-series distribution assumptions must be non-empty strings")


@dataclass(frozen=True)
class MetricEstimate:
    """One contract-bound point estimate and optional uncertainty."""

    contract: MetricContract
    value: float | None
    status: MetricStatus
    n_observations: int
    uncertainty: TimeSeriesDistribution | None
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract, MetricContract):
            raise TypeError("metric estimate contract must be a MetricContract")
        if (
            isinstance(self.n_observations, bool)
            or not isinstance(self.n_observations, int)
            or self.n_observations < 0
        ):
            raise MetricError("metric estimate n_observations must be a non-negative integer")
        if self.status == "ok":
            if self.value is None or not np.isfinite(self.value):
                raise MetricError("defined metric estimate value must be finite")
            if not isinstance(self.uncertainty, TimeSeriesDistribution):
                raise MetricError("defined metric estimate requires time-series uncertainty")
            if self.note is not None:
                raise MetricError("defined metric estimate must not carry an undefined-state note")
        elif self.status == "undefined":
            if self.value is not None or self.uncertainty is not None:
                raise MetricError("undefined metric estimate cannot carry numeric evidence")
            if not isinstance(self.note, str) or not self.note.strip():
                raise MetricError("undefined metric estimate requires an explanatory note")
        else:
            raise MetricError("metric estimate status must be 'ok' or 'undefined'")

    @property
    def uncertainty_variance(self) -> float | None:
        """Return bootstrap variance without forcing callers to square an error."""

        return None if self.uncertainty is None else self.uncertainty.variance


@dataclass(frozen=True)
class MetricSuiteConfig:
    """Predeclared metric and temporal-resampling policy."""

    bootstrap: BlockBootstrapConfig
    minimum_prediction_samples: int = 20
    minimum_trading_periods: int = 20
    reliability_bins: int = 10
    benchmark_name: str = "declared benchmark"

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap, BlockBootstrapConfig):
            raise TypeError("bootstrap must be a BlockBootstrapConfig")
        for name, value in (
            ("minimum_prediction_samples", self.minimum_prediction_samples),
            ("minimum_trading_periods", self.minimum_trading_periods),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 4 <= value <= _MAX_METRIC_ROWS:
                raise MetricError(f"{name} must be in [4, {_MAX_METRIC_ROWS}]")
        if (
            isinstance(self.reliability_bins, bool)
            or not isinstance(self.reliability_bins, int)
            or not 2 <= self.reliability_bins <= 100
        ):
            raise MetricError("reliability_bins must be an integer in [2, 100]")
        if not isinstance(self.benchmark_name, str) or not self.benchmark_name.strip():
            raise MetricError("benchmark_name must be a non-empty string")
        if self.bootstrap.block_length > min(
            self.minimum_prediction_samples, self.minimum_trading_periods
        ):
            raise MetricError("bootstrap block_length cannot exceed either minimum sample setting")


@dataclass(frozen=True)
class MetricReport:
    """Immutable unified metric result."""

    estimates: tuple[MetricEstimate, ...]
    prediction_start: str
    prediction_end: str
    trading_start: str
    trading_end: str
    effective_periods_per_year: float
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.estimates or any(
            not isinstance(estimate, MetricEstimate) for estimate in self.estimates
        ):
            raise MetricError("metric report estimates must be non-empty MetricEstimate records")
        names = [estimate.contract.name for estimate in self.estimates]
        if len(names) != len(set(names)):
            raise MetricError("metric report contract names must be unique")
        try:
            prediction_start = pd.Timestamp(self.prediction_start)
            prediction_end = pd.Timestamp(self.prediction_end)
            trading_start = pd.Timestamp(self.trading_start)
            trading_end = pd.Timestamp(self.trading_end)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MetricError("metric report periods must contain valid timestamps") from exc
        if (
            prediction_start is pd.NaT
            or prediction_end is pd.NaT
            or trading_start is pd.NaT
            or trading_end is pd.NaT
            or prediction_start > prediction_end
            or trading_start > trading_end
        ):
            raise MetricError("metric report periods violate temporal invariants")
        if not np.isfinite(self.effective_periods_per_year) or self.effective_periods_per_year <= 0:
            raise MetricError("effective_periods_per_year must be finite and positive")
        if not self.assumptions or any(
            not isinstance(value, str) or not value.strip() for value in self.assumptions
        ):
            raise MetricError("metric report assumptions must be non-empty strings")

    def by_name(self) -> dict[str, MetricEstimate]:
        """Index the immutable estimates by their unique contract name."""

        return {estimate.contract.name: estimate for estimate in self.estimates}


def _contract(
    name: str,
    unit: str,
    minimum_samples: int,
    aggregation: str,
    *,
    annualization: str = "none",
    benchmark: str = "none",
    invalid_state: str = "raise MetricError",
) -> MetricContract:
    return MetricContract(
        name=name,
        unit=unit,
        annualization=annualization,
        benchmark=benchmark,
        missingness="reject non-finite or missing required observations",
        minimum_samples=minimum_samples,
        aggregation=aggregation,
        invalid_state=invalid_state,
    )


def metric_contracts(config: MetricSuiteConfig) -> tuple[MetricContract, ...]:
    """Return the complete predeclared registry for a metric-suite run."""

    prediction_n = config.minimum_prediction_samples
    trading_n = config.minimum_trading_periods
    return (
        _contract("mse", "squared decimal return", prediction_n, "mean squared error"),
        _contract("mae", "decimal return", prediction_n, "mean absolute error"),
        _contract(
            "directional_accuracy",
            "proportion",
            prediction_n,
            "mean sign agreement; zero is non-positive",
        ),
        _contract(
            "pearson_ic",
            "correlation",
            2,
            "mean of valid same-date cross-sectional Pearson correlations",
            invalid_state="undefined when either cross-section is constant",
        ),
        _contract(
            "rank_ic",
            "correlation",
            2,
            "mean of valid same-date Spearman correlations with average ranks",
            invalid_state="undefined when either ranked cross-section is constant",
        ),
        _contract("brier_score", "squared probability error", prediction_n, "mean"),
        _contract(
            "expected_calibration_error",
            "absolute probability gap",
            prediction_n,
            f"count-weighted mean over {config.reliability_bins} fixed bins",
        ),
        _contract("total_return", "decimal return", trading_n, "geometric compound"),
        _contract(
            "annualized_return",
            "decimal return per year",
            trading_n,
            "elapsed-calendar-time geometric compound",
            annualization="365.2425 calendar days divided by observed span",
        ),
        _contract(
            "annualized_volatility",
            "decimal return per square-root year",
            trading_n,
            "sample standard deviation",
            annualization="square root of observed periods per elapsed calendar year",
        ),
        _contract(
            "sharpe",
            "ratio",
            trading_n,
            "arithmetic mean divided by sample volatility",
            annualization="square root of observed periods per elapsed calendar year",
            invalid_state="undefined for zero return variance",
        ),
        _contract(
            "max_drawdown",
            "decimal loss from prior peak",
            trading_n,
            "minimum compounded-wealth drawdown",
        ),
        _contract("hit_rate", "proportion", trading_n, "mean(return > 0)"),
        _contract(
            "annualized_excess_return",
            "decimal return per year",
            trading_n,
            "elapsed-time geometric strategy return minus benchmark return",
            annualization="365.2425 calendar days divided by observed span",
            benchmark=config.benchmark_name,
        ),
        _contract(
            "tracking_error",
            "decimal return per square-root year",
            trading_n,
            "sample standard deviation of strategy minus benchmark return",
            annualization="square root of observed periods per elapsed calendar year",
            benchmark=config.benchmark_name,
        ),
        _contract("average_turnover", "gross capital fraction per period", trading_n, "mean"),
        _contract("average_gross_exposure", "capital fraction", trading_n, "mean"),
        _contract("average_net_exposure", "capital fraction", trading_n, "mean"),
        _contract("total_transaction_cost", "accounting currency", trading_n, "sum"),
        _contract(
            "cost_bps_per_traded_notional",
            "basis points",
            trading_n,
            "10000 * total cost / gross traded notional",
            invalid_state="undefined when gross traded notional is zero",
        ),
        _contract("capacity_fill_ratio", "proportion", trading_n, "mean desired-notional fill"),
        _contract(
            "capacity_constrained_fraction",
            "proportion",
            trading_n,
            "mean fraction of desired notional constrained by capacity",
        ),
    )


def _frame_dates(
    frame: pd.DataFrame,
    *,
    allow_duplicates: bool,
    name: str,
) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not 1 <= len(frame) <= _MAX_METRIC_ROWS:
        raise MetricError(f"{name} must contain between 1 and {_MAX_METRIC_ROWS} rows")
    if "date" not in frame:
        raise MetricError(f"{name} is missing required column 'date'")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{name}.date must contain valid timestamps") from exc
    if dates.hasnans or not dates.is_monotonic_increasing:
        raise MetricError(f"{name}.date must be non-missing and monotonically non-decreasing")
    if not allow_duplicates and dates.has_duplicates:
        raise MetricError(f"{name}.date must be unique")
    return dates


def _finite_column(frame: pd.DataFrame, column: str, name: str) -> np.ndarray:
    if column not in frame:
        raise MetricError(f"{name} is missing required column {column!r}")
    try:
        values = frame[column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{name}.{column} must be numeric") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise MetricError(f"{name}.{column} must contain only finite scalar values")
    return values


def _bootstrap_indices(
    n_observations: int,
    config: BlockBootstrapConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.block_length > n_observations:
        raise MetricError("bootstrap block_length must not exceed the statistic sample count")
    n_blocks = int(np.ceil(n_observations / config.block_length))
    max_start = n_observations if config.circular else n_observations - config.block_length + 1
    starts = rng.integers(0, max_start, size=n_blocks)
    offsets = np.arange(config.block_length)
    indices = (starts[:, None] + offsets[None, :]).reshape(-1)
    if config.circular:
        indices %= n_observations
    return indices[:n_observations]


def _distribution(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    config: BlockBootstrapConfig,
) -> TimeSeriesDistribution:
    point = float(statistic(values))
    if not np.isfinite(point):
        raise MetricError("cannot resample an undefined or non-finite statistic")
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.n_resamples, dtype=float)
    for sample_index in range(config.n_resamples):
        indices = _bootstrap_indices(len(values), config, rng)
        samples[sample_index] = statistic(values[indices])
    return _distribution_result(point, samples, len(values), config)


def _distribution_result(
    point: float,
    samples: np.ndarray,
    n_observations: int,
    config: BlockBootstrapConfig,
) -> TimeSeriesDistribution:
    if not np.isfinite(samples).all():
        raise MetricError("bootstrap statistic produced a non-finite resample")
    tail = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    variance = float(np.var(samples, ddof=1))
    return TimeSeriesDistribution(
        estimate=point,
        lower=float(lower),
        upper=float(upper),
        variance=variance,
        standard_error=float(np.sqrt(variance)),
        n_observations=n_observations,
        n_resamples=config.n_resamples,
        block_length=config.block_length,
        confidence_level=config.confidence_level,
        seed=config.seed,
        circular=config.circular,
        assumptions=(
            "ordered observations preserve their original temporal order",
            "contiguous blocks represent material short-range dependence",
            "the evaluated process is sufficiently stable over the reported period",
            "the interval is descriptive evidence, not a future-performance guarantee",
        ),
    )


def _panel_distribution(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    statistic: Callable[[np.ndarray], float],
    config: BlockBootstrapConfig,
) -> TimeSeriesDistribution:
    """Resample contiguous dates while retaining each complete cross-section."""

    groups = [
        group.loc[:, columns].to_numpy(dtype=float)
        for _, group in panel.groupby("date", sort=False)
    ]
    if len(groups) < config.block_length:
        raise MetricError("prediction metrics require at least one bootstrap block of dates")
    point = float(statistic(np.concatenate(groups, axis=0)))
    if not np.isfinite(point):
        raise MetricError("cannot resample an undefined or non-finite panel statistic")
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.n_resamples, dtype=float)
    for sample_index in range(config.n_resamples):
        indices = _bootstrap_indices(len(groups), config, rng)
        resampled = np.concatenate([groups[index] for index in indices], axis=0)
        samples[sample_index] = statistic(resampled)
    return _distribution_result(point, samples, len(groups), config)


def _panel_estimate(
    contract: MetricContract,
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    statistic: Callable[[np.ndarray], float],
    bootstrap: BlockBootstrapConfig,
) -> MetricEstimate:
    if len(panel) < contract.minimum_samples:
        raise MetricError(
            f"{contract.name} requires at least {contract.minimum_samples} observations"
        )
    uncertainty = _panel_distribution(panel, columns, statistic, bootstrap)
    return MetricEstimate(
        contract=contract,
        value=uncertainty.estimate,
        status="ok",
        n_observations=len(panel),
        uncertainty=uncertainty,
    )


def _estimate(
    contract: MetricContract,
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    bootstrap: BlockBootstrapConfig,
) -> MetricEstimate:
    if len(values) < contract.minimum_samples:
        raise MetricError(
            f"{contract.name} requires at least {contract.minimum_samples} observations"
        )
    uncertainty = _distribution(values, statistic, bootstrap)
    return MetricEstimate(
        contract=contract,
        value=uncertainty.estimate,
        status="ok",
        n_observations=len(values),
        uncertainty=uncertainty,
    )


def _undefined(contract: MetricContract, n_observations: int, note: str) -> MetricEstimate:
    return MetricEstimate(
        contract=contract,
        value=None,
        status="undefined",
        n_observations=n_observations,
        uncertainty=None,
        note=note,
    )


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _ic_series(prediction_panel: pd.DataFrame, *, ranked: bool) -> np.ndarray:
    values: list[float] = []
    for _, group in prediction_panel.groupby("date", sort=False):
        target = group["target"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        if ranked:
            target = pd.Series(target).rank(method="average").to_numpy()
            prediction = pd.Series(prediction).rank(method="average").to_numpy()
        correlation = _correlation(target, prediction)
        if correlation is not None:
            values.append(correlation)
    return np.asarray(values, dtype=float)


def _elapsed_periods_per_year(dates: pd.DatetimeIndex) -> float:
    elapsed_days = (dates[-1] - dates[0]).total_seconds() / 86_400.0
    if elapsed_days <= 0:
        raise MetricError("trading dates must span positive elapsed calendar time")
    periods_per_year = (len(dates) - 1) * _CALENDAR_DAYS_PER_YEAR / elapsed_days
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise MetricError("could not infer a finite positive observation frequency")
    return float(periods_per_year)


def _compound(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def _annualized_return(returns: np.ndarray, periods_per_year: float) -> float:
    total = _compound(returns)
    if total <= -1.0:
        return -1.0
    return float((1.0 + total) ** (periods_per_year / (len(returns) - 1)) - 1.0)


def _annualized_volatility(returns: np.ndarray, periods_per_year: float) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    volatility = float(np.std(returns, ddof=1))
    if np.ptp(returns) == 0.0:
        return float("nan")
    return float(np.mean(returns) / volatility * np.sqrt(periods_per_year))


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    wealth_with_origin = np.concatenate(([1.0], wealth))
    peaks = np.maximum.accumulate(wealth_with_origin)
    return float(np.min(wealth_with_origin / peaks - 1.0))


def _prediction_estimates(
    panel: pd.DataFrame,
    config: MetricSuiteConfig,
    contracts: dict[str, MetricContract],
) -> list[MetricEstimate]:
    target = _finite_column(panel, "target", "prediction_panel")
    _finite_column(panel, "prediction", "prediction_panel")
    if len(target) < config.minimum_prediction_samples:
        raise MetricError(
            f"prediction_panel requires at least {config.minimum_prediction_samples} rows"
        )
    bootstrap = config.bootstrap
    estimates = [
        _panel_estimate(
            contracts["mse"],
            panel,
            ("target", "prediction"),
            lambda rows: float(np.mean((rows[:, 1] - rows[:, 0]) ** 2)),
            bootstrap,
        ),
        _panel_estimate(
            contracts["mae"],
            panel,
            ("target", "prediction"),
            lambda rows: float(np.mean(np.abs(rows[:, 1] - rows[:, 0]))),
            bootstrap,
        ),
        _panel_estimate(
            contracts["directional_accuracy"],
            panel,
            ("target", "prediction"),
            lambda rows: float(np.mean((rows[:, 0] > 0) == (rows[:, 1] > 0))),
            bootstrap,
        ),
    ]
    for name, ranked in (("pearson_ic", False), ("rank_ic", True)):
        ic_values = _ic_series(panel, ranked=ranked)
        contract = contracts[name]
        if len(ic_values) < max(contract.minimum_samples, bootstrap.block_length):
            estimates.append(
                _undefined(
                    contract,
                    len(ic_values),
                    "insufficient non-constant date cross-sections for block uncertainty",
                )
            )
        else:
            estimates.append(_estimate(contract, ic_values, np.mean, bootstrap))

    has_probability = "probability" in panel
    has_outcome = "outcome" in panel
    if has_probability != has_outcome:
        raise MetricError("probability and outcome must be supplied together")
    if not has_probability:
        return estimates
    probability = _finite_column(panel, "probability", "prediction_panel")
    outcome = _finite_column(panel, "outcome", "prediction_panel")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise MetricError("prediction_panel.probability must be in [0, 1]")
    if ((outcome != 0.0) & (outcome != 1.0)).any():
        raise MetricError("prediction_panel.outcome must be binary")
    squared_probability_error = (probability - outcome) ** 2
    brier_point = brier_score(probability, outcome)
    if not np.isclose(brier_point, np.mean(squared_probability_error)):
        raise MetricError("Brier statistic replay mismatch")
    estimates.append(
        _panel_estimate(
            contracts["brier_score"],
            panel,
            ("probability", "outcome"),
            lambda rows: float(np.mean((rows[:, 0] - rows[:, 1]) ** 2)),
            bootstrap,
        )
    )
    ece_point = expected_calibration_error(probability, outcome, n_bins=config.reliability_bins)
    ece_distribution = _panel_distribution(
        panel,
        ("probability", "outcome"),
        lambda rows: expected_calibration_error(
            rows[:, 0], rows[:, 1], n_bins=config.reliability_bins
        ),
        bootstrap,
    )
    if not np.isclose(ece_point, ece_distribution.estimate):
        raise MetricError("calibration statistic replay mismatch")
    estimates.append(
        MetricEstimate(
            contract=contracts["expected_calibration_error"],
            value=ece_point,
            status="ok",
            n_observations=len(probability),
            uncertainty=ece_distribution,
        )
    )
    return estimates


def _cost_estimates(
    frame: pd.DataFrame,
    bootstrap: BlockBootstrapConfig,
    contracts: dict[str, MetricContract],
) -> list[MetricEstimate]:
    if "transaction_cost" not in frame:
        return []
    costs = _finite_column(frame, "transaction_cost", "trading_frame")
    if (costs < 0).any():
        raise MetricError("trading_frame.transaction_cost must be non-negative")
    estimates = [_estimate(contracts["total_transaction_cost"], costs, np.sum, bootstrap)]
    if "traded_notional" not in frame:
        return estimates
    notionals = np.abs(_finite_column(frame, "traded_notional", "trading_frame"))
    if ((notionals == 0.0) & (costs != 0.0)).any():
        raise MetricError("transaction cost must be zero when gross traded notional is zero")
    if float(np.sum(notionals)) == 0.0:
        estimates.append(
            _undefined(
                contracts["cost_bps_per_traded_notional"],
                len(notionals),
                "gross traded notional is zero",
            )
        )
        return estimates
    active_trade = notionals > 0.0
    paired = np.column_stack((costs[active_trade], notionals[active_trade]))
    if len(paired) < bootstrap.block_length:
        raise MetricError("cost bps requires at least one bootstrap block of active trades")
    estimates.append(
        _estimate(
            contracts["cost_bps_per_traded_notional"],
            paired,
            lambda rows: float(10_000.0 * np.sum(rows[:, 0]) / np.sum(rows[:, 1])),
            bootstrap,
        )
    )
    return estimates


def _trading_estimates(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: MetricSuiteConfig,
    contracts: dict[str, MetricContract],
) -> tuple[list[MetricEstimate], float]:
    returns = _finite_column(frame, "return", "trading_frame")
    if len(returns) < config.minimum_trading_periods:
        raise MetricError(
            f"trading_frame requires at least {config.minimum_trading_periods} periods"
        )
    if (returns < -1.0).any():
        raise MetricError("trading_frame.return must be greater than or equal to -1")
    periods_per_year = _elapsed_periods_per_year(dates)
    bootstrap = config.bootstrap

    def annual_return_stat(sample: np.ndarray) -> float:
        return _annualized_return(sample, periods_per_year)

    def volatility_stat(sample: np.ndarray) -> float:
        return _annualized_volatility(sample, periods_per_year)

    estimates = [
        _estimate(contracts["total_return"], returns, _compound, bootstrap),
        _estimate(contracts["annualized_return"], returns, annual_return_stat, bootstrap),
        _estimate(contracts["annualized_volatility"], returns, volatility_stat, bootstrap),
    ]
    if np.ptp(returns) == 0.0:
        estimates.append(_undefined(contracts["sharpe"], len(returns), "return variance is zero"))
    else:
        estimates.append(
            _estimate(
                contracts["sharpe"],
                returns,
                lambda sample: _sharpe(sample, periods_per_year),
                bootstrap,
            )
        )
    estimates.extend(
        [
            _estimate(contracts["max_drawdown"], returns, _max_drawdown, bootstrap),
            _estimate(contracts["hit_rate"], (returns > 0).astype(float), np.mean, bootstrap),
        ]
    )
    if "benchmark_return" in frame:
        benchmark = _finite_column(frame, "benchmark_return", "trading_frame")
        if (benchmark < -1.0).any():
            raise MetricError("trading_frame.benchmark_return must be greater than or equal to -1")
        estimates.extend(
            [
                _estimate(
                    contracts["annualized_excess_return"],
                    np.column_stack((returns, benchmark)),
                    lambda rows: _annualized_return(rows[:, 0], periods_per_year)
                    - _annualized_return(rows[:, 1], periods_per_year),
                    bootstrap,
                ),
                _estimate(
                    contracts["tracking_error"],
                    returns - benchmark,
                    volatility_stat,
                    bootstrap,
                ),
            ]
        )
    optional_means = {
        "turnover": "average_turnover",
        "gross_exposure": "average_gross_exposure",
        "net_exposure": "average_net_exposure",
        "capacity_fill_ratio": "capacity_fill_ratio",
        "capacity_constrained_fraction": "capacity_constrained_fraction",
    }
    for column, name in optional_means.items():
        if column not in frame:
            continue
        values = _finite_column(frame, column, "trading_frame")
        if column in {"turnover", "gross_exposure"} and (values < 0).any():
            raise MetricError(f"trading_frame.{column} must be non-negative")
        if column.startswith("capacity_") and ((values < 0) | (values > 1)).any():
            raise MetricError(f"trading_frame.{column} must be in [0, 1]")
        estimates.append(_estimate(contracts[name], values, np.mean, bootstrap))
    estimates.extend(_cost_estimates(frame, bootstrap, contracts))
    return estimates, periods_per_year


def evaluate_metric_suite(
    prediction_panel: pd.DataFrame,
    trading_frame: pd.DataFrame,
    config: MetricSuiteConfig,
) -> MetricReport:
    """Evaluate one frozen prediction panel and chronological trading record.

    Required prediction columns are ``date``, ``target``, and ``prediction``.
    Supplying both ``probability`` and binary ``outcome`` adds calibration
    metrics.  Required trading columns are ``date`` and ``return``; optional
    governed columns are ``benchmark_return``, ``turnover``,
    ``gross_exposure``, ``net_exposure``, ``transaction_cost``,
    ``traded_notional``, ``capacity_fill_ratio``, and
    ``capacity_constrained_fraction``.
    """

    if not isinstance(config, MetricSuiteConfig):
        raise TypeError("config must be a MetricSuiteConfig")
    prediction_dates = _frame_dates(
        prediction_panel, allow_duplicates=True, name="prediction_panel"
    )
    trading_dates = _frame_dates(trading_frame, allow_duplicates=False, name="trading_frame")
    contracts = {contract.name: contract for contract in metric_contracts(config)}
    estimates = _prediction_estimates(prediction_panel, config, contracts)
    trading_estimates, periods_per_year = _trading_estimates(
        trading_frame, trading_dates, config, contracts
    )
    estimates.extend(trading_estimates)
    names = [estimate.contract.name for estimate in estimates]
    if len(names) != len(set(names)):
        raise MetricError("metric report contains duplicate contract names")
    return MetricReport(
        estimates=tuple(estimates),
        prediction_start=prediction_dates[0].isoformat(),
        prediction_end=prediction_dates[-1].isoformat(),
        trading_start=trading_dates[0].isoformat(),
        trading_end=trading_dates[-1].isoformat(),
        effective_periods_per_year=periods_per_year,
        assumptions=(
            "prediction and trading inputs were frozen before this evaluation",
            "all required values are finite; missing observations are rejected, not imputed",
            "annualization uses actual elapsed calendar time and observed sample count",
            "uncertainty uses contiguous moving blocks and is not IID",
            "capacity values are sensitivity diagnostics, not deployable-AUM guarantees",
            "no metric or interval establishes future profitability or live readiness",
        ),
    )
