"""Strict, typed configuration contracts for every supported YAML surface.

Configuration is untrusted input.  The public loaders in this module parse with
``yaml.safe_load``, reject unknown fields through Pydantic's ``extra='forbid'``
policy, validate cross-field invariants before any I/O or model execution, and
return normalized dictionaries for the existing domain APIs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
OpenUnitFloat = Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]


class ConfigValidationError(ValueError):
    """Raised when a configuration document is malformed or unsafe."""


class StrictConfig(BaseModel):
    """Base class that forbids coercion, mutation, and undeclared fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise ValueError("path must be a non-empty safe relative path")
    return value


def _ordered_unique(values: list[int], *, field_name: str) -> list[int]:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{field_name} must contain positive integers")
    if values != sorted(set(values)):
        raise ValueError(f"{field_name} must be strictly increasing and unique")
    return values


class SyntheticConfig(StrictConfig):
    n_symbols: PositiveInt = 20
    n_days: PositiveInt = 2500
    seed: NonNegativeInt = 42


class DataConfig(StrictConfig):
    source: Literal["yfinance", "csv", "synthetic", "signal_foundry"]
    symbols: list[str] = Field(default_factory=list)
    benchmark: str
    start: str
    end: str | None = None
    interval: Literal["1d"]
    cache_dir: str
    csv_dir: str
    quality_report: str
    synthetic: SyntheticConfig

    @field_validator("cache_dir", "csv_dir", "quality_report")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _safe_relative_path(value)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, values: list[str]) -> list[str]:
        if any(not value or value != value.strip() or not value.isascii() for value in values):
            raise ValueError("symbols must be non-empty, trimmed ASCII identifiers")
        if len(values) != len(set(values)):
            raise ValueError("symbols must be unique")
        return values

    @model_validator(mode="after")
    def validate_data_contract(self) -> DataConfig:
        if not self.benchmark or self.benchmark != self.benchmark.strip():
            raise ValueError("benchmark must be non-empty and trimmed")
        try:
            start = date.fromisoformat(self.start)
            end = None if self.end is None else date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError("start and end must be ISO-8601 calendar dates") from exc
        if end is not None and start >= end:
            raise ValueError("start must be earlier than end")
        if self.source in {"yfinance", "csv"} and not self.symbols:
            raise ValueError(f"{self.source} requires at least one symbol")
        if self.benchmark in self.symbols:
            raise ValueError("benchmark must not be duplicated in the tradable symbol list")
        return self


class MacdConfig(StrictConfig):
    fast: PositiveInt
    slow: PositiveInt
    signal: PositiveInt

    @model_validator(mode="after")
    def validate_windows(self) -> MacdConfig:
        if self.fast >= self.slow:
            raise ValueError("macd.fast must be smaller than macd.slow")
        return self


class FittedTransformConfig(StrictConfig):
    """Configuration for preprocessing learned exclusively from a training fold."""

    version: Literal["1.0.0"] = "1.0.0"
    enabled: bool = False
    imputation: Literal["median", "mean"] = "median"
    standardize: bool = True
    variance_threshold: NonNegativeFloat | None = None
    pca_components: PositiveInt | OpenUnitFloat | None = None
    pca_whiten: bool = False
    min_fit_rows: PositiveInt = 64

    @model_validator(mode="after")
    def validate_transform(self) -> FittedTransformConfig:
        if self.pca_whiten and self.pca_components is None:
            raise ValueError("pca_whiten requires pca_components")
        return self


class FeatureConfig(StrictConfig):
    registry_version: Literal["1.0.0"] = "1.0.0"
    cache_dir: str | None = None
    fitted_transform: FittedTransformConfig = Field(default_factory=FittedTransformConfig)
    return_lags: list[int]
    vol_windows: list[int]
    ma_windows: list[int]
    momentum_windows: list[int]
    rsi_window: PositiveInt
    macd: MacdConfig
    bollinger_window: PositiveInt
    mean_reversion_window: PositiveInt
    volume_window: PositiveInt
    beta_window: PositiveInt
    rolling_sharpe_window: PositiveInt
    drawdown_window: PositiveInt
    regime_vol_window: PositiveInt
    regime_trend_fast: PositiveInt
    regime_trend_slow: PositiveInt
    hmm_regime: bool
    hmm_refit_every: PositiveInt = 63
    hmm_min_train: PositiveInt = 252
    cross_sectional: bool
    output_dir: str | None = None

    @field_validator("return_lags", "vol_windows", "ma_windows", "momentum_windows")
    @classmethod
    def validate_window_lists(cls, values: list[int], info: Any) -> list[int]:
        return _ordered_unique(values, field_name=info.field_name)

    @field_validator("cache_dir", "output_dir")
    @classmethod
    def validate_output_dir(cls, value: str | None) -> str | None:
        return None if value is None else _safe_relative_path(value)

    @model_validator(mode="after")
    def validate_regime_windows(self) -> FeatureConfig:
        if self.regime_trend_fast >= self.regime_trend_slow:
            raise ValueError("regime_trend_fast must be smaller than regime_trend_slow")
        if self.hmm_regime and self.hmm_min_train < self.hmm_refit_every:
            raise ValueError("hmm_min_train must be at least hmm_refit_every")
        return self


class LabelSpecConfig(StrictConfig):
    """One versioned financial-label definition."""

    version: Literal["1.0.0"] = "1.0.0"
    name: str
    kind: Literal[
        "regression",
        "classification",
        "threshold",
        "quantile",
        "triple_barrier",
        "volatility_scaled",
        "meta_label",
    ]
    horizon: Annotated[int, Field(gt=0, le=2520)]
    timing: Literal["close_to_close"] = "close_to_close"
    price_field: Literal["close"] = "close"
    overlap_policy: Literal["allow", "non_overlapping"] = "allow"
    threshold: NonNegativeFloat | None = None
    quantiles: Annotated[int, Field(ge=2, le=20)] | None = None
    upper_barrier: PositiveFloat | None = None
    lower_barrier: PositiveFloat | None = None
    volatility_window: Annotated[int, Field(ge=2)] | None = None
    side_source: Literal["lagged_return", "column"] | None = None
    side_column: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if (
            not value
            or value != value.strip()
            or not value.isascii()
            or not all(character.isalnum() or character == "_" for character in value)
        ):
            raise ValueError(
                "label name must be an ASCII identifier containing letters, numbers, "
                "or underscores"
            )
        return value

    @model_validator(mode="after")
    def validate_parameters(self) -> LabelSpecConfig:
        supplied = {
            name
            for name, value in {
                "threshold": self.threshold,
                "quantiles": self.quantiles,
                "upper_barrier": self.upper_barrier,
                "lower_barrier": self.lower_barrier,
                "volatility_window": self.volatility_window,
                "side_source": self.side_source,
                "side_column": self.side_column,
            }.items()
            if value is not None
        }
        allowed = {
            "regression": set(),
            "classification": set(),
            "threshold": {"threshold"},
            "quantile": {"quantiles"},
            "triple_barrier": {"upper_barrier", "lower_barrier"},
            "volatility_scaled": {"volatility_window"},
            "meta_label": {"threshold", "side_source", "side_column"},
        }
        required = {
            "regression": set(),
            "classification": set(),
            "threshold": {"threshold"},
            "quantile": {"quantiles"},
            "triple_barrier": {"upper_barrier", "lower_barrier"},
            "volatility_scaled": {"volatility_window"},
            "meta_label": {"threshold", "side_source"},
        }
        unexpected = supplied - allowed[self.kind]
        missing = required[self.kind] - supplied
        if unexpected:
            raise ValueError(f"{self.kind} label has unsupported parameters: {sorted(unexpected)}")
        if missing:
            raise ValueError(f"{self.kind} label requires parameters: {sorted(missing)}")
        if self.side_source == "column":
            if (
                not self.side_column
                or self.side_column != self.side_column.strip()
                or not self.side_column.isascii()
            ):
                raise ValueError(
                    "side_source='column' requires a non-empty trimmed ASCII side_column"
                )
        elif self.side_column is not None:
            raise ValueError("side_column is only valid when side_source='column'")
        return self


class LabelDiagnosticsConfig(StrictConfig):
    """Bounded statistical-evidence settings for label diagnostics."""

    autocorrelation_lag: PositiveInt = 1
    temporal_periods: Annotated[int, Field(ge=2, le=12)] = 4
    sensitivity_scales: Annotated[list[PositiveFloat], Field(min_length=1, max_length=9)] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0]
    )

    @field_validator("sensitivity_scales")
    @classmethod
    def validate_scales(cls, values: list[float]) -> list[float]:
        if not values or values != sorted(set(values)):
            raise ValueError("sensitivity_scales must be strictly increasing and unique")
        if 1.0 not in values:
            raise ValueError("sensitivity_scales must include the 1.0 baseline")
        return values


class LabelsConfig(StrictConfig):
    """Strict YAML contract for governed financial-label materialization."""

    version: Literal["1.0.0"] = "1.0.0"
    benchmark_symbol: str
    missing_price_policy: Literal["raise"] = "raise"
    protected_boundaries: Annotated[list[str], Field(max_length=64)] = Field(default_factory=list)
    labels: Annotated[list[LabelSpecConfig], Field(min_length=1, max_length=64)]
    diagnostics: LabelDiagnosticsConfig = Field(default_factory=LabelDiagnosticsConfig)

    @field_validator("benchmark_symbol")
    @classmethod
    def validate_benchmark(cls, value: str) -> str:
        if not value or value != value.strip() or not value.isascii():
            raise ValueError("benchmark_symbol must be non-empty, trimmed ASCII")
        return value

    @field_validator("protected_boundaries")
    @classmethod
    def validate_boundaries(cls, values: list[str]) -> list[str]:
        try:
            parsed = [date.fromisoformat(value) for value in values]
        except ValueError as exc:
            raise ValueError("protected_boundaries must contain ISO-8601 calendar dates") from exc
        if parsed != sorted(set(parsed)):
            raise ValueError("protected_boundaries must be strictly increasing and unique")
        return values

    @model_validator(mode="after")
    def validate_label_contract(self) -> LabelsConfig:
        names = [label.name for label in self.labels]
        if not names:
            raise ValueError("labels must contain at least one definition")
        if len(names) != len(set(names)):
            raise ValueError("label names must be unique")
        return self


MODEL_PARAMETER_FIELDS: dict[str, frozenset[str]] = {
    "zero_baseline": frozenset(),
    "historical_mean": frozenset(),
    "momentum_baseline": frozenset({"feature", "scale"}),
    "linear": frozenset({"fit_intercept", "positive"}),
    "ridge": frozenset({"alpha", "fit_intercept", "max_iter", "tol"}),
    "lasso": frozenset({"alpha", "fit_intercept", "max_iter", "tol"}),
    "elastic_net": frozenset({"alpha", "l1_ratio", "fit_intercept", "max_iter", "tol"}),
    "huber": frozenset({"epsilon", "alpha", "max_iter", "tol"}),
    "random_forest": frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "n_jobs",
            "random_state",
        }
    ),
    "extra_trees": frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "n_jobs",
            "random_state",
        }
    ),
    "gradient_boosting": frozenset(
        {
            "backend",
            "learning_rate",
            "max_iter",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
            "random_state",
        }
    ),
    "lightgbm": frozenset(
        {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_samples_leaf",
            "l2_regularization",
            "n_jobs",
            "random_state",
        }
    ),
    "xgboost": frozenset(
        {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "l2_regularization",
            "n_jobs",
            "random_state",
        }
    ),
    "catboost": frozenset(
        {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_samples_leaf",
            "l2_regularization",
            "n_jobs",
            "random_state",
        }
    ),
    "small_mlp": frozenset(
        {
            "hidden_layer_sizes",
            "alpha",
            "learning_rate_init",
            "batch_size",
            "max_iter",
            "tol",
            "random_state",
        }
    ),
    "torch_mlp": frozenset(
        {"hidden_sizes", "dropout", "lr", "weight_decay", "epochs", "batch_size", "seed"}
    ),
    "torch_gru": frozenset(
        {"hidden_size", "num_layers", "dropout", "lr", "epochs", "batch_size", "seed"}
    ),
    "torch_tcn": frozenset(
        {"channels", "kernel_size", "dropout", "lr", "epochs", "batch_size", "seed"}
    ),
    "temporal_alpha": frozenset(
        {
            "seq_len",
            "hidden_size",
            "n_blocks",
            "dropout",
            "lr",
            "weight_decay",
            "max_epochs",
            "patience",
            "dates_per_batch",
            "ic_loss_weight",
            "aux_loss_weight",
            "val_fraction",
            "seed",
        }
    ),
    "sequence_cnn": frozenset(
        {
            "seq_len",
            "min_history",
            "hidden_size",
            "n_layers",
            "kernel_size",
            "n_heads",
            "dropout",
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "patience",
            "batch_size",
            "validation_fraction",
            "gradient_clip",
            "clip_z",
            "max_parameters",
            "max_windows",
            "max_tensor_bytes",
            "seed",
            "device",
        }
    ),
    "sequence_tcn": frozenset(),
    "sequence_lstm": frozenset(),
    "sequence_gru": frozenset(),
    "sequence_transformer": frozenset(),
    "ensemble": frozenset({"weighting", "members"}),
}

for _sequence_name in (
    "sequence_tcn",
    "sequence_lstm",
    "sequence_gru",
    "sequence_transformer",
):
    MODEL_PARAMETER_FIELDS[_sequence_name] = MODEL_PARAMETER_FIELDS["sequence_cnn"]


class ModelSpec(StrictConfig):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model_parameters(self) -> ModelSpec:
        allowed = MODEL_PARAMETER_FIELDS.get(self.name)
        if allowed is None:
            raise ValueError(f"unknown model {self.name!r}")
        unknown = set(self.params) - allowed
        if unknown:
            raise ValueError(f"unknown parameters for {self.name!r}: {sorted(unknown)}")
        if self.name == "ensemble":
            members = self.params.get("members")
            if not isinstance(members, list) or not members:
                raise ValueError("ensemble.params.members must be a non-empty list")
            for member in members:
                ModelSpec.model_validate(member)
        return self


class TemporalConfig(StrictConfig):
    seq_len: PositiveInt
    hidden_size: PositiveInt
    n_blocks: PositiveInt
    dropout: UnitFloat
    lr: PositiveFloat
    weight_decay: NonNegativeFloat
    max_epochs: PositiveInt
    patience: PositiveInt
    dates_per_batch: PositiveInt
    ic_loss_weight: NonNegativeFloat
    aux_loss_weight: NonNegativeFloat
    val_fraction: OpenUnitFloat
    seed: NonNegativeInt


class WalkForwardConfig(StrictConfig):
    scheme: Literal["expanding", "rolling"]
    min_train_days: PositiveInt
    test_days: PositiveInt
    step_days: PositiveInt
    embargo_days: NonNegativeInt
    max_windows: PositiveInt | None = None


class ModelsConfig(StrictConfig):
    target: str
    horizons: list[int]
    models: list[ModelSpec]
    temporal: TemporalConfig
    walk_forward: WalkForwardConfig
    seed: NonNegativeInt
    runs_dir: str

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, values: list[int]) -> list[int]:
        return _ordered_unique(values, field_name="horizons")

    @field_validator("runs_dir")
    @classmethod
    def validate_runs_dir(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def validate_model_contract(self) -> ModelsConfig:
        expected_targets = {f"fwd_ret_{horizon}" for horizon in self.horizons}
        if self.target not in expected_targets:
            raise ValueError("target must name one of the configured forward-return horizons")
        if not self.models:
            raise ValueError("models must contain at least one candidate")
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("top-level model names must be unique")
        if self.walk_forward.embargo_days < max(self.horizons):
            raise ValueError("walk_forward.embargo_days must cover the maximum label horizon")
        if self.temporal.seed != self.seed:
            raise ValueError("temporal.seed and root seed must match")
        return self


class StrategyParams(StrictConfig):
    quantile: OpenUnitFloat | None = None
    top_k: PositiveInt | None = None
    clip: PositiveFloat | None = None
    threshold: NonNegativeFloat | None = None
    allow_short: bool | None = None


class ExecutionConfig(StrictConfig):
    price_field: Literal["open"] = "open"
    adv_lookback: PositiveInt
    volatility_lookback: Annotated[int, Field(ge=2)]
    max_participation_rate: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] | None
    impact_coefficient: NonNegativeFloat
    impact_exponent: Annotated[float, Field(ge=0, le=4, allow_inf_nan=False)] = 0.5
    missing_price_policy: Literal["raise", "skip"]
    calibration_provenance: Annotated[str, Field(min_length=1, max_length=512)] = (
        "caller-supplied deterministic simulation assumption; not empirically calibrated"
    )


class CostConfig(StrictConfig):
    commission_bps: NonNegativeFloat
    half_spread_bps: NonNegativeFloat
    slippage_bps: NonNegativeFloat
    commission_per_share_usd: NonNegativeFloat = 0.0
    minimum_commission_usd: NonNegativeFloat = 0.0
    exchange_fee_bps: NonNegativeFloat = 0.0
    exchange_fee_per_share_usd: NonNegativeFloat = 0.0
    spread_slippage_multiplier: NonNegativeFloat = 0.0
    participation_slippage_bps: NonNegativeFloat = 0.0
    participation_slippage_exponent: Annotated[float, Field(ge=0, le=4, allow_inf_nan=False)] = 1.0
    volatility_slippage_bps_per_1pct: NonNegativeFloat = 0.0
    calibration_provenance: Annotated[str, Field(min_length=1, max_length=512)] = (
        "caller-supplied deterministic scenario assumption; not empirically calibrated"
    )


class LatencyConfig(StrictConfig):
    data_delay_sessions: Annotated[int, Field(ge=0, le=2520)] = 0
    feature_delay_sessions: Annotated[int, Field(ge=0, le=2520)] = 0
    inference_delay_sessions: Annotated[int, Field(ge=0, le=2520)] = 0
    submission_delay_sessions: Annotated[int, Field(ge=0, le=2520)] = 0
    fill_delay_sessions: Annotated[int, Field(ge=0, le=2520)] = 0
    calibration_provenance: Annotated[str, Field(min_length=1, max_length=512)] = (
        "predeclared deterministic simulation sensitivity; not calibrated from strategy "
        "test outcomes or presented as observed execution quality"
    )

    @model_validator(mode="after")
    def validate_total_delay(self) -> LatencyConfig:
        total = sum(
            (
                self.data_delay_sessions,
                self.feature_delay_sessions,
                self.inference_delay_sessions,
                self.submission_delay_sessions,
                self.fill_delay_sessions,
            )
        )
        if total >= 5040:
            raise ValueError("total logical latency must be less than 5040 sessions")
        return self


class PortfolioPolicyConfig(StrictConfig):
    max_weight: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    max_gross_exposure: PositiveFloat
    max_net_exposure: NonNegativeFloat = 1.0
    inverse_vol_scaling: bool
    turnover_cap: PositiveFloat | None

    @model_validator(mode="after")
    def validate_exposures(self) -> PortfolioPolicyConfig:
        if self.max_weight > self.max_gross_exposure:
            raise ValueError("max_weight cannot exceed max_gross_exposure")
        if self.max_net_exposure > self.max_gross_exposure:
            raise ValueError("max_net_exposure cannot exceed max_gross_exposure")
        return self


class RiskPolicyConfig(StrictConfig):
    vol_target: PositiveFloat | None = None
    vol_lookback: PositiveInt = 20
    max_leverage: PositiveFloat = 1.0
    drawdown_deleverage: OpenUnitFloat | None = None
    drawdown_cut: OpenUnitFloat = 0.5
    regime_filter: bool = False


class CapacityPolicyConfig(StrictConfig):
    enabled: bool = True
    aum_multiples: list[PositiveFloat]
    max_participation_rate: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    impact_exponent: PositiveFloat = 0.5
    variable_cost_fraction: UnitFloat = 0.5
    minimum_fill_ratio: UnitFloat

    @field_validator("aum_multiples")
    @classmethod
    def validate_aum_multiples(cls, values: list[float]) -> list[float]:
        if not values or values != sorted(set(values)):
            raise ValueError("aum_multiples must be strictly increasing and unique")
        return values


class BorrowFinancingConfig(StrictConfig):
    short_borrow_bps_annual: NonNegativeFloat
    cash_financing_bps_annual: NonNegativeFloat
    sessions_per_year: Annotated[int, Field(ge=1, le=366)] = 252
    calibration_provenance: Annotated[str, Field(min_length=1, max_length=512)] = (
        "predeclared deterministic simulation sensitivity; not calibrated from strategy "
        "test outcomes or presented as observed execution quality"
    )


class BacktestConfig(StrictConfig):
    strategy: Literal[
        "long_only_topk", "long_short", "rank_weighted", "confidence_weighted", "threshold"
    ]
    strategy_params: StrategyParams
    rebalance_frequency: PositiveInt
    execution_lag: PositiveInt
    liquidate_at_end: bool
    execution: ExecutionConfig
    costs: CostConfig
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    portfolio: PortfolioPolicyConfig
    risk: RiskPolicyConfig
    capacity: CapacityPolicyConfig
    initial_capital: PositiveFloat
    borrow_financing: BorrowFinancingConfig | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> BacktestConfig:
        configured = {
            name for name, value in self.strategy_params.model_dump().items() if value is not None
        }
        allowed_by_strategy = {
            "long_only_topk": {"top_k"},
            "long_short": {"quantile"},
            "rank_weighted": set(),
            "confidence_weighted": {"clip"},
            "threshold": {"threshold", "allow_short"},
        }
        unused = configured - allowed_by_strategy[self.strategy]
        if unused:
            raise ValueError(
                f"strategy_params contains settings unused by {self.strategy}: {sorted(unused)}"
            )
        if self.strategy == "long_only_topk" and self.strategy_params.top_k is None:
            raise ValueError("long_only_topk requires strategy_params.top_k")
        if self.strategy == "long_short" and self.strategy_params.quantile is None:
            raise ValueError("long_short requires strategy_params.quantile")
        if self.strategy == "confidence_weighted" and self.strategy_params.clip is None:
            raise ValueError("confidence_weighted requires strategy_params.clip")
        if self.strategy == "threshold" and self.strategy_params.threshold is None:
            raise ValueError("threshold requires strategy_params.threshold")
        return self


class StandalonePortfolioConfig(StrictConfig):
    scheme: Literal["equal_weight", "prediction_weighted", "inverse_vol"]
    max_weight: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    max_gross_exposure: PositiveFloat
    max_net_exposure: NonNegativeFloat
    turnover_cap: PositiveFloat | None
    vol_lookback: PositiveInt
    cash_buffer: UnitFloat

    @model_validator(mode="after")
    def validate_exposures(self) -> StandalonePortfolioConfig:
        if self.max_weight > self.max_gross_exposure:
            raise ValueError("max_weight cannot exceed max_gross_exposure")
        if self.max_net_exposure > self.max_gross_exposure:
            raise ValueError("max_net_exposure cannot exceed max_gross_exposure")
        return self


class StressScenario(StrictConfig):
    name: str
    market_shock: Annotated[float, Field(ge=-1, allow_inf_nan=False)]
    vol_multiplier: PositiveFloat


class RiskAnalyticsConfig(StrictConfig):
    stress_scenarios: list[StressScenario]

    @model_validator(mode="after")
    def validate_risk_contract(self) -> RiskAnalyticsConfig:
        names = [scenario.name for scenario in self.stress_scenarios]
        if not names or len(names) != len(set(names)):
            raise ValueError("stress_scenarios must have unique non-empty names")
        return self


class ProbabilityCalibrationConfig(StrictConfig):
    """Governed probability-calibration settings."""

    method: Literal["platt", "isotonic"]
    reliability_bins: Annotated[int, Field(ge=2, le=100)]


class BootstrapUncertaintyConfig(StrictConfig):
    """Bounded moving-block bootstrap settings."""

    n_resamples: Annotated[int, Field(ge=100, le=100_000)]
    block_length: Annotated[int, Field(ge=2, le=10_000_000)]
    confidence_level: OpenUnitFloat
    seed: NonNegativeInt
    circular: bool


class ConformalUncertaintyConfig(StrictConfig):
    """Block-conformal residual interval settings."""

    alpha: OpenUnitFloat
    block_length: Annotated[int, Field(ge=2, le=10_000_000)]
    min_blocks: Annotated[int, Field(ge=3, le=10_000)]


class QuantileRegressionUncertaintyConfig(StrictConfig):
    """Bounded linear quantile-regression settings."""

    lower_quantile: OpenUnitFloat
    upper_quantile: OpenUnitFloat
    alpha: NonNegativeFloat
    max_iter: Annotated[int, Field(ge=100, le=1_000_000)]
    max_samples: Annotated[int, Field(ge=10, le=10_000_000)]
    max_features: Annotated[int, Field(ge=1, le=10_000)]

    @model_validator(mode="after")
    def validate_quantiles(self) -> QuantileRegressionUncertaintyConfig:
        if not self.lower_quantile < 0.5 < self.upper_quantile:
            raise ValueError("quantiles must satisfy lower_quantile < 0.5 < upper_quantile")
        return self


class CalibrationUncertaintyConfig(StrictConfig):
    """Strict configuration surface for SF-S2-MR6."""

    version: Literal["1.0.0"]
    probability: ProbabilityCalibrationConfig
    bootstrap: BootstrapUncertaintyConfig
    conformal: ConformalUncertaintyConfig
    quantile_regression: QuantileRegressionUncertaintyConfig

    @model_validator(mode="after")
    def validate_temporal_blocks(self) -> CalibrationUncertaintyConfig:
        if self.bootstrap.block_length != self.conformal.block_length:
            raise ValueError(
                "bootstrap and conformal block lengths must match the predeclared dependence policy"
            )
        return self


class MetricSuitePolicyConfig(StrictConfig):
    """Strict configuration surface for SF-S2-MR7."""

    version: Literal["1.0.0"]
    minimum_prediction_samples: Annotated[int, Field(ge=4, le=10_000_000)]
    minimum_trading_periods: Annotated[int, Field(ge=4, le=10_000_000)]
    reliability_bins: Annotated[int, Field(ge=2, le=100)]
    benchmark_name: str
    bootstrap: BootstrapUncertaintyConfig

    @field_validator("benchmark_name")
    @classmethod
    def validate_benchmark_name(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("benchmark_name must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def validate_block_support(self) -> MetricSuitePolicyConfig:
        minimum = min(self.minimum_prediction_samples, self.minimum_trading_periods)
        if self.bootstrap.block_length > minimum:
            raise ValueError(
                "bootstrap.block_length cannot exceed either minimum sample requirement"
            )
        return self


class MultipleTestingPolicyConfig(StrictConfig):
    """Frozen complete-family correction policy for research governance."""

    method: Literal["holm_bonferroni", "benjamini_hochberg"]
    alpha: OpenUnitFloat
    family_size: Annotated[int, Field(ge=1, le=10_000)]
    assumptions: Annotated[list[str], Field(min_length=1, max_length=64)]
    failed_trial_p_value: Annotated[float, Field(ge=1.0, le=1.0)] = 1.0

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(
            not value or value != value.strip() for value in values
        ):
            raise ValueError("assumptions must be unique non-empty declarations")
        return values


class KillCriterionConfig(StrictConfig):
    """One predeclared candidate rejection condition."""

    name: str
    metric: str
    operator: Literal["lt", "le", "gt", "ge"]
    threshold: float

    @field_validator("name", "metric")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if (
            not value
            or len(value) > 128
            or value != value.strip()
            or not value.isascii()
            or not all(character.isalnum() or character in "._-" for character in value)
        ):
            raise ValueError("research-governance identifiers must be safe ASCII")
        return value

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("kill-criterion threshold must be finite")
        return value


class ResearchLedgerResourceConfig(StrictConfig):
    """Bounded local append-only ledger resource policy."""

    max_records: Annotated[int, Field(ge=10, le=1_000_000)]
    max_bytes: Annotated[int, Field(ge=4096, le=1_073_741_824)]


class ResearchGovernanceConfig(StrictConfig):
    """Strict configuration surface for SF-S2-MR8."""

    version: Literal["1.0.0"]
    correction: MultipleTestingPolicyConfig
    kill_criteria: Annotated[list[KillCriterionConfig], Field(min_length=1, max_length=64)]
    ledger: ResearchLedgerResourceConfig

    @model_validator(mode="after")
    def validate_governance(self) -> ResearchGovernanceConfig:
        names = [criterion.name for criterion in self.kill_criteria]
        if len(names) != len(set(names)):
            raise ValueError("kill-criterion names must be unique")
        if self.ledger.max_records < self.correction.family_size * 4 + 2:
            raise ValueError("ledger.max_records cannot hold the minimum trial event family")
        return self


class ResearchConfig(StrictConfig):
    holdout_start: str
    benchmark_symbol: str
    target: str
    horizons: list[int]
    selection_metric: Literal["rank_ic"]
    seed: NonNegativeInt

    @field_validator("holdout_start")
    @classmethod
    def validate_holdout_start(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("holdout_start must be an ISO-8601 calendar date") from exc
        return value

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, values: list[int]) -> list[int]:
        return _ordered_unique(values, field_name="horizons")

    @model_validator(mode="after")
    def validate_target(self) -> ResearchConfig:
        if self.target not in {f"fwd_ret_{horizon}" for horizon in self.horizons}:
            raise ValueError("target must name one of the configured horizons")
        return self


class ReadinessConfig(StrictConfig):
    rubric_version: Literal["1.0.0"]
    minimum_holdout_days: PositiveInt
    minimum_deflated_sharpe_probability: UnitFloat
    maximum_probability_of_backtest_overfitting: UnitFloat
    maximum_drawdown: UnitFloat
    minimum_annual_excess_return: float
    maximum_average_turnover: NonNegativeFloat
    require_complete_point_in_time: bool


class SignalFoundryResearchConfig(StrictConfig):
    research: ResearchConfig
    readiness: ReadinessConfig
    models: list[ModelSpec]
    features: FeatureConfig
    walk_forward: WalkForwardConfig
    backtest: BacktestConfig

    @model_validator(mode="after")
    def validate_experiment(self) -> SignalFoundryResearchConfig:
        if not self.models:
            raise ValueError("models must contain at least one candidate")
        if self.walk_forward.embargo_days < max(self.research.horizons):
            raise ValueError("walk_forward.embargo_days must cover the maximum label horizon")
        return self


class DecisionThresholdConfig(StrictConfig):
    """Strict configuration for the pure pre-portfolio decision policy."""

    schema_version: Literal["1.0.0"]
    required_margin: UnitFloat
    cost_multiplier: Annotated[float, Field(ge=1, le=100, allow_inf_nan=False)]
    cost_uncertainty_multiplier: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    uncertainty_penalty: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    maximum_total_cost: UnitFloat
    maximum_prediction_uncertainty: UnitFloat
    maximum_model_disagreement: UnitFloat
    maximum_regime_uncertainty: UnitFloat
    maximum_drift_score: UnitFloat
    maximum_data_age_seconds: Annotated[int, Field(ge=1, le=31_536_000)]
    maximum_absolute_expected_return: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    maximum_batch_size: Annotated[int, Field(ge=1, le=100_000)]

    @model_validator(mode="after")
    def validate_margin_support(self) -> DecisionThresholdConfig:
        if self.required_margin >= self.maximum_absolute_expected_return:
            raise ValueError("required_margin must be below maximum_absolute_expected_return")
        return self


class DecisionPolicyStudyConfig(StrictConfig):
    """Frozen deterministic synthetic study and resource budget."""

    schema_version: Literal["1.0.0"]
    scope: Literal["synthetic_engineering_only"]
    baselines: list[Literal["always_trade", "never_trade"]]
    seed: NonNegativeInt
    observation_count: Annotated[int, Field(ge=100, le=100_000)]
    period_count: Annotated[int, Field(ge=2, le=10_000)]
    anchor_time: str
    expected_return_scale: Annotated[float, Field(gt=0, le=0.10, allow_inf_nan=False)]
    realized_noise_scale: Annotated[float, Field(ge=0, le=0.10, allow_inf_nan=False)]
    minimum_expected_cost: UnitFloat
    maximum_expected_cost: UnitFloat
    cost_uncertainty_scale: UnitFloat
    prediction_uncertainty_scale: UnitFloat
    disagreement_scale: UnitFloat
    regime_uncertainty_scale: UnitFloat
    unsupported_regime_probability: UnitFloat
    stale_probability: UnitFloat
    future_probability: UnitFloat
    drift_probability: UnitFloat
    unit_turnover: Annotated[float, Field(gt=0, le=10, allow_inf_nan=False)]
    unit_notional: PositiveFloat
    period_capacity: PositiveFloat
    interpretation: str
    protected_holdout_access: Literal[False]
    broker_access: Literal[False]

    @field_validator("anchor_time")
    @classmethod
    def validate_anchor_time(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("anchor_time must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("anchor_time must include a timezone")
        return value

    @field_validator("interpretation")
    @classmethod
    def validate_interpretation(cls, value: str) -> str:
        if not value or value != value.strip() or len(value) > 512:
            raise ValueError(
                "interpretation must be non-empty, trimmed, and at most 512 characters"
            )
        return value

    @model_validator(mode="after")
    def validate_study(self) -> DecisionPolicyStudyConfig:
        if self.baselines != ["always_trade", "never_trade"]:
            raise ValueError("baselines must be frozen as always_trade, never_trade")
        if self.period_count > self.observation_count:
            raise ValueError("period_count cannot exceed observation_count")
        if self.minimum_expected_cost >= self.maximum_expected_cost:
            raise ValueError("minimum_expected_cost must be below maximum_expected_cost")
        if self.stale_probability + self.future_probability > 1.0:
            raise ValueError("stale_probability and future_probability cannot sum above one")
        return self


class DecisionPolicyExperimentConfig(StrictConfig):
    """Complete opt-in SF-S3-MR10 study configuration."""

    version: Literal["1.0.0"]
    policy: DecisionThresholdConfig
    study: DecisionPolicyStudyConfig

    @model_validator(mode="after")
    def validate_resource_budget(self) -> DecisionPolicyExperimentConfig:
        if self.study.observation_count > self.policy.maximum_batch_size:
            raise ValueError("study.observation_count cannot exceed policy.maximum_batch_size")
        return self


ConfigModel = (
    DataConfig
    | FeatureConfig
    | LabelsConfig
    | ModelsConfig
    | BacktestConfig
    | StandalonePortfolioConfig
    | RiskAnalyticsConfig
    | CalibrationUncertaintyConfig
    | MetricSuitePolicyConfig
    | ResearchGovernanceConfig
    | SignalFoundryResearchConfig
    | DecisionPolicyExperimentConfig
)

SCHEMAS: Mapping[str, type[ConfigModel]] = {
    "data": DataConfig,
    "features": FeatureConfig,
    "labels": LabelsConfig,
    "models": ModelsConfig,
    "backtest": BacktestConfig,
    "calibration": CalibrationUncertaintyConfig,
    "decision_policy": DecisionPolicyExperimentConfig,
    "metrics": MetricSuitePolicyConfig,
    "portfolio": StandalonePortfolioConfig,
    "research_governance": ResearchGovernanceConfig,
    "risk": RiskAnalyticsConfig,
    "signal_foundry_research": SignalFoundryResearchConfig,
}


def load_config(path: str | Path, kind: str) -> dict[str, Any]:
    """Load and normalize one declared configuration kind.

    Raises:
        ConfigValidationError: If the file is unreadable, is not a mapping, has
            an unknown kind or key, or violates a field/cross-field invariant.
    """

    schema = SCHEMAS.get(kind)
    if schema is None:
        raise ConfigValidationError(f"unknown configuration kind {kind!r}")
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigValidationError(f"could not parse {kind} configuration {source}") from exc
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"{kind} configuration root must be a mapping")
    try:
        validated = schema.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(f"invalid {kind} configuration {source}: {exc}") from exc
    return validated.model_dump(mode="python", exclude_none=False)


def load_data_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "data")


def load_feature_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "features")


def load_labels_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "labels")


def load_models_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "models")


def load_backtest_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "backtest")


def load_portfolio_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "portfolio")


def load_risk_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "risk")


def load_calibration_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "calibration")


def load_metrics_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "metrics")


def load_research_governance_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "research_governance")


def load_signal_foundry_research_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "signal_foundry_research")


def load_decision_policy_config(path: str | Path) -> dict[str, Any]:
    return load_config(path, "decision_policy")
