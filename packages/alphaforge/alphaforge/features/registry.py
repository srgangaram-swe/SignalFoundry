"""Versioned semantic contracts for causal AlphaForge features.

The registry describes existing feature mathematics; it does not execute them.
Stable identities bind names, inputs, parameters, lookbacks, warm-up policy,
and output schema so code and cached data cannot silently disagree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

REGISTRY_VERSION = "1.0.0"
ID_COLUMNS = ("date", "symbol")
_MISSING_POLICIES = frozenset({"warmup_nan", "none"})


class FeatureContractError(ValueError):
    """Raised when feature metadata or output violates its declared contract."""


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation for semantic hashing."""

    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FeatureContractError("feature metadata must be finite JSON-compatible data") from exc


def semantic_hash(value: Any) -> str:
    """Return the SHA-256 identity of a canonical semantic record."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_token(value: str, field: str) -> None:
    if not value or value != value.strip() or not value.isascii():
        raise FeatureContractError(f"{field} must be non-empty, trimmed ASCII")


@dataclass(frozen=True)
class FeatureDefinition:
    """Immutable contract for one independently versioned feature family."""

    name: str
    version: str
    inputs: tuple[str, ...]
    parameters: Mapping[str, Any]
    lookback_sessions: int
    warmup_sessions: int
    outputs: tuple[str, ...]
    missing_value_policy: str = "warmup_nan"
    fit_period: str = "causal-through-observation"

    def __post_init__(self) -> None:
        _validate_token(self.name, "feature name")
        if self.version != REGISTRY_VERSION:
            raise FeatureContractError(f"unsupported feature version {self.version!r}")
        if self.lookback_sessions < 0 or self.warmup_sessions < 0:
            raise FeatureContractError("lookback and warm-up must be non-negative")
        if not self.inputs or not self.outputs:
            raise FeatureContractError("feature inputs and outputs must be non-empty")
        for value in (*self.inputs, *self.outputs):
            _validate_token(value, "feature column")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise FeatureContractError("feature input and output columns must be unique")
        if self.missing_value_policy not in _MISSING_POLICIES:
            raise FeatureContractError("unsupported missing-value policy")
        _validate_token(self.fit_period, "fit period")
        normalized_parameters = json.loads(canonical_json(self.parameters))
        object.__setattr__(self, "parameters", _freeze_json(normalized_parameters))

    def _semantic_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "inputs": self.inputs,
            "parameters": _thaw_json(self.parameters),
            "lookback_sessions": self.lookback_sessions,
            "warmup_sessions": self.warmup_sessions,
            "outputs": self.outputs,
            "missing_value_policy": self.missing_value_policy,
            "fit_period": self.fit_period,
        }

    @property
    def feature_id(self) -> str:
        """Content identity of the complete semantic definition."""

        return semantic_hash(self._semantic_record())

    def record(self) -> dict[str, Any]:
        return {**self._semantic_record(), "feature_id": self.feature_id}


@dataclass(frozen=True, init=False)
class FeatureRegistry:
    """Exact-version registry with deterministic schema and identity."""

    version: str
    _definitions: tuple[FeatureDefinition, ...]

    def __init__(
        self, definitions: tuple[FeatureDefinition, ...], *, version: str = REGISTRY_VERSION
    ) -> None:
        if version != REGISTRY_VERSION:
            raise FeatureContractError(f"unsupported registry version {version!r}")
        if not definitions:
            raise FeatureContractError("feature registry must not be empty")
        keys = [(item.name, item.version) for item in definitions]
        outputs = [column for item in definitions for column in item.outputs]
        if len(keys) != len(set(keys)):
            raise FeatureContractError("duplicate feature name/version")
        if len(outputs) != len(set(outputs)):
            raise FeatureContractError("feature output columns overlap")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "_definitions", definitions)

    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return self._definitions

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(column for item in self._definitions for column in item.outputs)

    @property
    def required_warmup_sessions(self) -> int:
        return max(item.warmup_sessions for item in self._definitions)

    @property
    def registry_id(self) -> str:
        return semantic_hash(self.manifest())

    def resolve(self, name: str, version: str) -> FeatureDefinition:
        for definition in self._definitions:
            if definition.name == name and definition.version == version:
                return definition
        raise FeatureContractError(f"unknown feature/version {name!r}@{version!r}")

    def manifest(self) -> dict[str, Any]:
        return {
            "registry_version": self.version,
            "features": [item.record() for item in self._definitions],
        }


def _definition(
    name: str,
    *,
    inputs: tuple[str, ...],
    parameters: dict[str, Any],
    lookback: int,
    warmup: int,
    outputs: tuple[str, ...],
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        version=REGISTRY_VERSION,
        inputs=inputs,
        parameters=parameters,
        lookback_sessions=lookback,
        warmup_sessions=warmup,
        outputs=outputs,
    )


def build_default_registry(config: dict[str, Any] | None = None) -> FeatureRegistry:
    """Describe the exact columns emitted by :func:`build_features`."""

    cfg = config or {}
    if cfg.get("registry_version", REGISTRY_VERSION) != REGISTRY_VERSION:
        raise FeatureContractError("unsupported feature registry version")
    return_lags = tuple(cfg.get("return_lags", [1, 5, 20]))
    vol_windows = tuple(cfg.get("vol_windows", [5, 20, 60]))
    ma_windows = tuple(cfg.get("ma_windows", [10, 20, 50, 200]))
    momentum_windows = tuple(cfg.get("momentum_windows", [20, 60, 120]))
    macd = dict(cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9}))
    definitions = [
        _definition(
            "returns",
            inputs=("close",),
            parameters={"lags": return_lags},
            lookback=max(return_lags),
            warmup=max(return_lags),
            outputs=tuple(f"ret_{lag}" for lag in return_lags) + ("logret_1",),
        ),
        _definition(
            "volatility",
            inputs=("close",),
            parameters={"windows": vol_windows, "annualization_days": 252},
            lookback=max(vol_windows),
            warmup=max(vol_windows),
            outputs=tuple(f"vol_{window}" for window in vol_windows),
        ),
        _definition(
            "moving_average_ratio",
            inputs=("close",),
            parameters={"windows": ma_windows},
            lookback=max(ma_windows),
            warmup=max(ma_windows),
            outputs=tuple(f"ma_ratio_{window}" for window in ma_windows),
        ),
        _definition(
            "momentum",
            inputs=("close",),
            parameters={"windows": momentum_windows},
            lookback=max(momentum_windows),
            warmup=max(momentum_windows),
            outputs=tuple(f"momentum_{window}" for window in momentum_windows),
        ),
        _definition(
            "rsi",
            inputs=("close",),
            parameters={"window": int(cfg.get("rsi_window", 14)), "smoothing": "wilder-ewm"},
            lookback=int(cfg.get("rsi_window", 14)),
            warmup=int(cfg.get("rsi_window", 14)),
            outputs=("rsi",),
        ),
        _definition(
            "macd",
            inputs=("close",),
            parameters=macd,
            lookback=int(macd["slow"]) + int(macd["signal"]) - 1,
            warmup=int(macd["slow"]) + int(macd["signal"]) - 1,
            outputs=("macd", "macd_signal", "macd_hist"),
        ),
        _definition(
            "bollinger",
            inputs=("close",),
            parameters={"window": int(cfg.get("bollinger_window", 20)), "width": 2.0},
            lookback=int(cfg.get("bollinger_window", 20)),
            warmup=int(cfg.get("bollinger_window", 20)),
            outputs=("bollinger_z",),
        ),
        _definition(
            "mean_reversion",
            inputs=("close",),
            parameters={
                "return_window": int(cfg.get("mean_reversion_window", 5)),
                "normalization_window": 60,
            },
            lookback=int(cfg.get("mean_reversion_window", 5)) + 60,
            warmup=int(cfg.get("mean_reversion_window", 5)) + 60,
            outputs=("meanrev_z",),
        ),
        _definition(
            "volume",
            inputs=("close", "volume"),
            parameters={"window": int(cfg.get("volume_window", 20))},
            lookback=int(cfg.get("volume_window", 20)),
            warmup=int(cfg.get("volume_window", 20)),
            outputs=("volume_z", "volume_ratio", "log_dollar_volume"),
        ),
        _definition(
            "drawdown",
            inputs=("close",),
            parameters={
                "window": int(cfg.get("drawdown_window", 252)),
                "minimum_periods": 20,
            },
            lookback=int(cfg.get("drawdown_window", 252)),
            warmup=20,
            outputs=("drawdown",),
        ),
        _definition(
            "rolling_sharpe",
            inputs=("close",),
            parameters={
                "window": int(cfg.get("rolling_sharpe_window", 60)),
                "annualization_days": 252,
            },
            lookback=int(cfg.get("rolling_sharpe_window", 60)),
            warmup=int(cfg.get("rolling_sharpe_window", 60)),
            outputs=("rolling_sharpe",),
        ),
        _definition(
            "benchmark_relative",
            inputs=("close", "benchmark.close"),
            parameters={"window": int(cfg.get("beta_window", 60))},
            lookback=int(cfg.get("beta_window", 60)),
            warmup=int(cfg.get("beta_window", 60)),
            outputs=("beta", "bench_corr"),
        ),
        _definition(
            "market_regime",
            inputs=("benchmark.close",),
            parameters={
                "volatility_window": int(cfg.get("regime_vol_window", 20)),
                "trend_fast": int(cfg.get("regime_trend_fast", 50)),
                "trend_slow": int(cfg.get("regime_trend_slow", 200)),
            },
            lookback=max(
                int(cfg.get("regime_vol_window", 20)) * 3,
                int(cfg.get("regime_trend_slow", 200)),
            ),
            warmup=max(
                int(cfg.get("regime_vol_window", 20)) * 3,
                int(cfg.get("regime_trend_slow", 200)),
            ),
            outputs=("bench_vol", "bench_vol_pctile", "bench_trend", "high_vol_regime"),
        ),
    ]
    if cfg.get("hmm_regime", True):
        minimum = int(cfg.get("hmm_min_train", 252))
        definitions.append(
            _definition(
                "causal_hmm_regime",
                inputs=("benchmark.close",),
                parameters={
                    "states": 2,
                    "refit_every": int(cfg.get("hmm_refit_every", 63)),
                    "minimum_train": minimum,
                },
                lookback=minimum,
                warmup=minimum,
                outputs=("hmm_stress_prob",),
            )
        )
    if cfg.get("cross_sectional", True):
        source_columns = tuple(
            column
            for column in ("momentum_20", "momentum_60", "vol_20", "ret_5", "rolling_sharpe")
            if column in {value for definition in definitions for value in definition.outputs}
        )
        if source_columns:
            definitions.append(
                _definition(
                    "cross_sectional_ranks",
                    inputs=source_columns,
                    parameters={"method": "average", "percentile": True},
                    lookback=0,
                    warmup=0,
                    outputs=tuple(f"cs_rank_{column}" for column in source_columns),
                )
            )
    return FeatureRegistry(tuple(definitions))


def validate_feature_frame(frame: pd.DataFrame, registry: FeatureRegistry) -> None:
    """Fail closed when an emitted frame diverges from its registered schema."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FeatureContractError("feature frame must be a non-empty DataFrame")
    expected = [*ID_COLUMNS, *registry.output_columns]
    actual = list(frame.columns)
    if actual != expected:
        raise FeatureContractError(
            f"feature schema mismatch; missing={sorted(set(expected) - set(actual))}, "
            f"unknown={sorted(set(actual) - set(expected))}"
        )
    if frame[list(ID_COLUMNS)].isna().any().any():
        raise FeatureContractError("feature identifiers must not be missing")
    if frame.duplicated(list(ID_COLUMNS)).any():
        raise FeatureContractError("feature identifiers must be unique")
    numeric = frame[list(registry.output_columns)].apply(pd.to_numeric, errors="coerce")
    original_missing = frame[list(registry.output_columns)].isna()
    if (numeric.isna() & ~original_missing).any().any():
        raise FeatureContractError("feature outputs must be numeric")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise FeatureContractError("feature outputs must not contain infinity")
