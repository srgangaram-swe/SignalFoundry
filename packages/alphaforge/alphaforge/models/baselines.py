"""Naive baselines every ML model must beat to justify its complexity.

Each baseline implements the full :class:`~alphaforge.models.base.AlphaModel`
contract with JSON-safe, deterministic serialization. None is a deployable
strategy — they exist to bound what "adding value" means and are included
automatically in comparative evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alphaforge.models.base import AlphaModel, FeatureSchemaError, InvalidLabelError


class ZeroBaseline(AlphaModel):
    """Predicts zero return everywhere — the efficient-market null."""

    name = "zero_baseline"
    feature_agnostic = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ZeroBaseline:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def _fitted_state(self) -> dict[str, Any]:
        return {}


class HistoricalMeanBaseline(AlphaModel):
    """Predicts the training-set mean target, with its std as uncertainty."""

    name = "historical_mean"
    feature_agnostic = True

    def __init__(self) -> None:
        self.mean_: float = 0.0
        self.std_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HistoricalMeanBaseline:
        self.mean_ = float(y.mean())
        self.std_ = float(y.std(ddof=0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)

    def predict_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.std_)

    def _fitted_state(self) -> dict[str, Any]:
        return {"mean": self.mean_, "std": self.std_}

    def _load_fitted_state(self, state: dict[str, Any]) -> None:
        self.mean_ = float(state["mean"])
        self.std_ = float(state["std"])


class LagBaseline(AlphaModel):
    """Predicts the most recent observed return — the persistence/random-walk null.

    ``feature`` should name the column holding the latest completed return; the
    forecast for the next period is simply that value.
    """

    name = "lag_baseline"
    requires_raw_features = True

    def __init__(self, feature: str = "ret_1d") -> None:
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LagBaseline:
        if self.feature not in X.columns:
            raise FeatureSchemaError(f"lag feature {self.feature!r} not in X")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature].fillna(0.0).to_numpy(dtype=float)

    def required_features(self) -> tuple[str, ...]:
        return (self.feature,)

    def get_params(self) -> dict[str, Any]:
        return {"feature": self.feature}

    def _fitted_state(self) -> dict[str, Any]:
        return {}


class MovingAverageBaseline(AlphaModel):
    """Simple moving-average trend rule: long above the average, short below.

    ``feature`` should name a price-relative-to-moving-average column (e.g.
    ``ma_ratio_20`` = price / SMA - 1); the signal is its sign.
    """

    name = "moving_average_baseline"
    requires_raw_features = True

    def __init__(self, feature: str = "ma_ratio_20") -> None:
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MovingAverageBaseline:
        if self.feature not in X.columns:
            raise FeatureSchemaError(f"moving-average feature {self.feature!r} not in X")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.sign(X[self.feature].fillna(0.0).to_numpy(dtype=float))

    def required_features(self) -> tuple[str, ...]:
        return (self.feature,)

    def get_params(self) -> dict[str, Any]:
        return {"feature": self.feature}

    def _fitted_state(self) -> dict[str, Any]:
        return {}


class MomentumBaseline(AlphaModel):
    """Predicts a scaled momentum feature — the classic cross-sectional signal.

    No fitting: this is a rule, not a model, which is exactly why it is a useful
    baseline. If ML cannot beat ``0.05 * momentum_20``, the ML is noise.
    """

    name = "momentum_baseline"
    requires_raw_features = True

    def __init__(self, feature: str = "momentum_20", scale: float = 0.05) -> None:
        self.feature = feature
        self.scale = scale

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MomentumBaseline:
        if self.feature not in X.columns:
            raise FeatureSchemaError(f"momentum feature {self.feature!r} not in X")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.feature].fillna(0.0) * self.scale).to_numpy(dtype=float)

    def required_features(self) -> tuple[str, ...]:
        return (self.feature,)

    def get_params(self) -> dict[str, Any]:
        return {"feature": self.feature, "scale": self.scale}

    def _fitted_state(self) -> dict[str, Any]:
        return {}


class EqualProbabilityClassifier(AlphaModel):
    """The coin-flip classifier: the same up-probability for every observation."""

    name = "equal_probability"
    task = "classification"
    feature_agnostic = True

    def __init__(self, p_up: float = 0.5) -> None:
        if not 0.0 <= p_up <= 1.0:
            raise ValueError("p_up must be in [0, 1]")
        self.p_up = p_up

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EqualProbabilityClassifier:
        labels = set(np.unique(y.to_numpy(dtype=float)).tolist())
        if not labels <= {0.0, 1.0}:
            raise InvalidLabelError(f"classification labels must be binary {{0, 1}}, got {labels}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p_up)

    def get_params(self) -> dict[str, Any]:
        return {"p_up": self.p_up}

    def _fitted_state(self) -> dict[str, Any]:
        return {}


class _ConstantSignalBaseline(AlphaModel):
    """Predicts one constant signal for every row (feature-agnostic)."""

    feature_agnostic = True
    _signal: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _ConstantSignalBaseline:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._signal)

    def _fitted_state(self) -> dict[str, Any]:
        return {}


class BuyAndHoldBaseline(_ConstantSignalBaseline):
    """Always-long benchmark: a constant positive signal (hold the asset).

    As a per-row alpha signal it is uniform; the portfolio layer reads it as full,
    persistent long exposure — the return you earn by doing nothing.
    """

    name = "buy_and_hold"
    _signal = 1.0


class EqualWeightBaseline(_ConstantSignalBaseline):
    """Equal-weight benchmark: a uniform score so every asset gets equal weight.

    A constant cross-sectional score ranks all names equally, which a rank- or
    score-proportional portfolio turns into equal weights — the naive
    diversification benchmark active strategies must beat.
    """

    name = "equal_weight"
    _signal = 1.0
