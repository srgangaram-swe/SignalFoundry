"""Governed tabular benchmark models for forward-return research (SF-S2-MR5).

Every estimator embeds train-fold-only imputation and, where mathematically
appropriate, scaling. Public factories expose a deliberately narrow typed
hyperparameter surface with explicit resource bounds. Optional boosting
ecosystems are imported only when selected and never change another registry
name's semantics.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alphaforge.models.base import AlphaModel, TrainingDiagnostics

_MAX_ESTIMATORS = 2_000
_MAX_ITERATIONS = 50_000
_MAX_TREE_DEPTH = 64
_MAX_JOBS = 64


def _boolean(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _bounded_int(name: str, value: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _positive_float(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {comparator}")
    return number


def _fraction(name: str, value: float, *, include_zero: bool = False) -> float:
    number = _positive_float(name, value, allow_zero=include_zero)
    lower_ok = number >= 0.0 if include_zero else number > 0.0
    if not lower_ok or number > 1.0:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return number


def _jobs(value: int) -> int:
    return _bounded_int("n_jobs", value, minimum=1, maximum=_MAX_JOBS)


def _iteration_count(estimator: Any) -> int | None:
    for attribute in ("n_iter_", "n_estimators_", "tree_count_"):
        value = getattr(estimator, attribute, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.size:
            return int(np.max(array))
    members = getattr(estimator, "estimators_", None)
    if members is not None:
        return len(members)
    return None


class SklearnModel(AlphaModel):
    """Validated estimator pipeline with immutable termination diagnostics."""

    def __init__(
        self,
        estimator: Any,
        name: str,
        *,
        scale: bool,
        params: dict[str, Any],
        backend: str,
        iteration_limit: int | None = None,
        seed: int | None = None,
        requires_convergence: bool = False,
    ) -> None:
        steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        self.pipeline = Pipeline(steps)
        self.name = name
        self.columns_: list[str] | None = None
        self._params = dict(params)
        self._backend = backend
        self._iteration_limit = iteration_limit
        self._seed = seed
        self._requires_convergence = requires_convergence
        self._training_diagnostics: TrainingDiagnostics | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SklearnModel:
        self.columns_ = list(X.columns)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.pipeline.fit(X, y)
        messages = tuple(
            dict.fromkeys(f"{item.category.__name__}: {item.message}" for item in caught)
        )
        estimator = self.pipeline.named_steps["model"]
        iterations = _iteration_count(estimator)
        convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
        if convergence_warning:
            status: Literal["converged", "completed", "max_iterations", "warning"] = (
                "max_iterations"
                if iterations is not None
                and self._iteration_limit is not None
                and iterations >= self._iteration_limit
                else "warning"
            )
        elif messages:
            status = "warning"
        elif self._requires_convergence:
            status = (
                "completed"
                if iterations is None
                else (
                    "max_iterations"
                    if self._iteration_limit is not None and iterations >= self._iteration_limit
                    else "converged"
                )
            )
        else:
            status = "completed"
        self._training_diagnostics = TrainingDiagnostics(
            backend=self._backend,
            status=status,
            iterations=iterations,
            iteration_limit=self._iteration_limit,
            seed=self._seed,
            warnings=messages,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None:  # pragma: no cover - contract checks fitted state first
            raise RuntimeError("model feature schema is unavailable")
        return np.asarray(self.pipeline.predict(X[self.columns_]), dtype=float)

    def feature_importance(self) -> pd.Series | None:
        estimator = self.pipeline.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.ravel(estimator.coef_))
        else:
            return None
        if self.columns_ is None:
            return None
        return pd.Series(values, index=self.columns_).sort_values(ascending=False)

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def training_diagnostics(self) -> TrainingDiagnostics:
        self._ensure_fitted()
        if self._training_diagnostics is None:  # pragma: no cover - fit publishes atomically
            raise RuntimeError("training diagnostics are unavailable")
        return self._training_diagnostics


def make_linear(
    fit_intercept: bool = True,
    positive: bool = False,
) -> SklearnModel:
    params = {
        "fit_intercept": _boolean("fit_intercept", fit_intercept),
        "positive": _boolean("positive", positive),
    }
    return SklearnModel(
        LinearRegression(**params),
        "linear",
        scale=True,
        params=params,
        backend="sklearn.linear_model.LinearRegression",
    )


def make_ridge(
    alpha: float = 10.0,
    fit_intercept: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-4,
) -> SklearnModel:
    params = {
        "alpha": _positive_float("alpha", alpha, allow_zero=True),
        "fit_intercept": _boolean("fit_intercept", fit_intercept),
        "max_iter": _bounded_int("max_iter", max_iter, minimum=1, maximum=_MAX_ITERATIONS),
        "tol": _positive_float("tol", tol),
    }
    return SklearnModel(
        Ridge(**params),
        "ridge",
        scale=True,
        params=params,
        backend="sklearn.linear_model.Ridge",
        iteration_limit=max_iter,
        requires_convergence=True,
    )


def make_lasso(
    alpha: float = 1e-4,
    fit_intercept: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-4,
) -> SklearnModel:
    params = {
        "alpha": _positive_float("alpha", alpha),
        "fit_intercept": _boolean("fit_intercept", fit_intercept),
        "max_iter": _bounded_int("max_iter", max_iter, minimum=1, maximum=_MAX_ITERATIONS),
        "tol": _positive_float("tol", tol),
        "selection": "cyclic",
    }
    return SklearnModel(
        Lasso(**params),
        "lasso",
        scale=True,
        params=params,
        backend="sklearn.linear_model.Lasso",
        iteration_limit=max_iter,
        requires_convergence=True,
    )


def make_elastic_net(
    alpha: float = 1e-3,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-4,
) -> SklearnModel:
    params = {
        "alpha": _positive_float("alpha", alpha),
        "l1_ratio": _fraction("l1_ratio", l1_ratio, include_zero=True),
        "fit_intercept": _boolean("fit_intercept", fit_intercept),
        "max_iter": _bounded_int("max_iter", max_iter, minimum=1, maximum=_MAX_ITERATIONS),
        "tol": _positive_float("tol", tol),
        "selection": "cyclic",
    }
    return SklearnModel(
        ElasticNet(**params),
        "elastic_net",
        scale=True,
        params=params,
        backend="sklearn.linear_model.ElasticNet",
        iteration_limit=max_iter,
        requires_convergence=True,
    )


def make_huber(
    epsilon: float = 1.35,
    alpha: float = 1e-4,
    max_iter: int = 1_000,
    tol: float = 1e-5,
) -> SklearnModel:
    epsilon_value = _positive_float("epsilon", epsilon)
    if epsilon_value < 1.0:
        raise ValueError("epsilon must be >= 1")
    params = {
        "epsilon": epsilon_value,
        "alpha": _positive_float("alpha", alpha, allow_zero=True),
        "max_iter": _bounded_int("max_iter", max_iter, minimum=1, maximum=10_000),
        "tol": _positive_float("tol", tol),
    }
    return SklearnModel(
        HuberRegressor(**params),
        "huber",
        scale=True,
        params=params,
        backend="sklearn.linear_model.HuberRegressor",
        iteration_limit=max_iter,
        requires_convergence=True,
    )


def _forest_params(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float | Literal["sqrt", "log2"],
    n_jobs: int,
    random_state: int,
) -> dict[str, Any]:
    if max_depth is not None:
        _bounded_int("max_depth", max_depth, minimum=1, maximum=_MAX_TREE_DEPTH)
    if isinstance(max_features, str):
        if max_features not in {"sqrt", "log2"}:
            raise ValueError("max_features must be 'sqrt', 'log2', or a fraction")
    else:
        max_features = _fraction("max_features", max_features)
    return {
        "n_estimators": _bounded_int(
            "n_estimators", n_estimators, minimum=1, maximum=_MAX_ESTIMATORS
        ),
        "max_depth": max_depth,
        "min_samples_leaf": _bounded_int(
            "min_samples_leaf", min_samples_leaf, minimum=1, maximum=1_000_000
        ),
        "max_features": max_features,
        "n_jobs": _jobs(n_jobs),
        "random_state": _bounded_int("random_state", random_state, minimum=0, maximum=2**32 - 1),
    }


def make_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = 6,
    min_samples_leaf: int = 50,
    max_features: float | Literal["sqrt", "log2"] = 1.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> SklearnModel:
    params = _forest_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return SklearnModel(
        RandomForestRegressor(**params),
        "random_forest",
        scale=False,
        params=params,
        backend="sklearn.ensemble.RandomForestRegressor",
        iteration_limit=n_estimators,
        seed=random_state,
    )


def make_extra_trees(
    n_estimators: int = 200,
    max_depth: int | None = 6,
    min_samples_leaf: int = 50,
    max_features: float | Literal["sqrt", "log2"] = 1.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> SklearnModel:
    params = _forest_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return SklearnModel(
        ExtraTreesRegressor(**params),
        "extra_trees",
        scale=False,
        params=params,
        backend="sklearn.ensemble.ExtraTreesRegressor",
        iteration_limit=n_estimators,
        seed=random_state,
    )


def _boosting_params(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    min_samples_leaf: int,
    l2_regularization: float,
    n_jobs: int,
    random_state: int,
) -> dict[str, Any]:
    return {
        "n_estimators": _bounded_int(
            "n_estimators", n_estimators, minimum=1, maximum=_MAX_ESTIMATORS
        ),
        "max_depth": _bounded_int("max_depth", max_depth, minimum=1, maximum=_MAX_TREE_DEPTH),
        "learning_rate": _fraction("learning_rate", learning_rate),
        "min_samples_leaf": _bounded_int(
            "min_samples_leaf", min_samples_leaf, minimum=1, maximum=1_000_000
        ),
        "l2_regularization": _positive_float(
            "l2_regularization", l2_regularization, allow_zero=True
        ),
        "n_jobs": _jobs(n_jobs),
        "random_state": _bounded_int("random_state", random_state, minimum=0, maximum=2**32 - 1),
    }


def make_hist_gradient_boosting(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    random_state: int = 42,
) -> SklearnModel:
    governed = _boosting_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        n_jobs=1,
        random_state=random_state,
    )
    estimator_params = {
        "max_iter": governed["n_estimators"],
        "max_depth": governed["max_depth"],
        "learning_rate": governed["learning_rate"],
        "min_samples_leaf": governed["min_samples_leaf"],
        "l2_regularization": governed["l2_regularization"],
        "random_state": governed["random_state"],
    }
    return SklearnModel(
        HistGradientBoostingRegressor(**estimator_params),
        "gradient_boosting",
        scale=False,
        params={"backend": "sklearn", **governed},
        backend="sklearn.ensemble.HistGradientBoostingRegressor",
        iteration_limit=n_estimators,
        seed=random_state,
        requires_convergence=True,
    )


def make_lightgbm(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> SklearnModel:
    params = _boosting_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm requires the 'ml' extra: pip install 'alphaforge[ml]'") from exc
    estimator = lgb.LGBMRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        num_leaves=min(2 ** int(params["max_depth"]), 255),
        learning_rate=params["learning_rate"],
        min_child_samples=params["min_samples_leaf"],
        reg_lambda=params["l2_regularization"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    return SklearnModel(
        estimator,
        "lightgbm",
        scale=False,
        params=params,
        backend="lightgbm.LGBMRegressor",
        iteration_limit=n_estimators,
        seed=random_state,
    )


def make_xgboost(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_child_weight: float = 1.0,
    l2_regularization: float = 1.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> SklearnModel:
    governed = _boosting_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=1,
        l2_regularization=l2_regularization,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    params = {key: value for key, value in governed.items() if key != "min_samples_leaf"}
    params["min_child_weight"] = _positive_float(
        "min_child_weight", min_child_weight, allow_zero=True
    )
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost requires the 'ml' extra: pip install 'alphaforge[ml]'") from exc
    estimator = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["l2_regularization"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
        objective="reg:squarederror",
        tree_method="hist",
        verbosity=0,
    )
    return SklearnModel(
        estimator,
        "xgboost",
        scale=False,
        params=params,
        backend="xgboost.XGBRegressor",
        iteration_limit=n_estimators,
        seed=random_state,
    )


def make_catboost(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
    n_jobs: int = 1,
    random_state: int = 42,
) -> SklearnModel:
    params = _boosting_params(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    try:
        import catboost as cb
    except ImportError as exc:
        raise ImportError("catboost requires the 'ml' extra: pip install 'alphaforge[ml]'") from exc
    estimator = cb.CatBoostRegressor(
        iterations=params["n_estimators"],
        depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_data_in_leaf=params["min_samples_leaf"],
        l2_leaf_reg=params["l2_regularization"],
        random_seed=params["random_state"],
        thread_count=params["n_jobs"],
        loss_function="RMSE",
        task_type="CPU",
        grow_policy="Depthwise",
        bootstrap_type="No",
        random_strength=0.0,
        allow_writing_files=False,
        verbose=False,
    )
    return SklearnModel(
        estimator,
        "catboost",
        scale=False,
        params=params,
        backend="catboost.CatBoostRegressor",
        iteration_limit=n_estimators,
        seed=random_state,
    )


def make_gradient_boosting(
    random_state: int = 42,
    backend: Literal["sklearn", "lightgbm"] = "sklearn",
    max_iter: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
) -> AlphaModel:
    """Backward-compatible explicit backend selector; no environment fallback."""
    if backend == "sklearn":
        return make_hist_gradient_boosting(
            n_estimators=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
    if backend == "lightgbm":
        model = make_lightgbm(
            n_estimators=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        model.name = "gradient_boosting"
        model._params = {"backend": "lightgbm", **model._params}
        return model
    raise ValueError("gradient_boosting backend must be 'sklearn' or 'lightgbm'")


def make_small_mlp(
    hidden_layer_sizes: tuple[int, ...] | list[int] = (64, 32),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    batch_size: int = 256,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 42,
) -> SklearnModel:
    if not isinstance(hidden_layer_sizes, tuple | list) or not 1 <= len(hidden_layer_sizes) <= 2:
        raise ValueError("hidden_layer_sizes must contain one or two layers")
    layers = tuple(
        _bounded_int("hidden layer width", width, minimum=1, maximum=256)
        for width in hidden_layer_sizes
    )
    params: dict[str, Any] = {
        "hidden_layer_sizes": layers,
        "activation": "relu",
        "solver": "adam",
        "alpha": _positive_float("alpha", alpha, allow_zero=True),
        "learning_rate_init": _positive_float("learning_rate_init", learning_rate_init),
        "batch_size": _bounded_int("batch_size", batch_size, minimum=1, maximum=4_096),
        "max_iter": _bounded_int("max_iter", max_iter, minimum=1, maximum=1_000),
        "tol": _positive_float("tol", tol),
        "shuffle": False,
        "early_stopping": False,
        "random_state": _bounded_int("random_state", random_state, minimum=0, maximum=2**32 - 1),
    }
    return SklearnModel(
        MLPRegressor(**params),
        "small_mlp",
        scale=True,
        params=params,
        backend="sklearn.neural_network.MLPRegressor",
        iteration_limit=max_iter,
        seed=random_state,
        requires_convergence=True,
    )
