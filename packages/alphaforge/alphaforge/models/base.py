"""Unified, typed model contract for every alpha model (SF-S2-MR4/MR5).

Every model — sklearn wrapper, torch sequence model, ensemble, or naive
baseline — subclasses :class:`AlphaModel` and therefore shares one contract:

* ``fit(X, y)`` / ``predict(X)`` (abstract), plus optional ``predict_proba``,
  ``predict_uncertainty``, and ``feature_importance``;
* **validated fitted-state semantics** — a model must be fit before it predicts,
  enforced uniformly;
* **feature-schema checks** — the training feature schema is recorded at fit and
  a model's required features are checked at predict;
* immutable, typed training termination diagnostics when a backend exposes them;
* **versioned metadata** via :meth:`AlphaModel.metadata`; and
* **deterministic serialization** via :meth:`AlphaModel.save` / :meth:`load`
  (a stable, sorted-key JSON container for models with JSON-safe state — every
  naive baseline — and a joblib fallback for arbitrary fitted estimators).

The shared fitted-state and schema behaviour is installed once, in
:meth:`AlphaModel.__init_subclass__`, which wraps each subclass's own ``fit`` and
``predict``. A re-entrancy guard (``_fitting``) means a model that calls its own
``predict`` *during* ``fit`` is never tripped by the not-fitted check.

Errors are typed (:class:`ModelError` and subclasses) so callers can react to a
specific failure rather than parsing a message.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

#: Semantic version of the model contract; bumped when the interface changes.
CONTRACT_VERSION = "1.1.0"

_JSON_FORMAT = "alphaforge-model/json"
_JSON_FORMAT_VERSION = 1
_MAX_MODEL_ARTIFACT_BYTES = 512 * 1024 * 1024


class ModelError(Exception):
    """Base class for all model-contract errors."""


class NotFittedError(ModelError):
    """Raised when predict/serialization is attempted before ``fit``."""


class FeatureSchemaError(ModelError):
    """Raised when an input frame violates the model's feature schema."""


class InvalidLabelError(ModelError):
    """Raised when the target labels are unsupported for the model's task."""


class ProbabilityNotSupportedError(ModelError):
    """Raised when ``predict_proba`` is called on a non-probabilistic model."""


@dataclass(frozen=True)
class TrainingDiagnostics:
    """Immutable termination evidence for one fitted model.

    ``status`` distinguishes numerical convergence from algorithms that simply
    complete a declared finite budget. Tree ensembles, for example, do not have
    a convergence tolerance and therefore report ``completed`` rather than
    manufacturing a convergence claim.
    """

    backend: str
    status: Literal["converged", "completed", "max_iterations", "warning"]
    iterations: int | None
    iteration_limit: int | None
    seed: int | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible record."""
        return {
            "backend": self.backend,
            "status": self.status,
            "iterations": self.iterations,
            "iteration_limit": self.iteration_limit,
            "seed": self.seed,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable, versioned description of a model and its fitted state."""

    name: str
    task: str  # "regression" | "classification"
    contract_version: str
    fitted: bool
    n_features: int | None
    feature_names: tuple[str, ...] | None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (deterministic under ``sort_keys``)."""
        return {
            "name": self.name,
            "task": self.task,
            "contract_version": self.contract_version,
            "fitted": self.fitted,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names) if self.feature_names is not None else None,
            "params": dict(self.params),
        }


def _wrap_fit(fit_impl: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fit_impl)
    def wrapper(self: AlphaModel, X: pd.DataFrame, y: pd.Series, *args: Any, **kwargs: Any) -> Any:
        self._validate_fit_inputs(X, y)
        # A failed refit must never leave a partially mutated model marked as
        # usable. Invalidate first and publish fitted state only after success.
        self._is_fitted = False
        self._feature_names = None
        self._fitting = True
        try:
            result = fit_impl(self, X, y, *args, **kwargs)
        finally:
            self._fitting = False
        if result is not self:
            raise ModelError(f"{self.name}.fit must return self")
        self._record_fit_schema(X)
        self._is_fitted = True
        return result

    wrapper.__af_contract_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_predict(predict_impl: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(predict_impl)
    def wrapper(self: AlphaModel, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        if not self._fitting:
            self._ensure_fitted()
            self._validate_predict_features(X)
        elif not isinstance(X, pd.DataFrame):
            raise FeatureSchemaError("X must be a pandas DataFrame")
        prediction = predict_impl(self, X, *args, **kwargs)
        return self._validate_prediction_output(prediction, expected_rows=len(X))

    wrapper.__af_contract_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_optional_prediction(
    predict_impl: Callable[..., Any], *, probability: bool
) -> Callable[..., Any]:
    @functools.wraps(predict_impl)
    def wrapper(self: AlphaModel, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        self._ensure_fitted()
        self._validate_predict_features(X)
        prediction = predict_impl(self, X, *args, **kwargs)
        if prediction is None and not probability:
            return None
        values = self._validate_prediction_output(prediction, expected_rows=len(X))
        if probability and ((values < 0.0).any() or (values > 1.0).any()):
            raise ModelError(f"{self.name}.predict_proba returned values outside [0, 1]")
        return values

    wrapper.__af_contract_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def _wrap_feature_importance(importance_impl: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(importance_impl)
    def wrapper(self: AlphaModel, *args: Any, **kwargs: Any) -> Any:
        self._ensure_fitted()
        return importance_impl(self, *args, **kwargs)

    wrapper.__af_contract_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


class AlphaModel(ABC):
    """A model mapping a feature matrix to expected forward returns.

    X is a DataFrame of numeric features; rows may carry a (date, symbol)
    MultiIndex, which sequence models use to build causal windows. Tabular
    models simply ignore the index.
    """

    name: str = "alpha_model"
    #: "regression" (predict expected returns) or "classification" (predict_proba).
    task: str = "regression"
    #: Sequence models set this True so the driver attaches a (date, symbol) index.
    needs_sequence_index: bool = False
    #: True for models whose prediction ignores the feature columns (constants).
    feature_agnostic: bool = False
    #: Rule-based models may require original, untransformed semantic features.
    requires_raw_features: bool = False

    # Fitted-state, managed by the fit/predict wrappers. Class-level defaults act
    # as the "not fitted" state until the first fit sets instance attributes.
    _is_fitted: bool = False
    _fitting: bool = False
    _feature_names: tuple[str, ...] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        wrappers = (
            ("fit", _wrap_fit),
            ("predict", _wrap_predict),
            (
                "predict_proba",
                functools.partial(_wrap_optional_prediction, probability=True),
            ),
            (
                "predict_uncertainty",
                functools.partial(_wrap_optional_prediction, probability=False),
            ),
            ("feature_importance", _wrap_feature_importance),
        )
        for attr, wrap in wrappers:
            impl = cls.__dict__.get(attr)
            if impl is not None and not getattr(impl, "__af_contract_wrapped__", False):
                setattr(cls, attr, wrap(impl))

    # ----- required interface ------------------------------------------------
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> AlphaModel: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    # ----- optional interface (typed defaults) -------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of an up-move for classification models.

        Regression models do not support probabilities and raise
        :class:`ProbabilityNotSupportedError`.
        """
        self._ensure_fitted()
        self._validate_predict_features(X)
        raise ProbabilityNotSupportedError(
            f"{self.name} is a {self.task} model and does not implement predict_proba"
        )

    def predict_uncertainty(self, X: pd.DataFrame) -> np.ndarray | None:
        """Optional per-row predictive uncertainty (standard deviation)."""
        self._ensure_fitted()
        self._validate_predict_features(X)
        return None

    def feature_importance(self) -> pd.Series | None:
        """Optional per-feature importance (higher = more important)."""
        self._ensure_fitted()
        return None

    def training_diagnostics(self) -> TrainingDiagnostics | None:
        """Return fitted termination evidence when the backend exposes it."""
        self._ensure_fitted()
        return None

    def get_params(self) -> dict[str, Any]:
        """JSON-safe constructor parameters used to rebuild the model."""
        return {}

    def required_features(self) -> tuple[str, ...] | None:
        """Features that must be present at predict time.

        ``None`` means "the full training schema" (recorded at fit). Models that
        consume only specific columns override this to return just those.
        """
        return None

    # ----- fitted-state and schema helpers -----------------------------------
    def _validate_fit_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not isinstance(X, pd.DataFrame):
            raise FeatureSchemaError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise InvalidLabelError("y must be a pandas Series")
        if len(X) != len(y):
            raise FeatureSchemaError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise FeatureSchemaError("cannot fit on an empty training set")
        if not X.index.equals(y.index):
            raise FeatureSchemaError("X and y indices must be identical and in the same order")
        if not X.columns.is_unique:
            raise FeatureSchemaError("feature columns must be unique")
        if any(not isinstance(column, str) or not column for column in X.columns):
            raise FeatureSchemaError("feature columns must be non-empty strings")
        try:
            x_values = X.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise FeatureSchemaError("all feature columns must be numeric") from exc
        if np.isinf(x_values).any():
            raise FeatureSchemaError("features contain non-finite (inf) values")
        try:
            y_values = y.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise InvalidLabelError("labels must be numeric") from exc
        if not np.isfinite(y_values).all():
            raise InvalidLabelError("labels contain non-finite (NaN or inf) values")

    def _record_fit_schema(self, X: pd.DataFrame) -> None:
        if isinstance(X, pd.DataFrame):
            self._feature_names = tuple(str(c) for c in X.columns)

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(f"{self.name} must be fit before this operation")

    def _validate_predict_features(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise FeatureSchemaError("X must be a pandas DataFrame")
        if not X.columns.is_unique:
            raise FeatureSchemaError("feature columns must be unique")
        if self.feature_agnostic:
            return
        required = self.required_features()
        if required is None:
            required = self._feature_names
        if required is None:
            return
        missing = [column for column in required if column not in X.columns]
        if missing:
            raise FeatureSchemaError(f"{self.name}: missing required features {missing}")
        try:
            values = X.loc[:, list(required)].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise FeatureSchemaError(f"{self.name}: required features must be numeric") from exc
        if np.isinf(values).any():
            raise FeatureSchemaError(f"{self.name}: required features contain infinite values")

    def _validate_prediction_output(self, prediction: Any, *, expected_rows: int) -> np.ndarray:
        """Return a finite one-dimensional prediction vector or fail closed."""
        try:
            values = np.asarray(prediction, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ModelError(f"{self.name}.predict returned non-numeric values") from exc
        if values.ndim != 1 or values.shape[0] != expected_rows:
            raise ModelError(
                f"{self.name}.predict returned shape {values.shape}; expected ({expected_rows},)"
            )
        if not np.isfinite(values).all():
            raise ModelError(f"{self.name}.predict returned non-finite values")
        return values

    # ----- metadata and serialization ----------------------------------------
    def metadata(self) -> ModelMetadata:
        """Return this model's versioned metadata."""
        return ModelMetadata(
            name=self.name,
            task=self.task,
            contract_version=CONTRACT_VERSION,
            fitted=self._is_fitted,
            n_features=len(self._feature_names) if self._feature_names is not None else None,
            feature_names=self._feature_names,
            params=self.get_params(),
        )

    def _fitted_state(self) -> dict[str, Any] | None:
        """JSON-safe fitted state, or ``None`` to use the joblib fallback.

        Naive baselines override this to return a small, deterministic dict.
        """
        return None

    def _load_fitted_state(self, state: dict[str, Any]) -> None:  # noqa: B027
        """Restore fitted state produced by :meth:`_fitted_state`.

        An intentional no-op default: models with no JSON state (or that use the
        joblib fallback) need not override it.
        """

    def save(self, path: str | Path) -> None:
        """Serialize the model to ``path``.

        Models with JSON-safe state (every naive baseline) are written as a
        deterministic, sorted-key JSON container so two saves of an
        equally-fitted model are byte-identical. Other models fall back to
        joblib, which round-trips but is not byte-deterministic.
        """
        self._ensure_fitted()
        path = Path(path)
        if not path.parent.is_dir():
            raise ModelError(f"model destination directory does not exist: {path.parent}")
        state = self._fitted_state()
        if state is not None:
            container = {
                "format": _JSON_FORMAT,
                "format_version": _JSON_FORMAT_VERSION,
                "metadata": self.metadata().to_dict(),
                "state": state,
            }
            try:
                payload = (json.dumps(container, indent=2, sort_keys=True) + "\n").encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ModelError(f"{self.name} produced non-JSON-safe model state") from exc
            self._atomic_write(path, payload)
        else:
            import joblib

            temporary = self._temporary_path(path)
            try:
                joblib.dump(self, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> AlphaModel:
        """Load a model saved by :meth:`save`.

        JSON baseline artifacts are schema-validated and safe to load by
        default. Binary joblib artifacts can execute code during deserialization
        and therefore require ``trusted=True`` at an explicit trust boundary.
        """
        path = Path(path)
        try:
            artifact_size = path.stat().st_size
        except OSError as exc:
            raise ModelError(f"cannot inspect model artifact {path}") from exc
        if artifact_size <= 0 or artifact_size > _MAX_MODEL_ARTIFACT_BYTES:
            raise ModelError(
                f"model artifact size must be in [1, {_MAX_MODEL_ARTIFACT_BYTES}] bytes"
            )
        with open(path, "rb") as handle:
            prefix = handle.read(64).lstrip()
        if prefix.startswith(b"{"):
            try:
                container = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ModelError(f"invalid JSON model artifact: {path}") from exc
            cls._validate_json_container(container)
            from alphaforge.models.registry import create_model  # lazy: avoid import cycle

            meta = container["metadata"]
            try:
                model = create_model(meta["name"], **meta["params"])
                model._load_fitted_state(container["state"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelError(
                    "model artifact contains invalid constructor or fitted state"
                ) from exc
            names = meta["feature_names"]
            model._feature_names = tuple(names) if names is not None else None
            model._is_fitted = True
            return model
        if not trusted:
            raise ModelError(
                "refusing to deserialize an executable binary model artifact; "
                "pass trusted=True only for a verified, trusted artifact"
            )
        import joblib

        try:
            loaded = joblib.load(path)
        except Exception as exc:
            raise ModelError(f"could not deserialize trusted model artifact {path}") from exc
        if not isinstance(loaded, AlphaModel):
            raise ModelError(f"{path} did not contain an AlphaModel")
        loaded._ensure_fitted()
        return loaded

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        return Path(raw_path)

    @classmethod
    def _atomic_write(cls, path: Path, payload: bytes) -> None:
        temporary = cls._temporary_path(path)
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_json_container(container: Any) -> None:
        if not isinstance(container, dict):
            raise ModelError("JSON model artifact must be an object")
        if container.get("format") != _JSON_FORMAT:
            raise ModelError("unsupported JSON model artifact format")
        if container.get("format_version") != _JSON_FORMAT_VERSION:
            raise ModelError("unsupported JSON model artifact version")
        metadata = container.get("metadata")
        state = container.get("state")
        if not isinstance(metadata, dict) or not isinstance(state, dict):
            raise ModelError("JSON model metadata and state must be objects")
        if metadata.get("contract_version") != CONTRACT_VERSION:
            raise ModelError("model artifact contract version is incompatible")
        if metadata.get("fitted") is not True:
            raise ModelError("model artifact is not fitted")
        if metadata.get("task") not in {"regression", "classification"}:
            raise ModelError("model artifact task is invalid")
        if not isinstance(metadata.get("name"), str) or not metadata["name"]:
            raise ModelError("model artifact name is invalid")
        if not isinstance(metadata.get("params"), dict):
            raise ModelError("model artifact params must be an object")
        names = metadata.get("feature_names")
        if names is not None and (
            not isinstance(names, list)
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ModelError("model artifact feature_names are invalid")
        expected_count = len(names) if names is not None else None
        if metadata.get("n_features") != expected_count:
            raise ModelError("model artifact feature count does not match feature_names")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self._is_fitted})"
