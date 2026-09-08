"""Typed, leakage-safe contracts for learned feature representations.

The representation boundary is deliberately narrower than the model boundary:
``fit`` accepts features and temporal identities only.  Labels cannot be passed
to an unsupervised representation by accident, and every learned transform
publishes an immutable identity tied to its configuration, fitted state, input
schema, and exact training sample.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

REPRESENTATION_SCHEMA_VERSION = "1.0.0"
MAX_HARD_SAMPLES = 1_000_000
MAX_HARD_FEATURES = 4_096
MAX_HARD_TENSOR_BYTES = 2 * 1024**3
MAX_HARD_PARAMETERS = 50_000_000

FloatArray = NDArray[np.float64]
RepresentationKind = Literal[
    "raw",
    "pca",
    "incremental_pca",
    "robust_pca",
    "dense_autoencoder",
    "sequence_autoencoder",
    "denoising_autoencoder",
    "variational_autoencoder",
    "contrastive_timeseries",
]
NeuralRepresentationKind = Literal[
    "dense_autoencoder",
    "sequence_autoencoder",
    "denoising_autoencoder",
    "variational_autoencoder",
    "contrastive_timeseries",
]
StoppingReason = Literal[
    "closed_form",
    "max_epochs",
    "validation_patience",
]

REPRESENTATION_KINDS: tuple[RepresentationKind, ...] = (
    "raw",
    "pca",
    "incremental_pca",
    "robust_pca",
    "dense_autoencoder",
    "sequence_autoencoder",
    "denoising_autoencoder",
    "variational_autoencoder",
    "contrastive_timeseries",
)
NEURAL_REPRESENTATION_KINDS = frozenset(
    {
        "dense_autoencoder",
        "sequence_autoencoder",
        "denoising_autoencoder",
        "variational_autoencoder",
        "contrastive_timeseries",
    }
)


class RepresentationError(ValueError):
    """Base class for representation contract violations."""


class RepresentationNotFittedError(RepresentationError):
    """Raised when inference is attempted before fitting."""


class RepresentationSchemaError(RepresentationError):
    """Raised when an input batch violates its declared schema or alignment."""


class RepresentationResourceError(RepresentationError):
    """Raised before a representation would exceed a predeclared resource bound."""


class RepresentationCapabilityError(RepresentationError):
    """Raised when a representation does not implement a requested capability."""


class EmbeddingCollapseError(RepresentationError):
    """Raised when fitted embeddings have no material variation."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RepresentationError("identity payload must be finite JSON") from exc


def _readonly_float_array(values: object, *, name: str, dimensions: int) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RepresentationSchemaError(f"{name} must be numeric") from exc
    if array.ndim != dimensions:
        raise RepresentationSchemaError(f"{name} must be {dimensions}-dimensional")
    if not np.isfinite(array).all():
        raise RepresentationSchemaError(f"{name} must contain only finite values")
    result = np.ascontiguousarray(array.copy())
    result.setflags(write=False)
    return result


def array_sha256(values: object) -> str:
    """Return a shape- and dtype-bound SHA-256 digest for a numeric array."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_identity(payload: object, arrays: tuple[object, ...] = ()) -> str:
    """Hash canonical metadata and ordered numeric learned state."""

    digest = hashlib.sha256(_canonical_json(payload))
    for array in arrays:
        digest.update(array_sha256(array).encode("ascii"))
    return digest.hexdigest()


def named_seed(root_seed: int, name: str) -> int:
    """Derive a stable seed independent of candidate iteration order."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise RepresentationError("root_seed must be a non-negative integer")
    if not isinstance(name, str) or not name.strip():
        raise RepresentationError("seed stream name must be non-empty")
    payload = f"{root_seed}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def canonicalize_component_signs(components: object) -> FloatArray:
    """Resolve the arbitrary sign of each component by its largest loading.

    SVD eigenvectors are equivalent under sign reversal.  Choosing the sign of
    the largest-magnitude loading makes serialized state and transformed scores
    deterministic without changing the represented subspace.
    """

    matrix = _readonly_float_array(components, name="components", dimensions=2).copy()
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise RepresentationSchemaError("components must be non-empty")
    for row in matrix:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    matrix.setflags(write=False)
    return matrix


def subspace_projection(components: object) -> FloatArray:
    """Return the orthogonal projector for a full-row-rank component basis."""

    matrix = _readonly_float_array(components, name="components", dimensions=2)
    if matrix.shape[0] == 0 or matrix.shape[0] > matrix.shape[1]:
        raise RepresentationSchemaError("components must have shape [latent, features]")
    gram = matrix @ matrix.T
    try:
        inverse = np.linalg.pinv(gram, hermitian=True)
    except np.linalg.LinAlgError as exc:
        raise RepresentationError("component subspace is numerically singular") from exc
    projection = matrix.T @ inverse @ matrix
    projection = np.ascontiguousarray((projection + projection.T) / 2.0)
    projection.setflags(write=False)
    return projection


def subspace_fingerprint(components: object, *, decimals: int = 12) -> str:
    """Return a sign- and rotation-invariant identity for a component subspace."""

    if not 6 <= decimals <= 15:
        raise RepresentationError("subspace fingerprint decimals must be in [6, 15]")
    projection = np.ascontiguousarray(np.round(subspace_projection(components), decimals=decimals))
    # IEEE-754 distinguishes +0.0 and -0.0 at the byte level.  Rotation can
    # leave either signed zero after rounding even though both projections are
    # mathematically identical, so normalize that serialization ambiguity.
    projection[projection == 0.0] = 0.0
    return stable_identity(
        {"kind": "orthogonal_projection", "decimals": decimals},
        (projection,),
    )


def subspace_distance(first: object, second: object) -> float:
    """Return the normalized Frobenius distance between two component subspaces."""

    left = subspace_projection(first)
    right = subspace_projection(second)
    if left.shape != right.shape:
        raise RepresentationSchemaError("component subspaces must share an input dimension")
    denominator = max(float(np.sqrt(np.trace(left))), np.finfo(float).eps)
    return float(np.linalg.norm(left - right, ord="fro") / denominator)


@dataclass(frozen=True)
class RepresentationBatch:
    """Finite feature rows with exact temporal and entity alignment.

    ``values`` is copied and made read-only.  Rows must be ordered by date and
    uniquely identified by ``(date, symbol)``.  The contract intentionally has
    no target field.
    """

    values: FloatArray
    dates: tuple[str, ...]
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, name="representation values", dimensions=2)
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise RepresentationSchemaError("representation batch must be non-empty")
        if len(self.dates) != values.shape[0] or len(self.symbols) != values.shape[0]:
            raise RepresentationSchemaError("dates and symbols must align one-for-one with rows")
        if len(self.feature_names) != values.shape[1]:
            raise RepresentationSchemaError("feature names must align one-for-one with columns")
        if len(set(self.feature_names)) != len(self.feature_names) or any(
            not isinstance(name, str) or not name.strip() for name in self.feature_names
        ):
            raise RepresentationSchemaError("feature names must be unique non-empty strings")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in self.symbols):
            raise RepresentationSchemaError("symbols must be non-empty strings")
        try:
            date_index = pd.DatetimeIndex(pd.to_datetime(self.dates, errors="raise"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RepresentationSchemaError("dates must be valid timestamps") from exc
        if date_index.hasnans or not date_index.is_monotonic_increasing:
            raise RepresentationSchemaError("dates must be finite and monotonically non-decreasing")
        identities = tuple(zip(date_index.asi8.tolist(), self.symbols, strict=True))
        if len(set(identities)) != len(identities):
            raise RepresentationSchemaError("(date, symbol) identities must be unique")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dates", tuple(timestamp.isoformat() for timestamp in date_index))
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        dates: Iterable[object],
        symbols: Iterable[object],
    ) -> RepresentationBatch:
        """Create a batch without retaining mutable caller-owned storage."""

        if not isinstance(frame, pd.DataFrame):
            raise RepresentationSchemaError("frame must be a pandas DataFrame")
        return cls(
            values=frame.to_numpy(dtype=float),
            dates=tuple(str(value) for value in dates),
            symbols=tuple(str(value) for value in symbols),
            feature_names=tuple(str(value) for value in frame.columns),
        )

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])

    @property
    def tensor_bytes(self) -> int:
        return int(self.values.nbytes)

    @property
    def start(self) -> str:
        return self.dates[0]

    @property
    def end(self) -> str:
        return self.dates[-1]

    @property
    def batch_id(self) -> str:
        return stable_identity(
            {
                "dates": self.dates,
                "symbols": self.symbols,
                "feature_names": self.feature_names,
            },
            (self.values,),
        )

    def take(self, selector: object) -> RepresentationBatch:
        """Return an aligned immutable subset, preserving chronological order."""

        indices = np.asarray(selector)
        if indices.ndim != 1:
            raise RepresentationSchemaError("batch selector must be one-dimensional")
        try:
            selected_values = self.values[indices]
            selected_dates = tuple(np.asarray(self.dates, dtype=object)[indices].tolist())
            selected_symbols = tuple(np.asarray(self.symbols, dtype=object)[indices].tolist())
        except (IndexError, TypeError) as exc:
            raise RepresentationSchemaError("batch selector is invalid") from exc
        if len(selected_dates) == 0:
            raise RepresentationSchemaError("batch selector must retain at least one row")
        return RepresentationBatch(
            values=selected_values,
            dates=selected_dates,
            symbols=selected_symbols,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True)
class RepresentationConfig:
    """Bounded policy shared by linear and neural representations."""

    kind: RepresentationKind
    latent_dim: int = 4
    seed: int = 0
    hidden_dim: int = 16
    sequence_length: int = 8
    batch_size: int = 64
    incremental_batch_size: int = 128
    max_epochs: int = 20
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.20
    corruption_std: float = 0.10
    vae_beta: float = 0.01
    contrastive_temperature: float = 0.20
    augmentation_jitter_std: float = 0.03
    augmentation_scale_std: float = 0.05
    augmentation_mask_probability: float = 0.10
    min_embedding_variance: float = 1e-10
    max_samples: int = 100_000
    max_features: int = 512
    max_tensor_bytes: int = 512 * 1024**2
    max_parameters: int = 2_000_000

    def __post_init__(self) -> None:
        if self.kind not in REPRESENTATION_KINDS:
            raise RepresentationError(f"unsupported representation kind {self.kind!r}")
        bounded_integers = (
            ("latent_dim", self.latent_dim, 1, self.max_features),
            ("hidden_dim", self.hidden_dim, 1, 16_384),
            ("sequence_length", self.sequence_length, 2, 4_096),
            ("batch_size", self.batch_size, 2, 65_536),
            ("incremental_batch_size", self.incremental_batch_size, 2, 100_000),
            ("max_epochs", self.max_epochs, 1, 10_000),
            ("patience", self.patience, 1, self.max_epochs),
            ("max_samples", self.max_samples, 2, MAX_HARD_SAMPLES),
            ("max_features", self.max_features, 1, MAX_HARD_FEATURES),
            ("max_tensor_bytes", self.max_tensor_bytes, 1, MAX_HARD_TENSOR_BYTES),
            ("max_parameters", self.max_parameters, 1, MAX_HARD_PARAMETERS),
        )
        for integer_name, integer_value, minimum, maximum in bounded_integers:
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or not minimum <= integer_value <= maximum
            ):
                raise RepresentationError(
                    f"{integer_name} must be an integer in [{minimum}, {maximum}]"
                )
        if self.hidden_dim < self.latent_dim:
            raise RepresentationError("hidden_dim must be at least latent_dim")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise RepresentationError("seed must be a non-negative integer")
        positive: tuple[tuple[str, float], ...] = (
            ("learning_rate", self.learning_rate),
            ("contrastive_temperature", self.contrastive_temperature),
            ("min_embedding_variance", self.min_embedding_variance),
        )
        non_negative: tuple[tuple[str, float], ...] = (
            ("weight_decay", self.weight_decay),
            ("corruption_std", self.corruption_std),
            ("vae_beta", self.vae_beta),
            ("augmentation_jitter_std", self.augmentation_jitter_std),
            ("augmentation_scale_std", self.augmentation_scale_std),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise RepresentationError(f"{name} must be finite and positive")
        for name, value in non_negative:
            if not math.isfinite(value) or value < 0.0:
                raise RepresentationError(f"{name} must be finite and non-negative")
        for name, value in (
            ("validation_fraction", self.validation_fraction),
            ("augmentation_mask_probability", self.augmentation_mask_probability),
        ):
            if not math.isfinite(value) or not 0.0 < value < 0.5:
                raise RepresentationError(f"{name} must be finite and in (0, 0.5)")

    @property
    def config_id(self) -> str:
        return stable_identity({"schema_version": REPRESENTATION_SCHEMA_VERSION, **asdict(self)})


@dataclass(frozen=True)
class RepresentationState:
    """Immutable fitted-state and resource evidence without learned weights."""

    schema_version: str
    kind: RepresentationKind
    config_id: str
    state_id: str
    fit_batch_id: str
    input_features: tuple[str, ...]
    output_features: tuple[str, ...]
    fit_start: str
    fit_end: str
    fit_rows: int
    validation_rows: int
    iterations: int
    converged: bool
    stopping_reason: StoppingReason
    reconstruction_mse: float | None
    embedding_variance: float
    parameter_count: int
    parameter_bytes: int
    fit_wall_seconds: float
    fit_cpu_seconds: float
    device: Literal["cpu"]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != REPRESENTATION_SCHEMA_VERSION:
            raise RepresentationError("unsupported representation state schema version")
        if self.kind not in REPRESENTATION_KINDS:
            raise RepresentationError("representation state kind is unsupported")
        for name in ("config_id", "state_id", "fit_batch_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise RepresentationError(f"{name} must be a SHA-256 hex identity")
        if not self.input_features or not self.output_features:
            raise RepresentationError("representation schemas must be non-empty")
        if self.fit_rows < 2 or not 0 <= self.validation_rows < self.fit_rows:
            raise RepresentationError("representation fit/validation row counts are invalid")
        if self.iterations < 1:
            raise RepresentationError("representation iterations must be positive")
        if self.parameter_count < 0 or self.parameter_bytes < 0:
            raise RepresentationError("representation resource counts must be non-negative")
        finite_values = (
            self.embedding_variance,
            self.fit_wall_seconds,
            self.fit_cpu_seconds,
        )
        if not np.isfinite(finite_values).all() or any(value < 0.0 for value in finite_values):
            raise RepresentationError("representation state measurements must be finite")
        if self.embedding_variance < 0.0:
            raise RepresentationError("embedding variance must be non-negative")
        if self.reconstruction_mse is not None and (
            not math.isfinite(self.reconstruction_mse) or self.reconstruction_mse < 0.0
        ):
            raise RepresentationError("reconstruction MSE must be finite and non-negative")
        if self.device != "cpu":
            raise RepresentationError("SF-S3-MR8 reference states must report CPU execution")

    def to_dict(self) -> dict[str, Any]:
        """Return bounded JSON-safe public metadata."""

        payload = asdict(self)
        payload["input_features"] = list(self.input_features)
        payload["output_features"] = list(self.output_features)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class RepresentationOutput:
    """Immutable finite embeddings aligned exactly to their source batch."""

    values: FloatArray
    dates: tuple[str, ...]
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    state_id: str

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, name="embedding values", dimensions=2)
        if values.shape != (len(self.dates), len(self.feature_names)):
            raise RepresentationSchemaError("embedding shape does not match its row/column schema")
        if len(self.symbols) != values.shape[0]:
            raise RepresentationSchemaError("embedding symbols do not align with rows")
        if len(self.state_id) != 64:
            raise RepresentationSchemaError("embedding state_id must be a SHA-256 identity")
        object.__setattr__(self, "values", values)

    def to_frame(self) -> pd.DataFrame:
        """Return a detached frame without exposing mutable fitted state."""

        return pd.DataFrame(self.values.copy(), columns=self.feature_names)


def fit_standardizer(values: object) -> tuple[FloatArray, FloatArray]:
    """Fit a finite mean/scale transform on training rows only."""

    array = _readonly_float_array(values, name="standardizer values", dimensions=2)
    center = np.mean(array, axis=0)
    scale = np.std(array, axis=0, ddof=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return (
        _readonly_float_array(center, name="standardizer center", dimensions=1),
        _readonly_float_array(scale, name="standardizer scale", dimensions=1),
    )


def apply_standardizer(values: object, center: object, scale: object) -> FloatArray:
    """Apply an already-fitted standardizer without mutation or refitting."""

    array = _readonly_float_array(values, name="standardizer input", dimensions=2)
    center_array = _readonly_float_array(center, name="standardizer center", dimensions=1)
    scale_array = _readonly_float_array(scale, name="standardizer scale", dimensions=1)
    if array.shape[1] != len(center_array) or center_array.shape != scale_array.shape:
        raise RepresentationSchemaError("standardizer schema mismatch")
    if (scale_array <= 0.0).any():
        raise RepresentationSchemaError("standardizer scale must be positive")
    return _readonly_float_array(
        (array - center_array) / scale_array,
        name="standardized values",
        dimensions=2,
    )


def invert_standardizer(values: object, center: object, scale: object) -> FloatArray:
    """Invert an already-fitted standardizer."""

    array = _readonly_float_array(values, name="standardizer inverse input", dimensions=2)
    center_array = _readonly_float_array(center, name="standardizer center", dimensions=1)
    scale_array = _readonly_float_array(scale, name="standardizer scale", dimensions=1)
    if array.shape[1] != len(center_array) or center_array.shape != scale_array.shape:
        raise RepresentationSchemaError("standardizer inverse schema mismatch")
    return _readonly_float_array(
        array * scale_array + center_array,
        name="reconstructed values",
        dimensions=2,
    )


def embedding_variance(values: object) -> float:
    """Return total population variance across embedding coordinates."""

    array = _readonly_float_array(values, name="embedding diagnostics", dimensions=2)
    if len(array) < 2:
        return 0.0
    return float(np.var(array, axis=0, ddof=0).sum())


class BaseRepresentation(ABC):
    """Common schema, resource, fitted-state, and inference enforcement."""

    def __init__(self, config: RepresentationConfig) -> None:
        if not isinstance(config, RepresentationConfig):
            raise TypeError("config must be a RepresentationConfig")
        self.config = config
        self.state_: RepresentationState | None = None

    def _validate_fit_batch(self, batch: RepresentationBatch) -> None:
        if not isinstance(batch, RepresentationBatch):
            raise TypeError("fit requires a RepresentationBatch")
        if batch.n_rows > self.config.max_samples:
            raise RepresentationResourceError(
                f"sample count {batch.n_rows} exceeds limit {self.config.max_samples}"
            )
        if batch.n_features > self.config.max_features:
            raise RepresentationResourceError(
                f"feature count {batch.n_features} exceeds limit {self.config.max_features}"
            )
        if batch.tensor_bytes > self.config.max_tensor_bytes:
            raise RepresentationResourceError(
                f"input bytes {batch.tensor_bytes} exceed limit {self.config.max_tensor_bytes}"
            )
        if self.config.latent_dim > min(batch.n_rows, batch.n_features):
            raise RepresentationResourceError(
                "latent_dim cannot exceed either training rows or input features"
            )

    def _validate_inference_batch(self, batch: RepresentationBatch) -> RepresentationState:
        if self.state_ is None:
            raise RepresentationNotFittedError(
                f"{self.config.kind} representation must be fit before inference"
            )
        if not isinstance(batch, RepresentationBatch):
            raise TypeError("inference requires a RepresentationBatch")
        if batch.feature_names != self.state_.input_features:
            raise RepresentationSchemaError("representation inference feature schema mismatch")
        if (
            batch.n_rows > self.config.max_samples
            or batch.tensor_bytes > self.config.max_tensor_bytes
        ):
            raise RepresentationResourceError(
                "representation inference batch exceeds resource policy"
            )
        return self.state_

    def _output(self, values: object, batch: RepresentationBatch) -> RepresentationOutput:
        state = self._validate_inference_batch(batch)
        array = _readonly_float_array(values, name="embedding output", dimensions=2)
        if array.shape != (batch.n_rows, len(state.output_features)):
            raise RepresentationSchemaError("representation returned an invalid embedding shape")
        return RepresentationOutput(
            values=array,
            dates=batch.dates,
            symbols=batch.symbols,
            feature_names=state.output_features,
            state_id=state.state_id,
        )

    @abstractmethod
    def fit(self, batch: RepresentationBatch) -> Self:
        """Fit using feature rows only and return ``self``."""

    @abstractmethod
    def transform(self, batch: RepresentationBatch) -> RepresentationOutput:
        """Transform aligned rows without changing fitted state."""

    def reconstruct(self, batch: RepresentationBatch) -> FloatArray:
        """Reconstruct input features when the representation supports it."""

        self._validate_inference_batch(batch)
        raise RepresentationCapabilityError(
            f"{self.config.kind} representation does not support reconstruction"
        )
