"""Causal covariance and factor-risk contracts for portfolio optimization.

The public objects in this module are immutable, point-in-time records.  They
carry the observation interval and an exact source digest alongside the
covariance so an allocation can be traced to the information that was available
at its decision timestamp.  Estimators use observations strictly before
``as_of``; callers that omit ``as_of`` are explicitly requesting an offline
batch estimate over all supplied observations.

All matrices use periodic-return units.  Annualization belongs at reporting
boundaries and is never mixed into the optimizer's objective.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

MAX_ASSETS = 2_000
MAX_FACTORS = 64
MAX_WINDOW = 10_000
MAX_SOURCE_OBSERVATIONS = 100_000
# At eight bytes per float this caps the numeric payload at about 16 MiB before
# pandas sorting and complete-case copies.  The independent row/column ceilings
# still apply, so neither a long narrow nor a short wide source escapes bounds.
MAX_SOURCE_CELLS = 2_000_000
MIN_OBSERVATIONS = 10
EIGENVALUE_FLOOR = 1e-8
CONDITION_LIMIT = 1e8
# Backward-compatible public name used by the original MR2 test contract.
CONDITION_WARNING = CONDITION_LIMIT
ROUND_OFF_NEGATIVE_LIMIT = 1e-10
MATRIX_RECONCILIATION_RTOL = 1e-9
MATRIX_SYMMETRY_RTOL = 1e-12
SCALAR_DIAGNOSTIC_RTOL = 1e-10
ROUNDOFF_MULTIPLIER = 64.0


class RiskModelError(ValueError):
    """Raised when risk-model input or estimated state is unusable."""


def _finite_float(value: Any, *, name: str) -> float:
    """Return one finite real scalar with a structured boundary error."""
    if isinstance(value, (bool, np.bool_)):
        raise RiskModelError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskModelError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise RiskModelError(f"{name} must be finite")
    return result


def _unit_interval(value: Any, *, name: str) -> float:
    """Return a finite scalar in ``[0, 1]`` without accepting booleans."""
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise RiskModelError(f"{name} must lie in [0, 1]")
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    """Return one finite non-negative scalar."""
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise RiskModelError(f"{name} must be non-negative")
    return result


def _bounded_count(value: Any, *, name: str, maximum: int) -> int:
    """Return one non-negative integer no greater than ``maximum``."""
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise RiskModelError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= maximum:
        raise RiskModelError(f"{name} must lie in [0, {maximum}]")
    return result


def _frobenius_norm(value: FloatArray) -> float:
    """Return a finite Frobenius norm without an avoidable overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        maximum = float(np.max(np.abs(value))) if value.size else 0.0
    if not np.isfinite(maximum):
        return float("inf")
    if maximum == 0.0:
        return 0.0
    return float(maximum * np.linalg.norm(value / maximum, ord="fro"))


def _require_scaled_matrix_close(
    actual: FloatArray,
    expected: FloatArray,
    *,
    name: str,
    relative_tolerance: float,
    operation_scale: float | None = None,
) -> None:
    """Require matrix agreement under relative and round-off-scaled error.

    A fixed absolute tolerance in covariance units lets a decomposition be
    arbitrarily wrong when every variance is small.  The round-off term instead
    scales with the arithmetic that produced the matrices; an ill-scaled
    decomposition whose numerical uncertainty dominates the declared relative
    tolerance is rejected rather than granted a wider evidentiary budget.
    """
    if actual.shape != expected.shape:
        raise RiskModelError(f"{name} matrices have different shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.asarray(actual - expected, dtype=np.float64)
    error = _frobenius_norm(difference)
    if not np.isfinite(error):
        raise RiskModelError(f"{name} produced non-finite reconciliation error")
    # Exact equality remains evidence even when a conservative multiplication
    # error bound would exceed the requested relative tolerance at large n.
    if error == 0.0:
        return
    reference = max(_frobenius_norm(actual), _frobenius_norm(expected))
    if not np.isfinite(reference):
        raise RiskModelError(f"{name} contains a non-finite matrix norm")
    scale = (
        reference
        if operation_scale is None
        else _nonnegative_float(operation_scale, name=f"{name} operation scale")
    )
    dimension = max((*actual.shape, 1))
    roundoff = ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * dimension * scale
    relative_budget = relative_tolerance * reference
    if reference == 0.0:
        if error != 0.0:
            raise RiskModelError(f"{name} does not reconcile at zero scale")
        return
    if roundoff > relative_budget:
        raise RiskModelError(f"{name} is too ill-scaled for reliable reconciliation")
    if error > relative_budget + roundoff:
        relative_error = error / reference
        raise RiskModelError(
            f"{name} does not reconcile (relative Frobenius error {relative_error:.3e})"
        )


def _require_scaled_scalar_close(
    actual: Any,
    expected: float,
    *,
    name: str,
) -> None:
    """Require scalar diagnostics to match without a unit-based absolute floor."""
    supplied = _finite_float(actual, name=name)
    scale = max(abs(supplied), abs(expected))
    if scale == 0.0:
        return
    tolerance = (SCALAR_DIAGNOSTIC_RTOL + ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps) * scale
    if abs(supplied - expected) > tolerance:
        raise RiskModelError(f"{name} does not match covariance")


def _readonly_array(value: Any, *, ndim: int, name: str) -> FloatArray:
    """Return a finite, owned, read-only float64 array of the declared rank."""
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskModelError(f"{name} cannot be converted to float64") from exc
    if array.ndim != ndim:
        raise RiskModelError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise RiskModelError(f"{name} contains non-finite entries")
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    # A flag on an owning ndarray can be reversed with setflags(write=True).
    # Backing the public view with immutable bytes makes that bypass impossible.
    immutable = np.frombuffer(canonical.tobytes(order="C"), dtype=np.float64).reshape(
        canonical.shape
    )
    immutable.setflags(write=False)
    return immutable


def _timestamp(value: pd.Timestamp | str | None, *, name: str) -> pd.Timestamp | None:
    """Normalize one timestamp without discarding timezone information."""
    if value is None:
        return None
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskModelError(f"{name} must be a valid timestamp") from exc
    if pd.isna(result):
        raise RiskModelError(f"{name} must not be NaT")
    return result


def _exposure_timestamps(
    *,
    exposure_vintage: pd.Timestamp | str,
    exposure_available_at: pd.Timestamp | str,
    as_of: pd.Timestamp | str | None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Validate the point-in-time ordering of an exposure snapshot."""
    vintage = _timestamp(exposure_vintage, name="exposure_vintage")
    available = _timestamp(exposure_available_at, name="exposure_available_at")
    decision = _timestamp(as_of, name="as_of")
    if vintage is None or available is None:
        raise RiskModelError("exposure vintage and availability timestamps are required")
    if decision is None:
        raise RiskModelError("as_of is required for a point-in-time factor risk model")
    try:
        if vintage > available:
            raise RiskModelError("exposure_vintage must not follow exposure_available_at")
        if available > decision:
            raise RiskModelError("exposure_available_at must not follow as_of")
    except TypeError as exc:
        raise RiskModelError(
            "exposure vintage, availability, and as_of have incompatible timezone semantics"
        ) from exc
    return vintage, available, decision


def _source_digest(
    values: FloatArray,
    *,
    labels: tuple[str, ...],
    timestamps: pd.Index | None = None,
    extra_arrays: tuple[FloatArray, ...] = (),
) -> str:
    """Hash exact float bytes plus labels/timestamps for provenance identity."""
    digest = hashlib.sha256()
    digest.update(b"alphaforge-risk-source-v1\0")
    for label in labels:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
    if timestamps is not None:
        try:
            for item in timestamps:
                digest.update(pd.Timestamp(item).isoformat().encode("utf-8"))
                digest.update(b"\0")
        except (TypeError, ValueError, OverflowError) as exc:
            raise RiskModelError("source index contains an invalid timestamp") from exc
    for array in (values, *extra_arrays):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RiskModel:
    """Validated positive-definite covariance and point-in-time provenance.

    Args:
        assets: Unique asset labels in covariance order.
        covariance: ``(n, n)`` covariance in squared periodic-return units.
        periods_per_year: Number of periods used only for report annualization.
        min_eigenvalue: Independently verified smallest covariance eigenvalue.
        condition_number: Independently verified spectral condition number.
        ridge_applied: Diagonal stabilization in covariance units.  Only
            round-off-scale repair is permitted and it is always recorded.
        estimator: Stable estimator identifier.
        as_of: Decision boundary.  Estimated observations are strictly earlier.
        window_start/window_end: Inclusive interval actually used.
        n_observations: Complete observations actually used.
        source_hash: SHA-256 over exact source values, labels, and timestamps.

    The covariance is defensively copied and marked read-only.  Construction
    repeats structural and numerical validation, so direct construction cannot
    bypass :func:`validate_covariance`.
    """

    assets: tuple[str, ...]
    covariance: FloatArray
    periods_per_year: int
    min_eigenvalue: float
    condition_number: float
    ridge_applied: float
    estimator: str
    shrinkage_intensity: float | None = None
    estimator_parameters: tuple[tuple[str, str], ...] = ()
    as_of: pd.Timestamp | None = None
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None
    n_observations: int | None = None
    observations_considered: int | None = None
    observations_dropped: int = 0
    dropped_assets: tuple[str, ...] = ()
    source_hash: str | None = None

    def __post_init__(self) -> None:
        try:
            assets = tuple(str(asset) for asset in self.assets)
        except TypeError as exc:
            raise RiskModelError("assets must be an iterable of labels") from exc
        if not assets or any(not asset for asset in assets):
            raise RiskModelError("asset labels must be non-empty")
        if len(assets) > MAX_ASSETS:
            raise RiskModelError(f"risk model exceeds the {MAX_ASSETS}-asset ceiling")
        if len(set(assets)) != len(assets):
            raise RiskModelError("asset labels must be unique")
        if not isinstance(self.periods_per_year, int) or isinstance(self.periods_per_year, bool):
            raise RiskModelError("periods_per_year must be an integer")
        if self.periods_per_year < 1:
            raise RiskModelError("periods_per_year must be at least one")
        if not isinstance(self.estimator, str) or not self.estimator.strip():
            raise RiskModelError("estimator must be a non-empty identifier")

        matrix = _readonly_array(self.covariance, ndim=2, name="covariance")
        size = len(assets)
        if matrix.shape != (size, size):
            raise RiskModelError("covariance must be square over the asset labels")
        _require_scaled_matrix_close(
            matrix,
            matrix.T,
            name="covariance is not symmetric",
            relative_tolerance=MATRIX_SYMMETRY_RTOL,
        )
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
        except np.linalg.LinAlgError as exc:
            raise RiskModelError("covariance eigendecomposition failed") from exc
        smallest = float(eigenvalues[0])
        largest = float(eigenvalues[-1])
        if smallest <= 0.0 or largest <= 0.0:
            raise RiskModelError("covariance must be positive definite")
        condition = float(largest / smallest)
        if not np.isfinite(condition) or condition > CONDITION_LIMIT * (1.0 + 1e-10):
            raise RiskModelError(
                f"covariance condition number {condition:.3e} exceeds {CONDITION_LIMIT:.1e}"
            )
        _require_scaled_scalar_close(
            self.min_eigenvalue,
            smallest,
            name="min_eigenvalue",
        )
        _require_scaled_scalar_close(
            self.condition_number,
            condition,
            name="condition_number",
        )
        ridge_applied = _nonnegative_float(self.ridge_applied, name="ridge_applied")
        shrinkage_intensity = (
            None
            if self.shrinkage_intensity is None
            else _unit_interval(self.shrinkage_intensity, name="shrinkage_intensity")
        )
        try:
            parameters = tuple((str(key), str(value)) for key, value in self.estimator_parameters)
        except (TypeError, ValueError) as exc:
            raise RiskModelError("estimator_parameters must contain (key, value) pairs") from exc
        if parameters != tuple(sorted(parameters)) or len({key for key, _ in parameters}) != len(
            parameters
        ):
            raise RiskModelError("estimator_parameters must have unique keys in sorted order")
        if any(not key or not value for key, value in parameters):
            raise RiskModelError("estimator_parameters keys and values must be non-empty")

        as_of = _timestamp(self.as_of, name="as_of")
        start = _timestamp(self.window_start, name="window_start")
        end = _timestamp(self.window_end, name="window_end")
        if (start is None) != (end is None):
            raise RiskModelError("window_start and window_end must be supplied together")
        try:
            if start is not None and end is not None and end < start:
                raise RiskModelError("window_end precedes window_start")
            if as_of is not None and end is not None and not end < as_of:
                raise RiskModelError("risk-model window must end strictly before as_of")
        except TypeError as exc:
            raise RiskModelError(
                "risk-model timestamps have incompatible timezone semantics"
            ) from exc
        if self.n_observations is not None:
            if not isinstance(self.n_observations, int) or isinstance(self.n_observations, bool):
                raise RiskModelError("n_observations must be an integer")
            if self.n_observations < 1:
                raise RiskModelError("n_observations must be positive")
        if self.observations_considered is not None:
            if not isinstance(self.observations_considered, int) or isinstance(
                self.observations_considered, bool
            ):
                raise RiskModelError("observations_considered must be an integer")
            if self.observations_considered < 1:
                raise RiskModelError("observations_considered must be positive")
        if not isinstance(self.observations_dropped, int) or isinstance(
            self.observations_dropped, bool
        ):
            raise RiskModelError("observations_dropped must be an integer")
        if self.observations_dropped < 0:
            raise RiskModelError("observations_dropped must be non-negative")
        if (self.n_observations is None) != (self.observations_considered is None):
            raise RiskModelError(
                "n_observations and observations_considered must be supplied together"
            )
        if (
            self.n_observations is not None
            and self.observations_considered is not None
            and self.n_observations + self.observations_dropped != self.observations_considered
        ):
            raise RiskModelError("risk-model observation coverage does not reconcile")
        dropped_assets = tuple(str(asset) for asset in self.dropped_assets)
        if len(set(dropped_assets)) != len(dropped_assets) or any(
            not asset for asset in dropped_assets
        ):
            raise RiskModelError("dropped_assets must be unique and non-empty")
        if self.source_hash is not None and (
            not isinstance(self.source_hash, str)
            or (
                len(self.source_hash) != 64
                or any(c not in "0123456789abcdef" for c in self.source_hash)
            )
        ):
            raise RiskModelError("source_hash must be a lowercase SHA-256 digest")

        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "covariance", matrix)
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "window_start", start)
        object.__setattr__(self, "window_end", end)
        object.__setattr__(self, "dropped_assets", dropped_assets)
        object.__setattr__(self, "estimator_parameters", parameters)
        object.__setattr__(self, "ridge_applied", ridge_applied)
        object.__setattr__(self, "shrinkage_intensity", shrinkage_intensity)

    @property
    def ill_conditioned(self) -> bool:
        """Return ``False`` for every valid instance; invalid models are refused."""
        return bool(self.condition_number > CONDITION_LIMIT)

    @property
    def identity(self) -> str:
        """Return a deterministic identity over model values and provenance."""
        digest = hashlib.sha256()
        digest.update(b"alphaforge-risk-model-v2\0")
        digest.update(_source_digest(self.covariance, labels=self.assets).encode("ascii"))
        for value in (
            self.estimator,
            str(self.periods_per_year),
            self.source_hash or "",
            "" if self.as_of is None else self.as_of.isoformat(),
            "" if self.window_start is None else self.window_start.isoformat(),
            "" if self.window_end is None else self.window_end.isoformat(),
            "" if self.n_observations is None else str(self.n_observations),
            "" if self.observations_considered is None else str(self.observations_considered),
            str(self.observations_dropped),
            *self.dropped_assets,
            self.ridge_applied.hex(),
            "" if self.shrinkage_intensity is None else self.shrinkage_intensity.hex(),
            *(f"{key}={value}" for key, value in self.estimator_parameters),
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def volatilities(self) -> FloatArray:
        """Return an owned vector of periodic asset standard deviations."""
        return np.sqrt(np.diag(self.covariance)).copy()

    def portfolio_variance(self, weights: FloatArray) -> float:
        """Return ``w'Σw`` after validating shape and finiteness."""
        try:
            vector = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RiskModelError("weights cannot be converted to float64") from exc
        if vector.shape != (len(self.assets),):
            raise RiskModelError("weights must align with the risk-model assets")
        if not np.isfinite(vector).all():
            raise RiskModelError("weights must be finite")
        variance = float(vector @ self.covariance @ vector)
        if not np.isfinite(variance):
            raise RiskModelError("covariance produced non-finite portfolio variance")
        operation_scale = float(np.abs(vector) @ np.abs(self.covariance) @ np.abs(vector))
        if not np.isfinite(operation_scale):
            raise RiskModelError("portfolio-variance error bound is non-finite")
        roundoff = (
            ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * max(len(vector), 1) * operation_scale
        )
        if variance < -roundoff:
            raise RiskModelError("covariance produced negative portfolio variance")
        return max(variance, 0.0)

    def annualized_volatility(self, weights: FloatArray) -> float:
        """Return annualized ex-ante volatility."""
        return float(np.sqrt(self.portfolio_variance(weights) * self.periods_per_year))

    def to_frame(self) -> pd.DataFrame:
        """Return an owned labelled covariance frame."""
        return pd.DataFrame(self.covariance.copy(), index=self.assets, columns=self.assets)

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe conditioning, coverage, and provenance diagnostics."""
        return {
            "identity": self.identity,
            "estimator": self.estimator,
            "n_assets": len(self.assets),
            "periods_per_year": self.periods_per_year,
            "min_eigenvalue": self.min_eigenvalue,
            "condition_number": self.condition_number,
            "ill_conditioned": self.ill_conditioned,
            "ridge_applied": self.ridge_applied,
            "shrinkage_intensity": self.shrinkage_intensity,
            "estimator_parameters": dict(self.estimator_parameters),
            "as_of": None if self.as_of is None else self.as_of.isoformat(),
            "window_start": None if self.window_start is None else self.window_start.isoformat(),
            "window_end": None if self.window_end is None else self.window_end.isoformat(),
            "n_observations": self.n_observations,
            "observations_considered": self.observations_considered,
            "observations_dropped": self.observations_dropped,
            "complete_observation_fraction": (
                None
                if self.observations_considered is None or self.n_observations is None
                else self.n_observations / self.observations_considered
            ),
            "dropped_assets": list(self.dropped_assets),
            "source_hash": self.source_hash,
        }


def validate_covariance(
    covariance: pd.DataFrame | FloatArray,
    *,
    assets: tuple[str, ...] | None = None,
    periods_per_year: int = 252,
    estimator: str = "supplied",
    allow_ridge: bool = False,
    shrinkage_intensity: float | None = None,
    estimator_parameters: tuple[tuple[str, str], ...] = (),
    as_of: pd.Timestamp | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    n_observations: int | None = None,
    observations_considered: int | None = None,
    observations_dropped: int = 0,
    dropped_assets: tuple[str, ...] = (),
    source_hash: str | None = None,
) -> RiskModel:
    """Validate covariance structure and conditioning without silent repair.

    ``allow_ridge`` permits only round-off-scale stabilization.  A materially
    indefinite matrix, a singular supplied model, or a model above the declared
    condition limit is rejected with :class:`RiskModelError`.
    """
    if not isinstance(allow_ridge, bool):
        raise RiskModelError("allow_ridge must be a boolean")
    if isinstance(covariance, pd.DataFrame):
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise RiskModelError(f"covariance must be square, got shape {covariance.shape}")
        if covariance.shape[0] == 0:
            raise RiskModelError("covariance must cover at least one asset")
        if covariance.shape[0] > MAX_ASSETS:
            raise RiskModelError(f"covariance exceeds the {MAX_ASSETS}-asset ceiling")
        if covariance.index.has_duplicates or covariance.columns.has_duplicates:
            raise RiskModelError("covariance labels must be unique")
        if list(covariance.index) != list(covariance.columns):
            raise RiskModelError("covariance frame index and columns must match in order")
        inferred = tuple(str(column) for column in covariance.columns)
        if assets is not None:
            try:
                supplied_assets = tuple(str(asset) for asset in assets)
            except TypeError as exc:
                raise RiskModelError("assets must be an iterable of labels") from exc
            if supplied_assets != inferred:
                raise RiskModelError("supplied assets do not match covariance labels")
        assets = inferred
        try:
            matrix = np.array(covariance.to_numpy(), dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RiskModelError("covariance cannot be converted to float64") from exc
    else:
        if isinstance(covariance, np.ndarray):
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise RiskModelError(f"covariance must be square, got shape {covariance.shape}")
            if covariance.shape[0] == 0:
                raise RiskModelError("covariance must cover at least one asset")
            if covariance.shape[0] > MAX_ASSETS:
                raise RiskModelError(f"covariance exceeds the {MAX_ASSETS}-asset ceiling")
        try:
            matrix = np.array(covariance, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RiskModelError("covariance cannot be converted to float64") from exc
        if assets is None:
            raise RiskModelError("asset labels are required for an array covariance")
        try:
            assets = tuple(str(asset) for asset in assets)
        except TypeError as exc:
            raise RiskModelError("assets must be an iterable of labels") from exc

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RiskModelError(f"covariance must be square, got shape {matrix.shape}")
    if matrix.shape[0] != len(assets):
        raise RiskModelError("covariance dimension does not match asset labels")
    if not matrix.shape[0]:
        raise RiskModelError("covariance must cover at least one asset")
    if matrix.shape[0] > MAX_ASSETS:
        raise RiskModelError(f"covariance exceeds the {MAX_ASSETS}-asset ceiling")
    if not np.isfinite(matrix).all():
        raise RiskModelError("covariance contains non-finite entries")
    digest = source_hash if source_hash is not None else _source_digest(matrix, labels=assets)
    _require_scaled_matrix_close(
        matrix,
        matrix.T,
        name="covariance is not symmetric",
        relative_tolerance=MATRIX_SYMMETRY_RTOL,
    )
    symmetric = (matrix + matrix.T) / 2.0
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise RiskModelError("covariance eigendecomposition failed") from exc
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    if largest <= 0.0:
        raise RiskModelError("covariance has no positive variance direction")
    if smallest < -ROUND_OFF_NEGATIVE_LIMIT * largest:
        raise RiskModelError(f"covariance is materially indefinite (min eigenvalue {smallest:.3e})")

    ridge = 0.0
    required_floor = EIGENVALUE_FLOOR * largest
    if smallest <= 0.0:
        # Exact singularity is model uncertainty, not floating-point noise.  A
        # tiny negative eigenvalue may arise when a mathematically PSD estimate
        # is assembled from several products; that is the only repair policy.
        if smallest == 0.0 or not allow_ridge:
            raise RiskModelError(
                "covariance is singular or not positive definite "
                f"(min eigenvalue {smallest:.3e}); refusing stabilization"
            )
        ridge = float(required_floor - smallest)
        symmetric = symmetric + ridge * np.eye(symmetric.shape[0])
        try:
            eigenvalues = np.linalg.eigvalsh(symmetric)
        except np.linalg.LinAlgError as exc:
            raise RiskModelError("stabilized covariance eigendecomposition failed") from exc
        smallest = float(eigenvalues[0])
        largest = float(eigenvalues[-1])
    elif smallest < required_floor:
        condition = largest / smallest
        raise RiskModelError(
            "covariance is ill-conditioned "
            f"(min eigenvalue {smallest:.3e}, condition {condition:.3e}); refusing repair"
        )

    condition = float(largest / smallest)
    if not np.isfinite(condition) or condition > CONDITION_LIMIT * (1.0 + 1e-10):
        raise RiskModelError(
            f"covariance condition number {condition:.3e} exceeds {CONDITION_LIMIT:.1e}"
        )
    return RiskModel(
        assets=assets,
        covariance=symmetric,
        periods_per_year=periods_per_year,
        min_eigenvalue=smallest,
        condition_number=condition,
        ridge_applied=ridge,
        estimator=estimator,
        shrinkage_intensity=shrinkage_intensity,
        estimator_parameters=estimator_parameters,
        as_of=as_of,
        window_start=window_start,
        window_end=window_end,
        n_observations=n_observations,
        observations_considered=observations_considered,
        observations_dropped=observations_dropped,
        dropped_assets=dropped_assets,
        source_hash=digest,
    )


@dataclass(frozen=True)
class _CausalWindow:
    """One complete-case estimator window plus explicit coverage accounting."""

    frame: pd.DataFrame
    observations_considered: int
    observations_dropped: int
    dropped_assets: tuple[str, ...]


def _causal_window(
    returns: pd.DataFrame, *, as_of: pd.Timestamp | str | None, window: int
) -> _CausalWindow:
    """Validate and return the bounded complete-case window used by estimators."""
    if not isinstance(returns, pd.DataFrame):
        raise RiskModelError("returns must be a pandas DataFrame")
    source_rows, source_assets = returns.shape
    if not 1 <= source_rows <= MAX_SOURCE_OBSERVATIONS:
        raise RiskModelError(
            f"raw return observation count must lie in [1, {MAX_SOURCE_OBSERVATIONS}]"
        )
    if not 1 <= source_assets <= MAX_ASSETS:
        raise RiskModelError(f"raw return asset count must lie in [1, {MAX_ASSETS}]")
    if source_rows * source_assets > MAX_SOURCE_CELLS:
        raise RiskModelError(f"raw return source exceeds the {MAX_SOURCE_CELLS}-cell ceiling")
    if returns.index.has_duplicates:
        raise RiskModelError("returns index must be unique")
    if returns.columns.has_duplicates:
        raise RiskModelError("returns columns must be unique")
    if not isinstance(window, (int, np.integer)) or isinstance(window, (bool, np.bool_)):
        raise RiskModelError("window must be an integer")
    window = int(window)
    if not MIN_OBSERVATIONS <= window <= MAX_WINDOW:
        raise RiskModelError(f"window must lie in [{MIN_OBSERVATIONS}, {MAX_WINDOW}]")
    boundary = _timestamp(as_of, name="as_of")
    try:
        frame = returns.sort_index()
    except (TypeError, ValueError) as exc:
        raise RiskModelError("returns index cannot be ordered deterministically") from exc
    if boundary is not None:
        try:
            frame = frame.loc[frame.index < boundary]
        except TypeError as exc:
            raise RiskModelError(
                "returns index and as_of have incompatible timestamp semantics"
            ) from exc
    # Coverage is defined on the actual trailing estimation window. An asset
    # with old history but no observation inside that window is stale and must
    # be dropped/reported rather than eliminating every complete row.
    frame = frame.iloc[-window:]
    all_missing = tuple(str(column) for column in frame.columns[frame.isna().all(axis=0)])
    frame = frame.dropna(axis=1, how="all")
    considered = len(frame)
    frame = frame.dropna(axis=0, how="any")
    if frame.shape[1] < 1:
        raise RiskModelError("no asset has usable return history")
    if frame.shape[1] > MAX_ASSETS:
        raise RiskModelError(f"risk-model window exceeds the {MAX_ASSETS}-asset ceiling")
    if len(frame) < MIN_OBSERVATIONS:
        raise RiskModelError(
            f"only {len(frame)} complete observations strictly before {boundary}; "
            f"need {MIN_OBSERVATIONS}"
        )
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskModelError("return window cannot be converted to float64") from exc
    if not np.isfinite(values).all():
        raise RiskModelError("return window contains non-finite observations")
    return _CausalWindow(
        frame=frame,
        observations_considered=considered,
        observations_dropped=considered - len(frame),
        dropped_assets=all_missing,
    )


def shrinkage_covariance(
    returns: pd.DataFrame,
    *,
    as_of: pd.Timestamp | str | None = None,
    window: int = 252,
    intensity: float | None = None,
    periods_per_year: int = 252,
) -> RiskModel:
    """Estimate a causal constant-correlation shrinkage covariance.

    The estimator uses only complete observations strictly before ``as_of``.
    Its deterministic default shrinkage ``min(n_assets / n_observations, 1)``
    increases when sampling error is greatest.  Complexity is ``O(T n²+n³)``
    for ``T`` observations and ``n`` assets; both dimensions are bounded.
    """
    supplied_intensity = (
        None if intensity is None else _unit_interval(intensity, name="shrinkage intensity")
    )
    coverage = _causal_window(returns, as_of=as_of, window=window)
    frame = coverage.frame
    values = frame.to_numpy(dtype=np.float64)
    n_observations, n_assets = values.shape
    sample = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    variances = np.diag(sample).copy()
    if (variances <= 0.0).any():
        raise RiskModelError("return window contains an asset with zero sample variance")
    deviations = np.sqrt(variances)
    correlation = sample / np.outer(deviations, deviations)
    off_diagonal = correlation[~np.eye(n_assets, dtype=bool)]
    average_correlation = float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
    target = average_correlation * np.outer(deviations, deviations)
    np.fill_diagonal(target, variances)
    weight = (
        float(np.clip(n_assets / n_observations, 0.0, 1.0))
        if supplied_intensity is None
        else supplied_intensity
    )
    shrunk = (1.0 - weight) * sample + weight * target
    digest = _source_digest(
        values,
        labels=tuple(str(column) for column in frame.columns),
        timestamps=frame.index,
    )
    return validate_covariance(
        shrunk,
        assets=tuple(str(column) for column in frame.columns),
        periods_per_year=periods_per_year,
        estimator="constant_correlation_shrinkage_v1",
        allow_ridge=True,
        shrinkage_intensity=weight,
        estimator_parameters=(("window", str(window)),),
        as_of=as_of,
        window_start=pd.Timestamp(frame.index[0]),
        window_end=pd.Timestamp(frame.index[-1]),
        n_observations=n_observations,
        observations_considered=coverage.observations_considered,
        observations_dropped=coverage.observations_dropped,
        dropped_assets=coverage.dropped_assets,
        source_hash=digest,
    )


@dataclass(frozen=True)
class FactorRiskModel:
    """Validated point-in-time decomposition ``B F B' + D``.

    ``exposure_vintage`` identifies the economic snapshot and
    ``exposure_available_at`` records when that snapshot became observable.
    Both are part of model identity and must not follow the allocation
    decision.  Stabilization fields disclose the threshold, amount, and number
    of affected directions/assets instead of silently repairing an estimate.

    All arrays are defensively copied onto immutable bytes.  Direct
    construction repeats exposure rank/conditioning, factor-covariance, and
    scale-relative decomposition checks.
    """

    risk_model: RiskModel
    factors: tuple[str, ...]
    loadings: FloatArray
    factor_covariance: FloatArray
    specific_variances: FloatArray
    exposure_vintage: pd.Timestamp
    exposure_available_at: pd.Timestamp
    factor_variance_floor: float = 0.0
    factor_ridge_applied: float = 0.0
    factor_directions_stabilized: int = 0
    specific_variance_floor: float = 0.0
    specific_stabilization_total: float = 0.0
    specific_assets_stabilized: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.risk_model, RiskModel):
            raise RiskModelError("risk_model must be a validated RiskModel")
        vintage, available, _ = _exposure_timestamps(
            exposure_vintage=self.exposure_vintage,
            exposure_available_at=self.exposure_available_at,
            as_of=self.risk_model.as_of,
        )
        try:
            factors = tuple(str(factor) for factor in self.factors)
        except TypeError as exc:
            raise RiskModelError("factors must be an iterable of labels") from exc
        if not factors or len(factors) > MAX_FACTORS:
            raise RiskModelError(f"factor count must lie in [1, {MAX_FACTORS}]")
        if len(set(factors)) != len(factors) or any(not factor for factor in factors):
            raise RiskModelError("factor labels must be unique and non-empty")
        loadings = _readonly_array(self.loadings, ndim=2, name="factor loadings")
        factor_covariance = _readonly_array(
            self.factor_covariance, ndim=2, name="factor covariance"
        )
        specific = _readonly_array(self.specific_variances, ndim=1, name="specific variances")
        n_assets = len(self.risk_model.assets)
        n_factors = len(factors)
        if loadings.shape != (n_assets, n_factors):
            raise RiskModelError("factor loadings do not align with assets and factors")
        if factor_covariance.shape != (n_factors, n_factors):
            raise RiskModelError("factor covariance has the wrong shape")
        if specific.shape != (n_assets,) or (specific <= 0.0).any():
            raise RiskModelError("specific variances must be positive and asset-aligned")
        if n_assets <= n_factors:
            raise RiskModelError("factor model needs more assets than factors")
        try:
            loading_rank = int(np.linalg.matrix_rank(loadings))
            loading_condition = float(np.linalg.cond(loadings))
        except np.linalg.LinAlgError as exc:
            raise RiskModelError("factor exposure decomposition failed") from exc
        if loading_rank < n_factors:
            raise RiskModelError("factor exposure matrix is rank deficient")
        if not np.isfinite(loading_condition) or loading_condition > CONDITION_LIMIT:
            raise RiskModelError("factor exposure matrix is ill-conditioned")

        _require_scaled_matrix_close(
            factor_covariance,
            factor_covariance.T,
            name="factor covariance is not symmetric",
            relative_tolerance=MATRIX_SYMMETRY_RTOL,
        )
        try:
            factor_eigenvalues = np.linalg.eigvalsh(factor_covariance)
        except np.linalg.LinAlgError as exc:
            raise RiskModelError("factor covariance eigendecomposition failed") from exc
        smallest_factor = float(factor_eigenvalues[0])
        largest_factor = float(factor_eigenvalues[-1])
        if smallest_factor <= 0.0 or largest_factor <= 0.0:
            raise RiskModelError("factor covariance must be positive definite")
        factor_condition = float(largest_factor / smallest_factor)
        if not np.isfinite(factor_condition) or factor_condition > CONDITION_LIMIT * (
            1.0 + SCALAR_DIAGNOSTIC_RTOL
        ):
            raise RiskModelError("factor covariance is ill-conditioned")

        factor_floor = _nonnegative_float(self.factor_variance_floor, name="factor_variance_floor")
        factor_ridge = _nonnegative_float(self.factor_ridge_applied, name="factor_ridge_applied")
        factor_count = _bounded_count(
            self.factor_directions_stabilized,
            name="factor_directions_stabilized",
            maximum=n_factors,
        )
        specific_floor = _nonnegative_float(
            self.specific_variance_floor, name="specific_variance_floor"
        )
        specific_total = _nonnegative_float(
            self.specific_stabilization_total,
            name="specific_stabilization_total",
        )
        specific_count = _bounded_count(
            self.specific_assets_stabilized,
            name="specific_assets_stabilized",
            maximum=n_assets,
        )
        if (factor_ridge == 0.0) != (factor_count == 0):
            raise RiskModelError("factor stabilization amount and count do not reconcile")
        if factor_count and factor_floor == 0.0:
            raise RiskModelError("factor stabilization requires a positive variance floor")
        factor_roundoff = (
            ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * max(n_factors, 1) * largest_factor
        )
        pre_stabilization_eigenvalues = factor_eigenvalues - factor_ridge
        pre_stabilization_largest = float(pre_stabilization_eigenvalues[-1])
        pre_stabilization_smallest = float(pre_stabilization_eigenvalues[0])
        implied_factor_count = int(np.count_nonzero(pre_stabilization_eigenvalues < factor_floor))
        if implied_factor_count != factor_count:
            raise RiskModelError(
                "factor stabilization count does not match the disclosed ridge and floor"
            )
        if pre_stabilization_smallest < (
            -ROUND_OFF_NEGATIVE_LIMIT * max(pre_stabilization_largest, 0.0)
        ):
            raise RiskModelError("factor stabilization conceals material indefiniteness")
        maximum_plausible_ridge = (
            factor_floor
            + ROUND_OFF_NEGATIVE_LIMIT * max(pre_stabilization_largest, 0.0)
            + factor_roundoff
        )
        if factor_ridge > maximum_plausible_ridge:
            raise RiskModelError("factor stabilization ridge exceeds its plausible bound")
        if factor_count and abs(smallest_factor - factor_floor) > (
            factor_roundoff + SCALAR_DIAGNOSTIC_RTOL * factor_floor
        ):
            raise RiskModelError("factor stabilization amount does not reach its disclosed floor")
        if smallest_factor + factor_roundoff < factor_floor:
            raise RiskModelError("factor covariance remains below its disclosed variance floor")
        if (specific_total == 0.0) != (specific_count == 0):
            raise RiskModelError("specific stabilization amount and count do not reconcile")
        if specific_count and specific_floor == 0.0:
            raise RiskModelError("specific stabilization requires a positive variance floor")
        specific_roundoff = (
            ROUNDOFF_MULTIPLIER
            * np.finfo(np.float64).eps
            * max(n_assets, 1)
            * max(float(np.max(specific)), specific_floor)
        )
        maximum_specific_total = specific_count * specific_floor + specific_roundoff
        if specific_total > maximum_specific_total:
            raise RiskModelError("specific stabilization total exceeds its plausible bound")
        values_at_floor = int(np.count_nonzero(specific <= specific_floor + specific_roundoff))
        if values_at_floor < specific_count:
            raise RiskModelError(
                "specific stabilization count exceeds variances at the disclosed floor"
            )
        if float(np.min(specific)) + specific_roundoff < specific_floor:
            raise RiskModelError("specific variance remains below its disclosed floor")

        with np.errstate(over="ignore", invalid="ignore"):
            reconstructed = loadings @ factor_covariance @ loadings.T + np.diag(specific)
        if not np.isfinite(reconstructed).all():
            raise RiskModelError("factor decomposition produced non-finite covariance")
        try:
            operation_scale = _frobenius_norm(loadings) ** 2 * _frobenius_norm(
                factor_covariance
            ) + _frobenius_norm(np.diag(specific))
        except OverflowError as exc:
            raise RiskModelError("factor decomposition error bound overflowed") from exc
        _require_scaled_matrix_close(
            reconstructed,
            self.risk_model.covariance,
            name="factor decomposition",
            relative_tolerance=MATRIX_RECONCILIATION_RTOL,
            operation_scale=operation_scale,
        )
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "factor_covariance", factor_covariance)
        object.__setattr__(self, "specific_variances", specific)
        object.__setattr__(self, "exposure_vintage", vintage)
        object.__setattr__(self, "exposure_available_at", available)
        object.__setattr__(self, "factor_variance_floor", factor_floor)
        object.__setattr__(self, "factor_ridge_applied", factor_ridge)
        object.__setattr__(self, "factor_directions_stabilized", factor_count)
        object.__setattr__(self, "specific_variance_floor", specific_floor)
        object.__setattr__(self, "specific_stabilization_total", specific_total)
        object.__setattr__(self, "specific_assets_stabilized", specific_count)

    @property
    def identity(self) -> str:
        """Hash temporal provenance, factor names, and decomposition values."""
        digest = hashlib.sha256()
        digest.update(b"alphaforge-factor-risk-model-v2\0")
        digest.update(self.risk_model.identity.encode("ascii"))
        for factor in self.factors:
            digest.update(factor.encode("utf-8"))
            digest.update(b"\0")
        for array in (self.loadings, self.factor_covariance, self.specific_variances):
            canonical = np.ascontiguousarray(array, dtype="<f8")
            digest.update(str(canonical.shape).encode("ascii"))
            digest.update(canonical.tobytes(order="C"))
        for value in (
            self.exposure_vintage.isoformat(),
            self.exposure_available_at.isoformat(),
            self.factor_variance_floor.hex(),
            self.factor_ridge_applied.hex(),
            str(self.factor_directions_stabilized),
            self.specific_variance_floor.hex(),
            self.specific_stabilization_total.hex(),
            str(self.specific_assets_stabilized),
        ):
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def factor_exposures(self, weights: FloatArray) -> FloatArray:
        """Return ``B'w`` after shape/finiteness validation."""
        try:
            vector = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RiskModelError("weights cannot be converted to float64") from exc
        if vector.shape != (len(self.risk_model.assets),) or not np.isfinite(vector).all():
            raise RiskModelError("weights must be finite and asset-aligned")
        return np.asarray(self.loadings.T @ vector, dtype=np.float64)

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe factor coverage and decomposition metadata."""
        return {
            **self.risk_model.diagnostics(),
            "factor_model_identity": self.identity,
            "factors": list(self.factors),
            "n_factors": len(self.factors),
            "exposure_vintage": self.exposure_vintage.isoformat(),
            "exposure_available_at": self.exposure_available_at.isoformat(),
            "factor_variance_floor": self.factor_variance_floor,
            "factor_ridge_applied": self.factor_ridge_applied,
            "factor_directions_stabilized": self.factor_directions_stabilized,
            "specific_variance_floor": self.specific_variance_floor,
            "specific_stabilization_total": self.specific_stabilization_total,
            "specific_assets_stabilized": self.specific_assets_stabilized,
            "loading_condition_number": float(np.linalg.cond(self.loadings)),
            "factor_condition_number": float(np.linalg.cond(self.factor_covariance)),
            "specific_variance_trace_fraction": float(
                np.sum(self.specific_variances) / np.sum(np.diag(self.risk_model.covariance))
            ),
        }


def estimate_factor_risk_model(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    as_of: pd.Timestamp | str | None,
    exposure_vintage: pd.Timestamp | str,
    exposure_available_at: pd.Timestamp | str,
    window: int = 252,
    factor_shrinkage: float = 0.2,
    specific_shrinkage: float = 0.2,
    periods_per_year: int = 252,
) -> FactorRiskModel:
    """Estimate a causal linear factor model from point-in-time exposures.

    For each historical observation, factor returns are estimated by least
    squares ``r_t = B f_t + epsilon_t`` using the supplied exposure matrix
    ``B``.  Factor covariance is shrunk toward its diagonal and asset-specific
    variances toward their cross-sectional median.  The exact economic vintage
    and availability timestamp are required, validated against ``as_of``,
    included in identity, and reported in diagnostics.  Any numerical floor is
    authorized only by a non-zero corresponding shrinkage setting and is
    disclosed with its threshold, adjustment amount, and affected count.

    Raises:
        RiskModelError: On rank-deficient exposures, insufficient complete
            history, non-finite input, excessive dimensions, or failed PSD/
            conditioning/reconciliation checks.
    """
    vintage, available, decision = _exposure_timestamps(
        exposure_vintage=exposure_vintage,
        exposure_available_at=exposure_available_at,
        as_of=as_of,
    )
    factor_weight = _unit_interval(factor_shrinkage, name="factor_shrinkage")
    specific_weight = _unit_interval(specific_shrinkage, name="specific_shrinkage")
    if not isinstance(exposures, pd.DataFrame):
        raise RiskModelError("exposures must be a pandas DataFrame")
    if not 1 <= exposures.shape[0] <= MAX_ASSETS:
        raise RiskModelError(f"exposure asset count must lie in [1, {MAX_ASSETS}]")
    if not 1 <= exposures.shape[1] <= MAX_FACTORS:
        raise RiskModelError(f"factor count must lie in [1, {MAX_FACTORS}]")
    if exposures.index.has_duplicates or exposures.columns.has_duplicates:
        raise RiskModelError("exposure asset and factor labels must be unique")

    coverage = _causal_window(returns, as_of=decision, window=window)
    frame = coverage.frame
    missing = [column for column in frame.columns if column not in exposures.index]
    if missing:
        raise RiskModelError(f"exposures missing return assets: {missing[:5]}")
    aligned_exposures = exposures.reindex(frame.columns)
    factor_labels = tuple(str(column) for column in aligned_exposures.columns)
    try:
        loadings = aligned_exposures.to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskModelError("factor exposures cannot be converted to float64") from exc
    if not np.isfinite(loadings).all():
        raise RiskModelError("factor exposures must be finite")
    n_observations, n_assets = frame.shape
    n_factors = loadings.shape[1]
    if n_assets <= n_factors:
        raise RiskModelError("factor model needs more assets than factors")
    if n_observations <= n_factors + 1:
        raise RiskModelError("factor model needs more observations than factors plus one")
    try:
        loading_rank = int(np.linalg.matrix_rank(loadings))
        loading_condition = float(np.linalg.cond(loadings))
    except np.linalg.LinAlgError as exc:
        raise RiskModelError("factor exposure decomposition failed") from exc
    if loading_rank < n_factors:
        raise RiskModelError("factor exposure matrix is rank deficient")
    if not np.isfinite(loading_condition) or loading_condition > CONDITION_LIMIT:
        raise RiskModelError("factor exposure matrix is ill-conditioned")

    try:
        values = frame.to_numpy(dtype=np.float64)
        factor_returns = np.linalg.lstsq(loadings, values.T, rcond=None)[0].T
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        raise RiskModelError("factor-return least-squares estimation failed") from exc
    if not np.isfinite(factor_returns).all():
        raise RiskModelError("factor-return estimation produced non-finite values")
    factor_sample = np.atleast_2d(np.cov(factor_returns, rowvar=False, ddof=1))
    if not np.isfinite(factor_sample).all():
        raise RiskModelError("factor covariance estimate is non-finite")
    diagonal = np.diag(np.diag(factor_sample))
    factor_covariance = (1.0 - factor_weight) * factor_sample + factor_weight * diagonal
    try:
        factor_eigenvalues = np.linalg.eigvalsh(factor_covariance)
    except np.linalg.LinAlgError as exc:
        raise RiskModelError("factor covariance eigendecomposition failed") from exc
    largest = float(factor_eigenvalues[-1])
    if largest <= 0.0:
        raise RiskModelError("estimated factors have no positive variance")
    initial_factor_floor = EIGENVALUE_FLOOR * largest
    factor_count = int(np.count_nonzero(factor_eigenvalues < initial_factor_floor))
    factor_ridge = 0.0
    factor_roundoff_margin = 0.0
    if factor_count:
        if factor_weight == 0.0:
            raise RiskModelError(
                "factor covariance requires stabilization while factor_shrinkage is zero"
            )
        # Solving for a ridge that lands exactly on the admissible condition
        # boundary is not portable across eigensolver/LAPACK implementations:
        # their final eigenvalue round-off can place the matrix infinitesimally
        # outside the contract.  Add a dimension- and scale-aware floating-point
        # margin while retaining the same economic variance floor.
        factor_roundoff_margin = (
            ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * max(n_factors, 1) * largest
        )
        factor_ridge = (
            initial_factor_floor + factor_roundoff_margin - float(factor_eigenvalues[0])
        ) / (1.0 - EIGENVALUE_FLOOR)
        factor_covariance = factor_covariance + factor_ridge * np.eye(n_factors)
    # Report the actual stabilization target, including its numerical safety
    # margin, so downstream diagnostics can independently reconcile the ridge.
    factor_floor = EIGENVALUE_FLOOR * (largest + factor_ridge) + factor_roundoff_margin
    factor_count = int(np.count_nonzero(factor_eigenvalues < factor_floor))

    residuals = values - factor_returns @ loadings.T
    if not np.isfinite(residuals).all():
        raise RiskModelError("factor residual estimation produced non-finite values")
    raw_specific = np.var(residuals, axis=0, ddof=1)
    if not np.isfinite(raw_specific).all():
        raise RiskModelError("specific variance estimate is non-finite")
    positive = raw_specific[raw_specific > 0.0]
    if not positive.size:
        raise RiskModelError("factor model estimated no positive specific variance")
    specific_target = float(np.median(positive))
    specific_before_floor = (
        1.0 - specific_weight
    ) * raw_specific + specific_weight * specific_target
    specific_floor = EIGENVALUE_FLOOR * specific_target
    specific_adjustments = np.maximum(specific_floor - specific_before_floor, 0.0)
    specific_count = int(np.count_nonzero(specific_adjustments > 0.0))
    specific_total = float(np.sum(specific_adjustments))
    if specific_count and specific_weight == 0.0:
        raise RiskModelError(
            "specific variance requires stabilization while specific_shrinkage is zero"
        )
    specific = specific_before_floor + specific_adjustments
    with np.errstate(over="ignore", invalid="ignore"):
        asset_covariance = loadings @ factor_covariance @ loadings.T + np.diag(specific)
    if not np.isfinite(asset_covariance).all():
        raise RiskModelError("factor decomposition produced non-finite asset covariance")
    labels = tuple(str(column) for column in frame.columns)
    digest = _source_digest(
        values,
        labels=(
            *labels,
            "__factor_names__",
            *factor_labels,
            "__exposure_vintage__",
            vintage.isoformat(),
            "__exposure_available_at__",
            available.isoformat(),
        ),
        timestamps=frame.index,
        extra_arrays=(np.asarray(loadings, dtype=np.float64),),
    )
    risk_model = validate_covariance(
        asset_covariance,
        assets=labels,
        periods_per_year=periods_per_year,
        estimator="linear_factor_shrinkage_v1",
        allow_ridge=False,
        shrinkage_intensity=factor_weight,
        estimator_parameters=(
            ("exposure_available_at", available.isoformat()),
            ("exposure_vintage", vintage.isoformat()),
            ("factor_shrinkage", factor_weight.hex()),
            ("specific_shrinkage", specific_weight.hex()),
            ("window", str(window)),
        ),
        as_of=decision,
        window_start=pd.Timestamp(frame.index[0]),
        window_end=pd.Timestamp(frame.index[-1]),
        n_observations=n_observations,
        observations_considered=coverage.observations_considered,
        observations_dropped=coverage.observations_dropped,
        dropped_assets=coverage.dropped_assets,
        source_hash=digest,
    )
    return FactorRiskModel(
        risk_model=risk_model,
        factors=factor_labels,
        loadings=loadings,
        factor_covariance=factor_covariance,
        specific_variances=specific,
        exposure_vintage=vintage,
        exposure_available_at=available,
        factor_variance_floor=factor_floor,
        factor_ridge_applied=factor_ridge,
        factor_directions_stabilized=factor_count,
        specific_variance_floor=specific_floor,
        specific_stabilization_total=specific_total,
        specific_assets_stabilized=specific_count,
    )
