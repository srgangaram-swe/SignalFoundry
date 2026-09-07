r"""Certified constrained Markowitz optimization for SF-S4-MR2.

The public problem is a convex quadratic program in periodic-return units:

.. math::

   \min_w \; \frac{\lambda}{2} w^T\Sigma w - \mu^T w
       + c_1\lVert w-w_{prev}\rVert_1
       + c_2\lVert w-w_{prev}\rVert_2^2

subject to the declared budget, gross, net, position, long/short, turnover,
liquidity, target-return, and linear exposure limits.  Auxiliary variables
linearize ``|w|`` and ``|w-w_prev|``; they do not relax either constraint.

OSQP solves the canonical sparse QP.  Solver status is necessary but never
sufficient: every candidate receives an implementation-independent portfolio
audit plus primal, stationarity, complementarity, primal/dual-gap, and objective
reconstruction checks.  Only the exact ``solved`` status with a passing
certificate becomes ``optimal``.  Inaccurate, timed-out, iteration-exhausted,
or uncertified candidates remain unusable.

All input arrays are defensively copied and made read-only.  Problem and solve
identities hash exact float bytes, timestamps, formulation, solver version, and
settings; no rounded research input can alias another run.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import osqp
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from alphaforge.optimization.risk_model import RiskModel, RiskModelError
from alphaforge.portfolio.contracts import (
    CONSTRAINT_TOLERANCE,
    PortfolioConstraints,
    PortfolioError,
)

type FloatArray = NDArray[np.float64]

Formulation = Literal["minimum_variance", "target_return", "maximum_utility", "alpha_risk_cost"]
SolverStatus = Literal["optimal", "max_iterations", "infeasible", "failed"]

FORMULATIONS: frozenset[str] = frozenset(
    {"minimum_variance", "target_return", "maximum_utility", "alpha_risk_cost"}
)

# Refusal bounds, not optimization knobs.
MAX_ITERATIONS = 20_000
MAX_SOLVER_SECONDS = 5.0
MAX_OPTIMIZATION_ASSETS = 512
MAX_EXPOSURES = 256
MAX_PREVIOUS_ASSETS = 2 * MAX_OPTIMIZATION_ASSETS
MAX_DIAGNOSTIC_DEPTH = 8
MAX_DIAGNOSTIC_ITEMS = 4_096
MAX_TOLERANCE = 1e-5
OSQP_NUMERICAL_FLOOR = 1e-10
OBJECTIVE_NORMALIZATION_THRESHOLD = 1e-2

# The public feasibility and independent KKT certificate are intentionally
# looser than the internal solve request but far below an economically meaningful
# weight.  They are fixed so a caller cannot bless a poor solution by asking for
# a loose tolerance.
DEFAULT_TOLERANCE = 1e-9
AUDIT_TOLERANCE = 1e-7
KKT_TOLERANCE = 1e-7
OBJECTIVE_TOLERANCE = 1e-8
FLOAT_ROUNDOFF_FLOOR = 128.0 * np.finfo(np.float64).eps


class OptimizerError(PortfolioError):
    """Raised when an optimization request is invalid or cannot be certified."""


def _finite_float(value: Any, *, name: str) -> float:
    """Return one real finite scalar, mapping all type failures to the domain error."""
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or isinstance(value, (str, bytes, complex, np.complexfloating))
    ):
        raise OptimizerError(f"{name} must be a real numeric scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerError(f"{name} must be a real numeric scalar") from exc
    if not np.isfinite(result):
        raise OptimizerError(f"{name} must be finite")
    return result


def _audit_tolerance(*scales: float) -> float:
    """Return a unit-aware absolute tolerance without an implicit unit scale."""
    finite_scales = [abs(float(scale)) for scale in scales if np.isfinite(scale)]
    scale = max(finite_scales, default=0.0)
    return max(FLOAT_ROUNDOFF_FLOOR, AUDIT_TOLERANCE * scale)


def _roundoff_close(left: float, right: float, *terms: float) -> bool:
    """Compare reconstructed floats at their actual scale, including sub-unit values."""
    scale = max((abs(float(value)) for value in (left, right, *terms)), default=0.0)
    tolerance = max(np.nextafter(0.0, 1.0), 32.0 * np.finfo(np.float64).eps * scale)
    return abs(float(left) - float(right)) <= tolerance


def _readonly_array(
    value: Any,
    *,
    ndim: int,
    name: str,
    max_elements: int | None = None,
) -> FloatArray:
    """Return a finite float64 array backed by immutable storage."""
    if max_elements is not None:
        cheap_size: int | None = None
        if isinstance(value, np.ndarray):
            cheap_size = int(value.size)
        elif ndim == 1 and not isinstance(value, (str, bytes)):
            try:
                cheap_size = len(value)
            except (TypeError, OverflowError):
                cheap_size = None
        if cheap_size is not None and cheap_size > max_elements:
            raise OptimizerError(f"{name} exceeds the {max_elements}-element resource ceiling")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerError(f"{name} cannot be converted to float64") from exc
    if result.ndim != ndim:
        raise OptimizerError(f"{name} must be {ndim}-dimensional, got shape {result.shape}")
    if max_elements is not None and result.size > max_elements:
        raise OptimizerError(f"{name} exceeds the {max_elements}-element resource ceiling")
    if not np.isfinite(result).all():
        raise OptimizerError(f"{name} must be finite")
    canonical = np.ascontiguousarray(result, dtype=np.float64)
    immutable = np.frombuffer(canonical.tobytes(order="C"), dtype=np.float64).reshape(
        canonical.shape
    )
    immutable.setflags(write=False)
    return immutable


def _readonly_series(
    value: pd.Series,
    *,
    name: str,
    required_order: tuple[str, ...] | None = None,
    reject_extra: bool = False,
    max_entries: int = MAX_PREVIOUS_ASSETS,
) -> pd.Series:
    """Validate, canonicalize, copy, and freeze one labelled numeric vector."""
    if not isinstance(value, pd.Series):
        raise OptimizerError(f"{name} must be a pandas Series")
    if value.index.has_duplicates:
        raise OptimizerError(f"{name} index must be unique")
    if len(value) > max_entries:
        raise OptimizerError(f"{name} exceeds the {max_entries}-entry resource ceiling")
    try:
        labels = tuple(str(label) for label in value.index)
        numbers = np.array(value.to_numpy(dtype=np.float64), copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerError(f"{name} cannot be converted to a labelled float64 vector") from exc
    if len(set(labels)) != len(labels):
        raise OptimizerError(f"{name} labels collide after string normalization")
    if not np.isfinite(numbers).all():
        raise OptimizerError(f"{name} must be finite")
    mapping = dict(zip(labels, numbers, strict=True))
    if required_order is None:
        order = tuple(sorted(mapping))
    else:
        extra = set(mapping) - set(required_order)
        if reject_extra and extra:
            raise OptimizerError(f"{name} contains assets outside the risk model: {sorted(extra)}")
        if set(mapping) != set(required_order) and reject_extra:
            # Missing liquidity is an explicit zero cap, but expected returns are
            # required to be complete and call this helper without reject_extra.
            order = required_order
        elif set(mapping) == set(required_order):
            order = required_order
        else:
            order = (*required_order, *sorted(extra))
    frozen_values = _readonly_array(
        [mapping[label] for label in order],
        ndim=1,
        name=f"canonical {name}",
        max_elements=max_entries,
    )
    result = pd.Series(frozen_values, index=pd.Index(order), name=value.name, copy=False)
    result.to_numpy(copy=False).setflags(write=False)
    return result


def _timestamp(value: pd.Timestamp | str | None, *, name: str) -> pd.Timestamp | None:
    """Normalize a timestamp without discarding its timezone."""
    if value is None:
        return None
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise OptimizerError(f"{name} must be a valid timestamp") from exc
    if pd.isna(result):
        raise OptimizerError(f"{name} must not be NaT")
    return result


def _timestamp_iso(value: pd.Timestamp | str | None, *, name: str) -> str:
    """Return a normalized timestamp's ISO representation or an empty marker."""
    normalized = _timestamp(value, name=name)
    return "" if normalized is None else normalized.isoformat()


def _optional_timestamp_iso(value: pd.Timestamp | str | None, *, name: str) -> str | None:
    """Return a normalized timestamp's ISO representation or ``None``."""
    normalized = _timestamp(value, name=name)
    return None if normalized is None else normalized.isoformat()


def _gross_limit(constraints: PortfolioConstraints) -> float:
    """Return MR2's capital-scaled gross/leverage ceiling."""
    return min(constraints.max_gross, constraints.max_leverage) * constraints.deployable


def _deep_freeze(
    value: Any,
    *,
    depth: int = 0,
    remaining: list[int] | None = None,
) -> Any:
    """Recursively detach and bound JSON-oriented diagnostic values."""
    if remaining is None:
        remaining = [MAX_DIAGNOSTIC_ITEMS]
    if depth > MAX_DIAGNOSTIC_DEPTH:
        raise OptimizerError(
            f"risk_diagnostics exceeds the {MAX_DIAGNOSTIC_DEPTH}-level nesting ceiling"
        )
    remaining[0] -= 1
    if remaining[0] < 0:
        raise OptimizerError(
            f"risk_diagnostics exceeds the {MAX_DIAGNOSTIC_ITEMS}-item resource ceiling"
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in frozen:
                raise OptimizerError("risk_diagnostics keys collide after string normalization")
            frozen[normalized_key] = _deep_freeze(item, depth=depth + 1, remaining=remaining)
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        # Tuples eliminate ndarray write-flag bypasses from nested public
        # diagnostics while preserving deterministic JSON serialization.
        if value.size > remaining[0]:
            raise OptimizerError(
                f"risk_diagnostics exceeds the {MAX_DIAGNOSTIC_ITEMS}-item resource ceiling"
            )
        return _deep_freeze(value.tolist(), depth=depth + 1, remaining=remaining)
    if isinstance(value, pd.Series):
        return _readonly_series(
            value,
            name="risk diagnostic series",
            max_entries=min(MAX_DIAGNOSTIC_ITEMS, max(remaining[0], 0)),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item, depth=depth + 1, remaining=remaining) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_deep_freeze(item, depth=depth + 1, remaining=remaining) for item in value)
    if isinstance(value, np.generic):
        return _deep_freeze(value.item(), depth=depth + 1, remaining=remaining)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise OptimizerError(
        f"risk_diagnostics contains unsupported mutable value {type(value).__name__!r}"
    )


def _json_safe(value: Any) -> Any:
    """Return detached JSON-oriented containers for frozen diagnostics."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _hash_array(digest: Any, value: FloatArray) -> None:
    canonical = np.ascontiguousarray(value, dtype="<f8")
    _hash_text(digest, repr(canonical.shape))
    digest.update(canonical.tobytes(order="C"))


@dataclass(frozen=True)
class ExposureConstraint:
    """Bound one linear sector or factor exposure.

    ``lower <= loadings' w <= upper``.  ``loadings`` follows the risk-model
    asset order and is copied into immutable storage.
    """

    name: str
    loadings: FloatArray
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise OptimizerError("exposure constraint name must be a string")
        name = self.name.strip()
        if not name:
            raise OptimizerError("exposure constraint requires a name")
        loadings = _readonly_array(
            self.loadings,
            ndim=1,
            name=f"exposure {name!r} loadings",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        lower = _finite_float(self.lower, name=f"exposure {name!r} lower bound")
        upper = _finite_float(self.upper, name=f"exposure {name!r} upper bound")
        if upper < lower:
            raise OptimizerError(f"exposure {name!r} upper bound is below its lower bound")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def value(self, weights: FloatArray) -> float:
        """Return exposure achieved by ``weights``."""
        vector = _readonly_array(
            weights,
            ndim=1,
            name=f"weights for exposure {self.name!r}",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        if vector.shape != self.loadings.shape:
            raise OptimizerError(f"weights do not align with exposure {self.name!r}")
        with np.errstate(over="ignore", invalid="ignore"):
            achieved = float(self.loadings @ vector)
        if not np.isfinite(achieved):
            raise OptimizerError(f"exposure {self.name!r} calculation is non-finite")
        return achieved

    def violation(self, weights: FloatArray) -> float:
        """Return the non-negative exposure-bound violation."""
        achieved = self.value(weights)
        return float(max(self.lower - achieved, achieved - self.upper, 0.0))


@dataclass(frozen=True)
class CostModel:
    """Immediate rebalance cost in periodic-return units.

    ``linear_bps`` multiplies full L1 turnover. ``quadratic_bps`` multiplies
    squared weight changes.  Both are basis points per corresponding turnover
    unit and are active only for ``alpha_risk_cost``.
    """

    linear_bps: float = 0.0
    quadratic_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("linear_bps", "quadratic_bps"):
            value = _finite_float(getattr(self, name), name=name)
            if value < 0.0:
                raise OptimizerError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    @property
    def linear_rate(self) -> float:
        """Linear cost as return per unit L1 turnover."""
        return self.linear_bps / 10_000.0

    @property
    def quadratic_rate(self) -> float:
        """Quadratic cost as return per squared weight change."""
        return self.quadratic_bps / 10_000.0

    def value(self, weights: FloatArray, previous: FloatArray) -> float:
        """Return cost over two aligned vectors."""
        weights_array = _readonly_array(
            weights,
            ndim=1,
            name="cost weights",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        previous_array = _readonly_array(
            previous,
            ndim=1,
            name="cost previous weights",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        if weights_array.shape != previous_array.shape:
            raise OptimizerError("weights and previous must have identical shapes")
        trade = weights_array - previous_array
        with np.errstate(over="ignore", invalid="ignore"):
            cost = float(
                self.linear_rate * np.sum(np.abs(trade)) + self.quadratic_rate * np.sum(trade**2)
            )
        if not np.isfinite(cost):
            raise OptimizerError("cost calculation overflowed or became non-finite")
        return cost

    # Retained as explicit mathematical primitives for callers that use the
    # cost contract outside the QP.  The certified solver does not compose these
    # operators with a constraint projection.
    def smooth_gradient(self, weights: FloatArray, previous: FloatArray) -> FloatArray:
        """Return the quadratic-cost gradient."""
        weights_array = _readonly_array(
            weights,
            ndim=1,
            name="gradient weights",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        previous_array = _readonly_array(
            previous,
            ndim=1,
            name="gradient previous weights",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        if weights_array.shape != previous_array.shape:
            raise OptimizerError("weights and previous must have identical shapes")
        with np.errstate(over="ignore", invalid="ignore"):
            gradient = np.asarray(
                2.0 * self.quadratic_rate * (weights_array - previous_array),
                dtype=np.float64,
            )
        if not np.isfinite(gradient).all():
            raise OptimizerError("cost gradient overflowed or became non-finite")
        return gradient

    def proximal_step(self, weights: FloatArray, previous: FloatArray, step: float) -> FloatArray:
        """Return the exact unconstrained proximal step for linear turnover."""
        step_value = _finite_float(step, name="proximal step")
        if step_value <= 0.0:
            raise OptimizerError("proximal step must be positive")
        weights_array = _readonly_array(
            weights,
            ndim=1,
            name="proximal weights",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        previous_array = _readonly_array(
            previous,
            ndim=1,
            name="proximal previous weights",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        if weights_array.shape != previous_array.shape:
            raise OptimizerError("weights and previous must have identical shapes")
        with np.errstate(over="ignore", invalid="ignore"):
            trade = weights_array - previous_array
            shrunk = np.sign(trade) * np.maximum(np.abs(trade) - step_value * self.linear_rate, 0.0)
            result = np.asarray(previous_array + shrunk, dtype=np.float64)
        if not np.isfinite(result).all():
            raise OptimizerError("proximal cost step overflowed or became non-finite")
        return result


@dataclass(frozen=True)
class MeanVarianceProblem:
    """Immutable, timestamped mean-variance optimization request.

    Expected returns are periodic returns; covariance entries are squared
    periodic returns over the same frequency. ``expected_returns_periods_per_year``
    must equal the risk model's frequency. ``decision_timestamp`` must match a
    point-in-time risk model's ``as_of``, and alpha availability cannot be later
    than the decision.

    Previous weights may contain assets absent from the new risk universe. Those
    positions are mandatory liquidations: their L1 and quadratic changes enter
    turnover, cost, feasibility, reporting, and identity exactly once.
    """

    risk_model: RiskModel
    expected_returns: pd.Series | None = None
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    previous_weights: pd.Series | None = None
    exposures: tuple[ExposureConstraint, ...] = ()
    cost_model: CostModel = field(default_factory=CostModel)
    risk_aversion: float = 1.0
    target_return: float | None = None
    liquidity_caps: pd.Series | None = None
    budget: float | None = None
    decision_timestamp: pd.Timestamp | str | None = None
    expected_returns_available_at: pd.Timestamp | str | None = None
    expected_returns_periods_per_year: int | None = None
    expected_return_unit: Literal["periodic_return"] = "periodic_return"

    def __post_init__(self) -> None:
        if not isinstance(self.risk_model, RiskModel):
            raise OptimizerError("risk_model must be a validated RiskModel")
        if not isinstance(self.constraints, PortfolioConstraints):
            raise OptimizerError("constraints must be PortfolioConstraints")
        if not isinstance(self.cost_model, CostModel):
            raise OptimizerError("cost_model must be CostModel")
        if not isinstance(self.expected_return_unit, str) or (
            self.expected_return_unit != "periodic_return"
        ):
            raise OptimizerError("expected_return_unit must be 'periodic_return'")
        risk_aversion = _finite_float(self.risk_aversion, name="risk_aversion")
        if risk_aversion <= 0.0:
            raise OptimizerError("risk_aversion must be positive")
        target_return = (
            None
            if self.target_return is None
            else _finite_float(self.target_return, name="target_return")
        )

        budget = self.budget
        if self.budget is not None:
            budget = _finite_float(self.budget, name="budget")
            if 0.0 < abs(budget) <= FLOAT_ROUNDOFF_FLOOR:
                raise OptimizerError(
                    "non-zero budget magnitude is at or below the float64 audit floor"
                )
            if abs(budget) > self.constraints.max_net + CONSTRAINT_TOLERANCE:
                raise OptimizerError(
                    f"budget {budget} cannot satisfy max_net {self.constraints.max_net}"
                )
            if abs(budget) > _gross_limit(self.constraints) + CONSTRAINT_TOLERANCE:
                raise OptimizerError(
                    f"budget {budget} cannot satisfy deployable gross/leverage limits"
                )

        assets = self.risk_model.assets
        if len(assets) > MAX_OPTIMIZATION_ASSETS:
            raise OptimizerError(
                "optimization universe exceeds the "
                f"{MAX_OPTIMIZATION_ASSETS}-asset resource ceiling"
            )
        expected = self.expected_returns
        if expected is not None:
            if not isinstance(expected, pd.Series):
                raise OptimizerError("expected_returns must be a pandas Series")
            if tuple(str(label) for label in expected.index) != assets:
                raise OptimizerError(
                    "expected_returns must be indexed by the risk-model assets in order"
                )
            expected = _readonly_series(
                expected,
                name="expected_returns",
                required_order=assets,
                reject_extra=True,
                max_entries=MAX_OPTIMIZATION_ASSETS,
            )

        previous = self.previous_weights
        if previous is not None:
            previous = _readonly_series(
                previous,
                name="previous_weights",
                required_order=assets,
                reject_extra=False,
                max_entries=MAX_PREVIOUS_ASSETS,
            )
            previous_values = previous.to_numpy(dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                previous_l1 = float(np.sum(np.abs(previous_values)))
                previous_l2_squared = float(previous_values @ previous_values)
            if not (np.isfinite(previous_l1) and np.isfinite(previous_l2_squared)):
                raise OptimizerError("previous_weights aggregate magnitude is non-finite")

        liquidity = self.liquidity_caps
        if liquidity is not None:
            if not isinstance(liquidity, pd.Series):
                raise OptimizerError("liquidity_caps must be a pandas Series")
            if liquidity.index.has_duplicates:
                raise OptimizerError("liquidity_caps index must be unique")
            try:
                normalized = pd.Series(
                    np.asarray(liquidity, dtype=np.float64),
                    index=pd.Index([str(item) for item in liquidity.index]),
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise OptimizerError("liquidity_caps cannot be converted to float64") from exc
            if normalized.index.has_duplicates:
                raise OptimizerError("liquidity_caps labels collide after string normalization")
            extra = set(normalized.index) - set(assets)
            if extra:
                raise OptimizerError(
                    f"liquidity_caps contains assets outside the risk model: {sorted(extra)}"
                )
            normalized = normalized.reindex(assets, fill_value=0.0)
            if not np.isfinite(normalized.to_numpy()).all() or (normalized < 0.0).any():
                raise OptimizerError("liquidity_caps must be finite and non-negative")
            liquidity = _readonly_series(
                normalized,
                name="liquidity_caps",
                required_order=assets,
                reject_extra=True,
                max_entries=MAX_OPTIMIZATION_ASSETS,
            )

        try:
            iterator = iter(self.exposures)
        except TypeError as exc:
            raise OptimizerError(
                "exposures must be an iterable of ExposureConstraint objects"
            ) from exc
        supplied_exposures = tuple(itertools.islice(iterator, MAX_EXPOSURES + 1))
        if len(supplied_exposures) > MAX_EXPOSURES:
            raise OptimizerError(
                f"optimization request exceeds the {MAX_EXPOSURES}-exposure resource ceiling"
            )
        if any(not isinstance(item, ExposureConstraint) for item in supplied_exposures):
            raise OptimizerError("exposures must contain ExposureConstraint objects")
        exposures = tuple(sorted(supplied_exposures, key=lambda item: item.name))
        names = [item.name for item in exposures]
        if len(set(names)) != len(names):
            raise OptimizerError("exposure constraint names must be unique")
        for exposure in exposures:
            if exposure.loadings.shape != (len(assets),):
                raise OptimizerError(
                    f"exposure {exposure.name!r} loadings must cover {len(assets)} assets"
                )

        risk_as_of = _timestamp(self.risk_model.as_of, name="risk_model.as_of")
        decision = _timestamp(self.decision_timestamp, name="decision_timestamp")
        if decision is None:
            decision = risk_as_of
        if risk_as_of is not None and decision is not None and decision != risk_as_of:
            raise OptimizerError("decision_timestamp must equal risk_model.as_of")
        alpha_available = _timestamp(
            self.expected_returns_available_at, name="expected_returns_available_at"
        )
        if expected is not None and alpha_available is None:
            alpha_available = decision
        if decision is not None and alpha_available is not None:
            if (decision.tz is None) != (alpha_available.tz is None):
                raise OptimizerError(
                    "decision_timestamp and expected_returns_available_at must use "
                    "compatible timezone awareness"
                )
            if alpha_available > decision:
                raise OptimizerError("expected returns are not available at the decision timestamp")

        periods = self.expected_returns_periods_per_year
        if expected is not None:
            if periods is None:
                periods = self.risk_model.periods_per_year
            if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
                raise OptimizerError("expected_returns_periods_per_year must be a positive integer")
            if periods != self.risk_model.periods_per_year:
                raise OptimizerError(
                    "expected-return and covariance frequencies are dimensionally inconsistent"
                )
        elif periods is not None:
            raise OptimizerError("expected_returns_periods_per_year requires expected_returns")

        object.__setattr__(self, "expected_returns", expected)
        object.__setattr__(self, "previous_weights", previous)
        object.__setattr__(self, "liquidity_caps", liquidity)
        object.__setattr__(self, "exposures", exposures)
        object.__setattr__(self, "decision_timestamp", decision)
        object.__setattr__(self, "expected_returns_available_at", alpha_available)
        object.__setattr__(self, "expected_returns_periods_per_year", periods)
        object.__setattr__(self, "risk_aversion", risk_aversion)
        object.__setattr__(self, "target_return", target_return)
        object.__setattr__(self, "budget", budget)

    @property
    def assets(self) -> tuple[str, ...]:
        """Assets in optimization order."""
        return self.risk_model.assets

    def alpha(self) -> FloatArray:
        """Return an owned alpha vector, or zeros when alpha is absent."""
        if self.expected_returns is None:
            return np.zeros(len(self.assets), dtype=np.float64)
        return np.array(self.expected_returns.to_numpy(dtype=np.float64), copy=True)

    def previous(self) -> FloatArray:
        """Return current-universe previous weights; missing assets start at zero."""
        if self.previous_weights is None:
            return np.zeros(len(self.assets), dtype=np.float64)
        return np.array(
            self.previous_weights.reindex(self.assets, fill_value=0.0).to_numpy(dtype=np.float64),
            dtype=np.float64,
            copy=True,
        )

    def exited_previous(self) -> pd.Series:
        """Return previous positions absent from the new risk universe."""
        if self.previous_weights is None:
            return pd.Series(dtype=np.float64)
        mask = ~self.previous_weights.index.isin(self.assets)
        return self.previous_weights.loc[mask].copy()

    def caps(self) -> FloatArray | None:
        """Return owned current-universe liquidity caps."""
        if self.liquidity_caps is None:
            return None
        return np.array(self.liquidity_caps.to_numpy(dtype=np.float64), copy=True)

    @property
    def exit_turnover(self) -> float:
        """Mandatory L1 turnover needed to liquidate removed assets."""
        return float(self.exited_previous().abs().sum())

    @property
    def exit_squared_trade(self) -> float:
        """Mandatory squared trade from removed assets."""
        values = self.exited_previous().to_numpy(dtype=np.float64)
        return float(values @ values)

    def turnover(self, weights: FloatArray) -> float:
        """Return full-universe L1 turnover, including mandatory exits."""
        vector = _readonly_array(
            weights,
            ndim=1,
            name="turnover weights",
            max_elements=MAX_OPTIMIZATION_ASSETS,
        )
        if vector.shape != (len(self.assets),):
            raise OptimizerError("weights must be finite and aligned")
        previous = self.previous()
        with np.errstate(over="ignore", invalid="ignore"):
            current = float(np.sum(np.abs(vector - previous)))
            book_scale = max(
                float(np.sum(np.abs(vector))),
                float(np.sum(np.abs(previous))),
            )
            if current <= _audit_tolerance(book_scale):
                current = 0.0
            total = current + self.exit_turnover
        if not np.isfinite(total):
            raise OptimizerError("turnover calculation overflowed or became non-finite")
        return total

    def cost(self, weights: FloatArray) -> float:
        """Return full-universe cost, including mandatory exits."""
        current = self.cost_model.value(weights, self.previous())
        exits = (
            self.cost_model.linear_rate * self.exit_turnover
            + self.cost_model.quadratic_rate * self.exit_squared_trade
        )
        return float(current + exits)

    @property
    def identity(self) -> str:
        """Exact SHA-256 over every immutable problem input and timestamp."""
        digest = hashlib.sha256()
        digest.update(b"alphaforge-mean-variance-problem-v3\0")
        risk_identity = getattr(self.risk_model, "identity", None)
        if isinstance(risk_identity, str):
            _hash_text(digest, risk_identity)
        else:  # compatibility with an older validated RiskModel contract
            for asset in self.assets:
                _hash_text(digest, asset)
            _hash_array(digest, self.risk_model.covariance)
        _hash_array(digest, self.alpha())
        if self.previous_weights is None:
            _hash_text(digest, "previous:none")
        else:
            for label, value in self.previous_weights.items():
                _hash_text(digest, str(label))
                _hash_text(digest, float(value).hex())
        if self.liquidity_caps is None:
            _hash_text(digest, "liquidity:none")
        else:
            _hash_array(digest, self.caps())  # type: ignore[arg-type]
        for exposure in self.exposures:
            _hash_text(digest, exposure.name)
            _hash_array(digest, exposure.loadings)
            _hash_text(digest, exposure.lower.hex())
            _hash_text(digest, exposure.upper.hex())
        for key, value in sorted(self.constraints.to_dict().items()):
            _hash_text(digest, key)
            _hash_text(
                digest, repr(value) if isinstance(value, bool | type(None)) else float(value).hex()
            )
        for value in (
            self.cost_model.linear_bps,
            self.cost_model.quadratic_bps,
            self.risk_aversion,
        ):
            _hash_text(digest, float(value).hex())
        _hash_text(
            digest,
            "" if self.target_return is None else float(self.target_return).hex(),
        )
        _hash_text(digest, "" if self.budget is None else float(self.budget).hex())
        _hash_text(
            digest,
            _timestamp_iso(self.decision_timestamp, name="decision_timestamp"),
        )
        _hash_text(
            digest,
            _timestamp_iso(
                self.expected_returns_available_at,
                name="expected_returns_available_at",
            ),
        )
        _hash_text(
            digest,
            (
                ""
                if self.expected_returns_periods_per_year is None
                else str(self.expected_returns_periods_per_year)
            ),
        )
        _hash_text(digest, self.expected_return_unit)
        return digest.hexdigest()


@dataclass(frozen=True)
class SolverResult:
    """Immutable weights and independent optimality certificate."""

    weights: pd.Series
    formulation: str
    status: SolverStatus
    iterations: int
    max_iterations: int
    residual: float
    tolerance: float
    objective: float
    variance_term: float
    return_term: float
    cost_term: float
    ex_ante_volatility: float
    expected_return: float
    turnover: float
    gross: float
    net: float
    active_constraints: tuple[str, ...]
    problem_identity: str
    risk_diagnostics: Mapping[str, Any]
    audit_passed: bool
    audit_violations: tuple[str, ...] = ()
    solver_identity: str = ""
    solver_name: str = "osqp"
    solver_version: str = ""
    solver_status_detail: str = ""
    primal_residual: float = float("inf")
    dual_residual: float = float("inf")
    complementarity_residual: float = float("inf")
    objective_residual: float = float("inf")
    duality_gap: float = float("inf")
    kkt_passed: bool = False
    exit_turnover: float = 0.0
    solve_time_seconds: float = field(default=0.0, compare=False)
    decision_timestamp: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.weights, pd.Series) or self.weights.index.has_duplicates:
            raise OptimizerError("result weights must be a uniquely indexed Series")
        if not 1 <= len(self.weights) <= MAX_OPTIMIZATION_ASSETS:
            raise OptimizerError(
                "result weights must contain between 1 and " f"{MAX_OPTIMIZATION_ASSETS} assets"
            )
        try:
            labels = tuple(str(label) for label in self.weights.index)
            values = np.array(self.weights.to_numpy(dtype=np.float64), copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise OptimizerError("result weights cannot be converted to float64") from exc
        if len(set(labels)) != len(labels):
            raise OptimizerError("result weight labels collide after string normalization")
        if not np.isfinite(values).all():
            raise OptimizerError("result weights must be finite")
        if not isinstance(self.formulation, str) or self.formulation not in FORMULATIONS:
            raise OptimizerError(f"result has unknown formulation {self.formulation!r}")
        if not isinstance(self.status, str) or self.status not in {
            "optimal",
            "max_iterations",
            "infeasible",
            "failed",
        }:
            raise OptimizerError(f"result has unknown solver status {self.status!r}")
        if (
            not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or not 1 <= self.max_iterations <= MAX_ITERATIONS
        ):
            raise OptimizerError("result max_iterations is invalid")
        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or not 0 <= self.iterations <= self.max_iterations
        ):
            raise OptimizerError("result iterations are outside its declared budget")
        tolerance = _finite_float(self.tolerance, name="result tolerance")
        if not 0.0 < tolerance <= MAX_TOLERANCE:
            raise OptimizerError("result tolerance is invalid")

        finite_fields = (
            "objective",
            "variance_term",
            "return_term",
            "cost_term",
            "ex_ante_volatility",
            "expected_return",
            "turnover",
            "gross",
            "net",
            "exit_turnover",
            "solve_time_seconds",
        )
        for name in finite_fields:
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=f"result {name}"),
            )
        for name in (
            "variance_term",
            "cost_term",
            "ex_ante_volatility",
            "turnover",
            "gross",
            "exit_turnover",
            "solve_time_seconds",
        ):
            if getattr(self, name) < 0.0:
                raise OptimizerError(f"result {name} must be non-negative")
        if self.turnover + AUDIT_TOLERANCE < self.exit_turnover:
            raise OptimizerError("result turnover cannot be below mandatory exit turnover")

        certificate_fields = (
            "residual",
            "primal_residual",
            "dual_residual",
            "complementarity_residual",
            "objective_residual",
            "duality_gap",
        )
        certificate_values: list[float] = []
        for name in certificate_fields:
            raw_value = getattr(self, name)
            if (
                isinstance(raw_value, (bool, np.bool_))
                or not np.isscalar(raw_value)
                or isinstance(raw_value, (str, bytes, complex, np.complexfloating))
            ):
                raise OptimizerError(f"result {name} must be a real numeric scalar")
            try:
                certificate_value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise OptimizerError(f"result {name} must be a real numeric scalar") from exc
            if np.isnan(certificate_value) or certificate_value < 0.0:
                raise OptimizerError(f"result {name} must be non-negative and not NaN")
            certificate_values.append(certificate_value)
            object.__setattr__(self, name, certificate_value)
        expected_residual = max(certificate_values[1:])
        if self.residual != expected_residual:
            raise OptimizerError("result residual does not summarize its certificate")

        reconstructed_objective = self.variance_term - self.return_term + self.cost_term
        if not _roundoff_close(
            self.objective,
            reconstructed_objective,
            self.variance_term,
            self.return_term,
            self.cost_term,
        ):
            raise OptimizerError("result objective does not reconcile its terms")
        reconstructed_gross = float(np.sum(np.abs(values)))
        if not _roundoff_close(self.gross, reconstructed_gross, *values):
            raise OptimizerError("result gross does not reconcile its weights")
        reconstructed_net = float(np.sum(values))
        if not _roundoff_close(self.net, reconstructed_net, *values):
            raise OptimizerError("result net does not reconcile its weights")

        if isinstance(self.active_constraints, (str, bytes)):
            raise OptimizerError("result active_constraints must be a sequence of names")
        if isinstance(self.audit_violations, (str, bytes)):
            raise OptimizerError("result audit_violations must be a sequence of messages")
        try:
            active_iterator = iter(self.active_constraints)
            violation_iterator = iter(self.audit_violations)
        except TypeError as exc:
            raise OptimizerError("result certificate collections must be iterable") from exc
        active_constraints = tuple(itertools.islice(active_iterator, MAX_EXPOSURES + 17))
        audit_violations = tuple(itertools.islice(violation_iterator, MAX_EXPOSURES + 33))
        if len(active_constraints) > MAX_EXPOSURES + 16:
            raise OptimizerError("result active_constraints exceeds its resource ceiling")
        if len(audit_violations) > MAX_EXPOSURES + 32:
            raise OptimizerError("result audit_violations exceeds its resource ceiling")
        if any(not isinstance(item, str) or not item.strip() for item in active_constraints) or len(
            set(active_constraints)
        ) != len(active_constraints):
            raise OptimizerError("result active_constraints must be unique non-empty names")
        if any(not isinstance(item, str) or not item.strip() for item in audit_violations):
            raise OptimizerError("result audit_violations must be non-empty messages")
        if not isinstance(self.audit_passed, bool) or not isinstance(self.kkt_passed, bool):
            raise OptimizerError("result certificate flags must be booleans")
        if self.status == "optimal":
            if not self.audit_passed or not self.kkt_passed or audit_violations:
                raise OptimizerError("optimal result must carry a clean independent certificate")
            if (
                self.primal_residual > KKT_TOLERANCE
                or self.dual_residual > KKT_TOLERANCE
                or self.complementarity_residual > KKT_TOLERANCE
                or self.objective_residual > OBJECTIVE_TOLERANCE
                or self.duality_gap > KKT_TOLERANCE
            ):
                raise OptimizerError("optimal result certificate exceeds its numerical limits")
        elif self.audit_passed or self.kkt_passed or not audit_violations:
            raise OptimizerError(
                "non-optimal result must carry failed flags and at least one violation"
            )

        for name, identity_value in (
            ("problem_identity", self.problem_identity),
            ("solver_identity", self.solver_identity),
        ):
            if (
                not isinstance(identity_value, str)
                or len(identity_value) != 64
                or any(character not in "0123456789abcdef" for character in identity_value)
            ):
                raise OptimizerError(f"result {name} must be a lowercase SHA-256 digest")
        for name, metadata_value in (
            ("solver_name", self.solver_name),
            ("solver_version", self.solver_version),
            ("solver_status_detail", self.solver_status_detail),
        ):
            if not isinstance(metadata_value, str) or not metadata_value.strip():
                raise OptimizerError(f"result {name} must be non-empty")
        if self.decision_timestamp is not None:
            _timestamp(self.decision_timestamp, name="result decision_timestamp")
        if not isinstance(self.risk_diagnostics, Mapping):
            raise OptimizerError("result risk_diagnostics must be a mapping")

        immutable_values = _readonly_array(values, ndim=1, name="result weights")
        frozen = pd.Series(
            immutable_values,
            index=pd.Index(labels),
            name=self.weights.name,
            copy=False,
        )
        frozen.to_numpy(copy=False).setflags(write=False)
        object.__setattr__(self, "weights", frozen)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "active_constraints", active_constraints)
        object.__setattr__(self, "audit_violations", audit_violations)
        object.__setattr__(self, "risk_diagnostics", _deep_freeze(self.risk_diagnostics))

    @property
    def converged(self) -> bool:
        """Return whether exact solver status and the independent certificate passed."""
        return self.status == "optimal" and self.audit_passed and self.kkt_passed

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe research-ledger record."""
        return {
            "formulation": self.formulation,
            "status": self.status,
            "converged": self.converged,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "objective": self.objective,
            "variance_term": self.variance_term,
            "return_term": self.return_term,
            "cost_term": self.cost_term,
            "ex_ante_volatility": self.ex_ante_volatility,
            "expected_return": self.expected_return,
            "turnover": self.turnover,
            "exit_turnover": self.exit_turnover,
            "gross": self.gross,
            "net": self.net,
            "active_constraints": list(self.active_constraints),
            "problem_identity": self.problem_identity,
            "solver_identity": self.solver_identity,
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "solver_status_detail": self.solver_status_detail,
            "primal_residual": self.primal_residual,
            "dual_residual": self.dual_residual,
            "complementarity_residual": self.complementarity_residual,
            "objective_residual": self.objective_residual,
            "duality_gap": self.duality_gap,
            "kkt_passed": self.kkt_passed,
            "decision_timestamp": self.decision_timestamp,
            "risk": _json_safe(self.risk_diagnostics),
            "audit_passed": self.audit_passed,
            "audit_violations": list(self.audit_violations),
        }


@dataclass(frozen=True)
class _CanonicalQP:
    """Sparse canonical QP and metadata required for independent certification."""

    p: sparse.csc_matrix
    p_upper: sparse.csc_matrix
    q: FloatArray
    a: sparse.csc_matrix
    lower: FloatArray
    upper: FloatArray
    constant: float
    n_weights: int
    n_gross: int
    n_trade: int
    objective_scale: float


@dataclass(frozen=True)
class _Certificate:
    primal_residual: float
    dual_residual: float
    complementarity_residual: float
    objective_residual: float
    duality_gap: float
    portfolio_passed: bool
    kkt_passed: bool
    violations: tuple[str, ...]

    @property
    def residual(self) -> float:
        return max(
            self.primal_residual,
            self.dual_residual,
            self.complementarity_residual,
            self.objective_residual,
            self.duality_gap,
        )


def _formulation_terms(
    problem: MeanVarianceProblem, formulation: Formulation
) -> tuple[float, FloatArray, CostModel]:
    """Return effective ``(risk_aversion, alpha, cost)`` for one formulation."""
    if formulation == "minimum_variance":
        return 1.0, np.zeros(len(problem.assets)), CostModel()
    if formulation == "target_return":
        return 1.0, np.zeros(len(problem.assets)), CostModel()
    if formulation == "maximum_utility":
        return problem.risk_aversion, problem.alpha(), CostModel()
    return problem.risk_aversion, problem.alpha(), problem.cost_model


def _hstack_variables(
    weight: sparse.spmatrix,
    gross: sparse.spmatrix,
    trade: sparse.spmatrix | None,
) -> sparse.csc_matrix:
    parts: list[sparse.spmatrix] = [weight, gross]
    if trade is not None:
        parts.append(trade)
    return sparse.hstack(parts, format="csc")


def _build_qp(problem: MeanVarianceProblem, formulation: Formulation) -> _CanonicalQP:
    """Build the exact sparse QP with auxiliary absolute-value variables."""
    n = len(problem.assets)
    risk_aversion, alpha, cost = _formulation_terms(problem, formulation)
    previous = problem.previous()
    use_trade = bool(cost.linear_rate > 0.0 or problem.constraints.max_turnover is not None)
    n_trade = n if use_trade else 0
    total = 2 * n + n_trade

    quadratic_rate = cost.quadratic_rate
    with np.errstate(over="ignore", invalid="ignore"):
        p_weight = risk_aversion * problem.risk_model.covariance + 2.0 * quadratic_rate * np.eye(n)
    if not np.isfinite(p_weight).all():
        raise OptimizerError("quadratic objective overflowed or became non-finite")
    p = sparse.block_diag(
        (
            sparse.csc_matrix(p_weight),
            sparse.csc_matrix((n, n)),
            sparse.csc_matrix((n_trade, n_trade)),
        ),
        format="csc",
    )
    q = np.zeros(total, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        q[:n] = -alpha - 2.0 * quadratic_rate * previous
    if use_trade:
        q[2 * n :] = cost.linear_rate
    with np.errstate(over="ignore", invalid="ignore"):
        constant = float(
            quadratic_rate * (previous @ previous + problem.exit_squared_trade)
            + cost.linear_rate * problem.exit_turnover
        )
    if not (np.isfinite(q).all() and np.isfinite(constant)):
        raise OptimizerError("linear objective or constant overflowed or became non-finite")

    # OSQP's stopping criteria are absolute/relative in the submitted units.
    # Normalize the complete objective so covariance measured at 1e-12 is not
    # mistaken for a flat problem merely because constraints are order one.
    p_characteristic = float(np.max(np.abs(p.data))) if p.nnz else 0.0
    q_characteristic = float(np.max(np.abs(q))) if q.size else 0.0
    characteristic = max(p_characteristic, q_characteristic)
    if not np.isfinite(characteristic) or characteristic <= 0.0:
        raise OptimizerError("objective has no finite positive numerical scale")
    objective_scale = (
        1.0 / characteristic if characteristic < OBJECTIVE_NORMALIZATION_THRESHOLD else 1.0
    )
    if not np.isfinite(objective_scale):
        raise OptimizerError("objective scale is below float64's supported range")
    p = sparse.csc_matrix(p * objective_scale)
    q = np.asarray(q * objective_scale, dtype=np.float64)
    constant *= objective_scale
    if not (np.isfinite(p.data).all() and np.isfinite(q).all() and np.isfinite(constant)):
        raise OptimizerError("normalized objective overflowed or became non-finite")

    identity = sparse.eye(n, format="csc")
    zero = sparse.csc_matrix((n, n))
    trade_zero = sparse.csc_matrix((n, n)) if use_trade else None
    blocks: list[sparse.csc_matrix] = []
    lowers: list[FloatArray] = []
    uppers: list[FloatArray] = []

    def add(matrix: sparse.csc_matrix, lower: Any, upper: Any) -> None:
        rows = matrix.shape[0]
        lower_array = np.broadcast_to(np.asarray(lower, dtype=np.float64), (rows,)).copy()
        upper_array = np.broadcast_to(np.asarray(upper, dtype=np.float64), (rows,)).copy()
        blocks.append(matrix)
        lowers.append(lower_array)
        uppers.append(upper_array)

    caps = np.full(n, problem.constraints.max_position, dtype=np.float64)
    liquidity = problem.caps()
    if liquidity is not None:
        caps = np.minimum(caps, liquidity)
    weight_lower = np.zeros(n) if problem.constraints.long_only else -caps
    add(
        _hstack_variables(identity, zero, trade_zero),
        weight_lower,
        caps,
    )

    # g >= |w|, g >= 0, and sum(g) <= gross/leverage limit.
    add(_hstack_variables(identity, -identity, trade_zero), -np.inf, 0.0)
    add(_hstack_variables(-identity, -identity, trade_zero), -np.inf, 0.0)
    add(_hstack_variables(zero, identity, trade_zero), 0.0, np.inf)
    gross_limit = _gross_limit(problem.constraints)
    gross_row = sparse.csc_matrix(np.ones((1, n)))
    zero_row = sparse.csc_matrix((1, n))
    trade_zero_row = sparse.csc_matrix((1, n)) if use_trade else None
    add(
        _hstack_variables(zero_row, gross_row, trade_zero_row),
        -np.inf,
        gross_limit,
    )

    ones = sparse.csc_matrix(np.ones((1, n)))
    add(
        _hstack_variables(ones, zero_row, trade_zero_row),
        -problem.constraints.max_net,
        problem.constraints.max_net,
    )
    if problem.budget is not None:
        add(
            _hstack_variables(ones, zero_row, trade_zero_row),
            problem.budget,
            problem.budget,
        )

    for exposure in problem.exposures:
        loading = sparse.csc_matrix(exposure.loadings.reshape(1, -1))
        add(
            _hstack_variables(loading, zero_row, trade_zero_row),
            exposure.lower,
            exposure.upper,
        )

    if formulation == "target_return":
        assert problem.target_return is not None
        target = sparse.csc_matrix(problem.alpha().reshape(1, -1))
        add(
            _hstack_variables(target, zero_row, trade_zero_row),
            problem.target_return,
            np.inf,
        )

    if use_trade:
        assert trade_zero is not None
        # t >= w-p and t >= -(w-p).
        add(_hstack_variables(identity, zero, -identity), -np.inf, previous)
        add(_hstack_variables(-identity, zero, -identity), -np.inf, -previous)
        add(_hstack_variables(zero, zero, identity), 0.0, np.inf)
        if problem.constraints.max_turnover is not None:
            remaining = problem.constraints.max_turnover - problem.exit_turnover
            add(
                _hstack_variables(zero_row, zero_row, sparse.csc_matrix(np.ones((1, n)))),
                -np.inf,
                remaining,
            )

    a = sparse.vstack(blocks, format="csc")
    lower = np.concatenate(lowers)
    upper = np.concatenate(uppers)
    return _CanonicalQP(
        p=p,
        p_upper=sparse.triu(p, format="csc"),
        q=q,
        a=a,
        lower=lower,
        upper=upper,
        constant=constant,
        n_weights=n,
        n_gross=n,
        n_trade=n_trade,
        objective_scale=objective_scale,
    )


def _preflight(problem: MeanVarianceProblem, formulation: Formulation) -> None:
    """Reject provably invalid or infeasible requests before solver setup."""
    n = len(problem.assets)
    if formulation in {"target_return", "maximum_utility", "alpha_risk_cost"} and (
        problem.expected_returns is None
    ):
        raise OptimizerError(f"formulation {formulation!r} requires expected_returns")
    if formulation == "target_return" and problem.target_return is None:
        raise OptimizerError("formulation 'target_return' requires a target_return")
    if formulation == "minimum_variance" and problem.budget is None:
        raise OptimizerError("formulation 'minimum_variance' requires a budget equality")

    caps = np.full(n, problem.constraints.max_position, dtype=np.float64)
    if (liquidity := problem.caps()) is not None:
        caps = np.minimum(caps, liquidity)
    lower = np.zeros(n) if problem.constraints.long_only else -caps
    if problem.budget is not None and not (
        float(np.sum(lower)) - _audit_tolerance(problem.budget)
        <= problem.budget
        <= float(np.sum(caps)) + _audit_tolerance(problem.budget)
    ):
        raise OptimizerError(
            "constraint set is infeasible: position/liquidity bounds cannot satisfy budget"
        )
    if (
        problem.constraints.max_turnover is not None
        and problem.exit_turnover
        > problem.constraints.max_turnover + _audit_tolerance(problem.constraints.max_turnover)
    ):
        raise OptimizerError(
            "constraint set is infeasible: mandatory universe exits exceed max_turnover"
        )


def _solve_identity(
    problem: MeanVarianceProblem,
    formulation: Formulation,
    *,
    max_iterations: int,
    tolerance: float,
    effective_tolerance: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"alphaforge-certified-osqp-solve-v2\0")
    for value in (
        problem.identity,
        formulation,
        osqp.__version__,
        str(max_iterations),
        tolerance.hex(),
        effective_tolerance.hex(),
        MAX_SOLVER_SECONDS.hex(),
        "adaptive_rho=true",
        "adaptive_rho_interval=25",
        "polishing=true",
        "warm_starting=false",
        "scaled_termination=false",
        f"objective_normalization_threshold={OBJECTIVE_NORMALIZATION_THRESHOLD.hex()}",
        "objective_normalization_target=1.0",
    ):
        _hash_text(digest, value)
    return digest.hexdigest()


def _objective_parts(
    weights: FloatArray,
    problem: MeanVarianceProblem,
    formulation: Formulation,
) -> tuple[float, float, float, float]:
    """Independently reconstruct objective terms and actual expected return."""
    variance = problem.risk_model.portfolio_variance(weights)
    actual_return = float(problem.alpha() @ weights)
    if formulation in {"minimum_variance", "target_return"}:
        return 0.5 * variance, 0.0, 0.0, actual_return
    variance_term = 0.5 * problem.risk_aversion * variance
    if formulation == "maximum_utility":
        return variance_term, actual_return, 0.0, actual_return
    return variance_term, actual_return, problem.cost(weights), actual_return


def audit_solution(
    weights: FloatArray,
    problem: MeanVarianceProblem,
    *,
    formulation: Formulation | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Independently audit all original portfolio constraints.

    This function does not inspect the QP's auxiliary variables or solver state.
    It therefore verifies the actual financial quantities rather than accepting
    a potentially slack absolute-value epigraph.
    """
    if not isinstance(problem, MeanVarianceProblem):
        raise OptimizerError("problem must be a validated MeanVarianceProblem")
    try:
        vector = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerError("audit weights cannot be converted to float64") from exc
    if vector.shape != (len(problem.assets),):
        return False, ("weights do not align with the risk-model assets",)
    if not np.isfinite(vector).all():
        return False, ("weights are not finite",)

    violations: list[str] = []
    constraints = problem.constraints
    if problem.budget is not None:
        drift = abs(float(np.sum(vector)) - problem.budget)
        if drift > _audit_tolerance(problem.budget):
            violations.append(f"budget breached by {drift:.3e}")

    position_excess = float(np.max(np.abs(vector)) - constraints.max_position)
    if position_excess > _audit_tolerance(constraints.max_position):
        violations.append(f"max_position breached by {position_excess:.3e}")

    gross = float(np.sum(np.abs(vector)))
    gross_limit = constraints.max_gross * constraints.deployable
    leverage_limit = constraints.max_leverage * constraints.deployable
    if gross - gross_limit > _audit_tolerance(gross_limit):
        violations.append(f"max_gross breached by {gross - gross_limit:.3e}")
    if gross - leverage_limit > _audit_tolerance(leverage_limit):
        violations.append(f"max_leverage breached by {gross - leverage_limit:.3e}")

    net = abs(float(np.sum(vector)))
    if net - constraints.max_net > _audit_tolerance(constraints.max_net):
        violations.append(f"max_net breached by {net - constraints.max_net:.3e}")
    book_scale = max(gross, abs(problem.budget or 0.0))
    if constraints.long_only and float(np.min(vector)) < -_audit_tolerance(book_scale):
        violations.append(f"long_only breached by {abs(float(np.min(vector))):.3e}")

    turnover = problem.turnover(vector)
    if (
        constraints.max_turnover is not None
        and turnover - constraints.max_turnover > _audit_tolerance(constraints.max_turnover)
    ):
        violations.append(f"max_turnover breached by {turnover - constraints.max_turnover:.3e}")

    caps = problem.caps()
    if caps is not None:
        liquidity_excess_by_asset = np.abs(vector) - caps
        liquidity_tolerances = np.maximum(
            FLOAT_ROUNDOFF_FLOOR,
            AUDIT_TOLERANCE * np.maximum(np.abs(vector), caps),
        )
        liquidity_excess = float(np.max(liquidity_excess_by_asset))
        if np.any(liquidity_excess_by_asset > liquidity_tolerances):
            violations.append(f"liquidity cap breached by {liquidity_excess:.3e}")

    for exposure in problem.exposures:
        achieved = exposure.value(vector)
        amount = float(max(exposure.lower - achieved, achieved - exposure.upper, 0.0))
        exposure_scale = max(
            abs(exposure.lower),
            abs(exposure.upper),
            float(np.sum(np.abs(exposure.loadings * vector))),
        )
        if amount > _audit_tolerance(exposure_scale):
            violations.append(f"exposure {exposure.name!r} breached by {amount:.3e}")

    if formulation == "target_return":
        assert problem.target_return is not None
        alpha_contributions = problem.alpha() * vector
        achieved_return = float(np.sum(alpha_contributions))
        shortfall = problem.target_return - achieved_return
        target_scale = max(
            abs(problem.target_return),
            float(np.sum(np.abs(alpha_contributions))),
        )
        if shortfall > _audit_tolerance(target_scale):
            violations.append(f"target_return breached by {shortfall:.3e}")

    try:
        problem.risk_model.portfolio_variance(vector)
    except RiskModelError as exc:
        violations.append(f"risk model rejected the solution: {exc}")
    return not violations, tuple(violations)


def _certificate(
    qp: _CanonicalQP,
    z: FloatArray,
    dual: FloatArray,
    problem: MeanVarianceProblem,
    formulation: Formulation,
) -> _Certificate:
    """Independently reconstruct primal feasibility and the complete KKT system."""
    violations: list[str] = []
    if z.shape != qp.q.shape or dual.shape != qp.lower.shape:
        return _Certificate(
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            False,
            False,
            ("solver returned dimensionally inconsistent primal/dual vectors",),
        )
    if not (np.isfinite(z).all() and np.isfinite(dual).all()):
        return _Certificate(
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            False,
            False,
            ("solver returned non-finite primal/dual values",),
        )

    az = np.asarray(qp.a @ z, dtype=np.float64)
    lower_violation = np.where(np.isfinite(qp.lower), np.maximum(qp.lower - az, 0.0), 0.0)
    upper_violation = np.where(np.isfinite(qp.upper), np.maximum(az - qp.upper, 0.0), 0.0)
    primal_residual = float(max(np.max(lower_violation), np.max(upper_violation)))

    pz = np.asarray(qp.p @ z, dtype=np.float64)
    dual_action = np.asarray(qp.a.T @ dual, dtype=np.float64)
    stationarity = np.asarray(pz + qp.q + dual_action, dtype=np.float64)
    stationarity_scale = max(
        float(np.max(np.abs(pz))),
        float(np.max(np.abs(qp.q))),
        float(np.max(np.abs(dual_action))),
        np.finfo(np.float64).tiny,
    )
    dual_residual = float(np.max(np.abs(stationarity))) / stationarity_scale

    complementarity = 0.0
    sign_violation = 0.0
    for value, achieved, lower, upper in zip(dual, az, qp.lower, qp.upper, strict=True):
        if value > 0.0:
            if not np.isfinite(upper):
                sign_violation = max(sign_violation, float(value))
            else:
                complementarity = max(complementarity, float(abs(value * (upper - achieved))))
        elif value < 0.0:
            if not np.isfinite(lower):
                sign_violation = max(sign_violation, float(-value))
            else:
                complementarity = max(complementarity, float(abs(value * (achieved - lower))))
    sign_residual = sign_violation / stationarity_scale

    primal_variable_objective = float(0.5 * z @ (qp.p @ z) + qp.q @ z)
    primal_objective = primal_variable_objective + qp.constant
    weights = np.asarray(z[: qp.n_weights], dtype=np.float64)
    variance_term, return_term, cost_term, _ = _objective_parts(weights, problem, formulation)
    reconstructed = (variance_term - return_term + cost_term) * qp.objective_scale
    objective_reference = max(
        abs(primal_variable_objective),
        abs(reconstructed - qp.constant),
        0.5 * float(np.sum(np.abs(z * pz))),
        abs(float(qp.q @ z)),
        np.finfo(np.float64).tiny,
    )
    objective_residual = abs(primal_objective - reconstructed) / objective_reference
    complementarity_residual = max(complementarity / objective_reference, sign_residual)

    support = 0.0
    dual_support_valid = True
    dual_zero_tolerance = KKT_TOLERANCE * max(
        float(np.max(np.abs(dual))),
        stationarity_scale,
        np.finfo(np.float64).tiny,
    )
    for value, lower, upper in zip(dual, qp.lower, qp.upper, strict=True):
        if abs(value) <= dual_zero_tolerance:
            # A first-order solver can leave a numerically insignificant dual on
            # the unavailable side of a one-sided row. It contributes less than
            # the declared certificate tolerance and is canonically zero.
            continue
        if value >= 0.0:
            if not np.isfinite(upper):
                dual_support_valid = False
                break
            support += upper * value
        else:
            if not np.isfinite(lower):
                dual_support_valid = False
                break
            support += lower * value
    if dual_support_valid:
        dual_variable_objective = float(-0.5 * z @ (qp.p @ z) - support)
        dual_objective = dual_variable_objective + qp.constant
        duality_reference = max(
            abs(primal_variable_objective),
            abs(dual_variable_objective),
            objective_reference,
            np.finfo(np.float64).tiny,
        )
        duality_gap = abs(primal_objective - dual_objective) / duality_reference
    else:
        duality_gap = float("inf")

    portfolio_passed, portfolio_violations = audit_solution(
        weights, problem, formulation=formulation
    )
    violations.extend(portfolio_violations)
    if primal_residual > KKT_TOLERANCE:
        violations.append(f"QP primal residual {primal_residual:.3e} exceeds tolerance")
    if dual_residual > KKT_TOLERANCE:
        violations.append(f"KKT stationarity residual {dual_residual:.3e} exceeds tolerance")
    if complementarity_residual > KKT_TOLERANCE:
        violations.append(
            f"KKT complementarity residual {complementarity_residual:.3e} exceeds tolerance"
        )
    if objective_residual > OBJECTIVE_TOLERANCE:
        violations.append(
            f"objective reconstruction residual {objective_residual:.3e} exceeds tolerance"
        )
    if duality_gap > KKT_TOLERANCE:
        violations.append(f"primal/dual gap {duality_gap:.3e} exceeds tolerance")
    kkt_passed = (
        primal_residual <= KKT_TOLERANCE
        and dual_residual <= KKT_TOLERANCE
        and complementarity_residual <= KKT_TOLERANCE
        and objective_residual <= OBJECTIVE_TOLERANCE
        and duality_gap <= KKT_TOLERANCE
    )
    return _Certificate(
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        complementarity_residual=complementarity_residual,
        objective_residual=objective_residual,
        duality_gap=duality_gap,
        portfolio_passed=portfolio_passed,
        kkt_passed=kkt_passed,
        violations=tuple(violations),
    )


def solve_mean_variance(
    problem: MeanVarianceProblem,
    *,
    formulation: Formulation = "maximum_utility",
    max_iterations: int = 5_000,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SolverResult:
    """Solve and independently certify one convex mean-variance problem.

    Complexity is ``O(n^2)`` storage for the dense covariance and sparse linear
    constraints around it. Iterations, wall time, dimensions (through the risk
    contract), and logs are bounded. Only OSQP's exact ``solved`` status plus a
    passing independent certificate returns ``optimal``.
    """
    if not isinstance(problem, MeanVarianceProblem):
        raise OptimizerError("problem must be a validated MeanVarianceProblem")
    if not isinstance(formulation, str) or formulation not in FORMULATIONS:
        raise OptimizerError(f"unknown formulation {formulation!r}")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise OptimizerError("max_iterations must be an integer")
    if not 1 <= max_iterations <= MAX_ITERATIONS:
        raise OptimizerError(f"max_iterations must be in [1, {MAX_ITERATIONS}]")
    tolerance_value = _finite_float(tolerance, name="tolerance")
    if not 0.0 < tolerance_value <= MAX_TOLERANCE:
        raise OptimizerError(f"tolerance must be finite and in (0, {MAX_TOLERANCE:.0e}]")
    typed_formulation: Formulation = formulation
    _preflight(problem, typed_formulation)
    qp = _build_qp(problem, typed_formulation)
    effective_tolerance = max(tolerance_value, OSQP_NUMERICAL_FLOOR)
    solver_identity = _solve_identity(
        problem,
        typed_formulation,
        max_iterations=max_iterations,
        tolerance=tolerance_value,
        effective_tolerance=effective_tolerance,
    )

    solver = osqp.OSQP()
    try:
        solver.setup(
            P=qp.p_upper,
            q=qp.q,
            A=qp.a,
            l=qp.lower,
            u=qp.upper,
            verbose=False,
            max_iter=max_iterations,
            eps_abs=effective_tolerance,
            eps_rel=effective_tolerance,
            eps_prim_inf=effective_tolerance,
            eps_dual_inf=effective_tolerance,
            adaptive_rho=True,
            adaptive_rho_interval=25,
            polishing=True,
            polish_refine_iter=10,
            warm_starting=False,
            scaled_termination=False,
            check_termination=1,
            time_limit=MAX_SOLVER_SECONDS,
        )
        raw = solver.solve(raise_error=False)
    except osqp.OSQPException as exc:
        raise OptimizerError(f"OSQP setup/solve failed: {exc}") from exc

    detail = str(raw.info.status).lower()
    status_value = int(raw.info.status_val)
    if status_value == 1:
        status: SolverStatus = "optimal"
    elif status_value == 3:
        status = "infeasible"
    elif status_value == 7:
        status = "max_iterations"
    else:
        # Inaccurate success/infeasibility, time limit, non-convex, interrupted,
        # and unsolved are all unusable and deliberately not reclassified.
        status = "failed"

    raw_z = np.asarray(raw.x, dtype=np.float64) if raw.x is not None else np.array([])
    raw_dual = np.asarray(raw.y, dtype=np.float64) if raw.y is not None else np.array([])
    if status == "optimal":
        certificate = _certificate(qp, raw_z, raw_dual, problem, typed_formulation)
        if not certificate.portfolio_passed or not certificate.kkt_passed:
            status = "failed"
        weights = np.asarray(raw_z[: len(problem.assets)], dtype=np.float64)
    else:
        weights = np.zeros(len(problem.assets), dtype=np.float64)
        certificate = _Certificate(
            primal_residual=float(getattr(raw.info, "prim_res", float("inf"))),
            dual_residual=float(getattr(raw.info, "dual_res", float("inf"))),
            complementarity_residual=float("inf"),
            objective_residual=float("inf"),
            duality_gap=float("inf"),
            portfolio_passed=False,
            kkt_passed=False,
            violations=(f"solver status is {detail!r}; no portfolio is certified",),
        )

    return _package(
        weights,
        problem,
        formulation=typed_formulation,
        status=status,
        iterations=int(raw.info.iter),
        max_iterations=max_iterations,
        tolerance=effective_tolerance,
        certificate=certificate,
        solver_identity=solver_identity,
        solver_status_detail=detail,
        solve_time_seconds=float(raw.info.run_time),
    )


def _package(
    weights: FloatArray,
    problem: MeanVarianceProblem,
    *,
    formulation: Formulation,
    status: SolverStatus,
    iterations: int,
    max_iterations: int,
    tolerance: float,
    certificate: _Certificate,
    solver_identity: str,
    solver_status_detail: str,
    solve_time_seconds: float,
) -> SolverResult:
    """Package only independently reconstructed financial quantities."""
    variance_term, return_term, cost_term, actual_return = _objective_parts(
        weights, problem, formulation
    )
    audit_passed = status == "optimal" and certificate.portfolio_passed and certificate.kkt_passed
    violations = certificate.violations
    return SolverResult(
        weights=pd.Series(weights, index=problem.assets, name="target_weight"),
        formulation=formulation,
        status=status,
        iterations=iterations,
        max_iterations=max_iterations,
        residual=certificate.residual,
        tolerance=tolerance,
        objective=variance_term - return_term + cost_term,
        variance_term=variance_term,
        return_term=return_term,
        cost_term=cost_term,
        ex_ante_volatility=problem.risk_model.annualized_volatility(weights),
        expected_return=actual_return,
        turnover=problem.turnover(weights),
        gross=float(np.sum(np.abs(weights))),
        net=float(np.sum(weights)),
        active_constraints=_active_constraints(weights, problem, formulation=formulation),
        problem_identity=problem.identity,
        risk_diagnostics=problem.risk_model.diagnostics(),
        audit_passed=audit_passed,
        audit_violations=violations,
        solver_identity=solver_identity,
        solver_version=osqp.__version__,
        solver_status_detail=solver_status_detail,
        primal_residual=certificate.primal_residual,
        dual_residual=certificate.dual_residual,
        complementarity_residual=certificate.complementarity_residual,
        objective_residual=certificate.objective_residual,
        duality_gap=certificate.duality_gap,
        kkt_passed=certificate.kkt_passed,
        exit_turnover=problem.exit_turnover,
        solve_time_seconds=solve_time_seconds,
        decision_timestamp=_optional_timestamp_iso(
            problem.decision_timestamp, name="decision_timestamp"
        ),
    )


def _active_constraints(
    weights: FloatArray,
    problem: MeanVarianceProblem,
    *,
    formulation: Formulation,
) -> tuple[str, ...]:
    """Return deterministic names for independently observed active limits."""
    vector = np.asarray(weights, dtype=np.float64)
    constraints = problem.constraints
    active: list[str] = []
    if np.any(
        np.abs(np.abs(vector) - constraints.max_position)
        <= _audit_tolerance(constraints.max_position)
    ):
        active.append("max_position")
    if problem.budget is not None:
        active.append("budget")
    gross = float(np.sum(np.abs(vector)))
    gross_limit = constraints.max_gross * constraints.deployable
    if abs(gross - gross_limit) <= _audit_tolerance(gross_limit):
        active.append("max_gross")
    leverage_limit = constraints.max_leverage * constraints.deployable
    if abs(gross - leverage_limit) <= _audit_tolerance(leverage_limit):
        active.append("max_leverage")
    if abs(abs(float(np.sum(vector))) - constraints.max_net) <= _audit_tolerance(
        constraints.max_net
    ):
        active.append("max_net")
    if constraints.long_only and np.any(np.abs(vector) <= _audit_tolerance(gross)):
        active.append("long_only")
    if constraints.max_turnover is not None and (
        abs(problem.turnover(vector) - constraints.max_turnover)
        <= _audit_tolerance(constraints.max_turnover)
    ):
        active.append("max_turnover")
    if (caps := problem.caps()) is not None:
        cap_tolerances = np.maximum(
            FLOAT_ROUNDOFF_FLOOR,
            AUDIT_TOLERANCE * np.maximum(np.abs(vector), caps),
        )
        if np.any(np.abs(np.abs(vector) - caps) <= cap_tolerances):
            active.append("liquidity")
    for exposure in problem.exposures:
        achieved = exposure.value(vector)
        exposure_scale = max(
            abs(exposure.lower),
            abs(exposure.upper),
            float(np.sum(np.abs(exposure.loadings * vector))),
        )
        if min(abs(achieved - exposure.lower), abs(achieved - exposure.upper)) <= _audit_tolerance(
            exposure_scale
        ):
            active.append(f"exposure:{exposure.name}")
    if (
        formulation == "target_return"
        and problem.target_return is not None
        and (
            abs(float(problem.alpha() @ vector) - problem.target_return)
            <= _audit_tolerance(
                problem.target_return,
                float(np.sum(np.abs(problem.alpha() * vector))),
            )
        )
    ):
        active.append("target_return")
    return tuple(dict.fromkeys(active))


def analytic_minimum_variance(risk_model: RiskModel) -> FloatArray:
    """Return ``Sigma^-1 1 / (1' Sigma^-1 1)`` for validation."""
    if not isinstance(risk_model, RiskModel):
        raise OptimizerError("risk_model must be a validated RiskModel")
    ones = np.ones(len(risk_model.assets), dtype=np.float64)
    try:
        solved = np.linalg.solve(risk_model.covariance, ones)
    except np.linalg.LinAlgError as exc:
        raise OptimizerError("covariance is singular; analytic reference unavailable") from exc
    total = float(ones @ solved)
    if abs(total) < 1e-300:
        raise OptimizerError("degenerate covariance; analytic reference unavailable")
    result = np.asarray(solved / total, dtype=np.float64)
    if not np.isfinite(result).all():
        raise OptimizerError("analytic minimum-variance reference became non-finite")
    return result


def analytic_maximum_utility(
    risk_model: RiskModel,
    alpha: FloatArray,
    risk_aversion: float,
) -> FloatArray:
    """Return ``(1/lambda) Sigma^-1 mu`` for unconstrained validation."""
    if not isinstance(risk_model, RiskModel):
        raise OptimizerError("risk_model must be a validated RiskModel")
    try:
        vector = np.asarray(alpha, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerError("alpha cannot be converted to float64") from exc
    if vector.shape != (len(risk_model.assets),) or not np.isfinite(vector).all():
        raise OptimizerError("alpha must be finite and align with the risk model")
    risk_aversion_value = _finite_float(risk_aversion, name="risk_aversion")
    if risk_aversion_value <= 0.0:
        raise OptimizerError("risk_aversion must be positive")
    try:
        solved = np.linalg.solve(risk_model.covariance, vector)
    except np.linalg.LinAlgError as exc:
        raise OptimizerError("covariance is singular; analytic reference unavailable") from exc
    result = np.asarray(solved / risk_aversion_value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise OptimizerError("analytic maximum-utility reference became non-finite")
    return result
