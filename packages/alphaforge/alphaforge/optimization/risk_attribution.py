"""Validated portfolio-risk, realized-ledger, and scenario attribution.

This module explains research-period risk and accounting. It does not model
orders, fills, broker state, taxes, or executable P&L. Inputs are finite,
explicitly labelled, bounded, and interpreted as periodic simple returns.

Every exported dataclass validates and canonicalizes direct construction.
Sequences become immutable tuples, labels and identities are checked, derived
fields are reconciled, and non-finite or overflowed arithmetic fails closed.
Reconciliation tolerances contain no fixed domain-unit allowance: their bound
depends on operand scale, operation count, IEEE-754 epsilon, and unit-in-last-
place spacing. Tiny portfolios therefore receive tiny tolerances.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphaforge.optimization.risk_model import FactorRiskModel, RiskModel

type FloatArray = NDArray[np.float64]

MAX_ATTRIBUTION_ASSETS = 2_000
MAX_SCENARIOS = 64
MAX_SCENARIO_NAME_LENGTH = 128
MAX_SCENARIO_SIMPLE_RETURN = 10.0
RECONCILIATION_RTOL = 1e-9
ROUNDOFF_SAFETY_FACTOR = 32.0
MAX_RECONCILIATION_OPERATIONS = 100_000
_FLOAT_EPSILON = np.finfo(np.float64).eps
_FLOAT_TINY = np.finfo(np.float64).tiny


class RiskAttributionError(ValueError):
    """Raised when attribution input, state, or reconciliation is invalid."""


def _finite_float(value: Any, *, name: str) -> float:
    """Return one finite real scalar, rejecting booleans and overflow."""
    if isinstance(value, (bool, np.bool_)):
        raise RiskAttributionError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise RiskAttributionError(f"{name} must be finite")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise RiskAttributionError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise RiskAttributionError(f"{name} must be non-negative")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RiskAttributionError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise RiskAttributionError(f"{name} must be a positive integer")
    return result


def _label(value: Any, *, name: str, maximum: int = MAX_SCENARIO_NAME_LENGTH) -> str:
    if not isinstance(value, str):
        raise RiskAttributionError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized or normalized != value or len(normalized) > maximum:
        raise RiskAttributionError(
            f"{name} must be a non-empty trimmed string no longer than {maximum} characters"
        )
    return normalized


def _labels(
    value: Any,
    *,
    name: str,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise RiskAttributionError(f"{name} must be a sequence of labels")
    try:
        items = tuple(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} must be a sequence of labels") from exc
    if (not allow_empty and not items) or len(items) > maximum:
        minimum = 0 if allow_empty else 1
        raise RiskAttributionError(f"{name} count must lie in [{minimum}, {maximum}]")
    normalized = tuple(_label(item, name=f"{name} label") for item in items)
    if len(set(normalized)) != len(normalized):
        raise RiskAttributionError(f"{name} labels must be unique")
    return normalized


def _identity(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskAttributionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _typed_tuple[T](
    value: Any,
    expected_type: type[T],
    *,
    name: str,
    maximum: int,
    allow_empty: bool = True,
) -> tuple[T, ...]:
    if isinstance(value, (str, bytes)):
        raise RiskAttributionError(f"{name} must be a sequence")
    try:
        items = tuple(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} must be a sequence") from exc
    if (not allow_empty and not items) or len(items) > maximum:
        minimum = 0 if allow_empty else 1
        raise RiskAttributionError(f"{name} count must lie in [{minimum}, {maximum}]")
    if any(not isinstance(item, expected_type) for item in items):
        raise RiskAttributionError(f"{name} contains an invalid record type")
    return items


def _finite_array(value: Any, *, name: str, size: int | None = None) -> FloatArray:
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} must be numeric") from exc
    if result.ndim != 1 or (size is not None and result.shape != (size,)):
        raise RiskAttributionError(f"{name} must be one-dimensional and aligned")
    if not np.isfinite(result).all():
        raise RiskAttributionError(f"{name} must be finite")
    return result


def _checked_sum(values: Iterable[float], *, name: str) -> float:
    try:
        result = float(math.fsum(float(value) for value in values))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} overflowed or is not numeric") from exc
    if not np.isfinite(result):
        raise RiskAttributionError(f"{name} produced a non-finite value")
    return result


def _checked_product(left: float, right: float, *, name: str) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.multiply(np.float64(left), np.float64(right)))
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} overflowed") from exc
    if not np.isfinite(result):
        raise RiskAttributionError(f"{name} produced a non-finite value")
    return result


def _checked_add(left: float, right: float, *, name: str) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.add(np.float64(left), np.float64(right)))
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} overflowed") from exc
    if not np.isfinite(result):
        raise RiskAttributionError(f"{name} produced a non-finite value")
    return result


def _checked_subtract(left: float, right: float, *, name: str) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.subtract(np.float64(left), np.float64(right)))
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} overflowed") from exc
    if not np.isfinite(result):
        raise RiskAttributionError(f"{name} produced a non-finite value")
    return result


def _checked_vector_product(left: FloatArray, right: FloatArray, *, name: str) -> FloatArray:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(left * right, dtype=np.float64)
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError(f"{name} overflowed") from exc
    if not np.isfinite(result).all():
        raise RiskAttributionError(f"{name} produced non-finite values")
    return result


def _roundoff_limit(
    actual: float,
    expected: float,
    *,
    term_magnitudes: Iterable[float],
    operations: int,
) -> float:
    """Return a scale- and operation-aware IEEE-754 reconciliation bound."""
    if not 1 <= operations <= MAX_RECONCILIATION_OPERATIONS:
        raise RiskAttributionError("reconciliation operation count is outside its bound")
    magnitudes = tuple(
        abs(_finite_float(item, name="reconciliation term")) for item in term_magnitudes
    )
    sum_abs = _checked_sum(magnitudes, name="reconciliation magnitude")
    relative_scale = max(abs(actual), abs(expected), _FLOAT_TINY)
    arithmetic = ROUNDOFF_SAFETY_FACTOR * _FLOAT_EPSILON * operations * sum_abs
    spacing_scale = max(relative_scale, sum_abs, _FLOAT_TINY)
    spacing = ROUNDOFF_SAFETY_FACTOR * operations * abs(float(np.spacing(spacing_scale)))
    limit = RECONCILIATION_RTOL * relative_scale + arithmetic + spacing
    if not np.isfinite(limit):
        raise RiskAttributionError("reconciliation tolerance overflowed")
    return float(limit)


def _reconciliation_error(
    actual: float,
    expected: float,
    *,
    name: str,
    term_magnitudes: Iterable[float],
    operations: int,
) -> float:
    actual_value = _finite_float(actual, name=f"{name} actual")
    expected_value = _finite_float(expected, name=f"{name} expected")
    error = _checked_subtract(actual_value, expected_value, name=f"{name} error")
    limit = _roundoff_limit(
        actual_value,
        expected_value,
        term_magnitudes=term_magnitudes,
        operations=operations,
    )
    if abs(error) > limit:
        raise RiskAttributionError(
            f"{name} did not reconcile: actual={actual_value:.17g}, "
            f"expected={expected_value:.17g}, error={error:.3e}, bound={limit:.3e}"
        )
    return error


def _reconcile_sum(values: Iterable[float], expected: float, *, name: str) -> float:
    terms = tuple(_finite_float(item, name=f"{name} term") for item in values)
    actual = _checked_sum(terms, name=name)
    return _reconciliation_error(
        actual,
        expected,
        name=name,
        term_magnitudes=terms,
        operations=max(len(terms) - 1, 1),
    )


def _reconcile_product(left: float, right: float, expected: float, *, name: str) -> float:
    actual = _checked_product(left, right, name=name)
    return _reconciliation_error(
        actual,
        expected,
        name=name,
        term_magnitudes=(actual, expected),
        operations=1,
    )


def _canonical_error(recorded: Any, computed: float, *, name: str, scale: float = 0.0) -> float:
    """Verify a stored reconciliation residual against a freshly recomputed one.

    Both operands are *roundoff residuals*, so their own magnitude is noise and
    is not a meaningful tolerance scale. Judging them against each other with a
    purely relative bound is structurally unsatisfiable: a residual of 1.9e-9
    earns a bound of 1.9e-18, and any two independently accumulated float paths
    that legitimately differ in their last bits are then reported as a
    reconciliation failure.

    ``scale`` is the magnitude of the quantity whose reconciliation produced the
    residual — a variance, a return, an equity value. The residual of
    reconciling ``X`` is only interpretable relative to ``|X|``, so that is what
    bounds the comparison. Passing ``0.0`` preserves the strict
    residual-relative behaviour for fields where no underlying scale exists.
    """
    recorded_value = _finite_float(recorded, name=name)
    scale_value = abs(_finite_float(scale, name=f"{name} scale"))
    _reconciliation_error(
        recorded_value,
        computed,
        name=f"{name} field",
        term_magnitudes=(recorded_value, computed, scale_value),
        operations=1,
    )
    return computed


def _weight_vector(
    weights: pd.Series,
    *,
    expected_assets: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], FloatArray]:
    if not isinstance(weights, pd.Series):
        raise RiskAttributionError("weights must be a pandas Series indexed by asset")
    if weights.index.has_duplicates:
        raise RiskAttributionError("weights must have a unique asset index")
    assets = _labels(
        tuple(weights.index),
        name="weight assets",
        maximum=MAX_ATTRIBUTION_ASSETS,
    )
    if expected_assets is not None and assets != expected_assets:
        raise RiskAttributionError(
            "weights must exactly match the risk-model asset order; explicit reindexing is required"
        )
    values = _finite_array(weights.to_numpy(), name="weights", size=len(assets))
    return assets, values


def _return_vector(
    returns: pd.Series,
    *,
    assets: tuple[str, ...],
    name: str,
    maximum: float | None = None,
) -> FloatArray:
    if not isinstance(returns, pd.Series):
        raise RiskAttributionError(f"{name} must be a pandas Series indexed by asset")
    if returns.index.has_duplicates:
        raise RiskAttributionError(f"{name} must have a unique asset index")
    if tuple(returns.index) != assets:
        raise RiskAttributionError(
            f"{name} must exactly match the weight asset order; explicit reindexing is required"
        )
    values = _finite_array(returns.to_numpy(), name=name, size=len(assets))
    if (values < -1.0).any():
        raise RiskAttributionError(f"{name} contains a simple return below -1")
    if maximum is not None and (values > maximum).any():
        raise RiskAttributionError(f"{name} exceeds the declared maximum simple return {maximum}")
    return values


def _cash_return(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < -1.0 or result > MAX_SCENARIO_SIMPLE_RETURN:
        raise RiskAttributionError(f"{name} must lie in [-1, {MAX_SCENARIO_SIMPLE_RETURN}]")
    return result


@dataclass(frozen=True, slots=True)
class AssetRiskContribution:
    """One asset's Euler contribution in periodic risk units."""

    asset: str
    weight: float
    marginal_variance: float
    component_variance: float
    marginal_volatility: float
    component_volatility: float
    variance_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _label(self.asset, name="asset"))
        for field_name in (
            "weight",
            "marginal_variance",
            "component_variance",
            "marginal_volatility",
            "component_volatility",
            "variance_fraction",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), name=field_name),
            )
        _reconcile_product(
            self.weight,
            self.marginal_variance,
            self.component_variance,
            name=f"asset {self.asset!r} component variance",
        )


@dataclass(frozen=True, slots=True)
class FactorRiskContribution:
    """One factor's exposure and Euler contribution to portfolio risk."""

    factor: str
    exposure: float
    marginal_variance: float
    component_variance: float
    component_volatility: float
    variance_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", _label(self.factor, name="factor"))
        for field_name in (
            "exposure",
            "marginal_variance",
            "component_variance",
            "component_volatility",
            "variance_fraction",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), name=field_name),
            )
        _reconcile_product(
            self.exposure,
            self.marginal_variance,
            self.component_variance,
            name=f"factor {self.factor!r} component variance",
        )


@dataclass(frozen=True, slots=True)
class SpecificRiskContribution:
    """One asset's diagonal specific-risk contribution."""

    asset: str
    specific_variance: float
    component_variance: float
    component_volatility: float
    variance_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _label(self.asset, name="asset"))
        object.__setattr__(
            self,
            "specific_variance",
            _nonnegative_float(self.specific_variance, name="specific_variance"),
        )
        for field_name in ("component_variance", "component_volatility", "variance_fraction"):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), name=field_name),
            )
        if self.component_variance < 0.0:
            raise RiskAttributionError("specific component variance must be non-negative")


@dataclass(frozen=True, slots=True)
class ExAnteRiskAttribution:
    """Immutable, fully reconciled asset and optional factor-risk decomposition."""

    risk_model_identity: str
    assets: tuple[str, ...]
    periods_per_year: int
    portfolio_variance: float
    portfolio_volatility: float
    annualized_volatility: float
    variance_reconciliation_error: float
    volatility_reconciliation_error: float
    factor_variance: float | None
    specific_variance: float | None
    factor_specific_reconciliation_error: float | None
    asset_contributions: tuple[AssetRiskContribution, ...]
    factor_contributions: tuple[FactorRiskContribution, ...]
    specific_contributions: tuple[SpecificRiskContribution, ...]

    def __post_init__(self) -> None:
        identity = _identity(self.risk_model_identity, name="risk_model_identity")
        assets = _labels(
            self.assets,
            name="assets",
            maximum=MAX_ATTRIBUTION_ASSETS,
        )
        periods = _positive_int(self.periods_per_year, name="periods_per_year")
        variance = _nonnegative_float(self.portfolio_variance, name="portfolio_variance")
        volatility = _nonnegative_float(self.portfolio_volatility, name="portfolio_volatility")
        annualized = _nonnegative_float(self.annualized_volatility, name="annualized_volatility")
        asset_records = _typed_tuple(
            self.asset_contributions,
            AssetRiskContribution,
            name="asset_contributions",
            maximum=MAX_ATTRIBUTION_ASSETS,
            allow_empty=False,
        )
        factor_records = _typed_tuple(
            self.factor_contributions,
            FactorRiskContribution,
            name="factor_contributions",
            maximum=64,
        )
        specific_records = _typed_tuple(
            self.specific_contributions,
            SpecificRiskContribution,
            name="specific_contributions",
            maximum=MAX_ATTRIBUTION_ASSETS,
        )
        if tuple(item.asset for item in asset_records) != assets:
            raise RiskAttributionError("asset contributions must exactly match asset order")
        if len({item.factor for item in factor_records}) != len(factor_records):
            raise RiskAttributionError("factor contribution labels must be unique")
        if specific_records and tuple(item.asset for item in specific_records) != assets:
            raise RiskAttributionError("specific contributions must exactly match asset order")

        _reconcile_product(
            volatility,
            volatility,
            variance,
            name="portfolio volatility squared",
        )
        expected_annualized = _checked_product(
            volatility,
            float(np.sqrt(periods)),
            name="annualized volatility",
        )
        _reconciliation_error(
            annualized,
            expected_annualized,
            name="annualized volatility",
            term_magnitudes=(annualized, expected_annualized),
            operations=2,
        )
        variance_error = _reconcile_sum(
            (item.component_variance for item in asset_records),
            variance,
            name="asset component variance",
        )
        volatility_error = _reconcile_sum(
            (item.component_volatility for item in asset_records),
            volatility,
            name="asset component volatility",
        )
        for item in asset_records:
            if volatility > 0.0:
                _reconciliation_error(
                    item.marginal_volatility,
                    item.marginal_variance / volatility,
                    name=f"asset {item.asset!r} marginal volatility",
                    term_magnitudes=(item.marginal_volatility, item.marginal_variance / volatility),
                    operations=1,
                )
                _reconciliation_error(
                    item.component_volatility,
                    item.component_variance / volatility,
                    name=f"asset {item.asset!r} component volatility",
                    term_magnitudes=(
                        item.component_volatility,
                        item.component_variance / volatility,
                    ),
                    operations=1,
                )
                _reconciliation_error(
                    item.variance_fraction,
                    item.component_variance / variance,
                    name=f"asset {item.asset!r} variance fraction",
                    term_magnitudes=(item.variance_fraction, item.component_variance / variance),
                    operations=1,
                )
            elif any(
                value != 0.0
                for value in (
                    item.marginal_volatility,
                    item.component_volatility,
                    item.variance_fraction,
                    item.component_variance,
                )
            ):
                raise RiskAttributionError("zero-risk books require zero risk contributions")

        factor_error: float | None = None
        factor_variance: float | None
        specific_variance: float | None
        if not factor_records and not specific_records:
            if any(
                value is not None
                for value in (
                    self.factor_variance,
                    self.specific_variance,
                    self.factor_specific_reconciliation_error,
                )
            ):
                raise RiskAttributionError(
                    "covariance-only attribution cannot contain factor state"
                )
            factor_variance = None
            specific_variance = None
        else:
            if not factor_records or len(specific_records) != len(assets):
                raise RiskAttributionError(
                    "factor attribution requires factor and asset-specific contributions"
                )
            factor_variance = _nonnegative_float(
                self.factor_variance,
                name="factor_variance",
            )
            specific_variance = _nonnegative_float(
                self.specific_variance,
                name="specific_variance",
            )
            _reconcile_sum(
                (item.component_variance for item in factor_records),
                factor_variance,
                name="factor component variance",
            )
            _reconcile_sum(
                (item.component_variance for item in specific_records),
                specific_variance,
                name="specific component variance",
            )
            factor_error = _reconcile_sum(
                (factor_variance, specific_variance),
                variance,
                name="factor plus specific variance",
            )
            _reconcile_sum(
                (
                    *(item.component_volatility for item in factor_records),
                    *(item.component_volatility for item in specific_records),
                ),
                volatility,
                name="factor plus specific volatility",
            )
        object.__setattr__(self, "risk_model_identity", identity)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "periods_per_year", periods)
        object.__setattr__(self, "portfolio_variance", variance)
        object.__setattr__(self, "portfolio_volatility", volatility)
        object.__setattr__(self, "annualized_volatility", annualized)
        object.__setattr__(
            self,
            "variance_reconciliation_error",
            _canonical_error(
                self.variance_reconciliation_error,
                variance_error,
                name="variance_reconciliation_error",
                scale=variance,
            ),
        )
        object.__setattr__(
            self,
            "volatility_reconciliation_error",
            _canonical_error(
                self.volatility_reconciliation_error,
                volatility_error,
                name="volatility_reconciliation_error",
                scale=volatility,
            ),
        )
        object.__setattr__(self, "factor_variance", factor_variance)
        object.__setattr__(self, "specific_variance", specific_variance)
        if factor_error is None:
            object.__setattr__(self, "factor_specific_reconciliation_error", None)
        else:
            object.__setattr__(
                self,
                "factor_specific_reconciliation_error",
                _canonical_error(
                    self.factor_specific_reconciliation_error,
                    factor_error,
                    name="factor_specific_reconciliation_error",
                    scale=variance,
                ),
            )
        object.__setattr__(self, "asset_contributions", asset_records)
        object.__setattr__(self, "factor_contributions", factor_records)
        object.__setattr__(self, "specific_contributions", specific_records)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def asset_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(item) for item in self.asset_contributions)

    def factor_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(item) for item in self.factor_contributions)


def attribute_ex_ante_risk(
    model: RiskModel | FactorRiskModel,
    weights: pd.Series,
) -> ExAnteRiskAttribution:
    """Compute reconciled Euler asset and optional factor/specific risk."""
    factor_model: FactorRiskModel | None
    if isinstance(model, FactorRiskModel):
        factor_model = model
        risk_model = model.risk_model
    elif isinstance(model, RiskModel):
        factor_model = None
        risk_model = model
    else:
        raise RiskAttributionError("model must be a validated RiskModel or FactorRiskModel")
    assets, vector = _weight_vector(weights, expected_assets=risk_model.assets)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            covariance_times_weights = np.asarray(
                risk_model.covariance @ vector,
                dtype=np.float64,
            )
            component_variance = vector * covariance_times_weights
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError("asset risk attribution overflowed") from exc
    if not (np.isfinite(covariance_times_weights).all() and np.isfinite(component_variance).all()):
        raise RiskAttributionError("asset risk attribution produced non-finite values")
    portfolio_variance = _nonnegative_float(
        risk_model.portfolio_variance(vector),
        name="portfolio_variance",
    )
    portfolio_volatility = float(np.sqrt(portfolio_variance))
    if portfolio_volatility > 0.0:
        marginal_volatility = covariance_times_weights / portfolio_volatility
        component_volatility = component_variance / portfolio_volatility
        fractions = component_variance / portfolio_variance
    else:
        marginal_volatility = np.zeros_like(vector)
        component_volatility = np.zeros_like(vector)
        fractions = np.zeros_like(vector)
    variance_error = _reconcile_sum(
        component_variance,
        portfolio_variance,
        name="asset component variance",
    )
    volatility_error = _reconcile_sum(
        component_volatility,
        portfolio_volatility,
        name="asset component volatility",
    )
    asset_contributions = tuple(
        AssetRiskContribution(
            asset=asset,
            weight=float(vector[index]),
            marginal_variance=float(covariance_times_weights[index]),
            component_variance=float(component_variance[index]),
            marginal_volatility=float(marginal_volatility[index]),
            component_volatility=float(component_volatility[index]),
            variance_fraction=float(fractions[index]),
        )
        for index, asset in enumerate(assets)
    )

    factor_variance: float | None = None
    specific_variance: float | None = None
    factor_error: float | None = None
    factor_contributions: tuple[FactorRiskContribution, ...] = ()
    specific_contributions: tuple[SpecificRiskContribution, ...] = ()
    if factor_model is not None:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                exposures = factor_model.factor_exposures(vector)
                factor_marginal = np.asarray(
                    factor_model.factor_covariance @ exposures,
                    dtype=np.float64,
                )
                factor_components = exposures * factor_marginal
                specific_components = vector**2 * factor_model.specific_variances
        except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
            raise RiskAttributionError("factor risk attribution overflowed") from exc
        if not all(
            np.isfinite(value).all()
            for value in (
                exposures,
                factor_marginal,
                factor_components,
                specific_components,
            )
        ):
            raise RiskAttributionError("factor risk attribution produced non-finite values")
        factor_variance = _checked_sum(factor_components, name="factor variance")
        specific_variance = _checked_sum(specific_components, name="specific variance")
        factor_error = _reconcile_sum(
            (factor_variance, specific_variance),
            portfolio_variance,
            name="factor plus specific variance",
        )
        if portfolio_volatility > 0.0:
            factor_volatility = factor_components / portfolio_volatility
            specific_volatility = specific_components / portfolio_volatility
            factor_fractions = factor_components / portfolio_variance
            specific_fractions = specific_components / portfolio_variance
        else:
            factor_volatility = np.zeros_like(factor_components)
            specific_volatility = np.zeros_like(specific_components)
            factor_fractions = np.zeros_like(factor_components)
            specific_fractions = np.zeros_like(specific_components)
        _reconcile_sum(
            (*factor_volatility, *specific_volatility),
            portfolio_volatility,
            name="factor plus specific volatility",
        )
        factor_contributions = tuple(
            FactorRiskContribution(
                factor=factor,
                exposure=float(exposures[index]),
                marginal_variance=float(factor_marginal[index]),
                component_variance=float(factor_components[index]),
                component_volatility=float(factor_volatility[index]),
                variance_fraction=float(factor_fractions[index]),
            )
            for index, factor in enumerate(factor_model.factors)
        )
        specific_contributions = tuple(
            SpecificRiskContribution(
                asset=asset,
                specific_variance=float(factor_model.specific_variances[index]),
                component_variance=float(specific_components[index]),
                component_volatility=float(specific_volatility[index]),
                variance_fraction=float(specific_fractions[index]),
            )
            for index, asset in enumerate(assets)
        )
    annualized = _checked_product(
        portfolio_volatility,
        float(np.sqrt(risk_model.periods_per_year)),
        name="annualized volatility",
    )
    return ExAnteRiskAttribution(
        risk_model_identity=(
            factor_model.identity if factor_model is not None else risk_model.identity
        ),
        assets=assets,
        periods_per_year=risk_model.periods_per_year,
        portfolio_variance=portfolio_variance,
        portfolio_volatility=portfolio_volatility,
        annualized_volatility=annualized,
        variance_reconciliation_error=variance_error,
        volatility_reconciliation_error=volatility_error,
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        factor_specific_reconciliation_error=factor_error,
        asset_contributions=asset_contributions,
        factor_contributions=factor_contributions,
        specific_contributions=specific_contributions,
    )


@dataclass(frozen=True, slots=True)
class RealizedAssetContribution:
    """One asset's observed simple-return and currency-P&L contribution."""

    asset: str
    starting_weight: float
    asset_return: float
    return_contribution: float
    pnl_contribution: float

    def __post_init__(self) -> None:
        asset = _label(self.asset, name="asset")
        weight = _finite_float(self.starting_weight, name="starting_weight")
        asset_return = _cash_return(self.asset_return, name="asset_return")
        return_contribution = _finite_float(
            self.return_contribution,
            name="return_contribution",
        )
        pnl_contribution = _finite_float(self.pnl_contribution, name="pnl_contribution")
        _reconcile_product(
            weight,
            asset_return,
            return_contribution,
            name=f"asset {asset!r} realized return contribution",
        )
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "starting_weight", weight)
        object.__setattr__(self, "asset_return", asset_return)
        object.__setattr__(self, "return_contribution", return_contribution)
        object.__setattr__(self, "pnl_contribution", pnl_contribution)


@dataclass(frozen=True, slots=True)
class RealizedAttribution:
    """Observed ending-equity attribution with explicit cash and costs.

    The observed ending equity is supplied independently by the caller. Asset,
    cash, and cost contributions must reproduce both its return and P&L, while
    a separate ending-value path must reproduce the same ledger value.
    """

    assets: tuple[str, ...]
    initial_capital: float
    starting_cash_weight: float
    cash_return: float
    cost_return: float
    observed_ending_equity: float
    modeled_ending_equity: float
    gross_return: float
    net_return: float
    gross_pnl: float
    cost_pnl: float
    net_pnl: float
    cash_return_contribution: float
    cash_pnl_contribution: float
    cost_return_contribution: float
    cost_pnl_contribution: float
    return_reconciliation_error: float
    pnl_reconciliation_error: float
    ending_equity_reconciliation_error: float
    asset_contributions: tuple[RealizedAssetContribution, ...]

    def __post_init__(self) -> None:
        assets = _labels(
            self.assets,
            name="assets",
            maximum=MAX_ATTRIBUTION_ASSETS,
        )
        capital = _positive_float(self.initial_capital, name="initial_capital")
        cash_weight = _finite_float(self.starting_cash_weight, name="starting_cash_weight")
        cash_return = _cash_return(self.cash_return, name="cash_return")
        cost_return = _nonnegative_float(self.cost_return, name="cost_return")
        observed_ending = _positive_float(
            self.observed_ending_equity,
            name="observed_ending_equity",
        )
        modeled_ending = _finite_float(
            self.modeled_ending_equity,
            name="modeled_ending_equity",
        )
        records = _typed_tuple(
            self.asset_contributions,
            RealizedAssetContribution,
            name="asset_contributions",
            maximum=MAX_ATTRIBUTION_ASSETS,
            allow_empty=False,
        )
        if tuple(item.asset for item in records) != assets:
            raise RiskAttributionError("asset contributions must exactly match asset order")
        _reconcile_sum(
            (*[item.starting_weight for item in records], cash_weight),
            1.0,
            name="starting asset weights plus cash",
        )
        for item in records:
            _reconcile_product(
                item.return_contribution,
                capital,
                item.pnl_contribution,
                name=f"asset {item.asset!r} P&L contribution",
            )
        expected_cash_return = _checked_product(
            cash_weight,
            cash_return,
            name="cash return contribution",
        )
        expected_cash_pnl = _checked_product(
            expected_cash_return,
            capital,
            name="cash P&L contribution",
        )
        expected_cost_pnl = _checked_product(cost_return, capital, name="cost P&L")
        cash_return_contribution = _finite_float(
            self.cash_return_contribution,
            name="cash_return_contribution",
        )
        cash_pnl_contribution = _finite_float(
            self.cash_pnl_contribution,
            name="cash_pnl_contribution",
        )
        cost_return_contribution = _finite_float(
            self.cost_return_contribution,
            name="cost_return_contribution",
        )
        cost_pnl_contribution = _finite_float(
            self.cost_pnl_contribution,
            name="cost_pnl_contribution",
        )
        _reconciliation_error(
            cash_return_contribution,
            expected_cash_return,
            name="cash return contribution",
            term_magnitudes=(cash_return_contribution, expected_cash_return),
            operations=1,
        )
        _reconciliation_error(
            cash_pnl_contribution,
            expected_cash_pnl,
            name="cash P&L contribution",
            term_magnitudes=(cash_pnl_contribution, expected_cash_pnl),
            operations=1,
        )
        _reconciliation_error(
            cost_return_contribution,
            -cost_return,
            name="cost return contribution",
            term_magnitudes=(cost_return_contribution, cost_return),
            operations=1,
        )
        _reconciliation_error(
            cost_pnl_contribution,
            -expected_cost_pnl,
            name="cost P&L contribution",
            term_magnitudes=(cost_pnl_contribution, expected_cost_pnl),
            operations=1,
        )
        gross_return = _checked_sum(
            (*[item.return_contribution for item in records], expected_cash_return),
            name="gross realized return",
        )
        gross_pnl = _checked_sum(
            (*[item.pnl_contribution for item in records], expected_cash_pnl),
            name="gross realized P&L",
        )
        observed_net_pnl = _checked_subtract(
            observed_ending,
            capital,
            name="observed net P&L",
        )
        observed_net_return = observed_net_pnl / capital
        return_error = _reconcile_sum(
            (
                *[item.return_contribution for item in records],
                expected_cash_return,
                -cost_return,
            ),
            observed_net_return,
            name="realized return contributions",
        )
        pnl_error = _reconcile_sum(
            (
                *[item.pnl_contribution for item in records],
                expected_cash_pnl,
                -expected_cost_pnl,
            ),
            observed_net_pnl,
            name="realized P&L contributions",
        )
        ending_error = _reconciliation_error(
            modeled_ending,
            observed_ending,
            name="modeled versus observed ending equity",
            term_magnitudes=(modeled_ending, observed_ending),
            operations=2 * len(records) + 4,
        )
        for field_name, expected in (
            ("gross_return", gross_return),
            ("net_return", observed_net_return),
            ("gross_pnl", gross_pnl),
            ("cost_pnl", expected_cost_pnl),
            ("net_pnl", observed_net_pnl),
        ):
            supplied = _finite_float(getattr(self, field_name), name=field_name)
            _reconciliation_error(
                supplied,
                expected,
                name=field_name,
                term_magnitudes=(supplied, expected),
                operations=1,
            )
            object.__setattr__(self, field_name, expected)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "initial_capital", capital)
        object.__setattr__(self, "starting_cash_weight", cash_weight)
        object.__setattr__(self, "cash_return", cash_return)
        object.__setattr__(self, "cost_return", cost_return)
        object.__setattr__(self, "observed_ending_equity", observed_ending)
        object.__setattr__(self, "modeled_ending_equity", modeled_ending)
        object.__setattr__(self, "cash_return_contribution", expected_cash_return)
        object.__setattr__(self, "cash_pnl_contribution", expected_cash_pnl)
        object.__setattr__(self, "cost_return_contribution", -cost_return)
        object.__setattr__(self, "cost_pnl_contribution", -expected_cost_pnl)
        object.__setattr__(
            self,
            "return_reconciliation_error",
            _canonical_error(
                self.return_reconciliation_error,
                return_error,
                name="return_reconciliation_error",
                scale=observed_net_return,
            ),
        )
        object.__setattr__(
            self,
            "pnl_reconciliation_error",
            _canonical_error(
                self.pnl_reconciliation_error,
                pnl_error,
                name="pnl_reconciliation_error",
                scale=observed_net_pnl,
            ),
        )
        object.__setattr__(
            self,
            "ending_equity_reconciliation_error",
            _canonical_error(
                self.ending_equity_reconciliation_error,
                ending_error,
                name="ending_equity_reconciliation_error",
                scale=modeled_ending,
            ),
        )
        object.__setattr__(self, "asset_contributions", records)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contribution_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = [asdict(item) for item in self.asset_contributions]
        rows.extend(
            (
                {
                    "asset": "__cash__",
                    "starting_weight": self.starting_cash_weight,
                    "asset_return": self.cash_return,
                    "return_contribution": self.cash_return_contribution,
                    "pnl_contribution": self.cash_pnl_contribution,
                },
                {
                    "asset": "__cost__",
                    "starting_weight": 0.0,
                    "asset_return": 0.0,
                    "return_contribution": self.cost_return_contribution,
                    "pnl_contribution": self.cost_pnl_contribution,
                },
            )
        )
        return pd.DataFrame(rows)


def attribute_realized_performance(
    weights: pd.Series,
    realized_returns: pd.Series,
    *,
    initial_capital: float,
    observed_ending_equity: float,
    cash_weight: float,
    cash_return: float,
    cost_return: float,
) -> RealizedAttribution:
    """Reconcile supplied ending equity to asset, cash, and cost contributions."""
    assets, vector = _weight_vector(weights)
    returns = _return_vector(realized_returns, assets=assets, name="realized_returns")
    capital = _positive_float(initial_capital, name="initial_capital")
    observed_ending = _positive_float(
        observed_ending_equity,
        name="observed_ending_equity",
    )
    cash = _finite_float(cash_weight, name="cash_weight")
    cash_period_return = _cash_return(cash_return, name="cash_return")
    cost = _nonnegative_float(cost_return, name="cost_return")
    _reconcile_sum((*vector, cash), 1.0, name="starting asset weights plus cash")
    try:
        with np.errstate(over="raise", invalid="raise"):
            return_contributions = vector * returns
            pnl_contributions = return_contributions * capital
            asset_ending_values = capital * vector * (1.0 + returns)
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError("realized attribution arithmetic overflowed") from exc
    if not all(
        np.isfinite(values).all()
        for values in (return_contributions, pnl_contributions, asset_ending_values)
    ):
        raise RiskAttributionError("realized attribution produced non-finite values")
    cash_contribution = _checked_product(cash, cash_period_return, name="cash return")
    cash_pnl = _checked_product(cash_contribution, capital, name="cash P&L")
    cost_pnl = _checked_product(cost, capital, name="cost P&L")
    gross_return = _checked_sum(
        (*return_contributions, cash_contribution),
        name="gross realized return",
    )
    gross_pnl = _checked_sum((*pnl_contributions, cash_pnl), name="gross realized P&L")
    net_return = _checked_subtract(gross_return, cost, name="net realized return")
    net_pnl = _checked_subtract(gross_pnl, cost_pnl, name="net realized P&L")
    cash_ending_value = _checked_product(
        _checked_product(capital, cash, name="starting cash value"),
        _checked_add(1.0, cash_period_return, name="cash growth multiplier"),
        name="ending cash value",
    )
    modeled_ending = _checked_sum(
        (*asset_ending_values, cash_ending_value, -cost_pnl),
        name="modeled ending equity",
    )
    observed_net_pnl = _checked_subtract(
        observed_ending,
        capital,
        name="observed net P&L",
    )
    observed_net_return = observed_net_pnl / capital
    return_error = _reconcile_sum(
        (*return_contributions, cash_contribution, -cost),
        observed_net_return,
        name="realized return contributions",
    )
    pnl_error = _reconcile_sum(
        (*pnl_contributions, cash_pnl, -cost_pnl),
        observed_net_pnl,
        name="realized P&L contributions",
    )
    ending_error = _reconciliation_error(
        modeled_ending,
        observed_ending,
        name="modeled versus observed ending equity",
        term_magnitudes=(*asset_ending_values, cash_ending_value, cost_pnl, observed_ending),
        operations=2 * len(assets) + 4,
    )
    contributions = tuple(
        RealizedAssetContribution(
            asset=asset,
            starting_weight=float(vector[index]),
            asset_return=float(returns[index]),
            return_contribution=float(return_contributions[index]),
            pnl_contribution=float(pnl_contributions[index]),
        )
        for index, asset in enumerate(assets)
    )
    return RealizedAttribution(
        assets=assets,
        initial_capital=capital,
        starting_cash_weight=cash,
        cash_return=cash_period_return,
        cost_return=cost,
        observed_ending_equity=observed_ending,
        modeled_ending_equity=modeled_ending,
        gross_return=gross_return,
        net_return=net_return,
        gross_pnl=gross_pnl,
        cost_pnl=cost_pnl,
        net_pnl=net_pnl,
        cash_return_contribution=cash_contribution,
        cash_pnl_contribution=cash_pnl,
        cost_return_contribution=-cost,
        cost_pnl_contribution=-cost_pnl,
        return_reconciliation_error=return_error,
        pnl_reconciliation_error=pnl_error,
        ending_equity_reconciliation_error=ending_error,
        asset_contributions=contributions,
    )


@dataclass(frozen=True, slots=True)
class AssetWeightDrift:
    """One asset's weight change caused only by supplied period returns."""

    asset: str
    pre_return_weight: float
    post_return_weight: float
    drift: float

    def __post_init__(self) -> None:
        asset = _label(self.asset, name="asset")
        pre = _finite_float(self.pre_return_weight, name="pre_return_weight")
        post = _finite_float(self.post_return_weight, name="post_return_weight")
        drift = _finite_float(self.drift, name="drift")
        expected = _checked_subtract(post, pre, name="asset weight drift")
        _reconciliation_error(
            drift,
            expected,
            name=f"asset {asset!r} weight drift",
            term_magnitudes=(pre, post, drift),
            operations=1,
        )
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "pre_return_weight", pre)
        object.__setattr__(self, "post_return_weight", post)
        object.__setattr__(self, "drift", expected)


@dataclass(frozen=True, slots=True)
class FactorExposureDrift:
    """One factor's pre/post-return exposure and drift."""

    factor: str
    pre_return_exposure: float
    post_return_exposure: float
    drift: float

    def __post_init__(self) -> None:
        factor = _label(self.factor, name="factor")
        pre = _finite_float(self.pre_return_exposure, name="pre_return_exposure")
        post = _finite_float(self.post_return_exposure, name="post_return_exposure")
        drift = _finite_float(self.drift, name="drift")
        expected = _checked_subtract(post, pre, name="factor exposure drift")
        _reconciliation_error(
            drift,
            expected,
            name=f"factor {factor!r} exposure drift",
            term_magnitudes=(pre, post, drift),
            operations=1,
        )
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "pre_return_exposure", pre)
        object.__setattr__(self, "post_return_exposure", post)
        object.__setattr__(self, "drift", expected)


@dataclass(frozen=True, slots=True)
class ExposureDriftAttribution:
    """Validated pre/post-return asset, cash, gross/net, and factor drift."""

    assets: tuple[str, ...]
    factors: tuple[str, ...]
    risk_model_identity: str
    ending_equity_multiplier: float
    pre_return_cash_weight: float
    post_return_cash_weight: float
    pre_return_gross: float
    post_return_gross: float
    gross_drift: float
    pre_return_net: float
    post_return_net: float
    net_drift: float
    accounting_reconciliation_error: float
    asset_drifts: tuple[AssetWeightDrift, ...]
    factor_drifts: tuple[FactorExposureDrift, ...]

    def __post_init__(self) -> None:
        assets = _labels(self.assets, name="assets", maximum=MAX_ATTRIBUTION_ASSETS)
        factors = _labels(self.factors, name="factors", maximum=64)
        identity = _identity(self.risk_model_identity, name="risk_model_identity")
        ending = _positive_float(
            self.ending_equity_multiplier,
            name="ending_equity_multiplier",
        )
        pre_cash = _finite_float(self.pre_return_cash_weight, name="pre_return_cash_weight")
        post_cash = _finite_float(
            self.post_return_cash_weight,
            name="post_return_cash_weight",
        )
        asset_records = _typed_tuple(
            self.asset_drifts,
            AssetWeightDrift,
            name="asset_drifts",
            maximum=MAX_ATTRIBUTION_ASSETS,
            allow_empty=False,
        )
        factor_records = _typed_tuple(
            self.factor_drifts,
            FactorExposureDrift,
            name="factor_drifts",
            maximum=64,
            allow_empty=False,
        )
        if tuple(item.asset for item in asset_records) != assets:
            raise RiskAttributionError("asset drifts must exactly match asset order")
        if tuple(item.factor for item in factor_records) != factors:
            raise RiskAttributionError("factor drifts must exactly match factor order")
        pre_weights = tuple(item.pre_return_weight for item in asset_records)
        post_weights = tuple(item.post_return_weight for item in asset_records)
        _reconcile_sum((*pre_weights, pre_cash), 1.0, name="pre-return weights plus cash")
        accounting_error = _reconcile_sum(
            (*post_weights, post_cash),
            1.0,
            name="post-return weights plus cash",
        )
        expected = {
            "pre_return_gross": _checked_sum(map(abs, pre_weights), name="pre-return gross"),
            "post_return_gross": _checked_sum(map(abs, post_weights), name="post-return gross"),
            "pre_return_net": _checked_sum(pre_weights, name="pre-return net"),
            "post_return_net": _checked_sum(post_weights, name="post-return net"),
        }
        for field_name, value in expected.items():
            supplied = _finite_float(getattr(self, field_name), name=field_name)
            _reconciliation_error(
                supplied,
                value,
                name=field_name,
                term_magnitudes=(supplied, value),
                operations=max(len(asset_records) - 1, 1),
            )
            object.__setattr__(self, field_name, value)
        expected_gross_drift = _checked_subtract(
            expected["post_return_gross"],
            expected["pre_return_gross"],
            name="gross drift",
        )
        expected_net_drift = _checked_subtract(
            expected["post_return_net"],
            expected["pre_return_net"],
            name="net drift",
        )
        for field_name, value in (
            ("gross_drift", expected_gross_drift),
            ("net_drift", expected_net_drift),
        ):
            supplied = _finite_float(getattr(self, field_name), name=field_name)
            _reconciliation_error(
                supplied,
                value,
                name=field_name,
                term_magnitudes=(supplied, value),
                operations=1,
            )
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "risk_model_identity", identity)
        object.__setattr__(self, "ending_equity_multiplier", ending)
        object.__setattr__(self, "pre_return_cash_weight", pre_cash)
        object.__setattr__(self, "post_return_cash_weight", post_cash)
        object.__setattr__(
            self,
            "accounting_reconciliation_error",
            _canonical_error(
                self.accounting_reconciliation_error,
                accounting_error,
                name="accounting_reconciliation_error",
            ),
        )
        object.__setattr__(self, "asset_drifts", asset_records)
        object.__setattr__(self, "factor_drifts", factor_records)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def asset_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(item) for item in self.asset_drifts)

    def factor_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(item) for item in self.factor_drifts)


def attribute_exposure_drift(
    factor_model: FactorRiskModel,
    weights: pd.Series,
    realized_returns: pd.Series,
    *,
    cash_weight: float,
    cash_return: float = 0.0,
) -> ExposureDriftAttribution:
    """Attribute exposure drift caused by returns without assuming a trade."""
    if not isinstance(factor_model, FactorRiskModel):
        raise RiskAttributionError("factor_model must be a validated FactorRiskModel")
    assets, vector = _weight_vector(
        weights,
        expected_assets=factor_model.risk_model.assets,
    )
    returns = _return_vector(realized_returns, assets=assets, name="realized_returns")
    cash = _finite_float(cash_weight, name="cash_weight")
    cash_period_return = _cash_return(cash_return, name="cash_return")
    _reconcile_sum((*vector, cash), 1.0, name="asset weights plus explicit cash_weight")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ending_asset_values = vector * (1.0 + returns)
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError("exposure drift arithmetic overflowed") from exc
    ending_cash_value = _checked_product(
        cash,
        _checked_add(1.0, cash_period_return, name="cash growth multiplier"),
        name="ending cash value",
    )
    ending_equity = _checked_sum(
        (*ending_asset_values, ending_cash_value),
        name="post-return equity",
    )
    if ending_equity <= 0.0:
        raise RiskAttributionError("post-return equity must remain positive")
    post_weights = ending_asset_values / ending_equity
    post_cash = ending_cash_value / ending_equity
    accounting_error = _reconcile_sum(
        (*post_weights, post_cash),
        1.0,
        name="post-return weights plus cash",
    )
    try:
        pre_factor = factor_model.factor_exposures(vector)
        post_factor = factor_model.factor_exposures(post_weights)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RiskAttributionError("factor exposure drift failed") from exc
    asset_drifts = tuple(
        AssetWeightDrift(
            asset=asset,
            pre_return_weight=float(vector[index]),
            post_return_weight=float(post_weights[index]),
            drift=float(post_weights[index] - vector[index]),
        )
        for index, asset in enumerate(assets)
    )
    factor_drifts = tuple(
        FactorExposureDrift(
            factor=factor,
            pre_return_exposure=float(pre_factor[index]),
            post_return_exposure=float(post_factor[index]),
            drift=float(post_factor[index] - pre_factor[index]),
        )
        for index, factor in enumerate(factor_model.factors)
    )
    pre_gross = _checked_sum(np.abs(vector), name="pre-return gross")
    post_gross = _checked_sum(np.abs(post_weights), name="post-return gross")
    pre_net = _checked_sum(vector, name="pre-return net")
    post_net = _checked_sum(post_weights, name="post-return net")
    return ExposureDriftAttribution(
        assets=assets,
        factors=factor_model.factors,
        risk_model_identity=factor_model.identity,
        ending_equity_multiplier=ending_equity,
        pre_return_cash_weight=cash,
        post_return_cash_weight=post_cash,
        pre_return_gross=pre_gross,
        post_return_gross=post_gross,
        gross_drift=_checked_subtract(post_gross, pre_gross, name="gross drift"),
        pre_return_net=pre_net,
        post_return_net=post_net,
        net_drift=_checked_subtract(post_net, pre_net, name="net drift"),
        accounting_reconciliation_error=accounting_error,
        asset_drifts=asset_drifts,
        factor_drifts=factor_drifts,
    )


@dataclass(frozen=True, slots=True)
class ScenarioAssetContribution:
    """One asset's contribution under one named simple-return scenario."""

    asset: str
    starting_weight: float
    scenario_return: float
    return_contribution: float
    pnl_contribution: float

    def __post_init__(self) -> None:
        asset = _label(self.asset, name="asset")
        weight = _finite_float(self.starting_weight, name="starting_weight")
        scenario_return = _cash_return(self.scenario_return, name="scenario_return")
        return_contribution = _finite_float(
            self.return_contribution,
            name="return_contribution",
        )
        pnl_contribution = _finite_float(self.pnl_contribution, name="pnl_contribution")
        _reconcile_product(
            weight,
            scenario_return,
            return_contribution,
            name=f"scenario asset {asset!r} return contribution",
        )
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "starting_weight", weight)
        object.__setattr__(self, "scenario_return", scenario_return)
        object.__setattr__(self, "return_contribution", return_contribution)
        object.__setattr__(self, "pnl_contribution", pnl_contribution)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """One modeled scenario with explicit cash, cost, and ending-value checks."""

    name: str
    initial_capital: float
    starting_cash_weight: float
    cash_return: float
    cost_return: float
    cash_return_contribution: float
    cash_pnl_contribution: float
    cost_return_contribution: float
    cost_pnl_contribution: float
    modeled_portfolio_return: float
    modeled_scenario_pnl: float
    modeled_ending_equity: float
    return_reconciliation_error: float
    pnl_reconciliation_error: float
    ending_value_reconciliation_error: float
    contributions: tuple[ScenarioAssetContribution, ...]

    def __post_init__(self) -> None:
        name = _label(self.name, name="scenario name")
        capital = _positive_float(self.initial_capital, name="initial_capital")
        cash_weight = _finite_float(self.starting_cash_weight, name="starting_cash_weight")
        cash_return = _cash_return(self.cash_return, name="cash_return")
        cost_return = _nonnegative_float(self.cost_return, name="cost_return")
        records = _typed_tuple(
            self.contributions,
            ScenarioAssetContribution,
            name="scenario contributions",
            maximum=MAX_ATTRIBUTION_ASSETS,
            allow_empty=False,
        )
        if len({item.asset for item in records}) != len(records):
            raise RiskAttributionError("scenario contribution asset labels must be unique")
        _reconcile_sum(
            (*[item.starting_weight for item in records], cash_weight),
            1.0,
            name=f"scenario {name!r} starting weights plus cash",
        )
        for item in records:
            _reconcile_product(
                item.return_contribution,
                capital,
                item.pnl_contribution,
                name=f"scenario {name!r} asset {item.asset!r} P&L",
            )
        cash_return_contribution = _checked_product(
            cash_weight,
            cash_return,
            name=f"scenario {name!r} cash return",
        )
        cash_pnl_contribution = _checked_product(
            cash_return_contribution,
            capital,
            name=f"scenario {name!r} cash P&L",
        )
        cost_pnl = _checked_product(cost_return, capital, name=f"scenario {name!r} cost P&L")
        expected_fields = {
            "cash_return_contribution": cash_return_contribution,
            "cash_pnl_contribution": cash_pnl_contribution,
            "cost_return_contribution": -cost_return,
            "cost_pnl_contribution": -cost_pnl,
        }
        for field_name, expected in expected_fields.items():
            supplied = _finite_float(getattr(self, field_name), name=field_name)
            _reconciliation_error(
                supplied,
                expected,
                name=f"scenario {name!r} {field_name}",
                term_magnitudes=(supplied, expected),
                operations=1,
            )
            object.__setattr__(self, field_name, expected)
        modeled_return = _checked_sum(
            (
                *[item.return_contribution for item in records],
                cash_return_contribution,
                -cost_return,
            ),
            name=f"scenario {name!r} modeled return",
        )
        modeled_pnl = _checked_sum(
            (
                *[item.pnl_contribution for item in records],
                cash_pnl_contribution,
                -cost_pnl,
            ),
            name=f"scenario {name!r} modeled P&L",
        )
        modeled_ending = _finite_float(
            self.modeled_ending_equity,
            name="modeled_ending_equity",
        )
        independently_computed_ending = _checked_sum(
            (
                *(
                    _checked_product(
                        _checked_product(
                            capital,
                            item.starting_weight,
                            name=f"scenario {name!r} starting asset value",
                        ),
                        _checked_add(
                            1.0,
                            item.scenario_return,
                            name=f"scenario {name!r} asset growth",
                        ),
                        name=f"scenario {name!r} ending asset value",
                    )
                    for item in records
                ),
                _checked_product(
                    _checked_product(
                        capital,
                        cash_weight,
                        name=f"scenario {name!r} starting cash value",
                    ),
                    _checked_add(1.0, cash_return, name=f"scenario {name!r} cash growth"),
                    name=f"scenario {name!r} ending cash value",
                ),
                -cost_pnl,
            ),
            name=f"scenario {name!r} ending-value path",
        )
        return_error = _reconciliation_error(
            _finite_float(self.modeled_portfolio_return, name="modeled_portfolio_return"),
            modeled_return,
            name=f"scenario {name!r} modeled return",
            term_magnitudes=(self.modeled_portfolio_return, modeled_return),
            operations=len(records) + 1,
        )
        pnl_error = _reconciliation_error(
            _finite_float(self.modeled_scenario_pnl, name="modeled_scenario_pnl"),
            modeled_pnl,
            name=f"scenario {name!r} modeled scenario P&L",
            term_magnitudes=(self.modeled_scenario_pnl, modeled_pnl),
            operations=len(records) + 1,
        )
        ending_error = _reconciliation_error(
            modeled_ending,
            independently_computed_ending,
            name=f"scenario {name!r} ending-value path",
            term_magnitudes=(modeled_ending, independently_computed_ending),
            operations=2 * len(records) + 4,
        )
        _reconcile_sum((capital, modeled_pnl), modeled_ending, name=f"scenario {name!r} equity")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "initial_capital", capital)
        object.__setattr__(self, "starting_cash_weight", cash_weight)
        object.__setattr__(self, "cash_return", cash_return)
        object.__setattr__(self, "cost_return", cost_return)
        object.__setattr__(self, "modeled_portfolio_return", modeled_return)
        object.__setattr__(self, "modeled_scenario_pnl", modeled_pnl)
        object.__setattr__(self, "modeled_ending_equity", independently_computed_ending)
        object.__setattr__(
            self,
            "return_reconciliation_error",
            _canonical_error(
                self.return_reconciliation_error,
                return_error,
                name="return_reconciliation_error",
                scale=modeled_return,
            ),
        )
        object.__setattr__(
            self,
            "pnl_reconciliation_error",
            _canonical_error(
                self.pnl_reconciliation_error,
                pnl_error,
                name="pnl_reconciliation_error",
                scale=modeled_pnl,
            ),
        )
        object.__setattr__(
            self,
            "ending_value_reconciliation_error",
            _canonical_error(
                self.ending_value_reconciliation_error,
                ending_error,
                name="ending_value_reconciliation_error",
                scale=modeled_ending,
            ),
        )
        object.__setattr__(self, "contributions", records)


@dataclass(frozen=True, slots=True)
class ScenarioAnalysis:
    """Deterministically ordered modeled scenarios with explicit financing."""

    assets: tuple[str, ...]
    initial_capital: float
    starting_cash_weight: float
    cash_return: float
    cost_return: float
    results: tuple[ScenarioResult, ...]

    def __post_init__(self) -> None:
        assets = _labels(self.assets, name="assets", maximum=MAX_ATTRIBUTION_ASSETS)
        capital = _positive_float(self.initial_capital, name="initial_capital")
        cash_weight = _finite_float(self.starting_cash_weight, name="starting_cash_weight")
        cash_return = _cash_return(self.cash_return, name="cash_return")
        cost_return = _nonnegative_float(self.cost_return, name="cost_return")
        results = _typed_tuple(
            self.results,
            ScenarioResult,
            name="scenario results",
            maximum=MAX_SCENARIOS,
            allow_empty=False,
        )
        names = tuple(result.name for result in results)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise RiskAttributionError("scenario results must have unique sorted names")
        for result in results:
            if tuple(item.asset for item in result.contributions) != assets:
                raise RiskAttributionError("scenario contributions must exactly match asset order")
            for field_name, expected in (
                ("initial_capital", capital),
                ("starting_cash_weight", cash_weight),
                ("cash_return", cash_return),
                ("cost_return", cost_return),
            ):
                observed = _finite_float(getattr(result, field_name), name=field_name)
                _reconciliation_error(
                    observed,
                    expected,
                    name=f"scenario {result.name!r} {field_name}",
                    term_magnitudes=(observed, expected),
                    operations=1,
                )
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "initial_capital", capital)
        object.__setattr__(self, "starting_cash_weight", cash_weight)
        object.__setattr__(self, "cash_return", cash_return)
        object.__setattr__(self, "cost_return", cost_return)
        object.__setattr__(self, "results", results)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contribution_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            rows.extend(
                {"scenario": result.name, **asdict(contribution)}
                for contribution in result.contributions
            )
            rows.extend(
                (
                    {
                        "scenario": result.name,
                        "asset": "__cash__",
                        "starting_weight": result.starting_cash_weight,
                        "scenario_return": result.cash_return,
                        "return_contribution": result.cash_return_contribution,
                        "pnl_contribution": result.cash_pnl_contribution,
                    },
                    {
                        "scenario": result.name,
                        "asset": "__cost__",
                        "starting_weight": 0.0,
                        "scenario_return": 0.0,
                        "return_contribution": result.cost_return_contribution,
                        "pnl_contribution": result.cost_pnl_contribution,
                    },
                )
            )
        return pd.DataFrame(rows)


def evaluate_return_scenarios(
    weights: pd.Series,
    scenarios: Mapping[str, pd.Series],
    *,
    initial_capital: float,
    cash_weight: float,
    cash_return: float,
    cost_return: float,
) -> ScenarioAnalysis:
    """Evaluate bounded modeled scenarios with explicit cash and cost paths."""
    assets, vector = _weight_vector(weights)
    if not isinstance(scenarios, Mapping):
        raise RiskAttributionError("scenarios must be a mapping of name to return Series")
    if not 1 <= len(scenarios) <= MAX_SCENARIOS:
        raise RiskAttributionError(f"scenario count must lie in [1, {MAX_SCENARIOS}]")
    capital = _positive_float(initial_capital, name="initial_capital")
    cash = _finite_float(cash_weight, name="cash_weight")
    cash_period_return = _cash_return(cash_return, name="cash_return")
    cost = _nonnegative_float(cost_return, name="cost_return")
    _reconcile_sum((*vector, cash), 1.0, name="scenario starting weights plus cash")
    names = tuple(_label(name, name="scenario name") for name in scenarios)
    if len(set(names)) != len(names):
        raise RiskAttributionError("scenario names must be unique")
    results: list[ScenarioResult] = []
    for name in sorted(names):
        scenario_returns = _return_vector(
            scenarios[name],
            assets=assets,
            name=f"scenario {name!r}",
            maximum=MAX_SCENARIO_SIMPLE_RETURN,
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                return_contributions = vector * scenario_returns
                pnl_contributions = return_contributions * capital
                asset_ending_values = capital * vector * (1.0 + scenario_returns)
        except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
            raise RiskAttributionError(f"scenario {name!r} arithmetic overflowed") from exc
        if not all(
            np.isfinite(values).all()
            for values in (return_contributions, pnl_contributions, asset_ending_values)
        ):
            raise RiskAttributionError(f"scenario {name!r} produced non-finite values")
        cash_contribution = _checked_product(
            cash,
            cash_period_return,
            name=f"scenario {name!r} cash return",
        )
        cash_pnl = _checked_product(
            cash_contribution,
            capital,
            name=f"scenario {name!r} cash P&L",
        )
        cost_pnl = _checked_product(cost, capital, name=f"scenario {name!r} cost P&L")
        modeled_return = _checked_sum(
            (*return_contributions, cash_contribution, -cost),
            name=f"scenario {name!r} modeled return",
        )
        modeled_pnl = _checked_sum(
            (*pnl_contributions, cash_pnl, -cost_pnl),
            name=f"scenario {name!r} modeled P&L",
        )
        cash_ending_value = _checked_product(
            _checked_product(capital, cash, name=f"scenario {name!r} starting cash"),
            _checked_add(1.0, cash_period_return, name=f"scenario {name!r} cash growth"),
            name=f"scenario {name!r} ending cash",
        )
        ending_equity = _checked_sum(
            (*asset_ending_values, cash_ending_value, -cost_pnl),
            name=f"scenario {name!r} ending-value path",
        )
        return_error = _reconcile_sum(
            (*return_contributions, cash_contribution, -cost),
            modeled_return,
            name=f"scenario {name!r} return contributions",
        )
        pnl_error = _reconcile_sum(
            (*pnl_contributions, cash_pnl, -cost_pnl),
            modeled_pnl,
            name=f"scenario {name!r} P&L contributions",
        )
        ending_error = _reconcile_sum(
            (capital, modeled_pnl),
            ending_equity,
            name=f"scenario {name!r} modeled ending equity",
        )
        contributions = tuple(
            ScenarioAssetContribution(
                asset=asset,
                starting_weight=float(vector[index]),
                scenario_return=float(scenario_returns[index]),
                return_contribution=float(return_contributions[index]),
                pnl_contribution=float(pnl_contributions[index]),
            )
            for index, asset in enumerate(assets)
        )
        results.append(
            ScenarioResult(
                name=name,
                initial_capital=capital,
                starting_cash_weight=cash,
                cash_return=cash_period_return,
                cost_return=cost,
                cash_return_contribution=cash_contribution,
                cash_pnl_contribution=cash_pnl,
                cost_return_contribution=-cost,
                cost_pnl_contribution=-cost_pnl,
                modeled_portfolio_return=modeled_return,
                modeled_scenario_pnl=modeled_pnl,
                modeled_ending_equity=ending_equity,
                return_reconciliation_error=return_error,
                pnl_reconciliation_error=pnl_error,
                ending_value_reconciliation_error=ending_error,
                contributions=contributions,
            )
        )
    return ScenarioAnalysis(
        assets=assets,
        initial_capital=capital,
        starting_cash_weight=cash,
        cash_return=cash_period_return,
        cost_return=cost,
        results=tuple(results),
    )
