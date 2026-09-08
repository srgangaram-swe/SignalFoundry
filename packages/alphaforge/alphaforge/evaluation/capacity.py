"""Causal, auditable capacity scenarios for an observed trade/fill panel.

This module deliberately does not estimate ADV from the trade date or from
future observations.  Callers provide liquidity and cost information that was
available before each trade.  The resulting curve is a sensitivity analysis,
not a point estimate of deployable AUM.

For each scenario AUM, observed desired and traded notionals are scaled from a
reference AUM.  Scaled fills are capped at a configured fraction of lagged ADV.
Costs use a transparent power-law sensitivity anchored to the supplied cost at
the reference fill.  Both aggregate and row-level results are returned so every
curve point can be reconciled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

LiquiditySource = Literal["lagged_adv_notional", "lagged_adv_shares_x_reference_price"]
CostSource = Literal["lagged_cost_bps", "realized_total_cost"]
TemporalValidation = Literal["information_date_verified", "caller_attested_lagged_inputs"]

_NOTIONAL_RTOL = 1e-9
_NOTIONAL_ATOL = 1e-8


@dataclass(frozen=True)
class CapacityColumns:
    """Column mapping for a trade/fill panel.

    ``lagged_adv_notional`` is preferred when present.  Otherwise liquidity is
    derived from ``lagged_adv_shares * reference_price``.  Likewise,
    ``lagged_cost_bps`` is preferred over ``total_cost``.  Set a mapping to
    ``None`` to disable that source explicitly.

    Notionals may be signed, but desired and traded values must have the same
    direction.  Capacity calculations use their gross magnitudes.
    """

    date: str = "date"
    symbol: str = "symbol"
    desired_notional: str = "desired_notional"
    traded_notional: str = "traded_notional"
    lagged_adv_notional: str | None = "lagged_adv_notional"
    lagged_adv_shares: str | None = "lagged_adv_shares"
    reference_price: str | None = "reference_price"
    lagged_cost_bps: str | None = "lagged_cost_bps"
    total_cost: str | None = "total_cost"
    information_date: str | None = None

    @classmethod
    def for_fill_records(cls) -> CapacityColumns:
        """Return the mapping used by ``BacktestResult.fills`` records.

        Realized ``total_cost`` is selected explicitly, so diagnostics label
        the result as an ex-post anchored sensitivity.  The backtest decision
        date is intentionally not presented as the liquidity/cost information
        timestamp; callers can set ``information_date`` when that provenance is
        available separately.
        """

        return cls(
            date="fill_date",
            desired_notional="requested_notional",
            lagged_cost_bps=None,
        )

    def __post_init__(self) -> None:
        required = {
            "date": self.date,
            "symbol": self.symbol,
            "desired_notional": self.desired_notional,
            "traded_notional": self.traded_notional,
        }
        for field_name, column_name in required.items():
            if not isinstance(column_name, str) or not column_name.strip():
                raise ValueError(f"{field_name} column name must be a non-empty string")

        optional = {
            "lagged_adv_notional": self.lagged_adv_notional,
            "lagged_adv_shares": self.lagged_adv_shares,
            "reference_price": self.reference_price,
            "lagged_cost_bps": self.lagged_cost_bps,
            "total_cost": self.total_cost,
            "information_date": self.information_date,
        }
        for optional_field_name, optional_column_name in optional.items():
            if optional_column_name is not None and (
                not isinstance(optional_column_name, str) or not optional_column_name.strip()
            ):
                raise ValueError(
                    f"{optional_field_name} column name must be None or a non-empty string"
                )

        if len(set(required.values())) != len(required):
            raise ValueError("date, symbol, desired_notional, and traded_notional must be distinct")


@dataclass(frozen=True)
class CapacityConfig:
    """Assumptions for deterministic capacity sensitivity scenarios.

    ``reference_aum`` is the AUM represented by the input notionals.
    ``aum_values`` are normalized to unique ascending values for deterministic
    output.  ``variable_cost_fraction`` is the fraction of supplied cost bps
    assumed to scale with participation; the remainder is held constant.  The
    default exponent of 0.5 is a square-root sensitivity, not a fitted impact
    coefficient.
    """

    reference_aum: float
    aum_values: tuple[float, ...]
    max_participation_rate: float = 0.10
    impact_exponent: float = 0.50
    variable_cost_fraction: float = 0.50
    columns: CapacityColumns = field(default_factory=CapacityColumns)

    def __post_init__(self) -> None:
        reference_aum = _positive_finite_scalar(self.reference_aum, "reference_aum")
        object.__setattr__(self, "reference_aum", reference_aum)

        try:
            normalized_aum = tuple(
                sorted({_positive_finite_scalar(value, "aum_values") for value in self.aum_values})
            )
        except TypeError as exc:
            raise TypeError("aum_values must be an iterable of positive finite numbers") from exc
        if not normalized_aum:
            raise ValueError("aum_values must contain at least one scenario")
        object.__setattr__(self, "aum_values", normalized_aum)

        max_participation = _positive_finite_scalar(
            self.max_participation_rate, "max_participation_rate"
        )
        if max_participation > 1.0:
            raise ValueError("max_participation_rate must be <= 1")
        object.__setattr__(self, "max_participation_rate", max_participation)

        exponent = _positive_finite_scalar(self.impact_exponent, "impact_exponent")
        object.__setattr__(self, "impact_exponent", exponent)

        variable_fraction = _finite_scalar(self.variable_cost_fraction, "variable_cost_fraction")
        if not 0.0 <= variable_fraction <= 1.0:
            raise ValueError("variable_cost_fraction must be between 0 and 1")
        object.__setattr__(self, "variable_cost_fraction", variable_fraction)

        if not isinstance(self.columns, CapacityColumns):
            raise TypeError("columns must be a CapacityColumns instance")


@dataclass(frozen=True)
class CapacityDiagnostics:
    """Input provenance and interpretation guardrails for a capacity result."""

    n_input_rows: int
    n_active_trade_rows: int
    n_dates: int
    liquidity_source: LiquiditySource
    cost_source: CostSource
    temporal_validation: TemporalValidation
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class CapacityResult:
    """Aggregate capacity curve plus the row-level scenario audit trail."""

    curve: pd.DataFrame
    scenario_trades: pd.DataFrame
    diagnostics: CapacityDiagnostics
    config: CapacityConfig


@dataclass(frozen=True)
class _PreparedPanel:
    frame: pd.DataFrame
    liquidity_source: LiquiditySource
    cost_source: CostSource
    temporal_validation: TemporalValidation


def estimate_capacity(panel: pd.DataFrame, config: CapacityConfig) -> CapacityResult:
    """Build a deterministic AUM/participation/cost sensitivity curve.

    The input represents fills at ``config.reference_aum``.  Each row must
    contain a desired notional, a traded notional, and a causal liquidity
    estimate.  Cost can be supplied as lagged bps or as realized total dollars.
    In the latter case the analysis is ex-post anchored and diagnostics say so
    explicitly.

    No raw volume column is accepted and no rolling liquidity statistic is
    computed here.  When ``CapacityColumns.information_date`` is configured,
    every information timestamp must be strictly before its trade timestamp.
    """

    if not isinstance(config, CapacityConfig):
        raise TypeError("config must be a CapacityConfig instance")
    prepared = _prepare_panel(panel, config.columns)
    base = prepared.frame

    details = [_scenario_details(base, aum, config) for aum in config.aum_values]
    scenario_trades = pd.concat(details, ignore_index=True)
    curve = _aggregate_curve(scenario_trades)

    assumptions = _assumptions(prepared, config)
    diagnostics = CapacityDiagnostics(
        n_input_rows=len(base),
        n_active_trade_rows=int((base["base_traded_gross"] > 0.0).sum()),
        n_dates=int(base["date"].nunique()),
        liquidity_source=prepared.liquidity_source,
        cost_source=prepared.cost_source,
        temporal_validation=prepared.temporal_validation,
        assumptions=assumptions,
    )
    return CapacityResult(
        curve=curve,
        scenario_trades=scenario_trades,
        diagnostics=diagnostics,
        config=config,
    )


def _positive_finite_scalar(value: float, name: str) -> float:
    number = _finite_scalar(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return number


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_columns(panel: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(panel.columns))
    if missing:
        raise ValueError(f"trade panel is missing required columns: {missing}")


def _numeric_column(panel: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_rows = np.flatnonzero(~np.isfinite(values)).tolist()
        raise ValueError(f"{column!r} must contain only finite numeric values; bad rows={bad_rows}")
    return values


def _resolve_liquidity(
    panel: pd.DataFrame, columns: CapacityColumns
) -> tuple[np.ndarray, LiquiditySource]:
    direct = columns.lagged_adv_notional
    if direct is not None and direct in panel.columns:
        adv = _numeric_column(panel, direct)
        source: LiquiditySource = "lagged_adv_notional"
    else:
        shares = columns.lagged_adv_shares
        price = columns.reference_price
        if (
            shares is None
            or price is None
            or shares not in panel.columns
            or price not in panel.columns
        ):
            raise ValueError(
                "trade panel needs lagged ADV notional, or both lagged ADV shares and "
                "reference price, using the configured column mapping"
            )
        adv_shares = _numeric_column(panel, shares)
        reference_price = _numeric_column(panel, price)
        if (adv_shares <= 0.0).any():
            raise ValueError(f"{shares!r} must be strictly positive")
        if (reference_price <= 0.0).any():
            raise ValueError(f"{price!r} must be strictly positive")
        adv = adv_shares * reference_price
        if not np.isfinite(adv).all():
            raise ValueError("lagged ADV shares times reference price overflowed")
        source = "lagged_adv_shares_x_reference_price"

    if (adv <= 0.0).any():
        raise ValueError("lagged ADV notional must be strictly positive")
    return adv, source


def _resolve_cost(
    panel: pd.DataFrame,
    traded_gross: np.ndarray,
    columns: CapacityColumns,
) -> tuple[np.ndarray, CostSource]:
    cost_bps_column = columns.lagged_cost_bps
    if cost_bps_column is not None and cost_bps_column in panel.columns:
        cost_bps = _numeric_column(panel, cost_bps_column)
        if (cost_bps < 0.0).any():
            raise ValueError(f"{cost_bps_column!r} must be non-negative")
        return cost_bps, "lagged_cost_bps"

    total_cost_column = columns.total_cost
    if total_cost_column is None or total_cost_column not in panel.columns:
        raise ValueError(
            "trade panel needs lagged cost bps or realized total cost using the configured "
            "column mapping"
        )
    total_cost = _numeric_column(panel, total_cost_column)
    if (total_cost < 0.0).any():
        raise ValueError(f"{total_cost_column!r} must be non-negative")
    zero_fill_with_cost = (traded_gross == 0.0) & (total_cost > _NOTIONAL_ATOL)
    if zero_fill_with_cost.any():
        bad_rows = np.flatnonzero(zero_fill_with_cost).tolist()
        raise ValueError(f"non-zero total cost requires traded notional; bad rows={bad_rows}")

    cost_bps = np.divide(
        total_cost * 10_000.0,
        traded_gross,
        out=np.zeros_like(total_cost),
        where=traded_gross > 0.0,
    )
    if not np.isfinite(cost_bps).all():
        raise ValueError("total cost normalization produced non-finite cost bps")
    return cost_bps, "realized_total_cost"


def _prepare_panel(panel: pd.DataFrame, columns: CapacityColumns) -> _PreparedPanel:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("trade panel must contain at least one row")

    _require_columns(
        panel,
        [columns.date, columns.symbol, columns.desired_notional, columns.traded_notional],
    )

    dates = pd.to_datetime(panel[columns.date], errors="coerce", utc=True)
    if dates.isna().any():
        bad_rows = np.flatnonzero(dates.isna().to_numpy()).tolist()
        raise ValueError(f"{columns.date!r} contains invalid dates; bad rows={bad_rows}")
    normalized_dates = dates.dt.tz_convert(None)

    symbols_raw = panel[columns.symbol]
    invalid_symbol = symbols_raw.isna() | symbols_raw.astype(str).str.strip().eq("")
    if invalid_symbol.any():
        bad_rows = np.flatnonzero(invalid_symbol.to_numpy()).tolist()
        raise ValueError(f"{columns.symbol!r} contains missing/empty symbols; bad rows={bad_rows}")

    desired = _numeric_column(panel, columns.desired_notional)
    traded = _numeric_column(panel, columns.traded_notional)
    desired_gross = np.abs(desired)
    traded_gross = np.abs(traded)

    wrong_direction = (traded != 0.0) & (np.sign(desired) != np.sign(traded))
    if wrong_direction.any():
        bad_rows = np.flatnonzero(wrong_direction).tolist()
        raise ValueError(
            f"desired and traded notionals must have the same direction; bad rows={bad_rows}"
        )

    overfilled = (traded_gross > desired_gross) & ~np.isclose(
        traded_gross,
        desired_gross,
        rtol=_NOTIONAL_RTOL,
        atol=_NOTIONAL_ATOL,
    )
    if overfilled.any():
        bad_rows = np.flatnonzero(overfilled).tolist()
        raise ValueError(f"traded notional cannot exceed desired notional; bad rows={bad_rows}")
    # Eliminate tiny negative shortfalls admitted by floating-point tolerance.
    traded_gross = np.minimum(traded_gross, desired_gross)

    adv, liquidity_source = _resolve_liquidity(panel, columns)
    base_cost_bps, cost_source = _resolve_cost(panel, traded_gross, columns)

    temporal_validation: TemporalValidation = "caller_attested_lagged_inputs"
    if columns.information_date is not None:
        _require_columns(panel, [columns.information_date])
        information_dates = pd.to_datetime(
            panel[columns.information_date], errors="coerce", utc=True
        )
        if information_dates.isna().any():
            bad_rows = np.flatnonzero(information_dates.isna().to_numpy()).tolist()
            raise ValueError(
                f"{columns.information_date!r} contains invalid dates; bad rows={bad_rows}"
            )
        noncausal = information_dates >= dates
        if noncausal.any():
            bad_rows = np.flatnonzero(noncausal.to_numpy()).tolist()
            raise ValueError(
                "information_date must be strictly before the trade date for every row; "
                f"bad rows={bad_rows}"
            )
        temporal_validation = "information_date_verified"

    prepared = pd.DataFrame(
        {
            "date": normalized_dates,
            "symbol": symbols_raw.astype(str).str.strip(),
            "side": np.sign(desired).astype(np.int8),
            "base_desired_gross": desired_gross,
            "base_traded_gross": traded_gross,
            "lagged_adv_notional": adv,
            "base_cost_bps": base_cost_bps,
        }
    )
    # Canonical ordering makes curve aggregation invariant to input row order.
    sort_columns = [
        "date",
        "symbol",
        "side",
        "base_desired_gross",
        "base_traded_gross",
        "lagged_adv_notional",
        "base_cost_bps",
    ]
    prepared = prepared.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    prepared.insert(0, "trade_id", np.arange(len(prepared), dtype=np.int64))
    return _PreparedPanel(prepared, liquidity_source, cost_source, temporal_validation)


def _scenario_details(
    base: pd.DataFrame,
    scenario_aum: float,
    config: CapacityConfig,
) -> pd.DataFrame:
    multiple = scenario_aum / config.reference_aum
    desired = base["base_desired_gross"].to_numpy(dtype=float) * multiple
    empirical_fill = base["base_traded_gross"].to_numpy(dtype=float) * multiple
    adv = base["lagged_adv_notional"].to_numpy(dtype=float)
    participation_limit = adv * config.max_participation_rate
    modeled_fill = np.minimum(empirical_fill, participation_limit)

    if not all(np.isfinite(values).all() for values in (desired, empirical_fill, modeled_fill)):
        raise ValueError(f"scenario arithmetic overflowed for AUM={scenario_aum}")

    capacity_shortfall = empirical_fill - modeled_fill
    total_shortfall = desired - modeled_fill
    participation = modeled_fill / adv
    base_fill = base["base_traded_gross"].to_numpy(dtype=float)
    participation_ratio = np.divide(
        modeled_fill,
        base_fill,
        out=np.zeros_like(modeled_fill),
        where=base_fill > 0.0,
    )
    variable_fraction = config.variable_cost_fraction
    cost_multiplier = (1.0 - variable_fraction) + variable_fraction * np.power(
        participation_ratio, config.impact_exponent
    )
    modeled_cost_bps = base["base_cost_bps"].to_numpy(dtype=float) * cost_multiplier
    modeled_cost = modeled_fill * modeled_cost_bps / 10_000.0

    details = pd.DataFrame(
        {
            "scenario_aum": scenario_aum,
            "aum_multiple": multiple,
            "trade_id": base["trade_id"].to_numpy(),
            "date": base["date"].to_numpy(),
            "symbol": base["symbol"].to_numpy(),
            "side": base["side"].to_numpy(),
            "desired_gross_notional": desired,
            "empirical_fill_gross_notional": empirical_fill,
            "modeled_fill_gross_notional": modeled_fill,
            "total_shortfall_notional": total_shortfall,
            "capacity_shortfall_notional": capacity_shortfall,
            "lagged_adv_notional": adv,
            "participation_limit_notional": participation_limit,
            "participation_rate": participation,
            "is_capacity_constrained": capacity_shortfall > _NOTIONAL_ATOL,
            "modeled_cost_bps": modeled_cost_bps,
            "modeled_cost": modeled_cost,
        }
    )
    return details


def _aggregate_curve(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for scenario_aum, group in details.groupby("scenario_aum", sort=True):
        desired = float(group["desired_gross_notional"].sum())
        empirical_fill = float(group["empirical_fill_gross_notional"].sum())
        modeled_fill = float(group["modeled_fill_gross_notional"].sum())
        total_shortfall = float(group["total_shortfall_notional"].sum())
        capacity_shortfall = float(group["capacity_shortfall_notional"].sum())
        total_adv = float(group["lagged_adv_notional"].sum())
        modeled_cost = float(group["modeled_cost"].sum())
        active = group["empirical_fill_gross_notional"] > 0.0
        active_participation = group.loc[active, "participation_rate"]
        constrained = group["is_capacity_constrained"]
        n_active = int(active.sum())

        rows.append(
            {
                "scenario_aum": float(scenario_aum),
                "aum_multiple": float(group["aum_multiple"].iloc[0]),
                "n_dates": int(group["date"].nunique()),
                "n_trade_rows": n_active,
                "n_capacity_constrained": int(constrained.sum()),
                "capacity_constrained_fraction": (
                    float(constrained[active].mean()) if n_active else 0.0
                ),
                "desired_gross_notional": desired,
                "empirical_fill_gross_notional": empirical_fill,
                "modeled_fill_gross_notional": modeled_fill,
                "total_shortfall_notional": total_shortfall,
                "capacity_shortfall_notional": capacity_shortfall,
                "empirical_fill_ratio": empirical_fill / desired if desired > 0.0 else 0.0,
                "fill_ratio": modeled_fill / desired if desired > 0.0 else 0.0,
                "aggregate_participation_rate": modeled_fill / total_adv,
                "participation_p50": (
                    float(active_participation.quantile(0.50)) if n_active else 0.0
                ),
                "participation_p95": (
                    float(active_participation.quantile(0.95)) if n_active else 0.0
                ),
                "participation_max": float(active_participation.max()) if n_active else 0.0,
                "modeled_cost": modeled_cost,
                "modeled_cost_bps_per_traded_notional": (
                    modeled_cost / modeled_fill * 10_000.0 if modeled_fill > 0.0 else 0.0
                ),
                "modeled_cost_bps_per_desired_notional": (
                    modeled_cost / desired * 10_000.0 if desired > 0.0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_aum").reset_index(drop=True)


def _assumptions(prepared: _PreparedPanel, config: CapacityConfig) -> tuple[str, ...]:
    statements = [
        "Scenario outputs are sensitivities, not deployable-AUM forecasts or guarantees.",
        (
            "Fills preserve observed reference-AUM fill ratios, then cap each row at "
            f"{config.max_participation_rate:.2%} of caller-supplied lagged ADV."
        ),
        (
            f"{config.variable_cost_fraction:.0%} of supplied cost bps follows a "
            f"participation^{config.impact_exponent:g} sensitivity; the remainder is fixed."
        ),
        "No contemporaneous or future volume is read or inferred by this analysis.",
    ]
    if prepared.temporal_validation == "caller_attested_lagged_inputs":
        statements.append(
            "Lagged-input timing was not independently verified because no information-date "
            "column was configured."
        )
    else:
        statements.append("Every configured information date was verified before its trade date.")
    if prepared.cost_source == "realized_total_cost":
        statements.append(
            "Costs are ex-post anchored to realized total_cost and are not an ex-ante forecast."
        )
    else:
        statements.append("Costs are anchored to caller-supplied lagged cost estimates.")
    return tuple(statements)


__all__ = [
    "CapacityColumns",
    "CapacityConfig",
    "CapacityDiagnostics",
    "CapacityResult",
    "estimate_capacity",
]
