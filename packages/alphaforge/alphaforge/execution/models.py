"""Typed execution contracts and a causal daily-bar fill model.

The daily-bar model deliberately does not pretend to reconstruct an order
book.  It fills at a configured bar price, applies explicit spread and impact
assumptions, and can cap quantity using *lagged* average daily volume.  The
caller is responsible for supplying only information known before the fill.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd

from alphaforge.execution.costs import MAX_FILLED_SHARES, CostModel, FillCostBreakdown
from alphaforge.execution.events import MAX_ABSOLUTE_WEIGHT, MAX_MONEY, MAX_PRICE
from alphaforge.execution.frictions import ModelDeclaration

FillStatus = Literal["filled", "partial", "rejected"]
MissingPricePolicy = Literal["raise", "skip"]
_DEFAULT_EXECUTION_PROVENANCE = (
    "caller-supplied deterministic simulation assumption; not empirically calibrated"
)
MAX_ORDER_ID = 2**63 - 1
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}\Z", re.ASCII)


def _bounded_real(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _bounded_integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class ExecutionPolicy:
    """Assumptions used to turn a target rebalance into daily-bar fills."""

    price_field: Literal["open"] = "open"
    adv_lookback: int = 20
    volatility_lookback: int = 20
    max_participation_rate: float | None = None
    impact_coefficient: float = 0.0
    impact_exponent: float = 0.5
    missing_price_policy: MissingPricePolicy = "raise"
    calibration_provenance: str = _DEFAULT_EXECUTION_PROVENANCE

    def __post_init__(self) -> None:
        if self.price_field != "open":
            raise ValueError("daily-bar execution currently supports next-open fills only")
        try:
            adv_lookback = _bounded_integer(
                self.adv_lookback,
                name="adv_lookback",
                minimum=1,
                maximum=10_000,
            )
            volatility_lookback = _bounded_integer(
                self.volatility_lookback,
                name="volatility_lookback",
                minimum=2,
                maximum=10_000,
            )
        except ValueError as exc:
            raise ValueError("execution lookbacks must be positive bounded integers") from exc
        object.__setattr__(self, "adv_lookback", adv_lookback)
        object.__setattr__(self, "volatility_lookback", volatility_lookback)
        if self.max_participation_rate is not None:
            object.__setattr__(
                self,
                "max_participation_rate",
                _bounded_real(
                    self.max_participation_rate,
                    name="max_participation_rate",
                    minimum=np.nextafter(0.0, 1.0),
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "impact_coefficient",
            _bounded_real(
                self.impact_coefficient,
                name="impact_coefficient",
                minimum=0.0,
                maximum=1_000.0,
            ),
        )
        object.__setattr__(
            self,
            "impact_exponent",
            _bounded_real(
                self.impact_exponent,
                name="impact_exponent",
                minimum=0.0,
                maximum=4.0,
            ),
        )
        if self.missing_price_policy not in {"raise", "skip"}:
            raise ValueError("missing_price_policy must be 'raise' or 'skip'")
        provenance = self.calibration_provenance
        if not isinstance(provenance, str):
            raise ValueError("calibration_provenance must be a string")
        if (
            not provenance
            or provenance != provenance.strip()
            or not provenance.isprintable()
            or len(provenance) > 512
        ):
            raise ValueError(
                "calibration_provenance must be non-empty, printable, trim-exact, and at most "
                "512 characters; it is never normalized"
            )

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> ExecutionPolicy:
        if config is None:
            cfg: dict[str, object] = {}
        elif not isinstance(config, Mapping):
            raise ValueError("execution settings must be a mapping")
        elif any(not isinstance(key, str) for key in config):
            raise ValueError("execution setting keys must be strings")
        else:
            cfg = dict(config)
        allowed = {
            "price_field",
            "adv_lookback",
            "volatility_lookback",
            "max_participation_rate",
            "impact_coefficient",
            "impact_exponent",
            "missing_price_policy",
            "calibration_provenance",
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"unknown execution settings: {sorted(unknown)}")
        return cls(**cfg)  # type: ignore[arg-type]

    @property
    def declaration(self) -> ModelDeclaration:
        """Return causal input, timing, bounds, and failure semantics."""

        return ModelDeclaration(
            model_id="daily-bar-execution-policy",
            version="1.0.0",
            units=(
                ("adv", "lagged shares per logical trading session"),
                ("impact_coefficient", "dimensionless square-root/power-law coefficient"),
                ("participation", "filled shares divided by lagged ADV shares"),
                ("volatility", "lagged decimal return volatility"),
            ),
            calibration_provenance=self.calibration_provenance,
            domain="causal daily-bar next-open market-order simulation",
            parameter_bounds=(
                ("event_notional", f"finite USD in [0, {MAX_MONEY:.1e}]"),
                ("event_price", f"finite USD/share in (0, {MAX_PRICE:.1e}]"),
                ("impact_coefficient", "finite value in [0, 1000]"),
                ("impact_exponent", "finite value in [0, 4]"),
                ("lookbacks", "bounded integer trading sessions"),
                ("max_participation_rate", "None or finite ratio in (0, 1]"),
                ("order_quantity", f"finite absolute shares in [0, {MAX_FILLED_SHARES:.1e}]"),
                (
                    "target_weight",
                    f"finite value in [-{MAX_ABSOLUTE_WEIGHT:g}, {MAX_ABSOLUTE_WEIGHT:g}]",
                ),
            ),
            execution_timestamp="scheduled future-session open after all declared logical delays",
            failure_behavior=(
                "reject missing required causal ADV/volatility, invalid prices, non-positive "
                "all-in sell prices, malformed settings, and resource-bound overflow"
            ),
            limitations=(
                "Daily bars do not model queue priority, venues, auctions, or intraday paths.",
                "Impact parameters are sensitivities, not observed executable liquidity.",
            ),
        )

    @property
    def configuration_digest(self) -> str:
        """Return a stable identity for the declaration and exact policy."""

        payload = {
            "declaration_digest": self.declaration.digest,
            "price_field": self.price_field,
            "adv_lookback": self.adv_lookback,
            "volatility_lookback": self.volatility_lookback,
            "max_participation_rate": (
                None if self.max_participation_rate is None else self.max_participation_rate.hex()
            ),
            "impact_coefficient": self.impact_coefficient.hex(),
            "impact_exponent": self.impact_exponent.hex(),
            "missing_price_policy": self.missing_price_policy,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        digest = hashlib.sha256()
        digest.update(b"alphaforge.execution-policy.v1\x00")
        digest.update(encoded)
        return digest.hexdigest()


@dataclass(frozen=True)
class Order:
    """A day order generated from a close-time portfolio decision."""

    order_id: int
    symbol: str
    decision_date: pd.Timestamp
    fill_date: pd.Timestamp
    requested_shares: float
    target_weight: float
    pretrade_equity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "order_id",
            _bounded_integer(
                self.order_id,
                name="order_id",
                minimum=1,
                maximum=MAX_ORDER_ID,
            ),
        )
        if not isinstance(self.symbol, str) or _SYMBOL.fullmatch(self.symbol) is None:
            raise ValueError("symbol must be a bounded ASCII market identifier")
        for field_name in ("decision_date", "fill_date"):
            timestamp = getattr(self, field_name)
            if (
                not isinstance(timestamp, pd.Timestamp)
                or pd.isna(timestamp)
                or timestamp.tz is not None
                or timestamp != timestamp.normalize()
            ):
                raise ValueError(f"{field_name} must be a timezone-naive normalized Timestamp")
        if self.fill_date <= self.decision_date:
            raise ValueError("fill_date must be strictly after decision_date")
        object.__setattr__(
            self,
            "requested_shares",
            _bounded_real(
                self.requested_shares,
                name="requested_shares",
                minimum=-MAX_FILLED_SHARES,
                maximum=MAX_FILLED_SHARES,
            ),
        )
        object.__setattr__(
            self,
            "target_weight",
            _bounded_real(
                self.target_weight,
                name="target_weight",
                minimum=-MAX_ABSOLUTE_WEIGHT,
                maximum=MAX_ABSOLUTE_WEIGHT,
            ),
        )
        object.__setattr__(
            self,
            "pretrade_equity",
            _bounded_real(
                self.pretrade_equity,
                name="pretrade_equity",
                minimum=np.nextafter(0.0, 1.0),
                maximum=MAX_MONEY,
            ),
        )

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Fill:
    """Execution result with every modeled source of implementation shortfall."""

    order_id: int
    symbol: str
    decision_date: pd.Timestamp
    fill_date: pd.Timestamp
    status: FillStatus
    requested_shares: float
    filled_shares: float
    residual_shares: float
    reference_price: float
    fill_price: float
    target_weight: float
    pretrade_equity: float
    lagged_adv_shares: float
    lagged_volatility: float
    participation_rate: float
    commission: float
    spread_cost: float
    fixed_slippage_cost: float
    impact_cost: float
    impact_bps: float
    exchange_fee: float = 0.0
    spread_slippage_cost: float = 0.0
    participation_slippage_cost: float = 0.0
    volatility_slippage_cost: float = 0.0
    spread_slippage_bps: float = 0.0
    participation_slippage_bps: float = 0.0
    volatility_slippage_bps: float = 0.0
    cost_breakdown: FillCostBreakdown | None = None
    rejection_reason: str | None = None

    @property
    def traded_notional(self) -> float:
        return abs(self.filled_shares) * self.reference_price

    @property
    def total_cost(self) -> float:
        if self.cost_breakdown is not None:
            return self.cost_breakdown.total_cost
        return math.fsum(
            (
                self.commission,
                self.exchange_fee,
                self.spread_cost,
                self.fixed_slippage_cost,
                self.spread_slippage_cost,
                self.participation_slippage_cost,
                self.volatility_slippage_cost,
                self.impact_cost,
            )
        )

    def to_record(self) -> dict[str, object]:
        # Preserve the version-1 public fill table.  New component-level rows
        # are exposed through BacktestResult.friction_attribution instead of
        # silently widening a legacy table used by reporting integrations.
        record = {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "decision_date": self.decision_date,
            "fill_date": self.fill_date,
            "status": self.status,
            "requested_shares": self.requested_shares,
            "filled_shares": self.filled_shares,
            "residual_shares": self.residual_shares,
            "reference_price": self.reference_price,
            "fill_price": self.fill_price,
            "target_weight": self.target_weight,
            "pretrade_equity": self.pretrade_equity,
            "lagged_adv_shares": self.lagged_adv_shares,
            "lagged_volatility": self.lagged_volatility,
            "participation_rate": self.participation_rate,
            "commission": self.commission,
            "spread_cost": self.spread_cost,
            "fixed_slippage_cost": self.fixed_slippage_cost,
            "impact_cost": self.impact_cost,
            "impact_bps": self.impact_bps,
        }
        record["traded_notional"] = self.traded_notional
        record["total_cost"] = self.total_cost
        return record


class BarExecutionModel:
    """Deterministic next-open fill model for daily OHLCV research."""

    def __init__(self, costs: CostModel, policy: ExecutionPolicy) -> None:
        self.costs = costs
        self.policy = policy

    @classmethod
    def from_config(
        cls,
        costs: CostModel | dict | None = None,
        execution: ExecutionPolicy | dict | None = None,
    ) -> BarExecutionModel:
        cost_model = costs if isinstance(costs, CostModel) else CostModel.from_config(costs)
        policy = (
            execution
            if isinstance(execution, ExecutionPolicy)
            else ExecutionPolicy.from_config(execution)
        )
        return cls(cost_model, policy)

    def execute(
        self,
        order: Order,
        *,
        reference_price: float,
        lagged_adv_shares: float = np.nan,
        lagged_volatility: float = np.nan,
    ) -> Fill:
        """Execute one day order using inputs available before the open.

        ``lagged_adv_shares`` and ``lagged_volatility`` must already be lagged
        by the caller.  A participation limit with unavailable ADV rejects the
        order instead of silently assuming infinite liquidity.
        """
        if isinstance(reference_price, bool) or not isinstance(reference_price, Real):
            raise ValueError("reference_price must be a real number")
        if isinstance(lagged_adv_shares, bool) or not isinstance(lagged_adv_shares, Real):
            raise ValueError("lagged_adv_shares must be a real number or NaN")
        if isinstance(lagged_volatility, bool) or not isinstance(lagged_volatility, Real):
            raise ValueError("lagged_volatility must be a real number or NaN")
        try:
            normalized_reference = float(reference_price)
            normalized_adv = float(lagged_adv_shares)
            normalized_volatility = float(lagged_volatility)
        except (OverflowError, ValueError) as exc:
            raise ValueError("execution inputs must be representable as real numbers") from exc
        if math.isinf(normalized_adv) or math.isinf(normalized_volatility):
            raise ValueError("lagged execution inputs must be finite or NaN")
        if not np.isfinite(normalized_reference) or normalized_reference <= 0:
            if self.policy.missing_price_policy == "raise":
                raise ValueError(f"missing or invalid open price for {order.symbol}")
            return self._empty_fill(order, normalized_reference)
        if normalized_reference > MAX_PRICE:
            raise ValueError(f"reference_price must be at most {MAX_PRICE:g}")

        requested_abs = abs(order.requested_shares)
        fill_abs = requested_abs
        adv_is_valid = np.isfinite(normalized_adv) and normalized_adv > 0
        requires_adv = (
            self.policy.max_participation_rate is not None
            or self.policy.impact_coefficient > 0.0
            or self.costs.participation_slippage_bps > 0.0
        )
        requires_volatility = (
            self.policy.impact_coefficient > 0.0
            or self.costs.volatility_slippage_bps_per_1pct > 0.0
        )
        volatility_is_valid = np.isfinite(normalized_volatility) and normalized_volatility >= 0.0
        if requires_adv and not adv_is_valid:
            return self._empty_fill(
                order,
                normalized_reference,
                lagged_adv_shares=normalized_adv,
                lagged_volatility=normalized_volatility,
                rejection_reason="required_lagged_adv_unavailable",
            )
        if requires_volatility and not volatility_is_valid:
            return self._empty_fill(
                order,
                normalized_reference,
                lagged_adv_shares=normalized_adv,
                lagged_volatility=normalized_volatility,
                rejection_reason="required_lagged_volatility_unavailable",
            )
        if self.policy.max_participation_rate is not None:
            fill_abs = min(
                fill_abs,
                normalized_adv * self.policy.max_participation_rate,
            )

        if fill_abs <= 0 or requested_abs == 0:
            return self._empty_fill(
                order,
                normalized_reference,
                lagged_adv_shares=normalized_adv,
                lagged_volatility=normalized_volatility,
            )

        # A participation cap is a hard limit. Never round a capped quantity up
        # with a relative tolerance, even when the residue is small compared
        # with an unusually large order.
        status: FillStatus = "filled" if fill_abs == requested_abs else "partial"
        sign = float(np.sign(order.requested_shares))
        filled_shares = sign * fill_abs
        participation = fill_abs / normalized_adv if adv_is_valid else 0.0
        volatility = normalized_volatility if volatility_is_valid else 0.0
        impact_bps = (
            self.policy.impact_coefficient
            * volatility
            * 10_000.0
            * max(participation, 0.0) ** self.policy.impact_exponent
        )
        breakdown = self.costs.evaluate_fill(
            filled_shares=filled_shares,
            reference_price=normalized_reference,
            participation_rate=participation,
            lagged_volatility=volatility,
            impact_bps=impact_bps,
        )
        execution_bps = breakdown.embedded_shortfall_bps
        fill_price = normalized_reference * (1.0 + sign * execution_bps / 10_000.0)
        if not np.isfinite(fill_price) or fill_price <= 0.0:
            return self._empty_fill(
                order,
                normalized_reference,
                lagged_adv_shares=normalized_adv,
                lagged_volatility=normalized_volatility,
                rejection_reason="non_positive_or_non_finite_fill_price",
            )
        if fill_price > MAX_PRICE:
            raise ValueError("simulated fill price exceeds the event-contract resource ceiling")
        fill_notional = fill_abs * max(normalized_reference, fill_price)
        if not math.isfinite(fill_notional) or fill_notional > MAX_MONEY:
            raise ValueError("simulated fill notional exceeds the event-contract resource ceiling")
        observed_shortfall = fill_abs * abs(fill_price - normalized_reference)
        tolerance = 64.0 * math.ulp(
            max(observed_shortfall, breakdown.embedded_cost, fill_abs * normalized_reference, 1.0)
        )
        if abs(observed_shortfall - breakdown.embedded_cost) > tolerance:
            raise RuntimeError("price-embedded execution costs failed exact reconciliation")
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            decision_date=order.decision_date,
            fill_date=order.fill_date,
            status=status,
            requested_shares=order.requested_shares,
            filled_shares=filled_shares,
            residual_shares=order.requested_shares - filled_shares,
            reference_price=normalized_reference,
            fill_price=float(fill_price),
            target_weight=order.target_weight,
            pretrade_equity=order.pretrade_equity,
            lagged_adv_shares=normalized_adv,
            lagged_volatility=normalized_volatility,
            participation_rate=float(participation),
            commission=breakdown.commission,
            spread_cost=breakdown.spread_cost,
            fixed_slippage_cost=breakdown.fixed_slippage_cost,
            impact_cost=breakdown.impact_cost,
            impact_bps=float(impact_bps),
            exchange_fee=breakdown.exchange_fee,
            spread_slippage_cost=breakdown.spread_slippage_cost,
            participation_slippage_cost=breakdown.participation_slippage_cost,
            volatility_slippage_cost=breakdown.volatility_slippage_cost,
            spread_slippage_bps=breakdown.spread_slippage_bps,
            participation_slippage_bps=breakdown.participation_slippage_bps,
            volatility_slippage_bps=breakdown.volatility_slippage_bps,
            cost_breakdown=breakdown,
        )

    @staticmethod
    def _empty_fill(
        order: Order,
        reference_price: float,
        *,
        lagged_adv_shares: float = np.nan,
        lagged_volatility: float = np.nan,
        rejection_reason: str = "no_executable_quantity",
    ) -> Fill:
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            decision_date=order.decision_date,
            fill_date=order.fill_date,
            status="rejected",
            requested_shares=order.requested_shares,
            filled_shares=0.0,
            residual_shares=order.requested_shares,
            reference_price=float(reference_price),
            fill_price=float(reference_price),
            target_weight=order.target_weight,
            pretrade_equity=order.pretrade_equity,
            lagged_adv_shares=float(lagged_adv_shares),
            lagged_volatility=float(lagged_volatility),
            participation_rate=0.0,
            commission=0.0,
            spread_cost=0.0,
            fixed_slippage_cost=0.0,
            impact_cost=0.0,
            impact_bps=0.0,
            rejection_reason=rejection_reason,
        )
