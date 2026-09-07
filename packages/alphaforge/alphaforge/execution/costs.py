"""Causal, component-level execution-cost calculators.

Cash fees and price shortfall are intentionally different accounting paths.
Commission and exchange fees are debited from cash by ``FillApplied`` events;
spread, slippage, and market impact are embedded once in the simulated fill
price.  :class:`FillCostBreakdown` keeps those paths independently auditable
and prevents a caller from subtracting an embedded component a second time.

All defaults are caller-supplied scenario assumptions.  They are not presented
as estimates of broker charges or observed execution quality.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from alphaforge.execution.events import MAX_MONEY, MAX_PRICE
from alphaforge.execution.frictions import ModelDeclaration

MAX_COST_BPS = 1_000_000.0
MAX_PER_SHARE_USD = 1_000_000.0
MAX_MINIMUM_FEE_USD = 1_000_000_000.0
MAX_PARTICIPATION = 100.0
MAX_VOLATILITY = 100.0
MAX_MODELED_MONEY = 1.0e18
MAX_FILLED_SHARES = 1.0e15
MAX_PROVENANCE_LENGTH = 512
DEFAULT_CALIBRATION_PROVENANCE = (
    "caller-supplied deterministic scenario assumption; not empirically calibrated"
)


def _finite_number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Return one bounded float without accepting booleans as numbers."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        if minimum == 0.0:
            raise ValueError(f"{name} must be finite, non-negative, and at most {maximum}")
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _provenance(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("calibration_provenance must be a string")
    if (
        not value
        or value != value.strip()
        or not value.isprintable()
        or len(value) > MAX_PROVENANCE_LENGTH
    ):
        raise ValueError(
            "calibration_provenance must be non-empty, printable, trim-exact, and at most "
            f"{MAX_PROVENANCE_LENGTH} characters; it is never normalized"
        )
    return value


@dataclass(frozen=True, slots=True)
class FillCostBreakdown:
    """One fill's non-negative USD costs and their rate provenance.

    ``cash_fees`` are ledger debits.  ``embedded_cost`` is already represented
    by the difference between reference and fill price.  ``total_cost`` is the
    implementation shortfall and must never be debited separately.
    """

    commission: float
    exchange_fee: float
    spread_cost: float
    fixed_slippage_cost: float
    spread_slippage_cost: float
    participation_slippage_cost: float
    volatility_slippage_cost: float
    impact_cost: float
    half_spread_bps: float
    fixed_slippage_bps: float
    spread_slippage_bps: float
    participation_slippage_bps: float
    volatility_slippage_bps: float
    impact_bps: float

    def __post_init__(self) -> None:
        for field_name in (
            "commission",
            "exchange_fee",
            "spread_cost",
            "fixed_slippage_cost",
            "spread_slippage_cost",
            "participation_slippage_cost",
            "volatility_slippage_cost",
            "impact_cost",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(
                    getattr(self, field_name),
                    name=field_name,
                    minimum=0.0,
                    maximum=MAX_MODELED_MONEY,
                ),
            )
        for field_name in (
            "half_spread_bps",
            "fixed_slippage_bps",
            "spread_slippage_bps",
            "participation_slippage_bps",
            "volatility_slippage_bps",
            "impact_bps",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_number(
                    getattr(self, field_name),
                    name=field_name,
                    minimum=0.0,
                    maximum=MAX_COST_BPS,
                ),
            )
        # Force overflow detection while the record is being constructed.
        if not math.isfinite(self.total_cost) or self.total_cost > MAX_MODELED_MONEY:
            raise ValueError("modeled fill costs exceed the resource ceiling")

    @property
    def cash_fees(self) -> float:
        """Return cash-debited commission and exchange fees in USD."""

        return math.fsum((self.commission, self.exchange_fee))

    @property
    def embedded_cost(self) -> float:
        """Return price-embedded shortfall in USD."""

        return math.fsum(
            (
                self.spread_cost,
                self.fixed_slippage_cost,
                self.spread_slippage_cost,
                self.participation_slippage_cost,
                self.volatility_slippage_cost,
                self.impact_cost,
            )
        )

    @property
    def embedded_shortfall_bps(self) -> float:
        """Return total price-embedded shortfall in basis points."""

        return math.fsum(
            (
                self.half_spread_bps,
                self.fixed_slippage_bps,
                self.spread_slippage_bps,
                self.participation_slippage_bps,
                self.volatility_slippage_bps,
                self.impact_bps,
            )
        )

    @property
    def total_cost(self) -> float:
        """Return cash fees plus price-embedded shortfall in USD."""

        return math.fsum((self.cash_fees, self.embedded_cost))

    def components(self) -> tuple[tuple[str, str, float, float | None], ...]:
        """Return stable ``(name, accounting_path, USD, bps)`` records."""

        return (
            ("commission", "cash_fee", self.commission, None),
            ("exchange_fee", "cash_fee", self.exchange_fee, None),
            ("spread", "fill_price", self.spread_cost, self.half_spread_bps),
            (
                "fixed_slippage",
                "fill_price",
                self.fixed_slippage_cost,
                self.fixed_slippage_bps,
            ),
            (
                "spread_slippage",
                "fill_price",
                self.spread_slippage_cost,
                self.spread_slippage_bps,
            ),
            (
                "participation_slippage",
                "fill_price",
                self.participation_slippage_cost,
                self.participation_slippage_bps,
            ),
            (
                "volatility_slippage",
                "fill_price",
                self.volatility_slippage_cost,
                self.volatility_slippage_bps,
            ),
            ("market_impact", "fill_price", self.impact_cost, self.impact_bps),
        )


@dataclass(frozen=True, slots=True)
class CostModel:
    """Bounded fill-cost assumptions evaluated at the simulated fill.

    Rate fields use basis points of reference notional.  Per-share fields use
    USD/share.  ``minimum_commission_usd`` applies once per non-zero simulated
    fill, so a caller that splits an order into multiple fills pays the minimum
    on each fill.  ``volatility_slippage_bps_per_1pct`` is multiplied by lagged
    decimal volatility expressed in percentage points (``0.02 -> 2``).
    """

    commission_bps: float = 1.0
    half_spread_bps: float = 2.5
    slippage_bps: float = 2.0
    commission_per_share_usd: float = 0.0
    minimum_commission_usd: float = 0.0
    exchange_fee_bps: float = 0.0
    exchange_fee_per_share_usd: float = 0.0
    spread_slippage_multiplier: float = 0.0
    participation_slippage_bps: float = 0.0
    participation_slippage_exponent: float = 1.0
    volatility_slippage_bps_per_1pct: float = 0.0
    calibration_provenance: str = DEFAULT_CALIBRATION_PROVENANCE

    def __post_init__(self) -> None:
        bounded_bps = (
            "commission_bps",
            "half_spread_bps",
            "slippage_bps",
            "exchange_fee_bps",
            "participation_slippage_bps",
            "volatility_slippage_bps_per_1pct",
        )
        for field_name in bounded_bps:
            object.__setattr__(
                self,
                field_name,
                _finite_number(
                    getattr(self, field_name),
                    name=field_name,
                    minimum=0.0,
                    maximum=MAX_COST_BPS,
                ),
            )
        for field_name in ("commission_per_share_usd", "exchange_fee_per_share_usd"):
            object.__setattr__(
                self,
                field_name,
                _finite_number(
                    getattr(self, field_name),
                    name=field_name,
                    minimum=0.0,
                    maximum=MAX_PER_SHARE_USD,
                ),
            )
        object.__setattr__(
            self,
            "minimum_commission_usd",
            _finite_number(
                self.minimum_commission_usd,
                name="minimum_commission_usd",
                minimum=0.0,
                maximum=MAX_MINIMUM_FEE_USD,
            ),
        )
        object.__setattr__(
            self,
            "spread_slippage_multiplier",
            _finite_number(
                self.spread_slippage_multiplier,
                name="spread_slippage_multiplier",
                minimum=0.0,
                maximum=1_000.0,
            ),
        )
        object.__setattr__(
            self,
            "participation_slippage_exponent",
            _finite_number(
                self.participation_slippage_exponent,
                name="participation_slippage_exponent",
                minimum=0.0,
                maximum=4.0,
            ),
        )
        object.__setattr__(self, "calibration_provenance", _provenance(self.calibration_provenance))

    @property
    def rate(self) -> float:
        """Return the legacy constant-rate approximation.

        Per-share, minimum, participation, volatility, and market-impact terms
        require fill context and are deliberately absent from this helper.
        """

        constant_bps = math.fsum(
            (
                self.commission_bps,
                self.exchange_fee_bps,
                self.half_spread_bps,
                self.slippage_bps,
                self.half_spread_bps * self.spread_slippage_multiplier,
            )
        )
        return constant_bps / 10_000.0

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> CostModel:
        if config is None:
            cfg: dict[str, object] = {}
        elif not isinstance(config, Mapping):
            raise ValueError("transaction-cost settings must be a mapping")
        elif any(not isinstance(key, str) for key in config):
            raise ValueError("transaction-cost setting keys must be strings")
        else:
            cfg = dict(config)
        allowed = {
            "commission_bps",
            "half_spread_bps",
            "slippage_bps",
            "commission_per_share_usd",
            "minimum_commission_usd",
            "exchange_fee_bps",
            "exchange_fee_per_share_usd",
            "spread_slippage_multiplier",
            "participation_slippage_bps",
            "participation_slippage_exponent",
            "volatility_slippage_bps_per_1pct",
            "calibration_provenance",
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"unknown transaction-cost settings: {sorted(unknown)}")
        return cls(**cfg)  # type: ignore[arg-type]

    def costs(self, turnover: pd.Series) -> pd.Series:
        """Apply the backward-compatible constant-rate approximation."""

        return turnover.astype(float) * self.rate

    @property
    def declaration(self) -> ModelDeclaration:
        """Return the explicit units, timing, bounds, and failure contract."""

        return ModelDeclaration(
            model_id="daily-bar-fill-cost",
            version="1.0.0",
            units=(
                ("basis_point_rates", "basis points of reference notional"),
                ("per_share_rates", "USD per filled share"),
                ("result", "USD per simulated fill"),
                ("volatility", "lagged decimal return volatility"),
            ),
            calibration_provenance=self.calibration_provenance,
            domain="causal daily-bar next-open execution simulation",
            parameter_bounds=(
                ("basis_point_rates", f"finite values in [0, {MAX_COST_BPS:g}]"),
                ("filled_shares", f"finite absolute shares in [0, {MAX_FILLED_SHARES:.1e}]"),
                ("minimum_commission", f"finite USD in [0, {MAX_MINIMUM_FEE_USD:g}]"),
                ("participation", f"finite ratio in [0, {MAX_PARTICIPATION:g}]"),
                ("per_share_rates", f"finite USD/share in [0, {MAX_PER_SHARE_USD:g}]"),
                ("reference_price", f"finite USD/share in (0, {MAX_PRICE:.1e}]"),
                ("volatility", f"finite decimal value in [0, {MAX_VOLATILITY:g}]"),
            ),
            execution_timestamp=(
                "evaluated once for filled quantity at the scheduled next-open simulation"
            ),
            failure_behavior=(
                "reject malformed, missing causal, non-finite, non-positive-price, overflowed, "
                "or resource-unbounded inputs; rejected quantities incur no fill cost"
            ),
            limitations=(
                "Configured rates are sensitivities, not observed or executable broker terms.",
                "Minimum commission applies once per non-zero simulated fill.",
                "Taxes, rebates, venue tiers, and intraday queue dynamics are not modeled.",
            ),
        )

    @property
    def configuration_digest(self) -> str:
        """Return a stable identity for the declaration and exact parameters."""

        payload = {
            "declaration_digest": self.declaration.digest,
            **{
                field_name: float(getattr(self, field_name)).hex()
                for field_name in (
                    "commission_bps",
                    "half_spread_bps",
                    "slippage_bps",
                    "commission_per_share_usd",
                    "minimum_commission_usd",
                    "exchange_fee_bps",
                    "exchange_fee_per_share_usd",
                    "spread_slippage_multiplier",
                    "participation_slippage_bps",
                    "participation_slippage_exponent",
                    "volatility_slippage_bps_per_1pct",
                )
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        digest = hashlib.sha256()
        digest.update(b"alphaforge.fill-cost-model.v1\x00")
        digest.update(encoded)
        return digest.hexdigest()

    def evaluate_fill(
        self,
        *,
        filled_shares: float,
        reference_price: float,
        participation_rate: float,
        lagged_volatility: float,
        impact_bps: float,
    ) -> FillCostBreakdown:
        """Evaluate all fill-level costs from causal inputs.

        The caller must supply lagged participation and volatility.  Missing
        inputs are rejected even when a particular coefficient is zero so the
        resulting record always declares the values evaluated at the fill.
        """

        signed_shares = _finite_number(
            filled_shares,
            name="filled_shares",
            minimum=-MAX_FILLED_SHARES,
            maximum=MAX_FILLED_SHARES,
        )
        shares = abs(signed_shares)
        price = _finite_number(
            reference_price,
            name="reference_price",
            minimum=np.nextafter(0.0, 1.0),
            maximum=MAX_PRICE,
        )
        participation = _finite_number(
            participation_rate,
            name="participation_rate",
            minimum=0.0,
            maximum=MAX_PARTICIPATION,
        )
        volatility = _finite_number(
            lagged_volatility,
            name="lagged_volatility",
            minimum=0.0,
            maximum=MAX_VOLATILITY,
        )
        impact = _finite_number(
            impact_bps,
            name="impact_bps",
            minimum=0.0,
            maximum=MAX_COST_BPS,
        )
        if shares == 0.0:
            return FillCostBreakdown(*(0.0 for _ in range(14)))
        notional = shares * price
        if not math.isfinite(notional) or notional > MAX_MONEY:
            raise ValueError("reference notional exceeds the resource ceiling")
        commission = max(
            self.minimum_commission_usd,
            notional * self.commission_bps / 10_000.0 + shares * self.commission_per_share_usd,
        )
        exchange_fee = (
            notional * self.exchange_fee_bps / 10_000.0 + shares * self.exchange_fee_per_share_usd
        )
        spread_slippage_bps = self.half_spread_bps * self.spread_slippage_multiplier
        participation_slippage_bps = self.participation_slippage_bps * (
            participation**self.participation_slippage_exponent
        )
        volatility_slippage_bps = self.volatility_slippage_bps_per_1pct * (volatility * 100.0)

        def dollars(rate_bps: float) -> float:
            amount = notional * rate_bps / 10_000.0
            if not math.isfinite(amount) or amount > MAX_MODELED_MONEY:
                raise ValueError("modeled cost component exceeds the resource ceiling")
            return amount

        return FillCostBreakdown(
            commission=commission,
            exchange_fee=exchange_fee,
            spread_cost=dollars(self.half_spread_bps),
            fixed_slippage_cost=dollars(self.slippage_bps),
            spread_slippage_cost=dollars(spread_slippage_bps),
            participation_slippage_cost=dollars(participation_slippage_bps),
            volatility_slippage_cost=dollars(volatility_slippage_bps),
            impact_cost=dollars(impact),
            half_spread_bps=self.half_spread_bps,
            fixed_slippage_bps=self.slippage_bps,
            spread_slippage_bps=spread_slippage_bps,
            participation_slippage_bps=participation_slippage_bps,
            volatility_slippage_bps=volatility_slippage_bps,
            impact_bps=impact,
        )
