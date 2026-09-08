"""Explicit portfolio constraints and a true feasible projection (SF-S4-MR1).

Most portfolio bugs are not in the optimizer. They are in the step afterwards,
where target weights are forced to obey limits: a book is clipped to a position
cap and then renormalized to a gross target, which pushes the clipped names
straight back over the cap. The result satisfies neither constraint and no test
notices, because the numbers still look like weights.

This module therefore separates two things the codebase previously conflated:

* a :class:`PortfolioConstraints` record that states every limit in one place,
  in stated units, with a **feasibility check** — an impossible combination is
  refused up front rather than silently approximated; and
* :func:`project_to_feasible`, an alternating projection onto the intersection
  of those limits that **asserts its post-conditions** and raises if it cannot
  satisfy them.

Every constraint here is convex, so alternating projection converges to a point
in the intersection when one exists. That is exactly why the feasibility check
matters: on an empty intersection the iteration would wander forever and return
something plausible-looking.

Cash is explicit throughout. The issue's non-goal is hidden cash assumptions, so
gross exposure is stated against deployable capital after the cash buffer, and
:class:`AllocationResult` carries the cash weight rather than leaving it implied
by whatever the weights failed to sum to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Refusal thresholds, not tuning knobs.
MAX_UNIVERSE = 5_000
MAX_PROJECTION_ITERATIONS = 500

#: Absolute tolerance for declaring a constraint satisfied. Tighter than any
#: economically meaningful weight, loose enough to survive float accumulation.
CONSTRAINT_TOLERANCE = 1e-9


class PortfolioError(ValueError):
    """Base class for portfolio construction failures."""


class InfeasibleConstraintsError(PortfolioError):
    """Raised when no weight vector can satisfy every declared limit.

    Separated from :class:`PortfolioError` because it is actionable in a
    specific way: the caller must relax a limit or widen the universe, not
    retry.
    """


@dataclass(frozen=True)
class PortfolioConstraints:
    """Every limit a target book must satisfy, in one auditable record.

    Attributes:
        max_position: Largest absolute weight in any single name, as a fraction
            of deployable capital.
        max_gross: Ceiling on the sum of absolute weights. ``1.0`` is a fully
            invested unlevered book.
        max_net: Ceiling on the absolute value of the summed weights. Equal to
            ``max_gross`` for a long-only book; smaller for a market-neutral one.
        max_turnover: Ceiling on the sum of absolute weight changes at a
            rebalance. ``None`` disables the limit; ``0.0`` freezes the book.
        max_leverage: Ceiling on gross exposure per unit of equity. Kept
            separate from ``max_gross`` because they answer different questions:
            gross is a policy target, leverage is a solvency limit.
        cash_buffer: Fraction of capital deliberately left uninvested. Deployable
            capital is ``1 - cash_buffer``, and gross is measured against it.
        long_only: Forbid negative weights.

    Raises:
        PortfolioError: On an individually invalid limit.
    """

    max_position: float = 0.10
    max_gross: float = 1.0
    max_net: float = 1.0
    max_turnover: float | None = None
    max_leverage: float = 1.0
    cash_buffer: float = 0.0
    long_only: bool = False

    def __post_init__(self) -> None:
        for name in ("max_position", "max_gross", "max_leverage"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise PortfolioError(f"{name} must be finite and positive")
        if not np.isfinite(self.max_net) or self.max_net < 0.0:
            raise PortfolioError("max_net must be finite and non-negative")
        if self.max_turnover is not None and (
            not np.isfinite(self.max_turnover) or self.max_turnover < 0.0
        ):
            raise PortfolioError("max_turnover must be finite and non-negative when set")
        if not 0.0 <= self.cash_buffer < 1.0:
            raise PortfolioError("cash_buffer must lie in [0, 1)")
        if self.max_net > self.max_gross:
            raise PortfolioError("max_net cannot exceed max_gross")
        if self.max_gross > self.max_leverage:
            raise PortfolioError("max_gross cannot exceed max_leverage")
        if self.max_position > self.max_gross:
            raise PortfolioError("max_position cannot exceed max_gross")

    @property
    def deployable(self) -> float:
        """Capital available to invest after the cash buffer."""
        return 1.0 - self.cash_buffer

    def check_feasible(self, n_assets: int, liquidity_caps: FloatArray | None = None) -> None:
        """Refuse an empty constraint set before any projection is attempted.

        Raises:
            InfeasibleConstraintsError: If no vector can satisfy every limit.
        """
        if n_assets < 1:
            raise InfeasibleConstraintsError("cannot allocate over an empty universe")
        if n_assets > MAX_UNIVERSE:
            raise InfeasibleConstraintsError(f"universe exceeds the {MAX_UNIVERSE}-name ceiling")
        target = min(self.max_gross, self.deployable)
        reachable = self.max_position * n_assets
        if reachable < target - CONSTRAINT_TOLERANCE:
            raise InfeasibleConstraintsError(
                f"max_position {self.max_position} across {n_assets} names reaches gross "
                f"{reachable:.4f}, below the required {target:.4f}; relax the position cap "
                "or widen the universe"
            )
        if liquidity_caps is not None:
            available = float(np.sum(np.abs(liquidity_caps)))
            if available < target - CONSTRAINT_TOLERANCE:
                raise InfeasibleConstraintsError(
                    f"liquidity caps permit gross {available:.4f}, below the required "
                    f"{target:.4f}; reduce capital or the gross target"
                )
        if self.long_only and self.max_net + CONSTRAINT_TOLERANCE < target:
            raise InfeasibleConstraintsError(
                "a long-only book has net equal to gross, so max_net below the gross "
                "target is unsatisfiable"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for the run manifest."""
        return {
            "max_position": self.max_position,
            "max_gross": self.max_gross,
            "max_net": self.max_net,
            "max_turnover": self.max_turnover,
            "max_leverage": self.max_leverage,
            "cash_buffer": self.cash_buffer,
            "long_only": self.long_only,
        }


@dataclass(frozen=True)
class AllocationResult:
    """Target weights with the accounting that makes them auditable.

    ``cash_weight`` is carried explicitly rather than inferred from ``1 - sum``,
    because for a long-short book those are different numbers and conflating
    them is how a hidden leverage assumption enters a backtest.
    """

    weights: pd.Series
    policy: str
    cash_weight: float
    gross: float
    net: float
    turnover: float
    n_positions: int
    constraints: PortfolioConstraints
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.weights, pd.Series):
            raise PortfolioError("weights must be a pandas Series indexed by symbol")
        if self.weights.index.has_duplicates:
            raise PortfolioError("weights must have a unique symbol index")
        if not np.isfinite(self.weights.to_numpy(dtype=float)).all():
            raise PortfolioError("weights must be finite")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for the research ledger."""
        return {
            "policy": self.policy,
            "cash_weight": self.cash_weight,
            "gross": self.gross,
            "net": self.net,
            "turnover": self.turnover,
            "n_positions": self.n_positions,
            "constraints": self.constraints.to_dict(),
            **self.diagnostics,
        }


def project_to_feasible(
    target: FloatArray,
    constraints: PortfolioConstraints,
    *,
    previous: FloatArray | None = None,
    liquidity_caps: FloatArray | None = None,
) -> FloatArray:
    """Project ``target`` onto the intersection of every declared limit.

    Alternating projection over convex sets: box (position cap, long-only, and
    per-name liquidity), the L1 ball (gross), the net band, and the turnover
    ball around ``previous``. Each pass moves the point onto one set; repeating
    converges into the intersection.

    This is deliberately **not** clip-then-rescale. Rescaling to hit a gross
    target re-inflates names that the position cap had just pulled down, so the
    result violates the cap it was asked to respect — a bug that produces
    plausible-looking weights and no error.

    Raises:
        InfeasibleConstraintsError: If the limit set is empty, checked before
            iterating.
        PortfolioError: If the iteration fails to reach feasibility, rather than
            returning an infeasible book.
    """
    weights = np.asarray(target, dtype=np.float64).copy()
    if weights.ndim != 1:
        raise PortfolioError("target weights must be one-dimensional")
    if not np.isfinite(weights).all():
        raise PortfolioError("target weights must be finite")
    constraints.check_feasible(weights.size, liquidity_caps)

    caps = np.full(weights.size, constraints.max_position, dtype=np.float64)
    if liquidity_caps is not None:
        liquidity = np.asarray(liquidity_caps, dtype=np.float64)
        if liquidity.shape != weights.shape:
            raise PortfolioError("liquidity caps must align with the target weights")
        if (liquidity < 0.0).any() or not np.isfinite(liquidity).all():
            raise PortfolioError("liquidity caps must be finite and non-negative")
        caps = np.minimum(caps, liquidity)

    gross_target = min(constraints.max_gross, constraints.deployable, constraints.max_leverage)
    lower = np.zeros_like(caps) if constraints.long_only else -caps

    # A turnover cap and the exposure limits can conflict outright: moving at
    # most `max_turnover` from `previous` bounds how far gross and net can
    # travel, so a book that starts too far outside the limits cannot reach them.
    # Checking it here converts a confusing non-convergence into the actionable
    # statement that the trade budget is too small for the required correction.
    if constraints.max_turnover is not None and previous is not None:
        start = np.asarray(previous, dtype=np.float64)
        if start.shape != weights.shape:
            raise PortfolioError("previous weights must align with the target weights")
        budget = constraints.max_turnover
        net_excess = abs(float(np.sum(start))) - constraints.max_net
        gross_excess = float(np.sum(np.abs(start))) - gross_target
        if max(net_excess, gross_excess) > budget + CONSTRAINT_TOLERANCE:
            raise InfeasibleConstraintsError(
                f"a turnover budget of {budget} cannot correct the previous book: it is "
                f"{max(net_excess, gross_excess):.4f} outside the gross/net limits; raise "
                "max_turnover or relax the exposure limits"
            )

    # Names the policy actually selected. Redistribution may only ever move
    # capital between these: filling a name the policy set to zero would change
    # *which* assets the book holds, which is a different portfolio, not a
    # projection of this one.
    active = np.abs(np.asarray(target, dtype=np.float64)) > 0.0

    for _ in range(MAX_PROJECTION_ITERATIONS):
        weights = np.clip(weights, lower, caps)

        gross = float(np.sum(np.abs(weights)))
        if gross > gross_target + CONSTRAINT_TOLERANCE and gross > 0.0:
            weights *= gross_target / gross
        elif gross < gross_target - CONSTRAINT_TOLERANCE and gross > 0.0:
            # Clipping to the position cap frees capital. Without redistribution
            # the book silently under-deploys — it satisfies every limit while
            # holding unintended cash, which looks like a conservative choice
            # and is actually an accounting bug.
            weights = _fill_to_target(weights, gross_target, active, lower, caps)

        net = float(np.sum(weights))
        if abs(net) > constraints.max_net + CONSTRAINT_TOLERANCE:
            weights = _shift_net(weights, net, constraints, lower, caps)

        if constraints.max_turnover is not None and previous is not None:
            weights = _limit_turnover(weights, previous, constraints.max_turnover)

        if _satisfied(weights, constraints, previous, lower, caps, gross_target):
            return weights

    raise PortfolioError(
        "constraint projection did not converge; the limit set is likely "
        "degenerate even though it passed the feasibility check"
    )


def _fill_to_target(
    weights: FloatArray,
    gross_target: float,
    active: NDArray[np.bool_],
    lower: FloatArray,
    caps: FloatArray,
) -> FloatArray:
    """Distribute unused gross across active names that still have headroom.

    Proportional to each name's remaining headroom, so a name already at its cap
    absorbs nothing and the relative shape of the book is preserved. Iterated,
    because filling one name to its cap frees the next round's headroom
    calculation; it stops when the target is met or no headroom remains.
    """
    filled = weights.copy()
    for _ in range(64):
        gross = float(np.sum(np.abs(filled)))
        shortfall = gross_target - gross
        if shortfall <= CONSTRAINT_TOLERANCE:
            break
        headroom = np.where(active, caps - np.abs(filled), 0.0)
        available = float(np.sum(headroom))
        if available <= CONSTRAINT_TOLERANCE:
            break
        share = headroom / available * min(shortfall, available)
        # Grow each position along its own sign so directions never flip.
        sign = np.sign(filled)
        sign[sign == 0.0] = 1.0
        filled = np.clip(filled + sign * share, lower, caps)
    return filled


def _shift_net(
    weights: FloatArray,
    net: float,
    constraints: PortfolioConstraints,
    lower: FloatArray,
    caps: FloatArray,
) -> FloatArray:
    """Reduce |net| by trimming the side that causes the imbalance.

    Trimming proportionally on the offending side rather than shifting every
    weight by a constant keeps the relative ranking within that side intact —
    a constant shift would flip small positions through zero and change which
    names the book is even long.
    """
    excess = abs(net) - constraints.max_net
    side = weights > 0.0 if net > 0.0 else weights < 0.0
    magnitude = float(np.sum(np.abs(weights[side])))
    if magnitude <= 0.0:
        return weights
    scale = max(1.0 - excess / magnitude, 0.0)
    adjusted = weights.copy()
    adjusted[side] = weights[side] * scale
    return np.clip(adjusted, lower, caps)


def _limit_turnover(weights: FloatArray, previous: FloatArray, cap: float) -> FloatArray:
    """Shrink the trade toward ``previous`` until turnover fits its cap.

    Scaling the whole trade vector preserves its direction, so a capped
    rebalance is a partial move toward the same target rather than an
    arbitrary different book.
    """
    if previous.shape != weights.shape:
        raise PortfolioError("previous weights must align with the target weights")
    delta = weights - previous
    turnover = float(np.sum(np.abs(delta)))
    if turnover <= cap + CONSTRAINT_TOLERANCE or turnover <= 0.0:
        return weights
    return previous + delta * (cap / turnover)


def _satisfied(
    weights: FloatArray,
    constraints: PortfolioConstraints,
    previous: FloatArray | None,
    lower: FloatArray,
    caps: FloatArray,
    gross_target: float,
) -> bool:
    """Return whether every declared limit currently holds."""
    if (weights < lower - CONSTRAINT_TOLERANCE).any():
        return False
    if (weights > caps + CONSTRAINT_TOLERANCE).any():
        return False
    if float(np.sum(np.abs(weights))) > gross_target + CONSTRAINT_TOLERANCE:
        return False
    if abs(float(np.sum(weights))) > constraints.max_net + CONSTRAINT_TOLERANCE:
        return False
    return not (
        constraints.max_turnover is not None
        and previous is not None
        and float(np.sum(np.abs(weights - previous)))
        > constraints.max_turnover + CONSTRAINT_TOLERANCE
    )


def liquidity_caps_from_adv(adv: pd.Series, *, capital: float, participation: float) -> pd.Series:
    """Return the per-name weight cap implied by a participation limit.

    A position is only realizable if it can be traded without dominating the
    name's volume. ``participation`` is the fraction of average daily volume the
    book is willing to be, so the weight cap is
    ``participation * ADV / capital``.

    A name with zero or missing ADV receives a **zero** cap: untradeable rather
    than unconstrained. Defaulting an unknown ADV to "no limit" is how an
    illiquid name acquires a full-size position in a backtest.
    """
    if capital <= 0.0 or not np.isfinite(capital):
        raise PortfolioError("capital must be finite and positive")
    if not 0.0 < participation <= 1.0:
        raise PortfolioError("participation must lie in (0, 1]")
    values = adv.astype(float).to_numpy()
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    return pd.Series(values * participation / capital, index=adv.index, name="liquidity_cap")
