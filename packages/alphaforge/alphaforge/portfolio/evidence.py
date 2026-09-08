"""Net-of-cost allocation comparison across folds, regimes, and capacity (SF-S4-MR1).

An allocation policy that looks best gross, on one window, at one capital level,
has demonstrated nothing. This harness runs every policy over the same
chronological folds, splits each fold by regime, and repeats the whole grid at
several capital levels — because a policy's ranking routinely inverts as capital
grows and liquidity caps start to bind.

Three disciplines, each of which exists to prevent a specific self-deception:

**No holdout selection.** :func:`compare_allocation_policies` runs on
development folds only and takes the final holdout as a *separate, later*
argument that is scored once. There is no code path that lets holdout results
influence a parameter, because the function that chooses never sees them.

**Costs before comparison.** Turnover is charged at a declared rate on every
rebalance. A policy that rebalances hard can look better gross and worse net,
and gross-only comparison systematically favours exactly the policies that will
not survive execution.

**Capacity is a dimension, not a footnote.** The same grid runs at each capital
level with liquidity caps recomputed, so the level at which a policy stops
scaling is visible rather than discovered later.

Nothing here selects a winner. It produces the comparison; the rejection rule
belongs to the frozen qualification decision (SF-S4-MR9).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphaforge.portfolio.allocation import (
    apply_uncertainty_sizing,
    apply_volatility_target,
    inverse_volatility_portfolio,
    long_short_spread_portfolio,
    rank_weighted_portfolio,
    score_weighted_portfolio,
    top_k_portfolio,
)
from alphaforge.portfolio.contracts import (
    AllocationResult,
    InfeasibleConstraintsError,
    PortfolioConstraints,
    PortfolioError,
    liquidity_caps_from_adv,
)

#: Bars per year for annualization, matching the rest of the platform.
TRADING_DAYS = 252

#: Refusal thresholds, not tuning knobs.
MAX_FOLDS = 50
MAX_CAPITAL_LEVELS = 10
MAX_EVALUATION_DATES = 10_000
MAX_BACKTEST_ASSETS = 5_000
MAX_POLICIES = 32
MAX_REGIMES = 32
MAX_HAC_LAG = 20
MAX_PANEL_CELLS = 5_000_000
MAX_LABEL_LENGTH = 128
MAX_REASON_LENGTH = 200

_REQUIRED_RECORD_COLUMNS: frozenset[str] = frozenset(
    {
        "feasible",
        "carried",
        "reason",
        "gross_return",
        "net_return",
        "turnover",
        "cost",
    }
)

AllocationFn = Callable[..., AllocationResult]


def _finite_scalar(value: Any, *, name: str) -> float:
    """Return a finite non-boolean float through a structured evidence error."""
    if isinstance(value, (bool, np.bool_)):
        raise PortfolioError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise PortfolioError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BacktestPanel:
    """Aligned inputs for one allocation backtest.

    Attributes:
        scores: ``(date, symbol)`` predicted scores, wide.
        forward_returns: ``(date, symbol)`` return realized **after** the score
            date. Supplied pre-aligned and asserted, rather than shifted here,
            so the alignment is the caller's explicit statement.
        volatility: ``(date, symbol)`` trailing volatility for sizing.
        adv: ``(date, symbol)`` average daily volume in currency, for liquidity.
    """

    scores: pd.DataFrame
    forward_returns: pd.DataFrame
    volatility: pd.DataFrame
    adv: pd.DataFrame

    def __post_init__(self) -> None:
        frames = {
            "scores": self.scores,
            "forward_returns": self.forward_returns,
            "volatility": self.volatility,
            "adv": self.adv,
        }
        for name, frame in frames.items():
            if not isinstance(frame, pd.DataFrame):
                raise PortfolioError(f"{name} must be a pandas DataFrame")
        if self.scores.index.has_duplicates:
            raise PortfolioError("panel dates must be unique")
        if self.scores.columns.has_duplicates:
            raise PortfolioError("panel symbols must be unique")
        if not 1 <= len(self.scores.index) <= MAX_EVALUATION_DATES:
            raise PortfolioError(f"panel must hold 1..{MAX_EVALUATION_DATES} evidence dates")
        if not 1 <= len(self.scores.columns) <= MAX_BACKTEST_ASSETS:
            raise PortfolioError(f"panel must hold 1..{MAX_BACKTEST_ASSETS} evidence assets")
        if self.scores.size > MAX_PANEL_CELLS:
            raise PortfolioError(
                f"panel exceeds the {MAX_PANEL_CELLS}-cell per-frame evidence ceiling"
            )
        if bool(pd.isna(self.scores.index).any()):
            raise PortfolioError("panel dates must not contain missing labels")
        if any(
            not isinstance(symbol, str) or not symbol.strip() or len(symbol) > MAX_LABEL_LENGTH
            for symbol in self.scores.columns
        ):
            raise PortfolioError(
                f"panel symbols must be non-empty strings of at most {MAX_LABEL_LENGTH} characters"
            )
        for name, frame in frames.items():
            if frame.size > MAX_PANEL_CELLS:
                raise PortfolioError(f"{name} exceeds the {MAX_PANEL_CELLS}-cell evidence ceiling")
            try:
                values = frame.to_numpy(dtype=np.float64, copy=True)
            except (TypeError, ValueError) as exc:
                raise PortfolioError(f"{name} must contain numeric values") from exc
            if np.isinf(values).any():
                raise PortfolioError(f"{name} must not contain infinite values")
        for name, frame in frames.items():
            if name == "scores":
                continue
            if not frame.index.equals(self.scores.index):
                raise PortfolioError(f"{name} index must match the score index")
            if list(frame.columns) != list(self.scores.columns):
                raise PortfolioError(f"{name} columns must match the score columns")
        if not self.scores.index.is_monotonic_increasing:
            raise PortfolioError("panel dates must be sorted ascending")
        forward = self.forward_returns.to_numpy(dtype=np.float64, copy=False)
        volatility = self.volatility.to_numpy(dtype=np.float64, copy=False)
        adv = self.adv.to_numpy(dtype=np.float64, copy=False)
        if (forward[np.isfinite(forward)] < -1.0).any():
            raise PortfolioError("forward simple returns cannot be below -100%")
        if (volatility[np.isfinite(volatility)] < 0.0).any():
            raise PortfolioError("volatility must be non-negative when observed")
        if (adv[np.isfinite(adv)] < 0.0).any():
            raise PortfolioError("average daily volume must be non-negative when observed")

        # Detach the evidence snapshot from caller-owned frames. Pandas objects
        # remain intentionally usable by analysis code, but mutating an input
        # object after construction cannot silently rewrite this panel.
        for name, frame in frames.items():
            object.__setattr__(self, name, frame.copy(deep=True))


def _validated_evaluation_dates(panel: BacktestPanel, dates: pd.Index | None) -> pd.Index:
    """Return a bounded, unique chronological subset of the panel dates."""
    evaluation = panel.scores.index if dates is None else pd.Index(dates)
    if not 1 <= len(evaluation) <= MAX_EVALUATION_DATES:
        raise PortfolioError(f"evaluation dates must hold 1..{MAX_EVALUATION_DATES} entries")
    if evaluation.has_duplicates:
        raise PortfolioError("evaluation dates must be unique")
    if bool(pd.isna(evaluation).any()):
        raise PortfolioError("evaluation dates must not contain missing labels")
    if not evaluation.is_monotonic_increasing:
        raise PortfolioError("evaluation dates must be sorted ascending")
    try:
        missing = evaluation.difference(panel.scores.index)
    except (TypeError, ValueError) as exc:
        raise PortfolioError("evaluation dates must be comparable with panel dates") from exc
    if len(missing):
        raise PortfolioError(f"evaluation dates include {len(missing)} dates outside the panel")
    return evaluation


def _turnover_between(target: pd.Series, previous: pd.Series | None) -> float:
    """Return absolute traded weight over the union, including liquidations."""
    if not isinstance(target, pd.Series) or (
        previous is not None and not isinstance(previous, pd.Series)
    ):
        raise PortfolioError("turnover books must be pandas Series")
    if target.index.has_duplicates or (previous is not None and previous.index.has_duplicates):
        raise PortfolioError("turnover books must have unique asset labels")
    target_values = target.to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise PortfolioError("target weights must be finite")
    if previous is None:
        turnover = float(np.abs(target_values).sum())
        if not np.isfinite(turnover):
            raise PortfolioError("initial turnover is non-finite")
        return turnover
    previous_values = previous.to_numpy(dtype=float)
    if not np.isfinite(previous_values).all():
        raise PortfolioError("previous weights must be finite")
    union = target.index.union(previous.index, sort=False)
    turnover = float(
        (target.reindex(union).fillna(0.0) - previous.reindex(union).fillna(0.0)).abs().sum()
    )
    if not np.isfinite(turnover):
        raise PortfolioError("rebalance turnover is non-finite")
    return turnover


def _returns_for_book(panel: BacktestPanel, date: Any, weights: pd.Series) -> pd.Series:
    """Return finite forward returns for every non-zero holding, failing closed."""
    if not isinstance(weights, pd.Series) or weights.index.has_duplicates:
        raise PortfolioError("book weights must be a pandas Series with unique labels")
    weight_values = weights.to_numpy(dtype=float)
    if not np.isfinite(weight_values).all():
        raise PortfolioError("book weights must be finite")
    outside = weights.index.difference(panel.scores.columns)
    if len(outside):
        raise PortfolioError("book contains assets outside the evidence panel")
    aligned = panel.forward_returns.loc[date].reindex(weights.index).astype(float)
    active = np.abs(weight_values) > 1e-15
    values = aligned.to_numpy(dtype=float, copy=True)
    if active.any() and not np.isfinite(values[active]).all():
        raise PortfolioError("forward returns are missing or non-finite for an active holding")
    if active.any() and (values[active] < -1.0).any():
        raise PortfolioError("a simple asset return cannot be below -100%")
    values[~active] = 0.0
    return pd.Series(values, index=weights.index, dtype=float)


def _drift_book(
    weights: pd.Series, realized: pd.Series, *, cost: float
) -> tuple[float, float, pd.Series]:
    """Apply one self-financing period and return gross/net return and drifted weights.

    Costs are paid from equity at the rebalance. The end-of-period denominator is
    therefore ``1 + w'r - cost``; using the unchanged target weights at the next
    decision would silently assume a free rebalance back to target.
    """
    if not np.isfinite(cost) or cost < 0.0:
        raise PortfolioError("period cost must be finite and non-negative")
    if not isinstance(weights, pd.Series) or not isinstance(realized, pd.Series):
        raise PortfolioError("weights and realized returns must be pandas Series")
    if weights.index.has_duplicates or realized.index.has_duplicates:
        raise PortfolioError("weights and realized returns must have unique labels")
    missing = weights.index.difference(realized.index)
    if len(missing):
        raise PortfolioError("realized returns do not cover every book asset")
    vector = weights.to_numpy(dtype=float)
    observed = realized.reindex(weights.index).to_numpy(dtype=float)
    if not np.isfinite(vector).all() or not np.isfinite(observed).all():
        raise PortfolioError("weights and realized returns must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        gross_return = float(vector @ observed)
    if not np.isfinite(gross_return):
        raise PortfolioError("portfolio return calculation is non-finite")
    net_return = gross_return - cost
    equity = 1.0 + net_return
    if not np.isfinite(equity) or equity <= 1e-12:
        raise PortfolioError("portfolio equity was exhausted during the evidence period")
    drifted = vector * (1.0 + observed) / equity
    if not np.isfinite(drifted).all():
        raise PortfolioError("self-financing weight drift produced non-finite weights")
    return (
        gross_return,
        net_return,
        pd.Series(drifted, index=weights.index, name="pre_trade_weight"),
    )


def _weight_snapshot(weights: pd.Series) -> dict[str, float | int]:
    """Return exposure and concentration diagnostics for weights held this period."""
    values = weights.to_numpy(dtype=float)
    gross = float(np.sum(np.abs(values)))
    if not np.isfinite(gross):
        raise PortfolioError("book gross exposure is non-finite")
    concentration = float(np.sum(np.square(np.abs(values) / gross))) if gross > 0.0 else 0.0
    return {
        "gross": gross,
        "net_exposure": float(np.sum(values)),
        "n_positions": int(np.count_nonzero(np.abs(values) > 1e-12)),
        "concentration": concentration,
        "effective_n": float(1.0 / concentration) if concentration > 0.0 else 0.0,
    }


def _period_row(
    *,
    date: Any,
    weights: pd.Series,
    realized: pd.Series,
    feasible: bool,
    reason: str,
    turnover: float,
    cost: float,
    solver_status: str = "not_applicable",
    condition_number: float = float("nan"),
    ridge_applied: float = float("nan"),
    n_observations: float = float("nan"),
    observations_considered: float = float("nan"),
    complete_observation_fraction: float = float("nan"),
    primal_residual: float = float("nan"),
    dual_residual: float = float("nan"),
    complementarity_residual: float = float("nan"),
    objective_residual: float = float("nan"),
    duality_gap: float = float("nan"),
    max_certificate_residual: float = float("nan"),
    ex_ante_volatility: float = float("nan"),
) -> tuple[dict[str, Any], pd.Series]:
    """Build one honest evidence row and the next period's drifted book."""
    if not isinstance(feasible, (bool, np.bool_)):
        raise PortfolioError("period feasibility must be boolean")
    if not isinstance(reason, str):
        raise PortfolioError("period reason must be a string")
    if not feasible and not reason.strip():
        raise PortfolioError("an infeasible period requires a non-empty reason")
    if feasible and reason:
        raise PortfolioError("a feasible period cannot carry a failure reason")
    if not isinstance(solver_status, str) or not solver_status.strip():
        raise PortfolioError("solver status must be a non-empty string")
    if not np.isfinite(turnover) or turnover < 0.0:
        raise PortfolioError("period turnover must be finite and non-negative")
    if not np.isfinite(cost) or cost < 0.0:
        raise PortfolioError("period cost must be finite and non-negative")
    optional_nonnegative = {
        "condition_number": condition_number,
        "ridge_applied": ridge_applied,
        "n_observations": n_observations,
        "observations_considered": observations_considered,
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
        "complementarity_residual": complementarity_residual,
        "objective_residual": objective_residual,
        "duality_gap": duality_gap,
        "max_certificate_residual": max_certificate_residual,
        "ex_ante_volatility": ex_ante_volatility,
    }
    for name, value in optional_nonnegative.items():
        if np.isinf(value) or (np.isfinite(value) and value < 0.0):
            raise PortfolioError(f"{name} must be non-negative or unavailable")
    if np.isinf(complete_observation_fraction) or (
        np.isfinite(complete_observation_fraction)
        and not 0.0 <= complete_observation_fraction <= 1.0
    ):
        raise PortfolioError("complete_observation_fraction must lie in [0, 1] or be unavailable")
    gross_return, net_return, drifted = _drift_book(weights, realized, cost=cost)
    row: dict[str, Any] = {
        "date": date,
        "feasible": feasible,
        "carried": not feasible,
        "reason": reason[:MAX_REASON_LENGTH],
        "gross_return": gross_return,
        "net_return": net_return,
        "turnover": turnover,
        "cost": cost,
        "solver_status": solver_status,
        "condition_number": condition_number,
        "ridge_applied": ridge_applied,
        "n_observations": n_observations,
        "observations_considered": observations_considered,
        "complete_observation_fraction": complete_observation_fraction,
        "primal_residual": primal_residual,
        "dual_residual": dual_residual,
        "complementarity_residual": complementarity_residual,
        "objective_residual": objective_residual,
        "duality_gap": duality_gap,
        "max_certificate_residual": max_certificate_residual,
        "ex_ante_volatility": ex_ante_volatility,
        **_weight_snapshot(weights),
    }
    return row, drifted


def _carried_failure_row(
    panel: BacktestPanel,
    date: Any,
    previous: pd.Series | None,
    *,
    reason: str,
    solver_status: str = "failed",
    condition_number: float = float("nan"),
) -> tuple[dict[str, Any], pd.Series | None]:
    """Record a failed decision without erasing the book or gifting a cash day."""
    held = pd.Series(dtype=float) if previous is None else previous.copy()
    realized = _returns_for_book(panel, date, held)
    row, drifted = _period_row(
        date=date,
        weights=held,
        realized=realized,
        feasible=False,
        reason=reason,
        turnover=0.0,
        cost=0.0,
        solver_status=solver_status,
        condition_number=condition_number,
    )
    return row, (None if previous is None else drifted)


def _bound_sized_weights(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    liquidity_caps: pd.Series,
) -> pd.Series:
    """Uniformly de-risk a sizing overlay until every upper limit is respected."""
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PortfolioError("sizing overlay produced non-finite weights")
    if constraints.long_only and (values < -1e-12).any():
        raise PortfolioError("sizing overlay breached the long-only constraint")
    caps = liquidity_caps.reindex(weights.index).fillna(0.0).to_numpy(dtype=float)
    if (caps < 0.0).any() or not np.isfinite(caps).all():
        raise PortfolioError("liquidity caps must be finite and non-negative")

    scale = 1.0
    absolute = np.abs(values)
    gross = float(absolute.sum())
    gross_limit = min(constraints.max_gross, constraints.deployable, constraints.max_leverage)
    if gross > gross_limit and gross > 0.0:
        scale = min(scale, gross_limit / gross)
    largest = float(absolute.max()) if absolute.size else 0.0
    if largest > constraints.max_position:
        scale = min(scale, constraints.max_position / largest)
    net = abs(float(values.sum()))
    if net > constraints.max_net and net > 0.0:
        scale = min(scale, constraints.max_net / net)
    positive = absolute > 0.0
    if positive.any():
        scale = min(scale, float(np.min(caps[positive] / absolute[positive])))
    return pd.Series(values * max(min(scale, 1.0), 0.0), index=weights.index, name=weights.name)


def _newey_west_mean_standard_error(
    values: NDArray[np.float64],
) -> tuple[float, int]:
    """Return deterministic Bartlett-kernel HAC uncertainty for the sample mean.

    The lag rule ``floor(4 * (n / 100) ** (2 / 9))`` is declared in advance and
    bounded by both the available sample and :data:`MAX_HAC_LAG`. This preserves
    short-range serial dependence without allowing evidence size to create an
    unbounded quadratic calculation.
    """
    size = int(values.size)
    if size <= 1:
        return 0.0, 0
    lag = min(
        size - 1,
        MAX_HAC_LAG,
        max(int(np.floor(4.0 * (size / 100.0) ** (2.0 / 9.0))), 0),
    )
    centered = values - float(np.mean(values))
    long_run_variance = float(centered @ centered) / size
    for offset in range(1, lag + 1):
        autocovariance = float(centered[offset:] @ centered[:-offset]) / size
        bartlett_weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance
    return float(np.sqrt(max(long_run_variance, 0.0) / size)), lag


@dataclass(frozen=True)
class Fold:
    """One chronological evaluation window."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PortfolioError("fold name must be a non-empty string")
        name = self.name.strip()
        if len(name) > MAX_LABEL_LENGTH:
            raise PortfolioError(f"fold name must contain at most {MAX_LABEL_LENGTH} characters")
        try:
            start = pd.Timestamp(self.start)
            end = pd.Timestamp(self.end)
        except (TypeError, ValueError) as exc:
            raise PortfolioError(f"fold {name!r} bounds must be timestamps") from exc
        if pd.isna(start) or pd.isna(end):
            raise PortfolioError(f"fold {name!r} bounds must not be NaT")
        try:
            reversed_bounds = end < start
        except TypeError as exc:
            raise PortfolioError(f"fold {name!r} bounds must be timezone-compatible") from exc
        if reversed_bounds:
            raise PortfolioError(f"fold {name!r} ends before it starts")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def _validated_folds(panel: BacktestPanel, folds: tuple[Fold, ...]) -> tuple[Fold, ...]:
    """Validate bounded, named, non-overlapping evidence intervals."""
    if not folds:
        raise PortfolioError("at least one evaluation fold is required")
    if len(folds) > MAX_FOLDS:
        raise PortfolioError(f"folds exceed the {MAX_FOLDS}-entry ceiling")
    if any(not isinstance(fold, Fold) for fold in folds):
        raise PortfolioError("every evidence fold must be a Fold")
    names = [fold.name for fold in folds]
    if len(set(names)) != len(names):
        raise PortfolioError("fold names must be unique")
    previous: Fold | None = None
    for fold in folds:
        try:
            mask = (panel.scores.index >= fold.start) & (panel.scores.index <= fold.end)
        except TypeError as exc:
            raise PortfolioError(
                f"fold {fold.name!r} bounds are incompatible with panel dates"
            ) from exc
        if not bool(np.any(mask)):
            raise PortfolioError(f"fold {fold.name!r} contains no panel dates")
        if previous is not None and fold.start <= previous.end:
            raise PortfolioError("folds must be chronological and non-overlapping")
        previous = fold
    return folds


def _validated_regimes(panel: BacktestPanel, regimes: pd.Series | None) -> pd.Series | None:
    """Validate an optional bounded regime series covering every panel date."""
    if regimes is None:
        return None
    if not isinstance(regimes, pd.Series):
        raise PortfolioError("regimes must be a pandas Series when supplied")
    if regimes.index.has_duplicates:
        raise PortfolioError("regime dates must be unique")
    if not regimes.index.is_monotonic_increasing:
        raise PortfolioError("regime dates must be sorted ascending")
    missing = panel.scores.index.difference(regimes.index)
    if len(missing):
        raise PortfolioError(f"regimes are missing {len(missing)} panel dates")
    labels = tuple(str(value) for value in regimes.dropna().to_numpy())
    if any(not label.strip() or len(label) > MAX_LABEL_LENGTH for label in labels):
        raise PortfolioError(f"regime labels must contain 1..{MAX_LABEL_LENGTH} characters")
    distinct = set(labels) - {"unknown"}
    if len(distinct) > MAX_REGIMES:
        raise PortfolioError(f"regimes exceed the {MAX_REGIMES}-label ceiling")
    return regimes


def chronological_folds(index: pd.Index, *, n_folds: int) -> tuple[Fold, ...]:
    """Split an index into contiguous, non-overlapping chronological folds.

    Contiguous and ordered rather than shuffled: a random split of a time series
    lets a fold be trained around by its own future neighbours, which is the
    standard way a portfolio backtest acquires lookahead.
    """
    if not isinstance(index, pd.Index):
        raise PortfolioError("fold index must be a pandas Index")
    if not isinstance(n_folds, int) or isinstance(n_folds, bool):
        raise PortfolioError("n_folds must be an integer")
    if not 1 <= n_folds <= MAX_FOLDS:
        raise PortfolioError(f"n_folds must be in [1, {MAX_FOLDS}]")
    if not 1 <= len(index) <= MAX_EVALUATION_DATES:
        raise PortfolioError(f"fold index must hold 1..{MAX_EVALUATION_DATES} dates")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise PortfolioError("fold index must be unique and sorted ascending")
    if bool(pd.isna(index).any()):
        raise PortfolioError("fold index must not contain missing labels")
    if len(index) < n_folds:
        raise PortfolioError("index is shorter than the requested fold count")
    bounds = np.array_split(np.arange(len(index)), n_folds)
    return tuple(
        Fold(name=f"fold_{position + 1}", start=index[block[0]], end=index[block[-1]])
        for position, block in enumerate(bounds)
        if block.size
    )


def volatility_regimes(
    returns: pd.DataFrame, *, window: int = 63, quantile: float = 0.7
) -> pd.Series:
    """Label each date ``calm`` or ``stress`` by trailing cross-sectional volatility.

    The threshold is a quantile of the **trailing** series only, so a date's
    label never depends on volatility that had not happened yet. Warm-up dates
    are labelled ``unknown`` rather than guessed.
    """
    if not isinstance(returns, pd.DataFrame):
        raise PortfolioError("regime returns must be a pandas DataFrame")
    if not 1 <= len(returns.index) <= MAX_EVALUATION_DATES:
        raise PortfolioError(f"regime returns must hold 1..{MAX_EVALUATION_DATES} dates")
    if not 1 <= len(returns.columns) <= MAX_BACKTEST_ASSETS:
        raise PortfolioError(f"regime returns must hold 1..{MAX_BACKTEST_ASSETS} assets")
    if returns.size > MAX_PANEL_CELLS:
        raise PortfolioError(f"regime returns exceed the {MAX_PANEL_CELLS}-cell evidence ceiling")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise PortfolioError("regime return dates must be unique and sorted ascending")
    try:
        values = returns.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise PortfolioError("regime returns must be numeric") from exc
    if np.isinf(values).any():
        raise PortfolioError("regime returns must not contain infinite values")
    if not isinstance(window, int) or isinstance(window, bool):
        raise PortfolioError("regime window must be an integer")
    if window < 2:
        raise PortfolioError("regime window must be at least two observations")
    if window > len(returns.index):
        raise PortfolioError("regime window cannot exceed the available observations")
    quantile = _finite_scalar(quantile, name="regime quantile")
    if not 0.0 < quantile < 1.0:
        raise PortfolioError("regime quantile must be finite and lie in (0, 1)")
    dispersion = returns.mean(axis=1).rolling(window, min_periods=window).std()
    labels = pd.Series("unknown", index=returns.index, dtype=object)
    expanding_threshold = dispersion.expanding(min_periods=window).quantile(quantile)
    observed = dispersion.notna() & expanding_threshold.notna()
    labels[observed] = np.where(
        dispersion[observed] > expanding_threshold[observed], "stress", "calm"
    )
    return labels.rename("regime")


def _default_policies() -> dict[str, AllocationFn]:
    """Return the policy grid, each pinned to its declared parameters."""
    return {
        "top_k_equal": lambda scores, vol, constraints, previous, caps: top_k_portfolio(
            scores,
            constraints,
            k=max(len(scores) // 5, 1),
            weighting="equal",
            previous=previous,
            liquidity_caps=caps,
        ),
        "top_k_rank": lambda scores, vol, constraints, previous, caps: top_k_portfolio(
            scores,
            constraints,
            k=max(len(scores) // 5, 1),
            weighting="rank",
            previous=previous,
            liquidity_caps=caps,
        ),
        "long_short_spread": lambda scores, vol, constraints, previous, caps: (
            long_short_spread_portfolio(
                scores,
                constraints,
                k=max(len(scores) // 5, 1),
                previous=previous,
                liquidity_caps=caps,
            )
        ),
        "score_weighted": lambda scores, vol, constraints, previous, caps: (
            score_weighted_portfolio(scores, constraints, previous=previous, liquidity_caps=caps)
        ),
        "rank_weighted": lambda scores, vol, constraints, previous, caps: (
            rank_weighted_portfolio(scores, constraints, previous=previous, liquidity_caps=caps)
        ),
        "inverse_volatility": lambda scores, vol, constraints, previous, caps: (
            inverse_volatility_portfolio(
                scores, vol, constraints, previous=previous, liquidity_caps=caps
            )
        ),
    }


def run_allocation_backtest(
    panel: BacktestPanel,
    policy: AllocationFn,
    constraints: PortfolioConstraints,
    *,
    capital: float,
    cost_bps: float,
    participation: float = 0.05,
    uncertainty: pd.DataFrame | None = None,
    uncertainty_strength: float = 0.0,
    volatility_target: float | None = None,
    dates: pd.Index | None = None,
) -> pd.DataFrame:
    """Run one policy over the panel and return its per-date net record.

    Returns a frame with gross and net return, turnover, cost, gross exposure,
    net exposure and position count per rebalance date. A date whose constraints
    are infeasible at this capital level is recorded with ``feasible=False``
    rather than dropped, so capacity exhaustion is visible in the record instead
    of quietly shrinking the sample.
    """
    if not isinstance(panel, BacktestPanel):
        raise PortfolioError("panel must be a BacktestPanel")
    if not callable(policy):
        raise PortfolioError("policy must be callable")
    if not isinstance(constraints, PortfolioConstraints):
        raise PortfolioError("constraints must be PortfolioConstraints")
    cost_bps = _finite_scalar(cost_bps, name="cost_bps")
    capital = _finite_scalar(capital, name="capital")
    participation = _finite_scalar(participation, name="participation")
    uncertainty_strength = _finite_scalar(uncertainty_strength, name="uncertainty_strength")
    if cost_bps < 0.0:
        raise PortfolioError("cost_bps must be finite and non-negative")
    if capital <= 0.0:
        raise PortfolioError("capital must be finite and positive")
    if not 0.0 < participation <= 1.0:
        raise PortfolioError("participation must be finite and lie in (0, 1]")
    if not 0.0 <= uncertainty_strength <= 1.0:
        raise PortfolioError("uncertainty_strength must lie in [0, 1]")
    if volatility_target is not None:
        volatility_target = _finite_scalar(volatility_target, name="volatility_target")
        if volatility_target <= 0.0:
            raise PortfolioError("volatility_target must be finite and positive when supplied")
    if uncertainty is not None:
        if not isinstance(uncertainty, pd.DataFrame):
            raise PortfolioError("uncertainty must be a pandas DataFrame when supplied")
        if not uncertainty.index.equals(panel.scores.index):
            raise PortfolioError("uncertainty index must match the score index")
        if list(uncertainty.columns) != list(panel.scores.columns):
            raise PortfolioError("uncertainty columns must match the score columns")
        try:
            uncertainty_values = uncertainty.to_numpy(dtype=np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise PortfolioError("uncertainty must contain numeric values") from exc
        finite_uncertainty = uncertainty_values[np.isfinite(uncertainty_values)]
        if np.isinf(uncertainty_values).any() or (finite_uncertainty < 0.0).any():
            raise PortfolioError("uncertainty must be finite-or-missing and non-negative")
    evaluation_dates = _validated_evaluation_dates(panel, dates)
    previous: pd.Series | None = None
    rows: list[dict[str, Any]] = []

    for date in evaluation_dates:
        scores = panel.scores.loc[date].dropna()
        if scores.empty:
            row, previous = _carried_failure_row(
                panel, date, previous, reason="no finite scores on the decision date"
            )
            rows.append(row)
            continue
        if not np.isfinite(scores.to_numpy(dtype=float)).all():
            row, previous = _carried_failure_row(
                panel, date, previous, reason="scores are non-finite on the decision date"
            )
            rows.append(row)
            continue
        try:
            ex_ante_volatility = float("nan")
            volatility = panel.volatility.loc[date].reindex(scores.index)
            caps = liquidity_caps_from_adv(
                panel.adv.loc[date].reindex(scores.index).fillna(0.0),
                capital=capital,
                participation=participation,
            )
            result = policy(
                scores.copy(deep=True),
                volatility.copy(deep=True),
                constraints,
                None if previous is None else previous.copy(deep=True),
                caps.copy(deep=True),
            )
            if not isinstance(result, AllocationResult):
                raise PortfolioError("policy must return an AllocationResult")
            weights = result.weights.copy()
            outside = weights.index.difference(scores.index)
            if len(outside):
                raise PortfolioError("policy allocated assets outside the decision universe")
            if uncertainty is not None and uncertainty_strength > 0.0:
                weights = apply_uncertainty_sizing(
                    weights,
                    uncertainty.loc[date].reindex(weights.index),
                    strength=uncertainty_strength,
                )
            if volatility_target is not None:
                periodic_volatility = panel.volatility.loc[date].reindex(weights.index)
                if (
                    periodic_volatility.isna().any()
                    or not np.isfinite(periodic_volatility.to_numpy(dtype=float)).all()
                    or (periodic_volatility <= 0.0).any()
                ):
                    raise PortfolioError(
                        "volatility targeting requires finite positive volatility for every holding"
                    )
                covariance = pd.DataFrame(
                    np.diag(np.square(periodic_volatility.to_numpy(dtype=float))),
                    index=weights.index,
                    columns=weights.index,
                )
                weights, volatility_diagnostics = apply_volatility_target(
                    weights,
                    covariance,
                    target_volatility=volatility_target,
                    max_leverage=constraints.max_leverage,
                )
                ex_ante_volatility = float(
                    volatility_diagnostics.get("scaled_ex_ante_volatility", float("nan"))
                )
                if not np.isfinite(ex_ante_volatility) or ex_ante_volatility < 0.0:
                    raise PortfolioError(
                        "volatility sizing must expose finite non-negative ex-ante volatility"
                    )
            weights = _bound_sized_weights(weights, constraints, caps)
            turnover = _turnover_between(weights, previous)
            if (
                previous is not None
                and constraints.max_turnover is not None
                and turnover > constraints.max_turnover + 1e-9
            ):
                raise InfeasibleConstraintsError(
                    "the transformed target breaches max_turnover after explicit liquidation "
                    f"accounting ({turnover:.6f} > {constraints.max_turnover:.6f})"
                )
            cost = turnover * cost_bps / 10_000.0
            realized = _returns_for_book(panel, date, weights)
            row, drifted = _period_row(
                date=date,
                weights=weights,
                realized=realized,
                feasible=True,
                reason="",
                turnover=turnover,
                cost=cost,
                ex_ante_volatility=ex_ante_volatility,
            )
        except (InfeasibleConstraintsError, PortfolioError) as exc:
            row, previous = _carried_failure_row(
                panel, date, previous, reason=str(exc), solver_status="not_applicable"
            )
            rows.append(row)
            continue
        rows.append(row)
        previous = drifted

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _validated_record(record: pd.DataFrame) -> pd.DataFrame:
    """Return a detached evidence record after strict accounting validation."""
    if not isinstance(record, pd.DataFrame):
        raise PortfolioError("evidence record must be a pandas DataFrame")
    if len(record) > MAX_EVALUATION_DATES:
        raise PortfolioError(f"evidence record exceeds the {MAX_EVALUATION_DATES}-date ceiling")
    missing = sorted(_REQUIRED_RECORD_COLUMNS - set(record.columns))
    if missing:
        raise PortfolioError(f"evidence record is missing required columns: {missing}")
    if record.index.has_duplicates or not record.index.is_monotonic_increasing:
        raise PortfolioError("evidence record index must be unique and sorted ascending")

    normalized = record.copy(deep=True)
    for name in ("feasible", "carried"):
        values = normalized[name]
        if values.isna().any() or any(
            not isinstance(value, (bool, np.bool_)) for value in values.to_numpy()
        ):
            raise PortfolioError(f"evidence {name} flags must be strict booleans")
        normalized[name] = values.astype(bool)
    if not normalized["carried"].equals(~normalized["feasible"]):
        raise PortfolioError("evidence carried flags must be the inverse of feasible flags")

    reasons = normalized["reason"]
    if reasons.isna().any() or any(not isinstance(value, str) for value in reasons.to_numpy()):
        raise PortfolioError("evidence reasons must be strings")
    stripped = reasons.astype(str).str.strip()
    if (stripped.str.len() > MAX_REASON_LENGTH).any():
        raise PortfolioError(
            f"evidence reasons must contain at most {MAX_REASON_LENGTH} characters"
        )
    if (normalized["feasible"] & stripped.ne("")).any():
        raise PortfolioError("feasible evidence rows cannot carry failure reasons")
    if ((~normalized["feasible"]) & stripped.eq("")).any():
        raise PortfolioError("infeasible evidence rows require a failure reason")

    required_numeric = ("gross_return", "net_return", "turnover", "cost")
    for name in required_numeric:
        try:
            values = pd.to_numeric(normalized[name], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PortfolioError(f"evidence {name} must be numeric") from exc
        if not np.isfinite(values).all():
            raise PortfolioError(f"evidence {name} must be finite")
        normalized[name] = values
    if (normalized["turnover"] < 0.0).any() or (normalized["cost"] < 0.0).any():
        raise PortfolioError("evidence turnover and cost must be non-negative")
    if (normalized["net_return"] <= -1.0 + 1e-12).any():
        raise PortfolioError("evidence net returns must preserve strictly positive equity")
    expected_net = normalized["gross_return"].to_numpy(dtype=np.float64) - normalized[
        "cost"
    ].to_numpy(dtype=np.float64)
    if not np.allclose(
        normalized["net_return"].to_numpy(dtype=np.float64),
        expected_net,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise PortfolioError("evidence net returns do not reconcile to gross return minus cost")

    optional_nonnegative = (
        "gross",
        "n_positions",
        "concentration",
        "effective_n",
        "condition_number",
        "ridge_applied",
        "n_observations",
        "observations_considered",
        "primal_residual",
        "dual_residual",
        "complementarity_residual",
        "objective_residual",
        "duality_gap",
        "max_certificate_residual",
        "ex_ante_volatility",
    )
    for name in optional_nonnegative:
        if name not in normalized:
            continue
        try:
            values = pd.to_numeric(normalized[name], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PortfolioError(f"evidence {name} must be numeric when supplied") from exc
        if np.isinf(values).any() or (values[np.isfinite(values)] < 0.0).any():
            raise PortfolioError(f"evidence {name} must be non-negative or unavailable")
        normalized[name] = values
    if "complete_observation_fraction" in normalized:
        try:
            fractions = pd.to_numeric(
                normalized["complete_observation_fraction"], errors="raise"
            ).to_numpy(dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PortfolioError("evidence complete_observation_fraction must be numeric") from exc
        finite = fractions[np.isfinite(fractions)]
        if np.isinf(fractions).any() or ((finite < 0.0) | (finite > 1.0)).any():
            raise PortfolioError(
                "evidence complete_observation_fraction must lie in [0, 1] or be unavailable"
            )
        normalized["complete_observation_fraction"] = fractions
    if "net_exposure" in normalized:
        try:
            exposure = pd.to_numeric(normalized["net_exposure"], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise PortfolioError("evidence net_exposure must be numeric") from exc
        if np.isinf(exposure).any():
            raise PortfolioError("evidence net_exposure must be finite or unavailable")
        normalized["net_exposure"] = exposure
    if "solver_status" in normalized:
        statuses = normalized["solver_status"]
        if statuses.isna().any() or any(
            not isinstance(value, str) or not value.strip() for value in statuses.to_numpy()
        ):
            raise PortfolioError("evidence solver statuses must be non-empty strings")
    return normalized


def summarize_record(record: pd.DataFrame) -> dict[str, Any]:
    """Reduce a per-date record to complete net-of-cost risk evidence.

    VaR and CVaR are reported as positive losses at the empirical 95% level.
    Confidence bounds quantify sampling uncertainty in the annualized arithmetic
    mean through a deterministic, bounded Newey--West/Bartlett HAC estimator.
    This preserves short-range serial dependence instead of treating market bars
    as independent observations.
    """
    if not isinstance(record, pd.DataFrame):
        raise PortfolioError("evidence record must be a pandas DataFrame")
    empty: dict[str, Any] = {
        "n_dates": 0,
        "n_feasible": 0,
        "n_failures": 0,
        "feasible_fraction": 0.0,
        "coverage_fraction": 0.0,
        "performance_evidence_available": False,
        "net_return": float("nan"),
        "net_return_ci95_low": float("nan"),
        "net_return_ci95_high": float("nan"),
        "uncertainty_method": "newey_west_hac_bartlett",
        "hac_lag": 0,
        "net_sharpe": float("nan"),
        "realized_volatility": float("nan"),
        "gross_return": float("nan"),
        "cost_drag": float("nan"),
        "max_drawdown": float("nan"),
        "var_95_loss": float("nan"),
        "cvar_95_loss": float("nan"),
        "worst_bar_return": float("nan"),
        "mean_turnover": float("nan"),
        "max_turnover": float("nan"),
        "mean_gross_exposure": float("nan"),
        "max_gross_exposure": float("nan"),
        "mean_net_exposure": float("nan"),
        "mean_absolute_net_exposure": float("nan"),
        "mean_concentration": float("nan"),
        "max_concentration": float("nan"),
        "mean_effective_n": float("nan"),
        "mean_n_positions": float("nan"),
        "mean_ex_ante_volatility": float("nan"),
        "mean_condition_number": float("nan"),
        "max_condition_number": float("nan"),
        "mean_ridge_applied": float("nan"),
        "max_ridge_applied": float("nan"),
        "ridge_applied_fraction": float("nan"),
        "mean_n_observations": float("nan"),
        "min_n_observations": float("nan"),
        "mean_observations_considered": float("nan"),
        "mean_complete_observation_fraction": float("nan"),
        "min_complete_observation_fraction": float("nan"),
        "mean_primal_residual": float("nan"),
        "max_primal_residual": float("nan"),
        "mean_dual_residual": float("nan"),
        "max_dual_residual": float("nan"),
        "mean_complementarity_residual": float("nan"),
        "max_complementarity_residual": float("nan"),
        "mean_objective_residual": float("nan"),
        "max_objective_residual": float("nan"),
        "median_max_certificate_residual": float("nan"),
        "p95_max_certificate_residual": float("nan"),
        "max_certificate_residual": float("nan"),
        "median_duality_gap": float("nan"),
        "p95_duality_gap": float("nan"),
        "max_duality_gap": float("nan"),
        "solver_failures": 0,
        "failure_reasons": "",
    }
    if record.empty:
        return empty
    record = _validated_record(record)
    net: NDArray[np.float64] = record["net_return"].to_numpy(dtype=np.float64)
    equity: NDArray[np.float64] = np.cumprod(1.0 + net)
    if not np.isfinite(equity).all() or (equity <= 0.0).any():
        raise PortfolioError("evidence equity path is non-finite or exhausted")
    peak = np.maximum.accumulate(equity)
    deviation = float(np.std(net, ddof=1)) if net.size > 1 else 0.0
    standard_error, hac_lag = _newey_west_mean_standard_error(net)
    annualized_mean = float(np.mean(net) * TRADING_DAYS)
    quantile = float(np.quantile(net, 0.05))
    tail = net[net <= quantile]
    feasible = record["feasible"].astype(bool)
    failures = int((~feasible).sum())

    def _finite(column: str) -> NDArray[np.float64]:
        if column not in record:
            return np.asarray([], dtype=np.float64)
        values = record[column].to_numpy(dtype=float)
        return np.asarray(values[np.isfinite(values)], dtype=np.float64)

    def _mean(column: str) -> float:
        finite = _finite(column)
        return float(finite.mean()) if finite.size else float("nan")

    def _min(column: str) -> float:
        finite = _finite(column)
        return float(finite.min()) if finite.size else float("nan")

    def _max(column: str) -> float:
        finite = _finite(column)
        return float(finite.max()) if finite.size else float("nan")

    def _quantile(column: str, probability: float) -> float:
        finite = _finite(column)
        return float(np.quantile(finite, probability)) if finite.size else float("nan")

    solver_failures = 0
    if "solver_status" in record:
        statuses = record["solver_status"].astype(str)
        solver_failures = int((~statuses.isin({"optimal", "not_applicable"})).sum())
    reasons = record.loc[~feasible, "reason"].astype(str)
    reason_counts = reasons[reasons.str.len() > 0].value_counts().head(5)
    failure_reasons = " | ".join(
        f"{reason[:120]} ({count})" for reason, count in reason_counts.items()
    )

    result = {
        "n_dates": int(len(record)),
        "n_feasible": int(feasible.sum()),
        "n_failures": failures,
        "feasible_fraction": float(feasible.mean()),
        "coverage_fraction": float(feasible.mean()),
        "performance_evidence_available": bool(feasible.any()),
        "net_return": annualized_mean,
        "net_return_ci95_low": float(annualized_mean - 1.96 * standard_error * TRADING_DAYS),
        "net_return_ci95_high": float(annualized_mean + 1.96 * standard_error * TRADING_DAYS),
        "uncertainty_method": "newey_west_hac_bartlett",
        "hac_lag": hac_lag,
        "net_sharpe": (
            float(np.mean(net) / deviation * np.sqrt(TRADING_DAYS)) if deviation > 0.0 else 0.0
        ),
        "realized_volatility": float(deviation * np.sqrt(TRADING_DAYS)),
        "gross_return": float(record["gross_return"].mean() * TRADING_DAYS),
        "cost_drag": float(record["cost"].mean() * TRADING_DAYS),
        "max_drawdown": float(np.min(equity / peak - 1.0)),
        "var_95_loss": float(max(-quantile, 0.0)),
        "cvar_95_loss": float(max(-float(np.mean(tail)), 0.0)),
        "worst_bar_return": float(np.min(net)),
        "mean_turnover": float(record["turnover"].mean()),
        "max_turnover": float(record["turnover"].max()),
        "mean_gross_exposure": _mean("gross"),
        "max_gross_exposure": _max("gross"),
        "mean_net_exposure": _mean("net_exposure"),
        "mean_absolute_net_exposure": (
            float(record["net_exposure"].abs().mean()) if "net_exposure" in record else float("nan")
        ),
        "mean_concentration": _mean("concentration"),
        "max_concentration": _max("concentration"),
        "mean_effective_n": _mean("effective_n"),
        "mean_n_positions": _mean("n_positions"),
        "mean_ex_ante_volatility": _mean("ex_ante_volatility"),
        "mean_condition_number": _mean("condition_number"),
        "max_condition_number": _max("condition_number"),
        "mean_ridge_applied": _mean("ridge_applied"),
        "max_ridge_applied": _max("ridge_applied"),
        "ridge_applied_fraction": (
            float(np.mean(_finite("ridge_applied") > 0.0))
            if _finite("ridge_applied").size
            else float("nan")
        ),
        "mean_n_observations": _mean("n_observations"),
        "min_n_observations": _min("n_observations"),
        "mean_observations_considered": _mean("observations_considered"),
        "mean_complete_observation_fraction": _mean("complete_observation_fraction"),
        "min_complete_observation_fraction": _min("complete_observation_fraction"),
        "mean_primal_residual": _mean("primal_residual"),
        "max_primal_residual": _max("primal_residual"),
        "mean_dual_residual": _mean("dual_residual"),
        "max_dual_residual": _max("dual_residual"),
        "mean_complementarity_residual": _mean("complementarity_residual"),
        "max_complementarity_residual": _max("complementarity_residual"),
        "mean_objective_residual": _mean("objective_residual"),
        "max_objective_residual": _max("objective_residual"),
        "median_max_certificate_residual": _quantile("max_certificate_residual", 0.50),
        "p95_max_certificate_residual": _quantile("max_certificate_residual", 0.95),
        "max_certificate_residual": _max("max_certificate_residual"),
        "median_duality_gap": _quantile("duality_gap", 0.50),
        "p95_duality_gap": _quantile("duality_gap", 0.95),
        "max_duality_gap": _max("duality_gap"),
        "solver_failures": solver_failures,
        "failure_reasons": failure_reasons,
    }
    if not bool(feasible.any()):
        for name in (
            "net_return",
            "net_return_ci95_low",
            "net_return_ci95_high",
            "net_sharpe",
            "realized_volatility",
            "gross_return",
            "cost_drag",
            "max_drawdown",
            "var_95_loss",
            "cvar_95_loss",
            "worst_bar_return",
        ):
            result[name] = float("nan")
    return result


def compare_allocation_policies(
    panel: BacktestPanel,
    constraints: PortfolioConstraints,
    *,
    folds: tuple[Fold, ...],
    capital_levels: tuple[float, ...],
    cost_bps: float = 10.0,
    policies: dict[str, AllocationFn] | None = None,
    regimes: pd.Series | None = None,
    participation: float = 0.05,
) -> pd.DataFrame:
    """Compare every policy across folds, regimes, and capacity levels.

    **Development folds only.** The final holdout is deliberately not an
    argument: score it once with :func:`score_final_holdout` after every
    parameter is frozen. A function that cannot see the holdout cannot select on
    it.

    Rows are sorted by ``(policy, capital, fold, regime)`` — never by
    performance, because ranking a comparison table by its own metric invites
    reading the top row as a decision.
    """
    _validated_folds(panel, folds)
    regimes = _validated_regimes(panel, regimes)
    if not 1 <= len(capital_levels) <= MAX_CAPITAL_LEVELS:
        raise PortfolioError(f"capital_levels must hold 1..{MAX_CAPITAL_LEVELS} entries")
    try:
        normalized_capital = tuple(
            _finite_scalar(level, name="capital level") for level in capital_levels
        )
    except TypeError as exc:
        raise PortfolioError("capital_levels must be a bounded iterable") from exc
    if any(level <= 0.0 for level in normalized_capital):
        raise PortfolioError("every capital level must be finite and positive")
    if len(set(normalized_capital)) != len(normalized_capital):
        raise PortfolioError("capital_levels must be unique")
    policies = _default_policies() if policies is None else policies
    if not isinstance(policies, dict):
        raise PortfolioError("policies must be a dictionary")
    if not 1 <= len(policies) <= MAX_POLICIES:
        raise PortfolioError(f"policies must hold 1..{MAX_POLICIES} entries")
    if any(
        not isinstance(name, str) or not name.strip() or len(name) > MAX_LABEL_LENGTH
        for name in policies
    ):
        raise PortfolioError(f"policy names must contain 1..{MAX_LABEL_LENGTH} characters")
    if any(not callable(policy) for policy in policies.values()):
        raise PortfolioError("every policy must be callable")

    rows: list[dict[str, Any]] = []
    for name in sorted(policies):
        for capital in sorted(normalized_capital):
            record = run_allocation_backtest(
                panel,
                policies[name],
                constraints,
                capital=capital,
                cost_bps=cost_bps,
                participation=participation,
            )
            if record.empty:
                continue
            for fold in folds:
                window = record.loc[(record.index >= fold.start) & (record.index <= fold.end)]
                slices: list[tuple[str, pd.DataFrame]] = [("all", window)]
                if regimes is not None:
                    aligned = regimes.reindex(window.index)
                    string_labels = aligned.astype("string")
                    for label in sorted(set(string_labels.dropna()) - {"unknown"}):
                        slices.append((str(label), window.loc[string_labels == label]))
                for regime_label, block in slices:
                    if block.empty:
                        continue
                    rows.append(
                        {
                            "policy": name,
                            "capital": capital,
                            "fold": fold.name,
                            "regime": regime_label,
                            **summarize_record(block),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["policy", "capital", "fold", "regime"]).reset_index(drop=True)


def score_final_holdout(
    panel: BacktestPanel,
    policy: AllocationFn,
    constraints: PortfolioConstraints,
    *,
    holdout_dates: pd.Index,
    capital: float,
    cost_bps: float = 10.0,
    participation: float = 0.05,
) -> dict[str, Any]:
    """Score **one** frozen policy on the untouched holdout, once.

    Separate from the comparison by design. The holdout's only legitimate use is
    to report what a decision already made would have produced; running the
    comparison grid on it would convert it into another development fold and
    destroy the only unbiased estimate available.
    """
    record = run_allocation_backtest(
        panel,
        policy,
        constraints,
        capital=capital,
        cost_bps=cost_bps,
        participation=participation,
        dates=holdout_dates,
    )
    summary = summarize_record(record)
    summary["holdout"] = True
    summary["capital"] = capital
    summary["cost_bps"] = cost_bps
    return summary


def capacity_frontier(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return each policy's net performance as a function of capital.

    The number that decides whether a strategy is investable at size. A policy
    whose net Sharpe collapses between two capital levels has found its capacity
    ceiling, and reading only the smallest level would have hidden it.
    """
    if comparison.empty:
        return comparison
    overall = comparison.loc[comparison["regime"] == "all"]
    grouped = (
        overall.groupby(["policy", "capital"], as_index=False)
        .agg(
            net_sharpe=("net_sharpe", "mean"),
            net_return=("net_return", "mean"),
            cost_drag=("cost_drag", "mean"),
            feasible_fraction=("feasible_fraction", "mean"),
        )
        .sort_values(["policy", "capital"])
        .reset_index(drop=True)
    )
    return grouped
