"""Frozen Monte Carlo perturbation of the execution path (SF-S4-MR8).

A backtest reports one history. That history is a single draw from a process
that could plausibly have gone otherwise: the same signal reaching the desk a
bar late, fills landing at a slightly worse price, one order rejected, a day of
prices missing. If a result survives only the exact sequence that happened, it
is a description of that sequence and not of a strategy.

This module perturbs the parts of the path a strategy does not control and
reports the *distribution* of outcomes.

**What may be perturbed and what may not.** Perturbation applies to execution
mechanics — order sequence, fill price, signal timing, missed and delayed
trades, size, cost, liquidity, partial fills. It never touches the returns
themselves. Perturbing the P&L directly would manufacture the answer.

**Bounded distributions, declared before the run.** Every generator draws from a
bounded family with declared parameters, published under a content-derived
identity. `verify_frozen_perturbations` refuses a grid that differs from the one
recorded beforehand, so widening a distribution after seeing the tails is a
detectable error rather than a silent one.

**Frequency is not probability.** A Monte Carlo p-value states how often *this
model of the world* produced an outcome at least this bad. It is not the
probability of that outcome in the market, because the perturbation family is an
assumption. :class:`PerturbationOutcome` carries that caveat in its own record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

#: Refusal thresholds, not tuning knobs.
MAX_REPLICATES: Final = 20_000
MAX_KINDS: Final = 32
MAX_NAME_CHARS: Final = 64
MAX_MAGNITUDE: Final = 1.0

#: Fewer paths than this cannot support a tail estimate: the 5th percentile of
#: 50 draws is the second-worst path, which is noise, not a quantile.
MIN_REPLICATES: Final = 200

#: The perturbation kinds this module knows how to apply. A kind outside this
#: set is refused rather than silently ignored, because a stress grid that
#: quietly drops an arm reports coverage it does not have.
PERTURBATION_KINDS: Final = (
    "trade_order",
    "execution_price",
    "signal_timestamp",
    "missing_trade",
    "delayed_trade",
    "position_size",
    "cost_multiplier",
    "liquidity_haircut",
    "partial_fill",
)


class PerturbationError(ValueError):
    """Raised when a perturbation specification or run is unusable."""


class InsolventPathError(PerturbationError):
    """Raised when a perturbed path drives equity to or below zero.

    Separate from the general error because insolvency is a *result* worth
    counting, not a malfunction: :func:`run_perturbation_study` records these
    paths rather than discarding them.
    """


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PerturbationError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text or text != value:
        raise PerturbationError(f"{field_name} must be non-empty and free of padding")
    if len(text) > MAX_NAME_CHARS:
        raise PerturbationError(f"{field_name} exceeds {MAX_NAME_CHARS} characters")
    if not text.isascii():
        raise PerturbationError(f"{field_name} must be ASCII")
    return text


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerturbationError(f"{field_name} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise PerturbationError(f"{field_name} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    """One bounded perturbation family, declared before the run.

    Attributes:
        kind: Which mechanic is perturbed. Must be in :data:`PERTURBATION_KINDS`.
        magnitude: Scale of the disturbance, in units natural to the kind —
            a fraction of price for ``execution_price``, a probability for
            ``missing_trade``, a multiplier offset for ``cost_multiplier``.
            Bounded at :data:`MAX_MAGNITUDE` because a perturbation large enough
            to dominate the result is testing the perturbation, not the strategy.
        bound: Hard clip applied after drawing, so no single draw can escape the
            declared envelope even in the tail of an unbounded family.
    """

    kind: str
    magnitude: float
    bound: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _name(self.kind, field_name="perturbation kind"))
        if self.kind not in PERTURBATION_KINDS:
            raise PerturbationError(
                f"unsupported perturbation kind {self.kind!r}; supported: "
                f"{', '.join(PERTURBATION_KINDS)}"
            )
        magnitude = _finite(self.magnitude, field_name="magnitude")
        if not 0.0 < magnitude <= MAX_MAGNITUDE:
            raise PerturbationError(f"magnitude must lie in (0, {MAX_MAGNITUDE}]")
        bound = _finite(self.bound, field_name="bound")
        if not magnitude <= bound <= MAX_MAGNITUDE:
            raise PerturbationError(
                f"bound must lie in [magnitude, {MAX_MAGNITUDE}]; a bound below the "
                "magnitude would clip the distribution to a point"
            )
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "bound", bound)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly declaration."""
        return {"kind": self.kind, "magnitude": self.magnitude, "bound": self.bound}


@dataclass(frozen=True)
class FrozenPerturbationGrid:
    """The complete stress grid, frozen before any candidate is qualified.

    Raises:
        PerturbationError: On duplicate kinds, an empty grid, or an
            out-of-range replicate count.
    """

    name: str
    specs: tuple[PerturbationSpec, ...]
    replicates: int
    root_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="grid name"))
        specs = tuple(self.specs)
        if not specs:
            raise PerturbationError("a perturbation grid must declare at least one spec")
        if len(specs) > MAX_KINDS:
            raise PerturbationError(f"grid exceeds the {MAX_KINDS}-spec ceiling")
        kinds = [spec.kind for spec in specs]
        if len(set(kinds)) != len(kinds):
            raise PerturbationError(
                "each perturbation kind may appear once; two magnitudes for one kind "
                "makes the applied disturbance ambiguous"
            )
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, int):
            raise PerturbationError("replicates must be an int")
        if not MIN_REPLICATES <= self.replicates <= MAX_REPLICATES:
            raise PerturbationError(
                f"replicates must lie in [{MIN_REPLICATES}, {MAX_REPLICATES}]; fewer "
                "paths cannot support a tail estimate"
            )
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise PerturbationError("root_seed must be an int")
        if not 0 <= self.root_seed < 2**32:
            raise PerturbationError("root_seed must lie in [0, 2**32)")
        object.__setattr__(self, "specs", tuple(sorted(specs, key=lambda item: item.kind)))

    @property
    def identity(self) -> str:
        """Content identity frozen before evaluation."""
        return _digest(
            {
                "name": self.name,
                "specs": [spec.to_dict() for spec in self.specs],
                "replicates": self.replicates,
                "root_seed": self.root_seed,
            }
        )

    def seed_stream(self, kind: str, index: int) -> np.random.Generator:
        """Return the named, derived generator for one kind and replicate.

        Derived from ``(root_seed, hash(kind), index)`` rather than drawn from a
        shared advancing generator, so replicate ``k`` of the price stream is
        identical whether the study runs in order, in parallel, or resumes after
        a crash — and one kind's draws never shift another's.
        """
        stream = _name(kind, field_name="seed stream kind")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PerturbationError("seed stream index must be a non-negative int")
        label = int.from_bytes(hashlib.sha256(stream.encode("utf-8")).digest()[:4], "big")
        return np.random.default_rng(np.random.SeedSequence([self.root_seed, label, index]))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly frozen declaration."""
        return {
            "name": self.name,
            "identity": self.identity,
            "specs": [spec.to_dict() for spec in self.specs],
            "replicates": self.replicates,
            "root_seed": self.root_seed,
            "seed_derivation": "SeedSequence([root_seed, sha256(kind)[:4], replicate_index])",
            "frozen_before_qualification": True,
        }


def verify_frozen_perturbations(grid: FrozenPerturbationGrid, expected_identity: str) -> None:
    """Refuse a grid that differs from the one frozen before qualification.

    Raises:
        PerturbationError: On any divergence, naming both identities.
    """
    if not isinstance(expected_identity, str) or len(expected_identity) != 64:
        raise PerturbationError("expected_identity must be a full SHA-256 digest")
    if grid.identity != expected_identity:
        raise PerturbationError(
            f"perturbation grid {grid.name!r} does not match the frozen declaration: "
            f"executing {grid.identity[:12]}, frozen {expected_identity[:12]}. "
            "Widening a stress distribution after seeing the tails makes the grid a "
            "function of the outcome it is offered to stress."
        )


def standard_execution_grid(
    *, replicates: int = 2_000, root_seed: int = 0
) -> FrozenPerturbationGrid:
    """Return the default grid covering every mechanic the work item names.

    Magnitudes are deliberately modest: each represents an ordinary operational
    imperfection — a few basis points of slippage, a one-in-fifty missed order —
    rather than a crisis. A strategy that cannot survive ordinary imperfection
    does not need a crisis to fail.
    """
    return FrozenPerturbationGrid(
        name="standard_execution",
        specs=(
            PerturbationSpec(kind="trade_order", magnitude=1.0, bound=1.0),
            PerturbationSpec(kind="execution_price", magnitude=0.0010, bound=0.0100),
            PerturbationSpec(kind="signal_timestamp", magnitude=0.10, bound=0.50),
            PerturbationSpec(kind="missing_trade", magnitude=0.02, bound=0.10),
            PerturbationSpec(kind="delayed_trade", magnitude=0.05, bound=0.25),
            PerturbationSpec(kind="position_size", magnitude=0.05, bound=0.25),
            PerturbationSpec(kind="cost_multiplier", magnitude=0.25, bound=1.00),
            PerturbationSpec(kind="liquidity_haircut", magnitude=0.10, bound=0.50),
            PerturbationSpec(kind="partial_fill", magnitude=0.10, bound=0.50),
        ),
        replicates=replicates,
        root_seed=root_seed,
    )


# ---------------------------------------------------------------------------
# Applying a perturbation to one path
# ---------------------------------------------------------------------------


def _clip(draws: np.ndarray, bound: float) -> np.ndarray:
    return np.clip(draws, -bound, bound)


def perturb_path(
    returns: pd.Series,
    costs: pd.Series,
    spec: PerturbationSpec,
    generator: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """Apply one perturbation kind to a return and cost path.

    ``returns`` are the strategy's *per-bar realized* returns and ``costs`` the
    per-bar cost drag, both positive-cost convention. The perturbation acts on
    the mechanics that produce them; it never adds a free term to the return.

    Returns:
        The perturbed ``(returns, costs)`` pair, index-aligned to the input.

    Raises:
        PerturbationError: On misaligned or non-finite inputs.
    """
    if not isinstance(returns, pd.Series) or not isinstance(costs, pd.Series):
        raise PerturbationError("returns and costs must be pandas Series")
    if not returns.index.equals(costs.index):
        raise PerturbationError(
            "returns and costs must share an index; a cost charged on a bar the "
            "strategy did not trade is not a perturbation, it is a bug"
        )
    if returns.empty:
        raise PerturbationError("returns must be non-empty")
    values = returns.to_numpy(dtype=float)
    charges = costs.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(charges).all():
        raise PerturbationError("returns and costs must be finite")
    if (charges < 0.0).any():
        raise PerturbationError("costs use a positive-charge convention and cannot be negative")

    count = values.size
    perturbed = values.copy()
    perturbed_costs = charges.copy()

    if spec.kind == "trade_order":
        # Reordering the sequence of realized bar outcomes. The multiset is
        # conserved, so **compounded return is invariant under this arm** —
        # multiplication commutes, and prod(1+r) does not depend on order. That
        # is not a defect: the arm exists to stress *path-dependent* quantities,
        # and drawdown is strongly order-dependent even when the total is not.
        # `test_reordering_conserves_compounded_return_exactly` pins the
        # invariance so a future edit cannot quietly make this arm move a number
        # it has no mechanism to move.
        order = generator.permutation(count)
        perturbed = values[order]
        perturbed_costs = charges[order]
    elif spec.kind == "execution_price":
        # A worse or better fill shifts the bar's realized return directly.
        shocks = _clip(generator.normal(0.0, spec.magnitude, count), spec.bound)
        perturbed = values + shocks
    elif spec.kind == "signal_timestamp":
        # The signal is observed late, so the strategy holds the position it
        # would have held a bar ago: bar t earns bar t-1's outcome. Costs are
        # unchanged — it traded, just on stale information.
        #
        # An earlier draft blended a fraction of each bar into the next. That is
        # a *smoothing* operator, and smoothing a fixed-sum path always lowers
        # variance drag, so compounded return improved on every single replicate.
        # A timing error that can only help is not a stress, and
        # `test_timing_jitter_can_hurt_as_well_as_help` fails if this arm ever
        # becomes one-directional again. Substitution keeps the scale of the
        # returns intact and is genuinely two-sided.
        stale = generator.random(count) < spec.magnitude
        previous = np.concatenate(([values[0]], values[:-1]))
        perturbed = np.where(stale, previous, perturbed)
    elif spec.kind == "missing_trade":
        # An order never reaches the market: the bar earns nothing and costs
        # nothing. Both sides must drop together or the accounting breaks.
        missed = generator.random(count) < spec.magnitude
        perturbed = np.where(missed, 0.0, perturbed)
        perturbed_costs = np.where(missed, 0.0, perturbed_costs)
    elif spec.kind == "delayed_trade":
        # The fill lands one bar late: the return arrives shifted, and the bar
        # it left earns nothing.
        delayed = generator.random(count) < spec.magnitude
        shifted = np.zeros(count, dtype=float)
        shifted[1:] = np.where(delayed[:-1], values[:-1], 0.0)
        perturbed = np.where(delayed, 0.0, perturbed) + shifted
    elif spec.kind == "position_size":
        # Size error scales the bar's exposure, and therefore both its return
        # and the cost of holding it.
        scale = 1.0 + _clip(generator.normal(0.0, spec.magnitude, count), spec.bound)
        scale = np.maximum(scale, 0.0)
        perturbed = perturbed * scale
        perturbed_costs = perturbed_costs * scale
    elif spec.kind == "cost_multiplier":
        # Costs come in higher than modeled. One-sided by construction: a
        # symmetric cost shock would let the study average away the drag it is
        # supposed to be testing.
        inflation = np.abs(_clip(generator.normal(0.0, spec.magnitude, count), spec.bound))
        perturbed_costs = perturbed_costs * (1.0 + inflation)
    elif spec.kind == "liquidity_haircut":
        # Thin liquidity means the intended size cannot be reached, so the bar
        # captures less of its return while still paying to trade.
        haircut = np.abs(_clip(generator.normal(0.0, spec.magnitude, count), spec.bound))
        perturbed = perturbed * (1.0 - haircut)
    else:  # partial_fill
        # Only part of the order fills. The unfilled part earns nothing and is
        # not charged, but the filled part pays full freight.
        filled = 1.0 - np.abs(_clip(generator.normal(0.0, spec.magnitude, count), spec.bound))
        filled = np.clip(filled, 0.0, 1.0)
        perturbed = perturbed * filled
        perturbed_costs = perturbed_costs * filled

    return (
        pd.Series(perturbed, index=returns.index, name=returns.name),
        pd.Series(perturbed_costs, index=costs.index, name=costs.name),
    )


# ---------------------------------------------------------------------------
# Path metrics and structured failure analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PathOutcome:
    """Metrics for one perturbed path, including the ones that failed."""

    replicate: int
    kind: str
    net_return: float
    gross_return: float
    total_cost: float
    max_drawdown: float
    volatility: float
    insolvent: bool
    no_trade: bool
    reconciled: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "replicate": self.replicate,
            "kind": self.kind,
            "net_return": self.net_return,
            "gross_return": self.gross_return,
            "total_cost": self.total_cost,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "insolvent": self.insolvent,
            "no_trade": self.no_trade,
            "reconciled": self.reconciled,
        }


def _max_drawdown(equity: np.ndarray) -> float:
    """Return the deepest peak-to-trough fraction, as a non-positive number."""
    peaks = np.maximum.accumulate(equity)
    # A peak can only be non-positive once equity has already gone to zero; the
    # drawdown is then complete by definition and further ratios are undefined.
    safe = np.where(peaks > 0.0, peaks, np.nan)
    with np.errstate(invalid="ignore"):
        drawdowns = equity / safe - 1.0
    finite = drawdowns[np.isfinite(drawdowns)]
    if finite.size == 0:
        return -1.0
    return float(min(finite.min(), 0.0))


def evaluate_path(
    returns: pd.Series, costs: pd.Series, *, replicate: int, kind: str
) -> PathOutcome:
    """Compute one path's outcome, marking insolvency rather than raising.

    Equity compounds from 1.0. A path whose equity reaches zero is **insolvent**
    and stops compounding there — a strategy cannot lose more than everything
    and then recover, and letting equity go negative would let a blown-up path
    contribute a positive average.
    """
    gross = returns.to_numpy(dtype=float)
    charges = costs.to_numpy(dtype=float)
    net = gross - charges

    equity = np.empty(net.size + 1, dtype=float)
    equity[0] = 1.0
    insolvent = False
    for position, step in enumerate(net):
        if insolvent:
            equity[position + 1] = 0.0
            continue
        nxt = equity[position] * (1.0 + step)
        if nxt <= 0.0:
            equity[position + 1] = 0.0
            insolvent = True
        else:
            equity[position + 1] = nxt

    ending = float(equity[-1])
    gross_total = float(np.prod(1.0 + gross) - 1.0) if not insolvent else -1.0
    total_cost = float(charges.sum())
    no_trade = bool(np.all(gross == 0.0) and np.all(charges == 0.0))

    # Reconciliation: compounding the recorded net steps must reproduce the
    # recorded ending equity. A path that fails this is reported as unreconciled
    # rather than silently averaged in with the sound ones.
    if insolvent:
        reconciled = True
    else:
        expected = float(np.prod(1.0 + net))
        scale = max(abs(expected), abs(ending), 1.0)
        reconciled = bool(abs(expected - ending) <= 1e-9 * scale)

    return PathOutcome(
        replicate=replicate,
        kind=kind,
        net_return=ending - 1.0,
        gross_return=gross_total,
        total_cost=total_cost,
        max_drawdown=_max_drawdown(equity),
        volatility=float(np.std(net, ddof=1)) if net.size > 1 else 0.0,
        insolvent=insolvent,
        no_trade=no_trade,
        reconciled=reconciled,
    )


@dataclass(frozen=True, slots=True)
class PerturbationOutcome:
    """The distribution of outcomes under one perturbation kind."""

    kind: str
    baseline_net_return: float
    median_net_return: float
    percentile_5: float
    percentile_95: float
    worse_than_baseline: float
    baseline_max_drawdown: float
    median_max_drawdown: float
    worst_max_drawdown: float
    insolvent_paths: int
    unreconciled_paths: int
    no_trade_paths: int
    replicates: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly evidence row."""
        return {
            "kind": self.kind,
            "baseline_net_return": self.baseline_net_return,
            "median_net_return": self.median_net_return,
            "percentile_5": self.percentile_5,
            "percentile_95": self.percentile_95,
            "worse_than_baseline": self.worse_than_baseline,
            "baseline_max_drawdown": self.baseline_max_drawdown,
            "median_max_drawdown": self.median_max_drawdown,
            "worst_max_drawdown": self.worst_max_drawdown,
            "insolvent_paths": self.insolvent_paths,
            "unreconciled_paths": self.unreconciled_paths,
            "no_trade_paths": self.no_trade_paths,
            "replicates": self.replicates,
            "interpretation": (
                "worse_than_baseline is the frequency with which THIS perturbation "
                "model produced an outcome at or below the unperturbed path. It is "
                "not the probability of that outcome in the market: the perturbation "
                "family is an assumption, not a measurement."
            ),
        }


def run_perturbation_study(
    returns: pd.Series,
    costs: pd.Series,
    grid: FrozenPerturbationGrid,
    *,
    expected_identity: str | None = None,
) -> dict[str, Any]:
    """Run every declared perturbation kind and report the outcome distributions.

    Insolvent, no-trade, and unreconciled paths are **counted, not dropped**. A
    study that silently discards the paths where the strategy blew up reports the
    conditional distribution given survival, which is the number that flatters.

    Raises:
        PerturbationError: If the grid diverges from ``expected_identity``.
    """
    if expected_identity is not None:
        verify_frozen_perturbations(grid, expected_identity)

    baseline = evaluate_path(returns, costs, replicate=-1, kind="baseline")
    outcomes: list[PerturbationOutcome] = []
    paths: list[PathOutcome] = []

    for spec in grid.specs:
        nets = np.empty(grid.replicates, dtype=float)
        drawdowns = np.empty(grid.replicates, dtype=float)
        insolvent = 0
        unreconciled = 0
        no_trade = 0
        for index in range(grid.replicates):
            generator = grid.seed_stream(spec.kind, index)
            shifted_returns, shifted_costs = perturb_path(returns, costs, spec, generator)
            outcome = evaluate_path(shifted_returns, shifted_costs, replicate=index, kind=spec.kind)
            nets[index] = outcome.net_return
            drawdowns[index] = outcome.max_drawdown
            insolvent += int(outcome.insolvent)
            unreconciled += int(not outcome.reconciled)
            no_trade += int(outcome.no_trade)
            if index < 32:  # bounded retention; the full path set is not stored
                paths.append(outcome)
        lower, upper = np.quantile(nets, [0.05, 0.95])
        outcomes.append(
            PerturbationOutcome(
                kind=spec.kind,
                baseline_net_return=baseline.net_return,
                median_net_return=float(np.median(nets)),
                percentile_5=float(lower),
                percentile_95=float(upper),
                worse_than_baseline=float(np.mean(nets <= baseline.net_return)),
                baseline_max_drawdown=baseline.max_drawdown,
                median_max_drawdown=float(np.median(drawdowns)),
                worst_max_drawdown=float(drawdowns.min()),
                insolvent_paths=insolvent,
                unreconciled_paths=unreconciled,
                no_trade_paths=no_trade,
                replicates=grid.replicates,
            )
        )

    fragile = [item.kind for item in outcomes if item.percentile_5 <= 0.0 < baseline.net_return]
    return {
        "grid": grid.to_dict(),
        "baseline": baseline.to_dict(),
        "outcomes": [item.to_dict() for item in outcomes],
        "retained_paths": [item.to_dict() for item in paths],
        "kinds_whose_downside_crosses_zero": fragile,
        "total_insolvent_paths": sum(item.insolvent_paths for item in outcomes),
        "total_unreconciled_paths": sum(item.unreconciled_paths for item in outcomes),
        "reporting_rule": (
            "insolvent, no-trade, and unreconciled paths are counted rather than "
            "dropped; a study that discards its failures reports the distribution "
            "conditional on survival"
        ),
        "simulation_only": True,
    }


def replay_path(
    returns: pd.Series,
    costs: pd.Series,
    grid: FrozenPerturbationGrid,
    *,
    kind: str,
    replicate: int,
) -> PathOutcome:
    """Reproduce one recorded path exactly from its seed coordinates.

    The point of derived seed streams: any single path in a 2,000-replicate
    study can be recovered on its own, without re-running the study or knowing
    what any other kind drew.
    """
    spec = next((item for item in grid.specs if item.kind == kind), None)
    if spec is None:
        raise PerturbationError(f"grid {grid.name!r} declares no {kind!r} perturbation")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise PerturbationError("replicate must be a non-negative int")
    if replicate >= grid.replicates:
        raise PerturbationError(
            f"replicate {replicate} is outside the frozen study of {grid.replicates} paths"
        )
    generator = grid.seed_stream(kind, replicate)
    shifted_returns, shifted_costs = perturb_path(returns, costs, spec, generator)
    return evaluate_path(shifted_returns, shifted_costs, replicate=replicate, kind=kind)


def assert_perturbation_streams_isolated(grid: FrozenPerturbationGrid, *, draws: int = 8) -> None:
    """Refuse a grid whose perturbation kinds share a random stream.

    Two kinds drawing identical numbers would make their outcomes correlated
    artefacts of the seeding rather than independent stresses.

    Raises:
        PerturbationError: If any two kinds produce the same draws.
    """
    seen: dict[tuple[float, ...], str] = {}
    for spec in grid.specs:
        signature = tuple(grid.seed_stream(spec.kind, 0).random(draws).tolist())
        if signature in seen:
            raise PerturbationError(
                f"perturbation kinds {seen[signature]!r} and {spec.kind!r} share a "
                "random stream; their outcomes would be seeding artefacts"
            )
        seen[signature] = spec.kind


def failure_report(study: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize a study into a structured, machine-readable failure analysis.

    Ordered worst-first by 5th-percentile outcome, so the reader meets the
    strategy's weakest mechanic before its strongest.
    """
    outcomes = list(study.get("outcomes", ()))
    if not outcomes:
        raise PerturbationError("study contains no outcomes to summarize")
    ranked = sorted(outcomes, key=lambda item: (item["percentile_5"], item["kind"]))
    baseline = float(study["baseline"]["net_return"])
    return {
        "grid_identity": study["grid"]["identity"],
        "baseline_net_return": baseline,
        "most_damaging_kinds": [item["kind"] for item in ranked[:3]],
        "worst_case_net_return": ranked[0]["percentile_5"],
        "kinds_with_insolvency": [item["kind"] for item in outcomes if item["insolvent_paths"] > 0],
        "kinds_with_unreconciled_paths": [
            item["kind"] for item in outcomes if item["unreconciled_paths"] > 0
        ],
        "median_degradation": {
            item["kind"]: item["median_net_return"] - baseline for item in outcomes
        },
        "worst_drawdown_by_kind": {item["kind"]: item["worst_max_drawdown"] for item in outcomes},
        "deepest_drawdown_kind": min(
            outcomes, key=lambda item: (item["worst_max_drawdown"], item["kind"])
        )["kind"],
        "caveat": (
            "Monte Carlo frequency under a declared perturbation family is not a "
            "probability of loss in the market."
        ),
        "simulation_only": True,
    }


__all__ = [
    "MAX_MAGNITUDE",
    "MAX_REPLICATES",
    "MIN_REPLICATES",
    "PERTURBATION_KINDS",
    "FrozenPerturbationGrid",
    "InsolventPathError",
    "PathOutcome",
    "PerturbationError",
    "PerturbationOutcome",
    "PerturbationSpec",
    "assert_perturbation_streams_isolated",
    "evaluate_path",
    "failure_report",
    "perturb_path",
    "replay_path",
    "run_perturbation_study",
    "standard_execution_grid",
    "verify_frozen_perturbations",
]
