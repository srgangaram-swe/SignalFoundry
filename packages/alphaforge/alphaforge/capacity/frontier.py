"""Capacity frontier by complete simulation rerun (SF-S4-MR5).

The frontier answers one question: how does the strategy behave as the capital
it manages grows? The only honest way to answer it is to **rerun the entire
simulation** at each AUM — recomputing targets, orders, authorizations, fills,
costs, and positions — because capacity effects are path-dependent. A larger book
hits participation limits, which changes what fills, which changes the next
session's position, which changes the next target.

Scaling a completed return series by a capital ratio produces a smooth,
attractive, and entirely fictitious frontier: it assumes the same trades happened
at every size, which is precisely the assumption capacity analysis exists to
test. This module therefore never scales. :func:`capacity_frontier` calls the
supplied simulation once per scenario, from scratch.

Scenarios are **candidate-order isolated**: each runs against its own policy
instance and its own ledgers, and the results are sorted by AUM afterwards. A
run whose result depended on which scenario preceded it would be reporting
contamination, and the ordering test asserts it does not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alphaforge.capacity.contracts import (
    MAX_RECORDS,
    CapacityContractError,
    CapacityPolicyDeclaration,
    finite_quantity,
)

#: Refusal threshold, not a tuning knob. Each scenario is a full rerun, so an
#: unbounded grid is a configuration mistake rather than a long wait.
MAX_SCENARIOS = 24


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """One complete rerun at a frozen AUM.

    Every field is recomputed from that scenario's own simulation. ``feasible``
    is ``False`` when the run could not complete under the declared policy — a
    halted or infeasible scenario stays in the frontier as a refusal rather than
    being dropped, because dropping the hard scenarios is how a capacity curve
    acquires an optimistic tail.
    """

    aum: float
    feasible: bool
    desired_notional: float
    filled_notional: float
    shortfall_notional: float
    fill_ratio: float
    participation_utilization: float
    book_budget_utilization: float
    locate_rejections: int
    forced_buy_ins: int
    turnover: float
    gross_return: float
    net_return: float
    costs: float
    max_drawdown: float
    max_concentration: float
    reason: str = ""
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row for the frontier evidence."""
        return {
            "aum": self.aum,
            "feasible": self.feasible,
            "desired_notional": self.desired_notional,
            "filled_notional": self.filled_notional,
            "shortfall_notional": self.shortfall_notional,
            "fill_ratio": self.fill_ratio,
            "participation_utilization": self.participation_utilization,
            "book_budget_utilization": self.book_budget_utilization,
            "locate_rejections": self.locate_rejections,
            "forced_buy_ins": self.forced_buy_ins,
            "turnover": self.turnover,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "costs": self.costs,
            "max_drawdown": self.max_drawdown,
            "max_concentration": self.max_concentration,
            "reason": self.reason,
        }


SimulationFn = Callable[[float, CapacityPolicyDeclaration], ScenarioResult]


def validate_scenarios(aum_levels: Sequence[float]) -> tuple[float, ...]:
    """Return a strictly increasing, bounded, finite AUM grid.

    Strict ordering is required rather than merely sorted: a repeated level
    would produce two identical rows that look like corroboration, and the
    frontier's whole value is in the *differences* between levels.
    """
    if not isinstance(aum_levels, (list, tuple)):
        raise CapacityContractError("aum_levels must be a list or tuple")
    if not aum_levels:
        raise CapacityContractError("at least one AUM scenario is required")
    if len(aum_levels) > MAX_SCENARIOS:
        raise CapacityContractError(
            f"aum_levels exceeds the {MAX_SCENARIOS}-scenario ceiling; each scenario is a "
            "complete rerun"
        )
    levels = tuple(finite_quantity(level, name="aum", maximum=1e15) for level in aum_levels)
    if any(level <= 0.0 for level in levels):
        raise CapacityContractError("every AUM scenario must be strictly positive")
    if any(later <= earlier for earlier, later in zip(levels, levels[1:], strict=False)):
        raise CapacityContractError("aum_levels must be strictly increasing")
    return levels


def capacity_frontier(
    simulate: SimulationFn,
    declaration: CapacityPolicyDeclaration,
    *,
    aum_levels: Sequence[float],
    reference_aum: float,
) -> pd.DataFrame:
    """Rerun the complete simulation at every frozen AUM and tabulate the result.

    Args:
        simulate: Runs one full simulation at a given AUM under the declaration
            and returns its recomputed :class:`ScenarioResult`. Called exactly
            once per scenario.
        declaration: The frozen policy, identical across scenarios so the only
            varying input is capital.
        aum_levels: Strictly increasing grid.
        reference_aum: The baseline level, which must appear in the grid so the
            frontier always contains the run every other row is compared against.

    Returns:
        One row per scenario, sorted by AUM, with the reference row flagged.

    Raises:
        CapacityContractError: On a malformed grid, a missing reference level, or
            a simulation that returns a result for the wrong AUM.
    """
    levels = validate_scenarios(aum_levels)
    reference = finite_quantity(reference_aum, name="reference_aum", maximum=1e15)
    if not any(abs(level - reference) <= 1e-9 for level in levels):
        raise CapacityContractError(
            f"reference AUM {reference} is absent from the scenario grid; the frontier "
            "must contain the run its comparisons are anchored to"
        )

    rows: list[dict[str, Any]] = []
    for level in levels:
        result = simulate(level, declaration)
        if not isinstance(result, ScenarioResult):
            raise CapacityContractError("simulate must return a ScenarioResult")
        if abs(result.aum - level) > 1e-9:
            raise CapacityContractError(
                f"simulation returned AUM {result.aum} for scenario {level}; a scenario "
                "must report the capital it actually ran"
            )
        row = result.to_dict()
        row["is_reference"] = bool(abs(level - reference) <= 1e-9)
        row["policy_digest"] = declaration.digest
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("aum").reset_index(drop=True)
    return frame


def frontier_summary(frontier: pd.DataFrame) -> dict[str, Any]:
    """Reduce a frontier to the aggregate claims it supports, and no others.

    Deliberately reports the level at which feasibility or fill ratio first
    degrades rather than a "capacity" number. **No output of this function is a
    deployable AUM, a broker capacity, paper-trading readiness, or an expected
    profit**, and the returned record says so explicitly so the caveat travels
    with the data rather than living only in prose.
    """
    if frontier.empty:
        raise CapacityContractError("cannot summarize an empty frontier")
    required = {"aum", "feasible", "fill_ratio", "net_return"}
    missing = sorted(required - set(frontier.columns))
    if missing:
        raise CapacityContractError(f"frontier is missing required columns: {missing}")

    feasible = frontier.loc[frontier["feasible"]]
    infeasible = frontier.loc[~frontier["feasible"]]
    first_infeasible = float(infeasible["aum"].min()) if not infeasible.empty else None

    degraded = frontier.loc[frontier["fill_ratio"] < 0.99]
    first_degraded = float(degraded["aum"].min()) if not degraded.empty else None

    return {
        "scenarios": int(len(frontier)),
        "feasible_scenarios": int(len(feasible)),
        "max_feasible_aum": float(feasible["aum"].max()) if not feasible.empty else None,
        "first_infeasible_aum": first_infeasible,
        "first_degraded_fill_aum": first_degraded,
        "reference_net_return": (
            float(frontier.loc[frontier["is_reference"], "net_return"].iloc[0])
            if "is_reference" in frontier.columns and frontier["is_reference"].any()
            else None
        ),
        "interpretation": (
            "Simulated capacity under frozen synthetic assumptions. These figures are "
            "NOT a deployable AUM, broker capacity, paper-trading readiness, or expected "
            "profit; they describe only where this simulation's declared constraints "
            "begin to bind."
        ),
    }


def verify_row_aggregation(frontier: pd.DataFrame, scenario_rows: Sequence[ScenarioResult]) -> None:
    """Check that per-scenario rows reproduce the frontier exactly.

    The issue requires row-level evidence to aggregate to every reported point.
    Recomputing the table from the raw results and comparing is a cheap, direct
    check that nothing was smoothed, rescaled, or reordered between them.

    Raises:
        CapacityContractError: On any mismatch.
    """
    if len(frontier) != len(scenario_rows):
        raise CapacityContractError(
            f"frontier has {len(frontier)} rows against {len(scenario_rows)} scenario results"
        )
    expected = sorted(scenario_rows, key=lambda item: item.aum)
    for position, result in enumerate(expected):
        row = frontier.iloc[position]
        for field_name, value in result.to_dict().items():
            observed = row[field_name]
            if isinstance(value, float):
                if abs(float(observed) - value) > 1e-9:
                    raise CapacityContractError(
                        f"frontier row {position} field {field_name!r} is {observed}, "
                        f"expected {value}"
                    )
            elif observed != value:
                raise CapacityContractError(
                    f"frontier row {position} field {field_name!r} is {observed!r}, "
                    f"expected {value!r}"
                )


def bounded_scenario_count(count: int) -> int:
    """Validate a requested scenario count against the declared ceiling."""
    if isinstance(count, bool) or not isinstance(count, int):
        raise CapacityContractError("scenario count must be an int")
    if not 1 <= count <= MAX_SCENARIOS:
        raise CapacityContractError(f"scenario count must be in [1, {MAX_SCENARIOS}]")
    if count > MAX_RECORDS:  # pragma: no cover - defensive, MAX_SCENARIOS is far smaller
        raise CapacityContractError("scenario count exceeds the record ceiling")
    return count
