"""Stable regions, sensitivity cliffs, and family redundancy (SF-S4-MR6).

The issue's non-goal is blunt: do not report only the best parameter point. This
module exists to replace that number with the things that actually predict
out-of-sample survival.

The argmax of a noisy sweep is the least trustworthy point in it. It is where
skill and luck happen to align, and by construction it is the point most likely
to regress. A parameter setting whose neighbours also work is evidence of a real
effect; a spike surrounded by mediocrity is evidence of overfitting, and the two
are indistinguishable from the maximum alone.

Three analyses, each answering a question the maximum cannot:

* :func:`stable_regions` — where is performance *broadly* acceptable? A wide
  plateau tolerates the parameter drift that live trading guarantees.
* :func:`sensitivity_cliffs` — where does a single step off a value collapse the
  result? Operating next to a cliff means a small mis-specification is
  catastrophic rather than merely costly.
* :func:`family_redundancy` — which feature families can be removed without
  measurable loss? A family whose ablation costs nothing is complexity being
  carried for free, and every carried family is another thing to break.

Adjacency is defined by the **declared axis order**, which is why
:class:`~alphaforge.robustness.grid.RobustnessGrid` preserves it: "next to" is
only meaningful along the ordering the study declared.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.robustness.grid import RobustnessGrid, RobustnessGridError

#: Refusal threshold on the reported region count, so a degenerate sweep cannot
#: emit an unbounded report.
MAX_REPORTED_REGIONS = 256


class StabilityError(ValueError):
    """Raised when a stability request or its inputs are unusable."""


def _global_span(frame: pd.DataFrame) -> float:
    """Return the metric range over the whole grid.

    Every relative threshold in this module is measured against this, so a
    "20% tolerance" or a "35% cliff" always means the same fraction of the
    performance range the study actually spans — not a fraction of whatever
    noise happens to appear along one axis.
    """
    values = frame["metric"].to_numpy(dtype=float)
    return float(values.max() - values.min())


def _metric_frame(grid: RobustnessGrid, metrics: Mapping[str, float]) -> pd.DataFrame:
    """Return one validated row per grid point with its metric."""
    if not isinstance(metrics, Mapping) or not metrics:
        raise StabilityError("metrics must be a non-empty mapping of point_id to value")
    expected = {point.point_id for point in grid.points}
    observed = set(metrics)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise StabilityError(
            f"metrics must cover exactly the frozen grid; missing={missing[:8]}, "
            f"extra={extra[:8]}. A partial sweep cannot support a stability claim."
        )
    rows: list[dict[str, Any]] = []
    for point in grid.points:
        value = metrics[point.point_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StabilityError(f"metric for {point.point_id} must be a real number")
        if not math.isfinite(value):
            raise StabilityError(
                f"metric for {point.point_id} is non-finite; a failed trial must be "
                "recorded as a failure rather than smuggled in as a number"
            )
        rows.append({"point_id": point.point_id, **dict(point.parameters), "metric": float(value)})
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class StableRegion:
    """A contiguous run of acceptable settings along one axis."""

    axis: str
    values: tuple[Any, ...]
    size: int
    worst_metric: float
    best_metric: float
    median_metric: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly evidence row."""
        return {
            "axis": self.axis,
            "values": list(self.values),
            "size": self.size,
            "worst_metric": self.worst_metric,
            "best_metric": self.best_metric,
            "median_metric": self.median_metric,
        }


def stable_regions(
    grid: RobustnessGrid,
    metrics: Mapping[str, float],
    *,
    tolerance: float = 0.20,
    min_size: int = 2,
) -> tuple[StableRegion, ...]:
    """Find contiguous axis runs whose every setting stays near the best.

    For each axis the metric is aggregated across the other axes by **median**,
    not mean: a single catastrophic combination should not disqualify an
    otherwise sound setting, and a single spectacular one should not rescue a bad
    one. An axis value is acceptable when its aggregate sits within ``tolerance``
    of the best aggregate on that axis, measured relative to the observed spread
    so the threshold does not depend on the metric's units.

    Args:
        tolerance: Fraction of the axis's observed metric range a value may fall
            below the best and still count as acceptable.
        min_size: Shortest run reported. Defaults to 2 because a "region" of one
            point is the spike this analysis exists to distinguish from a plateau.

    Returns:
        Regions ordered by axis and then by position, longest first within an
        axis, so the broadest plateau is the first thing read.
    """
    if not 0.0 < tolerance < 1.0:
        raise StabilityError("tolerance must lie in (0, 1)")
    if isinstance(min_size, bool) or not isinstance(min_size, int) or min_size < 1:
        raise StabilityError("min_size must be a positive int")
    frame = _metric_frame(grid, metrics)
    # Scale by the *global* metric range, never by the axis's own aggregate
    # spread. An axis with no effect has an aggregate spread made entirely of
    # noise, and a tolerance relative to that denominator would judge it on a
    # scale thousands of times finer than the one the study actually operates on.
    span = _global_span(frame)

    regions: list[StableRegion] = []
    for axis, values in grid.axes.items():
        aggregate = frame.groupby(axis)["metric"].median()
        ordered = [aggregate.get(value, float("nan")) for value in values]
        if not np.isfinite(ordered).any():
            continue
        finite = np.asarray([item for item in ordered if np.isfinite(item)], dtype=float)
        if grid.higher_is_better:
            threshold = float(finite.max()) - tolerance * span
            acceptable = [np.isfinite(item) and item >= threshold for item in ordered]
        else:
            threshold = float(finite.min()) + tolerance * span
            acceptable = [np.isfinite(item) and item <= threshold for item in ordered]

        start: int | None = None
        for index in range(len(values) + 1):
            inside = index < len(values) and acceptable[index]
            if inside and start is None:
                start = index
            elif not inside and start is not None:
                run = list(range(start, index))
                if len(run) >= min_size:
                    block = np.asarray([ordered[position] for position in run], dtype=float)
                    regions.append(
                        StableRegion(
                            axis=axis,
                            values=tuple(values[position] for position in run),
                            size=len(run),
                            worst_metric=float(block.min()),
                            best_metric=float(block.max()),
                            median_metric=float(np.median(block)),
                        )
                    )
                start = None
        if len(regions) > MAX_REPORTED_REGIONS:
            raise StabilityError(
                f"stability analysis produced more than {MAX_REPORTED_REGIONS} regions; "
                "the sweep is too fine to summarize meaningfully"
            )
    return tuple(sorted(regions, key=lambda item: (item.axis, -item.size, item.values[0])))


@dataclass(frozen=True, slots=True)
class SensitivityCliff:
    """A large metric change between two adjacent axis settings."""

    axis: str
    from_value: Any
    to_value: Any
    from_metric: float
    to_metric: float
    drop: float
    relative_drop: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly evidence row."""
        return {
            "axis": self.axis,
            "from_value": self.from_value,
            "to_value": self.to_value,
            "from_metric": self.from_metric,
            "to_metric": self.to_metric,
            "drop": self.drop,
            "relative_drop": self.relative_drop,
        }


def sensitivity_cliffs(
    grid: RobustnessGrid,
    metrics: Mapping[str, float],
    *,
    threshold: float = 0.35,
    consistency: float = 0.6,
) -> tuple[SensitivityCliff, ...]:
    """Find adjacent settings between which the metric consistently collapses.

    A cliff is a step where the aggregate falls by more than ``threshold`` of the
    axis's observed range **and** the same deterioration holds across at least
    ``consistency`` of the other-axis combinations. Operating next to a real
    cliff is materially different from operating on a plateau at the same metric:
    the plateau tolerates mis-specification and drift, the cliff edge does not,
    and the point estimate is identical in both cases.

    The consistency requirement exists because a median aggregate over a
    multi-modal surface can jump between modes as the count crosses a boundary,
    manufacturing an apparent cliff on an axis that has no effect at all. Such an
    artefact does not survive being checked slice by slice, and reporting it
    would send an operator to defend a parameter that never mattered.

    Only *deteriorating* steps are reported. A sharp improvement is not a risk to
    the operator standing on the good side of it.
    """
    if not 0.0 < threshold < 1.0:
        raise StabilityError("threshold must lie in (0, 1)")
    if not 0.0 < consistency <= 1.0:
        raise StabilityError("consistency must lie in (0, 1]")
    frame = _metric_frame(grid, metrics)
    other_axes = list(grid.axes)
    # The global metric range, for the reason given in `stable_regions`: dividing
    # a noise-sized drop by a noise-sized per-axis denominator manufactures
    # cliffs on axes that have no effect at all.
    span = _global_span(frame)
    if span <= 0.0:
        return ()

    cliffs: list[SensitivityCliff] = []
    for axis, values in grid.axes.items():
        aggregate = frame.groupby(axis)["metric"].median()
        ordered = [aggregate.get(value, float("nan")) for value in values]
        finite = np.asarray([item for item in ordered if np.isfinite(item)], dtype=float)
        if finite.size < 2:
            continue
        conditioning = [item for item in other_axes if item != axis]
        for index in range(len(values) - 1):
            left, right = ordered[index], ordered[index + 1]
            if not (np.isfinite(left) and np.isfinite(right)):
                continue
            drop = (left - right) if grid.higher_is_better else (right - left)
            if drop / span < threshold:
                continue
            if not _consistently_deteriorates(
                frame,
                axis=axis,
                conditioning=conditioning,
                from_value=values[index],
                to_value=values[index + 1],
                higher_is_better=grid.higher_is_better,
                required=consistency,
            ):
                continue
            cliffs.append(
                SensitivityCliff(
                    axis=axis,
                    from_value=values[index],
                    to_value=values[index + 1],
                    from_metric=float(left),
                    to_metric=float(right),
                    drop=float(drop),
                    relative_drop=float(drop / span),
                )
            )
    return tuple(sorted(cliffs, key=lambda item: (-item.relative_drop, item.axis)))


def _consistently_deteriorates(
    frame: pd.DataFrame,
    *,
    axis: str,
    conditioning: Sequence[str],
    from_value: Any,
    to_value: Any,
    higher_is_better: bool,
    required: float,
) -> bool:
    """Whether the step degrades in enough held-out slices to be believed.

    With no other axes to condition on there is only one slice, and the aggregate
    comparison is already exact.
    """
    if not conditioning:
        return True
    left_rows = frame.loc[frame[axis] == from_value]
    right_rows = frame.loc[frame[axis] == to_value]
    if left_rows.empty or right_rows.empty:
        return False
    keys = list(conditioning)
    left = left_rows.set_index(keys)["metric"]
    right = right_rows.set_index(keys)["metric"]
    shared = left.index.intersection(right.index)
    if len(shared) == 0:
        return False
    deltas = left.loc[shared].to_numpy() - right.loc[shared].to_numpy()
    worse = deltas > 0.0 if higher_is_better else deltas < 0.0
    return bool(np.mean(worse) >= required)


@dataclass(frozen=True, slots=True)
class FamilyContribution:
    """What removing one feature family costs."""

    family: str
    full_metric: float
    ablated_metric: float
    contribution: float
    relative_contribution: float
    redundant: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly evidence row."""
        return {
            "family": self.family,
            "full_metric": self.full_metric,
            "ablated_metric": self.ablated_metric,
            "contribution": self.contribution,
            "relative_contribution": self.relative_contribution,
            "redundant": self.redundant,
        }


def family_redundancy(
    grid: RobustnessGrid,
    ablation_metrics: Mapping[str, float],
    *,
    redundancy_tolerance: float = 0.05,
) -> tuple[FamilyContribution, ...]:
    """Measure each family's marginal contribution by leave-one-out ablation.

    ``ablation_metrics`` must contain the ``full`` arm and one ``drop_<family>``
    arm per declared family — the complete set produced by
    :meth:`RobustnessGrid.ablation_points`. A partial set is refused, because
    reporting only the families somebody chose to ablate is selection by another
    name.

    A family is ``redundant`` when removing it changes the metric by less than
    ``redundancy_tolerance`` **relative to the full arm**. Redundancy is a claim
    about this grid and this metric only: a family that adds nothing on average
    may still matter in a regime this sweep did not contain, so the flag marks a
    candidate for removal rather than a decision to remove it.
    """
    if not 0.0 <= redundancy_tolerance < 1.0:
        raise StabilityError("redundancy_tolerance must lie in [0, 1)")
    if not grid.families:
        raise StabilityError("the grid declares no feature families to ablate")

    expected = {label for label, _ in grid.ablation_points()}
    observed = set(ablation_metrics)
    if observed != expected:
        raise StabilityError(
            f"ablation metrics must cover exactly {sorted(expected)}; got {sorted(observed)}. "
            "Reporting a subset of ablations is selection by another name."
        )
    for label, value in ablation_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StabilityError(f"ablation metric {label!r} must be a real number")
        if not math.isfinite(value):
            raise StabilityError(f"ablation metric {label!r} is non-finite")

    full = float(ablation_metrics["full"])
    scale = max(abs(full), 1e-12)
    contributions: list[FamilyContribution] = []
    for family in grid.families:
        ablated = float(ablation_metrics[f"drop_{family.name}"])
        contribution = (full - ablated) if grid.higher_is_better else (ablated - full)
        relative = contribution / scale
        contributions.append(
            FamilyContribution(
                family=family.name,
                full_metric=full,
                ablated_metric=ablated,
                contribution=contribution,
                relative_contribution=relative,
                redundant=bool(abs(relative) < redundancy_tolerance),
            )
        )
    return tuple(sorted(contributions, key=lambda item: (-item.contribution, item.family)))


def stability_report(
    grid: RobustnessGrid,
    metrics: Mapping[str, float],
    *,
    ablation_metrics: Mapping[str, float] | None = None,
    control_outcomes: Sequence[Any] = (),
    tolerance: float = 0.20,
    cliff_threshold: float = 0.35,
) -> dict[str, Any]:
    """Assemble the complete robustness record for the governed ledger.

    Deliberately reports the best point **alongside** its stability context and
    never on its own. The record states in its own text that the maximum is the
    least trustworthy point in a noisy sweep, so the caveat travels with the data
    rather than living only in a document somebody may not read.
    """
    frame = _metric_frame(grid, metrics)
    best_index = frame["metric"].idxmax() if grid.higher_is_better else frame["metric"].idxmin()
    best = frame.loc[best_index]
    regions = stable_regions(grid, metrics, tolerance=tolerance)
    cliffs = sensitivity_cliffs(grid, metrics, threshold=cliff_threshold)

    record: dict[str, Any] = {
        "study_id": grid.study_id,
        "grid_identity": grid.identity,
        "metric_name": grid.metric_name,
        "higher_is_better": grid.higher_is_better,
        "n_points": len(grid),
        "best_point": {
            "point_id": str(best["point_id"]),
            "metric": float(best["metric"]),
            "parameters": {axis: best[axis] for axis in grid.axes},
        },
        "metric_distribution": {
            "min": float(frame["metric"].min()),
            "median": float(frame["metric"].median()),
            "max": float(frame["metric"].max()),
            "std": float(frame["metric"].std(ddof=1)) if len(frame) > 1 else 0.0,
        },
        "stable_regions": [region.to_dict() for region in regions],
        "sensitivity_cliffs": [cliff.to_dict() for cliff in cliffs],
        "interpretation": (
            "The best point is the least trustworthy value in a noisy sweep: it is where "
            "skill and luck align and is the point most likely to regress. Read the stable "
            "regions and cliffs, not the maximum. A wide plateau tolerates the parameter "
            "drift live operation guarantees; a spike does not."
        ),
    }
    if ablation_metrics is not None:
        contributions = family_redundancy(grid, ablation_metrics)
        record["family_contributions"] = [item.to_dict() for item in contributions]
        record["redundant_families"] = [item.family for item in contributions if item.redundant]
    if control_outcomes:
        record["negative_controls"] = [outcome.to_dict() for outcome in control_outcomes]
        record["controls_all_exceeded"] = all(
            outcome.exceeds_null(0.05) for outcome in control_outcomes
        )
        record["control_note"] = (
            "Per-control p-values are uncorrected. Family-wise correction is applied by "
            "the governed ledger over the complete eligible family; a candidate that "
            "clears an individual control has not thereby cleared the family."
        )
    return record


def verify_rerun_determinism(first: Mapping[str, float], second: Mapping[str, float]) -> None:
    """Assert two runs of the same frozen grid produced identical metrics.

    Raises:
        RobustnessGridError: On any divergence, naming the first differing point.
            A study whose numbers move between runs cannot support a stability
            claim, because the instability being measured would be its own.
    """
    if set(first) != set(second):
        raise RobustnessGridError(
            "reruns covered different point sets; determinism cannot be established"
        )
    for point_id in sorted(first):
        left, right = float(first[point_id]), float(second[point_id])
        if left != right:
            raise RobustnessGridError(
                f"rerun diverged at {point_id}: {left!r} then {right!r}. A study whose "
                "own numbers move cannot measure anything else's stability."
            )
