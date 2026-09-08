"""Frozen negative controls under a leakage-safe temporal protocol (SF-S4-MR6).

A backtest result means nothing until you know what the same pipeline produces on
data that provably contains no signal. These controls generate that null.

Three kinds, because they fail differently and a study running only one has not
established much:

* **Feature permutation** shuffles the feature/label correspondence while leaving
  both marginal distributions intact. It asks whether the model used the
  *pairing* or merely the shapes.
* **Randomized labels** destroy the signal outright. Repeated, they give the null
  a *distribution*, which is what a p-value requires — a single control run is a
  point and supports nothing.
* **Representation placebo** substitutes a structurally similar but
  information-free representation, asking whether the representation earned its
  place or the surrounding pipeline did the work.

Two invariants make these controls trustworthy, and both are enforced
structurally rather than by convention:

**Controls never see the holdout.** :func:`permute_within_folds` and friends take
only training-fold data; there is no parameter through which holdout outcomes
could arrive. A control fit on the holdout would be calibrating the null against
the very data the candidate is judged on.

**Controls never share mutable random state with candidates.** Every draw comes
from a grid-derived named stream, so a control's randomness is a pure function of
``(root_seed, name, replicate)``. Advancing a shared generator would make results
depend on execution order, which is exactly the contamination these controls
exist to rule out.

Permutation is applied **within folds**, never across them. Shuffling globally
would move observations across the temporal boundary and leak future rows into a
training fold — producing a null that is *easier* than reality and a candidate
that clears it too easily.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphaforge.robustness.grid import (
    CONTROL_KINDS,
    ControlKind,
    NegativeControl,
    RobustnessGrid,
    RobustnessGridError,
)

FloatArray = NDArray[np.float64]

#: Refusal threshold. A null estimated from fewer rows than this is dominated by
#: its own sampling error and would make any comparison meaningless.
MIN_CONTROL_OBSERVATIONS = 30


class NegativeControlError(ValueError):
    """Raised when a control request would be invalid or leak information."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One contiguous training fold, identified by positional bounds."""

    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise NegativeControlError(f"fold {self.name!r} is empty or inverted")
        if self.start < 0:
            raise NegativeControlError(f"fold {self.name!r} starts before the series")

    @property
    def size(self) -> int:
        """Observations in this fold."""
        return self.stop - self.start


def contiguous_folds(n_observations: int, *, n_folds: int) -> tuple[Fold, ...]:
    """Split ``n_observations`` into contiguous, ordered folds.

    Contiguous and ordered rather than shuffled: a random split of a time series
    surrounds each fold with its own future, which is the standard way a control
    protocol quietly stops being leakage-safe.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 1:
        raise NegativeControlError("n_folds must be a positive int")
    if n_observations < n_folds:
        raise NegativeControlError("fewer observations than requested folds")
    bounds = np.array_split(np.arange(n_observations), n_folds)
    return tuple(
        Fold(name=f"fold_{index + 1}", start=int(block[0]), stop=int(block[-1]) + 1)
        for index, block in enumerate(bounds)
        if block.size
    )


def _validated_frame(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise NegativeControlError("features must be a pandas DataFrame")
    if features.empty:
        raise NegativeControlError("features must be non-empty")
    if len(features) < MIN_CONTROL_OBSERVATIONS:
        raise NegativeControlError(
            f"a null needs at least {MIN_CONTROL_OBSERVATIONS} observations; "
            f"got {len(features)}"
        )
    if features.columns.has_duplicates:
        raise NegativeControlError("feature columns must be unique")
    return features


def permute_within_folds(
    features: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    generator: np.random.Generator,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Shuffle feature rows **within** each fold, never across them.

    Breaks the feature/label correspondence while preserving each fold's marginal
    distributions and its temporal boundary. Shuffling globally would relocate
    observations across folds and leak future rows into training — producing a
    null that is easier than reality.

    Args:
        columns: Restrict permutation to these columns, which is how a *family*
            is challenged while the rest of the book is left intact. ``None``
            permutes every column.
    """
    frame = _validated_frame(features)
    if not folds:
        raise NegativeControlError("at least one fold is required")
    targets = list(frame.columns) if columns is None else list(columns)
    missing = [column for column in targets if column not in frame.columns]
    if missing:
        raise NegativeControlError(f"unknown columns for permutation: {missing}")

    permuted = frame.copy()
    # `.to_numpy()` may return a read-only view over the frame's block; copy so
    # the shuffle writes into memory this function owns.
    values = np.array(permuted[targets].to_numpy(), copy=True)
    for fold in folds:
        if fold.stop > len(frame):
            raise NegativeControlError(f"fold {fold.name!r} extends past the supplied observations")
        block = values[fold.start : fold.stop]
        order = generator.permutation(len(block))
        values[fold.start : fold.stop] = block[order]
    permuted[targets] = values
    return permuted


def randomize_labels_within_folds(
    labels: pd.Series, folds: Sequence[Fold], *, generator: np.random.Generator
) -> pd.Series:
    """Shuffle labels within each fold, destroying signal but not structure.

    Preserves each fold's label distribution — its mean, variance, and any
    class imbalance — so the resulting null differs from the candidate in
    exactly one respect: whether the labels line up with the features.
    """
    if not isinstance(labels, pd.Series):
        raise NegativeControlError("labels must be a pandas Series")
    if len(labels) < MIN_CONTROL_OBSERVATIONS:
        raise NegativeControlError(
            f"a null needs at least {MIN_CONTROL_OBSERVATIONS} observations; got {len(labels)}"
        )
    if not folds:
        raise NegativeControlError("at least one fold is required")
    values = labels.to_numpy().copy()
    for fold in folds:
        if fold.stop > len(labels):
            raise NegativeControlError(f"fold {fold.name!r} extends past the supplied observations")
        block = values[fold.start : fold.stop]
        values[fold.start : fold.stop] = block[generator.permutation(len(block))]
    return pd.Series(values, index=labels.index, name=labels.name)


def representation_placebo(
    features: pd.DataFrame, *, generator: np.random.Generator, columns: Sequence[str]
) -> pd.DataFrame:
    """Replace named columns with information-free surrogates of matched shape.

    Each surrogate is Gaussian noise scaled to the column's own mean and standard
    deviation, so it matches the original's first two moments and its position in
    the pipeline while carrying no relationship to anything.

    This is the control that isolates the *representation* from the machinery
    around it. If a candidate's advantage survives replacing its representation
    with matched noise, the advantage came from the surrounding pipeline — the
    labels, the portfolio construction, the cost model — and not from the
    representation it was credited to.
    """
    frame = _validated_frame(features)
    targets = list(columns)
    if not targets:
        raise NegativeControlError("a representation placebo must name at least one column")
    missing = [column for column in targets if column not in frame.columns]
    if missing:
        raise NegativeControlError(f"unknown columns for placebo: {missing}")

    placebo = frame.copy()
    for column in targets:
        original = frame[column].to_numpy(dtype=float)
        finite = original[np.isfinite(original)]
        if finite.size == 0:
            raise NegativeControlError(
                f"column {column!r} has no finite values to match a placebo against"
            )
        centre = float(np.mean(finite))
        spread = float(np.std(finite))
        # A degenerate column has no scale to match; a constant surrogate is the
        # honest substitute rather than inventing variance the original lacked.
        draw = (
            generator.normal(centre, spread, len(frame))
            if spread > 0.0
            else np.full(len(frame), centre)
        )
        placebo[column] = draw
    return placebo


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    """The null distribution produced by one frozen control."""

    control: str
    kind: ControlKind
    challenges: str
    replicate_metrics: tuple[float, ...]
    candidate_metric: float
    higher_is_better: bool

    def __post_init__(self) -> None:
        if len(self.replicate_metrics) < 2:
            raise NegativeControlError(
                "a null distribution needs at least two replicates; one draw is a point"
            )
        if not all(np.isfinite(self.replicate_metrics)):
            raise NegativeControlError("control replicate metrics must be finite")
        if not np.isfinite(self.candidate_metric):
            raise NegativeControlError("candidate metric must be finite")

    @property
    def null_mean(self) -> float:
        """Mean of the null distribution."""
        return float(np.mean(self.replicate_metrics))

    @property
    def null_std(self) -> float:
        """Standard deviation of the null distribution."""
        return float(np.std(self.replicate_metrics, ddof=1))

    @property
    def exceedances(self) -> int:
        """Replicates at least as extreme as the candidate."""
        values = np.asarray(self.replicate_metrics, dtype=float)
        if self.higher_is_better:
            return int(np.sum(values >= self.candidate_metric))
        return int(np.sum(values <= self.candidate_metric))

    @property
    def p_value(self) -> float:
        """Empirical one-sided p-value with the conservative ``+1`` correction.

        ``(exceedances + 1) / (replicates + 1)`` rather than
        ``exceedances / replicates``: the latter can report exactly zero, which
        claims more certainty than a finite number of draws can support. The
        correction also keeps the value strictly positive so the multiple-testing
        correction downstream is well defined.
        """
        return float((self.exceedances + 1) / (len(self.replicate_metrics) + 1))

    def exceeds_null(self, alpha: float = 0.05) -> bool:
        """Whether the candidate beats its own null at ``alpha``.

        Uncorrected. Family-wise correction is applied by the governed ledger
        over the complete family, and reading this flag as a decision would
        bypass it.
        """
        if not 0.0 < alpha < 1.0:
            raise NegativeControlError("alpha must lie in (0, 1)")
        return self.p_value <= alpha

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly evidence row."""
        return {
            "control": self.control,
            "kind": self.kind,
            "challenges": self.challenges,
            "replicates": len(self.replicate_metrics),
            "candidate_metric": self.candidate_metric,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "exceedances": self.exceedances,
            "p_value": self.p_value,
            "uncorrected_exceeds_null_at_5pct": self.exceeds_null(0.05),
            "note": (
                "empirical one-sided p-value with the conservative +1 correction; "
                "family-wise correction is applied separately over the complete family"
            ),
        }


def run_control(
    control: NegativeControl,
    grid: RobustnessGrid,
    *,
    candidate_metric: float,
    evaluate: Any,
) -> ControlOutcome:
    """Draw one control's null distribution using isolated named seed streams.

    ``evaluate`` receives ``(kind, generator, replicate)`` and returns that
    replicate's metric. It is supplied by the caller because the control layer
    deliberately knows nothing about models, portfolios, or costs — that
    separation is what stops a control from accidentally acquiring access to the
    holdout through a shared object.

    Each replicate draws from ``seed_stream(control.name, replicate)``, so a
    replicate is reproducible independently of the order the study runs.
    """
    if control.kind not in CONTROL_KINDS:
        raise NegativeControlError(f"unsupported control kind {control.kind!r}")
    metrics: list[float] = []
    for replicate in range(control.replicates):
        generator = grid.seed_stream(control.name, replicate)
        value = evaluate(control.kind, generator, replicate)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise NegativeControlError(
                f"control {control.name!r} replicate {replicate} returned "
                f"{type(value).__name__}, expected a real metric"
            )
        if not np.isfinite(value):
            raise NegativeControlError(
                f"control {control.name!r} replicate {replicate} returned a non-finite metric"
            )
        metrics.append(float(value))
    return ControlOutcome(
        control=control.name,
        kind=control.kind,
        challenges=control.challenges,
        replicate_metrics=tuple(metrics),
        candidate_metric=float(candidate_metric),
        higher_is_better=grid.higher_is_better,
    )


def assert_streams_isolated(grid: RobustnessGrid, names: Sequence[str], draws: int = 8) -> None:
    """Assert that named streams produce independent, order-independent draws.

    Two distinct names must never yield identical sequences — that would mean a
    candidate and its own null control were drawing the same randomness, and the
    control would be measuring the candidate rather than the null.

    Raises:
        RobustnessGridError: If any two named streams collide.
    """
    observed: dict[str, tuple[float, ...]] = {}
    for name in names:
        sequence = tuple(float(value) for value in grid.seed_stream(name, 0).random(draws))
        for other, existing in observed.items():
            if sequence == existing:
                raise RobustnessGridError(
                    f"seed streams {name!r} and {other!r} produce identical draws; "
                    "a control sharing a candidate's randomness measures the candidate"
                )
        observed[name] = sequence
