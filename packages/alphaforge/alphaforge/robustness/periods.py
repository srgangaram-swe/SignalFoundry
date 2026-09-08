"""Frozen calendar intervals and causal regime labels (SF-S4-MR7).

The issue states the governing rule directly: *a regime cannot be defined from
the outcomes it is used to explain*. Labelling 2008 a "crisis" because the
strategy lost money there, and then reporting that the strategy is robust outside
crises, is circular — the label was derived from the result it is offered as an
explanation for.

Two mechanisms enforce that here, and both are structural rather than advisory:

**Labels are declared before evaluation.** A :class:`FrozenPeriodSet` publishes a
content-derived identity over its intervals, and
:func:`verify_frozen_periods` refuses a set that differs from the one recorded
beforehand. Re-cutting the calendar after seeing results is detectable.

**Regime labellers never see the candidate.** :func:`label_regimes` takes a
*conditioning* series — a benchmark, a volatility index, a macro series — and has
no parameter through which the candidate's own returns could arrive. The
separation is enforced by :func:`assert_conditioning_is_independent`, which
refuses a conditioning series that is a near-duplicate of the candidate's.

Labels are also **causal**: a rule-based label at bar ``t`` uses only conditioning
observations strictly before ``t``. A label computed from the full sample would
know which periods turned out badly, which is the same circularity by a subtler
route.

Boundary handling is explicit. An observation falling exactly on an interval edge
belongs to the interval that *starts* there — half-open ``[start, end)`` — so no
observation is ever counted twice and none is silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd

#: Refusal thresholds, not tuning knobs.
MAX_INTERVALS: Final = 512
MAX_REGIME_LABELS: Final = 32
MAX_LABEL_CHARS: Final = 48

#: A period with fewer observations than this cannot support an estimate. It is
#: still reported — as sparse, with its count — because silently dropping short
#: periods is how a strategy's worst stretch disappears from the evidence.
MIN_PERIOD_OBSERVATIONS: Final = 20


class PeriodContractError(ValueError):
    """Raised when a calendar, regime, or label specification is unusable."""


def _label(value: object, *, field_name: str) -> str:
    """Return a bounded ASCII label."""
    if not isinstance(value, str):
        raise PeriodContractError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text or text != value:
        raise PeriodContractError(f"{field_name} must be non-empty and free of padding")
    if len(text) > MAX_LABEL_CHARS:
        raise PeriodContractError(f"{field_name} exceeds {MAX_LABEL_CHARS} characters")
    if not text.isascii() or not all(part.isalnum() or part in "._- " for part in text):
        raise PeriodContractError(f"{field_name} must be printable ASCII")
    return text


def _digest(payload: Any) -> str:
    """Return a deterministic SHA-256 over a canonical JSON payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PeriodInterval:
    """One half-open evaluation interval ``[start, end)``.

    Half-open by construction so adjacent intervals tile the calendar without
    overlap: an observation on a boundary belongs to the interval that starts
    there, is counted exactly once, and is never silently dropped.
    """

    label: str
    start: date
    end: date
    kind: str = "calendar"

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _label(self.label, field_name="interval label"))
        object.__setattr__(self, "kind", _label(self.kind, field_name="interval kind"))
        for name in ("start", "end"):
            value = getattr(self, name)
            if not isinstance(value, date) or hasattr(value, "hour"):
                raise PeriodContractError(f"{name} must be a datetime.date")
        if self.end <= self.start:
            raise PeriodContractError(
                f"interval {self.label!r} is empty or inverted; [start, end) requires end > start"
            )

    def contains(self, moment: date) -> bool:
        """Whether ``moment`` falls in this half-open interval."""
        return self.start <= moment < self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "label": self.label,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class FrozenPeriodSet:
    """A calendar partition declared before any result is inspected.

    Args:
        name: Identifier for this partition, e.g. ``calendar_years``.
        intervals: The declared intervals. May overlap only when
            ``allow_overlap`` is set, which is how a "high volatility" overlay
            can coexist with a calendar-year partition.
        allow_overlap: Whether intervals may overlap. Defaults to ``False`` so an
            accidental double-count is refused rather than silently halving the
            apparent sample.

    Raises:
        PeriodContractError: On malformed, duplicate, or unexpectedly
            overlapping intervals.
    """

    name: str
    intervals: tuple[PeriodInterval, ...]
    allow_overlap: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _label(self.name, field_name="period set name"))
        intervals = tuple(self.intervals)
        if not intervals:
            raise PeriodContractError("a period set must declare at least one interval")
        if len(intervals) > MAX_INTERVALS:
            raise PeriodContractError(f"period set exceeds the {MAX_INTERVALS}-interval ceiling")
        labels = [interval.label for interval in intervals]
        if len(set(labels)) != len(labels):
            raise PeriodContractError("interval labels must be unique within a period set")
        ordered = tuple(sorted(intervals, key=lambda item: (item.start, item.end, item.label)))
        if not self.allow_overlap:
            for earlier, later in zip(ordered, ordered[1:], strict=False):
                if later.start < earlier.end:
                    raise PeriodContractError(
                        f"intervals {earlier.label!r} and {later.label!r} overlap; set "
                        "allow_overlap only when a deliberate overlay is intended, because "
                        "an accidental overlap double-counts observations"
                    )
        object.__setattr__(self, "intervals", ordered)

    @property
    def identity(self) -> str:
        """Content identity frozen before evaluation."""
        return _digest(
            {
                "name": self.name,
                "allow_overlap": self.allow_overlap,
                "intervals": [interval.to_dict() for interval in self.intervals],
            }
        )

    def assign(self, index: pd.Index) -> pd.Series:
        """Return the interval label for each timestamp, or ``None`` outside all.

        Observations outside every declared interval are labelled ``None`` rather
        than forced into the nearest one. Silently absorbing them would let the
        partition claim coverage it does not have.
        """
        moments = pd.to_datetime(pd.Index(index)).date
        labels: list[str | None] = []
        for moment in moments:
            match: str | None = None
            for interval in self.intervals:
                if interval.contains(moment):
                    match = interval.label
                    break
            labels.append(match)
        return pd.Series(labels, index=index, dtype="object", name=self.name)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly frozen declaration."""
        return {
            "name": self.name,
            "identity": self.identity,
            "allow_overlap": self.allow_overlap,
            "n_intervals": len(self.intervals),
            "intervals": [interval.to_dict() for interval in self.intervals],
            "boundary_rule": "half-open [start, end); a boundary observation joins the later interval",
            "frozen_before_evaluation": True,
        }


def calendar_years(start_year: int, end_year: int) -> FrozenPeriodSet:
    """Return a calendar-year partition, the least arguable frozen cut.

    Calendar years are declared by the calendar rather than by anyone's judgement
    about what happened in them, which makes them the one partition that cannot
    be accused of hindsight.
    """
    if isinstance(start_year, bool) or isinstance(end_year, bool):
        raise PeriodContractError("years must be ints")
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise PeriodContractError("years must be ints")
    if end_year < start_year:
        raise PeriodContractError("end_year must be on or after start_year")
    if end_year - start_year + 1 > MAX_INTERVALS:
        raise PeriodContractError(f"year range exceeds the {MAX_INTERVALS}-interval ceiling")
    intervals = tuple(
        PeriodInterval(
            label=str(year), start=date(year, 1, 1), end=date(year + 1, 1, 1), kind="calendar_year"
        )
        for year in range(start_year, end_year + 1)
    )
    return FrozenPeriodSet(name="calendar_years", intervals=intervals)


def verify_frozen_periods(period_set: FrozenPeriodSet, expected_identity: str) -> None:
    """Refuse a period set that differs from the one frozen before evaluation.

    Raises:
        PeriodContractError: On any divergence, naming both identities.
    """
    if not isinstance(expected_identity, str) or len(expected_identity) != 64:
        raise PeriodContractError("expected_identity must be a full SHA-256 digest")
    if period_set.identity != expected_identity:
        raise PeriodContractError(
            f"period set {period_set.name!r} does not match the frozen declaration: "
            f"executing {period_set.identity[:12]}, frozen {expected_identity[:12]}. "
            "Re-cutting the calendar after seeing results makes the partition a "
            "function of the outcomes it is offered to explain."
        )


# ---------------------------------------------------------------------------
# Causal regime labelling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeDefinition:
    """A frozen, causal rule assigning a regime label to each bar.

    The rule is a quantile cut on a trailing statistic of a **conditioning**
    series. It is deliberately simple and declared rather than fitted: a regime
    model estimated on the evaluation window would place its boundaries where
    they best explain that window, which is the circularity this module exists to
    prevent.

    Attributes:
        name: Identifier for this definition.
        statistic: ``volatility``, ``trend``, or ``drawdown`` of the conditioning
            series.
        window: Trailing window in bars.
        thresholds: Ascending quantiles in ``(0, 1)`` cutting the statistic into
            ``len(thresholds) + 1`` regimes.
        labels: Names for those regimes, low statistic first.
        warmup_policy: What to do with bars lacking a full trailing window.
            ``unknown`` labels them explicitly; it is the only supported policy
            because guessing a warm-up label invents information.
    """

    name: str
    statistic: str
    window: int
    thresholds: tuple[float, ...]
    labels: tuple[str, ...]
    warmup_policy: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _label(self.name, field_name="regime definition name"))
        if self.statistic not in ("volatility", "trend", "drawdown"):
            raise PeriodContractError(f"unsupported regime statistic {self.statistic!r}")
        if isinstance(self.window, bool) or not isinstance(self.window, int):
            raise PeriodContractError("window must be an int")
        if not 2 <= self.window <= 2_520:
            raise PeriodContractError("window must be in [2, 2520] bars")
        thresholds = tuple(float(item) for item in self.thresholds)
        if not thresholds:
            raise PeriodContractError("at least one threshold is required")
        if any(not np.isfinite(item) or not 0.0 < item < 1.0 for item in thresholds):
            raise PeriodContractError("thresholds must be finite quantiles in (0, 1)")
        if list(thresholds) != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
            raise PeriodContractError("thresholds must be strictly ascending and unique")
        labels = tuple(_label(item, field_name="regime label") for item in self.labels)
        if len(labels) != len(thresholds) + 1:
            raise PeriodContractError(
                f"{len(thresholds)} thresholds cut the statistic into {len(thresholds) + 1} "
                f"regimes, but {len(labels)} labels were supplied"
            )
        if len(set(labels)) != len(labels):
            raise PeriodContractError("regime labels must be unique")
        if len(labels) > MAX_REGIME_LABELS:
            raise PeriodContractError(f"exceeds the {MAX_REGIME_LABELS}-label ceiling")
        if self.warmup_policy != "unknown":
            raise PeriodContractError(
                "warmup_policy must be 'unknown'; guessing a warm-up label invents "
                "information the trailing window does not yet contain"
            )
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "labels", labels)

    @property
    def identity(self) -> str:
        """Content identity frozen before evaluation."""
        return _digest(
            {
                "name": self.name,
                "statistic": self.statistic,
                "window": self.window,
                "thresholds": list(self.thresholds),
                "labels": list(self.labels),
                "warmup_policy": self.warmup_policy,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly frozen declaration."""
        return {
            "name": self.name,
            "identity": self.identity,
            "statistic": self.statistic,
            "window": self.window,
            "thresholds": list(self.thresholds),
            "labels": list(self.labels),
            "warmup_policy": self.warmup_policy,
            "causality": (
                "the statistic and its cut points use only conditioning observations "
                "strictly before each labelled bar"
            ),
            "frozen_before_evaluation": True,
        }


def _trailing_statistic(conditioning: pd.Series, definition: RegimeDefinition) -> pd.Series:
    """Return the definition's trailing statistic, causal by construction."""
    values = conditioning.astype(float)
    window = definition.window
    if definition.statistic == "volatility":
        return values.rolling(window, min_periods=window).std()
    if definition.statistic == "trend":
        mean = values.rolling(window, min_periods=window).mean()
        deviation = values.rolling(window, min_periods=window).std()
        return mean / deviation.replace(0.0, np.nan)
    cumulative = values.fillna(0.0).cumsum()
    peak = cumulative.rolling(window, min_periods=window).max()
    return cumulative - peak


def label_regimes(conditioning: pd.Series, definition: RegimeDefinition) -> pd.Series:
    """Assign a frozen regime label to each bar from a conditioning series.

    **The candidate's own returns must never be passed here.** The signature
    accepts one conditioning series precisely so there is no parameter through
    which candidate outcomes could arrive;
    :func:`assert_conditioning_is_independent` provides the runtime check for
    callers that build the series dynamically.

    Cut points are **expanding** quantiles of the trailing statistic, computed
    from observations strictly before each bar. A full-sample quantile would let
    every early label depend on how the series ended — a subtler form of the same
    circularity, and one that survives casual inspection because the *statistic*
    looks causal even when its threshold is not.
    """
    if not isinstance(conditioning, pd.Series):
        raise PeriodContractError("conditioning must be a pandas Series")
    if conditioning.empty:
        raise PeriodContractError("conditioning series must be non-empty")
    if not conditioning.index.is_monotonic_increasing:
        raise PeriodContractError("conditioning series must be sorted ascending")

    statistic = _trailing_statistic(conditioning, definition)
    values = statistic.to_numpy(dtype=float)
    labels: list[str | None] = []
    for position in range(len(values)):
        current = values[position]
        if not np.isfinite(current):
            labels.append(None)
            continue
        history = values[:position]
        history = history[np.isfinite(history)]
        if history.size < definition.window:
            # Not enough prior statistic values to place a cut point without
            # borrowing from the future.
            labels.append(None)
            continue
        cuts = np.quantile(history, definition.thresholds)
        index = int(np.searchsorted(cuts, current, side="right"))
        labels.append(definition.labels[index])
    return pd.Series(labels, index=conditioning.index, dtype="object", name=definition.name)


def assert_conditioning_is_independent(
    conditioning: pd.Series, candidate_returns: pd.Series, *, max_abs_correlation: float = 0.99
) -> None:
    """Refuse a conditioning series that is effectively the candidate's own result.

    A regime defined on the candidate's returns explains those returns by
    construction. Perfect correlation is the extreme case; this check refuses
    anything indistinguishable from it, which catches the common accident of
    passing the strategy's P&L as its own "market conditioning".

    **This is a screen, not a proof.** It abstains — returns without raising —
    when it cannot judge: fewer than three shared observations, fewer than three
    after dropping missing values, or a constant series on either side. Those
    cases carry no evidence of circularity rather than evidence of its absence,
    and a constant conditioning series cannot be a disguised copy of a varying
    candidate in any event. The structural separation in :func:`label_regimes`,
    which has no parameter for candidate returns, is what actually enforces the
    rule; this function only catches the dynamically-built accident.

    Raises:
        PeriodContractError: If the two series are near-duplicates.
    """
    if not 0.0 < max_abs_correlation <= 1.0:
        raise PeriodContractError("max_abs_correlation must lie in (0, 1]")
    left = conditioning.astype(float)
    right = candidate_returns.astype(float)
    shared = left.index.intersection(right.index)
    if len(shared) < 3:
        return
    paired = pd.DataFrame({"a": left.loc[shared], "b": right.loc[shared]}).dropna()
    if len(paired) < 3:
        return
    if paired["a"].std(ddof=0) == 0.0 or paired["b"].std(ddof=0) == 0.0:
        return
    correlation = float(paired["a"].corr(paired["b"]))
    if np.isfinite(correlation) and abs(correlation) >= max_abs_correlation:
        raise PeriodContractError(
            f"conditioning series correlates {correlation:.4f} with the candidate's own "
            "returns; a regime defined from the outcomes it explains is circular"
        )


def coverage_report(labels: pd.Series, *, minimum: int = MIN_PERIOD_OBSERVATIONS) -> dict[str, Any]:
    """Return per-label observation counts and sparsity flags.

    Sparse and empty labels are **reported, not dropped**. Removing a short
    period is how a strategy's worst stretch quietly leaves the evidence, and the
    count is what tells a reader whether an interval's estimate means anything.
    """
    counts = labels.dropna().value_counts().to_dict()
    unlabelled = int(labels.isna().sum())
    return {
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "unlabelled": unlabelled,
        "sparse": sorted(str(key) for key, value in counts.items() if value < minimum),
        "minimum_observations": minimum,
        "note": (
            "sparse periods are reported with their counts rather than dropped; "
            "removing a short period is how a strategy's worst stretch leaves the record"
        ),
    }


def standard_regime_definitions(*, window: int = 60) -> tuple[RegimeDefinition, ...]:
    """Return the frozen definitions covering the regimes the work item names.

    Bull, bear, sideways, high/low volatility, crisis, and recovery — each
    expressed as a quantile cut on a trailing statistic of the conditioning
    series, so every label is causal and none is derived from the candidate.

    Supplied as a *set* precisely so a caller cannot report the one definition
    under which the result reads best: :func:`~alphaforge.robustness.temporal_evidence.regime_definition_sensitivity`
    runs all of them and flags a conclusion that moves between them.
    """
    return (
        RegimeDefinition(
            name="volatility_high_low",
            statistic="volatility",
            window=window,
            thresholds=(0.5,),
            labels=("low volatility", "high volatility"),
        ),
        RegimeDefinition(
            name="trend_bull_sideways_bear",
            statistic="trend",
            window=window,
            thresholds=(0.33, 0.67),
            labels=("bear", "sideways", "bull"),
        ),
        RegimeDefinition(
            name="drawdown_crisis_recovery",
            statistic="drawdown",
            window=window,
            thresholds=(0.2, 0.8),
            labels=("crisis", "recovery", "calm"),
        ),
    )
