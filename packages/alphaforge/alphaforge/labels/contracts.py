"""Versioned financial-label contracts and point-in-time-safe materialization.

The legacy :func:`alphaforge.labels.build_labels` surface remains available for
existing experiments.  This module is the strict boundary for governed label
research: each output has an immutable semantic definition, every observation
has a normalized future-event interval, and protected temporal boundaries fail
closed before a label frame is returned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

LabelKind = Literal[
    "regression",
    "classification",
    "threshold",
    "quantile",
    "triple_barrier",
    "volatility_scaled",
    "meta_label",
]
TimingConvention = Literal["close_to_close"]
OverlapPolicy = Literal["allow", "non_overlapping"]
MissingPricePolicy = Literal["raise"]
SideSource = Literal["lagged_return", "column"]

LABEL_CONTRACT_VERSION: Literal["1.0.0"] = "1.0.0"
MAX_HORIZON_SESSIONS = 2_520
MAX_LABEL_DEFINITIONS = 64
MAX_LABEL_CELLS = 50_000_000
MAX_TRIPLE_BARRIER_EVALUATIONS = 50_000_000
MAX_PANEL_ROWS = 10_000_000


class LabelContractError(ValueError):
    """Base error for an invalid label contract or market panel."""


class LabelBoundaryError(LabelContractError):
    """Raised when a label's future interval crosses a protected boundary."""


def _canonical_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LabelDefinition:
    """Immutable semantics for one financial target.

    ``horizon`` is measured in observed market sessions.  A target recorded at
    session ``t`` requires prices strictly after the close at ``t`` through the
    close at ``t + horizon``.  Volatility scaling uses only trailing information
    available at ``t``; it does not extend the future interval.
    """

    name: str
    kind: LabelKind
    horizon: int
    timing: TimingConvention = "close_to_close"
    price_field: Literal["close"] = "close"
    overlap_policy: OverlapPolicy = "allow"
    threshold: float | None = None
    quantiles: int | None = None
    upper_barrier: float | None = None
    lower_barrier: float | None = None
    volatility_window: int | None = None
    side_source: SideSource | None = None
    side_column: str | None = None
    version: Literal["1.0.0"] = LABEL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        supported_kinds = {
            "regression",
            "classification",
            "threshold",
            "quantile",
            "triple_barrier",
            "volatility_scaled",
            "meta_label",
        }
        if self.version != LABEL_CONTRACT_VERSION:
            raise LabelContractError(f"unsupported label definition version {self.version!r}")
        if self.kind not in supported_kinds:
            raise LabelContractError(f"unsupported label kind {self.kind!r}")
        if self.timing != "close_to_close":
            raise LabelContractError(f"unsupported label timing {self.timing!r}")
        if self.price_field != "close":
            raise LabelContractError(f"unsupported label price field {self.price_field!r}")
        if self.overlap_policy not in {"allow", "non_overlapping"}:
            raise LabelContractError(f"unsupported label overlap policy {self.overlap_policy!r}")
        if (
            not self.name
            or len(self.name) > 128
            or self.name != self.name.strip()
            or not self.name.isascii()
            or not all(character.isalnum() or character == "_" for character in self.name)
        ):
            raise LabelContractError(
                "label name must be a non-empty ASCII identifier containing letters, "
                "numbers, or underscores"
            )
        if not 0 < self.horizon <= MAX_HORIZON_SESSIONS:
            raise LabelContractError(
                f"label horizon must be in [1, {MAX_HORIZON_SESSIONS}] sessions"
            )

        supplied = {
            name
            for name, value in {
                "threshold": self.threshold,
                "quantiles": self.quantiles,
                "upper_barrier": self.upper_barrier,
                "lower_barrier": self.lower_barrier,
                "volatility_window": self.volatility_window,
                "side_source": self.side_source,
                "side_column": self.side_column,
            }.items()
            if value is not None
        }
        allowed: dict[LabelKind, set[str]] = {
            "regression": set(),
            "classification": set(),
            "threshold": {"threshold"},
            "quantile": {"quantiles"},
            "triple_barrier": {"upper_barrier", "lower_barrier"},
            "volatility_scaled": {"volatility_window"},
            "meta_label": {"threshold", "side_source", "side_column"},
        }
        required: dict[LabelKind, set[str]] = {
            "regression": set(),
            "classification": set(),
            "threshold": {"threshold"},
            "quantile": {"quantiles"},
            "triple_barrier": {"upper_barrier", "lower_barrier"},
            "volatility_scaled": {"volatility_window"},
            "meta_label": {"threshold", "side_source"},
        }
        unexpected = supplied - allowed[self.kind]
        missing = required[self.kind] - supplied
        if unexpected:
            raise LabelContractError(
                f"{self.kind} label has unsupported parameters: {sorted(unexpected)}"
            )
        if missing:
            raise LabelContractError(f"{self.kind} label requires parameters: {sorted(missing)}")
        if self.threshold is not None and (not np.isfinite(self.threshold) or self.threshold < 0):
            raise LabelContractError("threshold must be finite and non-negative")
        if self.quantiles is not None and not 2 <= self.quantiles <= 20:
            raise LabelContractError("quantiles must be in [2, 20]")
        if self.upper_barrier is not None and (
            not np.isfinite(self.upper_barrier) or self.upper_barrier <= 0
        ):
            raise LabelContractError("upper_barrier must be finite and positive")
        if self.lower_barrier is not None and (
            not np.isfinite(self.lower_barrier) or self.lower_barrier <= 0
        ):
            raise LabelContractError("lower_barrier must be finite and positive")
        if self.volatility_window is not None and self.volatility_window < 2:
            raise LabelContractError("volatility_window must be at least two sessions")
        if self.side_source == "column":
            if (
                not self.side_column
                or len(self.side_column) > 128
                or self.side_column != self.side_column.strip()
                or not self.side_column.isascii()
            ):
                raise LabelContractError(
                    "meta_label with side_source='column' requires a trimmed ASCII side_column"
                )
        elif self.side_column is not None:
            raise LabelContractError(
                "side_column is only valid for meta_label with side_source='column'"
            )

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the normalized parameters that affect mathematical meaning."""

        values = {
            "threshold": self.threshold,
            "quantiles": self.quantiles,
            "upper_barrier": self.upper_barrier,
            "lower_barrier": self.lower_barrier,
            "volatility_window": self.volatility_window,
            "side_source": self.side_source,
            "side_column": self.side_column,
        }
        return {key: value for key, value in values.items() if value is not None}

    @property
    def identity(self) -> str:
        """SHA-256 identity over all semantic label fields."""

        return _canonical_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        """Serialize the complete public contract without implementation state."""

        payload: dict[str, Any] = {
            "version": self.version,
            "name": self.name,
            "kind": self.kind,
            "horizon_sessions": self.horizon,
            "timing_convention": self.timing,
            "price_field": self.price_field,
            "required_future_interval": "(t, t+h]",
            "overlap_policy": self.overlap_policy,
            "parameters": self.parameters,
        }
        if include_identity:
            payload["identity"] = self.identity
        return payload

    def scaled(self, scale: float) -> LabelDefinition:
        """Return a deterministic parameter-sensitivity variant.

        Only magnitude parameters are changed.  Classification, regression,
        quantile, and volatility-scaled labels have no magnitude parameter and
        therefore return an identical definition.
        """

        if not np.isfinite(scale) or scale <= 0:
            raise LabelContractError("sensitivity scale must be finite and positive")
        return replace(
            self,
            threshold=None if self.threshold is None else self.threshold * scale,
            upper_barrier=(None if self.upper_barrier is None else self.upper_barrier * scale),
            lower_barrier=(None if self.lower_barrier is None else self.lower_barrier * scale),
        )


@dataclass(frozen=True)
class LabelContract:
    """A complete, versioned label-materialization request."""

    benchmark_symbol: str
    definitions: tuple[LabelDefinition, ...]
    missing_price_policy: MissingPricePolicy = "raise"
    protected_boundaries: tuple[pd.Timestamp, ...] = ()
    version: Literal["1.0.0"] = LABEL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != LABEL_CONTRACT_VERSION:
            raise LabelContractError(f"unsupported label contract version {self.version!r}")
        if self.missing_price_policy != "raise":
            raise LabelContractError(
                f"unsupported missing-price policy {self.missing_price_policy!r}"
            )
        if (
            not self.benchmark_symbol
            or len(self.benchmark_symbol) > 128
            or self.benchmark_symbol != self.benchmark_symbol.strip()
            or not self.benchmark_symbol.isascii()
        ):
            raise LabelContractError("benchmark_symbol must be non-empty, trimmed ASCII")
        if not 1 <= len(self.definitions) <= MAX_LABEL_DEFINITIONS:
            raise LabelContractError(
                f"label contract requires between 1 and {MAX_LABEL_DEFINITIONS} definitions"
            )
        names = [definition.name for definition in self.definitions]
        if len(names) != len(set(names)):
            raise LabelContractError("label definition names must be unique")
        normalized_boundaries: list[pd.Timestamp] = []
        for boundary in self.protected_boundaries:
            normalized = pd.Timestamp(boundary)
            if normalized.tzinfo is not None:
                raise LabelContractError(
                    "protected boundaries must be timezone-naive session dates"
                )
            normalized_boundaries.append(normalized.normalize())
        if normalized_boundaries != sorted(set(normalized_boundaries)):
            raise LabelContractError("protected boundaries must be strictly increasing and unique")
        object.__setattr__(self, "protected_boundaries", tuple(normalized_boundaries))

    @property
    def identity(self) -> str:
        """SHA-256 identity over the complete ordered label contract."""

        return _canonical_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        """Serialize the contract for manifests and reproducible evidence."""

        payload: dict[str, Any] = {
            "version": self.version,
            "benchmark_symbol": self.benchmark_symbol,
            "missing_price_policy": self.missing_price_policy,
            "protected_boundaries": [
                boundary.date().isoformat() for boundary in self.protected_boundaries
            ],
            "definitions": [
                definition.to_dict(include_identity=True) for definition in self.definitions
            ],
        }
        if include_identity:
            payload["identity"] = self.identity
        return payload

    def scaled(self, scale: float) -> LabelContract:
        """Return a contract whose magnitude parameters are scaled uniformly."""

        return replace(
            self,
            definitions=tuple(definition.scaled(scale) for definition in self.definitions),
        )

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> LabelContract:
        """Construct a domain contract from a validated ``labels.yaml`` mapping."""

        boundaries = tuple(pd.Timestamp(value) for value in config.get("protected_boundaries", []))
        definitions = tuple(
            LabelDefinition(
                name=entry["name"],
                kind=entry["kind"],
                horizon=entry["horizon"],
                timing=entry["timing"],
                price_field=entry["price_field"],
                overlap_policy=entry["overlap_policy"],
                threshold=entry.get("threshold"),
                quantiles=entry.get("quantiles"),
                upper_barrier=entry.get("upper_barrier"),
                lower_barrier=entry.get("lower_barrier"),
                volatility_window=entry.get("volatility_window"),
                side_source=entry.get("side_source"),
                side_column=entry.get("side_column"),
                version=entry["version"],
            )
            for entry in config["labels"]
        )
        return cls(
            version=config["version"],
            benchmark_symbol=config["benchmark_symbol"],
            definitions=definitions,
            missing_price_policy=config["missing_price_policy"],
            protected_boundaries=boundaries,
        )


@dataclass(frozen=True)
class LabelDataset:
    """Materialized values and normalized future-event intervals."""

    contract: LabelContract
    values: pd.DataFrame
    events: pd.DataFrame

    def manifest(self) -> dict[str, Any]:
        """Return licensed-data-safe aggregate evidence for this materialization."""

        return {
            "schema_version": LABEL_CONTRACT_VERSION,
            "contract": self.contract.to_dict(),
            "rows": int(len(self.values)),
            "symbols": sorted(self.values["symbol"].unique().tolist()),
            "date_start": pd.Timestamp(self.values["date"].min()).date().isoformat(),
            "date_end": pd.Timestamp(self.values["date"].max()).date().isoformat(),
            "event_rows": int(len(self.events)),
            "observable_event_rows": int(self.events["observable"].sum()),
        }


def _validate_panel(panel: pd.DataFrame, contract: LabelContract) -> pd.DataFrame:
    if not 0 < len(panel) <= MAX_PANEL_ROWS:
        raise LabelContractError(f"label panel must contain between 1 and {MAX_PANEL_ROWS:,} rows")
    required = {"date", "symbol", "close"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise LabelContractError(f"label panel is missing required columns: {missing}")

    out = panel.copy()
    try:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise LabelContractError("label panel dates must be valid session timestamps") from exc
    if out["date"].dt.tz is not None:
        raise LabelContractError("label panel dates must be timezone-naive session labels")
    out["date"] = out["date"].dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    if (
        out["symbol"].eq("").any()
        or out["symbol"].str.strip().ne(out["symbol"]).any()
        or not out["symbol"].map(str.isascii).all()
    ):
        raise LabelContractError("symbols must be non-empty, trimmed ASCII identifiers")
    if out.duplicated(["date", "symbol"]).any():
        raise LabelContractError("label panel contains duplicate (date, symbol) observations")
    try:
        out["close"] = pd.to_numeric(out["close"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise LabelContractError("close prices must be numeric") from exc
    close_values = out["close"].to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or (close_values <= 0).any():
        raise LabelContractError("close prices must be finite and strictly positive")
    if contract.benchmark_symbol not in set(out["symbol"]):
        raise LabelContractError("benchmark_symbol is unavailable in the label panel")

    benchmark_dates = pd.Index(
        out.loc[out["symbol"].eq(contract.benchmark_symbol), "date"].sort_values().unique()
    )
    max_horizon = max(definition.horizon for definition in contract.definitions)
    if len(benchmark_dates) <= max_horizon:
        raise LabelContractError(
            "maximum label horizon overflows the available benchmark session history"
        )

    for symbol, group in out.groupby("symbol", sort=True):
        dates = pd.Index(group["date"].sort_values().unique())
        if len(dates) <= max_horizon:
            raise LabelContractError(
                f"maximum label horizon overflows available history for symbol {symbol!r}"
            )
        expected = benchmark_dates[
            (benchmark_dates >= dates.min()) & (benchmark_dates <= dates.max())
        ]
        unavailable = expected.difference(dates)
        if len(unavailable):
            first = pd.Timestamp(unavailable[0]).date().isoformat()
            raise LabelContractError(
                f"symbol {symbol!r} has {len(unavailable)} unavailable close prices "
                f"within its active interval; first missing session is {first}"
            )

    for definition in contract.definitions:
        if definition.side_source != "column":
            continue
        assert definition.side_column is not None
        if definition.side_column not in out.columns:
            raise LabelContractError(
                f"meta-label side column {definition.side_column!r} is unavailable"
            )
        side = pd.to_numeric(out[definition.side_column], errors="coerce")
        tradable = out["symbol"].ne(contract.benchmark_symbol)
        if side.loc[tradable].isna().any() or not side.loc[tradable].isin([-1.0, 1.0]).all():
            raise LabelContractError(
                f"meta-label side column {definition.side_column!r} must contain only -1 or 1"
            )
        out[definition.side_column] = side.astype(float)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _future_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon).divide(close).subtract(1.0)


def _triple_barrier(close: pd.Series, definition: LabelDefinition) -> pd.Series:
    assert definition.upper_barrier is not None
    assert definition.lower_barrier is not None
    prices = close.to_numpy(dtype=float)
    labels = np.full(len(prices), np.nan, dtype=float)
    for position in range(len(prices) - definition.horizon):
        path = prices[position + 1 : position + definition.horizon + 1] / prices[position] - 1.0
        upper = np.flatnonzero(path >= definition.upper_barrier)
        lower = np.flatnonzero(path <= -definition.lower_barrier)
        first_upper = int(upper[0]) if upper.size else definition.horizon
        first_lower = int(lower[0]) if lower.size else definition.horizon
        if first_upper < first_lower:
            labels[position] = 1.0
        elif first_lower < first_upper:
            labels[position] = -1.0
        else:
            labels[position] = 0.0
    return pd.Series(labels, index=close.index)


def _quantile_labels(values: pd.Series, quantiles: int) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    eligible = values.dropna()
    if len(eligible) < quantiles:
        return result
    ranks = eligible.rank(method="first")
    result.loc[eligible.index] = np.ceil(ranks * quantiles / len(eligible)).clip(
        lower=1, upper=quantiles
    )
    return result


def _definition_values(group: pd.DataFrame, definition: LabelDefinition) -> pd.Series:
    close = group["close"].astype(float)
    forward = _future_return(close, definition.horizon)
    if definition.kind == "regression" or definition.kind == "quantile":
        return forward
    if definition.kind == "classification":
        return forward.gt(0).astype(float).where(forward.notna())
    if definition.kind == "threshold":
        assert definition.threshold is not None
        values = np.select(
            [forward > definition.threshold, forward < -definition.threshold],
            [1.0, -1.0],
            default=0.0,
        )
        return pd.Series(values, index=group.index).where(forward.notna())
    if definition.kind == "triple_barrier":
        return _triple_barrier(close, definition)
    if definition.kind == "volatility_scaled":
        assert definition.volatility_window is not None
        trailing_vol = (
            close.pct_change()
            .rolling(
                definition.volatility_window,
                min_periods=definition.volatility_window,
            )
            .std(ddof=1)
        )
        horizon_vol = trailing_vol * np.sqrt(definition.horizon)
        return forward.divide(horizon_vol.replace(0, np.nan))
    if definition.kind == "meta_label":
        assert definition.threshold is not None
        if definition.side_source == "lagged_return":
            side = np.sign(close.pct_change()).replace(0, np.nan)
        else:
            assert definition.side_column is not None
            side = group[definition.side_column].astype(float)
        outcome = side.multiply(forward)
        return outcome.gt(definition.threshold).astype(float).where(outcome.notna())
    raise AssertionError(f"unsupported label kind: {definition.kind}")


def _event_frame(group: pd.DataFrame, definition: LabelDefinition) -> pd.DataFrame:
    dates = group["date"].reset_index(drop=True)
    events = pd.DataFrame(
        {
            "date": dates,
            "symbol": group["symbol"].iloc[0],
            "label": definition.name,
            "label_id": definition.identity,
            "required_future_start": dates.shift(-1),
            "required_future_end": dates.shift(-definition.horizon),
        }
    )
    events["observable"] = events["required_future_end"].notna()
    if definition.overlap_policy == "non_overlapping":
        positions = np.arange(len(events))
        selected = positions % definition.horizon == 0
        events["observable"] &= selected
    return events


def _reject_boundary_crossings(events: pd.DataFrame, boundaries: tuple[pd.Timestamp, ...]) -> None:
    for boundary in boundaries:
        crossing = events[
            events["observable"]
            & events["date"].lt(boundary)
            & events["required_future_end"].ge(boundary)
        ]
        if crossing.empty:
            continue
        first = crossing.iloc[0]
        raise LabelBoundaryError(
            f"label {first['label']!r} for symbol {first['symbol']!r} at "
            f"{pd.Timestamp(first['date']).date().isoformat()} crosses protected boundary "
            f"{boundary.date().isoformat()}"
        )


def build_label_set(panel: pd.DataFrame, contract: LabelContract) -> LabelDataset:
    """Materialize a governed label dataset.

    Args:
        panel: Long market panel containing at least ``date``, ``symbol``, and
            finite positive ``close``.  Dates are session labels, not timestamps
            at which a result became observable.
        contract: Versioned definitions and boundary policy.

    Returns:
        A deterministic value matrix keyed by ``(date, symbol)`` and a normalized
        event table recording the exact future interval required by every label.

    Raises:
        LabelContractError: On invalid contracts, duplicate or unavailable
            prices, insufficient history, cross-symbol calendar gaps, or invalid
            side inputs.
        LabelBoundaryError: If any observable future interval crosses a protected
            temporal boundary.

    Complexity:
        All labels except triple-barrier are linear in rows.  Triple-barrier
        evaluation is ``O(rows * horizon)`` and horizons are contract-bounded.
    """

    validated = _validate_panel(panel, contract)
    tradable = validated.loc[validated["symbol"].ne(contract.benchmark_symbol)].reset_index(
        drop=True
    )
    if tradable.empty:
        raise LabelContractError("label panel contains no tradable symbols")
    label_cells = len(tradable) * len(contract.definitions)
    if label_cells > MAX_LABEL_CELLS:
        raise LabelContractError(
            f"label request would materialize {label_cells:,} cells; "
            f"the limit is {MAX_LABEL_CELLS:,}"
        )
    triple_barrier_evaluations = len(tradable) * sum(
        definition.horizon
        for definition in contract.definitions
        if definition.kind == "triple_barrier"
    )
    if triple_barrier_evaluations > MAX_TRIPLE_BARRIER_EVALUATIONS:
        raise LabelContractError(
            "triple-barrier request exceeds the bounded path-evaluation budget"
        )

    frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for _, group in tradable.groupby("symbol", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        frame = group[["date", "symbol"]].copy()
        for definition in contract.definitions:
            frame[definition.name] = _definition_values(group, definition).to_numpy()
            events = _event_frame(group, definition)
            if definition.overlap_policy == "non_overlapping":
                frame.loc[~events["observable"].to_numpy(), definition.name] = np.nan
            event_frames.append(events)
        frames.append(frame)

    values = (
        pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    )
    for definition in contract.definitions:
        if definition.kind != "quantile":
            continue
        assert definition.quantiles is not None
        quantiles = definition.quantiles
        values[definition.name] = values.groupby("date", sort=False)[definition.name].transform(
            lambda series, q=quantiles: _quantile_labels(series, q)
        )

    events = (
        pd.concat(event_frames, ignore_index=True)
        .sort_values(["label", "symbol", "date"])
        .reset_index(drop=True)
    )
    _reject_boundary_crossings(events, contract.protected_boundaries)

    numeric = values[[definition.name for definition in contract.definitions]].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise LabelContractError("label materialization produced infinite values")
    return LabelDataset(contract=contract, values=values, events=events)
