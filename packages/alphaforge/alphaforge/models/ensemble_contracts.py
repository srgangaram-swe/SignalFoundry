"""Immutable boundaries for leakage-safe ensemble training and inference.

The training boundary contains only complete, temporally attested
out-of-fold (OOF) predictions.  The inference boundary deliberately has no
target field, so a final-holdout outcome cannot cross into an ensemble policy
by accident.  Both boundaries canonicalize candidate order and expose stable
SHA-256 identities for audit records and deterministic replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
import pandas as pd

ENSEMBLE_CONTRACT_VERSION = "1.0.0"
DEFAULT_MAX_PREDICTION_RECORDS = 1_000_000
DEFAULT_MAX_EXPERTS = 64
DEFAULT_MAX_DATES = 10_000
DEFAULT_MAX_SERIALIZED_BYTES = 256 * 1024 * 1024

_TRAINING_COLUMNS = frozenset(
    {
        "date",
        "symbol",
        "fold_id",
        "expert",
        "prediction",
        "target",
        "uncertainty",
        "regime_probability",
    }
)
_INFERENCE_COLUMNS = frozenset(
    {
        "date",
        "symbol",
        "expert",
        "prediction",
        "uncertainty",
        "regime_probability",
    }
)
_GOVERNED_METHODS = frozenset(
    {"static", "rank_vote", "stacking", "bayesian", "dynamic", "regime_gate"}
)


class EnsembleContractError(ValueError):
    """Base error for an invalid governed-ensemble boundary."""


class HoldoutLeakageError(EnsembleContractError):
    """Raised when final-holdout data enters a training-only boundary."""


class IncompleteOOFError(EnsembleContractError):
    """Raised when a training date lacks a declared expert prediction."""


def _identifier(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value != value.strip()
        or not value.isascii()
        or not all(character.isalnum() or character in "._-" for character in value)
    ):
        raise EnsembleContractError(
            f"{field} must be a non-empty safe ASCII identifier of at most 128 characters"
        )
    return value


def _date(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise EnsembleContractError(f"{field} must be an ISO-8601 calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EnsembleContractError(f"{field} must be an ISO-8601 calendar date") from exc
    if parsed.isoformat() != value:
        raise EnsembleContractError(f"{field} must use canonical YYYY-MM-DD form")
    return value


def _finite(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise EnsembleContractError(f"{field} must be finite")
    return float(value)


def _frame_numeric(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EnsembleContractError(f"{field} must be numeric")
    return float(value)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise EnsembleContractError(f"{field} must be a positive integer")
    return int(value)


def _frame_date(value: Any, field: str) -> str:
    if isinstance(value, str):
        return _date(value, field)
    if not isinstance(value, (date, pd.Timestamp)):
        raise EnsembleContractError(f"{field} must be a calendar date or midnight timestamp")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise EnsembleContractError(f"{field} is invalid") from exc
    if (
        pd.isna(timestamp)
        or timestamp.tzinfo is not None
        or timestamp.hour
        or timestamp.minute
        or timestamp.second
        or timestamp.microsecond
        or timestamp.nanosecond
    ):
        raise EnsembleContractError(f"{field} must be a timezone-naive midnight calendar date")
    return timestamp.date().isoformat()


def _optional_non_negative(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    checked = _finite(value, field)
    if checked < 0.0:
        raise EnsembleContractError(f"{field} must be non-negative")
    return checked


def _optional_probability(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    checked = _finite(value, field)
    if not 0.0 <= checked <= 1.0:
        raise EnsembleContractError(f"{field} must be in [0, 1]")
    return checked


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _identity(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class TemporalOOFFold:
    """Attestation for one forward validation block.

    ``training_end`` must strictly precede ``validation_start``.  Predictions
    carrying this fold id may only use dates inside the validation interval.
    """

    fold_id: int
    training_end: str
    validation_start: str
    validation_end: str
    embargo_days: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_id, bool)
            or not isinstance(self.fold_id, Integral)
            or self.fold_id < 0
        ):
            raise EnsembleContractError("fold_id must be a non-negative integer")
        object.__setattr__(self, "fold_id", int(self.fold_id))
        _date(self.training_end, "training_end")
        _date(self.validation_start, "validation_start")
        _date(self.validation_end, "validation_end")
        if not self.training_end < self.validation_start <= self.validation_end:
            raise EnsembleContractError(
                "fold dates must satisfy training_end < validation_start <= validation_end"
            )
        if (
            isinstance(self.embargo_days, bool)
            or not isinstance(self.embargo_days, Integral)
            or self.embargo_days < 0
        ):
            raise EnsembleContractError("embargo_days must be a non-negative integer")
        object.__setattr__(self, "embargo_days", int(self.embargo_days))
        gap_days = (
            date.fromisoformat(self.validation_start) - date.fromisoformat(self.training_end)
        ).days - 1
        if gap_days < self.embargo_days:
            raise EnsembleContractError(
                "fold dates do not provide the declared intervening calendar-day embargo"
            )


@dataclass(frozen=True, order=True)
class TrainingOOFPrediction:
    """One expert prediction generated outside its temporal training fold."""

    date: str
    symbol: str
    fold_id: int
    expert: str
    prediction: float
    uncertainty: float | None = None
    regime_probability: float | None = None

    def __post_init__(self) -> None:
        _date(self.date, "prediction.date")
        _identifier(self.symbol, "prediction.symbol")
        if (
            isinstance(self.fold_id, bool)
            or not isinstance(self.fold_id, Integral)
            or self.fold_id < 0
        ):
            raise EnsembleContractError("prediction.fold_id must be a non-negative integer")
        object.__setattr__(self, "fold_id", int(self.fold_id))
        _identifier(self.expert, "prediction.expert")
        object.__setattr__(self, "prediction", _finite(self.prediction, "prediction"))
        object.__setattr__(
            self,
            "uncertainty",
            _optional_non_negative(self.uncertainty, "prediction.uncertainty"),
        )
        object.__setattr__(
            self,
            "regime_probability",
            _optional_probability(self.regime_probability, "prediction.regime_probability"),
        )


@dataclass(frozen=True, order=True)
class TrainingOOFTarget:
    """One label paired to OOF predictions after expert fitting completed."""

    date: str
    symbol: str
    fold_id: int
    target: float

    def __post_init__(self) -> None:
        _date(self.date, "target.date")
        _identifier(self.symbol, "target.symbol")
        if (
            isinstance(self.fold_id, bool)
            or not isinstance(self.fold_id, Integral)
            or self.fold_id < 0
        ):
            raise EnsembleContractError("target.fold_id must be a non-negative integer")
        object.__setattr__(self, "fold_id", int(self.fold_id))
        object.__setattr__(self, "target", _finite(self.target, "target"))


@dataclass(frozen=True, order=True)
class InferencePrediction:
    """One target-free expert prediction for a frozen evaluation period."""

    date: str
    symbol: str
    expert: str
    prediction: float
    uncertainty: float | None = None
    regime_probability: float | None = None

    def __post_init__(self) -> None:
        _date(self.date, "prediction.date")
        _identifier(self.symbol, "prediction.symbol")
        _identifier(self.expert, "prediction.expert")
        object.__setattr__(self, "prediction", _finite(self.prediction, "prediction"))
        object.__setattr__(
            self,
            "uncertainty",
            _optional_non_negative(self.uncertainty, "prediction.uncertainty"),
        )
        object.__setattr__(
            self,
            "regime_probability",
            _optional_probability(self.regime_probability, "prediction.regime_probability"),
        )


@dataclass(frozen=True)
class TrainingOOFPanel:
    """Complete, canonical training-only prediction panel.

    Every ``(date, symbol, fold_id)`` target must have exactly one prediction
    from every expected expert.  All records must predate ``holdout_start`` and
    fall inside their fold's validation interval.
    """

    predictions: tuple[TrainingOOFPrediction, ...]
    targets: tuple[TrainingOOFTarget, ...]
    folds: tuple[TemporalOOFFold, ...]
    expected_experts: tuple[str, ...]
    holdout_start: str
    source_id: str
    max_records: int = DEFAULT_MAX_PREDICTION_RECORDS
    max_dates: int = DEFAULT_MAX_DATES

    def __post_init__(self) -> None:
        _date(self.holdout_start, "holdout_start")
        _identifier(self.source_id, "source_id")
        if (
            isinstance(self.max_records, bool)
            or not isinstance(self.max_records, Integral)
            or self.max_records <= 0
        ):
            raise EnsembleContractError("max_records must be a positive integer")
        if (
            isinstance(self.max_dates, bool)
            or not isinstance(self.max_dates, Integral)
            or self.max_dates <= 0
        ):
            raise EnsembleContractError("max_dates must be a positive integer")
        if any(not isinstance(expert, str) for expert in self.expected_experts):
            raise EnsembleContractError("expected_experts must contain safe ASCII identifiers")
        experts = tuple(sorted(self.expected_experts))
        if not experts or len(experts) > DEFAULT_MAX_EXPERTS or len(experts) != len(set(experts)):
            raise EnsembleContractError("expected_experts must contain 1..64 unique experts")
        for expert in experts:
            _identifier(expert, "expected_expert")
        predictions = tuple(sorted(self.predictions))
        targets = tuple(sorted(self.targets))
        folds = tuple(sorted(self.folds))
        object.__setattr__(self, "expected_experts", experts)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "folds", folds)
        if not predictions or not targets or not folds:
            raise EnsembleContractError("OOF panel predictions, targets, and folds cannot be empty")
        if len(predictions) > self.max_records:
            raise EnsembleContractError("OOF panel exceeds max_records")
        dates = {record.date for record in predictions}
        if len(dates) > self.max_dates:
            raise EnsembleContractError("OOF panel exceeds max_dates")
        fold_map = {fold.fold_id: fold for fold in folds}
        if len(fold_map) != len(folds):
            raise EnsembleContractError("fold ids must be unique")
        prior_end: str | None = None
        for fold in folds:
            if fold.validation_end >= self.holdout_start:
                raise HoldoutLeakageError("OOF fold reaches the declared final holdout")
            if prior_end is not None and fold.validation_start <= prior_end:
                raise EnsembleContractError("OOF validation folds must be disjoint and ordered")
            prior_end = fold.validation_end

        target_map = {(row.date, row.symbol, row.fold_id): row.target for row in targets}
        if len(target_map) != len(targets):
            raise EnsembleContractError("OOF target identities must be unique")
        folds_by_date: dict[str, set[int]] = {}
        for target in targets:
            folds_by_date.setdefault(target.date, set()).add(target.fold_id)
        if any(len(fold_ids) != 1 for fold_ids in folds_by_date.values()):
            raise EnsembleContractError("every complete OOF date must belong to exactly one fold")
        prediction_keys: set[tuple[str, str, int, str]] = set()
        observed: dict[tuple[str, str, int], set[str]] = {}
        regimes: dict[tuple[str, str, int], float | None] = {}
        for row in predictions:
            if row.expert not in experts:
                raise EnsembleContractError(f"undeclared expert {row.expert!r}")
            if row.date >= self.holdout_start:
                raise HoldoutLeakageError("final-holdout prediction entered training OOF panel")
            selected_fold = fold_map.get(row.fold_id)
            if selected_fold is None:
                raise EnsembleContractError(f"prediction references unknown fold {row.fold_id}")
            if not selected_fold.validation_start <= row.date <= selected_fold.validation_end:
                raise EnsembleContractError("prediction date lies outside its OOF validation fold")
            key = (row.date, row.symbol, row.fold_id)
            if key not in target_map:
                raise EnsembleContractError("prediction has no exactly aligned OOF target")
            prediction_key = (*key, row.expert)
            if prediction_key in prediction_keys:
                raise EnsembleContractError("OOF prediction identities must be unique")
            prediction_keys.add(prediction_key)
            observed.setdefault(key, set()).add(row.expert)
            if key in regimes and regimes[key] != row.regime_probability:
                raise EnsembleContractError("regime probability must agree across experts")
            regimes[key] = row.regime_probability
        if set(target_map) != set(observed):
            raise EnsembleContractError("every OOF target must have expert predictions")
        incomplete = [key for key, names in observed.items() if names != set(experts)]
        if incomplete:
            raise IncompleteOOFError(
                f"{len(incomplete)} OOF rows do not contain the complete expert set"
            )
        if {target.fold_id for target in targets} != set(fold_map):
            raise EnsembleContractError("every declared fold must contribute OOF targets")

    @property
    def identity(self) -> str:
        """Stable content identity independent of caller row/candidate order."""

        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic serialization payload."""

        return {
            "contract_version": ENSEMBLE_CONTRACT_VERSION,
            "source_id": self.source_id,
            "holdout_start": self.holdout_start,
            "expected_experts": list(self.expected_experts),
            "folds": [asdict(value) for value in self.folds],
            "predictions": [asdict(value) for value in self.predictions],
            "targets": [asdict(value) for value in self.targets],
        }

    def to_json(self) -> str:
        """Serialize deterministically without executable objects."""

        return _canonical_json(self.to_dict()) + "\n"

    @classmethod
    def from_json(
        cls,
        payload: str | bytes,
        *,
        max_bytes: int = DEFAULT_MAX_SERIALIZED_BYTES,
        max_records: int = DEFAULT_MAX_PREDICTION_RECORDS,
        max_dates: int = DEFAULT_MAX_DATES,
    ) -> TrainingOOFPanel:
        """Validate and load a resource-bounded JSON panel."""

        if not isinstance(payload, (str, bytes)):
            raise EnsembleContractError("serialized OOF panel must be text or bytes")
        max_bytes = _positive_integer(max_bytes, "max_bytes")
        max_records = _positive_integer(max_records, "max_records")
        max_dates = _positive_integer(max_dates, "max_dates")
        encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
        if len(encoded) > max_bytes:
            raise EnsembleContractError("serialized OOF panel exceeds max_bytes")
        try:
            document = json.loads(encoded)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EnsembleContractError("OOF panel is not valid UTF-8 JSON") from exc
        expected = {
            "contract_version",
            "source_id",
            "holdout_start",
            "expected_experts",
            "folds",
            "predictions",
            "targets",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise EnsembleContractError("OOF panel fields mismatch")
        if document["contract_version"] != ENSEMBLE_CONTRACT_VERSION:
            raise EnsembleContractError("unsupported ensemble contract version")
        try:
            return cls(
                predictions=tuple(TrainingOOFPrediction(**row) for row in document["predictions"]),
                targets=tuple(TrainingOOFTarget(**row) for row in document["targets"]),
                folds=tuple(TemporalOOFFold(**row) for row in document["folds"]),
                expected_experts=tuple(document["expected_experts"]),
                holdout_start=document["holdout_start"],
                source_id=document["source_id"],
                max_records=max_records,
                max_dates=max_dates,
            )
        except (TypeError, KeyError) as exc:
            raise EnsembleContractError("malformed OOF panel records") from exc

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        folds: tuple[TemporalOOFFold, ...],
        expected_experts: tuple[str, ...],
        holdout_start: str,
        source_id: str,
        max_records: int = DEFAULT_MAX_PREDICTION_RECORDS,
        max_dates: int = DEFAULT_MAX_DATES,
    ) -> TrainingOOFPanel:
        """Build the immutable boundary from an exact-schema DataFrame."""

        max_records = _positive_integer(max_records, "max_records")
        max_dates = _positive_integer(max_dates, "max_dates")
        if (
            not isinstance(frame, pd.DataFrame)
            or len(frame.columns) != len(_TRAINING_COLUMNS)
            or set(frame.columns) != _TRAINING_COLUMNS
        ):
            raise EnsembleContractError(
                f"training OOF frame fields mismatch; expected {sorted(_TRAINING_COLUMNS)}"
            )
        if len(frame) > max_records:
            raise EnsembleContractError("OOF frame exceeds max_records")
        records: list[TrainingOOFPrediction] = []
        target_values: dict[tuple[str, str, int], float] = {}
        for row in frame.itertuples(index=False):
            date_value = _frame_date(row.date, "training date")
            if isinstance(row.fold_id, bool) or not isinstance(row.fold_id, Integral):
                raise EnsembleContractError("fold_id must be an integer")
            if isinstance(row.target, bool) or not isinstance(row.target, Real):
                raise EnsembleContractError("target must be numeric")
            if isinstance(row.prediction, bool) or not isinstance(row.prediction, Real):
                raise EnsembleContractError("prediction must be numeric")
            if not isinstance(row.symbol, str) or not isinstance(row.expert, str):
                raise EnsembleContractError("symbol and expert must be strings")
            fold_id = int(row.fold_id)
            key = (date_value, row.symbol, fold_id)
            target = _finite(float(row.target), "target")
            if key in target_values and target_values[key] != target:
                raise EnsembleContractError("target must agree across expert rows")
            target_values[key] = target
            records.append(
                TrainingOOFPrediction(
                    date=date_value,
                    symbol=row.symbol,
                    fold_id=fold_id,
                    expert=row.expert,
                    prediction=float(row.prediction),
                    uncertainty=(
                        None
                        if pd.isna(row.uncertainty)
                        else _frame_numeric(row.uncertainty, "uncertainty")
                    ),
                    regime_probability=(
                        None
                        if pd.isna(row.regime_probability)
                        else _frame_numeric(row.regime_probability, "regime_probability")
                    ),
                )
            )
        targets = tuple(
            TrainingOOFTarget(date_value, symbol, fold_id, target)
            for (date_value, symbol, fold_id), target in target_values.items()
        )
        return cls(
            predictions=tuple(records),
            targets=targets,
            folds=folds,
            expected_experts=expected_experts,
            holdout_start=holdout_start,
            source_id=source_id,
            max_records=max_records,
            max_dates=max_dates,
        )

    def wide_frame(self) -> pd.DataFrame:
        """Return a sorted in-memory training matrix for policy fitting."""

        prediction_frame = pd.DataFrame(asdict(row) for row in self.predictions)
        wide = prediction_frame.pivot(
            index=["date", "symbol", "fold_id"],
            columns="expert",
            values="prediction",
        ).loc[:, list(self.expected_experts)]
        targets = pd.DataFrame(asdict(row) for row in self.targets).set_index(
            ["date", "symbol", "fold_id"]
        )
        regime = (
            prediction_frame.groupby(["date", "symbol", "fold_id"], sort=True)["regime_probability"]
            .first()
            .rename("regime_probability")
        )
        return (
            wide.join(targets[["target"]], how="inner")
            .join(regime, how="left")
            .reset_index()
            .sort_values(["date", "symbol", "fold_id"])
            .reset_index(drop=True)
        )


@dataclass(frozen=True)
class InferenceBatch:
    """Canonical target-free inference predictions; missing experts are allowed.

    Missing experts are intentionally retained for the policy layer to turn
    into explicit abstention decisions.  Duplicate identities and malformed
    values still fail at the boundary.
    """

    predictions: tuple[InferencePrediction, ...]
    expected_keys: tuple[tuple[str, str], ...]
    max_records: int = DEFAULT_MAX_PREDICTION_RECORDS

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_records, bool)
            or not isinstance(self.max_records, Integral)
            or self.max_records <= 0
        ):
            raise EnsembleContractError("max_records must be a positive integer")
        predictions = tuple(sorted(self.predictions))
        try:
            raw_expected_keys = tuple(self.expected_keys)
        except TypeError as exc:
            raise EnsembleContractError("expected inference keys are malformed") from exc
        validated_keys: list[tuple[str, str]] = []
        for key in raw_expected_keys:
            if not isinstance(key, tuple) or len(key) != 2:
                raise EnsembleContractError("expected inference keys must be (date, symbol) tuples")
            validated_keys.append(
                (
                    _date(key[0], "expected inference date"),
                    _identifier(key[1], "expected inference symbol"),
                )
            )
        expected_keys = tuple(sorted(validated_keys))
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "expected_keys", expected_keys)
        if not expected_keys or len(expected_keys) != len(set(expected_keys)):
            raise EnsembleContractError("expected inference keys must be non-empty and unique")
        if len(predictions) > self.max_records:
            raise EnsembleContractError("inference batch exceeds max_records")
        if len(expected_keys) > self.max_records:
            raise EnsembleContractError("inference expected keys exceed max_records")
        keys = {(row.date, row.symbol, row.expert) for row in predictions}
        if len(keys) != len(predictions):
            raise EnsembleContractError("inference prediction identities must be unique")
        observed_keys = {(row.date, row.symbol) for row in predictions}
        unexpected = sorted(observed_keys - set(expected_keys))
        if unexpected:
            raise EnsembleContractError(
                f"inference predictions contain {len(unexpected)} unexpected date-symbol keys"
            )

    @property
    def identity(self) -> str:
        return _identity(
            {
                "expected_keys": [list(key) for key in self.expected_keys],
                "predictions": [asdict(row) for row in self.predictions],
            }
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        expected_keys: tuple[tuple[str, str], ...],
        max_records: int = DEFAULT_MAX_PREDICTION_RECORDS,
    ) -> InferenceBatch:
        max_records = _positive_integer(max_records, "max_records")
        if (
            not isinstance(frame, pd.DataFrame)
            or len(frame.columns) != len(_INFERENCE_COLUMNS)
            or set(frame.columns) != _INFERENCE_COLUMNS
        ):
            raise EnsembleContractError(
                f"inference frame fields mismatch; expected {sorted(_INFERENCE_COLUMNS)}"
            )
        if len(frame) > max_records:
            raise EnsembleContractError("inference frame exceeds max_records")
        records: list[InferencePrediction] = []
        for row in frame.itertuples(index=False):
            date_value = _frame_date(row.date, "inference date")
            if not isinstance(row.symbol, str) or not isinstance(row.expert, str):
                raise EnsembleContractError("symbol and expert must be strings")
            records.append(
                InferencePrediction(
                    date=date_value,
                    symbol=row.symbol,
                    expert=row.expert,
                    prediction=_frame_numeric(row.prediction, "prediction"),
                    uncertainty=(
                        None
                        if pd.isna(row.uncertainty)
                        else _frame_numeric(row.uncertainty, "uncertainty")
                    ),
                    regime_probability=(
                        None
                        if pd.isna(row.regime_probability)
                        else _frame_numeric(row.regime_probability, "regime_probability")
                    ),
                )
            )
        return cls(
            tuple(records),
            expected_keys=expected_keys,
            max_records=max_records,
        )


@dataclass(frozen=True)
class EnsembleDecision:
    """One immutable combined prediction or explicit fail-closed abstention."""

    date: str
    symbol: str
    method: str
    prediction: float
    uncertainty: float | None
    status: Literal["combined", "abstained"]
    reason: str | None
    weights: tuple[tuple[str, float], ...]
    state_id: str

    def __post_init__(self) -> None:
        _date(self.date, "decision.date")
        _identifier(self.symbol, "decision.symbol")
        _identifier(self.method, "decision.method")
        if self.method not in _GOVERNED_METHODS:
            raise EnsembleContractError("decision.method is unsupported")
        object.__setattr__(self, "prediction", _finite(self.prediction, "decision.prediction"))
        object.__setattr__(
            self,
            "uncertainty",
            _optional_non_negative(self.uncertainty, "decision.uncertainty"),
        )
        if self.status not in {"combined", "abstained"}:
            raise EnsembleContractError("decision status must be 'combined' or 'abstained'")
        if self.status == "combined" and self.reason is not None:
            raise EnsembleContractError("combined decision cannot carry a fallback reason")
        if self.status == "abstained" and not self.reason:
            raise EnsembleContractError("abstained decision requires a reason")
        names = [name for name, _ in self.weights]
        values = np.asarray([value for _, value in self.weights], dtype=float)
        if len(names) != len(set(names)) or names != sorted(names) or not np.isfinite(values).all():
            raise EnsembleContractError("decision weights must be canonical, unique, and finite")
        for name in names:
            _identifier(name, "decision.weight.expert")
        if self.status == "combined" and not names:
            raise EnsembleContractError("combined decision requires expert weights")
        if self.status == "abstained" and np.any(values != 0.0):
            raise EnsembleContractError("abstained decision must have zero active weights")
        if (
            not isinstance(self.state_id, str)
            or len(self.state_id) != 64
            or any(character not in "0123456789abcdef" for character in self.state_id)
        ):
            raise EnsembleContractError("decision.state_id must be a SHA-256 identity")

    @property
    def identity(self) -> str:
        """Stable content identity for this auditable decision."""

        return _identity(asdict(self))


__all__ = [
    "DEFAULT_MAX_DATES",
    "DEFAULT_MAX_EXPERTS",
    "DEFAULT_MAX_PREDICTION_RECORDS",
    "DEFAULT_MAX_SERIALIZED_BYTES",
    "ENSEMBLE_CONTRACT_VERSION",
    "EnsembleContractError",
    "EnsembleDecision",
    "HoldoutLeakageError",
    "IncompleteOOFError",
    "InferenceBatch",
    "InferencePrediction",
    "TemporalOOFFold",
    "TrainingOOFPanel",
    "TrainingOOFPrediction",
    "TrainingOOFTarget",
]
