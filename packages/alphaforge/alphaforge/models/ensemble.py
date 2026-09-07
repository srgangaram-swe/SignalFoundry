"""Leakage-safe ensemble policies over immutable temporal OOF predictions.

``fit_governed_ensemble`` is the primary SF-S3-MR9 boundary.  It learns only
from :class:`~alphaforge.models.ensemble_contracts.TrainingOOFPanel`; final
inference accepts a target-free :class:`~alphaforge.models.ensemble_contracts.InferenceBatch`.
Static blending, rank voting, temporal OOF stacking, Bayesian averaging,
causal dynamic weighting, and regime gates therefore share the same identity,
fallback, serialization, and resource semantics.

``EnsembleModel`` remains as a compatibility adapter for the pre-existing
feature-level registry entry.  It is intentionally limited to equal/IC
weighting; advanced policies must use the governed OOF boundary.
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

from alphaforge.models.base import AlphaModel, ModelError
from alphaforge.models.ensemble_contracts import (
    ENSEMBLE_CONTRACT_VERSION,
    EnsembleContractError,
    EnsembleDecision,
    InferenceBatch,
    TrainingOOFPanel,
)

EnsembleMethod = Literal[
    "static",
    "rank_vote",
    "stacking",
    "bayesian",
    "dynamic",
    "regime_gate",
]
_METHODS = frozenset({"static", "rank_vote", "stacking", "bayesian", "dynamic", "regime_gate"})
_EPSILON = 1e-12
_DEFAULT_MAX_SERIALIZED_AUDITS = 10_000
_ABSOLUTE_MAX_AUDIT_RECORDS = 100_000
_ABSOLUTE_MAX_EXPERTS = 64


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _finite_scalar(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EnsembleContractError(f"{field} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise EnsembleContractError(f"{field} must be finite")
    return result


def _canonical_date(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise EnsembleContractError(f"{field} must be an ISO-8601 calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EnsembleContractError(f"{field} must be an ISO-8601 calendar date") from exc
    if parsed.isoformat() != value:
        raise EnsembleContractError(f"{field} must use canonical YYYY-MM-DD form")
    return value


def _safe_token(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value != value.strip()
        or not value.isascii()
        or not all(character.isalnum() or character in "._:-" for character in value)
    ):
        raise EnsembleContractError(f"{field} must be a bounded safe ASCII token")
    return value


def _normalize_non_negative(values: np.ndarray) -> np.ndarray:
    checked = np.asarray(values, dtype=np.float64)
    if checked.ndim != 1 or not np.isfinite(checked).all() or (checked < 0.0).any():
        raise EnsembleContractError("ensemble weights must be finite and non-negative")
    total = float(checked.sum())
    if total <= _EPSILON:
        raise EnsembleContractError("ensemble weights are degenerate")
    return checked / total


def _softmax(log_values: np.ndarray, *, floor: float) -> np.ndarray:
    values = np.asarray(log_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise EnsembleContractError("weight scores must be a finite vector")
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0 / len(values):
        raise EnsembleContractError("weight floor is infeasible for the expert count")
    shifted = values - float(values.max())
    probabilities = _normalize_non_negative(np.exp(np.clip(shifted, -700.0, 0.0)))
    return floor + (1.0 - len(values) * floor) * probabilities


def _weight_tuple(experts: tuple[str, ...], values: np.ndarray) -> tuple[tuple[str, float], ...]:
    return tuple((expert, float(weight)) for expert, weight in zip(experts, values, strict=True))


def _weight_array(experts: tuple[str, ...], values: tuple[tuple[str, float], ...]) -> np.ndarray:
    if len(values) != len(experts) or len({name for name, _ in values}) != len(values):
        raise EnsembleContractError("serialized weights do not match the expert set")
    mapping = dict(values)
    if set(mapping) != set(experts):
        raise EnsembleContractError("serialized weights do not match the expert set")
    return np.asarray([mapping[expert] for expert in experts], dtype=np.float64)


def _canonical_weight_array(
    experts: tuple[str, ...],
    values: tuple[tuple[str, float], ...],
    *,
    field: str,
) -> np.ndarray:
    try:
        names = tuple(name for name, _ in values)
    except (TypeError, ValueError) as exc:
        raise EnsembleContractError(f"{field} is malformed") from exc
    if names != experts:
        raise EnsembleContractError(f"{field} must use canonical expert order")
    try:
        return _weight_array(experts, values)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, EnsembleContractError):
            raise
        raise EnsembleContractError(f"{field} is malformed") from exc


@dataclass(frozen=True)
class GovernedEnsembleConfig:
    """Validated policy and resource settings for one ensemble candidate."""

    method: EnsembleMethod
    experts: tuple[str, ...]
    static_weights: tuple[tuple[str, float], ...] = ()
    ridge_penalty: float = 1.0
    bayesian_temperature: float = 1.0
    dynamic_decay: float = 0.94
    dynamic_temperature: float = 1.0
    min_weight: float = 1e-6
    regime_threshold: float = 0.5
    regime_min_confidence: float = 0.1
    min_regime_rows: int = 16
    fallback_prediction: float = 0.0
    max_audit_records: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in _METHODS:
            raise EnsembleContractError(f"unsupported ensemble method {self.method!r}")
        if any(not isinstance(expert, str) for expert in self.experts):
            raise EnsembleContractError("experts must contain safe ASCII identifiers")
        experts = tuple(sorted(self.experts))
        if (
            not experts
            or len(experts) > _ABSOLUTE_MAX_EXPERTS
            or len(experts) != len(set(experts))
            or any(
                not expert
                or len(expert) > 128
                or expert != expert.strip()
                or not expert.isascii()
                or not all(character.isalnum() or character in "._-" for character in expert)
                for expert in experts
            )
        ):
            raise EnsembleContractError("experts must contain 1..64 unique safe ASCII identifiers")
        object.__setattr__(self, "experts", experts)
        finite_positive = {
            "ridge_penalty": self.ridge_penalty,
            "bayesian_temperature": self.bayesian_temperature,
            "dynamic_temperature": self.dynamic_temperature,
        }
        for field, value in finite_positive.items():
            checked = _finite_scalar(value, field)
            if checked <= 0.0:
                raise EnsembleContractError(f"{field} must be finite and positive")
            object.__setattr__(self, field, checked)
        dynamic_decay = _finite_scalar(self.dynamic_decay, "dynamic_decay")
        if not 0.0 <= dynamic_decay < 1.0:
            raise EnsembleContractError("dynamic_decay must be in [0, 1)")
        object.__setattr__(self, "dynamic_decay", dynamic_decay)
        min_weight = _finite_scalar(self.min_weight, "min_weight")
        if not 0.0 <= min_weight <= 1.0 / len(experts):
            raise EnsembleContractError("min_weight must be in [0, 1 / number of experts]")
        object.__setattr__(self, "min_weight", min_weight)
        regime_threshold = _finite_scalar(self.regime_threshold, "regime_threshold")
        if not 0.0 < regime_threshold < 1.0:
            raise EnsembleContractError("regime_threshold must be in (0, 1)")
        object.__setattr__(self, "regime_threshold", regime_threshold)
        regime_min_confidence = _finite_scalar(self.regime_min_confidence, "regime_min_confidence")
        if not 0.0 <= regime_min_confidence < 0.5:
            raise EnsembleContractError("regime_min_confidence must be in [0, 0.5)")
        object.__setattr__(self, "regime_min_confidence", regime_min_confidence)
        if (
            isinstance(self.min_regime_rows, bool)
            or not isinstance(self.min_regime_rows, Integral)
            or self.min_regime_rows <= 0
        ):
            raise EnsembleContractError("min_regime_rows must be a positive integer")
        object.__setattr__(self, "min_regime_rows", int(self.min_regime_rows))
        object.__setattr__(
            self,
            "fallback_prediction",
            _finite_scalar(self.fallback_prediction, "fallback_prediction"),
        )
        if (
            isinstance(self.max_audit_records, bool)
            or not isinstance(self.max_audit_records, Integral)
            or self.max_audit_records <= 0
            or self.max_audit_records > _ABSOLUTE_MAX_AUDIT_RECORDS
        ):
            raise EnsembleContractError(
                f"max_audit_records must be in [1, {_ABSOLUTE_MAX_AUDIT_RECORDS}]"
            )
        object.__setattr__(self, "max_audit_records", int(self.max_audit_records))
        if self.method == "static":
            if not self.static_weights:
                equal = np.full(len(experts), 1.0 / len(experts))
                object.__setattr__(self, "static_weights", _weight_tuple(experts, equal))
            else:
                object.__setattr__(
                    self,
                    "static_weights",
                    _weight_tuple(
                        experts,
                        _normalize_non_negative(_weight_array(experts, self.static_weights)),
                    ),
                )
        elif self.static_weights:
            raise EnsembleContractError("static_weights are only valid for method='static'")

    @property
    def identity(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class EnsembleAuditRecord:
    """One deterministic training-state transition or fit decision."""

    sequence: int
    effective_after: str
    action: str
    status: Literal["updated", "fallback"]
    input_folds: tuple[int, ...]
    weights_before: tuple[tuple[str, float], ...]
    weights_after: tuple[tuple[str, float], ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, Integral)
            or self.sequence < 0
        ):
            raise EnsembleContractError("audit.sequence must be a non-negative integer")
        object.__setattr__(self, "sequence", int(self.sequence))
        _safe_token(self.effective_after, "audit.effective_after")
        _safe_token(self.action, "audit.action")
        if self.status not in {"updated", "fallback"}:
            raise EnsembleContractError("audit.status must be 'updated' or 'fallback'")
        if any(
            isinstance(fold, bool) or not isinstance(fold, Integral) or fold < 0
            for fold in self.input_folds
        ):
            raise EnsembleContractError("audit.input_folds must be non-negative integers")
        input_folds = tuple(int(fold) for fold in self.input_folds)
        if input_folds != tuple(sorted(set(input_folds))):
            raise EnsembleContractError("audit.input_folds must be sorted and unique")
        object.__setattr__(self, "input_folds", input_folds)
        try:
            before_names = tuple(name for name, _ in self.weights_before)
            after_names = tuple(name for name, _ in self.weights_after)
            before_values = np.asarray([value for _, value in self.weights_before], dtype=float)
            after_values = np.asarray([value for _, value in self.weights_after], dtype=float)
        except (TypeError, ValueError) as exc:
            raise EnsembleContractError("audit weights are malformed") from exc
        if (
            not before_names
            or any(not isinstance(name, str) for name in before_names)
            or any(not isinstance(name, str) for name in after_names)
        ):
            raise EnsembleContractError("audit weights must name a non-empty expert set")
        for name in before_names:
            _safe_token(name, "audit.weight.expert")
        if (
            len(set(before_names)) != len(before_names)
            or before_names != tuple(sorted(before_names))
            or before_names != after_names
            or not np.isfinite(before_values).all()
            or not np.isfinite(after_values).all()
        ):
            raise EnsembleContractError(
                "audit weights must be finite and use one canonical expert set"
            )
        object.__setattr__(
            self,
            "weights_before",
            tuple(
                (name, float(value))
                for name, value in zip(before_names, before_values, strict=True)
            ),
        )
        object.__setattr__(
            self,
            "weights_after",
            tuple(
                (name, float(value)) for name, value in zip(after_names, after_values, strict=True)
            ),
        )
        if self.status == "fallback":
            _safe_token(self.reason, "audit.reason")
        elif self.reason is not None:
            raise EnsembleContractError("updated audit cannot carry a fallback reason")

    @property
    def identity(self) -> str:
        """Stable content identity for one state transition."""

        return _sha256(asdict(self))


@dataclass(frozen=True)
class GovernedEnsembleState:
    """Immutable fitted state with deterministic non-executable serialization."""

    method: EnsembleMethod
    experts: tuple[str, ...]
    holdout_start: str
    training_panel_id: str
    config_id: str
    weights: tuple[tuple[str, float], ...]
    intercept: float
    feature_means: tuple[tuple[str, float], ...]
    feature_scales: tuple[tuple[str, float], ...]
    residual_variances: tuple[tuple[str, float], ...]
    target_scale: float
    regime_weights: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    regime_threshold: float
    regime_min_confidence: float
    fit_status: Literal["fitted", "fallback"]
    fallback_reason: str | None
    fallback_prediction: float
    audits: tuple[EnsembleAuditRecord, ...]
    condition_number: float

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in _METHODS:
            raise EnsembleContractError("state method is unsupported")
        if any(not isinstance(expert, str) for expert in self.experts):
            raise EnsembleContractError("state experts must be safe ASCII identifiers")
        experts = tuple(sorted(self.experts))
        if (
            experts != self.experts
            or not experts
            or len(experts) > _ABSOLUTE_MAX_EXPERTS
            or len(experts) != len(set(experts))
        ):
            raise EnsembleContractError(
                "state experts must contain 1..64 sorted unique identifiers"
            )
        for expert in experts:
            if (
                not expert
                or len(expert) > 128
                or expert != expert.strip()
                or not expert.isascii()
                or not all(character.isalnum() or character in "._-" for character in expert)
            ):
                raise EnsembleContractError("state experts must be safe ASCII identifiers")
        for digest in (self.training_panel_id, self.config_id):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise EnsembleContractError("state identities must be SHA-256 values")
        _canonical_date(self.holdout_start, "state.holdout_start")
        numeric_fields = (
            "intercept",
            "target_scale",
            "fallback_prediction",
            "condition_number",
            "regime_threshold",
            "regime_min_confidence",
        )
        for field in numeric_fields:
            object.__setattr__(self, field, _finite_scalar(getattr(self, field), f"state.{field}"))
        if self.target_scale < 0.0 or self.condition_number < 0.0:
            raise EnsembleContractError("state scales and condition number must be non-negative")
        if not 0.0 < self.regime_threshold < 1.0:
            raise EnsembleContractError("state regime_threshold must be in (0, 1)")
        if not 0.0 <= self.regime_min_confidence < 0.5:
            raise EnsembleContractError("state regime_min_confidence must be in [0, 0.5)")
        residuals = _canonical_weight_array(
            self.experts,
            self.residual_variances,
            field="state.residual_variances",
        )
        if not np.isfinite(residuals).all() or (residuals < 0.0).any():
            raise EnsembleContractError("state residual variances must be finite and non-negative")
        means = _canonical_weight_array(
            self.experts,
            self.feature_means,
            field="state.feature_means",
        )
        scales = _canonical_weight_array(
            self.experts,
            self.feature_scales,
            field="state.feature_scales",
        )
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise EnsembleContractError("state feature transforms must be finite")
        if self.method == "stacking" and (scales <= 0.0).any():
            raise EnsembleContractError("stacking feature scales must be positive")
        if self.fit_status not in {"fitted", "fallback"}:
            raise EnsembleContractError("state fit_status must be 'fitted' or 'fallback'")
        if self.fit_status == "fallback" and not self.fallback_reason:
            raise EnsembleContractError("fallback state requires a reason")
        if self.fit_status == "fitted" and self.fallback_reason is not None:
            raise EnsembleContractError("fitted state cannot carry a fallback reason")
        if self.fallback_reason is not None:
            _safe_token(self.fallback_reason, "state.fallback_reason")
        fitted_weights = _canonical_weight_array(
            self.experts,
            self.weights,
            field="state.weights",
        )
        if not np.isfinite(fitted_weights).all():
            raise EnsembleContractError("state weights must be finite")
        if self.fit_status == "fallback":
            if np.any(fitted_weights != 0.0):
                raise EnsembleContractError("fallback state must have zero active weights")
        elif self.method != "stacking" and (
            (fitted_weights < 0.0).any() or not np.isclose(fitted_weights.sum(), 1.0)
        ):
            raise EnsembleContractError(
                "non-stacking fitted weights must be non-negative and sum to one"
            )
        if self.method == "regime_gate" and self.fit_status == "fitted":
            if tuple(name for name, _ in self.regime_weights) != ("calm", "stress"):
                raise EnsembleContractError(
                    "regime gate requires canonical calm and stress weights"
                )
            for _, values in self.regime_weights:
                regime_values = _canonical_weight_array(
                    self.experts,
                    values,
                    field="state.regime_weights",
                )
                if (regime_values < 0.0).any() or not np.isclose(regime_values.sum(), 1.0):
                    raise EnsembleContractError("regime weights must be normalized")
        elif self.regime_weights:
            raise EnsembleContractError("regime weights are only valid for fitted regime gates")
        audits = tuple(self.audits)
        if any(not isinstance(record, EnsembleAuditRecord) for record in audits):
            raise EnsembleContractError("state audits must contain typed audit records")
        if len(audits) > _ABSOLUTE_MAX_AUDIT_RECORDS:
            raise EnsembleContractError("state exceeds the absolute audit-record ceiling")
        if tuple(record.sequence for record in audits) != tuple(range(len(audits))):
            raise EnsembleContractError("state audit sequences must be contiguous")
        for record in audits:
            _canonical_weight_array(
                self.experts,
                record.weights_before,
                field="state.audit.weights_before",
            )
            _canonical_weight_array(
                self.experts,
                record.weights_after,
                field="state.audit.weights_after",
            )
        if self.method in {"stacking", "dynamic"} and self.fit_status == "fitted" and not audits:
            raise EnsembleContractError("fitted adaptive state requires an audit trail")
        if self.method == "stacking":
            prior_folds: list[int] = []
            for index, record in enumerate(audits):
                if (
                    record.action != "temporal_meta_oof_fit"
                    or not record.effective_after.startswith("fold-")
                    or not record.effective_after.removeprefix("fold-").isdigit()
                ):
                    raise EnsembleContractError("stacking audit action or fold marker is invalid")
                current_fold = int(record.effective_after.removeprefix("fold-"))
                if (prior_folds and current_fold <= prior_folds[-1]) or record.input_folds != tuple(
                    prior_folds
                ):
                    raise EnsembleContractError("stacking audit fold chain is invalid")
                expected_status = "fallback" if index == 0 else "updated"
                if record.status != expected_status:
                    raise EnsembleContractError("stacking audit status sequence is invalid")
                if index == 0 and (record.input_folds or record.reason != "no_prior_meta_fold"):
                    raise EnsembleContractError("first stacking audit must name its fallback")
                if index > 0 and not record.input_folds:
                    raise EnsembleContractError("stacking update requires prior folds")
                prior_folds.append(current_fold)
        elif self.method == "dynamic":
            prior_date: str | None = None
            prior_after: tuple[tuple[str, float], ...] | None = None
            expected_initial = np.full(len(self.experts), 1.0 / len(self.experts))
            for index, record in enumerate(audits):
                if (
                    record.action != "update_after_oof_target_observed"
                    or record.status != "updated"
                    or len(record.input_folds) != 1
                ):
                    raise EnsembleContractError("dynamic audit transition is invalid")
                current_date = _canonical_date(
                    record.effective_after,
                    "dynamic audit effective_after",
                )
                if current_date >= self.holdout_start:
                    raise EnsembleContractError("dynamic audit reaches the frozen holdout")
                if prior_date is not None and current_date <= prior_date:
                    raise EnsembleContractError("dynamic audit dates must be strictly increasing")
                if prior_after is not None and record.weights_before != prior_after:
                    raise EnsembleContractError("dynamic audit weight chain is discontinuous")
                before = _weight_array(self.experts, record.weights_before)
                after = _weight_array(self.experts, record.weights_after)
                if (
                    (before < 0.0).any()
                    or (after < 0.0).any()
                    or not np.isclose(before.sum(), 1.0)
                    or not np.isclose(after.sum(), 1.0)
                ):
                    raise EnsembleContractError("dynamic audit weights must be normalized")
                if index == 0 and not np.allclose(before, expected_initial):
                    raise EnsembleContractError("dynamic audit must start from equal weights")
                prior_date = current_date
                prior_after = record.weights_after
            if audits and self.weights != audits[-1].weights_after:
                raise EnsembleContractError("dynamic state weights diverge from the audit trail")
        elif audits:
            raise EnsembleContractError("only stacking and dynamic states may contain audits")
        object.__setattr__(self, "audits", audits)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ENSEMBLE_CONTRACT_VERSION,
            "method": self.method,
            "experts": list(self.experts),
            "holdout_start": self.holdout_start,
            "training_panel_id": self.training_panel_id,
            "config_id": self.config_id,
            "weights": [list(value) for value in self.weights],
            "intercept": self.intercept,
            "feature_means": [list(value) for value in self.feature_means],
            "feature_scales": [list(value) for value in self.feature_scales],
            "residual_variances": [list(value) for value in self.residual_variances],
            "target_scale": self.target_scale,
            "regime_weights": [
                [regime, [list(value) for value in weights]]
                for regime, weights in self.regime_weights
            ],
            "regime_threshold": self.regime_threshold,
            "regime_min_confidence": self.regime_min_confidence,
            "fit_status": self.fit_status,
            "fallback_reason": self.fallback_reason,
            "fallback_prediction": self.fallback_prediction,
            "audits": [
                {
                    **asdict(record),
                    "input_folds": list(record.input_folds),
                    "weights_before": [list(value) for value in record.weights_before],
                    "weights_after": [list(value) for value in record.weights_after],
                }
                for record in self.audits
            ],
            "condition_number": self.condition_number,
        }

    @property
    def identity(self) -> str:
        """Stable fitted-state identity."""

        return _sha256(self._payload())

    def to_json(self) -> str:
        """Return a deterministic, non-executable state document."""

        envelope = {"state": self._payload(), "state_id": self.identity}
        return _canonical_json(envelope) + "\n"

    @classmethod
    def from_json(
        cls,
        payload: str | bytes,
        *,
        max_bytes: int = 16 * 1024 * 1024,
        max_audit_records: int = _DEFAULT_MAX_SERIALIZED_AUDITS,
    ) -> GovernedEnsembleState:
        """Load and integrity-check a bounded state document.

        The embedded digest detects accidental or unreviewed mutation; it is
        not a keyed authenticity proof. Callers that require provenance must
        compare ``identity`` with an independently trusted expected digest.
        """

        if not isinstance(payload, (str, bytes)):
            raise EnsembleContractError("serialized ensemble state must be text or bytes")
        encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, Integral)
            or max_bytes <= 0
            or len(encoded) > max_bytes
        ):
            raise EnsembleContractError("serialized ensemble state exceeds max_bytes")
        if (
            isinstance(max_audit_records, bool)
            or not isinstance(max_audit_records, Integral)
            or max_audit_records < 0
            or max_audit_records > _ABSOLUTE_MAX_AUDIT_RECORDS
        ):
            raise EnsembleContractError(
                f"max_audit_records must be in [0, {_ABSOLUTE_MAX_AUDIT_RECORDS}]"
            )
        try:
            envelope = json.loads(encoded)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EnsembleContractError("ensemble state is not valid UTF-8 JSON") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"state", "state_id"}:
            raise EnsembleContractError("ensemble state envelope fields mismatch")
        state = envelope["state"]
        expected = {
            "contract_version",
            "method",
            "experts",
            "holdout_start",
            "training_panel_id",
            "config_id",
            "weights",
            "intercept",
            "feature_means",
            "feature_scales",
            "residual_variances",
            "target_scale",
            "regime_weights",
            "regime_threshold",
            "regime_min_confidence",
            "fit_status",
            "fallback_reason",
            "fallback_prediction",
            "audits",
            "condition_number",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise EnsembleContractError("ensemble state fields mismatch")
        if state["contract_version"] != ENSEMBLE_CONTRACT_VERSION:
            raise EnsembleContractError("unsupported ensemble state version")
        if not isinstance(state["audits"], list) or len(state["audits"]) > max_audit_records:
            raise EnsembleContractError("serialized ensemble state exceeds max_audit_records")
        try:
            verified_state_id = _sha256(state)
        except (TypeError, ValueError) as exc:
            raise EnsembleContractError("ensemble state contains invalid numeric values") from exc
        if verified_state_id != envelope["state_id"]:
            raise EnsembleContractError("ensemble state identity mismatch")
        try:
            result = cls(
                method=state["method"],
                experts=tuple(state["experts"]),
                holdout_start=state["holdout_start"],
                training_panel_id=state["training_panel_id"],
                config_id=state["config_id"],
                weights=tuple((name, float(value)) for name, value in state["weights"]),
                intercept=float(state["intercept"]),
                feature_means=tuple((name, float(value)) for name, value in state["feature_means"]),
                feature_scales=tuple(
                    (name, float(value)) for name, value in state["feature_scales"]
                ),
                residual_variances=tuple(
                    (name, float(value)) for name, value in state["residual_variances"]
                ),
                target_scale=float(state["target_scale"]),
                regime_weights=tuple(
                    (
                        regime,
                        tuple((name, float(value)) for name, value in weights),
                    )
                    for regime, weights in state["regime_weights"]
                ),
                regime_threshold=float(state["regime_threshold"]),
                regime_min_confidence=float(state["regime_min_confidence"]),
                fit_status=state["fit_status"],
                fallback_reason=state["fallback_reason"],
                fallback_prediction=float(state["fallback_prediction"]),
                audits=tuple(
                    EnsembleAuditRecord(
                        sequence=record["sequence"],
                        effective_after=record["effective_after"],
                        action=record["action"],
                        status=record["status"],
                        input_folds=tuple(record["input_folds"]),
                        weights_before=tuple(
                            (name, float(value)) for name, value in record["weights_before"]
                        ),
                        weights_after=tuple(
                            (name, float(value)) for name, value in record["weights_after"]
                        ),
                        reason=record["reason"],
                    )
                    for record in state["audits"]
                ),
                condition_number=float(state["condition_number"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, EnsembleContractError):
                raise
            raise EnsembleContractError("malformed ensemble state records") from exc
        if result.identity != envelope["state_id"]:
            raise EnsembleContractError("ensemble state did not round-trip canonically")
        return result

    def predict(self, batch: InferenceBatch) -> tuple[EnsembleDecision, ...]:
        """Combine target-free predictions or emit explicit abstentions.

        The method is side-effect free: final-holdout predictions cannot mutate
        weights or gate state.
        """

        if any(row.date < self.holdout_start for row in batch.predictions):
            raise EnsembleContractError("inference records predate the frozen holdout boundary")
        if any(date_value < self.holdout_start for date_value, _ in batch.expected_keys):
            raise EnsembleContractError("expected inference keys predate the frozen holdout")
        by_key: dict[tuple[str, str], dict[str, Any]] = {key: {} for key in batch.expected_keys}
        for row in batch.predictions:
            by_key[(row.date, row.symbol)][row.expert] = row
        rank_values: dict[tuple[str, str, str], float] = {}
        incomplete_rank_dates: set[str] = set()
        insufficient_rank_dates: set[str] = set()
        if self.method == "rank_vote":
            dates = sorted({date_value for date_value, _ in by_key})
            for date_value in dates:
                symbols = sorted(symbol for row_date, symbol in by_key if row_date == date_value)
                if len(symbols) < 2:
                    insufficient_rank_dates.add(date_value)
                    continue
                for expert in self.experts:
                    rank_records = [by_key[(date_value, symbol)].get(expert) for symbol in symbols]
                    if any(record is None for record in rank_records):
                        incomplete_rank_dates.add(date_value)
                        continue
                    complete_rank_records = [
                        record for record in rank_records if record is not None
                    ]
                    values = pd.Series(
                        [record.prediction for record in complete_rank_records],
                        index=symbols,
                        dtype=float,
                    )
                    ranks = values.rank(method="average")
                    centered = (ranks - (len(values) + 1.0) / 2.0) / max(len(values), 1)
                    for symbol, value in centered.items():
                        rank_values[(date_value, symbol, expert)] = float(value)

        decisions: list[EnsembleDecision] = []
        state_id = self.identity
        global_weights = _weight_array(self.experts, self.weights)
        if self.method == "stacking":
            means = _weight_array(self.experts, self.feature_means)
            scales = _weight_array(self.experts, self.feature_scales)
        else:
            means = np.zeros(len(self.experts), dtype=float)
            scales = np.ones(len(self.experts), dtype=float)
        residuals = _weight_array(self.experts, self.residual_variances)
        regime_map = {name: weights for name, weights in self.regime_weights}
        for key in sorted(by_key):
            date_value, symbol = key
            current_records = by_key[key]
            reason: str | None = None
            if self.fit_status == "fallback":
                reason = f"fit_{self.fallback_reason}"
            elif self.method == "rank_vote" and date_value in insufficient_rank_dates:
                reason = "insufficient_rank_cross_section"
            elif self.method == "rank_vote" and date_value in incomplete_rank_dates:
                reason = "incomplete_date_for_rank_vote"
            else:
                missing = sorted(set(self.experts) - set(current_records))
                unknown = sorted(set(current_records) - set(self.experts))
                if missing:
                    reason = f"missing_experts:{','.join(missing)}"
                elif unknown:
                    reason = f"unknown_experts:{','.join(unknown)}"
            if reason is not None:
                decisions.append(
                    EnsembleDecision(
                        date=date_value,
                        symbol=symbol,
                        method=self.method,
                        prediction=self.fallback_prediction,
                        uncertainty=None,
                        status="abstained",
                        reason=reason,
                        weights=tuple((expert, 0.0) for expert in self.experts),
                        state_id=state_id,
                    )
                )
                continue

            rows = [current_records[expert] for expert in self.experts]
            predictions = np.asarray([row.prediction for row in rows], dtype=np.float64)
            weights = global_weights
            if self.method == "regime_gate":
                regimes = {row.regime_probability for row in rows}
                if len(regimes) != 1 or None in regimes:
                    reason = "missing_or_inconsistent_regime"
                else:
                    probability = float(next(iter(regimes)))
                    threshold = self.regime_threshold
                    confidence = self.regime_min_confidence
                    if abs(probability - threshold) < confidence:
                        reason = "uncertain_regime"
                    else:
                        regime = "stress" if probability >= threshold else "calm"
                        weights = _weight_array(self.experts, regime_map[regime])
            if reason is not None:
                decisions.append(
                    EnsembleDecision(
                        date=date_value,
                        symbol=symbol,
                        method=self.method,
                        prediction=self.fallback_prediction,
                        uncertainty=None,
                        status="abstained",
                        reason=reason,
                        weights=tuple((expert, 0.0) for expert in self.experts),
                        state_id=state_id,
                    )
                )
                continue

            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    if self.method == "stacking":
                        prediction = self.intercept + float(
                            np.dot((predictions - means) / scales, weights)
                        )
                    elif self.method == "rank_vote":
                        ranked = np.asarray(
                            [rank_values[(date_value, symbol, expert)] for expert in self.experts],
                            dtype=float,
                        )
                        prediction = self.intercept + self.target_scale * float(
                            np.dot(ranked, weights)
                        )
                    else:
                        prediction = float(np.dot(predictions, weights))
                    absolute = np.abs(weights)
                    absolute_sum = float(absolute.sum())
                    uncertainty_weights = (
                        absolute / absolute_sum
                        if absolute_sum > _EPSILON
                        else np.full(len(self.experts), 1.0 / len(self.experts))
                    )
                    supplied_uncertainty = np.asarray(
                        [0.0 if row.uncertainty is None else row.uncertainty for row in rows],
                        dtype=float,
                    )
                    supplied = np.square(supplied_uncertainty) + residuals
                    disagreement = float(
                        np.dot(
                            uncertainty_weights,
                            np.square(predictions - prediction),
                        )
                    )
                    variance = (
                        float(np.dot(np.square(uncertainty_weights), supplied)) + disagreement
                    )
            except (FloatingPointError, OverflowError) as exc:
                raise EnsembleContractError(
                    "ensemble inference arithmetic exceeded the finite numeric range"
                ) from exc
            if not np.isfinite(prediction) or not np.isfinite(variance) or variance < 0.0:
                raise EnsembleContractError(
                    "ensemble inference arithmetic produced an invalid result"
                )
            decisions.append(
                EnsembleDecision(
                    date=date_value,
                    symbol=symbol,
                    method=self.method,
                    prediction=prediction,
                    uncertainty=float(np.sqrt(max(variance, 0.0))),
                    status="combined",
                    reason=None,
                    weights=_weight_tuple(self.experts, weights),
                    state_id=state_id,
                )
            )
        return tuple(decisions)


def _fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            means = matrix.mean(axis=0)
            scales = matrix.std(axis=0)
            scales = np.where(scales <= _EPSILON, 1.0, scales)
            normalized = (matrix - means) / scales
            centered_target = target - float(target.mean())
            design = normalized.T @ normalized + penalty * np.eye(normalized.shape[1])
            coefficients = np.linalg.lstsq(
                design,
                normalized.T @ centered_target,
                rcond=None,
            )[0]
            condition = float(np.linalg.cond(design))
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise EnsembleContractError("stacking fit arithmetic is degenerate") from exc
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(coefficients).all()
        or not np.isfinite(condition)
    ):
        raise EnsembleContractError("stacking fit produced degenerate weights")
    return coefficients, float(target.mean()), means, scales, condition


def _performance_weights(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    temperature: float,
    floor: float,
) -> np.ndarray:
    mse = _mean_squared_residuals(matrix, target)
    scale = max(float(np.median(mse)), _EPSILON)
    return _softmax(-mse / (scale * temperature), floor=floor)


def _squared_residuals(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return finite squared residuals or fail at the numeric boundary."""

    try:
        with np.errstate(over="raise", invalid="raise"):
            squared = np.square(matrix - target[:, None])
    except (FloatingPointError, OverflowError) as exc:
        raise EnsembleContractError(
            "ensemble training arithmetic exceeded the finite numeric range"
        ) from exc
    if not np.isfinite(squared).all():
        raise EnsembleContractError("ensemble training arithmetic produced non-finite residuals")
    return squared


def _mean_squared_residuals(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.mean(_squared_residuals(matrix, target), axis=0)
    except (FloatingPointError, OverflowError) as exc:
        raise EnsembleContractError(
            "ensemble training loss exceeded the finite numeric range"
        ) from exc
    if not np.isfinite(result).all():
        raise EnsembleContractError("ensemble training loss is non-finite")
    return result


def _fallback_state(
    panel: TrainingOOFPanel,
    config: GovernedEnsembleConfig,
    *,
    reason: str,
    residuals: np.ndarray,
    target_scale: float,
    condition_number: float,
    audits: tuple[EnsembleAuditRecord, ...] = (),
) -> GovernedEnsembleState:
    zeros = np.zeros(len(config.experts), dtype=float)
    ones = np.ones(len(config.experts), dtype=float)
    return GovernedEnsembleState(
        method=config.method,
        experts=config.experts,
        holdout_start=panel.holdout_start,
        training_panel_id=panel.identity,
        config_id=config.identity,
        weights=_weight_tuple(config.experts, zeros),
        intercept=0.0,
        feature_means=_weight_tuple(config.experts, zeros),
        feature_scales=_weight_tuple(config.experts, ones),
        residual_variances=_weight_tuple(config.experts, residuals),
        target_scale=target_scale,
        regime_weights=(),
        regime_threshold=config.regime_threshold,
        regime_min_confidence=config.regime_min_confidence,
        fit_status="fallback",
        fallback_reason=reason,
        fallback_prediction=config.fallback_prediction,
        audits=audits,
        condition_number=condition_number,
    )


def fit_governed_ensemble(
    panel: TrainingOOFPanel,
    config: GovernedEnsembleConfig,
) -> GovernedEnsembleState:
    """Fit one deterministic ensemble using training OOF predictions only."""

    if panel.expected_experts != config.experts:
        raise EnsembleContractError("panel and config expert identities differ")
    frame = panel.wide_frame()
    matrix = frame.loc[:, list(config.experts)].to_numpy(dtype=np.float64)
    target = frame["target"].to_numpy(dtype=np.float64)
    residuals = _mean_squared_residuals(matrix, target)
    try:
        with np.errstate(over="raise", invalid="raise"):
            target_scale = float(np.std(target))
    except (FloatingPointError, OverflowError) as exc:
        raise EnsembleContractError(
            "ensemble training target scale exceeded the finite numeric range"
        ) from exc
    if not np.isfinite(target_scale):
        raise EnsembleContractError("ensemble training target scale is non-finite")
    base_equal = np.full(len(config.experts), 1.0 / len(config.experts))
    means = np.zeros(len(config.experts))
    scales = np.ones(len(config.experts))
    weights = base_equal
    intercept = 0.0
    condition_number = 1.0
    audits: list[EnsembleAuditRecord] = []
    regime_weights: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()

    if config.method != "static" and target_scale <= _EPSILON:
        return _fallback_state(
            panel,
            config,
            reason="degenerate_training_target",
            residuals=residuals,
            target_scale=target_scale,
            condition_number=condition_number,
        )

    if config.method == "static":
        weights = _weight_array(config.experts, config.static_weights)
    elif config.method == "rank_vote":
        weights = base_equal
        intercept = float(np.mean(target))
    elif config.method == "stacking":
        if len(panel.folds) > config.max_audit_records:
            raise EnsembleContractError("stacking folds exceed max_audit_records")
        # Audit meta-prediction fits use only folds strictly earlier than the
        # fold being predicted.  The first fold is an explicit fallback.
        for fold_id in sorted(frame["fold_id"].unique()):
            prior = frame["fold_id"] < fold_id
            empty_weights = _weight_tuple(config.experts, np.zeros(len(config.experts)))
            if not prior.any():
                audits.append(
                    EnsembleAuditRecord(
                        sequence=len(audits),
                        effective_after=f"fold-{fold_id}",
                        action="temporal_meta_oof_fit",
                        status="fallback",
                        input_folds=(),
                        weights_before=empty_weights,
                        weights_after=empty_weights,
                        reason="no_prior_meta_fold",
                    )
                )
            else:
                fold_weights, _, _, _, _ = _fit_ridge(
                    matrix[prior],
                    target[prior],
                    config.ridge_penalty,
                )
                audits.append(
                    EnsembleAuditRecord(
                        sequence=len(audits),
                        effective_after=f"fold-{fold_id}",
                        action="temporal_meta_oof_fit",
                        status="updated",
                        input_folds=tuple(
                            int(value) for value in sorted(frame.loc[prior, "fold_id"].unique())
                        ),
                        weights_before=empty_weights,
                        weights_after=_weight_tuple(config.experts, fold_weights),
                    )
                )
        weights, intercept, means, scales, condition_number = _fit_ridge(
            matrix,
            target,
            config.ridge_penalty,
        )
    elif config.method == "bayesian":
        mse = np.maximum(residuals, _EPSILON)
        log_evidence = -0.5 * len(target) * np.log(mse)
        weights = _softmax(
            log_evidence / config.bayesian_temperature,
            floor=config.min_weight,
        )
    elif config.method == "dynamic":
        losses = np.zeros(len(config.experts), dtype=float)
        weights = base_equal
        for date_value in sorted(frame["date"].unique()):
            date_mask = frame["date"] == date_value
            before_weights = weights.copy()
            daily_mse = _mean_squared_residuals(matrix[date_mask], target[date_mask])
            try:
                with np.errstate(over="raise", invalid="raise"):
                    losses = (
                        config.dynamic_decay * losses + (1.0 - config.dynamic_decay) * daily_mse
                    )
            except (FloatingPointError, OverflowError) as exc:
                raise EnsembleContractError(
                    "dynamic ensemble loss exceeded the finite numeric range"
                ) from exc
            if not np.isfinite(losses).all():
                raise EnsembleContractError("dynamic ensemble loss is non-finite")
            scale = max(float(np.median(losses)), _EPSILON)
            weights = _softmax(
                -losses / (scale * config.dynamic_temperature),
                floor=config.min_weight,
            )
            if len(audits) >= config.max_audit_records:
                raise EnsembleContractError("dynamic weighting exceeds max_audit_records")
            audits.append(
                EnsembleAuditRecord(
                    sequence=len(audits),
                    effective_after=str(date_value),
                    action="update_after_oof_target_observed",
                    status="updated",
                    input_folds=tuple(
                        int(value) for value in sorted(frame.loc[date_mask, "fold_id"].unique())
                    ),
                    weights_before=_weight_tuple(config.experts, before_weights),
                    weights_after=_weight_tuple(config.experts, weights),
                )
            )
    elif config.method == "regime_gate":
        regime = frame["regime_probability"].to_numpy(dtype=float)
        if not np.isfinite(regime).all():
            return _fallback_state(
                panel,
                config,
                reason="missing_training_regime",
                residuals=residuals,
                target_scale=target_scale,
                condition_number=condition_number,
            )
        calm_boundary = config.regime_threshold - config.regime_min_confidence
        calm = (
            regime < config.regime_threshold
            if config.regime_min_confidence == 0.0
            else regime <= calm_boundary
        )
        stress = regime >= config.regime_threshold + config.regime_min_confidence
        if int(calm.sum()) < config.min_regime_rows or int(stress.sum()) < config.min_regime_rows:
            return _fallback_state(
                panel,
                config,
                reason="insufficient_confident_regime_rows",
                residuals=residuals,
                target_scale=target_scale,
                condition_number=condition_number,
            )
        calm_weights = _performance_weights(
            matrix[calm],
            target[calm],
            temperature=config.bayesian_temperature,
            floor=config.min_weight,
        )
        stress_weights = _performance_weights(
            matrix[stress],
            target[stress],
            temperature=config.bayesian_temperature,
            floor=config.min_weight,
        )
        weights = _performance_weights(
            matrix,
            target,
            temperature=config.bayesian_temperature,
            floor=config.min_weight,
        )
        regime_weights = (
            ("calm", _weight_tuple(config.experts, calm_weights)),
            ("stress", _weight_tuple(config.experts, stress_weights)),
        )
        # Identity transforms keep the uniform state schema explicit.
        means = np.zeros(len(config.experts))
        scales = np.ones(len(config.experts))

    if len(audits) > config.max_audit_records:
        raise EnsembleContractError("ensemble fit exceeds max_audit_records")
    feature_means = _weight_tuple(config.experts, means)
    feature_scales = _weight_tuple(config.experts, scales)
    return GovernedEnsembleState(
        method=config.method,
        experts=config.experts,
        holdout_start=panel.holdout_start,
        training_panel_id=panel.identity,
        config_id=config.identity,
        weights=_weight_tuple(config.experts, weights),
        intercept=intercept,
        feature_means=feature_means,
        feature_scales=feature_scales,
        residual_variances=_weight_tuple(config.experts, residuals),
        target_scale=target_scale,
        regime_weights=regime_weights,
        regime_threshold=config.regime_threshold,
        regime_min_confidence=config.regime_min_confidence,
        fit_status="fitted",
        fallback_reason=None,
        fallback_prediction=config.fallback_prediction,
        audits=tuple(audits),
        condition_number=condition_number,
    )


def _rank_ic(y: pd.Series, pred: np.ndarray) -> float:
    frame = pd.DataFrame({"y": y.to_numpy(), "p": pred}).dropna()
    if len(frame) < 10 or frame["p"].std() == 0 or frame["y"].std() == 0:
        return 0.0
    return float(frame["y"].rank().corr(frame["p"].rank()))


class EnsembleModel(AlphaModel):
    """Compatibility adapter for the historic ``ensemble`` registry name.

    Equal weighting remains unchanged.  IC weighting is deterministic, exposes
    any insufficient-data fallback through ``weighting_status_``, and never
    suppresses member failures.  New stacking/gating research must use
    :func:`fit_governed_ensemble`.
    """

    name = "ensemble"

    def __init__(
        self,
        members: list[AlphaModel],
        weights: list[float] | None = None,
        weighting: str = "equal",
        val_fraction: float = 0.2,
        min_weight: float = 0.05,
    ):
        if not members:
            raise ValueError("ensemble needs at least one member")
        if weighting not in {"equal", "ic"}:
            raise ValueError("weighting must be 'equal' or 'ic'")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if not np.isfinite(min_weight) or min_weight < 0.0:
            raise ValueError("min_weight must be finite and non-negative")
        self.members = tuple(members)
        self.weighting = weighting
        self.val_fraction = val_fraction
        self.min_weight = min_weight
        raw = np.asarray(weights if weights is not None else [1.0] * len(members), dtype=float)
        if len(raw) != len(members):
            raise ValueError("weights length must match members")
        self.weights = _normalize_non_negative(raw)
        self.member_ics_: list[float] | None = None
        self.weighting_status_: str = "configured"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        if self.weighting == "ic":
            self._fit_ic_weights(X, y)
        for member in self.members:
            member.fit(X, y)
        return self

    def _fit_ic_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        n_val = max(10, int(len(X) * self.val_fraction))
        if len(X) - n_val < 10:
            self.member_ics_ = [0.0] * len(self.members)
            self.weights = np.full(len(self.members), 1.0 / len(self.members))
            self.weighting_status_ = "fallback_insufficient_rows"
            return
        X_train, y_train = X.iloc[:-n_val], y.iloc[:-n_val]
        X_validation, y_validation = X.iloc[-n_val:], y.iloc[-n_val:]
        correlations: list[float] = []
        for member in self.members:
            # A member failure is a model failure, not a zero-skill estimate.
            member.fit(X_train, y_train)
            correlations.append(_rank_ic(y_validation, np.asarray(member.predict(X_validation))))
        self.member_ics_ = correlations
        raw = np.clip(np.asarray(correlations, dtype=float), 0.0, None) + self.min_weight
        try:
            self.weights = _normalize_non_negative(raw)
            self.weighting_status_ = "fitted"
        except EnsembleContractError as exc:  # explicit compatibility fallback
            self.weights = np.full(len(self.members), 1.0 / len(self.members))
            self.weighting_status_ = f"fallback_{exc}"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.column_stack([member.predict(X) for member in self.members])
        if predictions.shape[1] != len(self.weights):
            raise ModelError("ensemble member prediction count changed after fit")
        return predictions @ self.weights

    def feature_importance(self) -> pd.Series | None:
        weighted: list[pd.Series] = []
        for weight, member in zip(self.weights, self.members, strict=True):
            importance = member.feature_importance()
            if importance is not None and float(importance.sum()) > 0.0:
                weighted.append(weight * importance / float(importance.sum()))
        if not weighted:
            return None
        return pd.concat(weighted, axis=1).fillna(0.0).sum(axis=1).sort_values(ascending=False)


__all__ = [
    "EnsembleAuditRecord",
    "EnsembleMethod",
    "EnsembleModel",
    "GovernedEnsembleConfig",
    "GovernedEnsembleState",
    "fit_governed_ensemble",
]
