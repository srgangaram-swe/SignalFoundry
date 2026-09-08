"""Append-only research governance and complete-family statistical correction.

This module is the trust boundary between a frozen research plan and observed
trial results.  A :class:`ResearchLedger` has one writer, records every state
transition in a hash chain, and maintains an atomic head receipt that detects
truncation, reordering, and mutation before another event can be appended.

The ledger is an audit mechanism, not a hostile-storage signature scheme.
An attacker able to rewrite both the ledger and its receipt can manufacture a
new chain; durable deployments must anchor receipts in an external immutable
store.  No API in this module deletes or overwrites an accepted record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

GOVERNANCE_SCHEMA_VERSION = "1.0.0"
LEDGER_FILE_NAME = "research_ledger.jsonl"
HEAD_FILE_NAME = "research_ledger.head.json"
LOCK_FILE_NAME = ".research_ledger.lock"
ROOT_TRIAL_ID = "ROOT"
GENESIS_HASH = "0" * 64
DEFAULT_MAX_LEDGER_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RECORDS = 100_000

CorrectionMethod = Literal["holm_bonferroni", "benjamini_hochberg"]
TrialStatus = Literal[
    "REGISTERED",
    "STARTED",
    "INTERRUPTED",
    "RESUMED",
    "SUCCEEDED",
    "FAILED",
]
KillOperator = Literal["lt", "le", "gt", "ge"]

_RECORD_FIELDS = {
    "schema_version",
    "sequence",
    "event_type",
    "trial_id",
    "occurred_at",
    "payload",
    "previous_hash",
    "record_hash",
}
_HEAD_FIELDS = {
    "schema_version",
    "record_count",
    "head_hash",
    "ledger_sha256",
}
_TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED"})
_EVENT_TO_STATUS: dict[str, TrialStatus] = {
    "TRIAL_REGISTERED": "REGISTERED",
    "TRIAL_STARTED": "STARTED",
    "TRIAL_INTERRUPTED": "INTERRUPTED",
    "TRIAL_RESUMED": "RESUMED",
    "TRIAL_SUCCEEDED": "SUCCEEDED",
    "TRIAL_FAILED": "FAILED",
}
_ALLOWED_TRANSITIONS: dict[str | None, frozenset[TrialStatus]] = {
    None: frozenset({"REGISTERED"}),
    "REGISTERED": frozenset({"STARTED", "FAILED"}),
    "STARTED": frozenset({"INTERRUPTED", "SUCCEEDED", "FAILED"}),
    "INTERRUPTED": frozenset({"RESUMED"}),
    "RESUMED": frozenset({"INTERRUPTED", "SUCCEEDED", "FAILED"}),
    "SUCCEEDED": frozenset(),
    "FAILED": frozenset(),
}


class ResearchGovernanceError(ValueError):
    """Raised when research evidence violates the frozen governance contract."""


class LedgerIntegrityError(ResearchGovernanceError):
    """Raised when the append-only ledger or its receipt fails verification."""


class DuplicateTrialError(ResearchGovernanceError):
    """Raised after an attempted duplicate registration has been audited."""


class IncompleteTrialFamilyError(ResearchGovernanceError):
    """Raised after an incomplete or cherry-picked family evaluation is audited."""


def _require_identifier(value: str, *, field: str, allow_root: bool = False) -> str:
    if allow_root and value == ROOT_TRIAL_ID:
        return value
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value != value.strip()
        or not value.isascii()
        or not all(character.isalnum() or character in "._-" for character in value)
    ):
        raise ResearchGovernanceError(
            f"{field} must be a non-empty ASCII identifier using letters, numbers, '.', '_', or '-'"
        )
    return value


def _parse_utc(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ResearchGovernanceError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchGovernanceError(f"{field} must be an ISO-8601 UTC timestamp") from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ResearchGovernanceError(f"{field} must be UTC")
    return parsed


def _freeze_json(value: Any, *, field: str) -> Any:
    """Validate and deeply freeze a bounded JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchGovernanceError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > 10_000:
            raise ResearchGovernanceError(f"{field} mapping is too large")
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or key in frozen:
                raise ResearchGovernanceError(f"{field} keys must be unique non-empty strings")
            frozen[key] = _freeze_json(item, field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 100_000:
            raise ResearchGovernanceError(f"{field} sequence is too large")
        return tuple(
            _freeze_json(item, field=f"{field}[{index}]") for index, item in enumerate(value)
        )
    raise ResearchGovernanceError(
        f"{field} contains a non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResearchGovernanceError("value is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class TrialSpec:
    """One eligible candidate and its explicit parent lineage."""

    trial_id: str
    parent_trial_id: str
    candidate: str
    configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_identifier(self.trial_id, field="trial_id")
        _require_identifier(self.parent_trial_id, field="parent_trial_id", allow_root=True)
        _require_identifier(self.candidate, field="candidate")
        if self.trial_id == self.parent_trial_id:
            raise ResearchGovernanceError("a trial cannot be its own parent")
        object.__setattr__(
            self,
            "configuration",
            _freeze_json(self.configuration, field=f"trial[{self.trial_id}].configuration"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "parent_trial_id": self.parent_trial_id,
            "candidate": self.candidate,
            "configuration": _thaw_json(self.configuration),
        }


@dataclass(frozen=True)
class MultipleTestingPolicy:
    """Frozen correction method and complete eligible-family assumptions."""

    method: CorrectionMethod
    alpha: float
    family_size: int
    assumptions: tuple[str, ...]
    failed_trial_p_value: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in {"holm_bonferroni", "benjamini_hochberg"}:
            raise ResearchGovernanceError("unsupported multiple-testing correction")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ResearchGovernanceError("correction alpha must be finite and in (0, 1)")
        if isinstance(self.family_size, bool) or self.family_size < 1:
            raise ResearchGovernanceError("family_size must be a positive integer")
        assumptions = tuple(self.assumptions)
        if (
            not assumptions
            or len(assumptions) != len(set(assumptions))
            or any(not item or item != item.strip() for item in assumptions)
        ):
            raise ResearchGovernanceError(
                "correction assumptions must be unique non-empty declarations"
            )
        if self.failed_trial_p_value != 1.0:
            raise ResearchGovernanceError("failed trials must receive conservative p-value 1.0")
        object.__setattr__(self, "assumptions", assumptions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "family_size": self.family_size,
            "assumptions": list(self.assumptions),
            "failed_trial_p_value": self.failed_trial_p_value,
        }


@dataclass(frozen=True)
class KillCriterion:
    """A predeclared per-trial condition that rejects a candidate."""

    name: str
    metric: str
    operator: KillOperator
    threshold: float

    def __post_init__(self) -> None:
        _require_identifier(self.name, field="kill criterion name")
        _require_identifier(self.metric, field="kill criterion metric")
        if self.operator not in {"lt", "le", "gt", "ge"}:
            raise ResearchGovernanceError("unsupported kill-criterion operator")
        if not math.isfinite(self.threshold):
            raise ResearchGovernanceError("kill-criterion threshold must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
        }

    def triggered(self, value: float) -> bool:
        if not math.isfinite(value):
            return True
        return {
            "lt": value < self.threshold,
            "le": value <= self.threshold,
            "gt": value > self.threshold,
            "ge": value >= self.threshold,
        }[self.operator]


@dataclass(frozen=True)
class FrozenResearchPlan:
    """Deeply immutable hypothesis, experiment, rejection, and lineage plan."""

    hypothesis: str
    mechanism: str
    dataset_id: str
    features: tuple[str, ...]
    label: str
    test_plan: Mapping[str, Any]
    validation: Mapping[str, Any]
    costs: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    rejection_thresholds: Mapping[str, Any]
    trials: tuple[TrialSpec, ...]
    correction: MultipleTestingPolicy
    kill_criteria: tuple[KillCriterion, ...]
    frozen_at: str
    schema_version: str = GOVERNANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GOVERNANCE_SCHEMA_VERSION:
            raise ResearchGovernanceError("unsupported research-plan schema version")
        for field_name in ("hypothesis", "mechanism", "label"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ResearchGovernanceError(f"{field_name} must be non-empty and trimmed")
        if len(self.dataset_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.dataset_id
        ):
            raise ResearchGovernanceError("dataset_id must be a lowercase SHA-256 identity")
        features = tuple(self.features)
        if (
            not features
            or len(features) != len(set(features))
            or any(not feature or feature != feature.strip() for feature in features)
        ):
            raise ResearchGovernanceError("features must be unique non-empty declarations")
        trials = tuple(self.trials)
        trial_ids = [trial.trial_id for trial in trials]
        if not trials or len(trial_ids) != len(set(trial_ids)):
            raise ResearchGovernanceError("trials must contain unique eligible trial IDs")
        if self.correction.family_size != len(trials):
            raise ResearchGovernanceError(
                "correction family_size must equal the complete eligible trial family"
            )
        criteria = tuple(self.kill_criteria)
        criterion_names = [criterion.name for criterion in criteria]
        if not criteria or len(criterion_names) != len(set(criterion_names)):
            raise ResearchGovernanceError("kill criteria must contain unique declarations")
        _validate_lineage(trials)
        _parse_utc(self.frozen_at, field="frozen_at")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "kill_criteria", criteria)
        for field_name in (
            "test_plan",
            "validation",
            "costs",
            "uncertainty",
            "rejection_thresholds",
        ):
            value = getattr(self, field_name)
            if not value:
                raise ResearchGovernanceError(f"{field_name} must be explicitly declared")
            object.__setattr__(
                self,
                field_name,
                _freeze_json(value, field=field_name),
            )

    @property
    def plan_hash(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "dataset_id": self.dataset_id,
            "features": list(self.features),
            "label": self.label,
            "test_plan": _thaw_json(self.test_plan),
            "validation": _thaw_json(self.validation),
            "costs": _thaw_json(self.costs),
            "uncertainty": _thaw_json(self.uncertainty),
            "rejection_thresholds": _thaw_json(self.rejection_thresholds),
            "trials": [trial.to_dict() for trial in self.trials],
            "correction": self.correction.to_dict(),
            "kill_criteria": [criterion.to_dict() for criterion in self.kill_criteria],
            "frozen_at": self.frozen_at,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> FrozenResearchPlan:
        """Validate an untrusted canonical plan document without coercion."""

        fields = {
            "schema_version",
            "hypothesis",
            "mechanism",
            "dataset_id",
            "features",
            "label",
            "test_plan",
            "validation",
            "costs",
            "uncertainty",
            "rejection_thresholds",
            "trials",
            "correction",
            "kill_criteria",
            "frozen_at",
        }
        if not isinstance(document, Mapping) or set(document) != fields:
            raise ResearchGovernanceError("frozen research-plan fields mismatch")
        features = document["features"]
        trials = document["trials"]
        correction = document["correction"]
        criteria = document["kill_criteria"]
        if (
            not isinstance(features, list)
            or not isinstance(trials, list)
            or not isinstance(correction, dict)
            or not isinstance(criteria, list)
        ):
            raise ResearchGovernanceError("frozen research-plan collection types mismatch")
        correction_fields = {
            "method",
            "alpha",
            "family_size",
            "assumptions",
            "failed_trial_p_value",
        }
        if set(correction) != correction_fields or not isinstance(correction["assumptions"], list):
            raise ResearchGovernanceError("multiple-testing policy fields mismatch")
        trial_fields = {"trial_id", "parent_trial_id", "candidate", "configuration"}
        criterion_fields = {"name", "metric", "operator", "threshold"}
        if any(not isinstance(item, dict) or set(item) != trial_fields for item in trials):
            raise ResearchGovernanceError("trial specification fields mismatch")
        if any(not isinstance(item, dict) or set(item) != criterion_fields for item in criteria):
            raise ResearchGovernanceError("kill-criterion fields mismatch")
        try:
            return cls(
                schema_version=document["schema_version"],
                hypothesis=document["hypothesis"],
                mechanism=document["mechanism"],
                dataset_id=document["dataset_id"],
                features=tuple(features),
                label=document["label"],
                test_plan=document["test_plan"],
                validation=document["validation"],
                costs=document["costs"],
                uncertainty=document["uncertainty"],
                rejection_thresholds=document["rejection_thresholds"],
                trials=tuple(
                    TrialSpec(
                        trial_id=item["trial_id"],
                        parent_trial_id=item["parent_trial_id"],
                        candidate=item["candidate"],
                        configuration=item["configuration"],
                    )
                    for item in trials
                ),
                correction=MultipleTestingPolicy(
                    method=correction["method"],
                    alpha=correction["alpha"],
                    family_size=correction["family_size"],
                    assumptions=tuple(correction["assumptions"]),
                    failed_trial_p_value=correction["failed_trial_p_value"],
                ),
                kill_criteria=tuple(
                    KillCriterion(
                        name=item["name"],
                        metric=item["metric"],
                        operator=item["operator"],
                        threshold=item["threshold"],
                    )
                    for item in criteria
                ),
                frozen_at=document["frozen_at"],
            )
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            if isinstance(exc, ResearchGovernanceError):
                raise
            raise ResearchGovernanceError("frozen research plan is malformed") from exc


def _validate_lineage(trials: tuple[TrialSpec, ...]) -> None:
    parents = {trial.trial_id: trial.parent_trial_id for trial in trials}
    for trial_id, parent in parents.items():
        if parent != ROOT_TRIAL_ID and parent not in parents:
            raise ResearchGovernanceError(f"trial {trial_id!r} has unknown parent {parent!r}")
        visited: set[str] = set()
        cursor = trial_id
        while cursor != ROOT_TRIAL_ID:
            if cursor in visited:
                raise ResearchGovernanceError("trial parent lineage contains a cycle")
            visited.add(cursor)
            cursor = parents[cursor]


@dataclass(frozen=True)
class FamilyEvaluation:
    """Complete-family corrected evidence and predeclared kill decisions."""

    method: CorrectionMethod
    alpha: float
    raw_p_values: Mapping[str, float]
    adjusted_p_values: Mapping[str, float]
    rejected: Mapping[str, bool]
    killed: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        for field_name in ("raw_p_values", "adjusted_p_values", "rejected", "killed"):
            object.__setattr__(
                self,
                field_name,
                _freeze_json(getattr(self, field_name), field=field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "raw_p_values": _thaw_json(self.raw_p_values),
            "adjusted_p_values": _thaw_json(self.adjusted_p_values),
            "rejected": _thaw_json(self.rejected),
            "killed": _thaw_json(self.killed),
        }


def adjust_p_values(
    p_values: Mapping[str, float],
    *,
    policy: MultipleTestingPolicy,
    eligible_trial_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, bool]]:
    """Correct exactly one complete eligible family.

    Missing, extra, duplicate, non-finite, or out-of-range values fail closed.
    Holm controls family-wise error under arbitrary dependence. Benjamini-
    Hochberg controls false discovery rate only under the assumptions declared
    in the frozen policy.
    """

    eligible = tuple(eligible_trial_ids)
    if len(eligible) != policy.family_size or len(eligible) != len(set(eligible)):
        raise IncompleteTrialFamilyError("eligible family count does not match frozen policy")
    expected = set(eligible)
    observed = set(p_values)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise IncompleteTrialFamilyError(
            f"correction requires complete eligible family; missing={missing}, extra={extra}"
        )
    values: list[tuple[str, float]] = []
    for trial_id in eligible:
        value = p_values[trial_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IncompleteTrialFamilyError(f"p-value for {trial_id!r} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise IncompleteTrialFamilyError(f"p-value for {trial_id!r} must be finite in [0, 1]")
        values.append((trial_id, numeric))
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted_ordered: list[float] = [0.0] * count
    if policy.method == "holm_bonferroni":
        running = 0.0
        for index, (_, value) in enumerate(ordered):
            running = max(running, min(1.0, (count - index) * value))
            adjusted_ordered[index] = running
    else:
        running = 1.0
        for reverse_index in range(count - 1, -1, -1):
            rank = reverse_index + 1
            running = min(running, min(1.0, count * ordered[reverse_index][1] / rank))
            adjusted_ordered[reverse_index] = running
    adjusted = {trial_id: adjusted_ordered[index] for index, (trial_id, _) in enumerate(ordered)}
    rejected = {trial_id: adjusted[trial_id] <= policy.alpha for trial_id in eligible}
    return adjusted, rejected


class ResearchLedger:
    """Verified, bounded, append-only ledger for one frozen research plan."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_bytes: int = DEFAULT_MAX_LEDGER_BYTES,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> None:
        self.directory = Path(directory)
        self.ledger_path = self.directory / LEDGER_FILE_NAME
        self.head_path = self.directory / HEAD_FILE_NAME
        self.lock_path = self.directory / LOCK_FILE_NAME
        if max_bytes <= 0 or max_records <= 0:
            raise ResearchGovernanceError("ledger resource limits must be positive")
        self.max_bytes = max_bytes
        self.max_records = max_records

    @classmethod
    def create(
        cls,
        directory: str | Path,
        plan: FrozenResearchPlan,
        *,
        max_bytes: int = DEFAULT_MAX_LEDGER_BYTES,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> ResearchLedger:
        ledger = cls(directory, max_bytes=max_bytes, max_records=max_records)
        ledger.directory.mkdir(parents=True, exist_ok=True)
        ledger._validate_paths(require_existing=False)
        if ledger.ledger_path.exists() or ledger.head_path.exists():
            raise FileExistsError("research ledger already exists and cannot be overwritten")
        ledger.ledger_path.write_bytes(b"")
        ledger._append(
            event_type="PLAN_FROZEN",
            trial_id=None,
            occurred_at=plan.frozen_at,
            payload={"plan": plan.to_dict(), "plan_hash": plan.plan_hash},
        )
        return ledger

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        expected_plan_hash: str | None = None,
        max_bytes: int = DEFAULT_MAX_LEDGER_BYTES,
        max_records: int = DEFAULT_MAX_RECORDS,
    ) -> ResearchLedger:
        ledger = cls(directory, max_bytes=max_bytes, max_records=max_records)
        records = ledger.verify()
        if expected_plan_hash is not None:
            actual = str(records[0]["payload"]["plan_hash"])
            if actual != expected_plan_hash:
                raise LedgerIntegrityError("ledger plan hash does not match expected frozen plan")
        return ledger

    def _validate_paths(self, *, require_existing: bool = True) -> None:
        for path in (self.directory, self.ledger_path, self.head_path):
            if path.is_symlink():
                raise LedgerIntegrityError("ledger paths must not be symbolic links")
        if require_existing and (not self.ledger_path.is_file() or not self.head_path.is_file()):
            raise LedgerIntegrityError("ledger and head receipt must both exist")

    def _acquire_lock(self) -> int:
        try:
            return os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise LedgerIntegrityError(
                "ledger writer lock already exists; fail closed and investigate interrupted writer"
            ) from exc

    def _release_lock(self, descriptor: int) -> None:
        os.close(descriptor)
        self.lock_path.unlink(missing_ok=True)

    def _write_head(self, record_count: int, head_hash: str) -> None:
        ledger_bytes = self.ledger_path.read_bytes()
        document = {
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "record_count": record_count,
            "head_hash": head_hash,
            "ledger_sha256": _sha256(ledger_bytes),
        }
        encoded = _canonical_bytes(document) + b"\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.directory,
                prefix=f".{HEAD_FILE_NAME}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.head_path)
        except BaseException:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise

    def _append(
        self,
        *,
        event_type: str,
        trial_id: str | None,
        occurred_at: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        event_time = _parse_utc(occurred_at, field="occurred_at")
        frozen_payload = _freeze_json(payload, field="payload")
        descriptor = self._acquire_lock()
        try:
            if self.head_path.exists():
                records = self.verify()
                previous_time = _parse_utc(
                    str(records[-1]["occurred_at"]),
                    field="previous occurred_at",
                )
                if event_time < previous_time:
                    raise ResearchGovernanceError("ledger event timestamps must be nondecreasing")
                previous_hash = str(records[-1]["record_hash"])
                sequence = len(records) + 1
            else:
                if self.ledger_path.read_bytes():
                    raise LedgerIntegrityError("unreceipted ledger content cannot be extended")
                previous_hash = GENESIS_HASH
                sequence = 1
            if sequence > self.max_records:
                raise LedgerIntegrityError("ledger record limit exceeded")
            body = {
                "schema_version": GOVERNANCE_SCHEMA_VERSION,
                "sequence": sequence,
                "event_type": event_type,
                "trial_id": trial_id,
                "occurred_at": occurred_at,
                "payload": _thaw_json(frozen_payload),
                "previous_hash": previous_hash,
            }
            record_hash = _sha256(_canonical_bytes(body))
            record = {**body, "record_hash": record_hash}
            encoded = _canonical_bytes(record) + b"\n"
            current_size = self.ledger_path.stat().st_size
            if current_size + len(encoded) > self.max_bytes:
                raise LedgerIntegrityError("ledger byte limit exceeded")
            with self.ledger_path.open("ab", buffering=0) as handle:
                handle.write(encoded)
                os.fsync(handle.fileno())
            self._write_head(sequence, record_hash)
            return MappingProxyType(record)
        finally:
            self._release_lock(descriptor)

    def verify(self) -> tuple[Mapping[str, Any], ...]:
        """Verify receipt, resource limits, schema, chain, plan, and state machine."""

        self._validate_paths()
        ledger_size = self.ledger_path.stat().st_size
        if ledger_size > self.max_bytes:
            raise LedgerIntegrityError("ledger exceeds configured byte limit")
        try:
            ledger_bytes = self.ledger_path.read_bytes()
            raw_lines = ledger_bytes.splitlines()
            head = json.loads(self.head_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LedgerIntegrityError("ledger or head receipt is unreadable") from exc
        if not isinstance(head, dict) or set(head) != _HEAD_FIELDS:
            raise LedgerIntegrityError("head receipt fields mismatch")
        if head["schema_version"] != GOVERNANCE_SCHEMA_VERSION:
            raise LedgerIntegrityError("unsupported head receipt schema")
        if len(raw_lines) > self.max_records:
            raise LedgerIntegrityError("ledger exceeds configured record limit")
        if head["record_count"] != len(raw_lines):
            raise LedgerIntegrityError("head receipt detects ledger truncation or extension")
        if head["ledger_sha256"] != _sha256(ledger_bytes):
            raise LedgerIntegrityError("head receipt detects ledger byte mutation")
        records: list[Mapping[str, Any]] = []
        states: dict[str, TrialStatus] = {}
        previous_hash = GENESIS_HASH
        previous_time: datetime | None = None
        plan_trial_ids: set[str] = set()
        for index, raw_line in enumerate(raw_lines, start=1):
            try:
                record = json.loads(raw_line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise LedgerIntegrityError(f"ledger record {index} is invalid JSON") from exc
            if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
                raise LedgerIntegrityError(f"ledger record {index} fields mismatch")
            if (
                record["schema_version"] != GOVERNANCE_SCHEMA_VERSION
                or record["sequence"] != index
                or record["previous_hash"] != previous_hash
            ):
                raise LedgerIntegrityError(f"ledger record {index} chain metadata mismatch")
            body = {key: record[key] for key in _RECORD_FIELDS if key != "record_hash"}
            expected_hash = _sha256(_canonical_bytes(body))
            if record["record_hash"] != expected_hash:
                raise LedgerIntegrityError(f"ledger record {index} hash mismatch")
            occurred_at = _parse_utc(record["occurred_at"], field=f"record[{index}].occurred_at")
            if previous_time is not None and occurred_at < previous_time:
                raise LedgerIntegrityError("ledger event timestamps must be nondecreasing")
            previous_time = occurred_at
            if index == 1:
                if record["event_type"] != "PLAN_FROZEN" or record["trial_id"] is not None:
                    raise LedgerIntegrityError("ledger must begin with exactly one frozen plan")
                plan_payload = record["payload"]
                if not isinstance(plan_payload, dict) or set(plan_payload) != {"plan", "plan_hash"}:
                    raise LedgerIntegrityError("frozen plan payload fields mismatch")
                try:
                    validated_plan = FrozenResearchPlan.from_dict(plan_payload["plan"])
                except ResearchGovernanceError as exc:
                    raise LedgerIntegrityError("frozen plan semantics are invalid") from exc
                if plan_payload["plan_hash"] != validated_plan.plan_hash:
                    raise LedgerIntegrityError("frozen plan hash mismatch")
                plan_trial_ids = {trial.trial_id for trial in validated_plan.trials}
            else:
                self._verify_event(record, states, plan_trial_ids)
            previous_hash = str(record["record_hash"])
            records.append(MappingProxyType(record))
        if not records or head["head_hash"] != previous_hash:
            raise LedgerIntegrityError("head receipt does not match ledger head")
        return tuple(records)

    @staticmethod
    def _verify_event(
        record: Mapping[str, Any],
        states: dict[str, TrialStatus],
        plan_trial_ids: set[str],
    ) -> None:
        event_type = record["event_type"]
        trial_id = record["trial_id"]
        if event_type in {"DUPLICATE_REJECTED", "FAMILY_EVALUATION_REJECTED", "FAMILY_EVALUATED"}:
            if event_type == "DUPLICATE_REJECTED" and trial_id not in plan_trial_ids:
                raise LedgerIntegrityError("duplicate event references unknown trial")
            if event_type != "DUPLICATE_REJECTED" and trial_id is not None:
                raise LedgerIntegrityError("family event must not name one trial")
            return
        new_status = _EVENT_TO_STATUS.get(str(event_type))
        if new_status is None or trial_id not in plan_trial_ids:
            raise LedgerIntegrityError("ledger event type or trial identity is invalid")
        previous = states.get(str(trial_id))
        if new_status not in _ALLOWED_TRANSITIONS[previous]:
            raise LedgerIntegrityError(
                f"invalid trial transition for {trial_id}: {previous} -> {new_status}"
            )
        states[str(trial_id)] = new_status

    def _plan(self) -> Mapping[str, Any]:
        return self.verify()[0]["payload"]["plan"]

    def _state(self) -> tuple[dict[str, TrialStatus], dict[str, Mapping[str, Any]]]:
        states: dict[str, TrialStatus] = {}
        terminal: dict[str, Mapping[str, Any]] = {}
        for record in self.verify()[1:]:
            event_type = str(record["event_type"])
            if event_type in _EVENT_TO_STATUS:
                trial_id = str(record["trial_id"])
                states[trial_id] = _EVENT_TO_STATUS[event_type]
                if states[trial_id] in _TERMINAL_STATUSES:
                    terminal[trial_id] = record
        return states, terminal

    def register_trial(self, trial_id: str, *, occurred_at: str) -> None:
        _require_identifier(trial_id, field="trial_id")
        plan = self._plan()
        specs = {str(spec["trial_id"]): spec for spec in plan["trials"] if isinstance(spec, dict)}
        if trial_id not in specs:
            raise ResearchGovernanceError("trial is not part of the frozen eligible family")
        states, _ = self._state()
        if trial_id in states:
            self._append(
                event_type="DUPLICATE_REJECTED",
                trial_id=trial_id,
                occurred_at=occurred_at,
                payload={"existing_status": states[trial_id], "reason": "duplicate_registration"},
            )
            raise DuplicateTrialError(f"trial {trial_id!r} is already registered")
        self._append(
            event_type="TRIAL_REGISTERED",
            trial_id=trial_id,
            occurred_at=occurred_at,
            payload={
                "plan_hash": self.verify()[0]["payload"]["plan_hash"],
                "spec": specs[trial_id],
            },
        )

    def transition(
        self,
        trial_id: str,
        status: TrialStatus,
        *,
        occurred_at: str,
        details: Mapping[str, Any],
    ) -> None:
        _require_identifier(trial_id, field="trial_id")
        if status not in {
            "STARTED",
            "INTERRUPTED",
            "RESUMED",
            "SUCCEEDED",
            "FAILED",
        }:
            raise ResearchGovernanceError("transition status is invalid")
        states, _ = self._state()
        previous = states.get(trial_id)
        if status not in _ALLOWED_TRANSITIONS[previous]:
            raise ResearchGovernanceError(
                f"invalid trial transition for {trial_id}: {previous} -> {status}"
            )
        frozen_details = _freeze_json(details, field="details")
        if status == "SUCCEEDED":
            if set(frozen_details) != {"metrics", "p_value"}:
                raise ResearchGovernanceError(
                    "successful trial details require exactly metrics and p_value"
                )
            p_value = frozen_details["p_value"]
            if (
                isinstance(p_value, bool)
                or not isinstance(p_value, (int, float))
                or not math.isfinite(float(p_value))
                or not 0.0 <= float(p_value) <= 1.0
            ):
                raise ResearchGovernanceError("successful trial p_value must be finite in [0, 1]")
            metrics = frozen_details["metrics"]
            if not isinstance(metrics, Mapping) or not metrics:
                raise ResearchGovernanceError("successful trial metrics must be non-empty")
        elif status == "FAILED":
            if "reason" not in frozen_details or not str(frozen_details["reason"]).strip():
                raise ResearchGovernanceError("failed trial requires a non-empty reason")
        elif not frozen_details:
            raise ResearchGovernanceError(f"{status.lower()} transition requires audit details")
        self._append(
            event_type=f"TRIAL_{status}",
            trial_id=trial_id,
            occurred_at=occurred_at,
            payload=frozen_details,
        )

    def evaluate_family(self, *, occurred_at: str) -> FamilyEvaluation:
        """Evaluate every frozen trial or audit and reject the incomplete attempt."""

        records = self.verify()
        if any(record["event_type"] == "FAMILY_EVALUATED" for record in records):
            self._append(
                event_type="FAMILY_EVALUATION_REJECTED",
                trial_id=None,
                occurred_at=occurred_at,
                payload={"reason": "family_already_evaluated"},
            )
            raise ResearchGovernanceError("frozen trial family has already been evaluated")
        plan = self._plan()
        eligible = tuple(str(spec["trial_id"]) for spec in plan["trials"])
        states, terminal = self._state()
        missing = [
            trial_id for trial_id in eligible if states.get(trial_id) not in _TERMINAL_STATUSES
        ]
        if missing:
            self._append(
                event_type="FAMILY_EVALUATION_REJECTED",
                trial_id=None,
                occurred_at=occurred_at,
                payload={
                    "reason": "incomplete_eligible_family",
                    "missing_or_nonterminal": missing,
                    "eligible_count": len(eligible),
                },
            )
            raise IncompleteTrialFamilyError(
                f"complete eligible family required; missing or nonterminal={missing}"
            )
        policy_doc = plan["correction"]
        policy = MultipleTestingPolicy(
            method=policy_doc["method"],
            alpha=float(policy_doc["alpha"]),
            family_size=int(policy_doc["family_size"]),
            assumptions=tuple(policy_doc["assumptions"]),
            failed_trial_p_value=float(policy_doc["failed_trial_p_value"]),
        )
        p_values: dict[str, float] = {}
        metrics: dict[str, Mapping[str, Any]] = {}
        for trial_id in eligible:
            record = terminal[trial_id]
            if states[trial_id] == "FAILED":
                p_values[trial_id] = policy.failed_trial_p_value
                metrics[trial_id] = MappingProxyType({})
            else:
                p_values[trial_id] = float(record["payload"]["p_value"])
                metrics[trial_id] = record["payload"]["metrics"]
        adjusted, rejected = adjust_p_values(
            p_values,
            policy=policy,
            eligible_trial_ids=eligible,
        )
        criteria = tuple(
            KillCriterion(
                name=item["name"],
                metric=item["metric"],
                operator=item["operator"],
                threshold=float(item["threshold"]),
            )
            for item in plan["kill_criteria"]
        )
        killed: dict[str, tuple[str, ...]] = {}
        for trial_id in eligible:
            triggered: list[str] = []
            if states[trial_id] == "FAILED":
                triggered.append("trial_failed")
            for criterion in criteria:
                value = metrics[trial_id].get(criterion.metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or criterion.triggered(float(value))
                ):
                    triggered.append(criterion.name)
            if not rejected[trial_id]:
                triggered.append("multiplicity_adjusted_evidence")
            killed[trial_id] = tuple(triggered)
        evaluation = FamilyEvaluation(
            method=policy.method,
            alpha=policy.alpha,
            raw_p_values=p_values,
            adjusted_p_values=adjusted,
            rejected=rejected,
            killed=killed,
        )
        self._append(
            event_type="FAMILY_EVALUATED",
            trial_id=None,
            occurred_at=occurred_at,
            payload=evaluation.to_dict(),
        )
        return evaluation
