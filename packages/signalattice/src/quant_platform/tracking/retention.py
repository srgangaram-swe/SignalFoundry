"""Authority-bound, crash-visible retention for unlinked CAS objects.

Retention is a policy layer over :class:`RunRegistry` and :class:`ArtifactStore`; it neither
deletes registry evidence nor performs pathname-based deletion.  Every destructive plan is a
bounded canonical envelope authenticated by the registry's persisted HMAC authority and bound to
the exact registry and CAS identities.  The protocol has two durable phases:

1. ``stage`` preflights every unstaged registry invariant and exact, physically eligible CAS
   generation, then repeats registry validation under ``BEGIN IMMEDIATE`` and atomically commits
   the immutable signed envelope plus pending tombstones for every candidate.
2. ``execute`` verifies and unlinks each exact CAS generation individually, then performs the
   tombstone's sole legal ``NULL -> deleted_at`` transition in a bounded per-object transaction.

Committing intent before unlinking ensures a process crash can never erase evidence that bytes may
have been selected or removed.  A pending tombstone prevents future run links and new claims.  If a
process dies after unlink but before finalization, the tombstone intentionally remains pending for
operator reconciliation: absence alone is ambiguous and is never promoted to proof of deletion.
Metadata and signed plan envelopes are append-only.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import sqlite3
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn, Protocol

from quant_platform.tracking.cas import (
    ArtifactGeneration,
    ArtifactStore,
    ArtifactStoreError,
    PublishedArtifact,
)
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactMetadata,
    BusyError,
    IntegrityError,
    RegistryError,
    RegistryLimits,
    RetentionPlan,
    RetentionPlanAuthenticationError,
    ValidationError,
    canonical_json,
    decode_bounded_json,
    require_digest,
    require_utc,
)

_PLAN_SCHEMA_VERSION = 1
_MAX_RETENTION_CANDIDATES = 1_000
_MAX_RECOVERY_PLANS = 1_000
_MAX_PLAN_ENVELOPE_BYTES = 1_048_576
_MAX_GRACE_SECONDS = 366 * 24 * 60 * 60
_MAX_TOTAL_BYTES = 2**63 - 1
_QUERY_TIMEOUT_SECONDS = 2.0
_SQLITE_PROGRESS_INSTRUCTIONS = 1_000
_REASON_CODE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_CAUSE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_REGISTRY_ID = re.compile(r"^[0-9a-f]{32}$")
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class RetentionError(RegistryError):
    """Base class for bounded retention-policy failures."""

    code = "retention_error"


class RetentionDisabledError(RetentionError):
    """Retention was invoked without an explicit enabled policy."""

    code = "retention_disabled"


class RetentionConfirmationError(RetentionError):
    """The caller did not confirm the exact authority-bound plan identity."""

    code = "retention_confirmation_failed"


class RetentionDriftError(RetentionError):
    """A candidate or durable intent changed after the plan was created."""

    code = "retention_candidate_drift"


class RetentionActiveWorkError(RetentionError):
    """Retention cannot safely proceed while registry work is running."""

    code = "retention_active_work"
    retryable = True


class RetentionIntegrityError(IntegrityError):
    """Registry or CAS state cannot safely satisfy a confirmed retention plan."""

    code = "retention_integrity_error"


class RetentionPartialFailure(RetentionError):
    """Execution stopped with durable tombstones describing all remaining work.

    Digest identifiers are safe to expose and let an operator reconcile the exact bounded plan.
    Filesystem paths, artifact bytes, and raw operating-system or SQLite errors are intentionally
    absent.
    """

    code = "retention_partial_failure"

    def __init__(
        self,
        *,
        plan_digest: str,
        finalized_digests: tuple[str, ...],
        pending_digests: tuple[str, ...],
        cause_code: str,
    ) -> None:
        super().__init__("retention stopped; durable pending intent requires reconciliation")
        self.plan_digest = require_digest(plan_digest, "plan_digest")
        self.finalized_digests = _validated_digest_tuple(
            finalized_digests,
            field_name="finalized_digests",
            allow_empty=True,
        )
        self.pending_digests = _validated_digest_tuple(
            pending_digests,
            field_name="pending_digests",
            allow_empty=False,
        )
        if set(self.finalized_digests).intersection(self.pending_digests):
            raise ValidationError("finalized and pending digests must be disjoint")
        if type(cause_code) is not str or _CAUSE_CODE.fullmatch(cause_code) is None:
            raise ValidationError("cause_code must be a bounded lowercase machine code")
        self.cause_code = cause_code


class _RegistryHandle(Protocol):
    """Authenticated registry authority required by retention."""

    path: Path
    limits: RegistryLimits

    @property
    def registry_id(self) -> str:
        """Return the verified persisted registry identity."""
        ...

    @property
    def artifact_store_id(self) -> str | None:
        """Return the verified durable CAS binding, if present."""
        ...

    @property
    def artifact_verifier(self) -> object | None:
        """Return the exact in-process verifier authority."""
        ...

    def bind_artifact_store(self, store_id: str) -> None:
        """Idempotently bind this registry to one durable CAS identity."""
        ...

    def sign_retention_payload(self, payload: bytes, *, store_id: str) -> str:
        """Return a domain-separated HMAC identity for canonical retention bytes."""
        ...

    def verify_retention_payload_signature(
        self,
        payload: bytes,
        *,
        store_id: str,
        signature: str,
    ) -> None:
        """Verify a retention HMAC under this exact persisted authority."""
        ...

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        """Open one schema-, identity-, and secret-bound registry connection."""
        ...

    def _now(self) -> datetime:
        """Return one validated instant from the registry's clock authority."""
        ...


@dataclass(slots=True)
class _OperationBudget:
    """Cooperative monotonic deadline shared by SQLite and Python validation."""

    deadline: float
    last_observed: float

    @classmethod
    def start(cls) -> _OperationBudget:
        now = time.monotonic()
        return cls(now + _QUERY_TIMEOUT_SECONDS, now)

    def expired(self) -> int:
        observed = time.monotonic()
        if observed < self.last_observed:
            return 1
        self.last_observed = observed
        return int(observed >= self.deadline)

    def checkpoint(self) -> None:
        observed = time.monotonic()
        if observed < self.last_observed:
            raise RetentionIntegrityError("retention monotonic clock moved backwards")
        self.last_observed = observed
        if observed >= self.deadline:
            raise RetentionIntegrityError("retention operation exceeded its bounded deadline")


@dataclass(frozen=True, slots=True)
class _StoredTombstone:
    """Strictly decoded immutable retention evidence from SQLite."""

    artifact_digest: str
    plan_digest: str
    planned_at: datetime
    deleted_at: datetime | None
    reason: str


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """Bounded destructive-policy configuration, disabled unless explicitly enabled.

    ``grace_period_seconds`` protects both registry publication time and the physical CAS
    generation timestamp. Each invocation is bounded by candidate count and aggregate bytes.
    """

    enabled: bool = False
    grace_period_seconds: int = 7 * 24 * 60 * 60
    max_candidates: int = 100
    max_total_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValidationError("enabled must be a boolean")
        _bounded_integer(
            self.grace_period_seconds,
            "grace_period_seconds",
            minimum=1,
            maximum=_MAX_GRACE_SECONDS,
        )
        _bounded_integer(
            self.max_candidates,
            "max_candidates",
            minimum=1,
            maximum=_MAX_RETENTION_CANDIDATES,
        )
        _bounded_integer(
            self.max_total_bytes,
            "max_total_bytes",
            minimum=1,
            maximum=_MAX_TOTAL_BYTES,
        )


@dataclass(frozen=True, slots=True)
class RetentionCandidate:
    """Exact registry metadata and physical CAS generation selected for deletion."""

    metadata: ArtifactMetadata
    last_changed_at: datetime
    last_changed_ns: int
    generation: ArtifactGeneration

    def __post_init__(self) -> None:
        if type(self.metadata) is not ArtifactMetadata:
            raise ValidationError("metadata must be ArtifactMetadata")
        if self.metadata.pinned:
            raise ValidationError("retention candidates may not be pinned")
        _published_artifact(self.metadata)
        object.__setattr__(
            self,
            "last_changed_at",
            require_utc(self.last_changed_at, "last_changed_at"),
        )
        exact_ns = _bounded_integer(
            self.last_changed_ns,
            "last_changed_ns",
            minimum=-(2**63),
            maximum=2**63 - 1,
        )
        if _nanoseconds_to_display_time(exact_ns) != self.last_changed_at:
            raise ValidationError(
                "last_changed_at must be the canonical display time for last_changed_ns"
            )
        if type(self.generation) is not ArtifactGeneration:
            raise ValidationError("generation must be an ArtifactGeneration")

    @property
    def digest(self) -> str:
        """Return the candidate's content digest."""

        return self.metadata.digest


@dataclass(frozen=True, slots=True)
class DecodedRetentionPlan:
    """Strict typed view of one canonical, unsigned retention-plan payload.

    This value carries no deletion authority.  Callers must separately authenticate the exact
    canonical payload against the registry/CAS identities before trusting persisted intent.
    """

    plan: RetentionPlan
    candidates: tuple[RetentionCandidate, ...]
    eligible_before: datetime
    policy: RetentionPolicy
    registry_id: str
    cas_store_id: str
    schema_version: int


@dataclass(frozen=True, slots=True)
class DigestConfirmedRetentionPlan:
    """Canonical signed deletion envelope bound to one registry and CAS generation set.

    ``digest`` is the operator-confirmed, domain-separated registry HMAC. ``payload_digest`` is a
    separately verifiable SHA-256 of the unsigned canonical payload; it detects storage corruption
    but never grants destructive authority.
    """

    plan: RetentionPlan
    candidates: tuple[RetentionCandidate, ...]
    eligible_before: datetime
    policy: RetentionPolicy
    registry_id: str
    cas_store_id: str
    payload_digest: str
    digest: str
    schema_version: int = _PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.plan) is not RetentionPlan:
            raise ValidationError("plan must be a RetentionPlan")
        _require_reason(self.plan.reason)
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValidationError("candidates must be a non-empty tuple")
        if len(self.candidates) > _MAX_RETENTION_CANDIDATES:
            raise ValidationError(f"candidates may not exceed {_MAX_RETENTION_CANDIDATES} entries")
        if any(type(candidate) is not RetentionCandidate for candidate in self.candidates):
            raise ValidationError("every candidate must be a RetentionCandidate")
        digests = tuple(candidate.digest for candidate in self.candidates)
        if digests != tuple(sorted(set(digests))):
            raise ValidationError("candidates must be unique and sorted by digest")
        if digests != self.plan.artifact_digests:
            raise ValidationError("candidate digests must exactly match the retention plan")
        eligible_before = require_utc(self.eligible_before, "eligible_before")
        object.__setattr__(self, "eligible_before", eligible_before)
        if eligible_before > self.plan.planned_at:
            raise ValidationError("eligible_before may not follow planned_at")
        if type(self.policy) is not RetentionPolicy:
            raise ValidationError("policy must be a RetentionPolicy")
        if type(self.schema_version) is not int or self.schema_version != _PLAN_SCHEMA_VERSION:
            raise ValidationError("retention plan schema_version is unsupported")
        _require_registry_id(self.registry_id)
        require_digest(self.cas_store_id, "cas_store_id")
        require_digest(self.payload_digest, "payload_digest")
        require_digest(self.digest, "plan_digest")
        payload = self.canonical_payload_bytes()
        expected_payload_digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(self.payload_digest, expected_payload_digest):
            raise ValidationError("payload_digest does not match canonical plan bytes")

    @property
    def total_bytes(self) -> int:
        """Return the plan's bounded aggregate byte count."""

        return sum(candidate.metadata.byte_size for candidate in self.candidates)

    @property
    def policy_digest(self) -> str:
        """Return a non-authorizing SHA-256 fingerprint of the exact policy."""

        return hashlib.sha256(
            canonical_json(_policy_payload(self.policy)).encode("utf-8")
        ).hexdigest()

    def canonical_payload_bytes(self) -> bytes:
        """Return the bounded canonical unsigned envelope bytes."""

        encoded = canonical_json(
            _plan_payload(
                plan=self.plan,
                candidates=self.candidates,
                eligible_before=self.eligible_before,
                policy=self.policy,
                registry_id=self.registry_id,
                cas_store_id=self.cas_store_id,
                schema_version=self.schema_version,
            )
        ).encode("utf-8")
        if not 2 <= len(encoded) <= _MAX_PLAN_ENVELOPE_BYTES:
            raise ValidationError("canonical retention payload exceeds its persisted byte bound")
        return encoded

    @classmethod
    def create(
        cls,
        *,
        plan: RetentionPlan,
        candidates: tuple[RetentionCandidate, ...],
        eligible_before: datetime,
        policy: RetentionPolicy,
        registry: _RegistryHandle,
        cas_store_id: str,
    ) -> DigestConfirmedRetentionPlan:
        """Sign a canonical plan with one verified registry/CAS authority."""

        if type(policy) is not RetentionPolicy:
            raise ValidationError("policy must be a RetentionPolicy")
        registry_id = _require_registry_id(registry.registry_id)
        store_id = require_digest(cas_store_id, "cas_store_id")
        payload = canonical_json(
            _plan_payload(
                plan=plan,
                candidates=candidates,
                eligible_before=eligible_before,
                policy=policy,
                registry_id=registry_id,
                cas_store_id=store_id,
                schema_version=_PLAN_SCHEMA_VERSION,
            )
        ).encode("utf-8")
        if not 2 <= len(payload) <= _MAX_PLAN_ENVELOPE_BYTES:
            raise ValidationError("canonical retention payload exceeds its persisted byte bound")
        payload_digest = hashlib.sha256(payload).hexdigest()
        digest = registry.sign_retention_payload(payload, store_id=store_id)
        return cls(
            plan=plan,
            candidates=candidates,
            eligible_before=eligible_before,
            policy=policy,
            registry_id=registry_id,
            cas_store_id=store_id,
            payload_digest=payload_digest,
            digest=digest,
        )


@dataclass(frozen=True, slots=True)
class RetentionIntent:
    """Durable phase-A state for one exact plan."""

    plan_digest: str
    pending_digests: tuple[str, ...]
    finalized_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        require_digest(self.plan_digest, "plan_digest")
        pending = _validated_digest_tuple(
            self.pending_digests,
            field_name="pending_digests",
            allow_empty=True,
        )
        finalized = _validated_digest_tuple(
            self.finalized_digests,
            field_name="finalized_digests",
            allow_empty=True,
        )
        if set(pending).intersection(finalized):
            raise ValidationError("pending and finalized digests must be disjoint")


@dataclass(frozen=True, slots=True)
class RetentionRecovery:
    """Bounded path-free summary of one staged plan requiring reconciliation."""

    plan_digest: str
    planned_at: datetime
    reason: str
    pending_count: int
    finalized_count: int

    def __post_init__(self) -> None:
        require_digest(self.plan_digest, "plan_digest")
        object.__setattr__(self, "planned_at", require_utc(self.planned_at, "planned_at"))
        _require_reason(self.reason)
        _bounded_integer(
            self.pending_count,
            "pending_count",
            minimum=1,
            maximum=_MAX_RETENTION_CANDIDATES,
        )
        _bounded_integer(
            self.finalized_count,
            "finalized_count",
            minimum=0,
            maximum=_MAX_RETENTION_CANDIDATES,
        )
        if self.pending_count + self.finalized_count > _MAX_RETENTION_CANDIDATES:
            raise ValidationError("recovery cohort exceeds the retention candidate bound")


@dataclass(frozen=True, slots=True)
class RetentionExecution:
    """Immutable summary of an exactly identified retention execution."""

    plan_digest: str
    newly_deleted_digests: tuple[str, ...]
    previously_deleted_digests: tuple[str, ...]
    deleted_bytes: int
    executed_at: datetime

    def __post_init__(self) -> None:
        require_digest(self.plan_digest, "plan_digest")
        newly = _validated_digest_tuple(
            self.newly_deleted_digests,
            field_name="newly_deleted_digests",
            allow_empty=True,
        )
        previous = _validated_digest_tuple(
            self.previously_deleted_digests,
            field_name="previously_deleted_digests",
            allow_empty=True,
        )
        if set(newly).intersection(previous):
            raise ValidationError("new and previous deletion digests must be disjoint")
        _bounded_integer(
            self.deleted_bytes,
            "deleted_bytes",
            minimum=0,
            maximum=_MAX_TOTAL_BYTES,
        )
        object.__setattr__(self, "executed_at", require_utc(self.executed_at, "executed_at"))


class RetentionController:
    """Plan and execute confirmed retention against one exact registry/CAS pair.

    Construction binds the registry durably to the initialized CAS and requires its configured
    verifier to be the *same object* supplied here. Planning is bounded and read-only after that
    explicit binding. ``stage`` persists a restart-recoverable signed envelope and cohort;
    ``load_plan``, ``list_recovery_plans``, and ``resume`` expose bounded reconciliation surfaces.
    """

    def __init__(
        self,
        registry: _RegistryHandle,
        artifact_store: ArtifactStore,
        *,
        policy: RetentionPolicy | None = None,
    ) -> None:
        if type(artifact_store) is not ArtifactStore:
            raise ValidationError("artifact_store must be an ArtifactStore")
        path = getattr(registry, "path", None)
        limits = getattr(registry, "limits", None)
        required_callables = (
            "_connect",
            "_now",
            "bind_artifact_store",
            "sign_retention_payload",
            "verify_retention_payload_signature",
        )
        if (
            not isinstance(path, Path)
            or type(limits) is not RegistryLimits
            or any(not callable(getattr(registry, name, None)) for name in required_callables)
        ):
            raise ValidationError("registry must expose authenticated retention authorities")
        if policy is not None and type(policy) is not RetentionPolicy:
            raise ValidationError("policy must be a RetentionPolicy")
        if getattr(registry, "artifact_verifier", None) is not artifact_store:
            raise ValidationError("registry artifact verifier must be the exact ArtifactStore")
        try:
            store_id = artifact_store.store_id
        except ArtifactStoreError:
            raise ValidationError("artifact_store must be initialized") from None
        registry.bind_artifact_store(store_id)
        bound_store_id = registry.artifact_store_id
        if type(bound_store_id) is not str or not hmac.compare_digest(bound_store_id, store_id):
            raise RetentionIntegrityError("registry CAS binding does not match the supplied store")
        self._registry = registry
        self._store = artifact_store
        self.policy = RetentionPolicy() if policy is None else policy

    def plan(self, *, reason: str) -> DigestConfirmedRetentionPlan | None:
        """Return a deterministic, signed, bounded plan or ``None`` when none is eligible.

        Registry eligibility requires unpinned metadata older than the configured grace period,
        no immutable run link, no prior tombstone, and an aggregate byte count within policy. The
        bounded candidate set is then fully inspected through the CAS descriptor boundary so the
        signed plan binds each exact physical generation. No bytes are mutated.
        """

        self._require_enabled()
        _require_reason(reason)
        instant = self._registry._now()
        try:
            eligible_before = instant - timedelta(seconds=self.policy.grace_period_seconds)
        except OverflowError:
            raise ValidationError(
                "planned_at is too early for the configured grace period"
            ) from None
        budget = _OperationBudget.start()
        connection = self._registry._connect(readonly=True)
        _install_progress_handler(connection, budget)
        try:
            rows = connection.execute(
                """
                SELECT a.digest, a.artifact_class, a.byte_size, a.media_type,
                       a.storage_relpath, a.created_at, a.pinned
                FROM sl_registry_artifacts AS a
                WHERE a.pinned = 0
                  AND a.created_at <= ?
                  AND a.byte_size <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM sl_registry_run_artifacts AS links
                      WHERE links.artifact_digest = a.digest
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM sl_registry_retention_tombstones AS tombstones
                      WHERE tombstones.artifact_digest = a.digest
                  )
                ORDER BY a.created_at ASC, a.digest ASC
                LIMIT ?
                """,
                (
                    _db_time(eligible_before),
                    self.policy.max_total_bytes,
                    self.policy.max_candidates,
                ),
            ).fetchall()
            if _has_running_work(connection):
                raise RetentionActiveWorkError(
                    "retention is unavailable while a job attempt is running"
                )
            budget.checkpoint()
        except sqlite3.Error as exc:
            _raise_sqlite(exc, operation="retention planning")
        finally:
            _close_connection(connection)

        selected: list[ArtifactMetadata] = []
        total_bytes = 0
        for row in rows:
            budget.checkpoint()
            candidate = _metadata_from_row(row)
            if total_bytes + candidate.byte_size > self.policy.max_total_bytes:
                break
            selected.append(candidate)
            total_bytes += candidate.byte_size
        if not selected:
            return None

        exact_candidates: list[RetentionCandidate] = []
        for metadata in sorted(selected, key=lambda candidate: candidate.digest):
            budget.checkpoint()
            try:
                record = self._store.inspect(_published_artifact(metadata))
            except ArtifactStoreError:
                raise RetentionIntegrityError(
                    "a retention candidate failed descriptor-safe CAS inspection"
                ) from None
            if record.last_changed_ns > _datetime_to_nanoseconds(eligible_before):
                continue
            exact_candidates.append(
                RetentionCandidate(
                    metadata,
                    record.last_changed_at,
                    record.last_changed_ns,
                    record.generation,
                )
            )
        if not exact_candidates:
            return None
        candidates = tuple(exact_candidates)
        plan = RetentionPlan(
            tuple(candidate.digest for candidate in candidates),
            instant,
            reason,
        )
        return DigestConfirmedRetentionPlan.create(
            plan=plan,
            candidates=candidates,
            eligible_before=eligible_before,
            policy=self.policy,
            registry=self._registry,
            cas_store_id=self._store.store_id,
        )

    def stage(
        self,
        plan: DigestConfirmedRetentionPlan,
        *,
        confirmed_digest: str,
    ) -> RetentionIntent:
        """Atomically commit the signed envelope and pending intent for an exact plan."""

        authority_instant = self._registry._now()
        self._preflight_new_intent(
            plan,
            confirmed_digest=confirmed_digest,
            authority_instant=authority_instant,
        )
        return self._stage(
            plan,
            confirmed_digest=confirmed_digest,
            authority_instant=authority_instant,
        )

    def _preflight_new_intent(
        self,
        plan: DigestConfirmedRetentionPlan,
        *,
        confirmed_digest: str,
        authority_instant: datetime,
    ) -> None:
        """Verify an unstaged cohort completely before durable deletion intent exists.

        Registry invariants are repeated under the phase-A write transaction. CAS generation
        checks remain outside that transaction, and the physical unlink retains its independent
        compare-and-delete guard against a generation change after this preflight.
        """

        self._validate_confirmation(plan, confirmed_digest, authority_instant)
        budget = _OperationBudget.start()
        connection = self._registry._connect(readonly=True)
        _install_progress_handler(connection, budget)
        try:
            connection.execute("BEGIN")
            if _has_running_work(connection):
                raise RetentionActiveWorkError(
                    "retention intent cannot proceed while a job attempt is running"
                )
            existing_tombstones = self._inspect_durable_plan_state(connection, plan, budget)
            budget.checkpoint()
            connection.execute("ROLLBACK")
        except sqlite3.Error as exc:
            _raise_sqlite(exc, operation="retention intent preflight")
        finally:
            _close_connection(connection)

        if existing_tombstones is not None:
            return
        for candidate in plan.candidates:
            budget.checkpoint()
            try:
                self._inspect_candidate_generation(plan, candidate)
            except ArtifactStoreError:
                raise RetentionDriftError(
                    "retention candidate failed exact CAS generation preflight"
                ) from None
        budget.checkpoint()

    def _inspect_durable_plan_state(
        self,
        connection: sqlite3.Connection,
        plan: DigestConfirmedRetentionPlan,
        budget: _OperationBudget,
    ) -> tuple[_StoredTombstone, ...] | None:
        """Validate existing plan state, candidate ownership, and registry eligibility."""

        envelope_rows = connection.execute(
            """
            SELECT plan_digest, payload_digest, payload_json, registry_id,
                   cas_store_id, planned_at, schema_version
            FROM sl_registry_retention_plans
            WHERE plan_digest = ?
            LIMIT 2
            """,
            (plan.digest,),
        ).fetchall()
        if len(envelope_rows) > 1:
            raise RetentionIntegrityError("retention plan identity is not unique")
        existing_rows = connection.execute(
            """
            SELECT artifact_digest, plan_digest, planned_at, deleted_at, reason
            FROM sl_registry_retention_tombstones
            WHERE plan_digest = ?
            ORDER BY artifact_digest
            LIMIT ?
            """,
            (plan.digest, _MAX_RETENTION_CANDIDATES + 1),
        ).fetchall()
        if len(existing_rows) > _MAX_RETENTION_CANDIDATES:
            raise RetentionIntegrityError("persisted retention cohort exceeds its bound")
        existing_tombstones = tuple(_tombstone_from_row(row) for row in existing_rows)

        if envelope_rows:
            stored_plan = self._plan_from_envelope_row(envelope_rows[0], budget=budget)
            if stored_plan != plan:
                raise RetentionDriftError(
                    "persisted signed retention envelope differs from the confirmed plan"
                )
            if not existing_tombstones:
                raise RetentionDriftError("signed retention envelope has no candidate cohort")
        elif existing_tombstones:
            raise RetentionIntegrityError("retention cohort is missing its signed envelope")

        if existing_tombstones:
            actual_digests = tuple(tombstone.artifact_digest for tombstone in existing_tombstones)
            if actual_digests != plan.plan.artifact_digests:
                raise RetentionDriftError(
                    "persisted retention intent does not match the confirmed candidate set"
                )
            for tombstone in existing_tombstones:
                budget.checkpoint()
                _require_exact_tombstone(tombstone, plan)
        else:
            for candidate in plan.candidates:
                budget.checkpoint()
                other = connection.execute(
                    """
                    SELECT plan_digest
                    FROM sl_registry_retention_tombstones
                    WHERE artifact_digest = ?
                    LIMIT 2
                    """,
                    (candidate.digest,),
                ).fetchall()
                if len(other) > 1:
                    raise RetentionIntegrityError("retention artifact identity is not unique")
                if other:
                    raise RetentionDriftError(
                        "a retention candidate belongs to another durable plan"
                    )

        for candidate in plan.candidates:
            budget.checkpoint()
            self._revalidate_candidate(connection, plan, candidate.metadata)
        return existing_tombstones if envelope_rows else None

    def _stage(
        self,
        plan: DigestConfirmedRetentionPlan,
        *,
        confirmed_digest: str,
        authority_instant: datetime,
    ) -> RetentionIntent:
        """Implement phase A using one caller-owned, validated authority instant."""

        self._validate_confirmation(plan, confirmed_digest, authority_instant)
        payload_bytes = plan.canonical_payload_bytes()
        payload_json = payload_bytes.decode("utf-8")
        budget = _OperationBudget.start()
        connection = self._registry._connect()
        _install_progress_handler(connection, budget)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if _has_running_work(connection):
                raise RetentionActiveWorkError(
                    "retention intent cannot proceed while a job attempt is running"
                )
            existing_tombstones = self._inspect_durable_plan_state(connection, plan, budget)

            if existing_tombstones is None:
                connection.execute(
                    """
                    INSERT INTO sl_registry_retention_plans(
                        plan_digest, payload_digest, payload_json, registry_id,
                        cas_store_id, planned_at, schema_version
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.digest,
                        plan.payload_digest,
                        payload_json,
                        plan.registry_id,
                        plan.cas_store_id,
                        _db_time(plan.plan.planned_at),
                        plan.schema_version,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO sl_registry_retention_tombstones(
                        artifact_digest, plan_digest, planned_at, deleted_at, reason
                    ) VALUES(?, ?, ?, NULL, ?)
                    """,
                    (
                        (
                            candidate.digest,
                            plan.digest,
                            _db_time(plan.plan.planned_at),
                            plan.plan.reason,
                        )
                        for candidate in plan.candidates
                    ),
                )
            budget.checkpoint()
            connection.execute("COMMIT")
        except RegistryError as exc:
            _rollback(connection, primary=exc)
            raise
        except sqlite3.Error as exc:
            _rollback(connection, primary=exc)
            _raise_sqlite(exc, operation="retention intent staging")
        finally:
            _close_connection(connection)
        return self._read_intent(plan)

    def load_plan(self, plan_digest: str) -> DigestConfirmedRetentionPlan:
        """Load and authenticate one persisted signed plan for restart recovery."""

        digest = require_digest(plan_digest, "plan_digest")
        budget = _OperationBudget.start()
        connection = self._registry._connect(readonly=True)
        _install_progress_handler(connection, budget)
        try:
            rows = connection.execute(
                """
                SELECT plan_digest, payload_digest, payload_json, registry_id,
                       cas_store_id, planned_at, schema_version
                FROM sl_registry_retention_plans
                WHERE plan_digest = ?
                LIMIT 2
                """,
                (digest,),
            ).fetchall()
            if len(rows) != 1:
                raise RetentionDriftError("signed retention plan is missing or ambiguous")
            plan = self._plan_from_envelope_row(rows[0], budget=budget)
        except sqlite3.Error as exc:
            _raise_sqlite(exc, operation="retention plan load")
        finally:
            _close_connection(connection)
        self._read_intent(plan)
        return plan

    def list_recovery_plans(self, *, limit: int = 100) -> tuple[RetentionRecovery, ...]:
        """List a bounded page of pending plans in deterministic recovery order.

        This diagnostic is path-free and does not authorize execution. Use :meth:`load_plan` to
        authenticate an exact envelope and :meth:`resume` with its HMAC confirmation to continue.
        """

        page_limit = _bounded_integer(
            limit,
            "limit",
            minimum=1,
            maximum=min(_MAX_RECOVERY_PLANS, self._registry.limits.max_page_size),
        )
        budget = _OperationBudget.start()
        connection = self._registry._connect(readonly=True)
        _install_progress_handler(connection, budget)
        try:
            plan_rows = connection.execute(
                """
                SELECT plans.plan_digest, plans.planned_at,
                       min(tombstones.reason) AS reason
                FROM sl_registry_retention_plans AS plans
                JOIN sl_registry_retention_tombstones AS tombstones
                  ON tombstones.plan_digest = plans.plan_digest
                WHERE EXISTS (
                    SELECT 1
                    FROM sl_registry_retention_tombstones AS pending
                    WHERE pending.plan_digest = plans.plan_digest
                      AND pending.deleted_at IS NULL
                )
                GROUP BY plans.plan_digest, plans.planned_at
                ORDER BY plans.planned_at, plans.plan_digest
                LIMIT ?
                """,
                (page_limit + 1,),
            ).fetchall()
            if len(plan_rows) > page_limit:
                plan_rows = plan_rows[:page_limit]
            summaries: list[RetentionRecovery] = []
            for row in plan_rows:
                budget.checkpoint()
                digest = _strict_text(row["plan_digest"], "plan_digest")
                counts = connection.execute(
                    """
                    SELECT deleted_at
                    FROM sl_registry_retention_tombstones
                    WHERE plan_digest = ?
                    ORDER BY artifact_digest
                    LIMIT ?
                    """,
                    (digest, _MAX_RETENTION_CANDIDATES + 1),
                ).fetchall()
                if len(counts) > _MAX_RETENTION_CANDIDATES:
                    raise RetentionIntegrityError("recovery cohort exceeds its row bound")
                pending_count = sum(row_["deleted_at"] is None for row_ in counts)
                finalized_count = len(counts) - pending_count
                summaries.append(
                    RetentionRecovery(
                        plan_digest=digest,
                        planned_at=_parse_time(row["planned_at"], "retention planned_at"),
                        reason=_strict_text(row["reason"], "retention reason"),
                        pending_count=pending_count,
                        finalized_count=finalized_count,
                    )
                )
            budget.checkpoint()
            return tuple(summaries)
        except sqlite3.Error as exc:
            _raise_sqlite(exc, operation="retention recovery listing")
        finally:
            _close_connection(connection)

    def resume(self, plan_digest: str, *, confirmed_digest: str) -> RetentionExecution:
        """Load, authenticate, and execute one exact persisted plan after restart."""

        plan = self.load_plan(plan_digest)
        return self.execute(plan, confirmed_digest=confirmed_digest)

    def execute(
        self,
        plan: DigestConfirmedRetentionPlan,
        *,
        confirmed_digest: str,
    ) -> RetentionExecution:
        """Stage and execute one exact plan, stopping at the first failed candidate.

        All pending objects are descriptor-verified against their signed generations and physical
        grace cutoff before the first unlink. Every already-finalized tombstone is chronology-
        checked against a fresh registry clock and its exact CAS key is proven absent. A failure
        leaves unfinished candidates as durable pending intent and raises
        :class:`RetentionPartialFailure`.
        """

        authority_instant = self._registry._now()
        self._preflight_new_intent(
            plan,
            confirmed_digest=confirmed_digest,
            authority_instant=authority_instant,
        )
        intent = self._stage(
            plan,
            confirmed_digest=confirmed_digest,
            authority_instant=authority_instant,
        )
        candidates_by_digest = {candidate.digest: candidate for candidate in plan.candidates}

        for digest in intent.finalized_digests:
            self._verify_finalized_candidate(plan, candidates_by_digest[digest])

        for digest in intent.pending_digests:
            try:
                self._inspect_candidate_generation(plan, candidates_by_digest[digest])
            except (RegistryError, ArtifactStoreError) as exc:
                current = self._read_intent(plan)
                if digest in current.finalized_digests:
                    continue
                raise RetentionPartialFailure(
                    plan_digest=plan.digest,
                    finalized_digests=current.finalized_digests,
                    pending_digests=current.pending_digests,
                    cause_code=_cause_code(exc),
                ) from None

        newly_deleted: list[str] = []
        for digest in intent.pending_digests:
            candidate = candidates_by_digest[digest]
            try:
                deleted = self._finalize_candidate(plan, candidate)
            except (RegistryError, ArtifactStoreError) as exc:
                current = self._read_intent(plan)
                if not current.pending_digests:
                    break
                raise RetentionPartialFailure(
                    plan_digest=plan.digest,
                    finalized_digests=current.finalized_digests,
                    pending_digests=current.pending_digests,
                    cause_code=_cause_code(exc),
                ) from None
            if deleted:
                newly_deleted.append(digest)

        final_intent = self._read_intent(plan)
        if final_intent.pending_digests:
            raise RetentionPartialFailure(
                plan_digest=plan.digest,
                finalized_digests=final_intent.finalized_digests,
                pending_digests=final_intent.pending_digests,
                cause_code="retention_finalize_incomplete",
            )
        previous = tuple(sorted(set(final_intent.finalized_digests) - set(newly_deleted)))
        deleted_bytes = sum(
            candidates_by_digest[digest].metadata.byte_size for digest in newly_deleted
        )
        executed_at = self._registry._now()
        if executed_at < plan.plan.planned_at:
            raise RetentionIntegrityError("registry clock moved behind the signed plan")
        return RetentionExecution(
            plan_digest=plan.digest,
            newly_deleted_digests=tuple(sorted(newly_deleted)),
            previously_deleted_digests=previous,
            deleted_bytes=deleted_bytes,
            executed_at=executed_at,
        )

    def _finalize_candidate(
        self,
        plan: DigestConfirmedRetentionPlan,
        candidate: RetentionCandidate,
    ) -> bool:
        if self._candidate_is_finalized(plan, candidate):
            return False
        self._inspect_candidate_generation(plan, candidate)
        before_unlink = self._registry._now()
        if before_unlink < plan.plan.planned_at:
            raise RetentionIntegrityError("registry clock moved behind the signed plan")
        self._store.unlink_verified(
            _published_artifact(candidate.metadata),
            expected_generation=candidate.generation,
        )
        # This sample is deliberately the first operation after verified unlink. A crash or clock
        # regression leaves explicit pending intent rather than manufacturing a deletion record.
        deleted_at = self._registry._now()
        if deleted_at < plan.plan.planned_at:
            raise RetentionIntegrityError(
                "CAS bytes were unlinked but the registry clock regressed before finalization"
            )

        budget = _OperationBudget.start()
        connection = self._registry._connect()
        _install_progress_handler(connection, budget)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._revalidate_candidate(connection, plan, candidate.metadata)
            rows = connection.execute(
                """
                SELECT artifact_digest, plan_digest, planned_at, deleted_at, reason
                FROM sl_registry_retention_tombstones
                WHERE artifact_digest = ? AND plan_digest = ?
                LIMIT 2
                """,
                (candidate.digest, plan.digest),
            ).fetchall()
            if len(rows) != 1:
                raise RetentionDriftError("candidate tombstone is missing or ambiguous")
            stored_tombstone = _tombstone_from_row(rows[0])
            _require_exact_tombstone(stored_tombstone, plan)
            if stored_tombstone.deleted_at is not None:
                _require_finalized_chronology(stored_tombstone, deleted_at)
                connection.execute("COMMIT")
                self._store.verify_absent(_published_artifact(candidate.metadata))
                return False
            changed = connection.execute(
                """
                UPDATE sl_registry_retention_tombstones
                SET deleted_at = ?
                WHERE artifact_digest = ? AND plan_digest = ? AND deleted_at IS NULL
                """,
                (_db_time(deleted_at), candidate.digest, plan.digest),
            ).rowcount
            if changed != 1:
                raise RetentionDriftError("candidate tombstone was not finalized exactly once")
            budget.checkpoint()
            connection.execute("COMMIT")
            return True
        except (RegistryError, ArtifactStoreError) as exc:
            _rollback(connection, primary=exc)
            raise
        except sqlite3.Error as exc:
            _rollback(connection, primary=exc)
            raise RetentionIntegrityError(
                "CAS bytes were unlinked but tombstone finalization is uncertain"
            ) from None
        finally:
            _close_connection(connection)

    def _candidate_is_finalized(
        self,
        plan: DigestConfirmedRetentionPlan,
        candidate: RetentionCandidate,
    ) -> bool:
        """Inspect one committed intent without holding a lock during CAS I/O."""

        budget = _OperationBudget.start()
        connection = self._registry._connect(readonly=True)
        _install_progress_handler(connection, budget)
        try:
            self._revalidate_candidate(connection, plan, candidate.metadata)
            rows = connection.execute(
                """
                SELECT artifact_digest, plan_digest, planned_at, deleted_at, reason
                FROM sl_registry_retention_tombstones
                WHERE artifact_digest = ? AND plan_digest = ?
                LIMIT 2
                """,
                (candidate.digest, plan.digest),
            ).fetchall()
            if len(rows) != 1:
                raise RetentionDriftError("candidate tombstone is missing or ambiguous")
            tombstone = _tombstone_from_row(rows[0])
            _require_exact_tombstone(tombstone, plan)
        except sqlite3.Error as exc:
            _raise_sqlite(exc, operation="retention candidate inspection")
        finally:
            _close_connection(connection)
        if tombstone.deleted_at is None:
            return False
        self._verify_finalized_tombstone(tombstone, candidate)
        return True

    def _inspect_candidate_generation(
        self,
        plan: DigestConfirmedRetentionPlan,
        candidate: RetentionCandidate,
    ) -> None:
        record = self._store.inspect(_published_artifact(candidate.metadata))
        if (
            record.last_changed_at != candidate.last_changed_at
            or record.last_changed_ns != candidate.last_changed_ns
            or record.generation != candidate.generation
        ):
            raise RetentionDriftError("CAS candidate generation changed after planning")
        if record.last_changed_ns > _datetime_to_nanoseconds(plan.eligible_before):
            raise RetentionDriftError("CAS candidate generation has not satisfied physical grace")

    def _verify_finalized_candidate(
        self,
        plan: DigestConfirmedRetentionPlan,
        candidate: RetentionCandidate,
    ) -> None:
        if not self._candidate_is_finalized(plan, candidate):
            raise RetentionDriftError("expected finalized candidate remains pending")

    def _verify_finalized_tombstone(
        self,
        tombstone: _StoredTombstone,
        candidate: RetentionCandidate,
    ) -> None:
        authority_instant = self._registry._now()
        _require_finalized_chronology(tombstone, authority_instant)
        try:
            self._store.verify_absent(_published_artifact(candidate.metadata))
        except ArtifactStoreError:
            raise RetentionIntegrityError(
                "finalized retention evidence conflicts with the exact CAS key"
            ) from None

    def _revalidate_candidate(
        self,
        connection: sqlite3.Connection,
        plan: DigestConfirmedRetentionPlan,
        candidate: ArtifactMetadata,
    ) -> None:
        rows = connection.execute(
            """
            SELECT digest, artifact_class, byte_size, media_type,
                   storage_relpath, created_at, pinned
            FROM sl_registry_artifacts
            WHERE digest = ?
            LIMIT 2
            """,
            (candidate.digest,),
        ).fetchall()
        if len(rows) != 1:
            raise RetentionDriftError("retention candidate metadata is missing or ambiguous")
        current = _metadata_from_row(rows[0])
        if current != candidate:
            raise RetentionDriftError("retention candidate metadata changed after planning")
        if current.pinned or current.created_at > plan.eligible_before:
            raise RetentionDriftError("retention candidate is no longer policy-eligible")
        linked = connection.execute(
            """
            SELECT 1
            FROM sl_registry_run_artifacts
            WHERE artifact_digest = ?
            LIMIT 1
            """,
            (candidate.digest,),
        ).fetchone()
        if linked is not None:
            raise RetentionDriftError("retention candidate is linked to immutable run evidence")

    def _read_intent(self, plan: DigestConfirmedRetentionPlan) -> RetentionIntent:
        budget = _OperationBudget.start()
        connection = self._registry._connect(readonly=True)
        _install_progress_handler(connection, budget)
        try:
            rows = connection.execute(
                """
                SELECT artifact_digest, plan_digest, planned_at, deleted_at, reason
                FROM sl_registry_retention_tombstones
                WHERE plan_digest = ?
                ORDER BY artifact_digest
                LIMIT ?
                """,
                (plan.digest, _MAX_RETENTION_CANDIDATES + 1),
            ).fetchall()
            budget.checkpoint()
        except sqlite3.Error as exc:
            _raise_sqlite(exc, operation="retention intent read")
        finally:
            _close_connection(connection)
        if len(rows) > _MAX_RETENTION_CANDIDATES:
            raise RetentionIntegrityError("durable retention cohort exceeds its row bound")
        tombstones = tuple(_tombstone_from_row(row) for row in rows)
        actual_digests = tuple(tombstone.artifact_digest for tombstone in tombstones)
        if actual_digests != plan.plan.artifact_digests:
            raise RetentionDriftError("durable retention intent has missing or extra candidates")
        candidates = {candidate.digest: candidate for candidate in plan.candidates}
        pending: list[str] = []
        finalized: list[str] = []
        for tombstone in tombstones:
            budget.checkpoint()
            _require_exact_tombstone(tombstone, plan)
            if tombstone.deleted_at is None:
                pending.append(tombstone.artifact_digest)
            else:
                self._verify_finalized_tombstone(
                    tombstone,
                    candidates[tombstone.artifact_digest],
                )
                finalized.append(tombstone.artifact_digest)
        return RetentionIntent(plan.digest, tuple(pending), tuple(finalized))

    def _plan_from_envelope_row(
        self,
        row: sqlite3.Row,
        *,
        budget: _OperationBudget,
    ) -> DigestConfirmedRetentionPlan:
        """Strictly decode, hash, authenticate, and scope-bind one stored envelope."""

        try:
            plan_digest = require_digest(
                _strict_text(row["plan_digest"], "plan_digest"),
                "plan_digest",
            )
            payload_digest = require_digest(
                _strict_text(row["payload_digest"], "payload_digest"),
                "payload_digest",
            )
            payload_json = _strict_text(row["payload_json"], "payload_json")
            try:
                payload_bytes = payload_json.encode("utf-8")
            except (MemoryError, UnicodeEncodeError):
                raise RetentionIntegrityError(
                    "stored retention envelope is not valid bounded UTF-8"
                ) from None
            if not 2 <= len(payload_bytes) <= _MAX_PLAN_ENVELOPE_BYTES:
                raise RetentionIntegrityError("stored retention envelope exceeds its byte bound")
            registry_id = _require_registry_id(_strict_text(row["registry_id"], "registry_id"))
            cas_store_id = require_digest(
                _strict_text(row["cas_store_id"], "cas_store_id"),
                "cas_store_id",
            )
            planned_at = _parse_time(row["planned_at"], "retention planned_at")
            schema_version = row["schema_version"]
            if type(schema_version) is not int or schema_version != _PLAN_SCHEMA_VERSION:
                raise RetentionIntegrityError("stored retention envelope version is unsupported")
            if not hmac.compare_digest(hashlib.sha256(payload_bytes).hexdigest(), payload_digest):
                raise RetentionIntegrityError("stored retention envelope hash is invalid")
            budget.checkpoint()
            decoded = decode_retention_plan_payload(
                payload_json,
                checkpoint=budget.checkpoint,
            )
            plan = DigestConfirmedRetentionPlan(
                plan=decoded.plan,
                candidates=decoded.candidates,
                eligible_before=decoded.eligible_before,
                policy=decoded.policy,
                registry_id=decoded.registry_id,
                cas_store_id=decoded.cas_store_id,
                payload_digest=payload_digest,
                digest=plan_digest,
                schema_version=decoded.schema_version,
            )
        except (KeyError, IndexError, TypeError, ValueError, ValidationError):
            raise RetentionIntegrityError("stored retention envelope is invalid") from None
        if (
            plan.registry_id != registry_id
            or plan.cas_store_id != cas_store_id
            or plan.plan.planned_at != planned_at
            or plan.schema_version != schema_version
        ):
            raise RetentionIntegrityError("stored retention envelope columns disagree with payload")
        if not hmac.compare_digest(plan.registry_id, self._registry.registry_id):
            raise RetentionIntegrityError("retention envelope belongs to another registry")
        if not hmac.compare_digest(plan.cas_store_id, self._store.store_id):
            raise RetentionIntegrityError("retention envelope belongs to another CAS")
        self._registry.verify_retention_payload_signature(
            payload_bytes,
            store_id=plan.cas_store_id,
            signature=plan.digest,
        )
        budget.checkpoint()
        return plan

    def _validate_confirmation(
        self,
        plan: DigestConfirmedRetentionPlan,
        confirmed_digest: str,
        authority_instant: datetime,
    ) -> None:
        self._require_enabled()
        if type(plan) is not DigestConfirmedRetentionPlan:
            raise ValidationError("plan must be a DigestConfirmedRetentionPlan")
        authority_instant = require_utc(authority_instant, "authority_instant")
        if plan.plan.planned_at > authority_instant:
            raise RetentionConfirmationError(
                "plan timestamp is in the future relative to the registry clock authority"
            )
        if plan.registry_id != self._registry.registry_id:
            raise RetentionConfirmationError("plan belongs to a different registry identity")
        if plan.cas_store_id != self._store.store_id:
            raise RetentionConfirmationError("plan belongs to a different CAS identity")
        store_id = self._store.store_id
        if not _same_object(self._registry.artifact_verifier, self._store):
            raise RetentionConfirmationError("registry verifier authority changed")
        if self._registry.artifact_store_id != store_id:
            raise RetentionConfirmationError("registry CAS binding changed")
        if len(plan.candidates) > self.policy.max_candidates:
            raise RetentionConfirmationError("plan exceeds the configured candidate bound")
        if plan.total_bytes > self.policy.max_total_bytes:
            raise RetentionConfirmationError("plan exceeds the configured aggregate byte bound")
        try:
            expected_cutoff = plan.plan.planned_at - timedelta(
                seconds=self.policy.grace_period_seconds
            )
        except OverflowError:
            raise RetentionConfirmationError(
                "plan timestamp cannot satisfy the configured grace period"
            ) from None
        if plan.eligible_before != expected_cutoff:
            raise RetentionConfirmationError("plan does not match the configured grace period")
        if plan.policy != self.policy:
            raise RetentionConfirmationError("plan does not match the configured retention policy")
        payload = plan.canonical_payload_bytes()
        try:
            self._registry.verify_retention_payload_signature(
                payload,
                store_id=plan.cas_store_id,
                signature=plan.digest,
            )
        except RetentionPlanAuthenticationError:
            raise RetentionConfirmationError(
                "plan signature is not valid for this registry/CAS authority"
            ) from None
        try:
            confirmation = require_digest(confirmed_digest, "confirmed_digest")
        except ValidationError:
            raise RetentionConfirmationError(
                "confirmed_digest must be the authority-bound plan digest"
            ) from None
        if not hmac.compare_digest(confirmation, plan.digest):
            raise RetentionConfirmationError(
                "confirmed_digest does not match the authority-bound plan digest"
            )

    def _require_enabled(self) -> None:
        if not self.policy.enabled:
            raise RetentionDisabledError(
                "retention is disabled; enable it through explicit local policy"
            )


def _policy_payload(policy: RetentionPolicy) -> dict[str, object]:
    return {
        "enabled": policy.enabled,
        "grace_period_seconds": policy.grace_period_seconds,
        "max_candidates": policy.max_candidates,
        "max_total_bytes": policy.max_total_bytes,
    }


def _plan_payload(
    *,
    plan: RetentionPlan,
    candidates: tuple[RetentionCandidate, ...],
    eligible_before: datetime,
    policy: RetentionPolicy,
    registry_id: str,
    cas_store_id: str,
    schema_version: int,
) -> dict[str, object]:
    return {
        "candidates": [
            {
                "artifact_class": candidate.metadata.artifact_class.value,
                "byte_size": candidate.metadata.byte_size,
                "created_at": _db_time(candidate.metadata.created_at),
                "digest": candidate.metadata.digest,
                "generation_token": candidate.generation.token,
                "last_changed_at": _db_time(candidate.last_changed_at),
                "last_changed_ns": candidate.last_changed_ns,
                "media_type": candidate.metadata.media_type,
                "pinned": candidate.metadata.pinned,
                "storage_relpath": candidate.metadata.storage_relpath,
            }
            for candidate in candidates
        ],
        "cas_store_id": require_digest(cas_store_id, "cas_store_id"),
        "eligible_before": _db_time(eligible_before),
        "planned_at": _db_time(plan.planned_at),
        "policy": _policy_payload(policy),
        "reason": plan.reason,
        "registry_id": _require_registry_id(registry_id),
        "schema_version": schema_version,
    }


def decode_retention_plan_payload(
    payload_json: str,
    *,
    checkpoint: Callable[[], None],
) -> DecodedRetentionPlan:
    """Decode and semantically validate one bounded canonical retention payload.

    The supplied checkpoint is invoked before and during candidate decoding so readiness and
    execution paths can enforce their own monotonic operation budgets without coupling this pure
    contract boundary to a particular deadline implementation.
    """

    if type(payload_json) is not str or not callable(checkpoint):
        raise RetentionIntegrityError("stored retention envelope has an invalid boundary type")
    try:
        raw = decode_bounded_json(
            payload_json,
            maximum_bytes=_MAX_PLAN_ENVELOPE_BYTES,
            require_canonical=True,
        )
    except ValidationError:
        raise RetentionIntegrityError("stored retention envelope JSON is invalid") from None
    try:
        return validate_decoded_retention_plan(
            payload_json,
            raw,
            checkpoint=checkpoint,
        )
    except RetentionIntegrityError:
        raise
    except (ArtifactStoreError, TypeError, ValueError, ValidationError):
        raise RetentionIntegrityError("stored retention envelope is invalid") from None


def validate_decoded_retention_plan(
    payload_json: str,
    decoded: object,
    *,
    checkpoint: Callable[[], None],
) -> DecodedRetentionPlan:
    """Validate a retention object produced by a caller's bounded strict JSON decoder."""

    if type(payload_json) is not str or not callable(checkpoint):
        raise RetentionIntegrityError("stored retention envelope has an invalid boundary type")
    raw = decoded
    checkpoint()
    if type(raw) is not dict or set(raw) != {
        "candidates",
        "cas_store_id",
        "eligible_before",
        "planned_at",
        "policy",
        "reason",
        "registry_id",
        "schema_version",
    }:
        raise RetentionIntegrityError("stored retention envelope has an invalid shape")
    try:
        recanonicalized = canonical_json(raw)
    except (RecursionError, MemoryError, TypeError, ValueError, ValidationError):
        raise RetentionIntegrityError("stored retention envelope JSON is invalid") from None
    if recanonicalized != payload_json:
        raise RetentionIntegrityError("stored retention envelope is not canonical JSON")
    schema_version = raw["schema_version"]
    if type(schema_version) is not int or schema_version != _PLAN_SCHEMA_VERSION:
        raise RetentionIntegrityError("stored retention envelope version is unsupported")
    registry_id = _require_registry_id(_strict_text(raw["registry_id"], "registry_id"))
    cas_store_id = require_digest(
        _strict_text(raw["cas_store_id"], "cas_store_id"),
        "cas_store_id",
    )
    planned_at = _parse_time(raw["planned_at"], "retention planned_at")
    eligible_before = _parse_time(raw["eligible_before"], "retention eligible_before")
    reason = _require_reason(_strict_text(raw["reason"], "retention reason"))
    policy_raw = raw["policy"]
    if type(policy_raw) is not dict or set(policy_raw) != {
        "enabled",
        "grace_period_seconds",
        "max_candidates",
        "max_total_bytes",
    }:
        raise RetentionIntegrityError("stored retention policy has an invalid shape")
    policy = RetentionPolicy(
        enabled=policy_raw["enabled"],
        grace_period_seconds=policy_raw["grace_period_seconds"],
        max_candidates=policy_raw["max_candidates"],
        max_total_bytes=policy_raw["max_total_bytes"],
    )
    candidates_raw = raw["candidates"]
    if (
        type(candidates_raw) is not list
        or not 1 <= len(candidates_raw) <= _MAX_RETENTION_CANDIDATES
    ):
        raise RetentionIntegrityError("stored retention candidates exceed their bound")
    candidates: list[RetentionCandidate] = []
    for candidate_raw in candidates_raw:
        checkpoint()
        if type(candidate_raw) is not dict or set(candidate_raw) != {
            "artifact_class",
            "byte_size",
            "created_at",
            "digest",
            "generation_token",
            "last_changed_at",
            "last_changed_ns",
            "media_type",
            "pinned",
            "storage_relpath",
        }:
            raise RetentionIntegrityError("stored retention candidate has an invalid shape")
        artifact_class_raw = _strict_text(
            candidate_raw["artifact_class"],
            "artifact_class",
        )
        metadata = ArtifactMetadata(
            digest=_strict_text(candidate_raw["digest"], "digest"),
            artifact_class=ArtifactClass(artifact_class_raw),
            byte_size=candidate_raw["byte_size"],
            media_type=_strict_text(candidate_raw["media_type"], "media_type"),
            storage_relpath=_strict_text(
                candidate_raw["storage_relpath"],
                "storage_relpath",
            ),
            created_at=_parse_time(candidate_raw["created_at"], "artifact created_at"),
            pinned=candidate_raw["pinned"],
        )
        try:
            generation = ArtifactGeneration(
                _strict_text(candidate_raw["generation_token"], "generation_token")
            )
        except ArtifactStoreError:
            raise RetentionIntegrityError(
                "stored retention candidate generation is invalid"
            ) from None
        candidates.append(
            RetentionCandidate(
                metadata,
                _parse_time(candidate_raw["last_changed_at"], "artifact last_changed_at"),
                _bounded_integer(
                    candidate_raw["last_changed_ns"],
                    "last_changed_ns",
                    minimum=-(2**63),
                    maximum=2**63 - 1,
                ),
                generation,
            )
        )
    candidates_tuple = tuple(candidates)
    try:
        expected_eligible_before = planned_at - timedelta(seconds=policy.grace_period_seconds)
    except OverflowError:
        raise RetentionIntegrityError(
            "stored retention policy cannot produce a valid grace cutoff"
        ) from None
    if eligible_before != expected_eligible_before:
        raise RetentionIntegrityError("stored retention grace cutoff is inconsistent")
    if len(candidates_tuple) > policy.max_candidates:
        raise RetentionIntegrityError("stored retention candidates exceed the policy bound")
    total_bytes = 0
    for candidate in candidates_tuple:
        checkpoint()
        total_bytes += candidate.metadata.byte_size
        if total_bytes > policy.max_total_bytes:
            raise RetentionIntegrityError("stored retention bytes exceed the policy bound")
        if (
            candidate.metadata.created_at > eligible_before
            or candidate.last_changed_ns > _datetime_to_nanoseconds(eligible_before)
        ):
            raise RetentionIntegrityError(
                "stored retention candidate has not satisfied its grace cutoff"
            )
    if not policy.enabled:
        raise RetentionIntegrityError("stored retention plan was created under disabled policy")
    try:
        plan = RetentionPlan(
            tuple(candidate.digest for candidate in candidates_tuple),
            planned_at,
            reason,
        )
    except ValidationError:
        raise RetentionIntegrityError("stored retention candidate set is invalid") from None
    return DecodedRetentionPlan(
        plan=plan,
        candidates=candidates_tuple,
        eligible_before=eligible_before,
        policy=policy,
        registry_id=registry_id,
        cas_store_id=cas_store_id,
        schema_version=schema_version,
    )


def _metadata_from_row(row: sqlite3.Row) -> ArtifactMetadata:
    try:
        digest_raw = row["digest"]
        artifact_class_raw = row["artifact_class"]
        byte_size_raw = row["byte_size"]
        media_type_raw = row["media_type"]
        storage_relpath_raw = row["storage_relpath"]
        pinned_raw = row["pinned"]
        if (
            type(digest_raw) is not str
            or type(artifact_class_raw) is not str
            or type(media_type_raw) is not str
            or type(storage_relpath_raw) is not str
        ):
            raise ValueError
        if type(byte_size_raw) is not int:
            raise ValueError
        if type(pinned_raw) is not int or pinned_raw not in (0, 1):
            raise ValueError
        metadata = ArtifactMetadata(
            digest=digest_raw,
            artifact_class=ArtifactClass(artifact_class_raw),
            byte_size=byte_size_raw,
            media_type=media_type_raw,
            storage_relpath=storage_relpath_raw,
            created_at=_parse_time(row["created_at"], "artifact created_at"),
            pinned=bool(pinned_raw),
        )
        _published_artifact(metadata)
        return metadata
    except (KeyError, TypeError, ValueError, ValidationError, ArtifactStoreError):
        raise RetentionIntegrityError("stored artifact metadata is invalid") from None


def _tombstone_from_row(row: sqlite3.Row) -> _StoredTombstone:
    """Decode one tombstone without SQLite's permissive type coercions."""

    try:
        artifact_digest = _strict_text(row["artifact_digest"], "artifact_digest")
        plan_digest = _strict_text(row["plan_digest"], "plan_digest")
        reason = _strict_text(row["reason"], "reason")
        require_digest(artifact_digest, "artifact_digest")
        require_digest(plan_digest, "plan_digest")
        _require_reason(reason)
        planned_at = _parse_time(row["planned_at"], "retention planned_at")
        deleted_raw = row["deleted_at"]
        deleted_at = (
            None if deleted_raw is None else _parse_time(deleted_raw, "retention deleted_at")
        )
        if deleted_at is not None and deleted_at < planned_at:
            raise ValueError
        return _StoredTombstone(
            artifact_digest=artifact_digest,
            plan_digest=plan_digest,
            planned_at=planned_at,
            deleted_at=deleted_at,
            reason=reason,
        )
    except RetentionIntegrityError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, ValidationError):
        raise RetentionIntegrityError("stored retention tombstone is invalid") from None


def _require_exact_tombstone(
    tombstone: _StoredTombstone,
    plan: DigestConfirmedRetentionPlan,
) -> None:
    if (
        not hmac.compare_digest(tombstone.plan_digest, plan.digest)
        or tombstone.planned_at != plan.plan.planned_at
        or tombstone.reason != plan.plan.reason
    ):
        raise RetentionDriftError("durable retention intent metadata changed")


def _require_finalized_chronology(
    tombstone: _StoredTombstone,
    authority_instant: datetime,
) -> None:
    instant = require_utc(authority_instant, "authority_instant")
    if tombstone.deleted_at is None:
        raise RetentionIntegrityError("pending tombstone has no finalized chronology")
    if not tombstone.planned_at <= tombstone.deleted_at <= instant:
        raise RetentionIntegrityError("finalized retention chronology is invalid")


def _published_artifact(candidate: ArtifactMetadata) -> PublishedArtifact:
    try:
        return PublishedArtifact(
            digest=candidate.digest,
            byte_size=candidate.byte_size,
            storage_key=candidate.storage_relpath,
        )
    except ArtifactStoreError:
        raise RetentionIntegrityError(
            "stored artifact key is not canonical for its digest"
        ) from None


def _parse_time(value: object, field_name: str) -> datetime:
    if type(value) is not str or len(value) != 27 or not value.endswith("Z"):
        raise RetentionIntegrityError(f"stored {field_name} is not canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise RetentionIntegrityError(f"stored {field_name} is not ISO-8601") from None
    normalized = require_utc(parsed, field_name)
    if _db_time(normalized) != value:
        raise RetentionIntegrityError(f"stored {field_name} is not canonical UTC text")
    return normalized


def _db_time(value: datetime) -> str:
    return require_utc(value, "timestamp").isoformat(timespec="microseconds").replace("+00:00", "Z")


def _datetime_to_nanoseconds(value: datetime) -> int:
    normalized = require_utc(value, "timestamp")
    delta = normalized - _UNIX_EPOCH
    exact = (
        delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000
    )
    return _bounded_integer(
        exact,
        "timestamp nanoseconds",
        minimum=-(2**63),
        maximum=2**63 - 1,
    )


def _nanoseconds_to_display_time(value: int) -> datetime:
    exact = _bounded_integer(
        value,
        "last_changed_ns",
        minimum=-(2**63),
        maximum=2**63 - 1,
    )
    seconds, nanoseconds = divmod(exact, 1_000_000_000)
    try:
        return _UNIX_EPOCH + timedelta(
            seconds=seconds,
            microseconds=nanoseconds // 1_000,
        )
    except (OverflowError, ValueError):
        raise ValidationError("last_changed_ns is outside the supported UTC range") from None


def _require_reason(value: str) -> str:
    if type(value) is not str or _REASON_CODE.fullmatch(value) is None:
        raise ValidationError(
            "reason must be a lowercase policy identifier of at most 128 ASCII characters"
        )
    return value


def _require_registry_id(value: str) -> str:
    if type(value) is not str or _REGISTRY_ID.fullmatch(value) is None:
        raise ValidationError("registry_id must be 32 lowercase hexadecimal characters")
    return value


def _strict_text(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise RetentionIntegrityError(f"stored {field_name} has the wrong storage type")
    return value


def _bounded_integer(value: int, field_name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{field_name} must be an integer in [{minimum}, {maximum}]")
    return value


def _validated_digest_tuple(
    value: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple or (not allow_empty and not value):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValidationError(f"{field_name} must be a {qualifier} tuple")
    if len(value) > _MAX_RETENTION_CANDIDATES:
        raise ValidationError(f"{field_name} may not exceed {_MAX_RETENTION_CANDIDATES} entries")
    if value != tuple(sorted(set(value))):
        raise ValidationError(f"{field_name} must contain unique sorted digests")
    for digest in value:
        require_digest(digest, field_name)
    return value


def _has_running_work(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sl_registry_jobs WHERE state = 'running' LIMIT 1"
        ).fetchone()
        is not None
    )


def _install_progress_handler(
    connection: sqlite3.Connection,
    budget: _OperationBudget,
) -> None:
    try:
        connection.set_progress_handler(budget.expired, _SQLITE_PROGRESS_INSTRUCTIONS)
    except sqlite3.Error as exc:
        _raise_sqlite(exc, operation="retention deadline installation")


def _close_connection(connection: sqlite3.Connection) -> None:
    """Remove callbacks and close once without replacing an active failure.

    SQLite descriptor cleanup is best effort only in the sense that every independent cleanup
    action is attempted. A cleanup failure is never discarded: it becomes a sanitized note on the
    exception already propagating from the operation, or a typed integrity failure when cleanup is
    the only failure. Descriptor close is deliberately not retried because an interrupted close
    has platform-dependent ownership semantics.
    """

    primary = sys.exception()
    failures: list[str] = []
    try:
        connection.set_progress_handler(None, 0)
    except sqlite3.Error as exc:
        failures.append(f"progress-handler removal ({type(exc).__name__})")
    try:
        connection.close()
    except sqlite3.Error as exc:
        failures.append(f"connection close ({type(exc).__name__})")
    if not failures:
        return

    if primary is not None:
        _add_failure_note(
            primary,
            "retention connection cleanup also failed closed: " + ", ".join(failures),
        )
        return
    raise RetentionIntegrityError("retention connection cleanup failed") from None


def _same_object(left: object | None, right: object) -> bool:
    """Compare authority identity without inviting static type narrowing."""

    return left is right


def _rollback(connection: sqlite3.Connection, *, primary: BaseException) -> None:
    if connection.in_transaction:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            _add_failure_note(primary, "retention transaction rollback also failed closed")


def _add_failure_note(primary: BaseException, note: str) -> None:
    """Annotate a primary failure without permitting diagnostic machinery to replace it."""

    try:
        primary.add_note(note)
    except Exception:
        # A hostile custom exception may override ``add_note``. The original exception still
        # carries the authoritative failure and must continue propagating unchanged.
        return


def _raise_sqlite(exc: sqlite3.Error, *, operation: str) -> NoReturn:
    code = getattr(exc, "sqlite_errorcode", None)
    primary_code = None if type(code) is not int else code & 0xFF
    if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        raise BusyError(f"{operation} remained busy past its configured bound") from None
    if primary_code == sqlite3.SQLITE_INTERRUPT:
        raise RetentionIntegrityError(f"{operation} exceeded its bounded deadline") from None
    raise RetentionIntegrityError(f"SQLite rejected {operation}") from None


def _artifact_cause_code(exc: ArtifactStoreError) -> str:
    return "artifact_integrity" if "Integrity" in type(exc).__name__ else "artifact_boundary"


def _cause_code(exc: RegistryError | ArtifactStoreError) -> str:
    if isinstance(exc, RegistryError):
        return exc.code
    return _artifact_cause_code(exc)


__all__ = [
    "DecodedRetentionPlan",
    "DigestConfirmedRetentionPlan",
    "RetentionActiveWorkError",
    "RetentionCandidate",
    "RetentionConfirmationError",
    "RetentionController",
    "RetentionDisabledError",
    "RetentionDriftError",
    "RetentionError",
    "RetentionExecution",
    "RetentionIntegrityError",
    "RetentionIntent",
    "RetentionPartialFailure",
    "RetentionPlanAuthenticationError",
    "RetentionPolicy",
    "RetentionRecovery",
    "decode_retention_plan_payload",
    "validate_decoded_retention_plan",
]
