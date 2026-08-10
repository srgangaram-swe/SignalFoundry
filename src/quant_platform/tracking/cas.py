"""Descriptor-anchored local content-addressed artifact storage.

The store is a filesystem mechanism, not a retention or registry policy. It
publishes bounded immutable bytes beneath one configured root and returns only
a digest, byte count, and deterministic root-relative key. All security-
relevant operations are anchored to already-open directory descriptors so a
symlink or directory replacement cannot redirect an operation after validation.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import math
import os
import secrets
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Final, Protocol, cast


class _FileLockModule(Protocol):
    """Minimum POSIX advisory-lock surface required by the CAS."""

    LOCK_EX: int
    LOCK_NB: int
    LOCK_SH: int
    LOCK_UN: int

    def flock(self, descriptor: int, operation: int) -> None:
        """Apply one advisory lock operation to an open descriptor."""


def _load_file_lock_module() -> _FileLockModule | None:
    """Load POSIX advisory locks without making package import platform-specific."""

    try:
        module = importlib.import_module("fcntl")
    except ModuleNotFoundError:
        return None
    return cast(_FileLockModule, module)


fcntl: _FileLockModule | None = _load_file_lock_module()

SHA256_HEX_LENGTH: Final = 64
DEFAULT_MAX_ARTIFACT_BYTES: Final = 64 * 1024 * 1024
DEFAULT_MAX_MANIFEST_BYTES: Final = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 1024 * 1024 * 1024
MAX_IN_MEMORY_READ_BYTES: Final = 64 * 1024 * 1024
PUBLICATION_LOCK_TIMEOUT_SECONDS: Final = 5.0
PUBLICATION_LOCK_POLL_SECONDS: Final = 0.001
COPY_CHUNK_BYTES: Final = 1024 * 1024
MAX_STAGING_ATTEMPTS: Final = 8
DEFAULT_INVENTORY_PAGE_SIZE: Final = 128
DEFAULT_INVENTORY_MAX_SCANNED_ENTRIES: Final = 10_000
DEFAULT_INVENTORY_TIME_BUDGET_SECONDS: Final = 5.0
MAX_INVENTORY_PAGE_SIZE: Final = 1_000
MAX_INVENTORY_SCANNED_ENTRIES: Final = 100_000
MAX_INVENTORY_TIME_BUDGET_SECONDS: Final = 60.0
MAX_CAS_PATH_BYTES: Final = 4_096
MAX_CAS_PATH_COMPONENT_BYTES: Final = 255

_STORE_ID_FILENAME: Final = ".store-id"
_STORE_ID_TEMP_PREFIX: Final = ".store-id-"
_STORE_ID_TEMP_SUFFIX: Final = ".tmp"
_STORE_ID_HEX_LENGTH: Final = 64
_STORE_ID_HEADER: Final = b"SIGNALATTICE_CAS_STORE_ID_V1\n"
_STORE_ID_CHECKSUM_DOMAIN: Final = b"signalattice.cas.store-id.v1\x00"
_ARTIFACT_GENERATION_DOMAIN: Final = b"signalattice.cas.artifact-generation.v1\x00"
_STORE_ID_PAYLOAD_BYTES: Final = (
    len(_STORE_ID_HEADER) + _STORE_ID_HEX_LENGTH + 1 + SHA256_HEX_LENGTH + 1
)

_STAGING_PREFIX: Final = "publish-"
_STAGING_SUFFIX: Final = ".tmp"
_STAGING_TOKEN_HEX_LENGTH: Final = 48
_UNIX_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)

_DIRECTORY_FLAGS: Final = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)

_DARWIN_ACL_GET_FD_NP: Any | None = None
_DARWIN_ACL_FREE: Any | None = None
if sys.platform == "darwin":
    _darwin_libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    _DARWIN_ACL_GET_FD_NP = _darwin_libsystem.acl_get_fd_np
    _DARWIN_ACL_GET_FD_NP.argtypes = [ctypes.c_int, ctypes.c_int]
    _DARWIN_ACL_GET_FD_NP.restype = ctypes.c_void_p
    _DARWIN_ACL_FREE = _darwin_libsystem.acl_free
    _DARWIN_ACL_FREE.argtypes = [ctypes.c_void_p]
    _DARWIN_ACL_FREE.restype = ctypes.c_int


class ArtifactStoreError(RuntimeError):
    """Base error for a failed or unsafe artifact-store operation.

    ``errno_code`` retains the machine-readable operating-system cause without
    retaining an ``OSError`` whose text may contain a sensitive source path.
    """

    def __init__(self, message: str, *, errno_code: int | None = None) -> None:
        super().__init__(message)
        self.errno_code = errno_code


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when artifact bytes or filesystem identity do not verify."""


class ArtifactBoundaryError(ArtifactStoreError):
    """Raised when a path, file type, or resource limit crosses the store boundary."""


def _add_failure_note(primary: BaseException, note: str) -> None:
    """Annotate a primary failure without allowing hostile diagnostics to replace it."""

    try:
        primary.add_note(note)
    except Exception:
        return


@dataclass(slots=True)
class _CleanupState:
    """Best-effort close-all state that preserves one authoritative primary failure."""

    primary: BaseException | None
    failures: list[tuple[str, str, int | None]]

    @classmethod
    def start(cls, primary: BaseException | None = None) -> _CleanupState:
        return cls(sys.exception() if primary is None else primary, [])

    def close(self, label: str, *descriptors: int | None) -> None:
        """Close every distinct descriptor once; interrupted closes are never retried."""

        observed: set[int] = set()
        for descriptor in descriptors:
            if descriptor is None or descriptor in observed:
                continue
            observed.add(descriptor)
            try:
                os.close(descriptor)
            except Exception as exc:
                self._record(label, exc)

    def attempt(self, label: str, operation: Callable[[], None]) -> None:
        """Attempt one independent cleanup action without stopping later actions."""

        try:
            operation()
        except Exception as exc:
            self._record(label, exc)

    def _record(self, label: str, exc: Exception) -> None:
        errno_code: int | None = None
        if isinstance(exc, ArtifactStoreError):
            errno_code = exc.errno_code
        elif isinstance(exc, OSError):
            errno_code = exc.errno
        self.failures.append((label, type(exc).__name__, errno_code))

    def finish(self, role: str) -> None:
        """Attach sanitized evidence to a primary, or raise a typed cleanup-only failure."""

        if not self.failures:
            return
        summary = ", ".join(f"{label} ({kind})" for label, kind, _ in self.failures)
        if self.primary is not None:
            _add_failure_note(self.primary, f"{role} also failed closed: {summary}")
            return
        raise ArtifactStoreError(
            f"{role} failed",
            errno_code=self.failures[0][2],
        ) from None


def _close_descriptors(*descriptors: int | None, role: str) -> None:
    """Close all supplied descriptors while respecting the active exception lifecycle."""

    cleanup = _CleanupState.start()
    cleanup.close("descriptor close", *descriptors)
    cleanup.finish(role)


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """Immutable public identity for one published artifact.

    ``storage_key`` is a POSIX root-relative identifier, never an absolute
    workstation path. The key is derived rather than caller-selected.
    """

    digest: str
    byte_size: int
    storage_key: str

    def __post_init__(self) -> None:
        _validate_digest(self.digest)
        if type(self.byte_size) is not int:
            raise ArtifactBoundaryError("artifact byte_size must be an integer")
        if self.byte_size < 0:
            raise ArtifactBoundaryError("artifact byte_size must be non-negative")
        if self.byte_size > MAX_ARTIFACT_BYTES:
            raise ArtifactBoundaryError(f"artifact byte_size cannot exceed {MAX_ARTIFACT_BYTES}")
        if type(self.storage_key) is not str:
            raise ArtifactBoundaryError("artifact storage key must be exact text")
        if self.storage_key != _storage_key(self.digest):
            raise ArtifactBoundaryError("artifact storage key does not match its digest")


@dataclass(frozen=True, slots=True)
class ArtifactGeneration:
    """Opaque exact descriptor-generation identity safe to persist and compare."""

    token: str

    def __post_init__(self) -> None:
        _validate_generation_token(self.token)


@dataclass(frozen=True, slots=True)
class ArtifactInventoryRecord:
    """Path-free verified identity of one grace-eligible CAS object.

    ``last_changed_ns`` is the exact conservative grace timestamp; ``last_changed_at`` is its
    human-readable UTC representation truncated to Python microseconds. ``generation`` is an
    opaque digest over store/object identity plus exact descriptor metadata, so raw device and
    inode identifiers are never exposed. ``storage_key`` is deterministic and root-relative.
    """

    digest: str
    byte_size: int
    storage_key: str
    last_changed_at: datetime
    last_changed_ns: int
    generation: ArtifactGeneration

    def __post_init__(self) -> None:
        PublishedArtifact(
            digest=self.digest,
            byte_size=self.byte_size,
            storage_key=self.storage_key,
        )
        object.__setattr__(
            self,
            "last_changed_at",
            _require_utc_datetime(self.last_changed_at, role="last_changed_at"),
        )
        exact_ns = _require_timestamp_nanoseconds(
            self.last_changed_ns,
            role="last_changed_ns",
        )
        if _nanoseconds_to_utc(exact_ns, role="last_changed_ns") != self.last_changed_at:
            raise ArtifactBoundaryError(
                "last_changed_at must be the canonical display time for last_changed_ns"
            )
        if type(self.generation) is not ArtifactGeneration:
            raise ArtifactBoundaryError("artifact generation must be a typed opaque identity")


@dataclass(frozen=True, slots=True)
class ArtifactInventoryCursor:
    """Immutable continuation bound to one store and one exact cutoff.

    The cursor is deliberately a typed local value rather than a raw digest.
    Advancing the grace cutoff or presenting it to another CAS fails closed,
    preventing a rescan from silently omitting newly eligible lower keys.
    """

    after_digest: str
    cutoff: datetime
    store_id: str

    def __post_init__(self) -> None:
        _validate_digest(self.after_digest)
        object.__setattr__(
            self,
            "cutoff",
            _require_utc_datetime(self.cutoff, role="cursor cutoff"),
        )
        _validate_store_id(self.store_id)


@dataclass(frozen=True, slots=True)
class ArtifactInventoryPage:
    """One deterministic page from a fully bounded object-tree inspection."""

    records: tuple[ArtifactInventoryRecord, ...]
    next_cursor: ArtifactInventoryCursor | None
    scanned_entries: int

    def __post_init__(self) -> None:
        if type(self.records) is not tuple or any(
            type(record) is not ArtifactInventoryRecord for record in self.records
        ):
            raise ArtifactBoundaryError(
                "artifact inventory records must be a tuple of typed records"
            )
        digests = tuple(record.digest for record in self.records)
        if digests != tuple(sorted(set(digests))):
            raise ArtifactBoundaryError("artifact inventory records must be unique and ordered")
        if len(self.records) > MAX_INVENTORY_PAGE_SIZE:
            raise ArtifactBoundaryError(
                f"artifact inventory page cannot exceed {MAX_INVENTORY_PAGE_SIZE} records"
            )
        if self.next_cursor is not None:
            if type(self.next_cursor) is not ArtifactInventoryCursor:
                raise ArtifactBoundaryError("artifact inventory cursor must be typed")
            if not digests or self.next_cursor.after_digest != digests[-1]:
                raise ArtifactBoundaryError("artifact inventory cursor must end its page")
        scanned = _bounded_non_negative_integer(
            self.scanned_entries,
            role="scanned_entries",
        )
        if scanned > MAX_INVENTORY_SCANNED_ENTRIES:
            raise ArtifactBoundaryError(
                f"scanned_entries cannot exceed {MAX_INVENTORY_SCANNED_ENTRIES}"
            )
        if scanned < len(self.records):
            raise ArtifactBoundaryError("scanned_entries cannot be smaller than returned records")


@dataclass(frozen=True, slots=True)
class StagingInventoryRecord:
    """Path-free identity of one stale, private CAS staging file.

    Discovery is deliberately separate from cleanup. Returning this record does
    not authorize removal and the store exposes no automatic stale-file deletion.
    """

    staging_id: str
    byte_size: int
    last_changed_at: datetime

    def __post_init__(self) -> None:
        _validate_staging_id(self.staging_id)
        _bounded_non_negative_integer(self.byte_size, role="staging byte_size")
        object.__setattr__(
            self,
            "last_changed_at",
            _require_utc_datetime(self.last_changed_at, role="last_changed_at"),
        )


@dataclass(frozen=True, slots=True)
class StagingInventoryCursor:
    """Immutable stale-staging continuation bound to a store and cutoff."""

    after_staging_id: str
    cutoff: datetime
    store_id: str

    def __post_init__(self) -> None:
        _validate_staging_id(self.after_staging_id)
        object.__setattr__(
            self,
            "cutoff",
            _require_utc_datetime(self.cutoff, role="cursor cutoff"),
        )
        _validate_store_id(self.store_id)


@dataclass(frozen=True, slots=True)
class StagingInventoryPage:
    """One deterministic page from a fully bounded staging-tree inspection."""

    records: tuple[StagingInventoryRecord, ...]
    next_cursor: StagingInventoryCursor | None
    scanned_entries: int

    def __post_init__(self) -> None:
        if type(self.records) is not tuple or any(
            type(record) is not StagingInventoryRecord for record in self.records
        ):
            raise ArtifactBoundaryError(
                "staging inventory records must be a tuple of typed records"
            )
        identifiers = tuple(record.staging_id for record in self.records)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ArtifactBoundaryError("staging inventory records must be unique and ordered")
        if len(self.records) > MAX_INVENTORY_PAGE_SIZE:
            raise ArtifactBoundaryError(
                f"staging inventory page cannot exceed {MAX_INVENTORY_PAGE_SIZE} records"
            )
        if self.next_cursor is not None:
            if type(self.next_cursor) is not StagingInventoryCursor:
                raise ArtifactBoundaryError("staging inventory cursor must be typed")
            if not identifiers or self.next_cursor.after_staging_id != identifiers[-1]:
                raise ArtifactBoundaryError("staging inventory cursor must end its page")
        scanned = _bounded_non_negative_integer(
            self.scanned_entries,
            role="scanned_entries",
        )
        if scanned > MAX_INVENTORY_SCANNED_ENTRIES:
            raise ArtifactBoundaryError(
                f"scanned_entries cannot exceed {MAX_INVENTORY_SCANNED_ENTRIES}"
            )
        if scanned < len(self.records):
            raise ArtifactBoundaryError("scanned_entries cannot be smaller than returned records")


@dataclass(slots=True)
class _InventoryBudget:
    """Cooperative entry and elapsed-time bounds for one local tree scan."""

    maximum_entries: int
    deadline_ns: int
    last_clock_ns: int
    scanned_entries: int = 0

    @classmethod
    def start(cls, *, maximum_entries: int, time_budget_seconds: float) -> _InventoryBudget:
        entry_limit = _bounded_inventory_integer(
            maximum_entries,
            role="max_scan_entries",
            maximum=MAX_INVENTORY_SCANNED_ENTRIES,
        )
        seconds = _bounded_inventory_seconds(time_budget_seconds)
        started = time.monotonic_ns()
        duration_ns = math.ceil(seconds * 1_000_000_000)
        return cls(
            maximum_entries=entry_limit,
            deadline_ns=started + duration_ns,
            last_clock_ns=started,
        )

    def checkpoint(self) -> None:
        """Fail when the cooperative monotonic deadline is exhausted."""

        observed = time.monotonic_ns()
        if observed < self.last_clock_ns:
            raise ArtifactIntegrityError("inventory monotonic clock moved backwards")
        self.last_clock_ns = observed
        if observed > self.deadline_ns:
            raise ArtifactBoundaryError("inventory exceeded its caller time budget")

    def observe_entry(self) -> None:
        """Account for one directory entry before retaining or interpreting it."""

        self.checkpoint()
        self.scanned_entries += 1
        if self.scanned_entries > self.maximum_entries:
            raise ArtifactBoundaryError("inventory exceeded its caller scan-entry bound")


@dataclass(frozen=True, slots=True)
class _Identity:
    """Descriptor identity and mutation-sensitive metadata."""

    device: int
    inode: int
    file_type: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def regular(cls, value: os.stat_result, *, role: str) -> _Identity:
        if not stat.S_ISREG(value.st_mode):
            raise ArtifactBoundaryError(f"{role} must be a regular file")
        return cls._from_stat(value)

    @classmethod
    def directory(cls, value: os.stat_result, *, role: str) -> _Identity:
        if not stat.S_ISDIR(value.st_mode):
            raise ArtifactBoundaryError(f"{role} must be a real directory")
        return cls._from_stat(value)

    @classmethod
    def _from_stat(cls, value: os.stat_result) -> _Identity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            file_type=stat.S_IFMT(value.st_mode),
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    def same_object(self, other: _Identity) -> bool:
        """Return whether two observations identify the same filesystem object."""

        return (
            self.device,
            self.inode,
            self.file_type,
        ) == (
            other.device,
            other.inode,
            other.file_type,
        )


def _artifact_generation(
    store_id: str,
    digest: str,
    identity: _Identity,
) -> ArtifactGeneration:
    """Hash exact descriptor state into a path-free, non-secret generation identity."""

    if type(identity) is not _Identity:
        raise ArtifactIntegrityError("CAS object generation identity is malformed")
    components = (
        _generation_integer(identity.device, role="device", lower=0, upper=2**64 - 1),
        _generation_integer(identity.inode, role="inode", lower=0, upper=2**64 - 1),
        _generation_integer(identity.file_type, role="file type", lower=0, upper=2**32 - 1),
        _generation_integer(identity.size, role="size", lower=0, upper=MAX_ARTIFACT_BYTES),
        _generation_integer(
            identity.mtime_ns,
            role="mtime_ns",
            lower=-(2**63),
            upper=2**63 - 1,
        ),
        _generation_integer(
            identity.ctime_ns,
            role="ctime_ns",
            lower=-(2**63),
            upper=2**63 - 1,
        ),
    )
    message = b"\x00".join(
        (
            _ARTIFACT_GENERATION_DOMAIN,
            _validate_store_id(store_id).encode("ascii"),
            _validate_digest(digest).encode("ascii"),
            *(str(component).encode("ascii") for component in components),
        )
    )
    return ArtifactGeneration(hashlib.sha256(message).hexdigest())


def _generation_integer(value: object, *, role: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ArtifactIntegrityError(f"CAS object generation {role} is outside its exact domain")
    return value


@dataclass(slots=True)
class _OpenedFile:
    descriptor: int
    parent_descriptor: int
    name: str
    identity: _Identity

    def close(self) -> None:
        _close_descriptors(
            self.descriptor,
            self.parent_descriptor,
            role="artifact source descriptor cleanup",
        )


@dataclass(slots=True)
class _StoreDescriptors:
    root: int
    staging: int
    objects: int

    def close(self) -> None:
        _close_descriptors(
            self.objects,
            self.staging,
            self.root,
            role="CAS store descriptor cleanup",
        )


def _validate_digest(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactBoundaryError("artifact digest must be a lowercase SHA-256")
    return value


def _validate_generation_token(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactBoundaryError("artifact generation token must be a lowercase SHA-256")
    return value


def _validate_store_id(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != _STORE_ID_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactBoundaryError("CAS store identity must be canonical lowercase hexadecimal")
    return value


def _store_id_payload(store_id: str) -> bytes:
    canonical = _validate_store_id(store_id).encode("ascii")
    checksum = hashlib.sha256(_STORE_ID_CHECKSUM_DOMAIN + canonical).hexdigest().encode("ascii")
    return _STORE_ID_HEADER + canonical + b"\n" + checksum + b"\n"


def _parse_store_id_payload(payload: bytes) -> str:
    if type(payload) is not bytes or len(payload) != _STORE_ID_PAYLOAD_BYTES:
        raise ArtifactIntegrityError("CAS store identity marker has an invalid size")
    lines = payload.splitlines(keepends=True)
    if len(lines) != 3 or lines[0] != _STORE_ID_HEADER:
        raise ArtifactIntegrityError("CAS store identity marker is not canonical")
    try:
        store_id = lines[1][:-1].decode("ascii")
        checksum = lines[2][:-1].decode("ascii")
    except UnicodeDecodeError:
        raise ArtifactIntegrityError("CAS store identity marker is not canonical") from None
    if not lines[1].endswith(b"\n") or not lines[2].endswith(b"\n"):
        raise ArtifactIntegrityError("CAS store identity marker is not canonical")
    try:
        canonical_id = _validate_store_id(store_id)
        canonical_checksum = _validate_digest(checksum)
    except ArtifactBoundaryError:
        raise ArtifactIntegrityError("CAS store identity marker is not canonical") from None
    expected = hashlib.sha256(_STORE_ID_CHECKSUM_DOMAIN + canonical_id.encode("ascii")).hexdigest()
    if not secrets.compare_digest(canonical_checksum, expected):
        raise ArtifactIntegrityError("CAS store identity marker checksum differs")
    return canonical_id


def _assert_no_darwin_extended_acl(descriptor: int, *, role: str) -> None:
    """Reject a macOS NFSv4-style ACL using only an open descriptor."""

    if sys.platform != "darwin":
        return
    if _DARWIN_ACL_GET_FD_NP is None or _DARWIN_ACL_FREE is None:
        raise ArtifactBoundaryError("descriptor ACL inspection is unavailable")
    ctypes.set_errno(0)
    acl_pointer = _DARWIN_ACL_GET_FD_NP(descriptor, 0x00000100)
    if acl_pointer:
        release_result = _DARWIN_ACL_FREE(acl_pointer)
        if release_result != 0:
            raise ArtifactStoreError("unable to release descriptor ACL inspection state")
        raise ArtifactBoundaryError(f"{role} must not have an extended ACL")
    observed_errno = ctypes.get_errno()
    no_acl_errors = {
        errno.ENOENT,
        getattr(errno, "ENOATTR", errno.ENOENT),
        getattr(errno, "ENODATA", errno.ENOENT),
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        errno.EOPNOTSUPP,
    }
    if observed_errno not in no_acl_errors:
        raise ArtifactStoreError(
            f"unable to inspect {role} ACL",
            errno_code=observed_errno or None,
        )


def _assert_no_posix_acl_xattrs(descriptor: int, *, role: str) -> None:
    """Reject Linux/POSIX ACL xattrs through ``flistxattr`` semantics."""

    list_xattrs = getattr(os, "listxattr", None)
    if list_xattrs is None:
        return
    try:
        names = list_xattrs(descriptor)
    except OSError as exc:
        unsupported = {
            getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
            errno.EOPNOTSUPP,
        }
        if exc.errno in unsupported:
            return
        raise ArtifactStoreError(
            f"unable to inspect {role} ACL xattrs",
            errno_code=exc.errno,
        ) from None
    acl_names = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        b"system.posix_acl_access",
        b"system.posix_acl_default",
    }
    if any(name in acl_names for name in names):
        raise ArtifactBoundaryError(f"{role} must not have a POSIX ACL")


def _assert_no_extended_acl(descriptor: int, *, role: str) -> None:
    _assert_no_darwin_extended_acl(descriptor, role=role)
    _assert_no_posix_acl_xattrs(descriptor, role=role)


def _bounded_positive_integer(
    value: int,
    *,
    role: str,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactBoundaryError(f"{role} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ArtifactBoundaryError(f"{role} cannot exceed {maximum}")
    return value


def _bounded_non_negative_integer(value: int, *, role: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactBoundaryError(f"{role} must be a non-negative integer")
    return value


def _bounded_inventory_integer(value: int, *, role: str, maximum: int) -> int:
    result = _bounded_positive_integer(value, role=role)
    if result > maximum:
        raise ArtifactBoundaryError(f"{role} cannot exceed {maximum}")
    return result


def _bounded_inventory_seconds(value: float) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ArtifactBoundaryError("time_budget_seconds must be finite and positive")
    result = float(value)
    if result > MAX_INVENTORY_TIME_BUDGET_SECONDS:
        raise ArtifactBoundaryError(
            f"time_budget_seconds cannot exceed {MAX_INVENTORY_TIME_BUDGET_SECONDS}"
        )
    return result


def _validate_staging_id(value: str) -> str:
    if type(value) is not str:
        raise ArtifactBoundaryError("staging identifier must be canonical")
    if not value.startswith(_STAGING_PREFIX) or not value.endswith(_STAGING_SUFFIX):
        raise ArtifactBoundaryError("staging identifier must be canonical")
    token = value[len(_STAGING_PREFIX) : -len(_STAGING_SUFFIX)]
    if len(token) != _STAGING_TOKEN_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise ArtifactBoundaryError("staging identifier must be canonical")
    return value


def _require_utc_datetime(value: datetime, *, role: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None:
        raise ArtifactBoundaryError(f"{role} must be a timezone-aware UTC datetime")
    try:
        offset = value.utcoffset()
        normalized = value.astimezone(UTC)
    except Exception:
        raise ArtifactBoundaryError(f"{role} must be a valid UTC datetime") from None
    if offset != timedelta(0):
        raise ArtifactBoundaryError(f"{role} must be normalized to UTC")
    return normalized


def _utc_datetime_to_nanoseconds(value: datetime, *, role: str) -> int:
    normalized = _require_utc_datetime(value, role=role)
    try:
        delta = normalized - _UNIX_EPOCH
    except (OverflowError, ValueError):  # pragma: no cover - datetime defensive guard
        raise ArtifactBoundaryError(f"{role} is outside the supported timestamp range") from None
    return (
        delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000
    )


def _require_timestamp_nanoseconds(value: object, *, role: str) -> int:
    if type(value) is not int or not -(2**63) <= value <= 2**63 - 1:
        raise ArtifactBoundaryError(f"{role} must be an exact signed 64-bit nanosecond value")
    return value


def _nanoseconds_to_utc(value: int, *, role: str) -> datetime:
    if type(value) is not int:
        raise ArtifactIntegrityError(f"{role} timestamp is invalid") from None
    if not -(2**63) <= value <= 2**63 - 1:
        raise ArtifactIntegrityError(
            f"{role} timestamp is outside the supported UTC range"
        ) from None
    exact = value
    seconds, nanoseconds = divmod(exact, 1_000_000_000)
    try:
        return _UNIX_EPOCH + timedelta(
            seconds=seconds,
            microseconds=nanoseconds // 1_000,
        )
    except (OverflowError, ValueError):
        raise ArtifactIntegrityError(
            f"{role} timestamp is outside the supported UTC range"
        ) from None


def _last_changed_nanoseconds(value: os.stat_result) -> int:
    """Use the conservative later content or metadata-change timestamp."""

    return max(value.st_mtime_ns, value.st_ctime_ns)


def _storage_key(digest: str) -> str:
    value = _validate_digest(digest)
    return f"objects/{value[:2]}/{value[2:4]}/{value}"


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except InterruptedError:
            continue
        except OSError as exc:
            raise ArtifactStoreError(
                "unable to write artifact staging bytes",
                errno_code=exc.errno,
            ) from None
        if written <= 0:
            raise ArtifactStoreError("artifact staging write made no progress")
        offset += written


class ArtifactStore:
    """Publish and verify bounded immutable files under one local CAS root.

    The root is created only by :meth:`initialize`; reads never bootstrap
    storage. Source, root, fan-out, staging, and destination acquisition rejects
    symlinks and special files. Publication uses a same-filesystem private file
    plus an atomic hard link, so an existing digest is never replaced. Streaming
    artifacts are capped at 1 GiB; in-memory verified reads are capped at 64 MiB.
    Resource use is O(1) memory for publish/verify and O(max_bytes) for
    ``read_verified``. Publishers take a bounded advisory lock on the already-
    verified root descriptor so another cooperating publisher cannot mistake a
    transient staging link for an external hard link. Ordinary verification and
    reads still require exactly one link and therefore fail closed on external
    hard-link escape.

    The implementation requires POSIX-style directory-descriptor operations.
    Platforms without those primitives fail closed rather than falling back to
    check-then-use pathname operations. Supplying ``expected_store_id`` makes
    initialization reopen-only: an absent root is never recreated under an
    externally bound identity.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        expected_store_id: str | None = None,
    ) -> None:
        self._require_descriptor_support()
        self._root, self._root_parts = self._validated_absolute_path(root, role="CAS root")
        if not self._root_parts:
            raise ArtifactBoundaryError("CAS root cannot be a filesystem root")
        self._max_artifact_bytes = _bounded_positive_integer(
            max_artifact_bytes,
            role="max_artifact_bytes",
            maximum=MAX_ARTIFACT_BYTES,
        )
        self._expected_store_id = (
            None if expected_store_id is None else _validate_store_id(expected_store_id)
        )
        self._root_identity: _Identity | None = None
        self._staging_identity: _Identity | None = None
        self._objects_identity: _Identity | None = None
        self._store_marker_identity: _Identity | None = None
        self._store_id: str | None = None

    @property
    def root(self) -> Path:
        """Return the configured lexical root for local operator diagnostics."""

        return self._root

    @property
    def store_id(self) -> str:
        """Return the path-free durable identity of this initialized CAS."""

        if self._store_id is None:
            raise ArtifactBoundaryError("CAS store is not initialized")
        return self._store_id

    def initialize(self) -> None:
        """Securely create and bind the private root, staging, and object trees.

        Missing components are created with mode 0700 while walking from the
        filesystem anchor. Existing symlink or non-directory components fail
        before any descendant is accessed. When an external store identity is
        expected, the complete root must already exist and no path component
        is created.
        """

        identities = (
            self._root_identity,
            self._staging_identity,
            self._objects_identity,
            self._store_marker_identity,
            self._store_id,
        )
        if any(identity is not None for identity in identities):
            if not all(identity is not None for identity in identities):
                raise ArtifactIntegrityError("CAS initialization state is inconsistent")
            store = self._open_initialized()
            try:
                self._assert_initialized_bindings(store.root)
            finally:
                store.close()
            return

        if self._expected_store_id is None:
            root_descriptor, root_created = self._open_root_with_creation_state(create=True)
        else:
            try:
                root_descriptor, root_created = self._open_root_with_creation_state(create=False)
            except ArtifactBoundaryError as exc:
                if exc.errno_code == errno.ENOENT:
                    raise ArtifactIntegrityError(
                        "expected CAS root is unavailable; automatic recreation is forbidden"
                    ) from None
                raise
        staging_descriptor: int | None = None
        objects_descriptor: int | None = None
        try:
            root_identity = _Identity.directory(os.fstat(root_descriptor), role="CAS root")
            self._assert_private_directory(root_descriptor, role="CAS root")
            try:
                staging_descriptor, staging_created = self._open_directory_at_with_creation_state(
                    root_descriptor,
                    ".staging",
                    role="CAS staging directory",
                    create=root_created,
                )
                objects_descriptor, objects_created = self._open_directory_at_with_creation_state(
                    root_descriptor,
                    "objects",
                    role="CAS objects directory",
                    create=root_created,
                )
                if root_created:
                    if not staging_created or not objects_created:
                        raise ArtifactIntegrityError(
                            "new CAS root acquired unexpected pre-existing components"
                        )
                    self._create_store_identity_marker(root_descriptor)
                    store_id, marker_identity = self._read_store_identity_marker(root_descriptor)
                else:
                    store_id, marker_identity = self._read_store_identity_marker(root_descriptor)
            except ArtifactBoundaryError as exc:
                if not root_created and exc.errno_code == errno.ENOENT:
                    raise ArtifactIntegrityError(
                        "existing CAS root is incomplete; automatic repair is forbidden"
                    ) from None
                raise
            self._assert_private_directory(
                staging_descriptor,
                role="CAS staging directory",
            )
            self._assert_private_directory(
                objects_descriptor,
                role="CAS objects directory",
            )
            self._fsync_directory_descriptor(root_descriptor)
            self._root_identity = root_identity
            self._staging_identity = _Identity.directory(
                os.fstat(staging_descriptor),
                role="CAS staging directory",
            )
            self._objects_identity = _Identity.directory(
                os.fstat(objects_descriptor),
                role="CAS objects directory",
            )
            self._store_marker_identity = marker_identity
            self._store_id = store_id
            self._assert_initialized_bindings(root_descriptor)
        except Exception:
            self._root_identity = None
            self._staging_identity = None
            self._objects_identity = None
            self._store_marker_identity = None
            self._store_id = None
            raise
        finally:
            _close_descriptors(
                objects_descriptor,
                staging_descriptor,
                root_descriptor,
                role="CAS initialization descriptor cleanup",
            )

    def publish(self, source: str | os.PathLike[str]) -> PublishedArtifact:
        """Copy one stable bounded source and atomically publish it by SHA-256.

        Publication never replaces an existing object. A concurrent or prior
        object is reused only after its descriptor, size, read-only mode, and
        complete digest verify. A failure after the destination link is created
        may leave a valid unlinked CAS object; callers must not create registry
        metadata unless this method returns successfully.
        """

        store = self._open_initialized()
        publication_lock_acquired = False
        opened_source: _OpenedFile | None = None
        staging_descriptor: int | None = None
        staging_name: str | None = None
        staging_identity: _Identity | None = None
        primary_error: BaseException | None = None
        try:
            opened_source = self._open_external_regular(source, role="artifact source")
            before = opened_source.identity
            if before.size > self._max_artifact_bytes:
                raise ArtifactBoundaryError(
                    f"artifact source exceeds {self._max_artifact_bytes} bytes"
                )

            staging_name, staging_descriptor = self._create_staging(store.staging)
            staging_identity = _Identity.regular(
                os.fstat(staging_descriptor),
                role="artifact staging file",
            )
            digest, copied = self._copy_and_hash(
                opened_source.descriptor,
                staging_descriptor,
                expected_size=before.size,
            )
            self._assert_source_unchanged(opened_source, before)
            if copied != before.size:
                raise ArtifactIntegrityError("artifact source size changed during publication")

            self._fsync_file_descriptor(staging_descriptor)
            self._make_file_read_only(staging_descriptor)
            self._fsync_file_descriptor(staging_descriptor)
            staging_identity = _Identity.regular(
                os.fstat(staging_descriptor),
                role="artifact staging file",
            )

            destination_parent = self._open_destination_parent(
                store.objects,
                digest,
                create=True,
            )
            try:
                self._acquire_publication_lock(store.root)
                publication_lock_acquired = True
                created = False
                try:
                    self._link_staged(
                        store.staging,
                        staging_name,
                        destination_parent,
                        digest,
                    )
                    created = True
                except FileExistsError:
                    pass
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise ArtifactStoreError(
                            "unable to atomically publish artifact",
                            errno_code=exc.errno,
                        ) from None

                self._assert_publication_candidate(
                    destination_parent,
                    digest,
                    expected_size=copied,
                    expected_link_count=2 if created else 1,
                    staging_identity=staging_identity if created else None,
                )
                if created:
                    self._fsync_directory_descriptor(destination_parent)
                self._assert_destination_parent_binding(
                    store.objects,
                    digest,
                    destination_parent,
                )
                self._unlink_name_if_same(
                    store.staging,
                    staging_name,
                    staging_identity,
                    missing_ok=False,
                    role="artifact staging cleanup",
                )
                self._fsync_directory_descriptor(store.staging)
                staging_name = None
                self._assert_publication_candidate(
                    destination_parent,
                    digest,
                    expected_size=copied,
                    expected_link_count=1,
                    staging_identity=None,
                )
                self._release_publication_lock(store.root)
                publication_lock_acquired = False

                self._verify_object(
                    destination_parent,
                    digest,
                    digest=digest,
                    expected_size=copied,
                    root_descriptor=store.root,
                )
                self._assert_destination_parent_binding(
                    store.objects,
                    digest,
                    destination_parent,
                )
            finally:
                _close_descriptors(
                    destination_parent,
                    role="artifact destination descriptor cleanup",
                )

            self._assert_initialized_bindings(store.root)
            return PublishedArtifact(
                digest=digest,
                byte_size=copied,
                storage_key=_storage_key(digest),
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup = _CleanupState.start(primary_error)
            cleanup.close("artifact staging descriptor close", staging_descriptor)
            if staging_name is not None and staging_identity is not None:
                cleanup.attempt(
                    "artifact staging removal",
                    lambda: self._cleanup_staging_file(
                        store.staging,
                        staging_name,
                        staging_identity,
                    ),
                )
            if opened_source is not None:
                cleanup.attempt("artifact source descriptor cleanup", opened_source.close)
            if publication_lock_acquired:
                cleanup.attempt(
                    "artifact publication lock release",
                    lambda: self._release_publication_lock(store.root),
                )
            cleanup.attempt("CAS store descriptor cleanup", store.close)
            cleanup.finish("artifact publication cleanup")

    def verify(self, artifact: PublishedArtifact) -> None:
        """Re-hash a published object and verify its recorded metadata."""

        self._validate_artifact_bound(artifact)
        store = self._open_initialized()
        try:
            parent = self._open_destination_parent(store.objects, artifact.digest, create=False)
            try:
                self._verify_object(
                    parent,
                    artifact.digest,
                    digest=artifact.digest,
                    expected_size=artifact.byte_size,
                    root_descriptor=store.root,
                )
                self._assert_destination_parent_binding(
                    store.objects,
                    artifact.digest,
                    parent,
                )
            finally:
                _close_descriptors(parent, role="artifact verification parent cleanup")
            self._assert_initialized_bindings(store.root)
        finally:
            store.close()

    def read_verified(
        self,
        artifact: PublishedArtifact,
        *,
        max_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
    ) -> bytes:
        """Return verified bounded bytes without exposing a filesystem path."""

        limit = _bounded_positive_integer(
            max_bytes,
            role="max_bytes",
            maximum=MAX_IN_MEMORY_READ_BYTES,
        )
        self._validate_artifact_bound(artifact)
        if artifact.byte_size > limit:
            raise ArtifactBoundaryError(f"artifact exceeds caller limit of {limit} bytes")
        store = self._open_initialized()
        try:
            parent = self._open_destination_parent(store.objects, artifact.digest, create=False)
            result: bytes
            try:
                descriptor, before = self._open_single_link_object(
                    store.root,
                    parent,
                    artifact.digest,
                    expected_size=artifact.byte_size,
                )
                try:
                    hasher = hashlib.sha256()
                    payload = bytearray()
                    while True:
                        remaining = limit + 1 - len(payload)
                        chunk = self._read_chunk(descriptor, min(COPY_CHUNK_BYTES, remaining))
                        if not chunk:
                            break
                        payload.extend(chunk)
                        hasher.update(chunk)
                        if len(payload) > limit:
                            raise ArtifactBoundaryError(
                                f"artifact exceeds caller limit of {limit} bytes"
                            )
                    if len(payload) != artifact.byte_size:
                        raise ArtifactIntegrityError(
                            "CAS object size differs from registry metadata"
                        )
                    if hasher.hexdigest() != artifact.digest:
                        raise ArtifactIntegrityError(
                            "CAS object digest differs from registry metadata"
                        )
                    after_status = os.fstat(descriptor)
                    after = _Identity.regular(after_status, role="CAS object")
                    self._assert_cas_object_status(after_status, expected_link_count=1)
                    if before != after:
                        raise ArtifactIntegrityError("CAS object changed during verified read")
                    self._assert_name_matches_descriptor(
                        parent,
                        artifact.digest,
                        after,
                        role="CAS object",
                    )
                    result = bytes(payload)
                    self._assert_destination_parent_binding(
                        store.objects,
                        artifact.digest,
                        parent,
                    )
                finally:
                    _close_descriptors(descriptor, role="verified read descriptor cleanup")
            finally:
                _close_descriptors(parent, role="verified read parent cleanup")
            self._assert_initialized_bindings(store.root)
            return result
        finally:
            store.close()

    def inspect(self, artifact: PublishedArtifact) -> ArtifactInventoryRecord:
        """Return one path-free verified object identity for retention policy.

        The exact canonical key is re-opened without creation and checked for
        private mode, owner, ACL absence, link count, size, digest, mutation,
        fan-out binding, and root binding before the generation timestamp is
        returned.
        """

        self._validate_artifact_bound(artifact)
        store = self._open_initialized()
        try:
            parent = self._open_destination_parent(
                store.objects,
                artifact.digest,
                create=False,
            )
            try:
                record, _ = self._inspect_inventory_object(
                    parent,
                    artifact.digest,
                    budget=None,
                    root_descriptor=store.root,
                )
                if record.byte_size != artifact.byte_size:
                    raise ArtifactIntegrityError("CAS object size differs from registry metadata")
                self._assert_destination_parent_binding(
                    store.objects,
                    artifact.digest,
                    parent,
                )
            finally:
                _close_descriptors(parent, role="artifact inspection parent cleanup")
            self._assert_initialized_bindings(store.root)
            return record
        finally:
            store.close()

    def verify_absent(self, artifact: PublishedArtifact) -> None:
        """Prove that an exact CAS key is absent without creating storage.

        The canonical fan-out must already exist from a prior publication and
        is acquired and rebound before return. Only ``ENOENT`` for the exact
        object name establishes absence. Missing fan-out components, any
        present regular file, symlink, or special file, any other operating-
        system error, or any parent mutation observed by either canonical
        rebind fails closed. POSIX cannot freeze the namespace after the final
        check, so callers must serialize maintenance against other same-UID
        writers. No host path is exposed.
        """

        self._validate_artifact_bound(artifact)
        store = self._open_initialized()
        try:
            parent = self._open_destination_parent(
                store.objects,
                artifact.digest,
                create=False,
            )
            try:
                before = _Identity.directory(
                    os.fstat(parent),
                    role="CAS fanout directory",
                )
                self._assert_name_absent(
                    parent,
                    artifact.digest,
                    role="CAS object",
                )
                self._assert_destination_parent_binding(
                    store.objects,
                    artifact.digest,
                    parent,
                )
                self._assert_initialized_bindings(store.root)
                after = _Identity.directory(
                    os.fstat(parent),
                    role="CAS fanout directory",
                )
                if before != after:
                    raise ArtifactIntegrityError(
                        "CAS fanout directory changed during absence proof"
                    )
                self._assert_name_absent(
                    parent,
                    artifact.digest,
                    role="CAS object",
                )
                final = _Identity.directory(
                    os.fstat(parent),
                    role="CAS fanout directory",
                )
                if after != final:
                    raise ArtifactIntegrityError(
                        "CAS fanout directory changed during absence proof"
                    )
                self._assert_destination_parent_binding(
                    store.objects,
                    artifact.digest,
                    parent,
                )
            finally:
                _close_descriptors(parent, role="artifact absence parent cleanup")
        finally:
            store.close()

    def inventory_objects(
        self,
        *,
        cutoff: datetime,
        cursor: ArtifactInventoryCursor | None = None,
        page_size: int = DEFAULT_INVENTORY_PAGE_SIZE,
        max_scan_entries: int = DEFAULT_INVENTORY_MAX_SCANNED_ENTRIES,
        time_budget_seconds: float = DEFAULT_INVENTORY_TIME_BUDGET_SECONDS,
    ) -> ArtifactInventoryPage:
        """Verify and page CAS objects changed no later than ``cutoff``.

        This read-only maintenance boundary is intended for comparing old CAS
        objects with a registry after a caller-selected publication grace
        period. Every object in the canonical two-level fan-out is type-, mode-,
        owner-, link-count-, size-, digest-, timestamp-, and descriptor-identity
        checked before any page is returned. Any unexpected entry fails the
        entire scan; no object is silently skipped.

        ``page_size`` bounds returned records, while ``max_scan_entries`` bounds
        all visited directories and files and ``time_budget_seconds`` supplies a
        cooperative elapsed-time deadline. To provide lexical continuation, a
        page rescans and verifies the complete bounded local tree. Consequently
        this API is intentionally local-scale: it is O(scan entries + artifact
        bytes) per page and O(scan entries) memory. The deadline cannot preempt a
        blocked filesystem syscall. A continuation is bound to this store and
        the exact cutoff, preventing omission when a caller advances the grace
        boundary between pages. Pagination is otherwise omission- and
        duplication-free only while the tree is unchanged; maintenance callers
        must coordinate with publishers because this traversal is not a snapshot.

        The method never initializes storage or issues a filesystem create,
        write, chmod, rename, link, or unlink operation, and it never returns an
        absolute host path. A filesystem may still update access-time metadata
        for reads according to its mount policy. Discovery does not authorize
        orphan deletion.
        """

        normalized_cutoff = _require_utc_datetime(cutoff, role="cutoff")
        cutoff_ns = _utc_datetime_to_nanoseconds(normalized_cutoff, role="cutoff")
        page_limit = _bounded_inventory_integer(
            page_size,
            role="page_size",
            maximum=MAX_INVENTORY_PAGE_SIZE,
        )
        if cursor is not None and type(cursor) is not ArtifactInventoryCursor:
            raise ArtifactBoundaryError("artifact inventory cursor must be typed")
        if cursor is not None and cursor.cutoff != normalized_cutoff:
            raise ArtifactBoundaryError("artifact inventory cursor cutoff does not match")
        cursor_value = None if cursor is None else cursor.after_digest
        budget = _InventoryBudget.start(
            maximum_entries=max_scan_entries,
            time_budget_seconds=time_budget_seconds,
        )

        try:
            store = self._open_initialized()
        except OSError as exc:  # pragma: no cover - descriptor I/O failure
            raise ArtifactStoreError(
                "unable to acquire CAS object inventory descriptors",
                errno_code=exc.errno,
            ) from None
        try:
            budget.checkpoint()
            if cursor is not None and not secrets.compare_digest(cursor.store_id, self.store_id):
                raise ArtifactBoundaryError("artifact inventory cursor belongs to another store")
            records = self._inspect_object_tree(
                store,
                cutoff_ns=cutoff_ns,
                budget=budget,
            )
            budget.checkpoint()
            if cursor_value is not None:
                cursor_positions = tuple(
                    index
                    for index, record in enumerate(records)
                    if secrets.compare_digest(record.digest, cursor_value)
                )
                if len(cursor_positions) != 1:
                    raise ArtifactBoundaryError(
                        "artifact inventory cursor position is absent from the verified tree"
                    )
                records = records[cursor_positions[0] + 1 :]
            budget.checkpoint()
            page_records = tuple(records[:page_limit])
            next_cursor = (
                ArtifactInventoryCursor(
                    after_digest=page_records[-1].digest,
                    cutoff=normalized_cutoff,
                    store_id=self.store_id,
                )
                if len(records) > len(page_records)
                else None
            )
            budget.checkpoint()
            return ArtifactInventoryPage(
                records=page_records,
                next_cursor=next_cursor,
                scanned_entries=budget.scanned_entries,
            )
        except OSError as exc:  # pragma: no cover - descriptor I/O failure
            raise ArtifactStoreError(
                "unable to inspect CAS object inventory",
                errno_code=exc.errno,
            ) from None
        finally:
            store.close()

    def discover_stale_staging(
        self,
        *,
        cutoff: datetime,
        cursor: StagingInventoryCursor | None = None,
        page_size: int = DEFAULT_INVENTORY_PAGE_SIZE,
        max_scan_entries: int = DEFAULT_INVENTORY_MAX_SCANNED_ENTRIES,
        time_budget_seconds: float = DEFAULT_INVENTORY_TIME_BUDGET_SECONDS,
    ) -> StagingInventoryPage:
        """Inspect and page canonical staging files changed by ``cutoff``.

        The same explicit page, complete-scan, and cooperative time bounds as
        :meth:`inventory_objects` apply. Staging files must have the exact
        publisher-generated identifier, current-user ownership, one link, and
        mode 0400 or 0600. Unexpected names, symlinks, special files, concurrent
        mutation, or unsafe permissions fail closed.

        This method is discovery only: it does not initialize the store or issue
        a create, write, chmod, rename, link, or unlink operation. A filesystem
        may still update access-time metadata for reads according to mount
        policy. Pagination assumes an unchanged tree and maintenance must be
        coordinated with publishers. The continuation is bound to this store
        and exact cutoff. Returned identifiers are path-free and reveal no
        absolute host path.
        """

        normalized_cutoff = _require_utc_datetime(cutoff, role="cutoff")
        cutoff_ns = _utc_datetime_to_nanoseconds(normalized_cutoff, role="cutoff")
        page_limit = _bounded_inventory_integer(
            page_size,
            role="page_size",
            maximum=MAX_INVENTORY_PAGE_SIZE,
        )
        if cursor is not None and type(cursor) is not StagingInventoryCursor:
            raise ArtifactBoundaryError("staging inventory cursor must be typed")
        if cursor is not None and cursor.cutoff != normalized_cutoff:
            raise ArtifactBoundaryError("staging inventory cursor cutoff does not match")
        cursor_value = None if cursor is None else cursor.after_staging_id
        budget = _InventoryBudget.start(
            maximum_entries=max_scan_entries,
            time_budget_seconds=time_budget_seconds,
        )

        try:
            store = self._open_initialized()
        except OSError as exc:  # pragma: no cover - descriptor I/O failure
            raise ArtifactStoreError(
                "unable to acquire CAS staging inventory descriptors",
                errno_code=exc.errno,
            ) from None
        try:
            budget.checkpoint()
            if cursor is not None and not secrets.compare_digest(cursor.store_id, self.store_id):
                raise ArtifactBoundaryError("staging inventory cursor belongs to another store")
            records = self._inspect_staging_tree(
                store,
                cutoff_ns=cutoff_ns,
                budget=budget,
            )
            budget.checkpoint()
            if cursor_value is not None:
                cursor_positions = tuple(
                    index
                    for index, record in enumerate(records)
                    if secrets.compare_digest(record.staging_id, cursor_value)
                )
                if len(cursor_positions) != 1:
                    raise ArtifactBoundaryError(
                        "staging inventory cursor position is absent from the verified tree"
                    )
                records = records[cursor_positions[0] + 1 :]
            budget.checkpoint()
            page_records = tuple(records[:page_limit])
            next_cursor = (
                StagingInventoryCursor(
                    after_staging_id=page_records[-1].staging_id,
                    cutoff=normalized_cutoff,
                    store_id=self.store_id,
                )
                if len(records) > len(page_records)
                else None
            )
            budget.checkpoint()
            return StagingInventoryPage(
                records=page_records,
                next_cursor=next_cursor,
                scanned_entries=budget.scanned_entries,
            )
        except OSError as exc:  # pragma: no cover - descriptor I/O failure
            raise ArtifactStoreError(
                "unable to inspect CAS staging inventory",
                errno_code=exc.errno,
            ) from None
        finally:
            store.close()

    def unlink_verified(
        self,
        artifact: PublishedArtifact,
        *,
        expected_generation: ArtifactGeneration | None = None,
    ) -> None:
        """Verify and unlink one registry-authorized retention object.

        Registry policy must first prove that the object is unlinked, unpinned,
        inactive, and selected by the exact confirmed retention plan. This
        method provides only the bounded filesystem integrity boundary. When a
        typed generation from :meth:`inspect` is supplied, a delete and republish of identical
        bytes fails closed even if both generations share the same display microsecond. The
        generation digest binds store ID, artifact digest, device, inode, file type, size, and
        exact modification/change nanoseconds without exposing raw filesystem identities.

        The exact fan-out parent is rebound immediately inside the destructive
        primitive and again after unlink. POSIX does not offer an atomic
        "unlink only while this dirfd remains beneath that root" operation, so
        a malicious same-UID process still has a narrow rename race between the
        final containment check and ``unlinkat``. The owner-only tree excludes
        other users; callers must serialize same-UID maintenance and publishing.
        """

        self._validate_artifact_bound(artifact)
        if expected_generation is not None and type(expected_generation) is not ArtifactGeneration:
            raise ArtifactBoundaryError("expected_generation must be a typed artifact generation")
        store = self._open_initialized()
        try:
            parent = self._open_destination_parent(store.objects, artifact.digest, create=False)
            try:
                descriptor = self._open_regular_at(
                    parent,
                    artifact.digest,
                    role="CAS object",
                    allowed_modes=(0o400,),
                )
                try:
                    before_status = os.fstat(descriptor)
                    before = _Identity.regular(before_status, role="CAS object")
                    self._assert_inventory_object_status(before_status)
                    if before.size != artifact.byte_size:
                        raise ArtifactIntegrityError(
                            "CAS object size differs from registry metadata"
                        )
                    if (
                        self._hash_descriptor(descriptor, maximum=artifact.byte_size)
                        != artifact.digest
                    ):
                        raise ArtifactIntegrityError(
                            "CAS object digest differs from registry metadata"
                        )
                    after_status = os.fstat(descriptor)
                    after = _Identity.regular(after_status, role="CAS object")
                    self._assert_inventory_object_status(after_status)
                    if before != after:
                        raise ArtifactIntegrityError("CAS object changed during verification")
                    observed_generation = _artifact_generation(
                        self.store_id,
                        artifact.digest,
                        after,
                    )
                    if (
                        expected_generation is not None
                        and observed_generation != expected_generation
                    ):
                        raise ArtifactIntegrityError("CAS object generation differs")
                    self._assert_name_matches_descriptor(
                        parent,
                        artifact.digest,
                        after,
                        role="CAS object",
                    )
                    self._unlink_name_if_same(
                        parent,
                        artifact.digest,
                        after,
                        missing_ok=False,
                        role="CAS object unlink",
                        require_full_identity=True,
                        containment_objects_descriptor=store.objects,
                        containment_root_descriptor=store.root,
                        containment_digest=artifact.digest,
                    )
                    self._fsync_directory_descriptor(parent)
                    if os.fstat(descriptor).st_nlink != 0:
                        raise ArtifactIntegrityError("CAS object unlink was not confirmed")
                    self._assert_destination_parent_binding(
                        store.objects,
                        artifact.digest,
                        parent,
                    )
                finally:
                    _close_descriptors(descriptor, role="artifact unlink descriptor cleanup")
            finally:
                _close_descriptors(parent, role="artifact unlink parent cleanup")
            self._assert_initialized_bindings(store.root)
        finally:
            store.close()

    @staticmethod
    def _require_descriptor_support() -> None:
        ArtifactStore._file_lock_module()
        required = (os.open, os.mkdir, os.stat, os.unlink, os.link)
        if any(operation not in os.supports_dir_fd for operation in required):
            raise ArtifactBoundaryError("CAS requires directory-descriptor filesystem operations")
        if os.stat not in os.supports_follow_symlinks or os.link not in os.supports_follow_symlinks:
            raise ArtifactBoundaryError("CAS requires no-follow filesystem operations")
        if _NOFOLLOW == 0:
            raise ArtifactBoundaryError("CAS requires O_NOFOLLOW support")

    @staticmethod
    def _file_lock_module() -> _FileLockModule:
        """Return POSIX advisory locks or fail before any filesystem mutation."""

        module = fcntl
        if module is None:
            raise ArtifactBoundaryError("CAS requires POSIX advisory file-lock operations")
        return module

    @staticmethod
    def _acquire_publication_lock(root_descriptor: int) -> None:
        """Acquire the cross-process publisher lock within a monotonic deadline."""

        ArtifactStore._acquire_store_lock(
            root_descriptor,
            exclusive=True,
            role="artifact publication",
        )

    @staticmethod
    def _acquire_verification_lock(root_descriptor: int) -> None:
        """Acquire a shared lock while opening one canonical single-link object."""

        ArtifactStore._acquire_store_lock(
            root_descriptor,
            exclusive=False,
            role="artifact verification",
        )

    @staticmethod
    def _acquire_store_lock(
        root_descriptor: int,
        *,
        exclusive: bool,
        role: str,
    ) -> None:
        """Acquire one advisory CAS lock within a monotonic contention deadline."""

        lock_api = ArtifactStore._file_lock_module()
        started = time.monotonic_ns()
        deadline = started + math.ceil(PUBLICATION_LOCK_TIMEOUT_SECONDS * 1_000_000_000)
        last_observed = started
        operation = lock_api.LOCK_EX if exclusive else lock_api.LOCK_SH
        while True:
            try:
                lock_api.flock(root_descriptor, operation | lock_api.LOCK_NB)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                    raise ArtifactStoreError(
                        f"unable to acquire {role} lock",
                        errno_code=exc.errno,
                    ) from None
            observed = time.monotonic_ns()
            if observed < last_observed:
                raise ArtifactIntegrityError(f"{role} lock monotonic clock moved backwards")
            last_observed = observed
            if observed >= deadline:
                raise ArtifactBoundaryError(f"{role} lock exceeded its bounded contention deadline")
            remaining_seconds = (deadline - observed) / 1_000_000_000
            time.sleep(min(PUBLICATION_LOCK_POLL_SECONDS, remaining_seconds))

    @staticmethod
    def _release_publication_lock(root_descriptor: int) -> None:
        """Release a previously acquired publisher lock without exposing host paths."""

        ArtifactStore._release_store_lock(root_descriptor, role="artifact publication")

    @staticmethod
    def _release_verification_lock(root_descriptor: int) -> None:
        """Release a previously acquired shared verification lock."""

        ArtifactStore._release_store_lock(root_descriptor, role="artifact verification")

    @staticmethod
    def _release_store_lock(root_descriptor: int, *, role: str) -> None:
        """Release one advisory CAS lock without exposing host paths."""

        lock_api = ArtifactStore._file_lock_module()
        try:
            lock_api.flock(root_descriptor, lock_api.LOCK_UN)
        except OSError as exc:
            raise ArtifactStoreError(
                f"unable to release {role} lock",
                errno_code=exc.errno,
            ) from None

    @staticmethod
    def _validated_absolute_path(
        value: str | os.PathLike[str],
        *,
        role: str,
    ) -> tuple[Path, tuple[str, ...]]:
        try:
            raw = os.fspath(value)
        except Exception:
            raise ArtifactBoundaryError(f"{role} path is invalid") from None
        if type(raw) is not str or not raw or "\x00" in raw:
            raise ArtifactBoundaryError(f"{role} path is invalid")
        try:
            lexical_parts = Path(raw).parts
            if ".." in lexical_parts:
                raise ArtifactBoundaryError(f"{role} path must not contain traversal")
            absolute = Path(os.path.abspath(raw))
            encoded_absolute = os.fsencode(str(absolute))
            encoded_parts = tuple(os.fsencode(part) for part in absolute.parts)
        except ArtifactBoundaryError:
            raise
        except Exception:
            raise ArtifactBoundaryError(f"{role} path is invalid") from None
        if len(encoded_absolute) > MAX_CAS_PATH_BYTES or any(
            len(part) > MAX_CAS_PATH_COMPONENT_BYTES for part in encoded_parts
        ):
            raise ArtifactBoundaryError(f"{role} path exceeds its portability bounds")
        anchor = Path(absolute.anchor)
        try:
            relative = absolute.relative_to(anchor)
        except ValueError:  # pragma: no cover - defensive platform guard
            raise ArtifactBoundaryError(f"{role} path is invalid") from None
        parts = tuple(part for part in relative.parts if part not in ("", "."))
        return absolute, parts

    def _open_root(self, *, create: bool) -> int:
        descriptor, _ = self._open_root_with_creation_state(create=create)
        return descriptor

    def _open_root_with_creation_state(self, *, create: bool) -> tuple[int, bool]:
        """Open the configured root and report whether this call created it."""

        anchor = self._root.anchor
        try:
            current = os.open(anchor, _DIRECTORY_FLAGS | _NOFOLLOW)
        except OSError as exc:  # pragma: no cover - platform/root failure
            raise ArtifactBoundaryError(
                "CAS filesystem anchor is unavailable",
                errno_code=exc.errno,
            ) from None
        try:
            root_created = False
            for index, part in enumerate(self._root_parts):
                next_descriptor, component_created = self._open_directory_at_with_creation_state(
                    current,
                    part,
                    role="CAS root component",
                    create=create,
                )
                previous_descriptor = current
                current = next_descriptor
                _close_descriptors(
                    previous_descriptor,
                    role="CAS root walk descriptor cleanup",
                )
                if index == len(self._root_parts) - 1:
                    root_created = component_created
            return current, root_created
        except Exception:
            _close_descriptors(current, role="CAS root acquisition cleanup")
            raise

    def _open_initialized(self) -> _StoreDescriptors:
        if (
            self._root_identity is None
            or self._staging_identity is None
            or self._objects_identity is None
            or self._store_marker_identity is None
            or self._store_id is None
        ):
            raise ArtifactBoundaryError("CAS store is not initialized")
        root = self._open_root(create=False)
        staging: int | None = None
        objects: int | None = None
        try:
            self._assert_descriptor_identity(root, self._root_identity, role="CAS root")
            self._assert_private_directory(root, role="CAS root")
            self._assert_store_identity_marker(root)
            staging = self._open_directory_at(
                root,
                ".staging",
                role="CAS staging directory",
                create=False,
            )
            self._assert_descriptor_identity(
                staging,
                self._staging_identity,
                role="CAS staging directory",
            )
            self._assert_private_directory(staging, role="CAS staging directory")
            objects = self._open_directory_at(
                root,
                "objects",
                role="CAS objects directory",
                create=False,
            )
            self._assert_descriptor_identity(
                objects,
                self._objects_identity,
                role="CAS objects directory",
            )
            self._assert_private_directory(objects, role="CAS objects directory")
            return _StoreDescriptors(root=root, staging=staging, objects=objects)
        except Exception:
            _close_descriptors(
                objects,
                staging,
                root,
                role="CAS descriptor acquisition cleanup",
            )
            raise

    def _validate_artifact_bound(self, artifact: PublishedArtifact) -> None:
        if type(artifact) is not PublishedArtifact:
            raise ArtifactBoundaryError("artifact must be a PublishedArtifact")
        if artifact.byte_size > self._max_artifact_bytes:
            raise ArtifactBoundaryError(
                f"artifact exceeds store limit of {self._max_artifact_bytes} bytes"
            )

    def _assert_initialized_bindings(self, root_descriptor: int) -> None:
        if (
            self._root_identity is None
            or self._staging_identity is None
            or self._objects_identity is None
            or self._store_marker_identity is None
            or self._store_id is None
        ):
            raise ArtifactBoundaryError("CAS store is not initialized")
        rebound_root = self._open_root(create=False)
        try:
            self._assert_private_directory(rebound_root, role="CAS root")
            self._assert_private_directory(root_descriptor, role="CAS root")
            self._assert_descriptor_identity(
                rebound_root,
                self._root_identity,
                role="CAS root",
            )
            self._assert_descriptor_identity(
                root_descriptor,
                self._root_identity,
                role="CAS root",
            )
            self._assert_store_identity_marker(root_descriptor)
            self._assert_name_identity(
                root_descriptor,
                ".staging",
                self._staging_identity,
                role="CAS staging directory",
                directory=True,
            )
            self._assert_name_identity(
                root_descriptor,
                "objects",
                self._objects_identity,
                role="CAS objects directory",
                directory=True,
            )
        finally:
            _close_descriptors(rebound_root, role="CAS root rebind descriptor cleanup")

    @staticmethod
    def _open_directory_at(
        parent_descriptor: int,
        name: str,
        *,
        role: str,
        create: bool,
    ) -> int:
        descriptor, _ = ArtifactStore._open_directory_at_with_creation_state(
            parent_descriptor,
            name,
            role=role,
            create=create,
        )
        return descriptor

    @staticmethod
    def _open_directory_at_with_creation_state(
        parent_descriptor: int,
        name: str,
        *,
        role: str,
        create: bool,
    ) -> tuple[int, bool]:
        if not name or name in (".", "..") or "/" in name or os.sep in name:
            raise ArtifactBoundaryError(f"{role} name is invalid")
        created = False
        if create:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
                created = True
            except FileExistsError:
                pass
            except OSError as exc:
                raise ArtifactStoreError(
                    f"unable to create {role}",
                    errno_code=exc.errno,
                ) from None
        try:
            descriptor = os.open(
                name,
                _DIRECTORY_FLAGS | _NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise ArtifactBoundaryError(
                f"{role} cannot be opened safely",
                errno_code=exc.errno,
            ) from None
        try:
            _Identity.directory(os.fstat(descriptor), role=role)
            observed = _Identity.directory(
                os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False),
                role=role,
            )
            opened = _Identity.directory(os.fstat(descriptor), role=role)
            if not observed.same_object(opened):
                raise ArtifactIntegrityError(f"{role} changed during acquisition")
            if create:
                ArtifactStore._fsync_directory_descriptor(parent_descriptor)
            return descriptor, created
        except Exception:
            _close_descriptors(descriptor, role=f"{role} acquisition cleanup")
            raise

    def _create_store_identity_marker(
        self,
        root_descriptor: int,
        *,
        store_id: str | None = None,
    ) -> None:
        """Atomically publish one checksummed store identity marker."""

        marker_store_id = (
            secrets.token_hex(_STORE_ID_HEX_LENGTH // 2)
            if store_id is None
            else _validate_store_id(store_id)
        )
        payload = _store_id_payload(marker_store_id)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW
        temporary_name = f"{_STORE_ID_TEMP_PREFIX}{secrets.token_hex(24)}{_STORE_ID_TEMP_SUFFIX}"
        descriptor: int | None = None
        temporary_identity: _Identity | None = None
        primary_error: BaseException | None = None
        try:
            try:
                descriptor = os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                raise ArtifactStoreError(
                    "unable to create CAS store identity staging file",
                    errno_code=exc.errno,
                ) from None
            initial_status = os.fstat(descriptor)
            temporary_identity = _Identity.regular(
                initial_status,
                role="CAS store identity staging file",
            )
            self._assert_private_file_metadata(
                initial_status,
                role="CAS store identity staging file",
                allowed_modes=(0o600,),
            )
            _assert_no_extended_acl(
                descriptor,
                role="CAS store identity staging file",
            )
            self._write_store_identity_payload(descriptor, payload)
            self._fsync_file_descriptor(descriptor)
            try:
                os.fchmod(descriptor, 0o400)
            except OSError as exc:
                raise ArtifactStoreError(
                    "unable to make CAS store identity marker read-only",
                    errno_code=exc.errno,
                ) from None
            self._fsync_file_descriptor(descriptor)
            readonly_status = os.fstat(descriptor)
            self._assert_private_file_metadata(
                readonly_status,
                role="CAS store identity staging file",
                allowed_modes=(0o400,),
            )
            _assert_no_extended_acl(
                descriptor,
                role="CAS store identity staging file",
            )
            temporary_identity = _Identity.regular(
                readonly_status,
                role="CAS store identity staging file",
            )
            try:
                os.link(
                    temporary_name,
                    _STORE_ID_FILENAME,
                    src_dir_fd=root_descriptor,
                    dst_dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                self._fsync_directory_descriptor(root_descriptor)
            except FileExistsError:
                raise ArtifactIntegrityError(
                    "new CAS root already contains a store identity marker"
                ) from None
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise ArtifactStoreError(
                        "unable to publish CAS store identity marker",
                        errno_code=exc.errno,
                    ) from None
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup = _CleanupState.start(primary_error)
            cleanup.close("CAS store identity staging descriptor close", descriptor)
            if temporary_identity is not None:
                cleanup.attempt(
                    "CAS store identity staging removal",
                    lambda: self._cleanup_store_identity_staging(
                        root_descriptor,
                        temporary_name,
                        temporary_identity,
                    ),
                )
            cleanup.finish("CAS store identity staging cleanup")

    def _cleanup_store_identity_staging(
        self,
        root_descriptor: int,
        temporary_name: str,
        temporary_identity: _Identity,
    ) -> None:
        """Remove one exact store-marker staging generation and flush its directory."""

        self._unlink_name_if_same(
            root_descriptor,
            temporary_name,
            temporary_identity,
            missing_ok=True,
            role="CAS store identity staging cleanup",
        )
        self._fsync_directory_descriptor(root_descriptor)

    @staticmethod
    def _write_store_identity_payload(descriptor: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(descriptor, payload[offset:])
            except InterruptedError:
                continue
            except OSError as exc:
                raise ArtifactStoreError(
                    "unable to write CAS store identity marker",
                    errno_code=exc.errno,
                ) from None
            if written <= 0:
                raise ArtifactStoreError("CAS store identity marker write made no progress")
            offset += written

    def _read_store_identity_marker(
        self,
        root_descriptor: int,
    ) -> tuple[str, _Identity]:
        descriptor = self._open_regular_at(
            root_descriptor,
            _STORE_ID_FILENAME,
            role="CAS store identity marker",
            allowed_modes=(0o400,),
        )
        try:
            before_status = os.fstat(descriptor)
            before = _Identity.regular(
                before_status,
                role="CAS store identity marker",
            )
            self._assert_store_identity_marker_status(before_status)
            payload = bytearray()
            while len(payload) <= _STORE_ID_PAYLOAD_BYTES:
                chunk = self._read_store_identity_chunk(
                    descriptor,
                    _STORE_ID_PAYLOAD_BYTES + 1 - len(payload),
                )
                if not chunk:
                    break
                payload.extend(chunk)
            store_id = _parse_store_id_payload(bytes(payload))
            if self._expected_store_id is not None and not secrets.compare_digest(
                store_id,
                self._expected_store_id,
            ):
                raise ArtifactIntegrityError(
                    "CAS store identity does not match expected external authority"
                )
            after_status = os.fstat(descriptor)
            after = _Identity.regular(
                after_status,
                role="CAS store identity marker",
            )
            self._assert_store_identity_marker_status(after_status)
            if before != after:
                raise ArtifactIntegrityError("CAS store identity marker changed during read")
            self._assert_name_matches_descriptor(
                root_descriptor,
                _STORE_ID_FILENAME,
                after,
                role="CAS store identity marker",
            )
            return store_id, after
        finally:
            _close_descriptors(descriptor, role="CAS store identity marker descriptor cleanup")

    @staticmethod
    def _read_store_identity_chunk(descriptor: int, maximum: int) -> bytes:
        while True:
            try:
                return os.read(descriptor, maximum)
            except InterruptedError:
                continue
            except OSError as exc:
                raise ArtifactStoreError(
                    "unable to read CAS store identity marker",
                    errno_code=exc.errno,
                ) from None

    def _assert_store_identity_marker(self, root_descriptor: int) -> None:
        if self._store_id is None or self._store_marker_identity is None:
            raise ArtifactBoundaryError("CAS store is not initialized")
        store_id, marker_identity = self._read_store_identity_marker(root_descriptor)
        if not secrets.compare_digest(store_id, self._store_id):
            raise ArtifactIntegrityError("CAS store identity marker value changed")
        if marker_identity != self._store_marker_identity:
            raise ArtifactIntegrityError("CAS store identity marker identity changed")

    @staticmethod
    def _assert_store_identity_marker_status(value: os.stat_result) -> None:
        ArtifactStore._assert_private_file_metadata(
            value,
            role="CAS store identity marker",
            allowed_modes=(0o400,),
        )
        if value.st_size != _STORE_ID_PAYLOAD_BYTES:
            raise ArtifactIntegrityError("CAS store identity marker has an invalid size")
        if value.st_nlink != 1:
            raise ArtifactIntegrityError(
                "CAS store identity marker has an unexpected hard-link count"
            )

    def _open_external_regular(
        self,
        value: str | os.PathLike[str],
        *,
        role: str,
    ) -> _OpenedFile:
        absolute, parts = self._validated_absolute_path(value, role=role)
        if not parts:
            raise ArtifactBoundaryError(f"{role} must name a file")
        try:
            parent = os.open(absolute.anchor, _DIRECTORY_FLAGS | _NOFOLLOW)
        except OSError as exc:  # pragma: no cover - platform/root failure
            raise ArtifactBoundaryError(
                f"{role} filesystem anchor is unavailable",
                errno_code=exc.errno,
            ) from None
        try:
            for part in parts[:-1]:
                next_descriptor = self._open_directory_at(
                    parent,
                    part,
                    role=f"{role} directory",
                    create=False,
                )
                previous_parent = parent
                parent = next_descriptor
                _close_descriptors(
                    previous_parent,
                    role=f"{role} directory walk cleanup",
                )
            name = parts[-1]
            before = self._stat_regular_at(parent, name, role=role)
            descriptor = self._open_regular_at(
                parent,
                name,
                role=role,
                allowed_modes=None,
            )
            opened = _Identity.regular(os.fstat(descriptor), role=role)
            if before != opened:
                _close_descriptors(descriptor, role=f"{role} descriptor cleanup")
                raise ArtifactIntegrityError(f"{role} changed during acquisition")
            return _OpenedFile(
                descriptor=descriptor,
                parent_descriptor=parent,
                name=name,
                identity=opened,
            )
        except Exception:
            _close_descriptors(parent, role=f"{role} parent descriptor cleanup")
            raise

    @staticmethod
    def _stat_regular_at(parent_descriptor: int, name: str, *, role: str) -> _Identity:
        try:
            status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactBoundaryError(
                f"{role} is unavailable",
                errno_code=exc.errno,
            ) from None
        return _Identity.regular(status, role=role)

    @staticmethod
    def _open_regular_at(
        parent_descriptor: int,
        name: str,
        *,
        role: str,
        allowed_modes: tuple[int, ...] | None,
    ) -> int:
        try:
            before_status = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ArtifactBoundaryError(
                f"{role} is unavailable",
                errno_code=exc.errno,
            ) from None
        if stat.S_ISLNK(before_status.st_mode):
            raise ArtifactBoundaryError(f"{role} cannot be opened safely")
        before = _Identity.regular(before_status, role=role)
        if allowed_modes is not None:
            ArtifactStore._assert_private_file_metadata(
                before_status,
                role=role,
                allowed_modes=allowed_modes,
            )
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ArtifactBoundaryError(
                f"{role} cannot be opened safely",
                errno_code=exc.errno,
            ) from None
        try:
            opened_status = os.fstat(descriptor)
            opened = _Identity.regular(opened_status, role=role)
            if not before.same_object(opened):
                raise ArtifactIntegrityError(f"{role} changed during acquisition")
            observed = _Identity.regular(
                os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False),
                role=role,
            )
            if not observed.same_object(opened):
                raise ArtifactIntegrityError(f"{role} changed during acquisition")
            if allowed_modes is not None:
                ArtifactStore._assert_private_file_metadata(
                    opened_status,
                    role=role,
                    allowed_modes=allowed_modes,
                )
                _assert_no_extended_acl(descriptor, role=role)
            return descriptor
        except Exception:
            _close_descriptors(descriptor, role=f"{role} descriptor cleanup")
            raise

    @staticmethod
    def _assert_private_file_metadata(
        value: os.stat_result,
        *,
        role: str,
        allowed_modes: tuple[int, ...],
    ) -> None:
        _Identity.regular(value, role=role)
        mode = stat.S_IMODE(value.st_mode)
        if mode not in allowed_modes:
            if role == "CAS object" and allowed_modes == (0o400,) and mode & 0o222:
                raise ArtifactIntegrityError("CAS object is not stored read-only")
            rendered_modes = "/".join(f"{allowed:o}" for allowed in allowed_modes)
            raise ArtifactBoundaryError(
                f"{role} permissions must be exact owner-only mode {rendered_modes}"
            )
        if hasattr(os, "geteuid") and value.st_uid != os.geteuid():
            raise ArtifactBoundaryError(f"{role} must be owned by the current user")

    def _assert_source_unchanged(self, source: _OpenedFile, before: _Identity) -> None:
        after = _Identity.regular(os.fstat(source.descriptor), role="artifact source")
        if before != after:
            raise ArtifactIntegrityError("artifact source changed during publication")
        self._assert_name_identity(
            source.parent_descriptor,
            source.name,
            before,
            role="artifact source",
            directory=False,
            require_full_identity=True,
        )

    def _create_staging(self, staging_descriptor: int) -> tuple[str, int]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW
        for _ in range(MAX_STAGING_ATTEMPTS):
            name = f"publish-{secrets.token_hex(24)}.tmp"
            try:
                descriptor = os.open(name, flags, 0o600, dir_fd=staging_descriptor)
            except FileExistsError:
                continue
            except OSError as exc:
                raise ArtifactStoreError(
                    "unable to create private staging file",
                    errno_code=exc.errno,
                ) from None
            try:
                status = os.fstat(descriptor)
                self._assert_private_file_metadata(
                    status,
                    role="artifact staging file",
                    allowed_modes=(0o600,),
                )
                _assert_no_extended_acl(descriptor, role="artifact staging file")
                return name, descriptor
            except Exception as exc:
                identity = _Identity.regular(
                    os.fstat(descriptor),
                    role="artifact staging file",
                )
                cleanup = _CleanupState.start(exc)
                cleanup.close("artifact staging descriptor close", descriptor)
                cleanup.attempt(
                    "artifact staging removal",
                    partial(
                        self._cleanup_staging_file,
                        staging_descriptor,
                        name,
                        identity,
                    ),
                )
                cleanup.finish("artifact staging acquisition cleanup")
                raise
        raise ArtifactStoreError("unable to allocate a unique artifact staging file")

    def _inspect_object_tree(
        self,
        store: _StoreDescriptors,
        *,
        cutoff_ns: int,
        budget: _InventoryBudget,
    ) -> list[ArtifactInventoryRecord]:
        records: list[ArtifactInventoryRecord] = []
        object_tree_identity, first_names = self._scan_directory_names(
            store.objects,
            budget=budget,
            role="CAS objects directory",
        )
        for first_name in first_names:
            self._validate_fanout_name(first_name)
            first_descriptor = self._open_directory_at(
                store.objects,
                first_name,
                role="CAS fanout directory",
                create=False,
            )
            try:
                self._assert_private_directory(
                    first_descriptor,
                    role="CAS fanout directory",
                )
                first_identity, second_names = self._scan_directory_names(
                    first_descriptor,
                    budget=budget,
                    role="CAS fanout directory",
                )
                for second_name in second_names:
                    self._validate_fanout_name(second_name)
                    second_descriptor = self._open_directory_at(
                        first_descriptor,
                        second_name,
                        role="CAS fanout directory",
                        create=False,
                    )
                    try:
                        self._assert_private_directory(
                            second_descriptor,
                            role="CAS fanout directory",
                        )
                        second_identity, object_names = self._scan_directory_names(
                            second_descriptor,
                            budget=budget,
                            role="CAS fanout directory",
                        )
                        for object_name in object_names:
                            digest = _validate_digest(object_name)
                            if digest[:2] != first_name or digest[2:4] != second_name:
                                raise ArtifactBoundaryError(
                                    "CAS object is outside its canonical digest fanout"
                                )
                            record, changed_ns = self._inspect_inventory_object(
                                second_descriptor,
                                digest,
                                budget=budget,
                            )
                            if changed_ns <= cutoff_ns:
                                if records and digest <= records[-1].digest:
                                    raise ArtifactIntegrityError(
                                        "CAS object traversal order changed"
                                    )
                                records.append(record)
                        self._assert_scanned_directory_unchanged(
                            first_descriptor,
                            second_name,
                            second_descriptor,
                            second_identity,
                            role="CAS fanout directory",
                        )
                    finally:
                        _close_descriptors(
                            second_descriptor,
                            role="CAS second-level inventory descriptor cleanup",
                        )
                self._assert_scanned_directory_unchanged(
                    store.objects,
                    first_name,
                    first_descriptor,
                    first_identity,
                    role="CAS fanout directory",
                )
            finally:
                _close_descriptors(
                    first_descriptor,
                    role="CAS first-level inventory descriptor cleanup",
                )
        self._assert_scanned_directory_unchanged(
            store.root,
            "objects",
            store.objects,
            object_tree_identity,
            role="CAS objects directory",
        )
        self._assert_initialized_bindings(store.root)
        budget.checkpoint()
        return records

    def _inspect_staging_tree(
        self,
        store: _StoreDescriptors,
        *,
        cutoff_ns: int,
        budget: _InventoryBudget,
    ) -> list[StagingInventoryRecord]:
        records: list[StagingInventoryRecord] = []
        staging_identity, staging_names = self._scan_directory_names(
            store.staging,
            budget=budget,
            role="CAS staging directory",
        )
        for staging_name in staging_names:
            _validate_staging_id(staging_name)
            record, changed_ns = self._inspect_staging_entry(
                store.staging,
                staging_name,
                budget=budget,
            )
            if changed_ns <= cutoff_ns:
                if records and staging_name <= records[-1].staging_id:
                    raise ArtifactIntegrityError("CAS staging traversal order changed")
                records.append(record)
        self._assert_scanned_directory_unchanged(
            store.root,
            ".staging",
            store.staging,
            staging_identity,
            role="CAS staging directory",
        )
        self._assert_initialized_bindings(store.root)
        budget.checkpoint()
        return records

    @staticmethod
    def _validate_fanout_name(value: str) -> str:
        if len(value) != 2 or any(character not in "0123456789abcdef" for character in value):
            raise ArtifactBoundaryError("CAS fanout entry is not canonical lowercase hexadecimal")
        return value

    @staticmethod
    def _scan_directory_names(
        descriptor: int,
        *,
        budget: _InventoryBudget,
        role: str,
    ) -> tuple[_Identity, list[str]]:
        budget.checkpoint()
        before = _Identity.directory(os.fstat(descriptor), role=role)
        names: list[str] = []
        try:
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    budget.observe_entry()
                    name = entry.name
                    if (
                        not isinstance(name, str)
                        or not name
                        or name in (".", "..")
                        or "/" in name
                        or os.sep in name
                        or "\x00" in name
                    ):
                        raise ArtifactBoundaryError(f"{role} contains an invalid entry name")
                    names.append(name)
        except OSError as exc:
            raise ArtifactStoreError(
                f"unable to inspect {role}",
                errno_code=exc.errno,
            ) from None
        budget.checkpoint()
        after = _Identity.directory(os.fstat(descriptor), role=role)
        if before != after:
            raise ArtifactIntegrityError(f"{role} changed during inventory")
        names.sort()
        return before, names

    def _inspect_inventory_object(
        self,
        parent_descriptor: int,
        digest: str,
        *,
        budget: _InventoryBudget | None,
        root_descriptor: int | None = None,
    ) -> tuple[ArtifactInventoryRecord, int]:
        if budget is not None:
            budget.checkpoint()
        if root_descriptor is None:
            descriptor = self._open_regular_at(
                parent_descriptor,
                digest,
                role="CAS object",
                allowed_modes=(0o400,),
            )
            before_status = os.fstat(descriptor)
            before = _Identity.regular(before_status, role="CAS object")
            self._assert_inventory_object_status(before_status)
        else:
            descriptor, before = self._open_single_link_object(
                root_descriptor,
                parent_descriptor,
                digest,
                expected_size=None,
            )
            before_status = os.fstat(descriptor)
        try:
            if before.size > self._max_artifact_bytes:
                raise ArtifactBoundaryError(
                    f"CAS object exceeds store limit of {self._max_artifact_bytes} bytes"
                )
            if (
                self._hash_descriptor(
                    descriptor,
                    maximum=before.size,
                    budget=budget,
                )
                != digest
            ):
                raise ArtifactIntegrityError("CAS object digest differs from its storage key")
            after_status = os.fstat(descriptor)
            after = _Identity.regular(after_status, role="CAS object")
            self._assert_inventory_object_status(after_status)
            if before != after:
                raise ArtifactIntegrityError("CAS object changed during inventory")
            self._assert_name_matches_descriptor(
                parent_descriptor,
                digest,
                after,
                role="CAS object",
            )
            changed_ns = _last_changed_nanoseconds(after_status)
            changed_at = _nanoseconds_to_utc(changed_ns, role="CAS object")
            return (
                ArtifactInventoryRecord(
                    digest=digest,
                    byte_size=after.size,
                    storage_key=_storage_key(digest),
                    last_changed_at=changed_at,
                    last_changed_ns=changed_ns,
                    generation=_artifact_generation(self.store_id, digest, after),
                ),
                changed_ns,
            )
        finally:
            _close_descriptors(descriptor, role="CAS inventory object descriptor cleanup")

    def _inspect_staging_entry(
        self,
        parent_descriptor: int,
        staging_id: str,
        *,
        budget: _InventoryBudget,
    ) -> tuple[StagingInventoryRecord, int]:
        budget.checkpoint()
        descriptor = self._open_regular_at(
            parent_descriptor,
            staging_id,
            role="CAS staging file",
            allowed_modes=(0o400, 0o600),
        )
        try:
            before_status = os.fstat(descriptor)
            before = _Identity.regular(before_status, role="CAS staging file")
            self._assert_staging_file_status(before_status)
            if before.size > self._max_artifact_bytes:
                raise ArtifactBoundaryError(
                    f"CAS staging file exceeds store limit of {self._max_artifact_bytes} bytes"
                )
            budget.checkpoint()
            after_status = os.fstat(descriptor)
            after = _Identity.regular(after_status, role="CAS staging file")
            self._assert_staging_file_status(after_status)
            if before != after:
                raise ArtifactIntegrityError("CAS staging file changed during inventory")
            self._assert_name_matches_descriptor(
                parent_descriptor,
                staging_id,
                after,
                role="CAS staging file",
            )
            changed_ns = _last_changed_nanoseconds(after_status)
            changed_at = _nanoseconds_to_utc(changed_ns, role="CAS staging file")
            return (
                StagingInventoryRecord(
                    staging_id=staging_id,
                    byte_size=after.size,
                    last_changed_at=changed_at,
                ),
                changed_ns,
            )
        finally:
            _close_descriptors(descriptor, role="CAS staging inventory descriptor cleanup")

    @staticmethod
    def _assert_inventory_object_status(value: os.stat_result) -> None:
        ArtifactStore._assert_cas_object_status(value, expected_link_count=1)

    @staticmethod
    def _assert_cas_object_status(
        value: os.stat_result,
        *,
        expected_link_count: int,
    ) -> None:
        """Validate one immutable object generation, including containment by link count."""

        _Identity.regular(value, role="CAS object")
        mode = stat.S_IMODE(value.st_mode)
        if mode & 0o222:
            raise ArtifactIntegrityError("CAS object is not stored read-only")
        if mode != 0o400:
            raise ArtifactBoundaryError("CAS object permissions are not private")
        if hasattr(os, "geteuid") and value.st_uid != os.geteuid():
            raise ArtifactBoundaryError("CAS object must be owned by the current user")
        if value.st_nlink != expected_link_count:
            raise ArtifactIntegrityError("CAS object has an unexpected hard-link count")

    @staticmethod
    def _assert_staging_file_status(value: os.stat_result) -> None:
        _Identity.regular(value, role="CAS staging file")
        if stat.S_IMODE(value.st_mode) not in (0o400, 0o600):
            raise ArtifactBoundaryError("CAS staging file permissions are unsafe")
        if hasattr(os, "geteuid") and value.st_uid != os.geteuid():
            raise ArtifactBoundaryError("CAS staging file must be owned by the current user")
        if value.st_nlink != 1:
            raise ArtifactIntegrityError("CAS staging file has an unexpected hard-link count")

    def _assert_scanned_directory_unchanged(
        self,
        parent_descriptor: int,
        name: str,
        descriptor: int,
        expected: _Identity,
        *,
        role: str,
    ) -> None:
        observed = _Identity.directory(os.fstat(descriptor), role=role)
        if observed != expected:
            raise ArtifactIntegrityError(f"{role} changed during inventory")
        self._assert_private_directory(descriptor, role=role)
        self._assert_name_identity(
            parent_descriptor,
            name,
            expected,
            role=role,
            directory=True,
            require_full_identity=True,
        )

    def _open_destination_parent(
        self,
        objects_descriptor: int,
        digest: str,
        *,
        create: bool,
    ) -> int:
        value = _validate_digest(digest)
        first: int | None = None
        try:
            first = self._open_directory_at(
                objects_descriptor,
                value[:2],
                role="CAS fanout directory",
                create=create,
            )
            self._assert_private_directory(first, role="CAS fanout directory")
            second = self._open_directory_at(
                first,
                value[2:4],
                role="CAS fanout directory",
                create=create,
            )
            self._assert_private_directory(second, role="CAS fanout directory")
            return second
        finally:
            _close_descriptors(first, role="CAS fanout descriptor cleanup")

    def _assert_destination_parent_binding(
        self,
        objects_descriptor: int,
        digest: str,
        expected_descriptor: int,
    ) -> None:
        rebound = self._open_destination_parent(objects_descriptor, digest, create=False)
        try:
            expected = _Identity.directory(
                os.fstat(expected_descriptor),
                role="CAS fanout directory",
            )
            observed = _Identity.directory(
                os.fstat(rebound),
                role="CAS fanout directory",
            )
            if not expected.same_object(observed):
                raise ArtifactIntegrityError("CAS fanout directory binding changed")
            self._assert_private_directory(rebound, role="CAS fanout directory")
        finally:
            _close_descriptors(rebound, role="CAS fanout rebind descriptor cleanup")

    @staticmethod
    def _link_staged(
        staging_directory: int,
        staging_name: str,
        destination_directory: int,
        destination_name: str,
    ) -> None:
        os.link(
            staging_name,
            destination_name,
            src_dir_fd=staging_directory,
            dst_dir_fd=destination_directory,
            follow_symlinks=False,
        )

    def _copy_and_hash(
        self,
        source_descriptor: int,
        staging_descriptor: int,
        *,
        expected_size: int,
    ) -> tuple[str, int]:
        hasher = hashlib.sha256()
        copied = 0
        while True:
            chunk = self._read_chunk(source_descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            copied += len(chunk)
            if copied > self._max_artifact_bytes or copied > expected_size:
                raise ArtifactBoundaryError("artifact source exceeded its declared size or limit")
            hasher.update(chunk)
            _write_all(staging_descriptor, chunk)
        return hasher.hexdigest(), copied

    @staticmethod
    def _read_chunk(descriptor: int, maximum: int) -> bytes:
        while True:
            try:
                return os.read(descriptor, maximum)
            except InterruptedError:
                continue
            except OSError as exc:
                raise ArtifactStoreError(
                    "unable to read artifact bytes",
                    errno_code=exc.errno,
                ) from None

    def _hash_descriptor(
        self,
        descriptor: int,
        *,
        maximum: int,
        budget: _InventoryBudget | None = None,
    ) -> str:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError as exc:
            raise ArtifactStoreError(
                "unable to rewind artifact bytes",
                errno_code=exc.errno,
            ) from None
        hasher = hashlib.sha256()
        observed = 0
        while True:
            if budget is not None:
                budget.checkpoint()
            chunk = self._read_chunk(descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum:
                raise ArtifactIntegrityError("CAS object exceeds its recorded size")
            hasher.update(chunk)
        if observed != maximum:
            raise ArtifactIntegrityError("CAS object size differs from registry metadata")
        return hasher.hexdigest()

    def _verify_object(
        self,
        parent_descriptor: int,
        name: str,
        *,
        digest: str,
        expected_size: int,
        root_descriptor: int,
    ) -> None:
        descriptor, before = self._open_single_link_object(
            root_descriptor,
            parent_descriptor,
            name,
            expected_size=expected_size,
        )
        try:
            if self._hash_descriptor(descriptor, maximum=expected_size) != digest:
                raise ArtifactIntegrityError("existing CAS object digest differs")
            after_status = os.fstat(descriptor)
            after = _Identity.regular(after_status, role="CAS object")
            self._assert_cas_object_status(after_status, expected_link_count=1)
            if before != after:
                raise ArtifactIntegrityError("CAS object changed during verification")
            self._assert_name_matches_descriptor(
                parent_descriptor,
                name,
                after,
                role="CAS object",
            )
        finally:
            _close_descriptors(descriptor, role="CAS object verification descriptor cleanup")

    def _open_single_link_object(
        self,
        root_descriptor: int,
        parent_descriptor: int,
        name: str,
        *,
        expected_size: int | None,
    ) -> tuple[int, _Identity]:
        """Open/status one canonical object while excluding transient publish links."""

        self._acquire_verification_lock(root_descriptor)
        descriptor: int | None = None
        primary_error: BaseException | None = None
        try:
            descriptor = self._open_regular_at(
                parent_descriptor,
                name,
                role="CAS object",
                allowed_modes=(0o400,),
            )
            status = os.fstat(descriptor)
            identity = _Identity.regular(status, role="CAS object")
            self._assert_cas_object_status(status, expected_link_count=1)
            if expected_size is not None and identity.size != expected_size:
                raise ArtifactIntegrityError("existing CAS object size differs")
            self._assert_name_matches_descriptor(
                parent_descriptor,
                name,
                identity,
                role="CAS object",
            )
            return descriptor, identity
        except BaseException as exc:
            primary_error = exc
            if descriptor is not None:
                _close_descriptors(
                    descriptor,
                    role="CAS object acquisition descriptor cleanup",
                )
                descriptor = None
            raise
        finally:
            cleanup = _CleanupState.start(primary_error)
            cleanup.attempt(
                "artifact verification lock release",
                lambda: self._release_verification_lock(root_descriptor),
            )
            if cleanup.failures:
                cleanup.close("CAS object descriptor close", descriptor)
            cleanup.finish("artifact verification cleanup")

    def _assert_publication_candidate(
        self,
        parent_descriptor: int,
        name: str,
        *,
        expected_size: int,
        expected_link_count: int,
        staging_identity: _Identity | None,
    ) -> None:
        """Validate the O(1) namespace state while the exclusive lock is held."""

        descriptor = self._open_regular_at(
            parent_descriptor,
            name,
            role="CAS object",
            allowed_modes=(0o400,),
        )
        try:
            status = os.fstat(descriptor)
            identity = _Identity.regular(status, role="CAS object")
            self._assert_cas_object_status(
                status,
                expected_link_count=expected_link_count,
            )
            if identity.size != expected_size:
                raise ArtifactIntegrityError("existing CAS object size differs")
            if staging_identity is not None and not identity.same_object(staging_identity):
                raise ArtifactIntegrityError("published CAS object differs from staging identity")
            self._assert_name_matches_descriptor(
                parent_descriptor,
                name,
                identity,
                role="CAS object",
            )
        finally:
            _close_descriptors(descriptor, role="CAS publication candidate descriptor cleanup")

    def _assert_name_matches_descriptor(
        self,
        parent_descriptor: int,
        name: str,
        expected: _Identity,
        *,
        role: str,
    ) -> None:
        self._assert_name_identity(
            parent_descriptor,
            name,
            expected,
            role=role,
            directory=False,
            require_full_identity=True,
        )

    @staticmethod
    def _assert_name_absent(
        parent_descriptor: int,
        name: str,
        *,
        role: str,
    ) -> None:
        try:
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return
            raise ArtifactStoreError(
                f"unable to prove {role} absence",
                errno_code=exc.errno,
            ) from None
        raise ArtifactIntegrityError(f"{role} is present")

    @staticmethod
    def _assert_name_identity(
        parent_descriptor: int,
        name: str,
        expected: _Identity,
        *,
        role: str,
        directory: bool,
        require_full_identity: bool = False,
    ) -> None:
        try:
            status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactIntegrityError(
                f"{role} binding is unavailable",
                errno_code=exc.errno,
            ) from None
        observed = (
            _Identity.directory(status, role=role)
            if directory
            else _Identity.regular(status, role=role)
        )
        if directory:
            if stat.S_IMODE(status.st_mode) != 0o700:
                raise ArtifactBoundaryError(f"{role} permissions must be exact owner-only mode 700")
            if hasattr(os, "geteuid") and status.st_uid != os.geteuid():
                raise ArtifactBoundaryError(f"{role} must be owned by the current user")
        matches = observed == expected if require_full_identity else observed.same_object(expected)
        if not matches:
            raise ArtifactIntegrityError(f"{role} binding changed")

    @staticmethod
    def _assert_descriptor_identity(
        descriptor: int,
        expected: _Identity,
        *,
        role: str,
    ) -> None:
        observed = _Identity.directory(os.fstat(descriptor), role=role)
        if not observed.same_object(expected):
            raise ArtifactIntegrityError(f"{role} identity changed")

    @staticmethod
    def _assert_private_directory(descriptor: int, *, role: str) -> None:
        status = os.fstat(descriptor)
        _Identity.directory(status, role=role)
        if stat.S_IMODE(status.st_mode) != 0o700:
            raise ArtifactBoundaryError(f"{role} permissions must be exact owner-only mode 700")
        if hasattr(os, "geteuid") and status.st_uid != os.geteuid():
            raise ArtifactBoundaryError(f"{role} must be owned by the current user")
        _assert_no_extended_acl(descriptor, role=role)

    def _cleanup_staging_file(
        self,
        staging_descriptor: int,
        staging_name: str,
        staging_identity: _Identity,
    ) -> None:
        """Remove one exact private staging generation and durably publish its absence."""

        self._unlink_name_if_same(
            staging_descriptor,
            staging_name,
            staging_identity,
            missing_ok=True,
            role="artifact staging cleanup",
        )
        self._fsync_directory_descriptor(staging_descriptor)

    def _unlink_name_if_same(
        self,
        parent_descriptor: int,
        name: str,
        expected: _Identity,
        *,
        missing_ok: bool,
        role: str,
        require_full_identity: bool = False,
        containment_objects_descriptor: int | None = None,
        containment_root_descriptor: int | None = None,
        containment_digest: str | None = None,
    ) -> None:
        containment = (
            containment_objects_descriptor,
            containment_root_descriptor,
            containment_digest,
        )
        if any(value is not None for value in containment):
            if not all(value is not None for value in containment):
                raise ArtifactBoundaryError("CAS unlink containment proof is incomplete")
            assert containment_objects_descriptor is not None
            assert containment_root_descriptor is not None
            assert containment_digest is not None
            self._assert_initialized_bindings(containment_root_descriptor)
            self._assert_destination_parent_binding(
                containment_objects_descriptor,
                containment_digest,
                parent_descriptor,
            )
        try:
            status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return
            raise ArtifactIntegrityError(f"{role} target disappeared") from None
        except OSError as exc:
            raise ArtifactIntegrityError(
                f"{role} target cannot be inspected",
                errno_code=exc.errno,
            ) from None
        observed = _Identity.regular(status, role=role)
        # Creating the destination hard link legitimately changes ctime and
        # link count on the staging inode. Cleanup is authorized by the stable
        # descriptor identity, not mutable timestamps.
        matches = observed == expected if require_full_identity else observed.same_object(expected)
        if not matches:
            raise ArtifactIntegrityError(f"{role} target changed; replacement preserved")
        try:
            os.unlink(name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            if not missing_ok:
                raise ArtifactIntegrityError(f"{role} target disappeared") from None
        except OSError as exc:
            raise ArtifactStoreError(
                f"{role} target cannot be removed",
                errno_code=exc.errno,
            ) from None

    @staticmethod
    def _fsync_file_descriptor(descriptor: int) -> None:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise ArtifactStoreError(
                "unable to durably flush artifact bytes",
                errno_code=exc.errno,
            ) from None

    @staticmethod
    def _make_file_read_only(descriptor: int) -> None:
        try:
            os.fchmod(descriptor, 0o400)
        except OSError as exc:
            raise ArtifactStoreError(
                "unable to make artifact staging bytes read-only",
                errno_code=exc.errno,
            ) from None

    @staticmethod
    def _fsync_directory_descriptor(descriptor: int) -> None:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise ArtifactStoreError(
                "unable to durably flush CAS directory",
                errno_code=exc.errno,
            ) from None


__all__ = [
    "ArtifactBoundaryError",
    "ArtifactGeneration",
    "ArtifactInventoryCursor",
    "ArtifactInventoryPage",
    "ArtifactInventoryRecord",
    "ArtifactIntegrityError",
    "ArtifactStore",
    "ArtifactStoreError",
    "MAX_ARTIFACT_BYTES",
    "MAX_IN_MEMORY_READ_BYTES",
    "PublishedArtifact",
    "StagingInventoryCursor",
    "StagingInventoryPage",
    "StagingInventoryRecord",
]
