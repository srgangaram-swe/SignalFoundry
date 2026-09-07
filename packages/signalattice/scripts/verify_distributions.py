"""Verify Signalattice release archives and isolated installed-distribution boundaries."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import re
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import default as default_email_policy
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Protocol

_MAX_ARCHIVE_CONTAINER_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 4_096
_MAX_ARCHIVE_FILES = 2_048
_MAX_ARCHIVE_PATH_BYTES = 1_024
_MAX_ARCHIVE_COMPONENT_BYTES = 255
_MAX_EXPANDED_TAR_BYTES = 40 * 1024 * 1024
_MAX_SPOOL_MEMORY_BYTES = 1 * 1024 * 1024
_STREAM_CHUNK_BYTES = 64 * 1024
_ZIP_EOCD_BYTES = 22
_ZIP_MAX_COMMENT_BYTES = 65_535
_ZIP_LOCAL_HEADER_BYTES = 30
_ZIP_CENTRAL_HEADER_BYTES = 46
_ZIP_DATA_DESCRIPTOR_FLAG = 0x08
_GZIP_HEADER_BYTES = 10
_GZIP_CANONICAL_FLAGS = 0x08
_PAX_MTIME = re.compile(r"^(?:0|[1-9][0-9]{0,9})(?:\.[0-9]{1,9})?$")
_TRACKING_MODULES = (
    "cas",
    "contracts",
    "migrations",
    "read_ports",
    "registry",
    "retention",
)
_SERVICE_MODULES = (
    "__init__",
    "admission",
    "api",
    "contracts",
    "entrypoint",
    "exporter",
    "http_protocol",
    "manifests",
    "metrics",
    "middleware",
    "models",
    "problems",
    "server",
    "telemetry",
    "telemetry_contracts",
)
_SERVICE_FRAMEWORK_DISTRIBUTIONS = frozenset({"fastapi", "starlette", "uvicorn"})
_SERVICE_FRAMEWORK_MODULES = ("fastapi", "starlette", "uvicorn")
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA_MARKER = re.compile(r'^\s*extra\s*==\s*["\']([A-Za-z0-9][A-Za-z0-9._-]*)["\']\s*$')
_EXPECTED_TRACKING_EXPORTS = [
    "ExperimentTracker",
    "LegacyTrackingError",
    "LegacyTrackingReadError",
    "LegacyTrackingWriteError",
    "RunContext",
    "get_tracker",
]
_REQUIRED_WHEEL_FILES = frozenset(
    {
        "quant_platform/py.typed",
        "quant_platform/service/__main__.py",
        "quant_platform/tracking/__init__.py",
        "quant_platform/tracking/experiment.py",
        *{f"quant_platform/tracking/{module_name}.py" for module_name in _TRACKING_MODULES},
        *{f"quant_platform/service/{module_name}.py" for module_name in _SERVICE_MODULES},
    }
)
_REQUIRED_WHEEL_METADATA_FILES = frozenset(
    {
        "METADATA",
        "RECORD",
        "WHEEL",
        "entry_points.txt",
        "licenses/LICENSE",
        "licenses/DISCLAIMER.md",
        "top_level.txt",
    }
)
_REQUIRED_SDIST_FILES = frozenset(
    {
        "DISCLAIMER.md",
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "docs/adr/0002-durable-local-registry.md",
        "docs/adr/0003-local-read-only-evidence-api.md",
        "docs/adr/0004-bounded-service-operability.md",
        "docs/api/openapi-v1.json",
        "docs/assets/service_operability_2026-08-09.png",
        "docs/assets/service_operability_2026-09-06.png",
        "docs/assets/service_operability_2026-09-06_patch1.png",
        "docs/api_service.md",
        "docs/benchmarks/service_operability_2026-08-09.json",
        "docs/benchmarks/service_operability_2026-09-06.json",
        "docs/benchmarks/service_operability_2026-09-06_patch1.json",
        "docs/run_registry.md",
        "docs/service_operations.md",
        "docs/threat_model.md",
        "scripts/benchmark_service_operability.py",
        "scripts/generate_service_openapi.py",
        "scripts/plot_service_operability.py",
        "scripts/summarize_experiments.py",
        "scripts/verify_service_container.py",
        "scripts/verify_distributions.py",
    }
)
_SDIST_ROOT_FILES = frozenset(
    {
        "DISCLAIMER.md",
        "LICENSE",
        "MANIFEST.in",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "setup.cfg",
    }
)
_SDIST_EGG_INFO_FILES = frozenset(
    {
        "src/signalattice.egg-info/PKG-INFO",
        "src/signalattice.egg-info/SOURCES.txt",
        "src/signalattice.egg-info/dependency_links.txt",
        "src/signalattice.egg-info/entry_points.txt",
        "src/signalattice.egg-info/requires.txt",
        "src/signalattice.egg-info/top_level.txt",
    }
)
_FORBIDDEN_SUFFIXES = (
    ".db",
    ".joblib",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pyc",
    ".pyo",
    ".sqlite",
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".DS_Store",
        ".env",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)
_FORBIDDEN_PART_KEYS = frozenset(part.casefold() for part in _FORBIDDEN_PARTS)
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_STEMS = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *{f"com{index}" for index in range(1, 10)},
        *{f"lpt{index}" for index in range(1, 10)},
    }
)
_NON_POSIX_CAS_SMOKE = "\n".join(
    (
        "import sys",
        "from pathlib import Path",
        "import quant_platform.tracking",
        "sys.modules.pop('fcntl', None)",
        "class BlockFcntl:",
        "    def find_spec(self, fullname, path=None, target=None):",
        "        if fullname == 'fcntl':",
        "            raise ModuleNotFoundError('blocked smoke capability', name=fullname)",
        "        return None",
        "sys.meta_path.insert(0, BlockFcntl())",
        "from quant_platform.tracking.cas import ArtifactBoundaryError, ArtifactStore",
        "root = Path(sys.argv[1])",
        "try:",
        "    ArtifactStore(root)",
        "except ArtifactBoundaryError as exc:",
        "    assert str(exc) == 'CAS requires POSIX advisory file-lock operations'",
        "else:",
        "    raise AssertionError('CAS construction did not fail closed')",
        "assert not root.exists()",
    )
)


class DistributionContractError(RuntimeError):
    """A built distribution violates its deterministic publication contract."""


def _normalize_distribution_name(value: str) -> str:
    """Return the canonical comparison key defined by Python packaging metadata."""

    return re.sub(r"[-_.]+", "-", value).lower()


def _parse_core_metadata(payload: bytes) -> Message:
    """Parse one bounded UTF-8 core metadata document without accepting parser defects."""

    if not 1 <= len(payload) <= _MAX_ARCHIVE_MEMBER_BYTES:
        raise DistributionContractError("wheel core metadata exceeds its byte ceiling")
    try:
        payload.decode("utf-8", errors="strict")
        document = BytesParser(policy=default_email_policy).parsebytes(payload)
    except (LookupError, MemoryError, UnicodeError, ValueError):
        raise DistributionContractError("wheel core metadata is malformed") from None
    if document.defects:
        raise DistributionContractError("wheel core metadata contains parser defects")
    return document


def _service_marker(requirement: str) -> tuple[str, str | None]:
    """Return a normalized distribution name and exact single-extra marker."""

    requirement_text, separator, marker_text = requirement.partition(";")
    name_match = _REQUIREMENT_NAME.match(requirement_text)
    if name_match is None:
        raise DistributionContractError("wheel contains a malformed dependency name")
    dependency_name = _normalize_distribution_name(name_match.group(1))
    if not separator:
        return dependency_name, None
    if ";" in marker_text:
        return dependency_name, "complex"
    marker_match = _EXTRA_MARKER.fullmatch(marker_text)
    if marker_match is None:
        return dependency_name, "complex"
    return dependency_name, _normalize_distribution_name(marker_match.group(1))


def _verify_service_extra_metadata(payload: bytes) -> None:
    """Prove service frameworks are opt-in and the service extra is exactly bounded."""

    document = _parse_core_metadata(payload)
    extras = tuple(
        _normalize_distribution_name(value)
        for value in document.get_all("Provides-Extra", failobj=[])
    )
    if extras.count("service") != 1:
        raise DistributionContractError("wheel must declare exactly one service extra")

    service_dependencies: list[str] = []
    for requirement in document.get_all("Requires-Dist", failobj=[]):
        if not isinstance(requirement, str) or len(requirement) > 2_048:
            raise DistributionContractError("wheel contains malformed dependency metadata")
        dependency_name, extra = _service_marker(str(requirement))
        if dependency_name in _SERVICE_FRAMEWORK_DISTRIBUTIONS:
            if extra is None or extra == "complex":
                raise DistributionContractError(
                    "service framework dependency is not isolated behind one optional extra"
                )
            if extra not in {"dev", "service"}:
                raise DistributionContractError(
                    "service framework dependency uses an unapproved optional extra"
                )
        if extra == "service":
            if dependency_name not in _SERVICE_FRAMEWORK_DISTRIBUTIONS:
                raise DistributionContractError(
                    "service extra contains a dependency outside its framework allowlist"
                )
            service_dependencies.append(dependency_name)
    if len(service_dependencies) != len(set(service_dependencies)):
        raise DistributionContractError("service extra contains a duplicate dependency")
    if frozenset(service_dependencies) != _SERVICE_FRAMEWORK_DISTRIBUTIONS:
        raise DistributionContractError("service extra omits a required framework dependency")


@dataclass(frozen=True, slots=True)
class _ZipEndRecord:
    """Bounded coordinates authenticated by the wheel's classic ZIP end record."""

    member_count: int
    central_offset: int
    central_size: int


@dataclass(frozen=True, slots=True)
class _ZipCentralRecord:
    """Security-relevant central-directory fields bound to one local member."""

    name: str
    name_bytes: bytes
    version_needed: int
    flags: int
    compression: int
    dos_time: int
    dos_date: int
    crc: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


class _BinaryDestination(Protocol):
    """Narrow write/rewind contract used by bounded archive expansion."""

    def write(self, data: bytes, /) -> int: ...

    def flush(self) -> None: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...


class _SeekableBinaryReader(Protocol):
    """Minimal reader contract used for compressed-header and tar-tail checks."""

    def read(self, size: int = -1, /) -> bytes: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...


class _SeekableBinaryStream(_BinaryDestination, _SeekableBinaryReader, Protocol):
    """Bounded binary spool contract required for exact tar-tail validation."""


def _check_archive_file(path: Path) -> None:
    """Reject indirection, special files, and oversized compressed containers before parsing."""

    try:
        metadata = path.lstat()
    except OSError:
        raise DistributionContractError(
            "distribution archive is not a readable regular file"
        ) from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise DistributionContractError("distribution archive must be a single-link regular file")
    if not 1 <= metadata.st_size <= _MAX_ARCHIVE_CONTAINER_BYTES:
        raise DistributionContractError("distribution archive exceeds its container byte ceiling")


def _member_path(name: str, *, is_dir: bool) -> PurePosixPath:
    """Return one canonical archive path or fail closed."""

    if (
        type(name) is not str
        or not name
        or not name.isascii()
        or len(name) > _MAX_ARCHIVE_PATH_BYTES
        or "\\" in name
        or "//" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise DistributionContractError("archive contains a noncanonical member name")
    trailing_slash = name.endswith("/")
    if trailing_slash and not is_dir:
        raise DistributionContractError("archive file member has a directory suffix")
    candidate = name[:-1] if trailing_slash else name
    path = PurePosixPath(candidate)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DistributionContractError("archive contains an unsafe member path")
    rendered = str(path) + ("/" if trailing_slash else "")
    if name != rendered:
        raise DistributionContractError("archive member name is not in canonical rendered form")
    if any(part.casefold() in _FORBIDDEN_PART_KEYS for part in path.parts):
        raise DistributionContractError("archive contains a forbidden generated or private path")
    for part in path.parts:
        if len(part) > _MAX_ARCHIVE_COMPONENT_BYTES:
            raise DistributionContractError("archive path component exceeds its byte ceiling")
        if part.endswith((".", " ")) or any(
            character in _WINDOWS_FORBIDDEN_CHARACTERS for character in part
        ):
            raise DistributionContractError("archive path is not portable to Windows filesystems")
        if part.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_STEMS:
            raise DistributionContractError("archive path uses a reserved Windows device basename")
    if path.name.endswith(_FORBIDDEN_SUFFIXES):
        raise DistributionContractError("archive contains a forbidden generated artifact type")
    return path


def _portable_path_key(path: PurePosixPath) -> str:
    """Return the cross-platform identity used for collision detection."""

    return "/".join(part.casefold() for part in path.parts)


def _check_member_counts(member_count: int, file_count: int) -> None:
    if not 1 <= member_count <= _MAX_ARCHIVE_MEMBERS:
        raise DistributionContractError("archive exceeds its member-count ceiling")
    if not 1 <= file_count <= _MAX_ARCHIVE_FILES:
        raise DistributionContractError("archive exceeds its file-count ceiling")


def _canonical_inventory(
    raw_members: tuple[tuple[str, bool], ...],
) -> tuple[PurePosixPath, ...]:
    """Parse one bounded inventory and reject aliases and file/directory collisions."""

    _check_member_counts(
        len(raw_members),
        sum(not is_dir for _, is_dir in raw_members),
    )
    parsed: list[PurePosixPath] = []
    kinds: dict[str, str] = {}
    required_directories: set[str] = set()
    for name, is_dir in raw_members:
        parsed.append(
            _register_canonical_member(
                name,
                is_dir=is_dir,
                kinds=kinds,
                required_directories=required_directories,
            )
        )

    _check_member_counts(
        len(parsed),
        sum(kind == "file" for kind in kinds.values()),
    )
    return tuple(parsed)


def _reject_unnecessary_directories(
    parsed: tuple[PurePosixPath, ...],
    raw_members: tuple[tuple[str, bool], ...],
) -> None:
    """Allow explicit directories only when at least one declared file requires them."""

    required: set[str] = set()
    declared: set[str] = set()
    for path, (_, is_dir) in zip(parsed, raw_members, strict=True):
        if is_dir:
            declared.add(_portable_path_key(path))
            continue
        parent = path.parent
        while str(parent) != ".":
            required.add(_portable_path_key(parent))
            parent = parent.parent
    if not declared.issubset(required):
        raise DistributionContractError(
            "archive contains an explicit directory outside its file inventory"
        )


def _register_canonical_member(
    name: str,
    *,
    is_dir: bool,
    kinds: dict[str, str],
    required_directories: set[str],
) -> PurePosixPath:
    """Register one canonical member and reject either ordering of a prefix collision."""

    member = _member_path(name, is_dir=is_dir)
    key = _portable_path_key(member)
    if key in kinds:
        raise DistributionContractError(
            "archive contains duplicate canonical or portable member paths"
        )
    if not is_dir and key in required_directories:
        raise DistributionContractError("archive contains a file/directory path collision")
    parent = member.parent
    while str(parent) != ".":
        parent_key = _portable_path_key(parent)
        if kinds.get(parent_key) == "file":
            raise DistributionContractError("archive contains a file/directory path collision")
        required_directories.add(parent_key)
        parent = parent.parent
    kinds[key] = "directory" if is_dir else "file"
    return member


def _check_sizes(sizes: tuple[int, ...]) -> None:
    if any(size < 0 or size > _MAX_ARCHIVE_MEMBER_BYTES for size in sizes):
        raise DistributionContractError("archive member exceeds its release byte ceiling")
    if sum(sizes) > _MAX_ARCHIVE_TOTAL_BYTES:
        raise DistributionContractError("archive exceeds its aggregate release byte ceiling")


def _preflight_zip_end_record(path: Path) -> _ZipEndRecord:
    """Read bounded classic-ZIP coordinates before materializing the central directory."""

    try:
        archive_size = path.stat().st_size
        tail_size = min(archive_size, _ZIP_EOCD_BYTES + _ZIP_MAX_COMMENT_BYTES)
        with path.open("rb") as source:
            source.seek(archive_size - tail_size)
            tail = source.read(tail_size)
    except OSError:
        raise DistributionContractError("wheel is unreadable or malformed") from None
    marker = b"PK\x05\x06"
    offset = tail.rfind(marker)
    if offset < 0 or len(tail) - offset < _ZIP_EOCD_BYTES:
        raise DistributionContractError("wheel has no bounded canonical end record")
    try:
        (
            signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_size,
        ) = struct.unpack_from("<4s4H2LH", tail, offset)
    except struct.error:
        raise DistributionContractError("wheel end record is malformed") from None
    if (
        signature != marker
        or comment_size != 0
        or comment_size != len(tail) - offset - _ZIP_EOCD_BYTES
    ):
        raise DistributionContractError("wheel end record or archive comment is not canonical")
    if (
        disk_number != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries in {0, 0xFFFF}
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        raise DistributionContractError("wheel uses an unsupported split or ZIP64 layout")
    if total_entries > _MAX_ARCHIVE_MEMBERS:
        raise DistributionContractError("archive exceeds its member-count ceiling")
    end_record_offset = archive_size - tail_size + offset
    if central_offset + central_size != end_record_offset:
        raise DistributionContractError("wheel central directory bounds are not canonical")
    return _ZipEndRecord(total_entries, central_offset, central_size)


def _decode_zip_name(raw_name: bytes, flags: int) -> str:
    """Decode a classic ZIP name exactly as the standard library will interpret it."""

    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        decoded = raw_name.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        raise DistributionContractError("wheel member name encoding is malformed") from None
    if decoded.encode(encoding) != raw_name:
        raise DistributionContractError("wheel member name encoding is not canonical")
    return decoded


def _read_zip_central_records(
    path: Path,
    end_record: _ZipEndRecord,
) -> tuple[_ZipCentralRecord, ...]:
    """Parse exactly the declared central records and reject undeclared record families."""

    try:
        with path.open("rb") as source:
            source.seek(end_record.central_offset)
            central = source.read(end_record.central_size)
    except OSError:
        raise DistributionContractError("wheel central directory cannot be read") from None
    if len(central) != end_record.central_size:
        raise DistributionContractError("wheel central directory is truncated")

    records: list[_ZipCentralRecord] = []
    cursor = 0
    for _ in range(end_record.member_count):
        fixed_end = cursor + _ZIP_CENTRAL_HEADER_BYTES
        if fixed_end > len(central):
            raise DistributionContractError("wheel central directory is truncated")
        try:
            (
                signature,
                _version_made,
                version_needed,
                flags,
                compression,
                dos_time,
                dos_date,
                crc,
                compressed_size,
                uncompressed_size,
                name_size,
                extra_size,
                comment_size,
                disk_start,
                _internal_attributes,
                _external_attributes,
                local_offset,
            ) = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        except struct.error:
            raise DistributionContractError("wheel central directory is malformed") from None
        if signature != b"PK\x01\x02":
            raise DistributionContractError("wheel contains an undeclared central record")
        if extra_size or comment_size:
            raise DistributionContractError("wheel member contains undeclared ZIP metadata")
        if disk_start != 0 or any(
            value == 0xFFFFFFFF for value in (compressed_size, uncompressed_size, local_offset)
        ):
            raise DistributionContractError("wheel uses an unsupported split or ZIP64 layout")
        record_end = fixed_end + name_size + extra_size + comment_size
        if record_end > len(central):
            raise DistributionContractError("wheel central record exceeds its declared bounds")
        raw_name = central[fixed_end : fixed_end + name_size]
        records.append(
            _ZipCentralRecord(
                name=_decode_zip_name(raw_name, flags),
                name_bytes=raw_name,
                version_needed=version_needed,
                flags=flags,
                compression=compression,
                dos_time=dos_time,
                dos_date=dos_date,
                crc=crc,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_offset=local_offset,
            )
        )
        cursor = record_end
    if cursor != len(central):
        raise DistributionContractError("wheel central directory has undeclared trailing records")
    return tuple(records)


def _verify_zip_local_layout(
    path: Path,
    end_record: _ZipEndRecord,
    central_records: tuple[_ZipCentralRecord, ...],
) -> None:
    """Bind every local header and payload extent exactly to its central record."""

    ordered = tuple(sorted(central_records, key=lambda record: record.local_offset))
    if len({record.local_offset for record in ordered}) != len(ordered):
        raise DistributionContractError("wheel local header offsets are not unique")
    expected_offset = 0
    try:
        with path.open("rb") as source:
            for central in ordered:
                if central.local_offset != expected_offset:
                    raise DistributionContractError(
                        "wheel contains a prefix, gap, or overlapping local member"
                    )
                source.seek(central.local_offset)
                fixed = source.read(_ZIP_LOCAL_HEADER_BYTES)
                if len(fixed) != _ZIP_LOCAL_HEADER_BYTES:
                    raise DistributionContractError("wheel local header is truncated")
                try:
                    (
                        signature,
                        version_needed,
                        flags,
                        compression,
                        dos_time,
                        dos_date,
                        crc,
                        compressed_size,
                        uncompressed_size,
                        name_size,
                        extra_size,
                    ) = struct.unpack("<4s5H3L2H", fixed)
                except struct.error:
                    raise DistributionContractError("wheel local header is malformed") from None
                if signature != b"PK\x03\x04":
                    raise DistributionContractError("wheel local header signature is malformed")
                if flags & _ZIP_DATA_DESCRIPTOR_FLAG:
                    raise DistributionContractError("wheel uses an unsupported data descriptor")
                if extra_size:
                    raise DistributionContractError(
                        "wheel local header contains undeclared ZIP metadata"
                    )
                local_name = source.read(name_size)
                if len(local_name) != name_size:
                    raise DistributionContractError("wheel local member name is truncated")
                observed = (
                    version_needed,
                    flags,
                    compression,
                    dos_time,
                    dos_date,
                    crc,
                    compressed_size,
                    uncompressed_size,
                    local_name,
                )
                declared = (
                    central.version_needed,
                    central.flags,
                    central.compression,
                    central.dos_time,
                    central.dos_date,
                    central.crc,
                    central.compressed_size,
                    central.uncompressed_size,
                    central.name_bytes,
                )
                if observed != declared:
                    raise DistributionContractError(
                        "wheel local header disagrees with its central record"
                    )
                expected_offset = (
                    central.local_offset + _ZIP_LOCAL_HEADER_BYTES + name_size + compressed_size
                )
                if expected_offset > end_record.central_offset:
                    raise DistributionContractError("wheel local member overlaps central metadata")
    except DistributionContractError:
        raise
    except OSError:
        raise DistributionContractError("wheel local layout cannot be read") from None
    if expected_offset != end_record.central_offset:
        raise DistributionContractError("wheel local layout does not meet its central directory")


def _verify_zip_directory_view(
    members: tuple[zipfile.ZipInfo, ...],
    central_records: tuple[_ZipCentralRecord, ...],
) -> None:
    """Ensure ``zipfile`` exposes exactly the independently parsed central directory."""

    if len(members) != len(central_records):
        raise DistributionContractError("wheel member count disagrees with its bounded end record")
    for member, record in zip(members, central_records, strict=True):
        if (
            member.filename != record.name
            or member.header_offset != record.local_offset
            or member.extract_version != record.version_needed
            or member.flag_bits != record.flags
            or member.compress_type != record.compression
            or record.crc != member.CRC
            or member.compress_size != record.compressed_size
            or member.file_size != record.uncompressed_size
        ):
            raise DistributionContractError(
                "wheel parser view disagrees with its bounded central directory"
            )


def _read_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    capture: bool = False,
) -> tuple[bytes, int, bytes | None]:
    """Stream one bounded member, forcing decompression, CRC, and declared-size validation."""

    digest = hashlib.sha256()
    total = 0
    captured = bytearray() if capture else None
    try:
        with archive.open(member, mode="r") as source:
            while chunk := source.read(_STREAM_CHUNK_BYTES):
                total += len(chunk)
                if total > _MAX_ARCHIVE_MEMBER_BYTES:
                    raise DistributionContractError(
                        "archive member exceeds its release byte ceiling"
                    )
                digest.update(chunk)
                if captured is not None:
                    captured.extend(chunk)
    except DistributionContractError:
        raise
    except (EOFError, OSError, RuntimeError, zipfile.BadZipFile, zlib.error):
        raise DistributionContractError("wheel member bytes are malformed") from None
    if total != member.file_size:
        raise DistributionContractError("wheel member size disagrees with its directory entry")
    return digest.digest(), total, None if captured is None else bytes(captured)


def _record_digest(value: str) -> bytes:
    match = re.fullmatch(r"sha256=([A-Za-z0-9_-]{43})", value)
    if match is None:
        raise DistributionContractError("wheel RECORD hash is not canonical SHA-256")
    encoded = match.group(1)
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=")
    except (binascii.Error, ValueError):
        raise DistributionContractError("wheel RECORD hash is malformed") from None
    if (
        len(decoded) != hashlib.sha256().digest_size
        or base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != encoded
    ):
        raise DistributionContractError("wheel RECORD hash is malformed")
    return decoded


def _verify_wheel_record(
    archive: zipfile.ZipFile,
    members: tuple[zipfile.ZipInfo, ...],
    parsed: tuple[PurePosixPath, ...],
    dist_info_root: str,
) -> None:
    """Bind every installable wheel file to exactly one canonical RECORD row."""

    actual = {
        str(member_path): member
        for member, member_path in zip(members, parsed, strict=True)
        if not member.is_dir()
    }
    record_name = f"{dist_info_root}/RECORD"
    record_member = actual.get(record_name)
    if record_member is None:
        raise DistributionContractError("wheel omits its required RECORD file")
    _, _, record_bytes = _read_zip_member(archive, record_member, capture=True)
    if record_bytes is None:
        raise DistributionContractError("wheel RECORD could not be read")
    try:
        record_text = record_bytes.decode("utf-8")
        rows = tuple(csv.reader(io.StringIO(record_text, newline=""), strict=True))
    except (csv.Error, MemoryError, UnicodeError):
        raise DistributionContractError("wheel RECORD is malformed") from None
    if not 1 <= len(rows) <= _MAX_ARCHIVE_FILES:
        raise DistributionContractError("wheel RECORD exceeds its row-count ceiling")

    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise DistributionContractError("wheel RECORD row has an unexpected shape")
        raw_name, hash_field, size_field = row
        canonical_name = str(_member_path(raw_name, is_dir=False))
        if canonical_name in recorded:
            raise DistributionContractError("wheel RECORD contains a duplicate canonical path")
        member = actual.get(canonical_name)
        if member is None:
            raise DistributionContractError("wheel RECORD references a missing member")
        recorded.add(canonical_name)
        if canonical_name == record_name:
            if hash_field or size_field:
                raise DistributionContractError("wheel RECORD must not self-hash")
            continue
        expected_digest = _record_digest(hash_field)
        if (
            len(size_field) > len(str(_MAX_ARCHIVE_MEMBER_BYTES))
            or re.fullmatch(r"0|[1-9][0-9]*", size_field) is None
        ):
            raise DistributionContractError("wheel RECORD size is not a canonical integer")
        try:
            expected_size = int(size_field)
        except ValueError:
            raise DistributionContractError("wheel RECORD size is malformed") from None
        if expected_size > _MAX_ARCHIVE_MEMBER_BYTES or expected_size != member.file_size:
            raise DistributionContractError("wheel RECORD size disagrees with its member")
        observed_digest, observed_size, _ = _read_zip_member(archive, member)
        if observed_size != expected_size or observed_digest != expected_digest:
            raise DistributionContractError("wheel RECORD hash or size verification failed")
    if recorded != set(actual):
        raise DistributionContractError("wheel RECORD does not cover the exact file inventory")


def verify_wheel(path: Path) -> None:
    """Verify the runtime-only wheel namespace and required tracking modules."""

    _check_archive_file(path)
    end_record = _preflight_zip_end_record(path)
    central_records = _read_zip_central_records(path, end_record)
    _verify_zip_local_layout(path, end_record, central_records)
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(archive.infolist())
            _verify_zip_directory_view(members, central_records)
            raw_inventory = tuple((member.filename, member.is_dir()) for member in members)
            parsed = _canonical_inventory(raw_inventory)
            _reject_unnecessary_directories(parsed, raw_inventory)
            _check_sizes(tuple(member.file_size for member in members if not member.is_dir()))
            if archive.comment:
                raise DistributionContractError("wheel contains an undeclared archive comment")
            for member in members:
                mode = member.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if member.flag_bits & 0x1:
                    raise DistributionContractError("wheel contains an encrypted member")
                if member.comment or member.extra:
                    raise DistributionContractError("wheel member contains undeclared ZIP metadata")
                if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise DistributionContractError("wheel uses an unsupported compression method")
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise DistributionContractError("wheel contains a non-file member")
                expected_type = stat.S_IFDIR if member.is_dir() else stat.S_IFREG
                if file_type != 0 and file_type != expected_type:
                    raise DistributionContractError(
                        "wheel member type disagrees with its canonical path form"
                    )

            dist_info_roots = {
                member.parts[0] for member in parsed if member.parts[0].endswith(".dist-info")
            }
            if len(dist_info_roots) != 1:
                raise DistributionContractError(
                    "wheel must contain exactly one dist-info directory"
                )
            dist_info_root = next(iter(dist_info_roots))
            if not dist_info_root.startswith("signalattice-"):
                raise DistributionContractError("wheel dist-info identity is not Signalattice")
            for member_path in parsed:
                if member_path.parts[0] not in {"quant_platform", dist_info_root}:
                    raise DistributionContractError(
                        "wheel contains a file outside its runtime allowlist"
                    )

            member_names = {
                str(member_path)
                for member, member_path in zip(members, parsed, strict=True)
                if not member.is_dir()
            }
            package_files = {name for name in member_names if name.startswith("quant_platform/")}
            if any(
                PurePosixPath(name).suffix != ".py" and PurePosixPath(name).name != "py.typed"
                for name in package_files
            ):
                raise DistributionContractError(
                    "wheel package contains a non-code runtime resource"
                )
            metadata_files = {
                str(PurePosixPath(*PurePosixPath(name).parts[1:]))
                for name in member_names
                if PurePosixPath(name).parts[0] == dist_info_root
            }
            if metadata_files != _REQUIRED_WHEEL_METADATA_FILES:
                raise DistributionContractError(
                    "wheel metadata does not match its exact publication inventory"
                )
            core_metadata_name = f"{dist_info_root}/METADATA"
            core_metadata_member = next(
                (
                    member
                    for member, member_path in zip(members, parsed, strict=True)
                    if str(member_path) == core_metadata_name and not member.is_dir()
                ),
                None,
            )
            if core_metadata_member is None:
                raise DistributionContractError("wheel omits its required core metadata")
            _, _, core_metadata = _read_zip_member(
                archive,
                core_metadata_member,
                capture=True,
            )
            if core_metadata is None:
                raise DistributionContractError("wheel core metadata could not be read")
            _verify_service_extra_metadata(core_metadata)
            missing = _REQUIRED_WHEEL_FILES.difference(member_names)
            if missing:
                raise DistributionContractError(
                    "wheel omits a required tracking or typing resource"
                )
            if any(
                name.startswith(("docs/", "scripts/", "configs/", "tests/"))
                for name in member_names
            ):
                raise DistributionContractError("runtime wheel contains repository-only resources")
            _verify_wheel_record(archive, members, parsed, dist_info_root)
    except (OSError, zipfile.BadZipFile):
        raise DistributionContractError("wheel is unreadable or malformed") from None


def _allowed_sdist_member(relative: PurePosixPath) -> bool:
    value = str(relative)
    if value in _SDIST_ROOT_FILES or value in _REQUIRED_SDIST_FILES:
        return True
    if value.startswith("src/quant_platform/"):
        return relative.suffix == ".py" or relative.name == "py.typed"
    if value.startswith("src/signalattice.egg-info/"):
        return value in _SDIST_EGG_INFO_FILES
    if value.startswith("tests/"):
        return relative.suffix == ".py"
    return False


def _preflight_gzip_header(source: _SeekableBinaryReader, path: Path) -> None:
    """Require the one canonical gzip header emitted by the pinned sdist builder."""

    header = source.read(_GZIP_HEADER_BYTES)
    if (
        len(header) != _GZIP_HEADER_BYTES
        or header[:3] != b"\x1f\x8b\x08"
        or header[3] != _GZIP_CANONICAL_FLAGS
    ):
        raise DistributionContractError(
            "source distribution gzip header contains unsupported metadata"
        )
    try:
        expected_name = path.name.removesuffix(".gz").encode("ascii")
    except UnicodeEncodeError:
        raise DistributionContractError(
            "source distribution gzip filename is not canonical"
        ) from None
    observed_name = bytearray()
    for _ in range(_MAX_ARCHIVE_COMPONENT_BYTES + 1):
        character = source.read(1)
        if character == b"\0":
            break
        if len(character) != 1:
            raise DistributionContractError("source distribution gzip header is truncated")
        observed_name.extend(character)
    else:
        raise DistributionContractError(
            "source distribution gzip filename exceeds its byte ceiling"
        )
    if bytes(observed_name) != expected_name:
        raise DistributionContractError("source distribution gzip filename is not canonical")
    source.seek(0)


def _expand_sdist(path: Path, destination: _BinaryDestination) -> int:
    """Expand exactly one gzip member under a ceiling before tar metadata parsing."""

    expanded_bytes = 0

    def persist(chunk: bytes) -> None:
        nonlocal expanded_bytes
        expanded_bytes += len(chunk)
        if expanded_bytes > _MAX_EXPANDED_TAR_BYTES:
            raise DistributionContractError(
                "source distribution exceeds its expanded-tar byte ceiling"
            )
        try:
            written = destination.write(chunk)
        except OSError:
            raise DistributionContractError(
                "source distribution expansion could not be persisted"
            ) from None
        if written != len(chunk):
            raise DistributionContractError("source distribution expansion could not be persisted")

    try:
        with path.open(mode="rb") as compressed:
            _preflight_gzip_header(compressed, path)
            decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
            finished = False
            while compressed_chunk := compressed.read(_STREAM_CHUNK_BYTES):
                pending = compressed_chunk
                while pending:
                    pending_size = len(pending)
                    chunk = decompressor.decompress(pending, _STREAM_CHUNK_BYTES)
                    pending = decompressor.unconsumed_tail
                    if not chunk and len(pending) >= pending_size:
                        raise DistributionContractError(
                            "source distribution gzip stream made no bounded progress"
                        )
                    persist(chunk)
                    if decompressor.eof:
                        if decompressor.unused_data or pending or compressed.read(1):
                            raise DistributionContractError(
                                "source distribution contains multiple gzip members"
                            )
                        finished = True
                        break
                if finished:
                    break
            if not finished or not decompressor.eof:
                raise DistributionContractError("source distribution gzip stream is truncated")
        if expanded_bytes == 0 or expanded_bytes % tarfile.BLOCKSIZE != 0:
            raise DistributionContractError("source distribution expanded tar is not block-aligned")
        try:
            destination.flush()
            destination.seek(0)
        except OSError:
            raise DistributionContractError(
                "source distribution expansion could not be persisted"
            ) from None
    except DistributionContractError:
        raise
    except (EOFError, OSError, zlib.error):
        raise DistributionContractError("source distribution gzip stream is malformed") from None
    return expanded_bytes


def _verify_tar_end_padding(
    source: _SeekableBinaryStream,
    *,
    logical_end: int,
    expanded_size: int,
) -> None:
    """Reject hidden bytes and noncanonical padding after the first tar end marker."""

    minimum_padding = 2 * tarfile.BLOCKSIZE
    padding_size = expanded_size - logical_end
    if (
        type(logical_end) is not int
        or logical_end < 0
        or logical_end % tarfile.BLOCKSIZE != 0
        or expanded_size % tarfile.BLOCKSIZE != 0
        or padding_size < minimum_padding
    ):
        raise DistributionContractError(
            "source distribution has no canonical two-record tar terminator"
        )
    try:
        source.seek(logical_end)
        remaining = padding_size
        while remaining:
            chunk = source.read(min(_STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                raise DistributionContractError("source distribution tar terminator is truncated")
            if any(chunk):
                raise DistributionContractError(
                    "source distribution contains hidden bytes after tar EOF"
                )
            remaining -= len(chunk)
    except DistributionContractError:
        raise
    except OSError:
        raise DistributionContractError(
            "source distribution tar terminator cannot be read"
        ) from None
    # tarfile writes the two-block terminator and then zero-fills to the next
    # RECORDSIZE boundary, so padding is 2*BLOCKSIZE plus a fill of up to
    # RECORDSIZE - BLOCKSIZE (9728). Total padding therefore reaches 10752,
    # above RECORDSIZE, for perfectly canonical archives -- roughly one archive
    # size in twenty. The bound belongs on the fill, which must stay under a
    # full record; an extra whole record of zeros is what non-canonical means.
    if (
        padding_size - minimum_padding >= tarfile.RECORDSIZE
        or expanded_size % tarfile.RECORDSIZE != 0
    ):
        raise DistributionContractError("source distribution tar end padding is not canonical")


def verify_sdist(path: Path) -> None:
    """Verify the source archive's narrow code, test, documentation, and utility allowlist."""

    _check_archive_file(path)
    try:
        with tempfile.SpooledTemporaryFile(
            max_size=_MAX_SPOOL_MEMORY_BYTES,
            mode="w+b",
        ) as expanded:
            expanded_size = _expand_sdist(path, expanded)
            with tarfile.open(fileobj=expanded, mode="r|") as archive:
                logical_end = _verify_sdist_members(archive)
            _verify_tar_end_padding(
                expanded,
                logical_end=logical_end,
                expanded_size=expanded_size,
            )
    except DistributionContractError:
        raise
    except (OSError, tarfile.TarError):
        raise DistributionContractError("source distribution is unreadable or malformed") from None


def _verify_sdist_members(archive: tarfile.TarFile) -> int:
    """Stream and validate one already size-bounded, uncompressed tar archive."""

    try:
        raw_members: list[tuple[str, bool]] = []
        parsed_members: list[PurePosixPath] = []
        declared_sizes: list[int] = []
        file_count = 0
        kinds: dict[str, str] = {}
        required_directories: set[str] = set()
        for raw in archive:
            if len(raw_members) >= _MAX_ARCHIVE_MEMBERS:
                raise DistributionContractError("archive exceeds its member-count ceiling")
            if archive.pax_headers:
                raise DistributionContractError(
                    "source distribution contains undeclared global PAX metadata"
                )
            if raw.pax_headers:
                if set(raw.pax_headers) != {"mtime"}:
                    raise DistributionContractError(
                        "source distribution contains undeclared member PAX metadata"
                    )
                pax_mtime = raw.pax_headers["mtime"]
                if type(pax_mtime) is not str or _PAX_MTIME.fullmatch(pax_mtime) is None:
                    raise DistributionContractError(
                        "source distribution member PAX mtime is not canonical"
                    )
            if not (raw.isfile() or raw.isdir()):
                raise DistributionContractError("source distribution contains a non-file member")
            if raw.linkname:
                raise DistributionContractError(
                    "source distribution file metadata contains an undeclared link target"
                )
            if raw.devmajor != 0 or raw.devminor != 0:
                raise DistributionContractError(
                    "source distribution file metadata contains device identifiers"
                )
            if type(raw.mode) is not int or raw.mode < 0 or raw.mode & ~0o777 or raw.mode & 0o022:
                raise DistributionContractError(
                    "source distribution member permissions are unsafe or noncanonical"
                )
            parsed_members.append(
                _register_canonical_member(
                    raw.name,
                    is_dir=raw.isdir(),
                    kinds=kinds,
                    required_directories=required_directories,
                )
            )
            raw_members.append((raw.name, raw.isdir()))
            if raw.isdir():
                continue
            file_count += 1
            if file_count > _MAX_ARCHIVE_FILES:
                raise DistributionContractError("archive exceeds its file-count ceiling")
            declared_sizes.append(raw.size)
            _check_sizes(tuple(declared_sizes))
            extracted = archive.extractfile(raw)
            if extracted is None:
                raise DistributionContractError("source distribution member cannot be read")
            observed_size = 0
            try:
                with extracted:
                    while chunk := extracted.read(_STREAM_CHUNK_BYTES):
                        observed_size += len(chunk)
                        if observed_size > _MAX_ARCHIVE_MEMBER_BYTES:
                            raise DistributionContractError(
                                "archive member exceeds its release byte ceiling"
                            )
            except DistributionContractError:
                raise
            except (EOFError, OSError, tarfile.TarError):
                raise DistributionContractError(
                    "source distribution member bytes are malformed"
                ) from None
            if observed_size != raw.size:
                raise DistributionContractError(
                    "source distribution member size disagrees with its header"
                )

        parsed = tuple(parsed_members)
        _reject_unnecessary_directories(parsed, tuple(raw_members))
        _check_member_counts(len(parsed), file_count)
        _check_sizes(tuple(declared_sizes))
        roots = {member.parts[0] for member in parsed}
        if len(roots) != 1:
            raise DistributionContractError("source distribution must have one release root")
        release_root = next(iter(roots))
        if not release_root.startswith("signalattice-"):
            raise DistributionContractError("source distribution root is not Signalattice")

        relative_files = {
            str(PurePosixPath(*member.parts[1:]))
            for member, (_, is_dir) in zip(parsed, raw_members, strict=True)
            if len(member.parts) > 1 and not is_dir
        }
        if any(not _allowed_sdist_member(PurePosixPath(relative)) for relative in relative_files):
            raise DistributionContractError(
                "source distribution contains a file outside its allowlist"
            )
        if _REQUIRED_SDIST_FILES.difference(relative_files):
            raise DistributionContractError("source distribution omits required operator resources")
        if _SDIST_EGG_INFO_FILES.difference(relative_files):
            raise DistributionContractError("source distribution omits required build metadata")
        if any(name.startswith("configs/") for name in relative_files):
            raise DistributionContractError(
                "repository-only example configs entered the source archive"
            )
        return archive.offset
    except DistributionContractError:
        raise
    except (OSError, tarfile.TarError):
        raise DistributionContractError("source distribution is unreadable or malformed") from None


def _module_location(module: ModuleType, prefix: Path) -> Path:
    value = getattr(module, "__file__", None)
    if type(value) is not str:
        raise DistributionContractError("installed tracking module has no concrete file location")
    location = Path(value).resolve()
    if not location.is_relative_to(prefix):
        raise DistributionContractError(
            "installed tracking import escaped its isolated environment"
        )
    return location


def _verify_non_posix_import_boundary(prefix: Path) -> None:
    """Prove the installed distribution remains importable when POSIX locks are absent."""

    if Path(sys.prefix).resolve() != prefix:
        raise DistributionContractError("package smoke is not running from the installed prefix")
    executable = Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix="signalattice-non-posix-smoke-") as directory:
        root = Path(directory) / "cas"
        try:
            result = subprocess.run(
                [str(executable), "-I", "-c", _NON_POSIX_CAS_SMOKE, str(root)],
                cwd=directory,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise DistributionContractError(
                "installed non-POSIX import smoke could not complete"
            ) from None
        if result.returncode != 0 or root.exists():
            raise DistributionContractError(
                "installed non-POSIX CAS boundary is not import-safe and fail-closed"
            )


def _installed_metadata_bytes(distribution: importlib.metadata.Distribution) -> bytes:
    """Read installed core metadata without relying on the caller's working tree."""

    try:
        metadata = distribution.read_text("METADATA")
    except (OSError, UnicodeError):
        raise DistributionContractError("installed core metadata cannot be read") from None
    if type(metadata) is not str:
        raise DistributionContractError("installed distribution omits core metadata")
    try:
        return metadata.encode("utf-8")
    except UnicodeEncodeError:
        raise DistributionContractError("installed core metadata is not valid UTF-8") from None


def _verify_missing_service_extra_cli(prefix: Path) -> None:
    """Prove a core install has no service frameworks and fails with operator guidance."""

    if Path(sys.prefix).resolve() != prefix:
        raise DistributionContractError("package smoke is not running from the installed prefix")
    if any(importlib.util.find_spec(name) is not None for name in _SERVICE_FRAMEWORK_MODULES):
        raise DistributionContractError(
            "core installation unexpectedly contains service frameworks"
        )

    executable = Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix="signalattice-core-service-smoke-") as directory:
        smoke_root = Path(directory)
        registry = smoke_root / "registry.db"
        registry.touch(mode=0o600)
        cas_root = smoke_root / "cas"
        cas_root.mkdir(mode=0o700)
        before = frozenset(smoke_root.iterdir())
        try:
            result = subprocess.run(
                [
                    str(executable),
                    "-I",
                    "-m",
                    "quant_platform.cli",
                    "serve-api",
                    "--registry-db",
                    str(registry),
                    "--cas-root",
                    str(cas_root),
                ],
                cwd=directory,
                capture_output=True,
                check=False,
                text=True,
                # A cold CLI import may initialize NumPy/Matplotlib runtime caches in the
                # isolated environment; keep the process bounded without treating cold start as
                # a ten-second correctness gate.
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise DistributionContractError(
                "core installation service-boundary smoke could not complete"
            ) from None
        rendered = result.stdout + result.stderr
        if (
            result.returncode == 0
            or "optional service dependencies" not in rendered
            or "signalattice[service]" not in rendered
        ):
            raise DistributionContractError(
                "core installation does not fail actionably when the service extra is absent"
            )
        if frozenset(smoke_root.iterdir()) != before or tuple(cas_root.iterdir()):
            raise DistributionContractError(
                "missing-service-extra failure mutated the supplied evidence paths"
            )


class _ServiceSmokePorts:
    """Read-port shape whose methods must remain untouched by liveness and schema reads."""

    @staticmethod
    def _unexpected(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("service metadata smoke attempted an evidence read")

    probe_evidence_readiness = _unexpected
    get_run = _unexpected
    list_runs = _unexpected
    get_artifact = _unexpected
    list_run_artifacts = _unexpected
    read_verified_manifest = _unexpected


async def _invoke_service_asgi(app: Any, path: str) -> tuple[int, dict[bytes, bytes], bytes]:
    """Invoke one body-free loopback GET directly through the installed ASGI boundary."""

    messages: list[dict[str, Any]] = []
    receive_count = 0

    async def receive() -> dict[str, Any]:
        nonlocal receive_count
        receive_count += 1
        if receive_count == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.5"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "GET",
        "root_path": "",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 30_000),
        "server": ("127.0.0.1", 8_765),
        "state": {},
    }
    try:
        await asyncio.wait_for(app(scope, receive, send), timeout=5)
    except (TimeoutError, OSError, RuntimeError):
        raise DistributionContractError("installed service ASGI smoke did not complete") from None

    starts = [message for message in messages if message.get("type") == "http.response.start"]
    bodies = [message for message in messages if message.get("type") == "http.response.body"]
    if len(starts) != 1 or not bodies or bool(bodies[-1].get("more_body", False)):
        raise DistributionContractError("installed service emitted an invalid ASGI response")
    status = starts[0].get("status")
    raw_headers = starts[0].get("headers")
    if type(status) is not int or type(raw_headers) is not list:
        raise DistributionContractError("installed service emitted malformed ASGI metadata")
    try:
        headers = {bytes(name).lower(): bytes(value) for name, value in raw_headers}
        body = b"".join(bytes(message.get("body", b"")) for message in bodies)
    except (TypeError, ValueError):
        raise DistributionContractError("installed service emitted malformed ASGI bytes") from None
    return status, headers, body


def _verify_installed_service_asgi() -> None:
    """Exercise liveness and deterministic OpenAPI without reading evidence storage."""

    from quant_platform.service.api import assert_read_only_route_inventory, create_app

    ports = _ServiceSmokePorts()
    app = create_app(ports, max_concurrency=2)
    assert_read_only_route_inventory(app)

    async def smoke() -> None:
        live_status, live_headers, live_body = await _invoke_service_asgi(app, "/health/live")
        metrics_status, metrics_headers, metrics_body = await _invoke_service_asgi(
            app,
            "/internal/metrics",
        )
        schema_status, schema_headers, schema_body = await _invoke_service_asgi(
            app,
            "/api/v1/openapi.json",
        )
        _, _, repeated_schema_body = await _invoke_service_asgi(app, "/api/v1/openapi.json")
        required_headers = {
            b"cache-control": b"no-store",
            b"referrer-policy": b"no-referrer",
            b"x-content-type-options": b"nosniff",
        }
        if live_status != 200 or json.loads(live_body) != {"schema_version": 1, "status": "live"}:
            raise DistributionContractError("installed service liveness smoke failed")
        if any(live_headers.get(name) != value for name, value in required_headers.items()):
            raise DistributionContractError("installed service omits required response hardening")
        if (
            metrics_status != 200
            or len(metrics_body) > 256 * 1024
            or metrics_headers.get(b"content-type") != b"text/plain; version=0.0.4; charset=utf-8"
            or b"signalattice_http_requests_total" not in metrics_body
        ):
            raise DistributionContractError("installed service metrics boundary is invalid")
        if schema_status != 200 or schema_body != repeated_schema_body:
            raise DistributionContractError("installed service OpenAPI output is not deterministic")
        if any(schema_headers.get(name) != value for name, value in required_headers.items()):
            raise DistributionContractError("installed OpenAPI response omits required hardening")
        try:
            schema = json.loads(schema_body)
            paths = schema["paths"]
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError):
            raise DistributionContractError(
                "installed service OpenAPI document is malformed"
            ) from None
        if (
            type(paths) is not dict
            or not paths
            or any(
                type(operations) is not dict or set(operations) != {"get"}
                for operations in paths.values()
            )
        ):
            raise DistributionContractError("installed service OpenAPI exposes a non-GET operation")
        operation_ids = [
            operation.get("operationId")
            for operations in paths.values()
            for operation in operations.values()
            if type(operation) is dict
        ]
        if (
            len(operation_ids) != len(paths)
            or any(type(value) is not str or not value for value in operation_ids)
            or len(operation_ids) != len(set(operation_ids))
        ):
            raise DistributionContractError(
                "installed OpenAPI operation IDs are not stable and unique"
            )
        if {"/docs", "/redoc", "/openapi.json", "/internal/metrics"}.intersection(paths):
            raise DistributionContractError(
                "installed service exposes interactive or internal documentation"
            )

    try:
        asyncio.run(smoke())
    except DistributionContractError:
        raise
    except (AssertionError, MemoryError, RuntimeError, ValueError):
        raise DistributionContractError("installed service ASGI/OpenAPI smoke failed") from None


def verify_installed_service_distribution(prefix: Path) -> None:
    """Verify the optional wheel service boundary from an isolated installation."""

    expected_prefix = prefix.resolve(strict=True)
    if Path(sys.prefix).resolve() != expected_prefix:
        raise DistributionContractError("service smoke is not running from the installed prefix")
    before = frozenset(Path.cwd().iterdir())
    distribution = importlib.metadata.distribution("signalattice")
    _verify_service_extra_metadata(_installed_metadata_bytes(distribution))
    modules = {
        name: importlib.import_module(
            "quant_platform.service" if name == "__init__" else f"quant_platform.service.{name}"
        )
        for name in _SERVICE_MODULES
    }
    for module in modules.values():
        _module_location(module, expected_prefix)
    if importlib.util.find_spec("httpx") is not None:
        raise DistributionContractError("service extra unexpectedly includes the HTTP test client")

    from quant_platform.service.server import ServerConfig, build_uvicorn_config

    server = build_uvicorn_config(_ServiceSmokePorts(), ServerConfig())
    configured_protocol = server.http
    protocol_base = modules["http_protocol"].SignalatticeH11Protocol
    if (
        type(configured_protocol) is not type
        or configured_protocol is protocol_base
        or not issubclass(configured_protocol, protocol_base)
        or configured_protocol.__module__ != "quant_platform.service.http_protocol"
        or configured_protocol.__name__ != "ConfiguredSignalatticeH11Protocol"
    ):
        raise DistributionContractError("installed service Uvicorn protocol adapter weakened")
    observed_profile = (
        server.host,
        server.port,
        server.uds,
        server.ws,
        server.lifespan,
        server.loop,
        server.interface,
        server.workers,
        server.reload,
        server.proxy_headers,
        server.forwarded_allow_ips,
        server.access_log,
        server.server_header,
        server.date_header,
        server.limit_concurrency,
        server.backlog,
        server.timeout_keep_alive,
        server.timeout_graceful_shutdown,
        server.h11_max_incomplete_event_size,
        server.reset_contextvars,
    )
    expected_profile = (
        "127.0.0.1",
        8_765,
        None,
        "none",
        "on",
        "asyncio",
        "asgi3",
        1,
        False,
        False,
        "",
        False,
        False,
        False,
        32,
        64,
        3,
        10,
        16 * 1_024,
        True,
    )
    if observed_profile != expected_profile:
        raise DistributionContractError("installed service Uvicorn profile weakened")
    _verify_installed_service_asgi()
    if frozenset(Path.cwd().iterdir()) != before:
        raise DistributionContractError("service imports or smoke created working-tree files")


def verify_installed_distribution(prefix: Path) -> None:
    """Import every tracking surface from one isolated, non-editable installation."""

    expected_prefix = prefix.resolve(strict=True)
    before = frozenset(Path.cwd().iterdir())
    package = importlib.import_module("quant_platform")
    tracking = importlib.import_module("quant_platform.tracking")
    modules = {
        name: importlib.import_module(f"quant_platform.tracking.{name}")
        for name in _TRACKING_MODULES
    }
    _module_location(package, expected_prefix)
    _module_location(tracking, expected_prefix)
    for module in modules.values():
        _module_location(module, expected_prefix)

    distribution = importlib.metadata.distribution("signalattice")
    if package.__version__ != distribution.version:
        raise DistributionContractError("installed package and distribution versions disagree")
    _verify_service_extra_metadata(_installed_metadata_bytes(distribution))
    installed_files = {str(file) for file in distribution.files or ()}
    if _REQUIRED_WHEEL_FILES.difference(installed_files):
        raise DistributionContractError("installed distribution omits a required tracking resource")
    if getattr(tracking, "__all__", None) != _EXPECTED_TRACKING_EXPORTS:
        raise DistributionContractError("tracking package exports changed")

    contracts = modules["contracts"]
    migrations = modules["migrations"]
    limits = contracts.RegistryLimits()
    request = contracts.SubmissionRequest("package-smoke", {"seed": 20260808})
    synthetic_migration = migrations.Migration(2, "package_smoke", "SELECT 1;")
    if limits.max_page_size < 1 or request.kind != "package-smoke":
        raise DistributionContractError("installed tracking contracts are not constructible")
    if synthetic_migration.checksum != hashlib.sha256(b"SELECT 1;").hexdigest():
        raise DistributionContractError("installed migration checksum contract is inconsistent")
    compiled = migrations.MIGRATIONS
    if (
        not compiled
        or tuple(migration.version for migration in compiled)
        != tuple(range(1, migrations.LATEST_SCHEMA_VERSION + 1))
        or any(len(migration.checksum) != 64 for migration in compiled)
    ):
        raise DistributionContractError("installed compiled migration ledger is not contiguous")
    if frozenset(Path.cwd().iterdir()) != before:
        raise DistributionContractError("tracking imports created files in the smoke directory")
    _verify_non_posix_import_boundary(expected_prefix)
    _verify_missing_service_extra_cli(expected_prefix)


def _single_archive(directory: Path, pattern: str, label: str) -> Path:
    matches = tuple(sorted(directory.glob(pattern)))
    if len(matches) != 1:
        raise DistributionContractError(f"dist directory must contain exactly one {label}")
    return matches[0]


def _distribution_pair(directory: Path) -> tuple[Path, Path]:
    """Resolve the exact two-file publication directory or fail closed."""

    try:
        entries = tuple(directory.iterdir())
    except OSError:
        raise DistributionContractError("dist directory cannot be inventoried") from None
    if len(entries) != 2:
        raise DistributionContractError(
            "dist directory must contain exactly the wheel and source distribution"
        )
    wheel = _single_archive(directory, "*.whl", "wheel")
    sdist = _single_archive(directory, "*.tar.gz", "source distribution")
    if frozenset(entries) != {wheel, sdist}:
        raise DistributionContractError("dist directory contains an undeclared publication entry")
    return wheel, sdist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument("--installed-prefix", type=Path)
    parser.add_argument("--service-prefix", type=Path)
    args = parser.parse_args()
    if args.dist_dir is None and args.installed_prefix is None and args.service_prefix is None:
        parser.error("at least one verification boundary is required")
    try:
        if args.dist_dir is not None:
            directory = args.dist_dir.resolve(strict=True)
            wheel, sdist = _distribution_pair(directory)
            verify_wheel(wheel)
            verify_sdist(sdist)
            print("distribution archives satisfy the publication allowlists")
        if args.installed_prefix is not None:
            verify_installed_distribution(args.installed_prefix)
            print("installed core distribution and missing-service boundary are verified")
        if args.service_prefix is not None:
            verify_installed_service_distribution(args.service_prefix)
            print("installed optional service distribution and ASGI contract are verified")
    except (DistributionContractError, FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(f"distribution verification failed: {exc}") from None


if __name__ == "__main__":
    main()
