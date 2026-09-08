"""Bounded verification of content-addressed evidence from a sibling repository.

Sprint 3 spans Signalattice and AlphaForge, but AlphaForge CI intentionally
does not clone or fetch Signalattice.  This module validates a committed,
non-executable receipt and can independently verify it against an already
available local Git object database.  Verification reads blobs from the pinned
commit; it never checks out a branch, consults the network, or trusts the
sibling worktree contents.

This local proof establishes that the exact bytes exist in the supplied object
database under the declared commit and that the checkout's configured origin
matches the receipt.  Without a network fetch or independently trusted remote
ref, it cannot prove that the commit is currently reachable from GitHub.  The
installed Git executable and local object database remain explicit trust
boundaries.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from alphaforge.research._bounded_io import (
    BoundedIOError,
    RegularFileSnapshot,
    bounded_diagnostic,
    parse_strict_json,
    read_regular_file_snapshot,
)

CROSS_REPOSITORY_SCHEMA_VERSION = "1.0.0"
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_EXTERNAL_SOURCES = 32
MAX_EXTERNAL_SOURCE_BYTES = 4 * 1024 * 1024
MAX_EXTERNAL_TOTAL_BYTES = 16 * 1024 * 1024
MAX_DOCUMENT_DEPTH = 64
MAX_DOCUMENT_NODES = 100_000
MAX_DIAGNOSTIC_CHARS = 4096
GIT_TIMEOUT_SECONDS = 15
GIT_STDERR_BYTES = 16_384

_RECEIPT_FIELDS = {
    "schema_version",
    "repository",
    "origin_url",
    "commit",
    "verification",
    "sources",
}
_VERIFICATION_FIELDS = {
    "mode",
    "verified_at_utc",
    "network_requests",
}
_SOURCE_FIELDS = {
    "family",
    "path",
    "git_blob_sha1",
    "bytes",
    "sha256",
    "claim_scope",
}


class CrossRepositoryProvenanceError(ValueError):
    """Raised when a receipt or pinned Git object fails validation."""


def _lower_hex(value: Any, *, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CrossRepositoryProvenanceError(
            f"{field} must be {length} lowercase hexadecimal characters"
        )
    return value


def _safe_text(value: Any, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\x00" in value
        or not value.isascii()
    ):
        raise CrossRepositoryProvenanceError(
            f"{field} must be non-empty, trimmed ASCII of at most {maximum} characters"
        )
    return value


def _safe_repository_path(value: Any) -> str:
    path_text = _safe_text(value, field="source.path", maximum=512)
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in path_text
        or ":" in path_text
        or any(
            not all(character.isalnum() or character in "._-" for character in part)
            for part in path.parts
        )
    ):
        raise CrossRepositoryProvenanceError(
            "source.path must be a safe repository-relative POSIX path"
        )
    return path_text


def _safe_repository_id(value: Any) -> str:
    repository = _safe_text(value, field="repository", maximum=256)
    parts = repository.split("/")
    if (
        len(parts) != 2
        or any(part in {"", ".", ".."} for part in parts)
        or any(
            not all(character.isalnum() or character in "._-" for character in part)
            for part in parts
        )
    ):
        raise CrossRepositoryProvenanceError(
            "repository must be a safe GitHub owner/repository identifier"
        )
    return repository


def _read_receipt_snapshot(path: str | Path) -> RegularFileSnapshot:
    try:
        return read_regular_file_snapshot(path, max_bytes=MAX_RECEIPT_BYTES)
    except BoundedIOError as exc:
        detail = str(exc)
        if "symlink" in detail:
            message = "receipt must be a regular non-symlink file"
        elif "bytes" in detail or "length" in detail:
            message = f"receipt bytes must be in [1, {MAX_RECEIPT_BYTES}]"
        else:
            message = f"unable to read receipt: {detail}"
        raise CrossRepositoryProvenanceError(
            bounded_diagnostic(
                "",
                message,
                maximum_chars=MAX_DIAGNOSTIC_CHARS,
            )
        ) from exc


@dataclass(frozen=True)
class ExternalSource:
    """One exact Git blob supporting a bounded cross-repository claim."""

    family: str
    path: str
    git_blob_sha1: str
    bytes: int
    sha256: str
    claim_scope: str

    def __post_init__(self) -> None:
        _safe_text(self.family, field="source.family", maximum=128)
        _safe_repository_path(self.path)
        _lower_hex(self.git_blob_sha1, length=40, field="source.git_blob_sha1")
        _lower_hex(self.sha256, length=64, field="source.sha256")
        _safe_text(self.claim_scope, field="source.claim_scope", maximum=1000)
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or not 0 < self.bytes <= MAX_EXTERNAL_SOURCE_BYTES
        ):
            raise CrossRepositoryProvenanceError(
                f"source.bytes must be in [1, {MAX_EXTERNAL_SOURCE_BYTES}]"
            )


@dataclass(frozen=True)
class CrossRepositoryReceipt:
    """Immutable receipt for a single repository and commit."""

    repository: str
    origin_url: str
    commit: str
    verified_at_utc: str
    sources: tuple[ExternalSource, ...]

    def __post_init__(self) -> None:
        repository = _safe_repository_id(self.repository)
        origin = _safe_text(self.origin_url, field="origin_url", maximum=512)
        if (
            not origin.startswith("https://github.com/")
            or not origin.endswith(".git")
            or "@" in origin
        ):
            raise CrossRepositoryProvenanceError(
                "origin_url must be a credential-free HTTPS GitHub repository URL"
            )
        if origin != f"https://github.com/{repository}.git":
            raise CrossRepositoryProvenanceError(
                "repository and origin_url must identify the same GitHub repository"
            )
        _lower_hex(self.commit, length=40, field="commit")
        timestamp = _safe_text(
            self.verified_at_utc,
            field="verified_at_utc",
            maximum=64,
        )
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
        except ValueError as exc:
            raise CrossRepositoryProvenanceError(
                "verified_at_utc must be a valid ISO-8601 UTC timestamp"
            ) from exc
        if not timestamp.endswith("Z") or parsed_timestamp.tzinfo != UTC:
            raise CrossRepositoryProvenanceError(
                "verified_at_utc must be an explicit UTC timestamp ending in Z"
            )
        if not 0 < len(self.sources) <= MAX_EXTERNAL_SOURCES:
            raise CrossRepositoryProvenanceError(
                f"source count must be in [1, {MAX_EXTERNAL_SOURCES}]"
            )
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise CrossRepositoryProvenanceError("source paths must be unique")
        total = sum(source.bytes for source in self.sources)
        if total > MAX_EXTERNAL_TOTAL_BYTES:
            raise CrossRepositoryProvenanceError(
                f"declared source bytes exceed {MAX_EXTERNAL_TOTAL_BYTES}"
            )


@dataclass(frozen=True)
class CrossRepositoryVerification:
    """Observed verification result without worktree-dependent state."""

    repository: str
    commit: str
    source_count: int
    total_bytes: int
    receipt_sha256: str


def _exact_fields(document: dict[str, Any], expected: set[str], context: str) -> None:
    observed = set(document)
    if observed != expected:
        raise CrossRepositoryProvenanceError(
            bounded_diagnostic(
                f"{context} fields mismatch: ",
                (
                    f"missing={sorted(expected - observed)[:16]}, "
                    f"extra={sorted(observed - expected)[:16]}"
                ),
                maximum_chars=MAX_DIAGNOSTIC_CHARS,
            )
        )


def _parse_cross_repository_receipt(
    snapshot: RegularFileSnapshot,
) -> CrossRepositoryReceipt:
    try:
        document = parse_strict_json(
            snapshot.data,
            maximum_depth=MAX_DOCUMENT_DEPTH,
            maximum_nodes=MAX_DOCUMENT_NODES,
        )
    except BoundedIOError as exc:
        raise CrossRepositoryProvenanceError(
            bounded_diagnostic(
                "receipt must be strict bounded UTF-8 JSON: ",
                exc,
                maximum_chars=MAX_DIAGNOSTIC_CHARS,
            )
        ) from exc
    if not isinstance(document, dict):
        raise CrossRepositoryProvenanceError("receipt root must be an object")
    _exact_fields(document, _RECEIPT_FIELDS, "receipt")
    if document["schema_version"] != CROSS_REPOSITORY_SCHEMA_VERSION:
        raise CrossRepositoryProvenanceError("unsupported receipt schema_version")
    verification = document["verification"]
    if not isinstance(verification, dict):
        raise CrossRepositoryProvenanceError("verification must be an object")
    _exact_fields(verification, _VERIFICATION_FIELDS, "verification")
    if verification["mode"] != "local_git_object_database":
        raise CrossRepositoryProvenanceError("unsupported verification mode")
    if (
        isinstance(verification["network_requests"], bool)
        or not isinstance(verification["network_requests"], int)
        or verification["network_requests"] != 0
    ):
        raise CrossRepositoryProvenanceError("receipt verification must be network-free")
    source_documents = document["sources"]
    if not isinstance(source_documents, list):
        raise CrossRepositoryProvenanceError("sources must be an array")
    if not 0 < len(source_documents) <= MAX_EXTERNAL_SOURCES:
        raise CrossRepositoryProvenanceError(f"source count must be in [1, {MAX_EXTERNAL_SOURCES}]")
    sources: list[ExternalSource] = []
    for index, item in enumerate(source_documents):
        if not isinstance(item, dict):
            raise CrossRepositoryProvenanceError(f"sources[{index}] must be an object")
        _exact_fields(item, _SOURCE_FIELDS, f"sources[{index}]")
        sources.append(ExternalSource(**item))
    return CrossRepositoryReceipt(
        repository=document["repository"],
        origin_url=document["origin_url"],
        commit=document["commit"],
        verified_at_utc=verification["verified_at_utc"],
        sources=tuple(sources),
    )


def load_cross_repository_receipt(path: str | Path) -> CrossRepositoryReceipt:
    """Load one strict receipt from a single bounded regular-file snapshot."""

    return _parse_cross_repository_receipt(_read_receipt_snapshot(path))


def _run_git(
    checkout: Path,
    arguments: list[str],
    *,
    maximum_stdout: int,
) -> bytes:
    if (
        isinstance(maximum_stdout, bool)
        or not isinstance(maximum_stdout, int)
        or maximum_stdout < 0
    ):
        raise CrossRepositoryProvenanceError("maximum_stdout must be a non-negative integer")
    env = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
    }

    def kill_process_tree(process: subprocess.Popen[bytes]) -> None:
        if os.name == "posix":
            with suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            with suppress(OSError):
                process.kill()

    def read_capped(
        stream: Any,
        *,
        maximum: int,
        output: bytearray,
        overflow: threading.Event,
        failures: list[OSError],
        process: subprocess.Popen[bytes],
    ) -> None:
        try:
            while chunk := stream.read(64 * 1024):
                remaining = maximum + 1 - len(output)
                if remaining > 0:
                    output.extend(chunk[:remaining])
                if len(output) > maximum or len(chunk) > remaining:
                    overflow.set()
                    kill_process_tree(process)
                    return
        except OSError as exc:
            failures.append(exc)
            kill_process_tree(process)
        finally:
            stream.close()

    try:
        process = subprocess.Popen(
            ["git", "-C", os.fspath(checkout), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise CrossRepositoryProvenanceError("bounded local Git command failed") from exc
    if process.stdout is None or process.stderr is None:
        kill_process_tree(process)
        process.wait()
        raise CrossRepositoryProvenanceError("unable to capture bounded local Git output")

    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    failures: list[OSError] = []
    stdout_reader = threading.Thread(
        target=read_capped,
        kwargs={
            "stream": process.stdout,
            "maximum": maximum_stdout,
            "output": stdout,
            "overflow": overflow,
            "failures": failures,
            "process": process,
        },
        daemon=True,
    )
    stderr_reader = threading.Thread(
        target=read_capped,
        kwargs={
            "stream": process.stderr,
            "maximum": GIT_STDERR_BYTES,
            "output": stderr,
            "overflow": overflow,
            "failures": failures,
            "process": process,
        },
        daemon=True,
    )
    stdout_reader.start()
    stderr_reader.start()
    try:
        returncode = process.wait(timeout=GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(process)
        process.wait()
        stdout_reader.join(timeout=1)
        stderr_reader.join(timeout=1)
        raise CrossRepositoryProvenanceError("bounded local Git command timed out") from exc
    stdout_reader.join()
    stderr_reader.join()
    if failures:
        raise CrossRepositoryProvenanceError(
            "unable to read bounded local Git output"
        ) from failures[0]
    if overflow.is_set():
        raise CrossRepositoryProvenanceError("local Git command exceeded output bounds")
    if returncode != 0:
        diagnostic = bytes(stderr).decode("utf-8", errors="replace").strip()[:1000]
        raise CrossRepositoryProvenanceError(
            f"local Git command rejected pinned evidence: {diagnostic}"
        )
    return bytes(stdout)


def verify_cross_repository_receipt(
    receipt_path: str | Path,
    *,
    checkout: str | Path,
) -> CrossRepositoryVerification:
    """Verify every receipt source against a pinned local Git commit.

    The current branch, index, and worktree are irrelevant. Only objects
    reachable by the exact receipt commit are read.
    """

    checkout_path = Path(checkout)
    if checkout_path.is_symlink() or not checkout_path.is_dir():
        raise CrossRepositoryProvenanceError("checkout must be a real non-symlink directory")
    receipt_file = Path(receipt_path)
    initial_receipt = _read_receipt_snapshot(receipt_file)
    receipt = _parse_cross_repository_receipt(initial_receipt)
    inside = _run_git(
        checkout_path,
        ["rev-parse", "--is-inside-work-tree"],
        maximum_stdout=64,
    )
    if inside.strip() != b"true":
        raise CrossRepositoryProvenanceError("checkout is not a Git worktree")
    origin = _run_git(
        checkout_path,
        ["config", "--get", "remote.origin.url"],
        maximum_stdout=1024,
    )
    try:
        origin_text = origin.decode("utf-8", errors="strict").strip()
    except UnicodeError as exc:
        raise CrossRepositoryProvenanceError("origin is not valid UTF-8") from exc
    if origin_text != receipt.origin_url:
        raise CrossRepositoryProvenanceError("origin mismatch")
    commit_type = _run_git(
        checkout_path,
        ["cat-file", "-t", receipt.commit],
        maximum_stdout=16,
    )
    if commit_type.strip() != b"commit":
        raise CrossRepositoryProvenanceError("receipt commit does not name an exact commit object")
    total_bytes = 0
    for source in receipt.sources:
        tree_record = _run_git(
            checkout_path,
            ["ls-tree", "-z", "--full-tree", receipt.commit, "--", source.path],
            maximum_stdout=len(source.path.encode("ascii")) + 128,
        )
        records = tree_record.split(b"\0")
        if len(records) != 2 or records[-1] or not records[0]:
            raise CrossRepositoryProvenanceError(
                f"Git tree path must resolve exactly once: {source.path}"
            )
        try:
            metadata, observed_path = records[0].split(b"\t", 1)
            mode, object_type, blob_bytes = metadata.split(b" ", 2)
            decoded_path = observed_path.decode("ascii")
            blob = blob_bytes.decode("ascii")
        except (ValueError, UnicodeError) as exc:
            raise CrossRepositoryProvenanceError(
                f"Git returned malformed tree metadata for {source.path}"
            ) from exc
        if decoded_path != source.path:
            raise CrossRepositoryProvenanceError(f"Git tree path mismatch for {source.path}")
        if mode != b"100644" or object_type != b"blob":
            raise CrossRepositoryProvenanceError(
                f"Git evidence path must be a non-executable regular blob: {source.path}"
            )
        if blob != source.git_blob_sha1:
            raise CrossRepositoryProvenanceError(
                f"Git blob mismatch for {source.path}: expected "
                f"{source.git_blob_sha1}, observed {blob}"
            )
        observed_size_text = (
            _run_git(
                checkout_path,
                ["cat-file", "-s", blob],
                maximum_stdout=64,
            )
            .decode("ascii")
            .strip()
        )
        try:
            observed_size = int(observed_size_text)
        except ValueError as exc:
            raise CrossRepositoryProvenanceError(
                f"Git returned an invalid byte count for {source.path}"
            ) from exc
        if observed_size != source.bytes:
            raise CrossRepositoryProvenanceError(
                f"byte-size mismatch for {source.path}: expected "
                f"{source.bytes}, observed {observed_size}"
            )
        content = _run_git(
            checkout_path,
            ["cat-file", "blob", blob],
            maximum_stdout=source.bytes,
        )
        if len(content) != source.bytes:
            raise CrossRepositoryProvenanceError(f"short Git blob read for {source.path}")
        observed_sha256 = hashlib.sha256(content).hexdigest()
        if observed_sha256 != source.sha256:
            raise CrossRepositoryProvenanceError(
                f"SHA-256 mismatch for {source.path}: expected "
                f"{source.sha256}, observed {observed_sha256}"
            )
        total_bytes += len(content)
    final_receipt = _read_receipt_snapshot(receipt_file)
    if final_receipt.data != initial_receipt.data:
        raise CrossRepositoryProvenanceError("receipt changed during verification")
    return CrossRepositoryVerification(
        repository=receipt.repository,
        commit=receipt.commit,
        source_count=len(receipt.sources),
        total_bytes=total_bytes,
        receipt_sha256=initial_receipt.sha256,
    )
