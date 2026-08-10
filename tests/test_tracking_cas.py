"""Adversarial tests for the descriptor-anchored local artifact store."""

from __future__ import annotations

import errno
import hashlib
import os
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest

from quant_platform.tracking import cas as cas_module
from quant_platform.tracking.cas import (
    MAX_ARTIFACT_BYTES,
    MAX_IN_MEMORY_READ_BYTES,
    ArtifactBoundaryError,
    ArtifactGeneration,
    ArtifactIntegrityError,
    ArtifactInventoryCursor,
    ArtifactInventoryPage,
    ArtifactInventoryRecord,
    ArtifactStore,
    ArtifactStoreError,
    PublishedArtifact,
    StagingInventoryCursor,
    StagingInventoryPage,
    StagingInventoryRecord,
)


class InjectedFailure(RuntimeError):
    """Test-only failure used to prove cleanup preserves primary errors."""


class _ExplosiveStr(str):
    def __eq__(self, other: object) -> bool:
        del other
        raise RuntimeError("hostile string comparison payload")


class _ExplosiveInt(int):
    def __le__(self, other: object) -> bool:
        del other
        raise RuntimeError("hostile integer comparison payload")


class _ExplosiveFloat(float):
    def __float__(self) -> float:
        raise RuntimeError("hostile float conversion payload")


class _ExplosivePathLike:
    def __fspath__(self) -> str:
        raise OSError(errno.EIO, "sensitive hostile filesystem payload")


class _ExplosiveTimezone(tzinfo):
    def utcoffset(self, value: datetime | None) -> timedelta:
        del value
        raise OSError(errno.EIO, "sensitive hostile timezone payload")

    def dst(self, value: datetime | None) -> timedelta:
        del value
        return timedelta(0)


class _ExplosiveEquality:
    def __eq__(self, other: object) -> bool:
        del other
        raise RuntimeError("hostile storage-key equality payload")


class _ExplosivePrimary(InjectedFailure):
    def add_note(self, note: str) -> None:
        del note
        raise RuntimeError("hostile diagnostic replacement payload")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Use the real macOS path rather than the /var compatibility symlink."""

    return tmp_path.resolve()


def _store(workspace: Path, *, maximum: int = 1024) -> ArtifactStore:
    store = ArtifactStore(workspace / "cas", max_artifact_bytes=maximum)
    store.initialize()
    return store


def _object_path(store: ArtifactStore, artifact: PublishedArtifact) -> Path:
    return store.root.joinpath(*artifact.storage_key.split("/"))


def _timestamp_nanoseconds(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    delta = normalized - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000 + delta.microseconds * 1_000
    )


def _future_cutoff() -> datetime:
    return datetime.now(UTC) + timedelta(days=1)


def _past_cutoff() -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC)


def _staging_path(store: ArtifactStore, token: str, payload: bytes = b"partial") -> Path:
    path = store.root / ".staging" / f"publish-{token}.tmp"
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _mkdir_private(path: Path) -> None:
    """Create each missing CAS test directory with exact mode 0700."""

    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        directory.chmod(0o700)


def test_module_imports_without_fcntl_and_store_fails_before_mutation(
    workspace: Path,
) -> None:
    """Keep non-POSIX package imports usable while the CAS fails closed."""

    root = workspace / "unsupported-cas"
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "import quant_platform.tracking",
            "sys.modules.pop('fcntl', None)",
            "class BlockFcntl:",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname == 'fcntl':",
            "            raise ModuleNotFoundError('blocked test capability', name=fullname)",
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
            "print('non-POSIX import boundary verified')",
        )
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=workspace,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "non-POSIX import boundary verified\n"
    assert not root.exists()


def test_store_rejects_missing_dirfd_capability_before_mutation(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = workspace / "unsupported-dirfd-cas"
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd.difference({os.open}))

    with pytest.raises(
        ArtifactBoundaryError,
        match="requires directory-descriptor filesystem operations",
    ):
        ArtifactStore(root)

    assert not root.exists()


@pytest.mark.parametrize("payload", [b"", b"abc", b"\x00\xff" * 513])
def test_publish_verify_read_and_unlink_round_trip(
    workspace: Path,
    payload: bytes,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(payload)
    store = _store(workspace, maximum=max(1, len(payload)))

    artifact = store.publish(source)

    digest = hashlib.sha256(payload).hexdigest()
    assert artifact == PublishedArtifact(
        digest=digest,
        byte_size=len(payload),
        storage_key=f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
    )
    assert store.read_verified(artifact, max_bytes=max(1, len(payload))) == payload
    store.verify(artifact)
    assert stat_mode(_object_path(store, artifact)) & 0o222 == 0

    store.unlink_verified(artifact)

    assert not _object_path(store, artifact).exists()


def test_inspect_returns_verified_path_free_generation(workspace: Path) -> None:
    source = workspace / "inspect-source.bin"
    source.write_bytes(b"inspectable")
    store = _store(workspace)
    artifact = store.publish(source)

    record = store.inspect(artifact)

    assert record.digest == artifact.digest
    assert record.byte_size == artifact.byte_size
    assert record.storage_key == artifact.storage_key
    assert record.last_changed_at.tzinfo is UTC
    assert not Path(record.storage_key).is_absolute()


def test_verify_absent_rejects_missing_parent_and_accepts_deleted_object(
    workspace: Path,
) -> None:
    store = _store(workspace)
    absent_digest = hashlib.sha256(b"never-published").hexdigest()
    absent = PublishedArtifact(
        digest=absent_digest,
        byte_size=0,
        storage_key=f"objects/{absent_digest[:2]}/{absent_digest[2:4]}/{absent_digest}",
    )
    with pytest.raises(ArtifactBoundaryError, match="fanout"):
        store.verify_absent(absent)

    source = workspace / "deleted-source.bin"
    source.write_bytes(b"deleted")
    published = store.publish(source)
    store.unlink_verified(published)

    store.verify_absent(published)


def test_verify_absent_rejects_fanout_restore_race_with_present_object(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "absence-race-source.bin"
    source.write_bytes(b"present-through-restore")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    parent = destination.parent
    displaced = workspace / "absence-race-fanout"
    original_open_parent = store._open_destination_parent

    def restore_before_error(*args: Any, **kwargs: Any) -> int:
        parent.rename(displaced)
        try:
            return original_open_parent(*args, **kwargs)
        except ArtifactBoundaryError:
            displaced.rename(parent)
            raise

    monkeypatch.setattr(store, "_open_destination_parent", restore_before_error)

    with pytest.raises(ArtifactBoundaryError, match="fanout"):
        store.verify_absent(artifact)

    assert destination.exists()


def test_verify_absent_rejects_first_level_ancestor_swap_after_root_rebind(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"present-in-replacement-fanout"
    source = workspace / "ancestor-swap-source.bin"
    source.write_bytes(payload)
    store = _store(workspace)
    artifact = store.publish(source)
    store.unlink_verified(artifact)

    destination = _object_path(store, artifact)
    first_level = store.root / "objects" / artifact.digest[:2]
    displaced = workspace / "displaced-first-level-fanout"
    replacement = workspace / "replacement-first-level-fanout"
    replacement.mkdir(mode=0o700)
    replacement.chmod(0o700)
    replacement_second = replacement / artifact.digest[2:4]
    replacement_second.mkdir(mode=0o700)
    replacement_second.chmod(0o700)
    replacement_object = replacement_second / artifact.digest
    replacement_object.write_bytes(payload)
    replacement_object.chmod(0o400)

    original_assert_bindings = store._assert_initialized_bindings
    swapped = False

    def swap_after_root_rebind(root_descriptor: int) -> None:
        nonlocal swapped
        original_assert_bindings(root_descriptor)
        if not swapped:
            first_level.rename(displaced)
            replacement.rename(first_level)
            swapped = True

    monkeypatch.setattr(store, "_assert_initialized_bindings", swap_after_root_rebind)

    with pytest.raises(ArtifactIntegrityError, match="fanout directory binding changed"):
        store.verify_absent(artifact)

    assert destination.is_file()


@pytest.mark.parametrize("restored_kind", ["regular", "symlink", "fifo"])
def test_verify_absent_rejects_restored_names_of_every_type(
    workspace: Path,
    restored_kind: str,
) -> None:
    if restored_kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation requires POSIX")
    source = workspace / "restore-source.bin"
    source.write_bytes(b"restore")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    store.unlink_verified(artifact)
    if restored_kind == "regular":
        destination.write_bytes(b"replacement")
        destination.chmod(0o400)
    elif restored_kind == "symlink":
        outside = workspace / "outside-restored.bin"
        outside.write_bytes(b"outside")
        destination.symlink_to(outside)
    else:
        os.mkfifo(destination)

    with pytest.raises(ArtifactIntegrityError, match="is present"):
        store.verify_absent(artifact)


def test_unlink_generation_guard_rejects_fresh_republication(workspace: Path) -> None:
    source = workspace / "generation-source.bin"
    source.write_bytes(b"same-content")
    store = _store(workspace)
    artifact = store.publish(source)
    stale_generation = store.inspect(artifact).generation
    store.unlink_verified(
        artifact,
        expected_generation=stale_generation,
    )

    republished = store.publish(source)
    assert republished == artifact
    destination = _object_path(store, republished)
    future_ns = int((datetime.now(UTC) + timedelta(days=1)).timestamp() * 1_000_000_000)
    os.utime(destination, ns=(future_ns, future_ns))
    fresh_generation = store.inspect(republished).generation
    assert fresh_generation != stale_generation

    with pytest.raises(ArtifactIntegrityError, match="generation differs"):
        store.unlink_verified(
            republished,
            expected_generation=stale_generation,
        )
    assert destination.exists()

    store.unlink_verified(
        republished,
        expected_generation=fresh_generation,
    )
    store.verify_absent(republished)


def test_unlink_generation_guard_distinguishes_submicrosecond_replacement(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "submicro-generation-source.bin"
    source.write_bytes(b"same-content")
    store = _store(workspace)
    artifact = store.publish(source)
    controlled_ns = 1_000_001
    original_generation = cas_module._artifact_generation

    def exact_generation(
        store_id: str,
        digest: str,
        identity: cas_module._Identity,
    ) -> ArtifactGeneration:
        controlled = replace(
            identity,
            device=1,
            inode=1,
            mtime_ns=controlled_ns,
            ctime_ns=controlled_ns,
        )
        return original_generation(store_id, digest, controlled)

    monkeypatch.setattr(cas_module, "_artifact_generation", exact_generation)
    monkeypatch.setattr(
        cas_module,
        "_last_changed_nanoseconds",
        lambda _status: controlled_ns,
    )
    stale = store.inspect(artifact)
    store.unlink_verified(artifact, expected_generation=stale.generation)

    republished = store.publish(source)
    controlled_ns = 1_000_999
    fresh = store.inspect(republished)
    assert fresh.last_changed_at == stale.last_changed_at
    assert fresh.last_changed_ns != stale.last_changed_ns
    assert fresh.generation != stale.generation

    with pytest.raises(ArtifactIntegrityError, match="generation differs"):
        store.unlink_verified(republished, expected_generation=stale.generation)
    assert _object_path(store, republished).exists()

    store.unlink_verified(republished, expected_generation=fresh.generation)
    store.verify_absent(republished)


def test_unlink_rebinds_parent_inside_destructive_primitive(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "containment-source.bin"
    source.write_bytes(b"containment")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    parent = destination.parent
    detached = workspace / "detached-fanout"
    original_unlink = store._unlink_name_if_same
    mutated = False

    def detach_before_unlink(*args: Any, **kwargs: Any) -> None:
        nonlocal mutated
        if not mutated:
            parent.rename(detached)
            mutated = True
        original_unlink(*args, **kwargs)

    monkeypatch.setattr(store, "_unlink_name_if_same", detach_before_unlink)

    with pytest.raises(ArtifactStoreError, match="fanout"):
        store.unlink_verified(artifact)

    assert (detached / artifact.digest).exists()


def stat_mode(path: Path) -> int:
    """Return permission bits without following a symlink."""

    return os.lstat(path).st_mode


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"digest": "A" * 64, "byte_size": 1, "storage_key": "x"}, "lowercase"),
        (
            {
                "digest": "0" * 64,
                "byte_size": -1,
                "storage_key": "objects/00/00/" + "0" * 64,
            },
            "non-negative",
        ),
        (
            {
                "digest": "0" * 64,
                "byte_size": True,
                "storage_key": "objects/00/00/" + "0" * 64,
            },
            "integer",
        ),
        (
            {"digest": "0" * 64, "byte_size": 1, "storage_key": "objects/00/00/other"},
            "storage key",
        ),
    ],
)
def test_published_artifact_rejects_noncanonical_metadata(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ArtifactBoundaryError, match=message):
        PublishedArtifact(**kwargs)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5, "10"])
def test_store_rejects_invalid_resource_bound(workspace: Path, maximum: Any) -> None:
    with pytest.raises(ArtifactBoundaryError, match="positive integer"):
        ArtifactStore(workspace / "cas", max_artifact_bytes=maximum)


def test_artifact_resource_ceilings_accept_boundary_and_reject_one_over(
    workspace: Path,
) -> None:
    digest = hashlib.sha256(b"").hexdigest()
    storage_key = f"objects/{digest[:2]}/{digest[2:4]}/{digest}"

    PublishedArtifact(
        digest=digest,
        byte_size=MAX_ARTIFACT_BYTES,
        storage_key=storage_key,
    )
    ArtifactStore(workspace / "boundary-cas", max_artifact_bytes=MAX_ARTIFACT_BYTES)

    with pytest.raises(ArtifactBoundaryError, match="cannot exceed"):
        PublishedArtifact(
            digest=digest,
            byte_size=MAX_ARTIFACT_BYTES + 1,
            storage_key=storage_key,
        )
    with pytest.raises(ArtifactBoundaryError, match="cannot exceed"):
        ArtifactStore(
            workspace / "oversized-cas",
            max_artifact_bytes=MAX_ARTIFACT_BYTES + 1,
        )


def test_public_cas_contracts_reject_hostile_scalar_subclasses_before_use(
    workspace: Path,
) -> None:
    digest = hashlib.sha256(b"hostile-contract").hexdigest()
    storage_key = f"objects/{digest[:2]}/{digest[2:4]}/{digest}"

    with pytest.raises(ArtifactBoundaryError, match="digest"):
        PublishedArtifact(_ExplosiveStr(digest), 1, storage_key)
    with pytest.raises(ArtifactBoundaryError, match="storage key must be exact text"):
        PublishedArtifact(digest, 1, _ExplosiveEquality())  # type: ignore[arg-type]
    with pytest.raises(ArtifactBoundaryError, match="positive integer"):
        ArtifactStore(workspace / "hostile-integer", max_artifact_bytes=_ExplosiveInt(1))
    with pytest.raises(ArtifactBoundaryError, match="store identity"):
        ArtifactStore(
            workspace / "hostile-store-id",
            expected_store_id=_ExplosiveStr("a" * 64),
        )
    with pytest.raises(ArtifactBoundaryError, match="staging identifier"):
        StagingInventoryRecord(
            _ExplosiveStr(f"publish-{'a' * 48}.tmp"),
            1,
            datetime.now(UTC),
        )

    store = _store(workspace)
    with pytest.raises(ArtifactBoundaryError, match="positive integer"):
        store.inventory_objects(cutoff=_future_cutoff(), page_size=_ExplosiveInt(1))
    with pytest.raises(ArtifactBoundaryError, match="finite and positive"):
        store.inventory_objects(
            cutoff=_future_cutoff(),
            time_budget_seconds=_ExplosiveFloat(1.0),
        )


def test_cas_path_and_timezone_boundaries_map_hostile_protocol_failures(
    workspace: Path,
) -> None:
    with pytest.raises(ArtifactBoundaryError, match="path is invalid") as path_error:
        ArtifactStore(_ExplosivePathLike())  # type: ignore[arg-type]
    assert path_error.value.__cause__ is None
    assert "sensitive" not in str(path_error.value)

    oversized_component = workspace / ("x" * (cas_module.MAX_CAS_PATH_COMPONENT_BYTES + 1))
    with pytest.raises(ArtifactBoundaryError, match="portability bounds"):
        ArtifactStore(oversized_component)
    bounded_components = ["x" * 250] * 17
    oversized_total = Path("/").joinpath(*bounded_components)
    assert len(os.fsencode(str(oversized_total))) > cas_module.MAX_CAS_PATH_BYTES
    with pytest.raises(ArtifactBoundaryError, match="portability bounds"):
        ArtifactStore(oversized_total)

    hostile_time = datetime(2026, 8, 8, 12, tzinfo=_ExplosiveTimezone())
    store = _store(workspace)
    with pytest.raises(ArtifactBoundaryError, match="valid UTC datetime") as time_error:
        store.inventory_objects(cutoff=hostile_time)
    assert time_error.value.__cause__ is None
    assert "sensitive" not in str(time_error.value)


def test_descriptor_cleanup_attempts_all_closes_and_preserves_hostile_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors = cas_module._StoreDescriptors(root=101, staging=102, objects=103)
    calls: list[int] = []

    def fail_every_close(descriptor: int) -> None:
        calls.append(descriptor)
        raise OSError(errno.EIO, "sensitive descriptor payload")

    monkeypatch.setattr(cas_module.os, "close", fail_every_close)
    with pytest.raises(ArtifactStoreError, match="descriptor cleanup failed") as cleanup_error:
        descriptors.close()
    assert calls == [103, 102, 101]
    assert cleanup_error.value.__cause__ is None
    assert "sensitive" not in str(cleanup_error.value)

    calls.clear()
    primary = _ExplosivePrimary("authoritative primary failure")
    with pytest.raises(_ExplosivePrimary) as captured:
        try:
            raise primary
        finally:
            descriptors.close()
    assert captured.value is primary
    assert calls == [103, 102, 101]


def test_descriptor_cleanup_never_retries_an_interrupted_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def interrupt_first_close(descriptor: int) -> None:
        calls.append(descriptor)
        if descriptor == 103:
            raise InterruptedError(errno.EINTR, "interrupted descriptor close")

    monkeypatch.setattr(cas_module.os, "close", interrupt_first_close)
    with pytest.raises(ArtifactStoreError, match="descriptor cleanup failed"):
        cas_module._StoreDescriptors(root=101, staging=102, objects=103).close()
    assert calls == [103, 102, 101]


def test_read_does_not_initialize_storage(workspace: Path) -> None:
    root = workspace / "missing-cas"
    store = ArtifactStore(root)
    empty_digest = hashlib.sha256(b"").hexdigest()
    artifact = PublishedArtifact(
        digest=empty_digest,
        byte_size=0,
        storage_key=f"objects/{empty_digest[:2]}/{empty_digest[2:4]}/{empty_digest}",
    )

    with pytest.raises(ArtifactBoundaryError, match="not initialized"):
        store.verify(artifact)

    assert not root.exists()


def test_root_and_source_traversal_are_rejected_before_access(workspace: Path) -> None:
    root_with_traversal = f"{workspace}/unused/../cas"
    with pytest.raises(ArtifactBoundaryError, match="traversal"):
        ArtifactStore(root_with_traversal)
    assert not (workspace / "cas").exists()

    source = workspace / "source.bin"
    source.write_bytes(b"bounded")
    store = _store(workspace)
    source_with_traversal = f"{workspace}/unused/../source.bin"
    with pytest.raises(ArtifactBoundaryError, match="traversal"):
        store.publish(source_with_traversal)
    assert list((store.root / ".staging").iterdir()) == []


def test_root_symlink_and_symlink_component_are_rejected(workspace: Path) -> None:
    target = workspace / "target"
    target.mkdir()
    root_link = workspace / "root-link"
    root_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ArtifactBoundaryError, match="cannot be opened safely"):
        ArtifactStore(root_link).initialize()

    parent_link = workspace / "parent-link"
    parent_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ArtifactBoundaryError, match="cannot be opened safely"):
        ArtifactStore(parent_link / "cas").initialize()
    assert not (target / "cas").exists()


def test_source_symlink_and_symlink_component_are_rejected(workspace: Path) -> None:
    real_parent = workspace / "real"
    real_parent.mkdir()
    source = real_parent / "source.bin"
    source.write_bytes(b"payload")
    source_link = workspace / "source-link"
    source_link.symlink_to(source)
    parent_link = workspace / "source-parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    store = _store(workspace)

    with pytest.raises(ArtifactBoundaryError):
        store.publish(source_link)
    with pytest.raises(ArtifactBoundaryError):
        store.publish(parent_link / source.name)

    assert list((store.root / ".staging").iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation requires POSIX")
def test_special_file_is_rejected_without_blocking(workspace: Path) -> None:
    fifo = workspace / "artifact.fifo"
    os.mkfifo(fifo)
    store = _store(workspace)

    with pytest.raises(ArtifactBoundaryError, match="regular file"):
        store.publish(fifo)
    with pytest.raises(ArtifactBoundaryError, match="regular file"):
        store.publish(workspace)


def test_oversized_source_fails_without_staging_or_object(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"12345")
    store = _store(workspace, maximum=4)

    with pytest.raises(ArtifactBoundaryError, match="exceeds 4 bytes"):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []
    assert list((store.root / "objects").iterdir()) == []


def test_source_content_mutation_during_copy_fails_and_cleans_staging(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"original")
    store = _store(workspace)
    original_read = store._read_chunk
    mutated = False

    def mutate_after_first_read(descriptor: int, maximum: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, maximum)
        if chunk and not mutated:
            mutated = True
            source.write_bytes(b"modified")
        return chunk

    monkeypatch.setattr(store, "_read_chunk", mutate_after_first_read)

    with pytest.raises(ArtifactIntegrityError, match="source changed"):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []
    assert list((store.root / "objects").iterdir()) == []


def test_source_path_replacement_during_copy_is_detected(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    displaced = workspace / "displaced.bin"
    source.write_bytes(b"original")
    store = _store(workspace)
    original_read = store._read_chunk
    replaced = False

    def replace_after_first_read(descriptor: int, maximum: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, maximum)
        if chunk and not replaced:
            replaced = True
            source.rename(displaced)
            source.write_bytes(b"original")
        return chunk

    monkeypatch.setattr(store, "_read_chunk", replace_after_first_read)

    with pytest.raises(ArtifactIntegrityError, match="source changed"):
        store.publish(source)

    assert source.read_bytes() == b"original"
    assert list((store.root / ".staging").iterdir()) == []


def test_copy_failure_cleans_descriptor_identified_staging(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)

    def fail_after_partial_write(
        source_descriptor: int,
        staging_descriptor: int,
        *,
        expected_size: int,
    ) -> tuple[str, int]:
        del source_descriptor, expected_size
        os.write(staging_descriptor, b"partial")
        raise InjectedFailure("copy failed")

    monkeypatch.setattr(store, "_copy_and_hash", fail_after_partial_write)

    with pytest.raises(InjectedFailure, match="copy failed"):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []


def test_cleanup_preserves_replacement_and_does_not_mask_primary_error(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)
    staging = store.root / ".staging"

    def replace_staging_name(
        source_descriptor: int,
        staging_descriptor: int,
        *,
        expected_size: int,
    ) -> tuple[str, int]:
        del source_descriptor, staging_descriptor, expected_size
        original = next(staging.iterdir())
        original.rename(staging / "displaced.tmp")
        original.write_bytes(b"replacement")
        raise InjectedFailure("primary failure")

    monkeypatch.setattr(store, "_copy_and_hash", replace_staging_name)

    with pytest.raises(InjectedFailure, match="primary failure") as captured:
        store.publish(source)

    assert (staging / "displaced.tmp").exists()
    replacement = next(path for path in staging.iterdir() if path.name != "displaced.tmp")
    assert replacement.read_bytes() == b"replacement"
    assert any("cleanup also failed closed" in note for note in captured.value.__notes__)


def test_existing_identical_object_is_verified_and_reused(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"same bytes")
    store = _store(workspace)

    first = store.publish(source)
    first_identity = os.lstat(_object_path(store, first))
    second = store.publish(source)
    second_identity = os.lstat(_object_path(store, second))

    assert first == second
    assert (first_identity.st_dev, first_identity.st_ino) == (
        second_identity.st_dev,
        second_identity.st_ino,
    )
    assert second_identity.st_nlink == 1
    assert list((store.root / ".staging").iterdir()) == []


def test_verify_read_and_republish_reject_external_hard_link(workspace: Path) -> None:
    source = workspace / "hard-link-source.bin"
    source.write_bytes(b"contained bytes")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    os.link(destination, workspace / "external-object-link")

    with pytest.raises(ArtifactIntegrityError, match="hard-link count"):
        store.verify(artifact)
    with pytest.raises(ArtifactIntegrityError, match="hard-link count"):
        store.read_verified(artifact, max_bytes=len(source.read_bytes()))
    with pytest.raises(ArtifactIntegrityError, match="hard-link count"):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []


def test_verified_read_ceiling_accepts_boundary_and_rejects_one_over(
    workspace: Path,
) -> None:
    source = workspace / "read-boundary.bin"
    source.write_bytes(b"bounded")
    store = _store(workspace)
    artifact = store.publish(source)

    assert store.read_verified(artifact, max_bytes=MAX_IN_MEMORY_READ_BYTES) == b"bounded"
    with pytest.raises(ArtifactBoundaryError, match="cannot exceed"):
        store.read_verified(artifact, max_bytes=MAX_IN_MEMORY_READ_BYTES + 1)


def test_existing_corrupt_object_is_never_replaced(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"trusted")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    destination.chmod(0o600)
    destination.write_bytes(b"corrupt")
    destination.chmod(0o400)

    with pytest.raises(ArtifactIntegrityError, match="digest differs"):
        store.publish(source)

    assert destination.read_bytes() == b"corrupt"
    assert list((store.root / ".staging").iterdir()) == []


def test_existing_writable_object_fails_closed(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"trusted")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    destination.chmod(0o600)

    with pytest.raises(ArtifactIntegrityError, match="not stored read-only"):
        store.verify(artifact)
    with pytest.raises(ArtifactIntegrityError, match="not stored read-only"):
        store.publish(source)


def test_fanout_symlink_cannot_redirect_publication(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store = _store(workspace)
    outside = workspace / "outside"
    outside.mkdir()
    (store.root / "objects" / digest[:2]).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactBoundaryError):
        store.publish(source)

    assert list(outside.iterdir()) == []


def test_destination_symlink_is_not_followed_or_replaced(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store = _store(workspace)
    parent = store.root / "objects" / digest[:2] / digest[2:4]
    _mkdir_private(parent)
    outside = workspace / "outside.bin"
    outside.write_bytes(b"outside")
    (parent / digest).symlink_to(outside)

    with pytest.raises(ArtifactBoundaryError):
        store.publish(source)

    assert outside.read_bytes() == b"outside"
    assert (parent / digest).is_symlink()


def test_fanout_replacement_during_publication_fails_closed(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    store = _store(workspace)
    original_verify = store._verify_object
    replaced = False

    def replace_after_verify(*args: Any, **kwargs: Any) -> None:
        nonlocal replaced
        original_verify(*args, **kwargs)
        if not replaced:
            replaced = True
            fanout = store.root / "objects" / digest[:2]
            fanout.rename(store.root / "objects" / "displaced-fanout")
            fanout.mkdir()

    monkeypatch.setattr(store, "_verify_object", replace_after_verify)

    with pytest.raises(ArtifactBoundaryError):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []


def test_concurrent_identical_publication_has_one_object_without_sleep(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"concurrent payload")
    store = _store(workspace)
    rendezvous = threading.Barrier(2)
    original_acquire = ArtifactStore._acquire_publication_lock

    def synchronized_acquire(root_descriptor: int) -> None:
        rendezvous.wait(timeout=5)
        original_acquire(root_descriptor)

    monkeypatch.setattr(
        ArtifactStore,
        "_acquire_publication_lock",
        staticmethod(synchronized_acquire),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.publish, source) for _ in range(2)]
        artifacts = [future.result(timeout=10) for future in futures]

    assert artifacts[0] == artifacts[1]
    destination = _object_path(store, artifacts[0])
    assert destination.read_bytes() == b"concurrent payload"
    assert os.lstat(destination).st_nlink == 1
    assert list((store.root / ".staging").iterdir()) == []


def test_verify_waits_out_transient_publication_link_without_accepting_two_links(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"publish while verify waits"
    source = workspace / "publish-read-source.bin"
    source.write_bytes(payload)
    store = _store(workspace)
    digest = hashlib.sha256(payload).hexdigest()
    expected = PublishedArtifact(
        digest=digest,
        byte_size=len(payload),
        storage_key=f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
    )
    linked = threading.Event()
    release_publisher = threading.Event()
    reader_entered = threading.Event()
    original_link = ArtifactStore._link_staged
    original_acquire = ArtifactStore._acquire_verification_lock

    def pause_with_transient_link(*args: Any, **kwargs: Any) -> None:
        original_link(*args, **kwargs)
        linked.set()
        if not release_publisher.wait(timeout=5):
            raise AssertionError("publisher release barrier timed out")

    def signal_reader(root_descriptor: int) -> None:
        reader_entered.set()
        original_acquire(root_descriptor)

    monkeypatch.setattr(ArtifactStore, "_link_staged", staticmethod(pause_with_transient_link))
    monkeypatch.setattr(
        ArtifactStore,
        "_acquire_verification_lock",
        staticmethod(signal_reader),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        publisher = executor.submit(store.publish, source)
        assert linked.wait(timeout=5)
        verifier = executor.submit(store.verify, expected)
        assert reader_entered.wait(timeout=5)
        release_publisher.set()
        assert publisher.result(timeout=10) == expected
        verifier.result(timeout=10)

    assert os.lstat(_object_path(store, expected)).st_nlink == 1


def test_publication_lock_contention_is_bounded_after_staging_preparation(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "contended-source.bin"
    source.write_bytes(b"must not be read")
    store = _store(workspace)
    deadline_ns = int(cas_module.PUBLICATION_LOCK_TIMEOUT_SECONDS * 1_000_000_000)
    times = iter((0, deadline_ns))
    prepared = False
    original_copy = store._copy_and_hash

    def record_preparation(*args: Any, **kwargs: Any) -> tuple[str, int]:
        nonlocal prepared
        result = original_copy(*args, **kwargs)
        prepared = True
        return result

    def remain_contended(_descriptor: int, _operation: int) -> None:
        assert prepared
        raise OSError(errno.EWOULDBLOCK, "sensitive contention detail")

    monkeypatch.setattr(store, "_copy_and_hash", record_preparation)
    monkeypatch.setattr(cas_module.fcntl, "flock", remain_contended)
    monkeypatch.setattr(cas_module.time, "monotonic_ns", lambda: next(times))
    monkeypatch.setattr(cas_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(ArtifactBoundaryError, match="bounded contention deadline") as caught:
        store.publish(source)

    assert "sensitive" not in str(caught.value)
    assert list((store.root / ".staging").iterdir()) == []
    assert not any(path.is_file() for path in (store.root / "objects").rglob("*"))


def test_precommit_file_fsync_failure_publishes_nothing(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)

    def fail_sync(descriptor: int) -> None:
        del descriptor
        raise ArtifactStoreError("injected file sync failure")

    monkeypatch.setattr(store, "_fsync_file_descriptor", fail_sync)

    with pytest.raises(ArtifactStoreError, match="injected file sync failure"):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []
    assert list((store.root / "objects").iterdir()) == []


def test_atomic_link_failure_cleans_staging_and_publishes_nothing(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)

    def fail_link(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError(errno.EIO, "injected")

    monkeypatch.setattr(store, "_link_staged", fail_link)

    with pytest.raises(ArtifactStoreError, match="atomically publish"):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []


def test_postcommit_directory_fsync_failure_preserves_verifiable_object(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)
    original_sync = store._fsync_directory_descriptor
    calls = 0

    def fail_destination_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ArtifactStoreError("injected postcommit sync failure")
        original_sync(descriptor)

    monkeypatch.setattr(store, "_fsync_directory_descriptor", fail_destination_sync)

    with pytest.raises(ArtifactStoreError, match="postcommit sync failure"):
        store.publish(source)

    monkeypatch.setattr(store, "_fsync_directory_descriptor", original_sync)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = PublishedArtifact(
        digest=digest,
        byte_size=len(source.read_bytes()),
        storage_key=f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
    )
    store.verify(artifact)
    assert list((store.root / ".staging").iterdir()) == []


def test_verified_read_detects_concurrent_object_mutation(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"original")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    original_read = store._read_chunk
    mutated = False

    def mutate_after_read(descriptor: int, maximum: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, maximum)
        if chunk and not mutated:
            mutated = True
            destination.chmod(0o600)
            destination.write_bytes(b"modified")
            destination.chmod(0o400)
        return chunk

    monkeypatch.setattr(store, "_read_chunk", mutate_after_read)

    with pytest.raises(ArtifactIntegrityError, match="changed during verified read"):
        store.read_verified(artifact)


def test_verified_read_rejects_bound_before_opening_object(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)
    artifact = store.publish(source)

    with pytest.raises(ArtifactBoundaryError, match="caller limit of 3 bytes"):
        store.read_verified(artifact, max_bytes=3)
    with pytest.raises(ArtifactBoundaryError, match="positive integer"):
        store.read_verified(artifact, max_bytes=True)


def test_all_existing_object_operations_enforce_store_bound(workspace: Path) -> None:
    store = _store(workspace, maximum=4)
    digest = hashlib.sha256(b"12345").hexdigest()
    artifact = PublishedArtifact(
        digest=digest,
        byte_size=5,
        storage_key=f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
    )

    with pytest.raises(ArtifactBoundaryError, match="store limit of 4 bytes"):
        store.verify(artifact)
    with pytest.raises(ArtifactBoundaryError, match="store limit of 4 bytes"):
        store.read_verified(artifact, max_bytes=10)
    with pytest.raises(ArtifactBoundaryError, match="store limit of 4 bytes"):
        store.unlink_verified(artifact)


def test_unlink_detects_replacement_and_preserves_it(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"original")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    original_unlink = store._unlink_name_if_same

    def replace_before_unlink(
        parent_descriptor: int,
        name: str,
        expected: Any,
        *,
        missing_ok: bool,
        role: str,
        **kwargs: Any,
    ) -> None:
        destination.unlink()
        destination.write_bytes(b"replacement")
        destination.chmod(0o400)
        original_unlink(
            parent_descriptor,
            name,
            expected,
            missing_ok=missing_ok,
            role=role,
            **kwargs,
        )

    monkeypatch.setattr(store, "_unlink_name_if_same", replace_before_unlink)

    with pytest.raises(ArtifactIntegrityError, match="replacement preserved"):
        store.unlink_verified(artifact)

    assert destination.read_bytes() == b"replacement"


def test_unlink_rejects_unexpected_external_hard_link(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    extra_link = workspace / "external-hard-link"
    os.link(destination, extra_link)

    with pytest.raises(ArtifactIntegrityError, match="hard-link count"):
        store.unlink_verified(artifact)

    assert destination.exists()
    assert extra_link.exists()


def test_replaced_root_identity_fails_closed(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)
    artifact = store.publish(source)
    displaced = workspace / "displaced-cas"
    store.root.rename(displaced)
    store.root.mkdir()
    (store.root / ".staging").mkdir()
    (store.root / "objects").mkdir()

    with pytest.raises(ArtifactIntegrityError, match="root identity changed"):
        store.verify(artifact)
    with pytest.raises(ArtifactIntegrityError, match="root identity changed"):
        store.initialize()


def test_repeated_initialize_validates_without_rebinding(workspace: Path) -> None:
    store = _store(workspace)
    store_id = store.store_id

    store.initialize()

    assert (store.root / ".staging").is_dir()
    assert (store.root / "objects").is_dir()
    assert store.store_id == store_id


def test_store_identity_is_stable_across_reopen_and_unique_per_root(
    workspace: Path,
) -> None:
    first = _store(workspace)
    first_id = first.store_id
    assert len(first_id) == 64
    assert set(first_id) <= set("0123456789abcdef")

    reopened = ArtifactStore(first.root, max_artifact_bytes=1024)
    reopened.initialize()
    second = ArtifactStore(workspace / "other-cas", max_artifact_bytes=1024)
    second.initialize()

    assert reopened.store_id == first_id
    assert second.store_id != first_id
    assert not Path(first_id).is_absolute()


def test_expected_store_identity_detects_valid_marker_substitution(
    workspace: Path,
) -> None:
    first = _store(workspace)
    expected = first.store_id
    verified_reopen = ArtifactStore(first.root, expected_store_id=expected)
    verified_reopen.initialize()
    assert verified_reopen.store_id == expected

    other = ArtifactStore(workspace / "substitution-source-cas")
    other.initialize()
    marker = first.root / ".store-id"
    marker.unlink()
    marker.write_bytes((other.root / ".store-id").read_bytes())
    marker.chmod(0o400)

    with pytest.raises(ArtifactIntegrityError, match="external authority"):
        ArtifactStore(
            first.root,
            expected_store_id=expected,
        ).initialize()


def test_expected_store_identity_rejects_missing_root_without_recreation(
    workspace: Path,
) -> None:
    source = workspace / "missing-root-source.bin"
    source.write_bytes(b"must-not-be-rebound-to-an-empty-store")
    original = _store(workspace)
    artifact = original.publish(source)
    expected = original.store_id
    displaced = workspace / "displaced-complete-cas"
    original.root.rename(displaced)

    with pytest.raises(ArtifactIntegrityError, match="expected CAS root is unavailable"):
        ArtifactStore(
            original.root,
            expected_store_id=expected,
        ).initialize()

    assert not original.root.exists()
    assert displaced.joinpath(*artifact.storage_key.split("/")).read_bytes() == source.read_bytes()


def test_expected_store_identity_must_be_canonical(workspace: Path) -> None:
    with pytest.raises(ArtifactBoundaryError, match="canonical lowercase"):
        ArtifactStore(workspace / "cas", expected_store_id="not-a-store-id")


@pytest.mark.parametrize(
    "existing_components",
    [(), (".staging",), ("objects",), (".staging", "objects")],
)
def test_existing_partial_root_is_never_repaired_or_adopted(
    workspace: Path,
    existing_components: tuple[str, ...],
) -> None:
    root = workspace / "partial-cas"
    root.mkdir(mode=0o700)
    for name in existing_components:
        component = root / name
        component.mkdir(mode=0o700)
        component.chmod(0o700)
    before = tuple(sorted(path.name for path in root.iterdir()))

    with pytest.raises(ArtifactIntegrityError, match="incomplete"):
        ArtifactStore(root).initialize()

    assert tuple(sorted(path.name for path in root.iterdir())) == before
    assert not (root / ".store-id").exists()


def test_initialization_crash_leaves_partial_root_that_cannot_be_repaired(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = workspace / "crashed-cas"
    store = ArtifactStore(root)

    def fail_marker(*args: Any, **kwargs: Any) -> None:
        raise InjectedFailure("marker publication interrupted")

    monkeypatch.setattr(store, "_create_store_identity_marker", fail_marker)
    with pytest.raises(InjectedFailure, match="interrupted"):
        store.initialize()

    assert root.is_dir()
    assert (root / ".staging").is_dir()
    assert (root / "objects").is_dir()
    assert not (root / ".store-id").exists()
    with pytest.raises(ArtifactIntegrityError, match="incomplete"):
        ArtifactStore(root).initialize()
    assert not (root / ".store-id").exists()


@pytest.mark.parametrize("missing_component", [".staging", "objects"])
def test_existing_initialized_root_never_recreates_missing_component(
    workspace: Path,
    missing_component: str,
) -> None:
    source = workspace / "preserved-source.bin"
    source.write_bytes(b"preserved")
    store = _store(workspace)
    artifact = store.publish(source)
    component = store.root / missing_component
    displaced = workspace / f"displaced-{missing_component.lstrip('.')}"
    component.rename(displaced)

    with pytest.raises(ArtifactIntegrityError, match="incomplete"):
        ArtifactStore(
            store.root,
            expected_store_id=store.store_id,
        ).initialize()

    assert not component.exists()
    assert displaced.exists()
    if missing_component == "objects":
        assert displaced.joinpath(*artifact.storage_key.split("/")[1:]).exists()


def test_populated_existing_root_without_marker_fails_without_adoption(
    workspace: Path,
) -> None:
    source = workspace / "populated-source.bin"
    source.write_bytes(b"must-survive")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    (store.root / ".store-id").unlink()

    with pytest.raises(ArtifactIntegrityError, match="incomplete"):
        ArtifactStore(store.root).initialize()

    assert destination.read_bytes() == b"must-survive"
    assert not (store.root / ".store-id").exists()


def test_store_identity_marker_tamper_and_substitution_fail_closed(
    workspace: Path,
) -> None:
    first = _store(workspace)
    marker = first.root / ".store-id"
    original = marker.read_bytes()
    tampered = bytearray(original)
    tampered[0] ^= 1
    marker.chmod(0o600)
    marker.write_bytes(tampered)
    marker.chmod(0o400)
    with pytest.raises(ArtifactIntegrityError, match="marker"):
        first.inventory_objects(cutoff=_future_cutoff())

    marker.chmod(0o600)
    marker.write_bytes(original)
    marker.chmod(0o400)
    other = ArtifactStore(workspace / "other-cas")
    other.initialize()
    replacement = (other.root / ".store-id").read_bytes()
    marker.chmod(0o600)
    marker.write_bytes(replacement)
    marker.chmod(0o400)
    with pytest.raises(ArtifactIntegrityError, match="marker"):
        first.inventory_objects(cutoff=_future_cutoff())


@pytest.mark.parametrize("mutation", ["unsafe-mode", "hardlink", "symlink", "fifo"])
def test_store_identity_marker_boundary_mutations_fail_closed(
    workspace: Path,
    mutation: str,
) -> None:
    if mutation == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation requires POSIX")
    root = workspace / mutation
    root.mkdir(mode=0o700)
    store = ArtifactStore(root / "cas")
    store.initialize()
    marker = store.root / ".store-id"
    if mutation == "unsafe-mode":
        marker.chmod(0o644)
    elif mutation == "hardlink":
        os.link(marker, root / "marker-hardlink")
    elif mutation == "symlink":
        copy = root / "marker-copy"
        copy.write_bytes(marker.read_bytes())
        copy.chmod(0o400)
        marker.unlink()
        marker.symlink_to(copy)
    else:
        marker.unlink()
        os.mkfifo(marker)

    with pytest.raises((ArtifactBoundaryError, ArtifactIntegrityError), match="marker"):
        store.inventory_objects(cutoff=_future_cutoff())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL integration")
@pytest.mark.parametrize("target", ["root", "object"])
def test_store_rejects_real_macos_extended_acls(
    workspace: Path,
    target: str,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"acl-boundary")
    store = _store(workspace)
    artifact = store.publish(source)
    path = store.root if target == "root" else _object_path(store, artifact)
    permission = (
        "group:everyone allow list,search" if target == "root" else "group:everyone allow read"
    )
    subprocess.run(
        ["/bin/chmod", "+a", permission, os.fspath(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    try:
        with pytest.raises(ArtifactBoundaryError, match="extended ACL"):
            store.verify(artifact)
    finally:
        subprocess.run(
            ["/bin/chmod", "-N", os.fspath(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )


def test_posix_acl_xattrs_are_rejected_from_descriptor(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.open(workspace, os.O_RDONLY)
    monkeypatch.setattr(
        cas_module.os,
        "listxattr",
        lambda value: ["system.posix_acl_access"],
        raising=False,
    )
    try:
        with pytest.raises(ArtifactBoundaryError, match="POSIX ACL"):
            cas_module._assert_no_posix_acl_xattrs(descriptor, role="test object")
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("unsafe_mode", [0o777, 0o755, 0o711])
def test_non_owner_only_store_directories_fail_closed(
    workspace: Path,
    unsafe_mode: int,
) -> None:
    root = workspace / "cas"
    root.mkdir(mode=unsafe_mode)
    root.chmod(unsafe_mode)
    store = ArtifactStore(root)

    with pytest.raises(ArtifactBoundaryError, match="exact owner-only mode 700"):
        store.initialize()
    assert not (root / ".staging").exists()

    root.chmod(0o700)
    with pytest.raises(ArtifactIntegrityError, match="incomplete"):
        store.initialize()
    assert not (root / ".staging").exists()

    safe_store = ArtifactStore(workspace / "fresh-cas")
    safe_store.initialize()
    (safe_store.root / "objects").chmod(unsafe_mode)
    with pytest.raises(ArtifactBoundaryError, match="exact owner-only mode 700"):
        safe_store.publish(workspace / "missing-source")


def test_replaced_objects_directory_fails_closed(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)
    artifact = store.publish(source)
    objects = store.root / "objects"
    displaced = store.root / "objects-displaced"
    objects.rename(displaced)
    objects.mkdir()

    with pytest.raises(ArtifactIntegrityError, match="objects directory identity changed"):
        store.verify(artifact)


def test_error_text_does_not_disclose_source_or_root_path(workspace: Path) -> None:
    secret_name = "token-super-secret-source"
    source = workspace / secret_name
    store = _store(workspace)

    with pytest.raises(ArtifactBoundaryError) as captured:
        store.publish(source)

    rendered = str(captured.value)
    assert secret_name not in rendered
    assert str(workspace) not in rendered
    assert captured.value.errno_code == errno.ENOENT
    assert captured.value.__cause__ is None
    formatted = "".join(
        traceback.format_exception(
            type(captured.value),
            captured.value,
            captured.value.__traceback__,
        )
    )
    assert secret_name not in formatted
    assert str(workspace) not in formatted


def test_partial_os_writes_are_retried_until_complete(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"partial writes must not truncate")
    store = _store(workspace)
    original_write = cas_module.os.write

    def partial_write(descriptor: int, payload: bytes) -> int:
        return original_write(descriptor, payload[: max(1, len(payload) // 3)])

    monkeypatch.setattr(cas_module.os, "write", partial_write)

    artifact = store.publish(source)

    assert store.read_verified(artifact) == source.read_bytes()


def test_zero_progress_write_fails_and_cleans_staging(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)

    monkeypatch.setattr(cas_module.os, "write", lambda descriptor, payload: 0)

    with pytest.raises(ArtifactStoreError, match="made no progress"):
        store.publish(source)

    assert list((store.root / ".staging").iterdir()) == []


def test_os_write_failure_is_typed_and_cleans_staging(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)

    def fail_write(descriptor: int, payload: bytes) -> int:
        del descriptor, payload
        raise OSError(errno.ENOSPC, "injected secret path")

    monkeypatch.setattr(cas_module.os, "write", fail_write)

    with pytest.raises(ArtifactStoreError, match="staging bytes") as captured:
        store.publish(source)

    assert captured.value.errno_code == errno.ENOSPC
    assert captured.value.__cause__ is None
    assert "secret" not in str(captured.value)
    assert list((store.root / ".staging").iterdir()) == []


def test_fchmod_failure_is_precommit_and_cleans_staging(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"payload")
    store = _store(workspace)

    def fail_fchmod(descriptor: int, mode: int) -> None:
        del descriptor, mode
        raise OSError(errno.EPERM, "injected")

    monkeypatch.setattr(cas_module.os, "fchmod", fail_fchmod)

    with pytest.raises(ArtifactStoreError, match="read-only") as captured:
        store.publish(source)

    assert captured.value.errno_code == errno.EPERM
    assert list((store.root / ".staging").iterdir()) == []
    assert list((store.root / "objects").iterdir()) == []


def test_inventory_reads_do_not_initialize_missing_storage(workspace: Path) -> None:
    root = workspace / "missing-cas"
    store = ArtifactStore(root)

    with pytest.raises(ArtifactBoundaryError, match="not initialized"):
        store.inventory_objects(cutoff=_future_cutoff())
    with pytest.raises(ArtifactBoundaryError, match="not initialized"):
        store.discover_stale_staging(cutoff=_future_cutoff())

    assert not root.exists()


def test_empty_inventory_is_typed_bounded_and_path_free(workspace: Path) -> None:
    store = _store(workspace)

    objects = store.inventory_objects(cutoff=_future_cutoff())
    staging = store.discover_stale_staging(cutoff=_future_cutoff())

    assert objects == ArtifactInventoryPage(records=(), next_cursor=None, scanned_entries=0)
    assert staging == StagingInventoryPage(records=(), next_cursor=None, scanned_entries=0)


def test_object_inventory_paginates_deterministically_without_duplicates(
    workspace: Path,
) -> None:
    store = _store(workspace)
    cutoff = _future_cutoff()
    published: list[PublishedArtifact] = []
    for index in range(5):
        source = workspace / f"source-{index}.bin"
        source.write_bytes(f"payload-{index}".encode())
        published.append(store.publish(source))

    cursor: ArtifactInventoryCursor | None = None
    observed: list[ArtifactInventoryRecord] = []
    scan_counts: set[int] = set()
    while True:
        page = store.inventory_objects(
            cutoff=cutoff,
            cursor=cursor,
            page_size=2,
            max_scan_entries=100,
        )
        observed.extend(page.records)
        scan_counts.add(page.scanned_entries)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    expected = sorted(published, key=lambda artifact: artifact.digest)
    assert [record.digest for record in observed] == [artifact.digest for artifact in expected]
    assert [record.byte_size for record in observed] == [
        artifact.byte_size for artifact in expected
    ]
    assert [record.storage_key for record in observed] == [
        artifact.storage_key for artifact in expected
    ]
    assert len({record.digest for record in observed}) == len(observed)
    assert len(scan_counts) == 1
    assert all(record.last_changed_at.tzinfo is UTC for record in observed)
    assert all(not Path(record.storage_key).is_absolute() for record in observed)

    repeated = store.inventory_objects(
        cutoff=cutoff,
        page_size=2,
        max_scan_entries=100,
    )
    assert repeated.records == tuple(observed[:2])
    assert repeated.next_cursor == ArtifactInventoryCursor(
        after_digest=observed[1].digest,
        cutoff=cutoff,
        store_id=store.store_id,
    )


def test_inventory_cutoff_is_a_conservative_inclusive_grace_boundary(
    workspace: Path,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"bounded")
    store = _store(workspace)
    artifact = store.publish(source)
    staging = _staging_path(store, "a" * 48)

    assert store.inventory_objects(cutoff=_past_cutoff()).records == ()
    assert store.discover_stale_staging(cutoff=_past_cutoff()).records == ()

    object_page = store.inventory_objects(cutoff=_future_cutoff())
    staging_page = store.discover_stale_staging(cutoff=_future_cutoff())
    assert [record.digest for record in object_page.records] == [artifact.digest]
    assert [record.staging_id for record in staging_page.records] == [staging.name]


def test_staging_discovery_paginates_and_never_removes_files(workspace: Path) -> None:
    store = _store(workspace)
    cutoff = _future_cutoff()
    paths = [
        _staging_path(store, token * 48, f"partial-{token}".encode()) for token in ("0", "1", "f")
    ]

    first = store.discover_stale_staging(cutoff=cutoff, page_size=2)
    second = store.discover_stale_staging(
        cutoff=cutoff,
        cursor=first.next_cursor,
        page_size=2,
    )

    assert [record.staging_id for record in first.records] == [path.name for path in paths[:2]]
    assert first.next_cursor == StagingInventoryCursor(
        after_staging_id=paths[1].name,
        cutoff=cutoff,
        store_id=store.store_id,
    )
    assert [record.staging_id for record in second.records] == [paths[2].name]
    assert second.next_cursor is None
    assert [record.byte_size for record in (*first.records, *second.records)] == [
        len(path.read_bytes()) for path in paths
    ]
    assert all(path.exists() for path in paths)
    assert all(not Path(record.staging_id).is_absolute() for record in first.records)


def test_inventory_cursor_rejects_cutoff_advance_and_store_mismatch(
    workspace: Path,
) -> None:
    store = _store(workspace)
    cutoff = datetime.now(UTC) + timedelta(hours=1)
    for index in range(3):
        source = workspace / f"cursor-source-{index}.bin"
        source.write_bytes(f"cursor-payload-{index}".encode())
        store.publish(source)
    first = store.inventory_objects(cutoff=cutoff, page_size=1)
    assert first.next_cursor is not None

    with pytest.raises(ArtifactBoundaryError, match="cutoff does not match"):
        store.inventory_objects(
            cutoff=cutoff + timedelta(seconds=1),
            cursor=first.next_cursor,
            page_size=1,
        )

    other = ArtifactStore(workspace / "cursor-other-cas")
    other.initialize()
    with pytest.raises(ArtifactBoundaryError, match="another store"):
        other.inventory_objects(
            cutoff=cutoff,
            cursor=first.next_cursor,
            page_size=1,
        )


def test_inventory_cursors_must_name_an_existing_verified_position(
    workspace: Path,
) -> None:
    store = _store(workspace)
    source = workspace / "cursor-membership-source.bin"
    source.write_bytes(b"cursor-membership")
    store.publish(source)
    staging = _staging_path(store, "a" * 48)
    cutoff = _future_cutoff()

    forged_object_cursor = ArtifactInventoryCursor(
        after_digest="f" * 64,
        cutoff=cutoff,
        store_id=store.store_id,
    )
    with pytest.raises(ArtifactBoundaryError, match="position is absent"):
        store.inventory_objects(cutoff=cutoff, cursor=forged_object_cursor)

    forged_staging_cursor = StagingInventoryCursor(
        after_staging_id=f"publish-{'f' * 48}.tmp",
        cutoff=cutoff,
        store_id=store.store_id,
    )
    assert forged_staging_cursor.after_staging_id != staging.name
    with pytest.raises(ArtifactBoundaryError, match="position is absent"):
        store.discover_stale_staging(cutoff=cutoff, cursor=forged_staging_cursor)


def test_cutoff_bound_cursor_prevents_newly_eligible_lower_key_omission(
    workspace: Path,
) -> None:
    store = _store(workspace)
    published: list[PublishedArtifact] = []
    for index in range(4):
        source = workspace / f"omission-source-{index}.bin"
        source.write_bytes(f"omission-payload-{index}".encode())
        published.append(store.publish(source))
    ordered = sorted(published, key=lambda artifact: artifact.digest)
    newly_eligible = _object_path(store, ordered[0])
    future_ns = int((datetime.now(UTC) + timedelta(days=2)).timestamp() * 1_000_000_000)
    os.utime(newly_eligible, ns=(future_ns, future_ns))
    cutoff = datetime.now(UTC) + timedelta(hours=1)

    first = store.inventory_objects(cutoff=cutoff, page_size=1)
    assert first.next_cursor is not None
    assert first.records[0].digest == ordered[1].digest

    with pytest.raises(ArtifactBoundaryError, match="cutoff does not match"):
        store.inventory_objects(
            cutoff=cutoff + timedelta(days=3),
            cursor=first.next_cursor,
            page_size=1,
        )


def test_inventory_records_are_frozen_and_validate_public_invariants() -> None:
    digest = hashlib.sha256(b"record").hexdigest()
    cutoff = datetime.now(UTC)
    observed_at = datetime.now(UTC)
    store_id = "a" * 64
    record = ArtifactInventoryRecord(
        digest=digest,
        byte_size=6,
        storage_key=f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
        last_changed_at=observed_at,
        last_changed_ns=_timestamp_nanoseconds(observed_at),
        generation=ArtifactGeneration("a" * 64),
    )
    staging = StagingInventoryRecord(
        staging_id=f"publish-{'0' * 48}.tmp",
        byte_size=1,
        last_changed_at=datetime.now(UTC),
    )

    with pytest.raises(FrozenInstanceError):
        record.byte_size = 7  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        staging.byte_size = 2  # type: ignore[misc]
    with pytest.raises(ArtifactBoundaryError, match="timezone-aware"):
        ArtifactInventoryRecord(
            digest=digest,
            byte_size=6,
            storage_key=f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
            last_changed_at=datetime.now(),
            last_changed_ns=0,
            generation=ArtifactGeneration("a" * 64),
        )
    with pytest.raises(ArtifactBoundaryError, match="canonical"):
        StagingInventoryRecord(
            staging_id="../escape",
            byte_size=1,
            last_changed_at=datetime.now(UTC),
        )
    with pytest.raises(ArtifactBoundaryError, match="tuple of typed records"):
        ArtifactInventoryPage(records=[record], next_cursor=None, scanned_entries=1)  # type: ignore[arg-type]
    with pytest.raises(ArtifactBoundaryError, match="unique and ordered"):
        ArtifactInventoryPage(records=(record, record), next_cursor=None, scanned_entries=1)
    with pytest.raises(ArtifactBoundaryError, match="cursor must end"):
        ArtifactInventoryPage(
            records=(record,),
            next_cursor=ArtifactInventoryCursor(
                after_digest="f" * 64,
                cutoff=cutoff,
                store_id=store_id,
            ),
            scanned_entries=1,
        )
    with pytest.raises(ArtifactBoundaryError, match="cursor must be typed"):
        ArtifactInventoryPage(
            records=(record,),
            next_cursor=record.digest,  # type: ignore[arg-type]
            scanned_entries=1,
        )
    with pytest.raises(ArtifactBoundaryError, match="non-negative"):
        StagingInventoryPage(records=(), next_cursor=None, scanned_entries=-1)


def test_inventory_values_normalize_utc_and_enforce_hard_result_bounds() -> None:
    zero_offset = timezone(timedelta(0), name="zero-offset")
    observed_at = datetime(2026, 8, 8, 12, tzinfo=zero_offset)
    digest = hashlib.sha256(b"utc-record").hexdigest()
    record = ArtifactInventoryRecord(
        digest=digest,
        byte_size=10,
        storage_key=f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
        last_changed_at=observed_at,
        last_changed_ns=_timestamp_nanoseconds(observed_at),
        generation=ArtifactGeneration("b" * 64),
    )
    cursor = ArtifactInventoryCursor(
        after_digest=digest,
        cutoff=observed_at,
        store_id="b" * 64,
    )
    staging = StagingInventoryRecord(
        staging_id=f"publish-{'a' * 48}.tmp",
        byte_size=0,
        last_changed_at=observed_at,
    )
    staging_cursor = StagingInventoryCursor(
        after_staging_id=staging.staging_id,
        cutoff=observed_at,
        store_id="b" * 64,
    )

    assert record.last_changed_at.tzinfo is UTC
    assert staging.last_changed_at.tzinfo is UTC
    assert cursor.cutoff.tzinfo is UTC
    assert staging_cursor.cutoff.tzinfo is UTC
    with pytest.raises(FrozenInstanceError):
        cursor.store_id = "c" * 64  # type: ignore[misc]
    with pytest.raises(ArtifactBoundaryError, match="smaller than returned"):
        ArtifactInventoryPage(records=(record,), next_cursor=None, scanned_entries=0)
    with pytest.raises(ArtifactBoundaryError, match="cannot exceed 100000"):
        ArtifactInventoryPage(
            records=(),
            next_cursor=None,
            scanned_entries=100_001,
        )

    too_many = tuple(
        ArtifactInventoryRecord(
            digest=value,
            byte_size=0,
            storage_key=f"objects/{value[:2]}/{value[2:4]}/{value}",
            last_changed_at=observed_at,
            last_changed_ns=_timestamp_nanoseconds(observed_at),
            generation=ArtifactGeneration(hashlib.sha256(value.encode()).hexdigest()),
        )
        for index in range(1_001)
        for value in (f"{index:064x}",)
    )
    with pytest.raises(ArtifactBoundaryError, match="cannot exceed 1000 records"):
        ArtifactInventoryPage(
            records=too_many,
            next_cursor=None,
            scanned_entries=1_001,
        )

    staging_page = StagingInventoryPage(
        records=(staging,),
        next_cursor=staging_cursor,
        scanned_entries=1,
    )
    assert staging_page.next_cursor == staging_cursor
    with pytest.raises(ArtifactBoundaryError, match="tuple of typed records"):
        StagingInventoryPage(
            records=[staging],  # type: ignore[arg-type]
            next_cursor=None,
            scanned_entries=1,
        )
    with pytest.raises(ArtifactBoundaryError, match="unique and ordered"):
        StagingInventoryPage(
            records=(staging, staging),
            next_cursor=None,
            scanned_entries=2,
        )
    with pytest.raises(ArtifactBoundaryError, match="cursor must be typed"):
        StagingInventoryPage(
            records=(staging,),
            next_cursor=staging.staging_id,  # type: ignore[arg-type]
            scanned_entries=1,
        )
    with pytest.raises(ArtifactBoundaryError, match="cursor must end"):
        StagingInventoryPage(
            records=(staging,),
            next_cursor=StagingInventoryCursor(
                after_staging_id=f"publish-{'b' * 48}.tmp",
                cutoff=observed_at,
                store_id="b" * 64,
            ),
            scanned_entries=1,
        )
    with pytest.raises(ArtifactBoundaryError, match="smaller than returned"):
        StagingInventoryPage(
            records=(staging,),
            next_cursor=None,
            scanned_entries=0,
        )
    with pytest.raises(ArtifactBoundaryError, match="cannot exceed 100000"):
        StagingInventoryPage(
            records=(),
            next_cursor=None,
            scanned_entries=100_001,
        )
    too_many_staging = tuple(
        StagingInventoryRecord(
            staging_id=f"publish-{index:048x}.tmp",
            byte_size=0,
            last_changed_at=observed_at,
        )
        for index in range(1_001)
    )
    with pytest.raises(ArtifactBoundaryError, match="cannot exceed 1000 records"):
        StagingInventoryPage(
            records=too_many_staging,
            next_cursor=None,
            scanned_entries=1_001,
        )


@pytest.mark.parametrize(
    ("page_size", "max_scan_entries", "time_budget_seconds", "message"),
    [
        (0, 10, 1.0, "positive integer"),
        (1001, 10, 1.0, "cannot exceed 1000"),
        (1, 0, 1.0, "positive integer"),
        (1, 100_001, 1.0, "cannot exceed 100000"),
        (1, 10, 0.0, "finite and positive"),
        (1, 10, float("inf"), "finite and positive"),
        (1, 10, 61.0, "cannot exceed 60"),
    ],
)
def test_inventory_rejects_invalid_caller_bounds_before_scanning(
    workspace: Path,
    page_size: int,
    max_scan_entries: int,
    time_budget_seconds: float,
    message: str,
) -> None:
    store = _store(workspace)

    with pytest.raises(ArtifactBoundaryError, match=message):
        store.inventory_objects(
            cutoff=_future_cutoff(),
            page_size=page_size,
            max_scan_entries=max_scan_entries,
            time_budget_seconds=time_budget_seconds,
        )


def test_inventory_rejects_invalid_cutoff_and_cursors(workspace: Path) -> None:
    store = _store(workspace)

    with pytest.raises(ArtifactBoundaryError, match="timezone-aware"):
        store.inventory_objects(cutoff=datetime.now())
    with pytest.raises(ArtifactBoundaryError, match="cursor must be typed"):
        store.inventory_objects(cutoff=_future_cutoff(), cursor="0" * 63)  # type: ignore[arg-type]
    with pytest.raises(ArtifactBoundaryError, match="cursor must be typed"):
        store.discover_stale_staging(
            cutoff=_future_cutoff(),
            cursor="publish-bad.tmp",  # type: ignore[arg-type]
        )


def test_inventory_scan_entry_bound_fails_before_unbounded_accumulation(
    workspace: Path,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"bounded")
    store = _store(workspace)
    store.publish(source)

    with pytest.raises(ArtifactBoundaryError, match="scan-entry bound"):
        store.inventory_objects(cutoff=_future_cutoff(), max_scan_entries=1)

    _staging_path(store, "0" * 48)
    _staging_path(store, "1" * 48)
    with pytest.raises(ArtifactBoundaryError, match="scan-entry bound"):
        store.discover_stale_staging(cutoff=_future_cutoff(), max_scan_entries=1)


def test_inventory_cooperative_time_budget_and_clock_regression_fail_closed(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(workspace)
    times = iter((0, 2_000_000_000))
    monkeypatch.setattr(cas_module.time, "monotonic_ns", lambda: next(times))

    with pytest.raises(ArtifactBoundaryError, match="time budget"):
        store.inventory_objects(cutoff=_future_cutoff(), time_budget_seconds=1.0)

    times = iter((10, 9))
    monkeypatch.setattr(cas_module.time, "monotonic_ns", lambda: next(times))
    with pytest.raises(ArtifactIntegrityError, match="clock moved backwards"):
        store.discover_stale_staging(cutoff=_future_cutoff())


def test_inventory_checks_deadline_after_cursor_filtering(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(workspace)
    cutoff = _future_cutoff()
    for index in range(2):
        source = workspace / f"deadline-source-{index}.bin"
        source.write_bytes(f"deadline-payload-{index}".encode())
        store.publish(source)
    full_page = store.inventory_objects(cutoff=cutoff)
    cursor = ArtifactInventoryCursor(
        after_digest=full_page.records[0].digest,
        cutoff=cutoff,
        store_id=store.store_id,
    )
    now_ns = 0

    class ExpiringRecords(list[ArtifactInventoryRecord]):
        def __iter__(self) -> Any:
            nonlocal now_ns
            now_ns = 2_000_000_000
            return super().__iter__()

    records = ExpiringRecords(full_page.records)
    monkeypatch.setattr(
        store,
        "_inspect_object_tree",
        lambda *args, **kwargs: records,
    )
    monkeypatch.setattr(cas_module.time, "monotonic_ns", lambda: now_ns)

    with pytest.raises(ArtifactBoundaryError, match="time budget"):
        store.inventory_objects(
            cutoff=cutoff,
            cursor=cursor,
            time_budget_seconds=1.0,
        )


@pytest.mark.parametrize("name", ["AA", "g0", "000", ".DS_Store"])
def test_object_inventory_rejects_noncanonical_top_level_entries(
    workspace: Path,
    name: str,
) -> None:
    store = _store(workspace)
    _mkdir_private(store.root / "objects" / name)

    with pytest.raises(ArtifactBoundaryError, match="fanout entry"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_object_inventory_rejects_noncanonical_second_level_and_digest_fanout(
    workspace: Path,
) -> None:
    store = _store(workspace)
    first = store.root / "objects" / "aa"
    _mkdir_private(first)
    _mkdir_private(first / "GG")
    with pytest.raises(ArtifactBoundaryError, match="fanout entry"):
        store.inventory_objects(cutoff=_future_cutoff())

    (first / "GG").rmdir()
    second = first / "bb"
    _mkdir_private(second)
    digest = hashlib.sha256(b"misplaced").hexdigest()
    object_path = second / digest
    object_path.write_bytes(b"misplaced")
    object_path.chmod(0o400)
    with pytest.raises(ArtifactBoundaryError, match="canonical digest fanout"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_object_inventory_rejects_noncanonical_object_name(workspace: Path) -> None:
    store = _store(workspace)
    parent = store.root / "objects" / "aa" / "bb"
    _mkdir_private(parent)
    path = parent / ("0" * 63)
    path.write_bytes(b"invalid")
    path.chmod(0o400)

    with pytest.raises(ArtifactBoundaryError, match="lowercase SHA-256"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_object_inventory_rejects_fanout_and_object_symlinks(workspace: Path) -> None:
    store = _store(workspace)
    outside = workspace / "outside"
    outside.mkdir()
    (store.root / "objects" / "aa").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactBoundaryError, match="opened safely"):
        store.inventory_objects(cutoff=_future_cutoff())

    (store.root / "objects" / "aa").unlink()
    second = store.root / "objects" / "aa" / "bb"
    _mkdir_private(second)
    payload = b"outside-object"
    digest = hashlib.sha256(payload).hexdigest()
    canonical_parent = store.root / "objects" / digest[:2] / digest[2:4]
    _mkdir_private(canonical_parent)
    outside_file = workspace / "outside.bin"
    outside_file.write_bytes(payload)
    (canonical_parent / digest).symlink_to(outside_file)
    with pytest.raises(ArtifactBoundaryError, match="opened safely"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_staging_discovery_rejects_names_symlinks_and_special_files(
    workspace: Path,
) -> None:
    store = _store(workspace)
    invalid = store.root / ".staging" / "leftover.tmp"
    invalid.write_bytes(b"partial")
    with pytest.raises(ArtifactBoundaryError, match="canonical"):
        store.discover_stale_staging(cutoff=_future_cutoff())

    invalid.unlink()
    outside = workspace / "outside.bin"
    outside.write_bytes(b"outside")
    symlink = store.root / ".staging" / f"publish-{'1' * 48}.tmp"
    symlink.symlink_to(outside)
    with pytest.raises(ArtifactBoundaryError, match="opened safely"):
        store.discover_stale_staging(cutoff=_future_cutoff())

    symlink.unlink()
    if hasattr(os, "mkfifo"):
        fifo = store.root / ".staging" / f"publish-{'2' * 48}.tmp"
        os.mkfifo(fifo)
        with pytest.raises(ArtifactBoundaryError, match="regular file"):
            store.discover_stale_staging(cutoff=_future_cutoff())


def test_object_inventory_rejects_writable_corrupt_and_hardlinked_objects(
    workspace: Path,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"trusted")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)

    destination.chmod(0o600)
    with pytest.raises(ArtifactIntegrityError, match="not stored read-only"):
        store.inventory_objects(cutoff=_future_cutoff())

    destination.write_bytes(b"corrupt")
    destination.chmod(0o400)
    with pytest.raises(ArtifactIntegrityError, match="digest differs"):
        store.inventory_objects(cutoff=_future_cutoff())

    destination.chmod(0o600)
    destination.write_bytes(b"trusted")
    destination.chmod(0o400)
    external_link = workspace / "unexpected-link"
    os.link(destination, external_link)
    with pytest.raises(ArtifactIntegrityError, match="hard-link count"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_inventory_enforces_store_byte_bound_on_objects_and_staging(
    workspace: Path,
) -> None:
    store = _store(workspace, maximum=4)
    payload = b"12345"
    digest = hashlib.sha256(payload).hexdigest()
    parent = store.root / "objects" / digest[:2] / digest[2:4]
    _mkdir_private(parent)
    object_path = parent / digest
    object_path.write_bytes(payload)
    object_path.chmod(0o400)

    with pytest.raises(ArtifactBoundaryError, match="object exceeds store limit"):
        store.inventory_objects(cutoff=_future_cutoff())

    object_path.unlink()
    staging = _staging_path(store, "5" * 48, payload)
    with pytest.raises(ArtifactBoundaryError, match="staging file exceeds store limit"):
        store.discover_stale_staging(cutoff=_future_cutoff())
    assert staging.read_bytes() == payload


def test_staging_discovery_rejects_unsafe_permissions_and_hardlinks(
    workspace: Path,
) -> None:
    store = _store(workspace)
    staging = _staging_path(store, "3" * 48)
    staging.chmod(0o644)

    with pytest.raises(ArtifactBoundaryError, match="permissions"):
        store.discover_stale_staging(cutoff=_future_cutoff())

    staging.chmod(0o600)
    os.link(staging, workspace / "staging-hard-link")
    with pytest.raises(ArtifactIntegrityError, match="hard-link count"):
        store.discover_stale_staging(cutoff=_future_cutoff())


def test_object_inventory_rejects_special_file_at_canonical_key(workspace: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation requires POSIX")
    store = _store(workspace)
    digest = "aabb" + "0" * 60
    parent = store.root / "objects" / "aa" / "bb"
    _mkdir_private(parent)
    os.mkfifo(parent / digest)

    with pytest.raises(ArtifactBoundaryError, match="regular file"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_regular_file_acquisition_rejects_fifo_before_open(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation requires POSIX")
    parent = workspace / "fifo-parent"
    parent.mkdir(mode=0o700)
    fifo = parent / "candidate"
    os.mkfifo(fifo)
    parent_descriptor = os.open(parent, os.O_RDONLY)
    original_open = cas_module.os.open
    candidate_opened = False

    def observe_open(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal candidate_opened
        if path == fifo.name and kwargs.get("dir_fd") == parent_descriptor:
            candidate_opened = True
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(cas_module.os, "open", observe_open)
    try:
        with pytest.raises(ArtifactBoundaryError, match="regular file"):
            ArtifactStore._open_regular_at(
                parent_descriptor,
                fifo.name,
                role="test FIFO",
                allowed_modes=None,
            )
    finally:
        os.close(parent_descriptor)

    assert candidate_opened is False


def test_inventory_detects_directory_mutation_after_entry_inspection(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"trusted")
    store = _store(workspace)
    artifact = store.publish(source)
    parent = _object_path(store, artifact).parent
    original = store._inspect_inventory_object

    def mutate_after_inspection(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        (parent / "unexpected").write_bytes(b"mutation")
        return result

    monkeypatch.setattr(store, "_inspect_inventory_object", mutate_after_inspection)

    with pytest.raises(ArtifactIntegrityError, match="changed during inventory"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_inventory_detects_object_and_staging_mutation_during_inspection(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"trusted")
    store = _store(workspace)
    artifact = store.publish(source)
    destination = _object_path(store, artifact)
    original_hash = store._hash_descriptor

    def mutate_after_hash(*args: Any, **kwargs: Any) -> str:
        result = original_hash(*args, **kwargs)
        destination.chmod(0o600)
        destination.write_bytes(b"trusted")
        destination.chmod(0o400)
        return result

    monkeypatch.setattr(store, "_hash_descriptor", mutate_after_hash)
    with pytest.raises(ArtifactIntegrityError, match="changed during inventory"):
        store.inventory_objects(cutoff=_future_cutoff())

    monkeypatch.setattr(store, "_hash_descriptor", original_hash)
    staging = _staging_path(store, "6" * 48)
    original_status_check = store._assert_staging_file_status
    checks = 0

    def mutate_staging_after_first_check(value: os.stat_result) -> None:
        nonlocal checks
        original_status_check(value)
        checks += 1
        if checks == 1:
            staging.write_bytes(b"mutated")

    monkeypatch.setattr(store, "_assert_staging_file_status", mutate_staging_after_first_check)
    with pytest.raises(ArtifactIntegrityError, match="changed during inventory"):
        store.discover_stale_staging(cutoff=_future_cutoff())


def test_inventory_rejects_unrepresentable_filesystem_timestamp(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"trusted")
    store = _store(workspace)
    store.publish(source)
    monkeypatch.setattr(cas_module, "_last_changed_nanoseconds", lambda value: 10**40)

    with pytest.raises(ArtifactIntegrityError, match="supported UTC range"):
        store.inventory_objects(cutoff=_future_cutoff())


def test_inventory_is_observational_and_does_not_mutate_tree(workspace: Path) -> None:
    source = workspace / "source.bin"
    source.write_bytes(b"trusted")
    store = _store(workspace)
    store.publish(source)
    _staging_path(store, "4" * 48)

    def snapshot() -> tuple[tuple[str, int, int, int, int, str | None], ...]:
        entries: list[tuple[str, int, int, int, int, str | None]] = []
        for path in sorted(store.root.rglob("*")):
            status = os.lstat(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            entries.append(
                (
                    path.relative_to(store.root).as_posix(),
                    stat_mode(path),
                    status.st_size,
                    status.st_mtime_ns,
                    status.st_ctime_ns,
                    digest,
                )
            )
        return tuple(entries)

    before = snapshot()
    store.inventory_objects(cutoff=_future_cutoff())
    store.discover_stale_staging(cutoff=_future_cutoff())
    after = snapshot()

    assert after == before
