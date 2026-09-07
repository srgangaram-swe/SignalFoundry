"""Tests for bounded sibling-repository provenance verification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alphaforge.research import cross_repository_provenance as provenance_module
from alphaforge.research.cross_repository_provenance import (
    CrossRepositoryProvenanceError,
    load_cross_repository_receipt,
    verify_cross_repository_receipt,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    repository = tmp_path / "source"
    repository.mkdir(parents=True)
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Test Owner")
    _git(repository, "config", "user.email", "owner@example.invalid")
    _git(
        repository,
        "remote",
        "add",
        "origin",
        "https://github.com/example/source.git",
    )
    evidence = repository / "docs" / "evidence.json"
    evidence.parent.mkdir()
    content = b'{"scope":"aggregate-only"}\n'
    evidence.write_bytes(content)
    _git(repository, "add", "docs/evidence.json")
    _git(repository, "commit", "-m", "Add evidence")
    commit = _git(repository, "rev-parse", "HEAD")
    blob = _git(repository, "rev-parse", f"{commit}:docs/evidence.json")
    document = {
        "schema_version": "1.0.0",
        "repository": "example/source",
        "origin_url": "https://github.com/example/source.git",
        "commit": commit,
        "verification": {
            "mode": "local_git_object_database",
            "verified_at_utc": "2026-07-26T22:00:00Z",
            "network_requests": 0,
        },
        "sources": [
            {
                "family": "aggregate_fixture",
                "path": "docs/evidence.json",
                "git_blob_sha1": blob,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "claim_scope": "aggregate fixture contract only",
            }
        ],
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return repository, receipt, document


def test_verification_reads_pinned_commit_not_dirty_worktree(tmp_path: Path) -> None:
    repository, receipt, _ = _fixture(tmp_path)

    first = verify_cross_repository_receipt(receipt, checkout=repository)
    (repository / "docs" / "evidence.json").write_text(
        '{"scope":"dirty-worktree"}\n',
        encoding="utf-8",
    )
    second = verify_cross_repository_receipt(receipt, checkout=repository)

    assert first == second
    assert first.source_count == 1
    assert first.total_bytes == len(b'{"scope":"aggregate-only"}\n')


def test_committed_receipt_is_network_free_and_semantically_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_subprocess(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("committed receipt loading must not invoke Git or a network path")

    monkeypatch.setattr(provenance_module.subprocess, "Popen", reject_subprocess)
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/signal_foundry_sprint_3/cross_repository_provenance.json"
    )
    receipt = load_cross_repository_receipt(receipt_path)

    assert receipt.repository == "srgangaram-swe/Signalattice"
    assert receipt.origin_url == "https://github.com/srgangaram-swe/Signalattice.git"
    assert receipt.commit == "000ae12de3b409e5f409b53fb191aa003b105318"
    assert len(receipt.sources) == 9
    assert sum(source.bytes for source in receipt.sources) == 142_578
    assert {source.family for source in receipt.sources} == {
        "adaptive_decomposition",
        "spectral_descriptors",
        "state_space",
        "time_frequency_vision",
    }
    assert all(
        0 < source.bytes <= provenance_module.MAX_EXTERNAL_SOURCE_BYTES
        for source in receipt.sources
    )
    assert all(len(source.git_blob_sha1) == 40 for source in receipt.sources)
    assert all(len(source.sha256) == 64 for source in receipt.sources)
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        == "982ea9c4f2204148cd532824a807dcee4e9c769948c4ae29193d267c5b5d6391"
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda document: document["sources"][0].update({"sha256": "0" * 64}),
            "SHA-256 mismatch",
        ),
        (
            lambda document: document["sources"][0].update({"bytes": 1}),
            "byte-size mismatch",
        ),
        (
            lambda document: document.update(
                {"origin_url": "https://github.com/example/other.git"}
            ),
            "same GitHub repository",
        ),
        (
            lambda document: document.update(
                {
                    "repository": "example/other",
                    "origin_url": "https://github.com/example/other.git",
                }
            ),
            "origin mismatch",
        ),
        (
            lambda document: document.update({"repository": "example/other"}),
            "same GitHub repository",
        ),
        (
            lambda document: document["sources"][0].update({"path": "../escape"}),
            "safe repository-relative",
        ),
    ],
)
def test_tampered_receipts_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    repository, receipt, document = _fixture(tmp_path)
    mutation(document)
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CrossRepositoryProvenanceError, match=match):
        verify_cross_repository_receipt(receipt, checkout=repository)


def test_loader_rejects_unknown_fields_duplicates_and_symlinks(tmp_path: Path) -> None:
    _, receipt, document = _fixture(tmp_path)
    document["unknown"] = True
    receipt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CrossRepositoryProvenanceError, match="fields mismatch"):
        load_cross_repository_receipt(receipt)

    _, receipt, document = _fixture(tmp_path / "duplicate")
    document["sources"].append(dict(document["sources"][0]))
    receipt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CrossRepositoryProvenanceError, match="unique"):
        load_cross_repository_receipt(receipt)

    _, receipt, _ = _fixture(tmp_path / "linked")
    link = tmp_path / "receipt-link.json"
    link.symlink_to(receipt)
    with pytest.raises(CrossRepositoryProvenanceError, match="symlink"):
        load_cross_repository_receipt(link)


def test_receipt_size_is_bounded_before_verification(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(CrossRepositoryProvenanceError, match="receipt bytes"):
        load_cross_repository_receipt(oversized)
    with pytest.raises(CrossRepositoryProvenanceError, match="receipt bytes"):
        verify_cross_repository_receipt(oversized, checkout=tmp_path)


def test_verifier_rejects_symlinked_checkout(tmp_path: Path) -> None:
    repository, receipt, _ = _fixture(tmp_path)
    link = tmp_path / "checkout-link"
    link.symlink_to(repository, target_is_directory=True)

    with pytest.raises(CrossRepositoryProvenanceError, match="symlink"):
        verify_cross_repository_receipt(receipt, checkout=link)


def test_verifier_ignores_git_replace_refs(tmp_path: Path) -> None:
    repository, receipt, document = _fixture(tmp_path)
    claimed_commit = document["commit"]
    evidence = repository / "docs" / "evidence.json"
    replacement_content = b'{"scope":"replacement-only"}\n'
    evidence.write_bytes(replacement_content)
    _git(repository, "add", "docs/evidence.json")
    _git(repository, "commit", "-m", "Add replacement-only evidence")
    replacement_commit = _git(repository, "rev-parse", "HEAD")
    replacement_blob = _git(
        repository,
        "rev-parse",
        f"{replacement_commit}:docs/evidence.json",
    )
    document["sources"][0].update(
        {
            "git_blob_sha1": replacement_blob,
            "bytes": len(replacement_content),
            "sha256": hashlib.sha256(replacement_content).hexdigest(),
        }
    )
    receipt.write_text(json.dumps(document), encoding="utf-8")
    _git(repository, "replace", claimed_commit, replacement_commit)

    with pytest.raises(CrossRepositoryProvenanceError, match="Git blob mismatch"):
        verify_cross_repository_receipt(receipt, checkout=repository)


def test_verifier_rejects_symlink_and_executable_tree_modes(tmp_path: Path) -> None:
    repository, receipt, document = _fixture(tmp_path)
    evidence = repository / "docs" / "evidence.json"
    evidence.unlink()
    evidence.symlink_to("../../outside.json")
    _git(repository, "add", "docs/evidence.json")
    _git(repository, "commit", "-m", "Replace evidence with a symlink")
    symlink_commit = _git(repository, "rev-parse", "HEAD")
    symlink_blob = _git(repository, "rev-parse", f"{symlink_commit}:docs/evidence.json")
    symlink_content = b"../../outside.json"
    document["commit"] = symlink_commit
    document["sources"][0].update(
        {
            "git_blob_sha1": symlink_blob,
            "bytes": len(symlink_content),
            "sha256": hashlib.sha256(symlink_content).hexdigest(),
        }
    )
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CrossRepositoryProvenanceError, match="regular blob"):
        verify_cross_repository_receipt(receipt, checkout=repository)

    evidence.unlink()
    evidence.write_text('{"scope":"executable"}\n', encoding="utf-8")
    evidence.chmod(0o755)
    _git(repository, "add", "docs/evidence.json")
    _git(repository, "commit", "-m", "Replace evidence with an executable")
    executable_commit = _git(repository, "rev-parse", "HEAD")
    executable_blob = _git(
        repository,
        "rev-parse",
        f"{executable_commit}:docs/evidence.json",
    )
    executable_content = evidence.read_bytes()
    document["commit"] = executable_commit
    document["sources"][0].update(
        {
            "git_blob_sha1": executable_blob,
            "bytes": len(executable_content),
            "sha256": hashlib.sha256(executable_content).hexdigest(),
        }
    )
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CrossRepositoryProvenanceError, match="regular blob"):
        verify_cross_repository_receipt(receipt, checkout=repository)


def test_receipt_parser_rejects_duplicates_deep_documents_and_boolean_counts(
    tmp_path: Path,
) -> None:
    _, receipt, document = _fixture(tmp_path)
    duplicate = receipt.read_text(encoding="utf-8").replace(
        "{",
        '{"repository":"example/source",',
        1,
    )
    receipt.write_text(duplicate, encoding="utf-8")
    with pytest.raises(CrossRepositoryProvenanceError, match="duplicate key"):
        load_cross_repository_receipt(receipt)

    receipt.write_text(
        '{"nested":' + "[" * 65 + "0" + "]" * 65 + "}",
        encoding="utf-8",
    )
    with pytest.raises(CrossRepositoryProvenanceError, match="depth"):
        load_cross_repository_receipt(receipt)

    document["verification"]["network_requests"] = False
    receipt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CrossRepositoryProvenanceError, match="network-free"):
        load_cross_repository_receipt(receipt)


def test_receipt_snapshot_rejects_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, receipt, _ = _fixture(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(receipt.read_bytes())
    original_open = os.open
    swapped = False

    def swap_before_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if Path(os.fsdecode(path)) == receipt and not swapped:
            swapped = True
            receipt.unlink()
            receipt.symlink_to(replacement)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    from alphaforge.research import _bounded_io

    monkeypatch.setattr(_bounded_io.os, "open", swap_before_open)
    with pytest.raises(CrossRepositoryProvenanceError, match="non-symlink"):
        load_cross_repository_receipt(receipt)
    assert swapped


def test_git_environment_disables_replacements_and_lazy_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, receipt, _ = _fixture(tmp_path)
    observed_environments: list[dict[str, str]] = []
    original_popen = subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        observed_environments.append(dict(kwargs["env"]))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(provenance_module.subprocess, "Popen", recording_popen)
    verify_cross_repository_receipt(receipt, checkout=repository)

    assert observed_environments
    assert all(
        environment["GIT_NO_REPLACE_OBJECTS"] == "1" for environment in observed_environments
    )
    assert all(environment["GIT_NO_LAZY_FETCH"] == "1" for environment in observed_environments)
    assert all(environment["GIT_TERMINAL_PROMPT"] == "0" for environment in observed_environments)


def test_git_output_is_capped_while_streaming(tmp_path: Path) -> None:
    repository, _, _ = _fixture(tmp_path)
    _git(repository, "config", "audit.oversized", "x" * 4096)

    with pytest.raises(CrossRepositoryProvenanceError, match="output bounds"):
        provenance_module._run_git(
            repository,
            ["config", "--get", "audit.oversized"],
            maximum_stdout=32,
        )
