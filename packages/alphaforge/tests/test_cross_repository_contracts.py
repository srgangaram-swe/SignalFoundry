"""Validator contracts for cross-repository provenance receipts (SF-S3-MR11).

A receipt asserts that a named blob, at a named commit, in a *second* repository
supports a claim in this one. Every field is therefore attacker-controlled from
this repository's point of view, and the validators are the trust boundary.

These tests pin the refusals rather than the happy path: a path that escapes the
repository, an origin URL carrying embedded credentials, a repository identifier
that disagrees with its own URL, a timestamp without an explicit zone, and
declared byte counts that would let a receipt authorize an unbounded read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from alphaforge.research.cross_repository_provenance import (
    MAX_EXTERNAL_SOURCE_BYTES,
    CrossRepositoryProvenanceError,
    CrossRepositoryReceipt,
    ExternalSource,
    load_cross_repository_receipt,
    verify_cross_repository_receipt,
)

BLOB = "a" * 40
DIGEST = "b" * 64


def _source(**overrides: object) -> ExternalSource:
    fields: dict[str, object] = {
        "family": "spectral",
        "path": "docs/spectral_features.md",
        "git_blob_sha1": BLOB,
        "bytes": 128,
        "sha256": DIGEST,
        "claim_scope": "descriptor definitions",
    }
    fields.update(overrides)
    return ExternalSource(**fields)  # type: ignore[arg-type]


def _receipt(**overrides: object) -> CrossRepositoryReceipt:
    fields: dict[str, object] = {
        "repository": "srgangaram-swe/Signalattice",
        "origin_url": "https://github.com/srgangaram-swe/Signalattice.git",
        "commit": "c" * 40,
        "verified_at_utc": "2026-07-26T12:00:00Z",
        "sources": (_source(),),
    }
    fields.update(overrides)
    return CrossRepositoryReceipt(**fields)  # type: ignore[arg-type]


def test_a_well_formed_receipt_is_accepted() -> None:
    receipt = _receipt()
    assert receipt.repository == "srgangaram-swe/Signalattice"
    assert len(receipt.sources) == 1


# ---------------------------------------------------------------------------
# ExternalSource
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"git_blob_sha1": "A" * 40}, "git_blob_sha1"),
        ({"git_blob_sha1": "a" * 39}, "git_blob_sha1"),
        ({"git_blob_sha1": 1234}, "git_blob_sha1"),
        ({"sha256": "b" * 63}, "sha256"),
        ({"sha256": "z" * 64}, "sha256"),
        ({"family": ""}, "source.family"),
        ({"family": " leading"}, "source.family"),
        ({"family": "non-ascii-\u00e9"}, "source.family"),
        ({"family": "with\x00null"}, "source.family"),
        ({"claim_scope": "x" * 1001}, "source.claim_scope"),
    ],
)
def test_source_field_validation(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match=message):
        _source(**overrides)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../escape.md",
        "docs/../../escape.md",
        "docs\\windows.md",
        "C:/drive.md",
        "docs/we ird.md",
    ],
)
def test_source_paths_must_be_contained_posix_paths(path: str) -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match="safe repository-relative"):
        _source(path=path)


@pytest.mark.parametrize("count", [0, -1, True, MAX_EXTERNAL_SOURCE_BYTES + 1])
def test_source_bytes_must_be_a_bounded_positive_int(count: object) -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match="source.bytes"):
        _source(bytes=count)


# ---------------------------------------------------------------------------
# CrossRepositoryReceipt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repository",
    ["Signalattice", "a/b/c", "/Signalattice", "owner/", "owner/..", "own er/repo"],
)
def test_repository_must_be_a_safe_owner_repo_identifier(repository: str) -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match="repository|origin"):
        _receipt(repository=repository)


@pytest.mark.parametrize(
    "origin",
    [
        "http://github.com/srgangaram-swe/Signalattice.git",
        "https://github.com/srgangaram-swe/Signalattice",
        "git@github.com:srgangaram-swe/Signalattice.git",
        "https://token@github.com/srgangaram-swe/Signalattice.git",
    ],
)
def test_origin_url_must_be_credential_free_https(origin: str) -> None:
    """A receipt must never carry an embedded credential into a log or manifest."""
    with pytest.raises(CrossRepositoryProvenanceError, match="origin_url"):
        _receipt(origin_url=origin)


def test_repository_and_origin_must_agree() -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match="same GitHub repository"):
        _receipt(origin_url="https://github.com/srgangaram-swe/AlphaForge.git")


@pytest.mark.parametrize(
    "timestamp",
    ["2026-07-26T12:00:00", "2026-07-26T12:00:00+02:00", "not-a-timestamp", "2026-13-01T00:00:00Z"],
)
def test_verified_at_must_be_explicit_utc(timestamp: str) -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match="verified_at_utc"):
        _receipt(verified_at_utc=timestamp)


def test_commit_must_be_a_full_lowercase_sha1() -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match="commit"):
        _receipt(commit="c" * 39)


def test_receipt_requires_at_least_one_source() -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match="source count"):
        _receipt(sources=())


def test_duplicate_source_paths_are_refused() -> None:
    """Two rows for one path make the receipt's byte accounting ambiguous."""
    with pytest.raises(CrossRepositoryProvenanceError, match="paths must be unique"):
        _receipt(sources=(_source(), _source()))


def test_declared_total_bytes_are_bounded() -> None:
    sources = tuple(
        _source(path=f"docs/f{index}.md", bytes=MAX_EXTERNAL_SOURCE_BYTES) for index in range(8)
    )
    with pytest.raises(CrossRepositoryProvenanceError, match="declared source bytes"):
        _receipt(sources=sources)


# ---------------------------------------------------------------------------
# Receipt document parsing
# ---------------------------------------------------------------------------


def _document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": "1.0.0",
        "repository": "srgangaram-swe/Signalattice",
        "origin_url": "https://github.com/srgangaram-swe/Signalattice.git",
        "commit": "c" * 40,
        "verification": {
            "mode": "local_git_object_database",
            "network_requests": 0,
            "verified_at_utc": "2026-07-26T12:00:00Z",
        },
        "sources": [
            {
                "family": "spectral",
                "path": "docs/spectral_features.md",
                "git_blob_sha1": BLOB,
                "bytes": 128,
                "sha256": DIGEST,
                "claim_scope": "descriptor definitions",
            }
        ],
    }
    document.update(overrides)
    return document


def _write(tmp_path: Path, document: object) -> Path:
    target = tmp_path / "receipt.json"
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def test_a_well_formed_receipt_document_loads(tmp_path: Path) -> None:
    receipt = load_cross_repository_receipt(_write(tmp_path, _document()))
    assert receipt.commit == "c" * 40
    assert receipt.sources[0].family == "spectral"


def test_committed_sprint_3_receipt_loads() -> None:
    """The receipt this MR actually publishes must satisfy its own parser."""
    receipt = load_cross_repository_receipt(
        Path("docs/evidence/signal_foundry_sprint_3/cross_repository_provenance.json")
    )
    assert receipt.repository == "srgangaram-swe/Signalattice"
    assert receipt.sources


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "root must be an object"),
        (_document(schema_version="9.9.9"), "schema_version"),
        (_document(verification=[]), "verification must be an object"),
        (
            _document(
                verification={
                    "mode": "trust_me",
                    "network_requests": 0,
                    "verified_at_utc": "2026-07-26T12:00:00Z",
                }
            ),
            "verification mode",
        ),
        (
            _document(
                verification={
                    "mode": "local_git_object_database",
                    "network_requests": 3,
                    "verified_at_utc": "2026-07-26T12:00:00Z",
                }
            ),
            "network-free",
        ),
        (_document(sources={}), "sources must be an array"),
        (_document(sources=[]), "source count"),
        (_document(sources=["not-an-object"]), r"sources\[0\] must be an object"),
    ],
)
def test_malformed_receipt_documents_fail_closed(
    tmp_path: Path, document: object, message: str
) -> None:
    with pytest.raises(CrossRepositoryProvenanceError, match=message):
        load_cross_repository_receipt(_write(tmp_path, document))


def test_unexpected_or_missing_receipt_fields_are_named(tmp_path: Path) -> None:
    """An exact-field check names both sides so a drifted schema is diagnosable."""
    extra = _document()
    extra["unexpected"] = 1
    with pytest.raises(CrossRepositoryProvenanceError, match="extra=.*unexpected"):
        load_cross_repository_receipt(_write(tmp_path, extra))

    missing = _document()
    del missing["commit"]
    with pytest.raises(CrossRepositoryProvenanceError, match="missing=.*commit"):
        load_cross_repository_receipt(_write(tmp_path, missing))


def test_non_json_receipt_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"{not json")
    with pytest.raises(CrossRepositoryProvenanceError, match="strict bounded UTF-8 JSON"):
        load_cross_repository_receipt(target)


# ---------------------------------------------------------------------------
# Verification against a real Git object database
# ---------------------------------------------------------------------------


def _git(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        },
    )
    return result.stdout.strip()


@pytest.fixture
def external_checkout(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """A real second repository holding one committed evidence blob."""
    checkout = tmp_path / "external"
    checkout.mkdir()
    _git(checkout, "init", "--quiet", "--initial-branch", "main")
    _git(checkout, "remote", "add", "origin", "https://github.com/srgangaram-swe/Signalattice.git")
    payload = b"# Spectral features\n\nCausal descriptors.\n"
    (checkout / "docs").mkdir()
    (checkout / "docs" / "spectral_features.md").write_bytes(payload)
    _git(checkout, "add", "docs/spectral_features.md")
    _git(checkout, "commit", "--quiet", "-m", "add evidence")
    commit = _git(checkout, "rev-parse", "HEAD")
    blob = _git(checkout, "rev-parse", "HEAD:docs/spectral_features.md")
    source = {
        "family": "spectral",
        "path": "docs/spectral_features.md",
        "git_blob_sha1": blob,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "claim_scope": "descriptor definitions",
    }
    return checkout, _document(commit=commit, sources=[source])


def test_verification_reads_only_the_pinned_commit(
    tmp_path: Path, external_checkout: tuple[Path, dict[str, object]]
) -> None:
    checkout, document = external_checkout
    receipt = _write(tmp_path, document)
    result = verify_cross_repository_receipt(receipt, checkout=checkout)
    assert result.repository == "srgangaram-swe/Signalattice"
    assert result.source_count == 1
    assert result.total_bytes == document["sources"][0]["bytes"]  # type: ignore[index]

    # The worktree is irrelevant: only objects reachable from the commit matter.
    (checkout / "docs" / "spectral_features.md").write_bytes(b"tampered after commit\n")
    again = verify_cross_repository_receipt(receipt, checkout=checkout)
    assert again.receipt_sha256 == result.receipt_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("git_blob_sha1", "d" * 40, "Git blob mismatch"),
        ("bytes", 9_999, "byte-size mismatch|tree metadata|resolve exactly once"),
        ("sha256", "e" * 64, "SHA-256 mismatch"),
        ("path", "docs/absent.md", "resolve exactly once"),
    ],
)
def test_verification_detects_a_tampered_source_row(
    tmp_path: Path,
    external_checkout: tuple[Path, dict[str, object]],
    field: str,
    value: object,
    message: str,
) -> None:
    checkout, document = external_checkout
    sources = [dict(document["sources"][0])]  # type: ignore[index]
    sources[0][field] = value
    receipt = _write(tmp_path, {**document, "sources": sources})
    with pytest.raises(CrossRepositoryProvenanceError, match=message):
        verify_cross_repository_receipt(receipt, checkout=checkout)


def test_verification_rejects_a_foreign_commit(
    tmp_path: Path, external_checkout: tuple[Path, dict[str, object]]
) -> None:
    checkout, document = external_checkout
    receipt = _write(tmp_path, {**document, "commit": "f" * 40})
    with pytest.raises(
        CrossRepositoryProvenanceError, match="exact commit object|rejected pinned evidence"
    ):
        verify_cross_repository_receipt(receipt, checkout=checkout)


def test_verification_rejects_an_origin_mismatch(
    tmp_path: Path, external_checkout: tuple[Path, dict[str, object]]
) -> None:
    checkout, document = external_checkout
    _git(checkout, "remote", "set-url", "origin", "https://github.com/srgangaram-swe/Other.git")
    with pytest.raises(CrossRepositoryProvenanceError, match="origin mismatch"):
        verify_cross_repository_receipt(_write(tmp_path, document), checkout=checkout)


def test_verification_requires_a_real_git_worktree(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    receipt = _write(tmp_path, _document())
    with pytest.raises(CrossRepositoryProvenanceError, match="worktree|rejected pinned evidence"):
        verify_cross_repository_receipt(receipt, checkout=plain)

    with pytest.raises(CrossRepositoryProvenanceError, match="non-symlink directory"):
        verify_cross_repository_receipt(receipt, checkout=tmp_path / "absent")
