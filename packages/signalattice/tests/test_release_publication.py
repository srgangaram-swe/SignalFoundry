"""Publication regressions: signed Git objects and complete downloaded bytes."""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZipInfo

import pytest
import yaml

from quant_platform.release.archive import (
    archive_inventory,
    pack_stage,
    verify_archive,
    verify_downloads,
)
from quant_platform.release.inventory import InventoryError, build_inventory

ROOT = Path(__file__).resolve().parents[1]


def _release_script() -> Any:
    spec = importlib.util.spec_from_file_location("release_script", ROOT / "scripts/release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    result = tmp_path / "stage"
    (result / "console/assets").mkdir(parents=True)
    (result / "console/assets/app.js").write_text("console.log('fixture')\n")
    (result / "release-descriptor.json").write_text('{"build_kind":"publication"}\n')
    (result / "LICENSE").write_text("MIT\n")
    return result


def test_complete_archive_is_reproducible_and_includes_nested_assets(stage: Path) -> None:
    first, second = stage.parent / "a.zip", stage.parent / "b.zip"
    pack_stage(stage, first)
    pack_stage(stage, second)
    assert first.read_bytes() == second.read_bytes()
    assert archive_inventory(first) == build_inventory(stage)
    verify_archive(stage, first)
    with pytest.raises(FileExistsError):
        pack_stage(stage, first)


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "renamed"])
def test_archive_refuses_every_member_set_or_content_change(stage: Path, mutation: str) -> None:
    archive = stage.parent / "release.zip"
    pack_stage(stage, archive)
    target = stage / "LICENSE"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (stage / "extra").write_text("unexpected")
    elif mutation == "changed":
        target.write_text("bit flip")
    else:
        target.rename(stage / "license-renamed")
    with pytest.raises(InventoryError, match="differs"):
        verify_archive(stage, archive)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "a/./b", "a//b"])
def test_untrusted_zip_paths_are_rejected_without_extraction(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "hostile.zip"
    with ZipFile(archive, "w") as output:
        entry = ZipInfo(name)
        entry.external_attr = (stat.S_IFREG | 0o644) << 16
        output.writestr(entry, "hostile")
    with pytest.raises(InventoryError, match="unsafe"):
        archive_inventory(archive)


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFDIR, stat.S_IFIFO])
def test_nonregular_members_are_rejected(tmp_path: Path, kind: int) -> None:
    archive = tmp_path / "hostile.zip"
    with ZipFile(archive, "w") as output:
        entry = ZipInfo("member")
        entry.external_attr = (kind | 0o644) << 16
        output.writestr(entry, "target")
    with pytest.raises(InventoryError, match="unsafe"):
        archive_inventory(archive)


def test_symlink_directories_and_recursive_output_are_rejected(stage: Path) -> None:
    with pytest.raises(InventoryError, match="outside"):
        pack_stage(stage, stage / "recursive.zip")
    (stage / "linked").symlink_to(stage / "console", target_is_directory=True)
    with pytest.raises(InventoryError, match="symlink"):
        pack_stage(stage, stage.parent / "archive.zip")


def test_download_verification_checks_bytes_not_just_asset_count(tmp_path: Path) -> None:
    expected, downloaded = tmp_path / "expected", tmp_path / "downloaded"
    expected.mkdir()
    downloaded.mkdir()
    (expected / "wheel").write_bytes(b"good")
    (downloaded / "wheel").write_bytes(b"evil")
    with pytest.raises(InventoryError, match="modified"):
        verify_downloads(expected, downloaded)
    (downloaded / "wheel").write_bytes(b"good")
    verify_downloads(expected, downloaded)


def test_unsigned_and_wrong_key_tags_fail_real_git_verification(tmp_path: Path) -> None:
    def run(*args: str, succeeds: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=20)
        assert (result.returncode == 0) is succeeds, result.stderr[-500:]
        return result

    run("git", "init", "-b", "main")
    run("git", "config", "user.name", "srgangaram-swe")
    run("git", "config", "user.email", "srgangaram-swe@users.noreply.github.com")
    run("git", "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "fixture")
    run("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "key"))
    allowed = tmp_path / "allowed"
    allowed.write_text("srgangaram-swe " + (tmp_path / "key.pub").read_text())
    options = ("-c", "gpg.format=ssh", "-c", f"gpg.ssh.allowedSignersFile={allowed}")
    run("git", "-c", "tag.gpgsign=false", "tag", "-a", "unsigned", "-m", "unsigned")
    run("git", *options, "verify-tag", "unsigned", succeeds=False)
    run(
        "git",
        *options,
        "-c",
        f"user.signingkey={tmp_path / 'key'}",
        "tag",
        "-s",
        "signed",
        "-m",
        "signed",
    )
    run("git", *options, "verify-tag", "signed")
    allowed.write_text("")
    run("git", *options, "verify-tag", "signed", succeeds=False)


def test_verifier_binds_descriptor_to_actual_checkout(
    stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _release_script()
    # Fail before inventory/provenance use with a complete independently valid descriptor.
    from tests.test_release_machinery import _descriptor

    (stage / release.DESCRIPTOR_NAME).write_text(_descriptor().model_dump_json())
    (stage / release.INVENTORY_NAME).write_text("[]")
    (stage / release.PROVENANCE_NAME).write_text("{}")
    monkeypatch.setattr(release, "_git", lambda *args: "b" * 40)
    assert release.verify(ROOT, stage) == release.EXIT_REFUSED


def test_release_workflow_keeps_authority_and_verification_boundaries() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    dry = workflow["jobs"]["dry-run"]
    publish = workflow["jobs"]["publish"]
    assert dry["permissions"] == {"contents": "read"}
    assert "environment" not in dry
    assert publish["environment"] == "release"
    steps = publish["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "scripts/release.py publication --staging" in commands
    assert "scripts/release.py publication-gate" in commands
    assert 'tag -s "$RELEASE_TAG"' in commands
    assert "credential.helper=!gh auth git-credential" in commands
    assert "verify-downloads" in commands and "verify-archive" in commands
    assert "--source-digest" in commands and "--signer-workflow" in commands
    assert commands.index("verify-downloads") < commands.index("--draft=false")
    attestation = next(step for step in steps if "attest-build-provenance@" in step.get("uses", ""))
    assert attestation["with"]["subject-path"].strip() == "build/assets/*"
    assert all("${{ inputs." not in step.get("run", "") for step in steps)


def test_corrupt_zip_is_a_domain_refusal(tmp_path: Path) -> None:
    archive = tmp_path / "corrupt.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(InventoryError, match="invalid"):
        archive_inventory(archive)


def test_transport_cli_roundtrip(stage: Path) -> None:
    import sys

    archive = stage.parent / "complete.zip"
    for command in ("pack", "verify-archive"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/release_transport.py"),
                command,
                str(stage),
                str(archive),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
    assert (
        json.loads((stage / "release-descriptor.json").read_text())["build_kind"] == "publication"
    )


@pytest.mark.parametrize("kind", ["dry-run", "publication"])
def test_build_identity_is_chosen_before_artifacts_are_described(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    release = _release_script()
    destination = tmp_path / "new-stage"
    monkeypatch.setattr(release, "_source_commit", lambda root: "a" * 40)
    monkeypatch.setattr(release, "_git", lambda *args: "1700000000")
    monkeypatch.setattr(release, "_build_console", lambda *args: True)
    monkeypatch.setattr(release, "_copy_schemas_and_evidence", lambda *args: None)
    monkeypatch.setattr(release, "verify", lambda *args: 0)

    def fake_distributions(root: Path, stage: Path, env: object, **kwargs: object) -> None:
        (stage / "python").mkdir()
        (stage / "python/pkg.whl").write_bytes(b"fixture wheel")
        (stage / "python/pkg.tar.gz").write_bytes(b"fixture sdist")

    monkeypatch.setattr(release, "_build_python_distributions", fake_distributions)
    assert release.dry_run(ROOT, destination, release_version=None, build_kind=kind) == 0
    descriptor = json.loads((destination / release.DESCRIPTOR_NAME).read_text())
    provenance = json.loads((destination / release.PROVENANCE_NAME).read_text())
    assert descriptor["build_kind"] == kind
    assert provenance["predicate"]["runDetails"]["builder"]["id"].endswith("/" + kind)
    assert release.dry_run(ROOT, destination, release_version=None, build_kind=kind) == 2


@pytest.mark.parametrize("failure", [None, "dry-run", "not-main", "single-parent", "stale"])
def test_publication_gate_checks_the_observed_repository(
    stage: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from tests.test_release_machinery import _descriptor

    release = _release_script()
    descriptor = _descriptor(build_kind="dry-run" if failure == "dry-run" else "publication")
    (stage / release.DESCRIPTOR_NAME).write_text(descriptor.model_dump_json())
    monkeypatch.setattr(release, "verify", lambda *args: 0)
    monkeypatch.setattr(release, "_source_commit", lambda *args: "a" * 40)

    def git(root: Path, *args: str) -> str:
        if args[0] == "branch":
            return "dev" if failure == "not-main" else "main"
        if args[0] == "rev-parse":
            return "b" * 40 if failure == "stale" and args[1] == "origin/main" else "a" * 40
        if args[0] == "show":
            return "a" * 40 if failure == "single-parent" else "a" * 40 + " " + "b" * 40
        if args[0] == "rev-list":
            return "a" * 40
        return ""

    monkeypatch.setattr(release, "_git", git)
    assert release.publication_gate(ROOT, stage, "a" * 40) == (0 if failure is None else 2)
