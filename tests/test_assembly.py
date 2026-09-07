"""Real Git integration, exact-history preservation and hostile-input regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

import foundry_build.assembly as assembly_module
import foundry_build.context as context_module
import foundry_build.workflows as workflow_module
from foundry_build.assembly import (
    APPROVED_TREES,
    archive_ref,
    canonical,
    import_sources,
    read_json,
    resolve_findings,
    source_record,
    verify,
)
from foundry_build.context import prepare
from foundry_build.git import AssemblyError, git, oid, resolve
from foundry_build.workflows import generate, relocate

ROOT = Path(__file__).resolve().parents[1]


def _commit(root: Path, name: str, contents: str) -> str:
    (root / name).write_text(contents)
    git(root, "add", name)
    git(root, "commit", "-m", "test: record fixture")
    return resolve(root, "HEAD")


def _ledger(mirror: Path, source: str, destination: Path) -> None:
    """Minimal test ledger, derived from real fixture objects, never public evidence."""
    refs = dict(
        line.split(" ", 1)
        for line in git(mirror, "for-each-ref", "--format=%(refname) %(objectname)")
        .decode()
        .splitlines()
    )
    objects = {}
    for value in (
        git(mirror, "rev-list", "--objects", "--no-object-names", "--all")
        .decode()
        .splitlines()
    ):
        kind = git(mirror, "cat-file", "-t", value).decode().strip()
        payload = git(mirror, "cat-file", kind, value)
        objects[value] = {
            "kind": kind,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    fields = {"source": source, "refs": refs, "objects": objects, "blockers": []}
    destination.mkdir()
    shards = []
    for field, content in fields.items():
        name = f"{field}-000000.json"
        payload = canonical(content)
        (destination / name).write_bytes(payload)
        shards.append(
            {
                "field": field,
                "path": name,
                "mapping": isinstance(content, dict),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    compact = (
        json.dumps(fields, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (destination / "manifest.json").write_bytes(
        canonical(
            {"shards": shards, "ledger_sha256": hashlib.sha256(compact).hexdigest()}
        )
    )


@pytest.fixture
def frozen(tmp_path: Path) -> tuple[Path, Path]:
    backup = tmp_path / "backup"
    backup.mkdir()
    for source in ("alphaforge", "signalattice"):
        original = tmp_path / source
        original.mkdir()
        git(original, "init", "-b", "dev")
        _commit(original, "LICENSE", "Test fixture notice\n")
        _commit(original, "module.py", "VALUE = 7\n")
        git(original, "tag", "-a", "v0.1", "-m", "signed-content identity fixture")
        git(original, "switch", "-c", "abandoned")
        _commit(original, "abandoned.py", "VALUE = 99\n")
        git(original, "switch", "dev")
        mirror = backup / f"{source}.git"
        git(tmp_path, "clone", "--mirror", str(original), str(mirror))
        _ledger(mirror, source, backup / f"{source}-ledger")
    destination = tmp_path / "unified"
    destination.mkdir()
    git(destination, "init", "-b", "feat/assembly-test")
    _commit(destination, "README.md", "Fixture bootstrap\n")
    return destination, backup


@pytest.fixture
def imported(frozen: tuple[Path, Path]) -> tuple[Path, dict[str, Any]]:
    root, backup = frozen
    return root, import_sources(root, backup)


def test_real_import_ancestry_tags_blobs_and_original_refs(
    frozen: tuple[Path, Path],
) -> None:
    root, backup = frozen
    before = {
        source: git(backup / f"{source}.git", "show-ref")
        for source in ("alphaforge", "signalattice")
    }
    manifest = import_sources(root, backup)
    report = verify(root, manifest)
    for source in before:
        mirror = backup / f"{source}.git"
        assert git(mirror, "show-ref") == before[source]
        assert report[source]["commits"] == 3
        assert report[source]["active_blockers"] == 0
        assert not (root / "packages" / source / "abandoned.py").exists()
        assert resolve(root, f"refs/tags/{source}/v0.1") == resolve(mirror, "v0.1")
    # Verify an independent clean-room clone, not merely the producing worktree.
    clone = root.parent / "clean-room"
    git(root.parent, "clone", "--no-local", str(root), str(clone))
    assert verify(clone, manifest) == report


@pytest.mark.parametrize("tree", sorted(APPROVED_TREES))
def test_owner_determination_is_exact(tree: str) -> None:
    finding = {"code": "missing-license", "tree": tree}
    assert resolve_findings("alphaforge", [finding]) == [finding]
    with pytest.raises(AssemblyError, match="duplicate"):
        resolve_findings("alphaforge", [finding, finding])
    with pytest.raises(AssemblyError, match="unresolved"):
        resolve_findings("signalattice", [finding])


@pytest.mark.parametrize(
    "finding",
    [
        {"code": "secret-marker", "tree": next(iter(APPROVED_TREES))},
        {"code": "missing-license", "tree": "0" * 40},
        {"code": "missing-license", "tree": next(iter(APPROVED_TREES)), "path": "x"},
        {},
    ],
)
def test_other_findings_never_get_waived(finding: dict[str, str]) -> None:
    with pytest.raises(AssemblyError, match="unresolved"):
        resolve_findings("alphaforge", [finding])


@pytest.mark.parametrize("ref", ["refs/heads/dev", "refs/pull/2/head", "refs/tags/v1"])
def test_namespaced_mapping(ref: str) -> None:
    assert archive_ref("alphaforge", ref).startswith("refs/tags/")


@pytest.mark.parametrize("ref", ["HEAD", "refs/heads/../x", "refs/heads/a b"])
def test_bad_ref(ref: str) -> None:
    with pytest.raises(AssemblyError):
        archive_ref("alphaforge", ref)


@pytest.mark.parametrize("payload", [b"{", b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff"])
def test_json_ambiguity_fails_closed(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(payload)
    with pytest.raises(AssemblyError):
        read_json(path)


def test_input_limits_and_nonregular_files(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(canonical({"x": 1}))
    assert read_json(path) == {"x": 1}
    with pytest.raises(AssemblyError):
        read_json(path, 1)
    link = tmp_path / "link"
    link.symlink_to(path)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for invalid in (link, fifo, tmp_path, tmp_path / "missing"):
        with pytest.raises(AssemblyError):
            read_json(invalid)


def test_ledger_hash_corruption_is_rejected(frozen: tuple[Path, Path]) -> None:
    _, backup = frozen
    directory = backup / "alphaforge-ledger"
    (directory / "source-000000.json").write_bytes(canonical("signalattice"))
    with pytest.raises(AssemblyError, match="hash-mismatch"):
        source_record(backup / "alphaforge.git", "alphaforge", directory)


def test_protected_branch_or_dirty_import_refused(frozen: tuple[Path, Path]) -> None:
    root, backup = frozen
    (root / "README.md").write_text("dirty\n")
    with pytest.raises(AssemblyError, match="dirty"):
        import_sources(root, backup)
    git(root, "switch", "-c", "dev")
    with pytest.raises(AssemblyError, match="unsafe-import"):
        import_sources(root, backup)


@pytest.mark.parametrize(
    "change", ["commit", "tree", "object", "mapping", "blocker", "schema", "kind"]
)
def test_manifest_tampering_is_rejected(
    imported: tuple[Path, dict[str, Any]], change: str
) -> None:
    root, manifest = imported
    altered = copy.deepcopy(manifest)
    record = altered["sources"]["alphaforge"]
    if change == "commit":
        record["commits"].append("0" * 40)
    elif change == "tree":
        record["tree"] = "0" * 40
    elif change == "object":
        next(iter(record["objects"].values()))["sha256"] = "0" * 64
    elif change == "mapping":
        next(iter(record["refs"].values()))["target"] = "refs/tags/foreign"
    elif change == "blocker":
        record["active_blockers"] = [{"code": "secret-marker"}]
    elif change == "kind":
        next(iter(record["objects"].values()))["kind"] = "--filters"
    else:
        altered["schema_version"] = 99
    with pytest.raises(AssemblyError):
        verify(root, altered)


def test_context_is_offline_relocated_and_repeatable(
    imported: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = imported
    for source in ("alphaforge", "signalattice"):
        package = prepare(root, source, manifest)
        assert prepare(root, source, manifest) == package
        assert (
            Path(git(package, "rev-parse", "--show-toplevel").decode().strip())
            == package
        )
        assert resolve(package, "HEAD") == manifest["sources"][source]["dev"]
        assert git(package, "show", "HEAD:module.py") == b"VALUE = 7\n"
        assert git(package, "status", "--porcelain") == b""
        assert (package / ".git").is_file()
        assert not (package / ".git").is_symlink()


def test_context_never_overwrites_user_marker(
    imported: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = imported
    marker = root / "packages/alphaforge/.git"
    marker.write_text("existing user state\n")
    with pytest.raises(AssemblyError, match="foreign"):
        prepare(root, "alphaforge", manifest)
    assert marker.read_text() == "existing user state\n"


def test_context_refuses_dirty_source(imported: tuple[Path, dict[str, Any]]) -> None:
    root, manifest = imported
    (root / "packages/alphaforge/module.py").write_text("VALUE = 9\n")
    with pytest.raises(AssemblyError, match="dirty-package"):
        prepare(root, "alphaforge", manifest)


def test_git_bounds_and_sanitized_errors(tmp_path: Path) -> None:
    git(tmp_path, "init", "-b", "dev")
    with pytest.raises(AssemblyError, match="output-limit"):
        git(tmp_path, "--version", ceiling=1)
    with pytest.raises(AssemblyError, match="timeout"):
        git(tmp_path, "--version", seconds=0)
    with pytest.raises(AssemblyError, match="command-failed"):
        git(tmp_path, "rev-parse", "private-payload-never-echoed")
    with pytest.raises(AssemblyError, match="unavailable"):
        git(tmp_path / "missing", "status")
    with pytest.raises(AssemblyError, match="invalid-object"):
        oid("abc")


def test_root_workflows_preserve_required_gates_and_no_publication() -> None:
    workflow = generate(ROOT)
    jobs = workflow["jobs"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "signalattice-publish" not in jobs
    assert "signalattice-service-supply-chain" in jobs
    assert "signalattice-service-supply-chain" not in jobs["signalattice"]["needs"]
    assert "signalattice-dry-run" in jobs["signalattice"]["needs"]
    for source in ("alphaforge", "signalattice"):
        for gate in (
            "test",
            "quality",
            "dependency-review",
            "dependency-audit",
            "secret-scan",
        ):
            assert f"{source}-{gate}" in jobs[source]["needs"]
        assert jobs[source]["if"] == "always()"
    text = canonical(workflow).decode()
    assert "--cov-fail-under=78" in text
    assert "--cov-fail-under=80" in text
    assert "--cov-fail-under=90" in text
    assert "--require-hashes" in text
    for job in jobs.values():
        assert 0 < job["timeout-minutes"] <= 45
        assert "environment" not in job
        for step in job["steps"]:
            if "uses" in step:
                assert len(step["uses"].rsplit("@", 1)[1]) == 40


def test_published_ledgers_bind_every_frozen_path_and_object() -> None:
    assembly = read_json(ROOT / "provenance/assembly.json")
    for source, record in assembly["sources"].items():
        directory = ROOT / "provenance/ledgers" / source
        manifest = read_json(directory / "manifest.json")
        assert manifest["ledger_sha256"] == record["ledger_sha256"]
        fields: dict[str, Any] = {}
        for shard in manifest["shards"]:
            assert Path(shard["path"]).name == shard["path"]
            path = directory / shard["path"]
            payload = path.read_bytes()
            assert len(payload) < 950_000
            assert hashlib.sha256(payload).hexdigest() == shard["sha256"]
            data = read_json(path)
            if shard["mapping"]:
                existing = fields.setdefault(shard["field"], {})
                assert not set(existing) & set(data)
                existing.update(data)
            else:
                assert shard["field"] not in fields
                fields[shard["field"]] = data
        compact = (
            json.dumps(fields, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        assert hashlib.sha256(compact).hexdigest() == record["ledger_sha256"]
        assert fields["objects"] == record["objects"]
        assert fields["refs"] == {
            ref: row["oid"] for ref, row in record["refs"].items()
        }
        assert (
            resolve_findings(source, fields["blockers"]) == record["resolved_findings"]
        )
        rows = fields["trees"][record["tree"]]
        actual = git(ROOT, "ls-tree", "-rz", record["tree"]).split(b"\0")
        assert len(rows) == len([entry for entry in actual if entry])
        for row in rows:
            assert row["target"] == f"packages/{source}/{row['path']}"
            assert row["oid"] in fields["objects"]


@pytest.mark.parametrize("source", ["alphaforge", "signalattice"])
def test_github_snapshot_matches_source_freeze(source: str) -> None:
    directory = ROOT / "provenance/github"
    hashes = read_json(directory / "manifest.json")
    path = directory / f"{source}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
    captured = read_json(path)
    record = read_json(ROOT / "provenance/assembly.json")["sources"][source]
    assert captured["advertised_refs"] == {
        ref: row["oid"] for ref, row in record["refs"].items()
    }
    assert captured["github"]["repository"] == record["repository"]
    assert (
        captured["github"]["source_url"] == f"https://github.com/{record['repository']}"
    )
    assert all("body" not in row for row in captured["github"]["discussions"])


def test_job_relocation_does_not_mutate_original() -> None:
    original = {
        "runs-on": "ubuntu-latest",
        "steps": [{"run": "pytest --strict-markers"}],
    }
    saved = copy.deepcopy(original)
    relocated = relocate("alphaforge", original)
    assert original == saved
    assert relocated["steps"] == original["steps"]
    assert relocated["defaults"]["run"]["working-directory"] == "packages/alphaforge"


@pytest.mark.parametrize("distribution", ["wheel", "sdist", "service-wheel"])
def test_relocation_preserves_absolute_distribution_interpreters(
    distribution: str,
) -> None:
    interpreter = f"/tmp/signalattice-{distribution}/bin/python"
    commands = (
        f"{interpreter} -m pip install --upgrade pip\n"
        f"{interpreter} -m pip install dist/example.whl\n"
        f"{interpreter} -m pip check\n"
    )
    result = relocate("signalattice", {"steps": [{"run": commands}]})
    assert result["steps"][0]["run"] == commands


def test_relocation_changes_only_exact_bootstrap_lines() -> None:
    commands = (
        "python -m pip install --upgrade pip\n"
        'python -m pip install -e ".[dev]"\n'
        "python -m pip install ./dist/example.whl\n"
    )
    result = relocate("signalattice", {"steps": [{"run": commands}]})
    assert result["steps"][0]["run"] == (
        "uv sync --locked --extra dev\n" "python -m pip install ./dist/example.whl\n"
    )


@pytest.mark.parametrize("case", ["path", "identity", "duplicate", "source", "refs"])
def test_ledger_boundary_failures(frozen: tuple[Path, Path], case: str) -> None:
    _, backup = frozen
    directory = backup / "alphaforge-ledger"
    path = directory / "manifest.json"
    manifest = read_json(path)
    if case == "path":
        manifest["shards"][0]["path"] = "../outside"
    elif case == "identity":
        manifest["ledger_sha256"] = "0" * 64
    elif case == "duplicate":
        manifest["shards"].append(manifest["shards"][0])
    elif case == "refs":
        mirror = backup / "alphaforge.git"
        git(mirror, "update-ref", "refs/heads/new", resolve(mirror, "dev"))
    path.write_bytes(canonical(manifest))
    with pytest.raises(AssemblyError):
        source_record(
            backup / "alphaforge.git",
            "signalattice" if case == "source" else "alphaforge",
            directory,
        )


def test_missing_determination_and_changed_archive(
    imported: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = imported
    changed = copy.deepcopy(manifest)
    record = changed["sources"]["alphaforge"]
    record["resolved_findings"] = [
        {"code": "missing-license", "tree": next(iter(APPROVED_TREES))}
    ]
    with pytest.raises(AssemblyError, match="determination"):
        verify(root, changed)
    changed = copy.deepcopy(manifest)
    next(iter(changed["sources"]["alphaforge"]["refs"].values()))["oid"] = "0" * 40
    with pytest.raises(AssemblyError, match="archive-ref"):
        verify(root, changed)


@pytest.mark.parametrize("case", ["unknown", "root", "tree", "reservation", "orphan"])
def test_context_preconditions(
    imported: tuple[Path, dict[str, Any]], case: str
) -> None:
    root, manifest = imported
    source = "alphaforge"
    parent = root / ".git/source-contexts"
    parent.mkdir()
    if case == "unknown":
        source = "foreign"
    elif case == "root":
        (root / "packages/alphaforge").rename(root / "displaced")
    elif case == "tree":
        manifest["sources"][source]["tree"] = "0" * 40
    elif case == "reservation":
        (parent / ".alphaforge.reservation").mkdir()
    else:
        (parent / "alphaforge.git").mkdir()
    with pytest.raises(AssemblyError):
        prepare(root, source, manifest)


def test_context_stale_state_is_not_silently_refreshed(
    imported: tuple[Path, dict[str, Any]],
) -> None:
    root, manifest = imported
    package = prepare(root, "alphaforge", manifest)
    marker = root / ".git/source-contexts/alphaforge.git/foundry-identity"
    original = marker.read_bytes()
    marker.write_text("stale\n")
    with pytest.raises(AssemblyError, match="stale"):
        prepare(root, "alphaforge", manifest)
    marker.write_bytes(original)
    git(package, "symbolic-ref", "HEAD", "refs/heads/abandoned")
    with pytest.raises(AssemblyError, match="head-drift"):
        prepare(root, "alphaforge", manifest)


def test_context_io_fault_cleans_owned_stage(
    imported: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = imported

    def fail(*args: Any) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(context_module.os, "rename", fail)
    with pytest.raises(AssemblyError, match="io-failure"):
        prepare(root, "alphaforge", manifest)
    assert list((root / ".git/source-contexts").iterdir()) == []
    assert not (root / "packages/alphaforge/.git").exists()


def test_public_assembly_and_context_commands(
    frozen: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, backup = frozen
    monkeypatch.setattr(
        assembly_module, "__file__", str(root / "foundry_build/assembly.py")
    )
    monkeypatch.setattr(sys, "argv", ["assembly", "import", "--backup", str(backup)])
    assert assembly_module.main() == 0
    monkeypatch.setattr(sys, "argv", ["assembly", "verify"])
    assert assembly_module.main() == 0
    monkeypatch.setattr(sys, "argv", ["assembly", "import"])
    with pytest.raises(SystemExit) as error:
        assembly_module.main()
    assert error.value.code == 2
    monkeypatch.setattr(
        context_module, "__file__", str(root / "foundry_build/context.py")
    )
    monkeypatch.setattr(sys, "argv", ["context", "alphaforge"])
    assert context_module.main() == 0
    (root / "provenance/assembly.json").write_text("invalid")
    with pytest.raises(SystemExit) as error:
        context_module.main()
    assert error.value.code == 2


def test_workflow_generation_and_drift_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = generate(ROOT)
    monkeypatch.setattr(
        workflow_module, "__file__", str(tmp_path / "foundry_build/workflows.py")
    )
    monkeypatch.setattr(workflow_module, "generate", lambda root: workflow)
    monkeypatch.setattr(sys, "argv", ["workflows"])
    assert workflow_module.main() == 0
    monkeypatch.setattr(sys, "argv", ["workflows", "--check"])
    assert workflow_module.main() == 0
    (tmp_path / ".github/workflows/qualification.yml").write_text("drift")
    with pytest.raises(SystemExit) as error:
        workflow_module.main()
    assert error.value.code == 2
