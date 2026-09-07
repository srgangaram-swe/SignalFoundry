"""Offline adversarial contracts and a real full-history scratch-import drill."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.preservation.contracts import findings, static_interfaces, validate_paths
from scripts.preservation.git import Git, Limits, PreservationError, bounded_run, oid
from scripts.preservation.github import capture
from scripts.preservation.inventory import canonical, inventory, require_migration_clear, sha256
from scripts.preservation.publish import publish, read_ledger


def test_category_and_conditional_declarations() -> None:
    from scripts.preservation.contracts import category

    cases = {
        ".github/workflows/ci.yml": "workflow",
        "tests/test.py": "test",
        "reports/figure.png": "evidence",
        "docs/guide.md": "documentation",
        "apps/main.tsx": "gui-api",
        "configs/run.yaml": "configuration",
        "scripts/run.py": "cli-tooling",
        "package/model.py": "package",
        "LICENSE": "other-preserved",
    }
    assert {path: category(path) for path in cases} == cases
    source = b"if True:\n    X: int = 1\nelse:\n    X = 2\ntry:\n    import os\nexcept ImportError:\n    Y = 1\nfinally:\n    Z = 2\n"
    assert {row["name"] for row in static_interfaces("x.py", source, 100)} == {"X", "os", "Y", "Z"}


def command(root: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=srgangaram-swe",
            "-c",
            "user.email=srgangaram-swe@users.noreply.github.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=15,
    )
    return result.stdout.decode().strip()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    command(root, "init", "-b", "main")
    (root / "LICENSE").write_text("test-reviewed-license\n")
    (root / "module.py").write_text("VALUE = 1\ndef run(x: int) -> int:\n    return x + VALUE\n")
    (root / "tool.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "tool.sh").chmod(0o755)
    command(root, "add", ".")
    command(root, "commit", "-m", "fixture: initial")
    command(root, "branch", "dev")
    command(root, "branch", "prod")
    command(root, "tag", "-a", "v0.1.0", "-m", "fixture tag")
    command(root, "checkout", "-b", "unmerged")
    (root / "unmerged.py").write_text("UNMERGED = True\n")
    command(root, "add", ".")
    command(root, "commit", "-m", "fixture: unmerged history")
    command(root, "checkout", "main")
    return root


def ledger_for(root: Path, **kwargs: Any) -> dict[str, Any]:
    policy = {sha256((root / "LICENSE").read_bytes()): "fixture-only reviewed license"}
    return inventory(root, "alphaforge", policy, **kwargs)


def test_inventory_independent_git_and_determinism(source: Path) -> None:
    before = Git(source).refs()
    first = ledger_for(source)
    assert canonical(first) == canonical(ledger_for(source))
    assert first["refs"] == before == Git(source).refs()
    expected = set(
        command(source, "rev-list", "--all", "--objects", "--no-object-names").splitlines()
    )
    assert expected == set(first["objects"])
    for tree, rows in first["trees"].items():
        independent = command(source, "ls-tree", "-r", tree).splitlines()
        assert len(independent) == len(rows)
        for row in rows:
            content = subprocess.run(
                ["git", "cat-file", "blob", row["oid"]],
                cwd=source,
                check=True,
                capture_output=True,
                timeout=5,
            ).stdout
            assert row["sha256"] == hashlib.sha256(content).hexdigest()
            assert row["target"] == "packages/alphaforge/" + row["path"]
    require_migration_clear(first)
    assert first["summary"]["interface_records"] == 3


@pytest.mark.parametrize(
    "paths",
    [
        ["A", "a"],
        ["dir", "dir/file"],
        ["dir/file", "dir"],
        ["x", "x"],
        ["é.py", "e\u0301.py"],
        ["A/file", "a/other"],
        ["../x"],
        ["/absolute"],
        ["a//b"],
        ["a/./b"],
        ["a\\b"],
        ["a\nfile"],
        [""],
        ["x."],
        ["C:drive"],
    ],
)
def test_hostile_paths(paths: list[str]) -> None:
    with pytest.raises(PreservationError):
        validate_paths(paths)


def test_path_permutation_property() -> None:
    randomizer = random.Random(785)
    paths = [f"p{i % 7}/d{i % 11}/file-{i}.py" for i in range(300)]
    for _ in range(30):
        randomizer.shuffle(paths)
        validate_paths(paths)
    for _ in range(30):
        randomizer.shuffle(paths)
        with pytest.raises(PreservationError):
            validate_paths([*paths, paths[0].upper()])


@pytest.mark.parametrize(
    ("path", "content", "code"),
    [
        ("data/raw/input.csv", b"value", "raw-or-runtime-data"),
        (".env", b"x", "runtime-or-secret-path"),
        ("model.pt", b"x", "secret-or-model-artifact"),
        ("asset", b"version https://git-lfs.github.com/spec/v1\n", "lfs-external-object"),
        ("file", b"-----BEGIN " + b"PRIVATE KEY-----", "secret-marker"),
        ("file", b"ghp_" + b"A" * 36, "secret-marker"),
    ],
)
def test_findings_redact_payload(path: str, content: bytes, code: str) -> None:
    assert code in findings(path, content)
    assert content not in canonical(findings(path, content))


def test_static_interfaces_no_execution() -> None:
    content = b"""raise RuntimeError("must never execute")
from package import Export as Public
__all__ = ["Public"]
class Model:
    def fit(self): pass
    def _private(self): pass
@router.get("/v1/models")
def models(): pass
parser.add_argument("--seed")
router.include_router(dynamic_router)
data.get("not an API")
"""
    records = static_interfaces("module.py", content, 1000)
    assert {row["name"] for row in records} == {
        "Public",
        "__all__",
        "Model",
        "Model.fit",
        "models",
        "get",
        "add_argument",
        "include_router",
    }
    assert any(row.get("literals") == ["/v1/models"] for row in records)
    assert any(row.get("dynamic") for row in records)
    assert static_interfaces("notes.md", b"not python", 1) == []
    with pytest.raises(PreservationError):
        static_interfaces("x.py", b"a =", 100)
    with pytest.raises(PreservationError):
        static_interfaces("x.py", b"a = 1", 1)


def test_deep_receiver_does_not_recursively_format_source() -> None:
    content = ("a." * 600 + 'router.get("/v1/test")').encode()
    records = static_interfaces("deep.py", content, 10_000)
    assert len(records) == 1
    assert records[0]["literals"] == ["/v1/test"]


@pytest.mark.parametrize(
    "change", ["license", "missing-license", "symlink", "lfs", "secret", "raw"]
)
def test_fail_closed_inventory(source: Path, change: str) -> None:
    policy = {sha256((source / "LICENSE").read_bytes()): "reviewed fixture"}
    if change == "license":
        (source / "LICENSE").write_text("unreviewed license")
    elif change == "missing-license":
        (source / "LICENSE").unlink()
    elif change == "symlink":
        (source / "link").symlink_to("/outside")
    elif change == "lfs":
        (source / "pointer").write_text("version https://git-lfs.github.com/spec/v1\n")
    elif change == "secret":
        (source / "secret.txt").write_bytes(b"ghp_" + b"A" * 36)
    else:
        (source / "data/raw").mkdir(parents=True)
        (source / "data/raw/private.csv").write_text("must not publish")
    command(source, "add", "-A")
    command(source, "commit", "-m", "fixture: rejection")
    ledger = inventory(source, "alphaforge", policy)
    assert ledger["blockers"]
    assert b"must not publish" not in canonical(ledger)
    with pytest.raises(PreservationError, match="migration-blocked"):
        require_migration_clear(ledger)


@pytest.mark.parametrize("state", ["shallow", "replace", "alternates", "graft", "missing-object"])
def test_malformed_repository(source: Path, state: str) -> None:
    head = command(source, "rev-parse", "HEAD")
    if state == "shallow":
        (source / ".git/shallow").write_text(head + "\n")
    elif state == "replace":
        command(source, "update-ref", "refs/replace/" + head, head)
    elif state == "alternates":
        (source / ".git/objects/info/alternates").write_text("/not-an-object-store\n")
    elif state == "graft":
        (source / ".git/info/grafts").write_text(head + "\n")
    else:
        (source / ".git/objects" / head[:2] / head[2:]).unlink()
    with pytest.raises(PreservationError):
        ledger_for(source)


def test_limits_and_command_failures(source: Path) -> None:
    for value in (0, -1, float("inf"), float("nan"), True):
        with pytest.raises(PreservationError):
            Limits(command_seconds=value)
    with pytest.raises(PreservationError, match="object-count-limit"):
        ledger_for(source, limits=Limits(objects=1))
    with pytest.raises(PreservationError, match="object-byte-limit"):
        ledger_for(source, limits=Limits(blob_bytes=1))
    with pytest.raises(PreservationError, match="command-output-limit"):
        bounded_run([sys.executable, "-c", "print('x' * 10000)"], source, 5, 100)
    with pytest.raises(PreservationError, match="command-timeout"):
        bounded_run([sys.executable, "-c", "import time; time.sleep(2)"], source, 0.05, 100)
    with pytest.raises(PreservationError, match="process-unavailable"):
        bounded_run(["/does-not-exist"], source, 1, 100)
    with pytest.raises(PreservationError):
        oid("HEAD; arbitrary")


def test_publication_roundtrip_faults_and_cli(source: Path, tmp_path: Path) -> None:
    ledger = ledger_for(source)
    output = tmp_path / "ledger"
    publish(ledger, output)
    assert read_ledger(output) == ledger
    with pytest.raises(PreservationError):
        publish(ledger, output)
    second = tmp_path / "second"
    publish(ledger, second)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    result = subprocess.run(
        [sys.executable, "-m", "scripts.preservation", "verify", str(output)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    (output / "refs-000000.json").write_text("{}")
    with pytest.raises(PreservationError):
        read_ledger(output)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.preservation", "verify", str(output)],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert b"no import authorized" in result.stderr


def test_full_history_import_and_bundle_recovery_drill(source: Path, tmp_path: Path) -> None:
    """Independent Git plumbing proves the ADR on fixtures, never real sources."""
    original_refs = Git(source).refs()
    second = tmp_path / "independent-source"
    second.mkdir()
    command(second, "init", "-b", "main")
    (second / "LICENSE").write_text("independent fixture license\n")
    (second / "module.py").write_text("SECOND = True\n")
    command(second, "add", ".")
    command(second, "commit", "-m", "fixture: independent root")
    command(second, "tag", "-a", "v0.1.0", "-m", "independent tag with same name")
    second_refs = Git(second).refs()
    target = tmp_path / "target"
    target.mkdir()
    command(target, "init", "-b", "dev")
    command(target, "commit", "--allow-empty", "-m", "fixture: bootstrap")
    command(target, "fetch", str(source), "+refs/*:refs/archive/alphaforge/*")
    command(target, "fetch", str(second), "+refs/*:refs/archive/signalattice/*")
    command(
        target, "read-tree", "--prefix=packages/alphaforge/", "refs/archive/alphaforge/heads/main"
    )
    command(
        target,
        "read-tree",
        "--prefix=packages/signalattice/",
        "refs/archive/signalattice/heads/main",
    )
    tree = command(target, "write-tree")
    parents = sorted(
        {
            command(target, "rev-parse", f"{value}^{{commit}}")
            for value in [*original_refs.values(), *second_refs.values()]
        }
    )
    arguments = ["commit-tree", tree, "-p", "HEAD"]
    for parent in parents:
        arguments.extend(["-p", parent])
    merged = command(target, *arguments, "-m", "fixture: preserve source ancestry")
    command(target, "update-ref", "refs/heads/dev", merged)
    for parent in parents:
        command(target, "merge-base", "--is-ancestor", parent, "dev")
    assert command(target, "rev-parse", "dev:packages/alphaforge") == command(
        source, "rev-parse", "main^{tree}"
    )
    assert (
        command(target, "rev-parse", "refs/archive/alphaforge/tags/v0.1.0")
        == original_refs["refs/tags/v0.1.0"]
    )
    assert Git(source).refs() == original_refs
    assert Git(second).refs() == second_refs
    assert command(target, "rev-parse", "dev:packages/signalattice") == command(
        second, "rev-parse", "main^{tree}"
    )
    assert (
        command(target, "rev-parse", "refs/archive/signalattice/tags/v0.1.0")
        == second_refs["refs/tags/v0.1.0"]
    )
    bundle = tmp_path / "recovery.bundle"
    command(target, "bundle", "create", str(bundle), "--all")
    command(target, "bundle", "verify", str(bundle))
    recovered = tmp_path / "recovered.git"
    command(tmp_path, "clone", "--mirror", str(bundle), str(recovered))
    command(recovered, "fsck", "--full", "--no-dangling")
    assert Git(recovered).refs() == Git(target).refs()
    for value in original_refs.values():
        assert command(recovered, "cat-file", "-p", value) == command(
            source, "cat-file", "-p", value
        )


def test_github_snapshot_pagination_privacy_and_race() -> None:
    counter = 0

    def request(endpoint: str) -> Any:
        nonlocal counter
        if "/branches?" in endpoint:
            counter += 1
            return [{"name": "main", "commit": {"sha": "a" * 40}}]
        if "/issues?" in endpoint:
            return [
                {
                    "number": 1,
                    "html_url": "https://github.com/srgangaram-swe/AlphaForge/issues/1",
                    "state": "closed",
                    "body": "private body not retained",
                }
            ]
        if endpoint.endswith("/protection"):
            return {"required_status_checks": {"strict": True}, "secret": "not retained"}
        if "?" in endpoint:
            return []
        return {
            "default_branch": "main",
            "license": {"spdx_id": "MIT"},
            "html_url": "https://github.com/srgangaram-swe/AlphaForge",
        }

    snapshot = capture("srgangaram-swe/AlphaForge", request)
    assert counter == 2
    assert len(snapshot["discussions"]) == 1
    assert b"private body" not in canonical(snapshot)
    assert b"not retained" not in canonical(snapshot)
    with pytest.raises(PreservationError):
        capture("other/repository", request)
    with pytest.raises(PreservationError):
        capture("srgangaram-swe/AlphaForge", lambda endpoint: [{}] * 100)
    calls = 0

    def moving(endpoint: str) -> Any:
        nonlocal calls
        result = request(endpoint)
        if "/branches?" in endpoint:
            calls += 1
            result[0]["commit"]["sha"] = ("a" if calls == 1 else "b") * 40
        return result

    with pytest.raises(PreservationError):
        capture("srgangaram-swe/AlphaForge", moving)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "shards", "traversal", "duplicate", "extra", "symlink", "digest", "mapping"],
)
def test_malformed_publication(source: Path, tmp_path: Path, mutation: str) -> None:
    output = tmp_path / "malformed"
    publish(ledger_for(source), output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == "schema":
        manifest["schema_version"] = 2
    elif mutation == "shards":
        manifest["shards"] = "not a list"
    elif mutation == "traversal":
        manifest["shards"][0]["path"] = "../outside.json"
    elif mutation == "duplicate":
        manifest["shards"].append(manifest["shards"][0])
    elif mutation == "extra":
        (output / "extra.json").write_text("{}")
    elif mutation == "symlink":
        shard = output / manifest["shards"][0]["path"]
        shard.unlink()
        shard.symlink_to(source / "LICENSE")
    elif mutation == "digest":
        manifest["ledger_sha256"] = "0" * 64
    else:
        manifest["shards"][0]["mapping"] = True
    manifest_path.write_bytes(canonical(manifest))
    with pytest.raises(PreservationError):
        read_ledger(output)


def test_cli_in_process(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.preservation.__main__ import main

    policy = tmp_path / "licenses.json"
    policy.write_bytes(
        canonical({"alphaforge": {sha256((source / "LICENSE").read_bytes()): "fixture"}})
    )
    output = tmp_path / "cli-ledger"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preservation",
            "inventory",
            "--repository",
            str(source),
            "--source",
            "alphaforge",
            "--licenses",
            str(policy),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    monkeypatch.setattr(sys, "argv", ["preservation", "verify", str(output)])
    assert main() == 0
    (output / "extra").touch()
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2


def test_remaining_resource_and_state_guards(source: Path, tmp_path: Path) -> None:
    with pytest.raises(PreservationError):
        inventory(source, "unknown", {"0" * 64: "fixture"})
    with pytest.raises(PreservationError):
        inventory(source, "alphaforge", {})
    for limit in (
        Limits(total_blob_bytes=1),
        Limits(paths=1),
        Limits(path_records=1),
        Limits(interface_records=1),
        Limits(refs=1),
    ):
        with pytest.raises(PreservationError):
            ledger_for(source, limits=limit)
    git = Git(source)
    git.deadline = 0
    with pytest.raises(PreservationError, match="inventory-timeout"):
        git.run("status")
    command(source, "config", "remote.origin.promisor", "true")
    with pytest.raises(PreservationError, match="partial-repository"):
        Git(source)
    with pytest.raises(PreservationError):
        require_migration_clear({"schema_version": 1, "blockers": [], "migration_gate": "PASS"})
    (tmp_path / ".reserved.reservation").mkdir()
    with pytest.raises(PreservationError, match="destination-reserved"):
        publish({}, tmp_path / "reserved")
    with pytest.raises(PreservationError, match="publication-shard-limit"):
        publish({"huge": "x" * 1_000_000}, tmp_path / "huge")
    assert not (tmp_path / "huge").exists()
    assert not (tmp_path / ".huge.reservation").exists()


def test_reference_evidence_integrity_and_scientific_limits() -> None:
    from scripts.preservation.evidence import summarize

    root = Path("docs/evidence/signal_foundry_sprint_6")
    result = summarize([root / "alphaforge", root / "signalattice"])
    assert result == json.loads((root / "reference/summary.json").read_bytes())
    assert result["runtime_parity"] == "NOT_RUN"
    assert result["migration"] == "NOT_PERFORMED"
    for source_record in result["sources"]:
        assert source_record["tracked_dev_files"] == source_record["mapped_dev_files"]
        ledger = read_ledger(root / source_record["source"])
        metadata = json.loads(
            (root / "reference" / f"{source_record['source']}-github.json").read_bytes()
        )
        assert ledger["refs"] == metadata["advertised_refs"]
        if source_record["source"] == "alphaforge":
            assert {row["code"] for row in ledger["blockers"]} == {"missing-license"}
        else:
            require_migration_clear(ledger)


def test_plot_uses_verified_aggregates(source: Path, tmp_path: Path) -> None:
    from scripts.preservation.evidence import plot, summarize

    output = tmp_path / "plot-ledger"
    publish(ledger_for(source), output)
    summary = summarize([output])
    plot(summary, tmp_path / "first.png")
    plot(summary, tmp_path / "second.png")
    assert (tmp_path / "first.png").read_bytes() == (tmp_path / "second.png").read_bytes()
    with pytest.raises(PreservationError):
        summarize([output, output])


def test_publication_fault_cleanup(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import publish_preservation_evidence as evidence_command
    from scripts.preservation import publish as publication

    with pytest.raises(PreservationError, match="invalid-ledger-field"):
        publish({"../outside": "rejected"}, tmp_path / "unsafe")
    assert not (tmp_path / "outside").exists()
    ledger = ledger_for(source)
    ledger_path = tmp_path / "valid-ledger"
    publish(ledger, ledger_path)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected I/O fault")

    monkeypatch.setattr(publication.os, "rename", fail)
    with pytest.raises(PreservationError, match="publication-io-failure"):
        publish(ledger, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not (tmp_path / ".failed.reservation").exists()
    with pytest.raises(OSError):
        evidence_command.publish_evidence([ledger_path], tmp_path / "failed-evidence")
    assert not (tmp_path / "failed-evidence").exists()
