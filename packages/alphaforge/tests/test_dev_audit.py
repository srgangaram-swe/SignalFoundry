"""Adversarial export/graph tests and real locked export integration for #123."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import tomli_w
import yaml
from packaging.markers import default_environment
from packaging.version import Version

from scripts.dev_audit import (
    EXPORT_ARGUMENTS,
    MAX_BYTES,
    AuditContractError,
    LockGraph,
    parse_export,
    read_bounded,
    validate_export,
)

DIGEST = "sha256:" + "a" * 64
BASE = "c9371ef4c9edd0a148b751b17b4b939d8905d300"


def environment() -> dict[str, str]:
    return {key: str(value) for key, value in default_environment().items()}


def package(name: str, version: str = "1.0", **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "source": {"registry": "https://pypi.org/simple"},
        "wheels": [{"hash": DIGEST}],
        **fields,
    }


@pytest.fixture
def document() -> dict[str, Any]:
    return {
        "version": 1,
        "revision": 3,
        "package": [
            {
                "name": "alphaforge",
                "version": "0.3.0",
                "source": {"editable": "."},
                "dependencies": [{"name": "runtime"}],
                "optional-dependencies": {"dev": [{"name": "pip-audit"}]},
            },
            package("runtime"),
            package("pip-audit", dependencies=[{"name": "pip"}]),
            package("pip", "26.2"),
        ],
    }


def encode(document: dict[str, Any]) -> bytes:
    return tomli_w.dumps(document).encode()


def exported(document: dict[str, Any]) -> bytes:
    return "".join(
        f"{item['name']}=={item['version']} --hash={DIGEST}\n"
        for item in document["package"]
        if item["name"] != "alphaforge"
    ).encode()


def test_exact_runtime_and_dev_union_and_deterministic_summary(document):
    first = validate_export(encode(document), exported(document))
    assert first == validate_export(encode(document), exported(document))
    assert first["runtime_packages"] == 1
    assert first["additional_dev_packages"] == 2
    assert {item["name"] for item in first["active_packages"]} == {"runtime", "pip", "pip-audit"}


@pytest.mark.parametrize("name", ["runtime", "pip", "pip-audit"])
def test_omitted_direct_or_transitive_dependency_fails(document, name):
    text = b"\n".join(
        line
        for line in exported(document).splitlines()
        if not line.startswith(name.encode() + b"==")
    )
    with pytest.raises(AuditContractError, match="closure-mismatch"):
        validate_export(encode(document), text)


def test_extra_reached_after_base_and_cycles_are_not_lost(document):
    document["package"][2]["dependencies"] += [
        {"name": "runtime", "extra": ["cache"]},
        {"name": "runtime"},
    ]
    document["package"][1]["optional-dependencies"] = {"cache": [{"name": "cache"}]}
    document["package"].append(package("cache", dependencies=[{"name": "pip-audit"}]))
    report = validate_export(encode(document), exported(document))
    assert report["additional_dev_packages"] == 3


@pytest.mark.parametrize("version", ["26.1.2", "26.2rc1"])
def test_unpatched_pip_cannot_pass_even_with_consistent_lock_and_hashes(document, version):
    document["package"][-1]["version"] = version
    with pytest.raises(AuditContractError, match="unpatched"):
        validate_export(encode(document), exported(document))


@pytest.mark.parametrize(
    "line",
    [
        "--index-url https://example.invalid/private-token",
        "--extra-index-url https://example.invalid",
        "-r nested.txt",
        "-e .",
        "pip @ https://example.invalid/pip.whl",
        "pip>=26.2",
        "pip==26.*",
        "pip==26.2",
        "pip[extra]==26.2",
        f"pip==26.2 --hash=md5:{'a' * 32}",
        f"pip==26.2 --hash={DIGEST} --trusted-host pypi.org",
        f"pip==26.2 --hash={DIGEST} --hash={DIGEST}",
        f"pip==26.2 --hash={DIGEST} \\",
        "# comment",
        "\x00pip==26.2",
    ],
)
def test_rejects_noncanonical_or_executable_requirement_syntax(line):
    with pytest.raises(AuditContractError):
        parse_export(line.encode())


def test_continuation_and_windows_line_endings_have_same_pins():
    single = f"pip==26.2 --hash={DIGEST}\n".encode()
    folded = f"pip==26.2 \\\r\n    --hash={DIGEST}\r\n".encode()
    assert parse_export(single) == parse_export(folded)


@pytest.mark.parametrize("content", [b"", b"\xff", b" " * (MAX_BYTES + 1), b"\n"])
def test_empty_oversized_and_non_ascii_exports_fail(content):
    with pytest.raises(AuditContractError):
        parse_export(content)


def test_hash_removal_mutation_and_duplicate_active_records_fail(document):
    content = exported(document)
    for bad in (
        content.replace(DIGEST.encode(), ("sha256:" + "b" * 64).encode(), 1),
        content + content.splitlines()[0] + b"\n",
        content + f"runtime==1.0; python_version >= '3' --hash={DIGEST}\n".encode(),
    ):
        with pytest.raises(AuditContractError):
            validate_export(encode(document), bad)


def test_marker_omission_is_detected_and_inactive_hashes_are_still_checked(document):
    document["package"][2]["dependencies"].append(
        {"name": "windows", "marker": "sys_platform == 'win32'"}
    )
    document["package"].append(package("windows"))
    content = exported(document).replace(b"windows==1.0", b"windows==1.0; sys_platform == 'win32'")
    env = environment()
    env["sys_platform"] = "linux"
    assert len(validate_export(encode(document), content, env)["active_packages"]) == 3
    env["sys_platform"] = "win32"
    assert len(validate_export(encode(document), content, env)["active_packages"]) == 4
    with pytest.raises(AuditContractError, match="closure-mismatch"):
        validate_export(encode(document), content.replace(b"win32", b"darwin"), env)
    with pytest.raises(AuditContractError, match="hash-mismatch"):
        validate_export(
            encode(document), content.replace(DIGEST.encode(), b"sha256:" + b"b" * 64), env
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(version=99),
        lambda d: d.update(package=[]),
        lambda d: d["package"].append(copy.deepcopy(d["package"][-1])),
        lambda d: d["package"][1].update(source={"registry": "https://example.invalid"}),
        lambda d: d["package"][1].update(wheels=[]),
        lambda d: d["package"][1].update(wheels=[{"hash": "wrong"}]),
        lambda d: d["package"][1].update(version=5),
        lambda d: d["package"][1].update(name="../payload"),
        lambda d: d["package"][2].update(dependencies=[{"name": "missing"}]),
        lambda d: d["package"][2].update(dependencies=[{"name": "pip", "extra": ["missing"]}]),
        lambda d: d["package"][2].update(
            dependencies=[{"name": "pip", "marker": "invalid marker"}]
        ),
        lambda d: d["package"][2].update(dependencies=[{"name": "pip", "unknown": True}]),
    ],
)
def test_malformed_lock_and_graph_fail_closed(document, mutation):
    mutation(document)
    with pytest.raises(AuditContractError):
        graph = LockGraph(encode(document))
        graph.closure(("dev",), environment())


def test_versioned_edges_resolve_forks_and_ambiguous_edges_fail(document):
    document["package"].append(package("pip", "26.3"))
    graph = LockGraph(encode(document))
    with pytest.raises(AuditContractError, match="ambiguous"):
        graph.closure(("dev",), environment())
    document["package"][2]["dependencies"] = [{"name": "pip", "version": "26.2"}]
    assert ("pip", Version("26.2")) in LockGraph(encode(document)).closure(("dev",), environment())


def test_file_boundary_and_cli_do_not_echo_untrusted_content(tmp_path):
    secret = "not-a-real-credential-should-not-appear"
    bad = tmp_path / "bad.txt"
    bad.write_text(f"--index-url https://example.invalid/{secret}")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.dev_audit", str(bad)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert secret not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    link = tmp_path / "link"
    link.symlink_to(bad)
    for path in (link, tmp_path, tmp_path / "missing"):
        with pytest.raises(AuditContractError):
            read_bounded(path)
    bad.write_bytes(b"x" * (MAX_BYTES + 1))
    with pytest.raises(AuditContractError, match="size-limit"):
        read_bounded(bad)


def test_real_uv_export_and_supported_platform_marker_matrix(tmp_path):
    destination = tmp_path / "dev.txt"
    subprocess.run(
        ["uv", *EXPORT_ARGUMENTS, "--output-file", str(destination)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    lock, content = read_bounded(Path("uv.lock")), read_bounded(destination)
    for python in ("3.12", "3.13", "3.14"):
        for system, machine, os_name, platform in (
            ("linux", "x86_64", "posix", "Linux"),
            ("darwin", "arm64", "posix", "Darwin"),
            ("win32", "AMD64", "nt", "Windows"),
        ):
            env = environment()
            env.update(
                python_version=python,
                python_full_version=python + ".0",
                implementation_version=python + ".0",
                sys_platform=system,
                platform_machine=machine,
                os_name=os_name,
                platform_system=platform,
            )
            report = validate_export(lock, content, env)
            assert report["additional_dev_packages"] > 40
    cli = subprocess.run(
        [sys.executable, "-m", "scripts.dev_audit", str(destination)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert json.loads(cli.stdout) == validate_export(lock, content)


def test_pip_is_the_only_changed_locked_third_party_package():
    previous = subprocess.run(
        ["git", "show", f"{BASE}:uv.lock"], check=True, capture_output=True, timeout=15
    ).stdout
    old = tomllib.loads(previous.decode())
    current = tomllib.loads(Path("uv.lock").read_text())

    def unchanged(lock):
        return [item for item in lock["package"] if item["name"] not in {"alphaforge", "pip"}]

    assert unchanged(old) == unchanged(current)
    assert {key: value for key, value in old.items() if key != "package"} == {
        key: value for key, value in current.items() if key != "package"
    }


def test_required_security_job_keeps_runtime_gate_and_enforces_dev_audit():
    workflow = yaml.safe_load(Path(".github/workflows/security.yml").read_text())
    job = workflow["jobs"]["dependency-audit"]
    steps = {step.get("name"): step.get("run", "") for step in job["steps"]}
    assert job["timeout-minutes"] == 15
    runtime = steps["Export hashed runtime, data, and ML dependency resolution"]
    assert "--extra data" in runtime and "--extra ml" in runtime
    exported = steps["Export and verify hashed runtime plus development tool closure"]
    for argument in EXPORT_ARGUMENTS:
        assert argument in exported
    assert "scripts.dev_audit" in exported
    install = steps["Install clean hash-checked development audit environment"]
    assert "--require-hashes" in install and "--only-binary :all:" in install
    audit = steps["Audit development tools without executing a second resolver"]
    for flag in ("--strict", "--require-hashes", "--disable-pip", "--timeout 15"):
        assert flag in audit
    assert "--ignore-vuln" not in audit and "||" not in audit


def test_schema_resource_and_environment_boundaries(document, monkeypatch):
    import scripts.dev_audit as audit

    for content in (b"", b"[broken", b"\xff", b" " * (MAX_BYTES + 1)):
        with pytest.raises(AuditContractError):
            LockGraph(content)
    for changed in (
        {**document, "package": "wrong"},
        {**document, "package": ["not-a-mapping"]},
        {**document, "package": document["package"][1:]},
    ):
        with pytest.raises(AuditContractError):
            LockGraph(encode(changed))
    with pytest.raises(AuditContractError, match="incomplete-marker"):
        validate_export(encode(document), exported(document), {})
    monkeypatch.setattr(audit, "MAX_PACKAGES", 2)
    with pytest.raises(AuditContractError, match="package-count-limit"):
        LockGraph(encode(document))
    with pytest.raises(AuditContractError, match="package-count-limit"):
        parse_export(exported(document))


def test_edge_bound_and_malformed_marker_type(document, monkeypatch):
    import scripts.dev_audit as audit

    document["package"][1]["dependencies"] = [{"name": "pip"}, {"name": "pip-audit"}]
    monkeypatch.setattr(audit, "MAX_EDGES", 4)
    with pytest.raises(AuditContractError, match="edge-count-limit"):
        LockGraph(encode(document))
    monkeypatch.setattr(audit, "MAX_EDGES", 32768)
    document["package"][1]["dependencies"][0]["marker"] = 1
    with pytest.raises(AuditContractError, match="environment-marker"):
        LockGraph(encode(document)).closure(("dev",), environment())


def test_resolution_markers_select_one_fork(document):
    document["package"][-1]["resolution-markers"] = ["sys_platform != 'win32'"]
    other = package("pip", "26.3")
    other["resolution-markers"] = ["sys_platform == 'win32'"]
    document["package"].append(other)
    env = environment()
    env["sys_platform"] = "win32"
    assert ("pip", Version("26.3")) in LockGraph(encode(document)).closure(("dev",), env)


def test_main_in_process_covers_success_and_redacted_failure(
    document, tmp_path, monkeypatch, capsys
):
    from scripts.dev_audit import main

    lock, requirements = tmp_path / "lock", tmp_path / "requirements"
    lock.write_bytes(encode(document))
    requirements.write_bytes(exported(document))
    monkeypatch.setattr(sys, "argv", ["dev_audit", str(requirements), "--lock", str(lock)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["additional_dev_packages"] == 2
    requirements.write_text("--index-url https://example.invalid/never-echo")
    assert main() == 2
    assert "never-echo" not in capsys.readouterr().err
    requirements.write_bytes(b"")
    with pytest.raises(AuditContractError, match="size-limit"):
        read_bounded(requirements)


def test_deep_parser_failure_is_mapped_to_domain_error(monkeypatch):
    import scripts.dev_audit as audit

    def exhausted(*args, **kwargs):
        raise RecursionError("untrusted parser text")

    monkeypatch.setattr(audit, "Requirement", exhausted)
    with pytest.raises(AuditContractError, match="invalid-export-requirement"):
        audit.parse_export(f"pip==26.2 --hash={DIGEST}".encode())
