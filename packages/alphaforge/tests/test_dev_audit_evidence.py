"""Saved-response provenance, failed audits and exclusive publication contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import sys

import pytest
from packaging.version import Version

from scripts.dev_audit import MAX_BYTES, AuditContractError
from scripts.dev_audit_evidence import audit_findings, build_evidence, main, publish
from tests.test_dev_audit import document, encode, exported

__all__ = ["document"]  # Reuse the tiny lock fixture, not a network service.


def report(packages, finding=False):
    return json.dumps(
        {
            "dependencies": [
                {
                    "name": name,
                    "version": str(version),
                    "vulns": [{"id": "PYSEC-2026-3721"}] if finding and name == "pip" else [],
                }
                for name, version in sorted(packages)
            ],
            "fixes": [],
        }
    ).encode()


@pytest.fixture
def inputs(document):
    document["package"][0]["optional-dependencies"].update(data=[], ml=[])
    old = copy.deepcopy(document)
    old["package"][-1]["version"] = "26.1.2"

    def keys(doc):
        return {
            (item["name"], Version(item["version"]))
            for item in doc["package"]
            if item["name"] != "alphaforge"
        }

    return (
        encode(old),
        encode(document),
        exported(document),
        report(keys(old), True),
        report(keys(document)),
    )


def test_paired_audit_evidence_and_bitwise_repeatable_publication(inputs, tmp_path):
    evidence = build_evidence(*inputs)
    assert evidence["changed_pip_records"] == 1
    assert evidence["unchanged_third_party_package_records"] == 2
    assert evidence["newly_audited_dev_packages"] == 2
    assert len(evidence["before_findings"]) == 1 and evidence["after_findings"] == []
    one, two = tmp_path / "one", tmp_path / "two"
    publish(evidence, one)
    publish(evidence, two)
    for path in one.iterdir():
        assert path.read_bytes() == (two / path.name).read_bytes()
    manifest = json.loads((one / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((one / name).read_bytes()).hexdigest() == digest
    with pytest.raises(AuditContractError, match="exists"):
        publish(evidence, one)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"dependencies": [], "fixes": []},
        {
            "dependencies": [{"name": "pip", "version": "26.2", "skip_reason": "missing"}],
            "fixes": [],
        },
        {"dependencies": [{"name": "pip", "version": "bad", "vulns": []}], "fixes": []},
        {"dependencies": [{"name": "pip", "version": "26.2", "vulns": ["bad"]}], "fixes": []},
        {
            "dependencies": [
                {"name": "pip", "version": "26.2", "vulns": [{"id": "https://untrusted.invalid"}]}
            ],
            "fixes": [],
        },
        {
            "dependencies": [{"name": "pip", "version": "26.2", "vulns": []}],
            "fixes": [{"name": "pip"}],
        },
        {"dependencies": [None], "fixes": []},
    ],
)
def test_failed_skipped_or_malformed_audits_cannot_be_published(payload):
    with pytest.raises(AuditContractError):
        audit_findings(json.dumps(payload).encode(), {("pip", Version("26.2"))})


@pytest.mark.parametrize("content", [b"", b"bad", b" " * (MAX_BYTES + 1)])
def test_bad_audit_serialization_is_bounded(content):
    with pytest.raises(AuditContractError):
        audit_findings(content, set())


@pytest.mark.parametrize(
    "content",
    [
        b'{"dependencies": [], "dependencies": [], "fixes": []}',
        b'{"dependencies": [], "fixes": [], "untrusted": NaN}',
        b'{"dependencies": [{"name":"pip","name":"pip","version":"26.2","vulns":[]}], "fixes": []}',
    ],
)
def test_ambiguous_and_nonstandard_json_is_rejected(content):
    with pytest.raises(AuditContractError, match="invalid-audit-json"):
        audit_findings(content, set())


def test_existing_advisory_and_unrelated_lock_drift_block_publication(inputs, document):
    before, after, requirements, old, new = inputs
    payload = json.loads(new)
    payload["dependencies"][0]["vulns"] = [{"id": "CVE-2026-11111"}]
    with pytest.raises(AuditContractError, match="has-advisories"):
        build_evidence(before, after, requirements, old, json.dumps(payload).encode())
    document["package"][1]["version"] = "2.0"
    with pytest.raises(AuditContractError, match="unrelated-lock-drift"):
        build_evidence(encode(document), after, requirements, old, new)


def test_failed_plot_cleans_only_owned_stage_and_preserves_destination(
    inputs, tmp_path, monkeypatch
):
    import scripts.dev_audit_evidence as module

    def fail(*args):
        raise OSError("injected output failure")

    monkeypatch.setattr(module, "plot_evidence", fail)
    with pytest.raises(AuditContractError, match="publication-failed"):
        publish(build_evidence(*inputs), tmp_path / "out")
    assert list(tmp_path.iterdir()) == []
    (tmp_path / ".out.reservation").mkdir()
    with pytest.raises(AuditContractError, match="unavailable"):
        publish(build_evidence(*inputs), tmp_path / "out")
    assert (tmp_path / ".out.reservation").exists()


def test_cli_success_and_invalid_input(inputs, tmp_path, monkeypatch, capsys):
    args = ["dev_audit_evidence"]
    for name, content in zip(
        ("before-lock", "after-lock", "requirements", "before-audit", "after-audit"),
        inputs,
        strict=True,
    ):
        path = tmp_path / name
        path.write_bytes(content)
        args.extend(["--" + name, str(path)])
    args.extend(["--output", str(tmp_path / "out")])
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 0
    assert main() == 2
    assert "exists" in capsys.readouterr().err
