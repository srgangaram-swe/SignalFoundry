"""Keep canonical license recognition and redistribution warnings independently intact."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from scripts import release, verify_distributions

from quant_platform.release import identity

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_mit_and_unchanged_disclaimer_are_separate() -> None:
    license_text = (ROOT / "LICENSE").read_text()
    disclaimer = (ROOT / "DISCLAIMER.md").read_text()
    assert license_text.startswith("MIT License\n\nCopyright (c) 2026 srgangaram-swe\n")
    assert license_text.endswith("SOFTWARE.\n")
    assert "DISCLAIMER" not in license_text
    # Canonical MIT text with this repository's existing copyright notice.
    assert hashlib.sha256(license_text.encode()).hexdigest() == (
        "ec1e512f1009e289ecb15fbceee1f361dd3e843d336751477baf9e2b5804e7e1"
    )
    assert disclaimer == (
        "DISCLAIMER: This software is provided for educational and portfolio purposes\n"
        "only. It is NOT financial advice and is NOT intended for live trading or\n"
        "investment decisions. Past performance of any strategy, simulated or real,\n"
        "does not guarantee future results. Use at your own risk.\n"
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "MIT"
    assert set(project["license-files"]) == {"LICENSE", "DISCLAIMER.md"}
    for name in ("LICENSE", "DISCLAIMER.md"):
        assert name in verify_distributions._REQUIRED_SDIST_FILES
        assert f"licenses/{name}" in verify_distributions._REQUIRED_WHEEL_METADATA_FILES


def test_complete_stage_preserves_both_notices(tmp_path: Path) -> None:
    release._copy_schemas_and_evidence(ROOT, tmp_path)
    for name in ("LICENSE", "DISCLAIMER.md"):
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes()
        assert release._classify(name) == "license"


def _project(tmp_path: Path, lock: Any) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.3.1"\n')
    (tmp_path / "web").mkdir()
    (tmp_path / "web/package.json").write_text('{"version":"0.3.1"}')
    (tmp_path / "web/package-lock.json").write_text(json.dumps(lock))
    return tmp_path


@pytest.mark.parametrize("field", ["top", "root"])
def test_npm_lock_drift_is_rejected(tmp_path: Path, monkeypatch: Any, field: str) -> None:
    monkeypatch.setattr(identity, "_installed_version", lambda: None)
    lock: dict[str, Any] = {"version": "0.3.1", "packages": {"": {"version": "0.3.1"}}}
    if field == "top":
        lock["version"] = "0.1.0"
    else:
        lock["packages"][""]["version"] = "0.1.0"
    with pytest.raises(identity.ReleaseIdentityError, match="console-lock"):
        identity.assert_versions_agree(_project(tmp_path, lock))


@pytest.mark.parametrize(
    "lock",
    [
        [],
        {},
        {"packages": []},
        {"packages": {"": []}},
        {"version": "0.3.1", "packages": {"": {"version": None}}},
    ],
)
def test_malformed_npm_identity_fails_closed(tmp_path: Path, lock: Any) -> None:
    with pytest.raises(identity.ReleaseIdentityError, match="package-lock"):
        identity.assert_versions_agree(_project(tmp_path, lock))


def test_npm_dependency_versions_do_not_impersonate_release(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(identity, "_installed_version", lambda: None)
    lock = {
        "version": "0.3.1",
        "packages": {"": {"version": "0.3.1"}, "node_modules/react": {"version": "19.2.8"}},
    }
    assert identity.assert_versions_agree(_project(tmp_path, lock)) == "0.3.1"


def test_unreadable_json_and_array_console_identity_are_refused(tmp_path: Path) -> None:
    _project(tmp_path, {})
    (tmp_path / "web/package.json").write_text("[]")
    with pytest.raises(identity.ReleaseIdentityError, match="object"):
        identity.assert_versions_agree(tmp_path)
    (tmp_path / "web/package.json").write_text("{")
    with pytest.raises(identity.ReleaseIdentityError, match="JSON"):
        identity.assert_versions_agree(tmp_path)


def test_console_cannot_silently_omit_its_lock(tmp_path: Path) -> None:
    _project(tmp_path, {})
    (tmp_path / "web/package-lock.json").unlink()
    with pytest.raises(identity.ReleaseIdentityError, match="console lockfile.*missing"):
        identity.assert_versions_agree(tmp_path)
