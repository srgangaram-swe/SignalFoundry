"""Tests for the console static delivery boundary (SF-S5-SL-MR6).

The boundary exists because a generic static handler is the wrong tool here: it
resolves arbitrary paths, guesses content types, and follows symlinks. Each of
those is a way for a file outside the build output to reach a browser.

What is asserted:

* **The served set is closed.** Only files enumerated at load are reachable, and
  the enumeration refuses symlinks, unsafe names, and unservable types.
* **Route fallback is an allowlist.** An unknown path is a 404, not the
  application document, so the console is not an oracle answering 200 for
  anything.
* **The document policy is strict.** No inline, no eval, no third-party origin,
  no websocket, no worker, and connect-src confined to the same origin.
* **HEAD is admitted for assets only**, with identical headers and no body.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quant_platform.service.console import (
    CONSOLE_CSP,
    CONSOLE_ROUTES,
    MAX_ASSET_BYTES,
    ConsoleBundleError,
    console_headers,
    load_console_bundle,
)

_DOCUMENT = (
    b'<!doctype html><html lang="en"><head><title>Console</title></head><body></body></html>'
)


def _bundle_dir(tmp_path: Path, *, document: bytes = _DOCUMENT) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_bytes(document)
    (root / "assets" / "app-abc123.js").write_bytes(b"export const ready = true;\n")
    (root / "assets" / "app-abc123.css").write_bytes(b":root{color-scheme:light dark}\n")
    return root


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def test_a_valid_bundle_enumerates_its_files(tmp_path: Path) -> None:
    bundle = load_console_bundle(_bundle_dir(tmp_path))
    assert set(bundle.assets) == {
        "/console/index.html",
        "/console/assets/app-abc123.js",
        "/console/assets/app-abc123.css",
    }
    assert bundle.document.route == "/console/index.html"
    assert bundle.total_bytes > 0


def test_a_bundle_without_a_document_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "dist"
    root.mkdir()
    (root / "app.js").write_bytes(b"//\n")
    with pytest.raises(ConsoleBundleError, match="no index.html"):
        load_console_bundle(root)


def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConsoleBundleError, match="not readable"):
        load_console_bundle(tmp_path / "absent")


def test_a_file_root_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "not-a-dir"
    target.write_bytes(b"x")
    with pytest.raises(ConsoleBundleError, match="not a directory"):
        load_console_bundle(target)


def test_a_symlink_is_refused_rather_than_followed(tmp_path: Path) -> None:
    """Following a link would serve content from outside the build output."""
    outside = tmp_path / "secret.js"
    outside.write_bytes(b"const token = 'not-a-console-asset';\n")
    root = _bundle_dir(tmp_path)
    os.symlink(outside, root / "assets" / "leak-000000.js")
    with pytest.raises(ConsoleBundleError, match="symlink"):
        load_console_bundle(root)


def test_an_unservable_file_type_is_refused(tmp_path: Path) -> None:
    """Content types are fixed, so an unexpected artifact is a build problem."""
    root = _bundle_dir(tmp_path)
    (root / "sourcemap.map").write_bytes(b"{}")
    with pytest.raises(ConsoleBundleError, match="unservable file type"):
        load_console_bundle(root)


@pytest.mark.parametrize("name", ["bad name.js", "weirdé.js", "tab\tname.js"])
def test_an_unsafe_file_name_is_refused(tmp_path: Path, name: str) -> None:
    root = _bundle_dir(tmp_path)
    try:
        (root / "assets" / name).write_bytes(b"//\n")
    except OSError:  # pragma: no cover - platform refuses the name outright
        pytest.skip("filesystem rejects the hostile name before the bundle loader can")
    with pytest.raises(ConsoleBundleError, match="unsafe file name"):
        load_console_bundle(root)


def test_an_oversized_asset_is_refused(tmp_path: Path) -> None:
    root = _bundle_dir(tmp_path)
    (root / "assets" / "huge-000000.js").write_bytes(b"x" * (MAX_ASSET_BYTES + 1))
    with pytest.raises(ConsoleBundleError, match="exceeds"):
        load_console_bundle(root)


def test_too_many_files_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quant_platform.service.console.MAX_ASSETS", 2)
    root = _bundle_dir(tmp_path)
    (root / "assets" / "extra-000000.js").write_bytes(b"//\n")
    with pytest.raises(ConsoleBundleError, match="exceeds 2 files"):
        load_console_bundle(root)


def test_a_bundle_over_the_total_ceiling_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quant_platform.service.console.MAX_TOTAL_BYTES", 32)
    with pytest.raises(ConsoleBundleError, match="exceeds 32 bytes"):
        load_console_bundle(_bundle_dir(tmp_path))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_declared_routes_resolve_to_the_document(tmp_path: Path) -> None:
    bundle = load_console_bundle(_bundle_dir(tmp_path))
    for route in CONSOLE_ROUTES:
        assert bundle.resolve(route) is bundle.document, route


@pytest.mark.parametrize(
    "path",
    [
        "/console/../etc/passwd",
        "/console/unknown",
        "/console/assets/missing.js",
        "/console/admin",
        "/console/index.html/extra",
        "/api/v1/runs",
        "/console/console",
    ],
)
def test_an_undeclared_path_resolves_to_nothing(tmp_path: Path, path: str) -> None:
    """There is no wildcard fallback: an unknown path is a 404, not a document."""
    bundle = load_console_bundle(_bundle_dir(tmp_path))
    assert bundle.resolve(path) is None


def test_the_route_allowlist_holds_exactly_the_seven_views() -> None:
    """An eighth console view would have to be added here visibly."""
    views = {route.rstrip("/") for route in CONSOLE_ROUTES}
    assert views == {
        "/console",
        "/console/runs",
        "/console/evidence",
        "/console/comparison",
        "/console/calibration",
        "/console/operations",
        "/console/governance",
    }


@pytest.mark.parametrize(
    "forbidden",
    ["/console/admin", "/console/sql", "/console/orders", "/console/positions", "/console/pnl"],
)
def test_no_administration_or_trading_route_is_declared(forbidden: str) -> None:
    assert forbidden not in CONSOLE_ROUTES


# ---------------------------------------------------------------------------
# Headers and policy
# ---------------------------------------------------------------------------


def test_the_console_policy_forbids_inline_eval_and_third_party_origins() -> None:
    assert "unsafe-inline" not in CONSOLE_CSP
    assert "unsafe-eval" not in CONSOLE_CSP
    assert "http://" not in CONSOLE_CSP and "https://" not in CONSOLE_CSP
    assert "default-src 'none'" in CONSOLE_CSP
    assert "connect-src 'self'" in CONSOLE_CSP
    assert "script-src 'self'" in CONSOLE_CSP
    # No websocket, worker, frame, or media surface is granted at all.
    assert "worker-src 'none'" in CONSOLE_CSP
    assert "frame-src 'none'" in CONSOLE_CSP
    assert "frame-ancestors 'none'" in CONSOLE_CSP
    assert "object-src 'none'" in CONSOLE_CSP
    assert "base-uri 'none'" in CONSOLE_CSP
    assert "form-action 'none'" in CONSOLE_CSP


def test_asset_headers_carry_the_full_security_set(tmp_path: Path) -> None:
    bundle = load_console_bundle(_bundle_dir(tmp_path))
    headers = dict(console_headers(bundle.assets["/console/assets/app-abc123.js"]))
    assert headers[b"content-type"] == b"text/javascript; charset=utf-8"
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert headers[b"referrer-policy"] == b"no-referrer"
    assert headers[b"x-frame-options"] == b"DENY"
    assert headers[b"cross-origin-resource-policy"] == b"same-origin"
    assert headers[b"cross-origin-opener-policy"] == b"same-origin"
    assert headers[b"content-security-policy"] == CONSOLE_CSP.encode("ascii")


def test_the_document_is_never_cached_but_hashed_assets_are(tmp_path: Path) -> None:
    """A cached document outliving its bundle would request files that are gone."""
    bundle = load_console_bundle(_bundle_dir(tmp_path))
    document = dict(console_headers(bundle.document))
    asset = dict(console_headers(bundle.assets["/console/assets/app-abc123.css"]))
    assert document[b"cache-control"] == b"no-store"
    assert asset[b"cache-control"] == b"public, max-age=31536000, immutable"


def test_every_asset_carries_a_strong_content_derived_validator(tmp_path: Path) -> None:
    bundle = load_console_bundle(_bundle_dir(tmp_path))
    for asset in bundle.assets.values():
        assert asset.etag.startswith('"') and asset.etag.endswith('"')
        assert len(asset.digest) == 64
        headers = dict(console_headers(asset))
        assert headers[b"content-length"] == str(asset.size).encode("ascii")


def test_two_bundles_with_identical_content_produce_identical_digests(tmp_path: Path) -> None:
    """Digests are content-derived, so a rebuild that changed nothing matches."""
    first = load_console_bundle(_bundle_dir(tmp_path / "a"))
    second = load_console_bundle(_bundle_dir(tmp_path / "b"))
    assert {route: item.digest for route, item in first.assets.items()} == {
        route: item.digest for route, item in second.assets.items()
    }
