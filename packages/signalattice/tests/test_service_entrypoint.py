"""Narrow-entry-point and dependency-boundary tests for the service image."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from quant_platform.service import entrypoint
from quant_platform.service.server import ServiceStartupError


def _arguments(root: Path) -> list[str]:
    return [
        "--registry-db",
        str(root / "registry.sqlite3"),
        "--cas-root",
        str(root / "cas"),
        "--socket-path",
        str(root / "api.sock"),
        "--digest-key-file",
        str(root / "digest-key"),
    ]


def test_entrypoint_parses_only_absolute_bounded_paths(tmp_path: Path) -> None:
    parsed = entrypoint.parse_service_arguments(_arguments(tmp_path))

    assert parsed == entrypoint.ServiceArguments(
        registry_db=tmp_path / "registry.sqlite3",
        cas_root=tmp_path / "cas",
        socket_path=tmp_path / "api.sock",
        digest_key_file=tmp_path / "digest-key",
    )

    relative = _arguments(tmp_path)
    relative[1] = "registry.sqlite3"
    with pytest.raises(entrypoint.ServiceArgumentError, match="absolute"):
        entrypoint.parse_service_arguments(relative)


def test_entrypoint_argument_failure_does_not_reflect_untrusted_value() -> None:
    marker = "SERVICE_SECRET_CANARY_0123456789abcdef"
    stderr = io.StringIO()

    status = entrypoint.main(["--unknown", marker], stderr=stderr)

    assert status == 2
    assert stderr.getvalue() == "signalattice-service: invalid service arguments\n"
    assert marker not in stderr.getvalue()


def test_entrypoint_rejects_duplicate_abbreviated_control_and_oversized_arguments(
    tmp_path: Path,
) -> None:
    valid = _arguments(tmp_path)
    abbreviated = list(valid)
    abbreviated[0] = "--registry"
    controlled = list(valid)
    controlled[1] = str(tmp_path / "registry\x1b.sqlite3")
    oversized = list(valid)
    oversized[1] = "/" + "r" * (4 * 1_024)
    duplicate = [*valid, "--registry-db", str(tmp_path / "other.sqlite3")]

    for candidate in ([], abbreviated, controlled, oversized, duplicate):
        with pytest.raises(entrypoint.ServiceArgumentError):
            entrypoint.parse_service_arguments(candidate)


def test_entrypoint_passes_exact_private_profile_to_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    runtime = Path("/tmp/signalattice-entrypoint-test")

    def serve(
        registry_db: Path,
        cas_root: Path,
        *,
        digest_key_file: Path,
        config: entrypoint.ServerConfig,
    ) -> None:
        observed.update(
            registry_db=registry_db,
            cas_root=cas_root,
            digest_key_file=digest_key_file,
            config=config,
        )

    monkeypatch.setattr(entrypoint, "serve_existing_evidence", serve)

    assert entrypoint.main(_arguments(runtime), stderr=io.StringIO()) == 0
    assert observed == {
        "registry_db": runtime / "registry.sqlite3",
        "cas_root": runtime / "cas",
        "digest_key_file": runtime / "digest-key",
        "config": entrypoint.ServerConfig(socket_path=runtime / "api.sock"),
    }


def test_entrypoint_preserves_stable_redacted_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Path("/tmp/signalattice-entrypoint-test")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ServiceStartupError("existing evidence state failed startup verification")

    monkeypatch.setattr(entrypoint, "serve_existing_evidence", fail)
    stderr = io.StringIO()

    status = entrypoint.main(_arguments(runtime), stderr=stderr)

    assert status == 2
    assert stderr.getvalue() == (
        "signalattice-service: existing evidence state failed startup verification\n"
    )
    assert str(runtime) not in stderr.getvalue()


def test_entrypoint_does_not_misreport_an_unexpected_crash_as_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "unexpected-internal-canary"
    runtime = Path("/tmp/signalattice-entrypoint-test")

    def crash(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(marker)

    monkeypatch.setattr(entrypoint, "serve_existing_evidence", crash)
    stderr = io.StringIO()

    with pytest.raises(RuntimeError, match=marker):
        entrypoint.main(_arguments(runtime), stderr=stderr)
    assert stderr.getvalue() == ""


def test_service_import_graph_excludes_research_and_legacy_adapter_dependencies() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = "\n".join(
        (
            "import sys",
            "sys.path.insert(0, sys.argv[1])",
            "import quant_platform.service.entrypoint",
            "forbidden = {'numpy', 'pandas', 'scipy', 'sklearn', 'pyarrow', 'duckdb', "
            "             'matplotlib', 'seaborn', 'yaml', 'typer', 'rich'}",
            "loaded = sorted(name for name in forbidden if name in sys.modules)",
            "assert not loaded, loaded",
            "assert 'quant_platform.tracking.experiment' not in sys.modules",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
