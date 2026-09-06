"""Traceability, reproducibility, and honesty gates for service evidence."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib.image as mpimg
import pytest
from scripts import benchmark_service_operability as benchmark
from scripts import plot_service_operability as plot

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "docs" / "benchmarks" / "service_operability_2026-09-06_patch1.json"
PLOT_PATH = ROOT / "docs" / "assets" / "service_operability_2026-09-06_patch1.png"
OPERATIONS_PATH = ROOT / "docs" / "service_operations.md"
MAKEFILE_PATH = ROOT / "Makefile"
Publisher = Callable[[bytes, Path], None]
PUBLISHERS: tuple[object, ...] = (
    pytest.param(benchmark._publish_bytes_no_replace, id="benchmark-json"),
    pytest.param(plot._publish_bytes_no_replace, id="seaborn-plot"),
)


def _load_committed_evidence() -> dict[str, Any]:
    payload = EVIDENCE_PATH.read_bytes()
    assert len(payload) <= 2 * 1024 * 1024
    document = json.loads(payload)
    assert type(document) is dict
    return cast(dict[str, Any], document)


def _isolated_child_result(
    mode: str,
    *,
    before: int = 64 * 1024 * 1024,
    after: int = 68 * 1024 * 1024,
) -> dict[str, object]:
    return {
        "protocol": benchmark._ISOLATED_RSS_PROTOCOL,
        "mode": mode,
        "workload_sha256": benchmark._isolated_rss_workload_sha256(),
        "runtime_platform": (
            f"{benchmark.platform.system()}-{benchmark.platform.release()}-"
            f"{benchmark.platform.machine()}"
        ),
        "python": benchmark.platform.python_version(),
        "python_implementation": benchmark.platform.python_implementation(),
        "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "rss_unit": "bytes",
        "rss_before_app_bytes": before,
        "rss_after_shutdown_peak_bytes": after,
        "workload_growth_bytes": after - before,
        "warmup_requests": benchmark.WARMUP_REQUESTS,
        "measured_requests": benchmark.STEADY_REQUESTS_PER_ROUND,
        "successful_requests": benchmark.STEADY_REQUESTS_PER_ROUND,
        "status_counts": {"200": benchmark.STEADY_REQUESTS_PER_ROUND},
        "lifespan_shutdown_completed": True,
        "network_requests": 0,
        "processes": 1,
    }


@pytest.mark.parametrize("publisher", PUBLISHERS)
def test_publication_creates_one_canonical_regular_file(
    tmp_path: Path,
    publisher: Publisher,
) -> None:
    destination = tmp_path / "artifact.bin"
    payload = b"bounded-reference-artifact\n"

    publisher(payload, destination)

    metadata = destination.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o644
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize("publisher", PUBLISHERS)
def test_publication_accepts_identical_regular_file_without_replacing_it(
    tmp_path: Path,
    publisher: Publisher,
) -> None:
    destination = tmp_path / "artifact.bin"
    payload = b"bounded-reference-artifact\n"
    publisher(payload, destination)
    before = destination.lstat()

    publisher(payload, destination)

    after = destination.lstat()
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize("publisher", PUBLISHERS)
def test_publication_refuses_different_existing_regular_file(
    tmp_path: Path,
    publisher: Publisher,
) -> None:
    destination = tmp_path / "artifact.bin"
    foreign = b"foreign-owner-content\n"
    destination.write_bytes(foreign)
    destination.chmod(0o644)
    before = destination.lstat()

    with pytest.raises(ValueError, match="different bytes"):
        publisher(b"candidate-content\n", destination)

    after = destination.lstat()
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )
    assert destination.read_bytes() == foreign


@pytest.mark.parametrize("publisher", PUBLISHERS)
def test_publication_refuses_symlink_without_touching_foreign_target(
    tmp_path: Path,
    publisher: Publisher,
) -> None:
    foreign_target = tmp_path / "foreign.bin"
    foreign = b"foreign-owner-content\n"
    foreign_target.write_bytes(foreign)
    destination = tmp_path / "artifact.bin"
    destination.symlink_to(foreign_target)
    symlink_identity = destination.lstat()

    with pytest.raises(ValueError, match="verify existing"):
        publisher(b"candidate-content\n", destination)

    after = destination.lstat()
    assert stat.S_ISLNK(after.st_mode)
    assert (after.st_dev, after.st_ino) == (symlink_identity.st_dev, symlink_identity.st_ino)
    assert foreign_target.read_bytes() == foreign


@pytest.mark.parametrize("publisher", PUBLISHERS)
def test_publication_refuses_nonregular_target(
    tmp_path: Path,
    publisher: Publisher,
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.mkdir()
    foreign_marker = destination / "foreign-marker"
    foreign_marker.write_bytes(b"do-not-delete\n")

    with pytest.raises(ValueError, match="regular file"):
        publisher(b"candidate-content\n", destination)

    assert destination.is_dir()
    assert foreign_marker.read_bytes() == b"do-not-delete\n"


@pytest.mark.parametrize("publisher", PUBLISHERS)
def test_publication_loser_preserves_file_created_by_destination_race(
    tmp_path: Path,
    publisher: Publisher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    foreign = b"concurrent-owner-content\n"
    raced_identity: list[os.stat_result] = []

    def create_racing_destination(
        source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        del source, follow_symlinks
        assert Path(target) == destination
        destination.write_bytes(foreign)
        destination.chmod(0o644)
        raced_identity.append(destination.lstat())
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)

    monkeypatch.setattr(benchmark.os, "link", create_racing_destination)

    with pytest.raises(ValueError, match="different bytes"):
        publisher(b"candidate-content\n", destination)

    assert len(raced_identity) == 1
    after = destination.lstat()
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        raced_identity[0].st_dev,
        raced_identity[0].st_ino,
        raced_identity[0].st_mtime_ns,
    )
    assert destination.read_bytes() == foreign
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_distribution_summary_is_interpolated_ordered_and_raw_free() -> None:
    summary = benchmark.summarize_distribution((4.0, 1.0, 3.0, 2.0))

    assert summary["count"] == 4
    assert summary["min"] == 1.0
    assert summary["p50"] == 2.5
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["p99"] == pytest.approx(3.97)
    assert summary["max"] == 4.0
    assert "samples" not in summary
    points = cast(list[dict[str, float]], summary["empirical_percentiles"])
    assert [point["percentile"] for point in points] == sorted(
        point["percentile"] for point in points
    )
    assert points[0] == {"percentile": 0.0, "value_ms": 1.0}
    assert points[-1] == {"percentile": 100.0, "value_ms": 4.0}


@pytest.mark.parametrize("values", [(), (-1.0,), (float("nan"),), (float("inf"),)])
def test_distribution_summary_rejects_empty_nonfinite_or_negative_input(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="non-empty, finite, and non-negative"):
        benchmark.summarize_distribution(values)


def test_benchmark_canary_is_bounded_injected_and_never_named_in_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "SIGNALATTICE-S5-WORKFLOW-CANARY-20260809"
    monkeypatch.setenv("SERVICE_SECRET_CANARY", marker)

    assert benchmark._service_canary_from_environment() == marker
    ports = benchmark._SyntheticPorts(
        fault=benchmark._PortFault.CORRUPT_MANIFEST,
        canary_marker=marker,
    )
    malformed = ports.read_verified_manifest(
        benchmark._DIAGNOSTIC_DIGEST,
        benchmark._MANIFEST_MEDIA_TYPE,
        512 * 1_024,
    )
    assert marker.encode("ascii") in malformed

    monkeypatch.setenv("SERVICE_SECRET_CANARY", "short")
    with pytest.raises(RuntimeError) as raised:
        benchmark._service_canary_from_environment()
    assert marker not in str(raised.value)
    assert "short" not in str(raised.value)


def test_isolated_rss_schema_accepts_only_compatible_exact_results() -> None:
    valid = _isolated_child_result("disabled")
    assert benchmark._validate_isolated_rss_child_result(valid, expected_mode="disabled") == valid

    unknown = dict(valid)
    unknown["unexpected"] = "field"
    with pytest.raises(benchmark._IsolatedRssProtocolError, match="closed schema"):
        benchmark._validate_isolated_rss_child_result(unknown, expected_mode="disabled")

    boolean_count = dict(valid)
    boolean_count["network_requests"] = False
    with pytest.raises(benchmark._IsolatedRssProtocolError, match="network boundary"):
        benchmark._validate_isolated_rss_child_result(boolean_count, expected_mode="disabled")

    incomplete = dict(valid)
    incomplete["lifespan_shutdown_completed"] = False
    with pytest.raises(benchmark._IsolatedRssProtocolError, match="lifespan shutdown"):
        benchmark._validate_isolated_rss_child_result(incomplete, expected_mode="disabled")

    mismatched_workload = dict(valid)
    mismatched_workload["workload_sha256"] = "0" * 64
    with pytest.raises(benchmark._IsolatedRssProtocolError, match="workload identity"):
        benchmark._validate_isolated_rss_child_result(mismatched_workload, expected_mode="disabled")


def test_isolated_rss_delta_retains_signed_observation_and_enforces_cap() -> None:
    disabled = _isolated_child_result("disabled", after=70 * 1024 * 1024)
    enabled_lower = _isolated_child_result("enabled", after=69 * 1024 * 1024)

    lower = benchmark._aggregate_isolated_rss_measurements(disabled, enabled_lower)

    assert lower["observed_signed_delta_bytes"] == -(1024 * 1024)
    assert lower["nonnegative_delta_bytes"] == 0
    assert lower["limit_bytes"] == 100 * 1024 * 1024
    assert lower["comparison_compatible"] is True
    assert lower["within_limit"] is True

    enabled_higher = _isolated_child_result("enabled", after=73 * 1024 * 1024)
    higher = benchmark._aggregate_isolated_rss_measurements(disabled, enabled_higher)
    assert higher["observed_signed_delta_bytes"] == 3 * 1024 * 1024
    assert higher["nonnegative_delta_bytes"] == 3 * 1024 * 1024

    enabled_excessive = _isolated_child_result(
        "enabled",
        after=70 * 1024 * 1024 + benchmark.MAX_TELEMETRY_RSS_DELTA_BYTES + 1,
    )
    with pytest.raises(RuntimeError, match="100 MiB"):
        benchmark._aggregate_isolated_rss_measurements(disabled, enabled_excessive)


def test_isolated_rss_measurement_runs_two_clean_compatible_processes() -> None:
    measurement = benchmark._measure_isolated_telemetry_rss()

    assert measurement["process_order"] == ["disabled", "enabled"]
    assert measurement["nonnegative_delta_bytes"] <= benchmark.MAX_TELEMETRY_RSS_DELTA_BYTES
    for mode in ("disabled", "enabled"):
        arm = cast(dict[str, Any], measurement[mode])
        assert arm["mode"] == mode
        assert arm["warmup_requests"] == benchmark.WARMUP_REQUESTS
        assert arm["measured_requests"] == benchmark.STEADY_REQUESTS_PER_ROUND
        assert arm["status_counts"] == {"200": benchmark.STEADY_REQUESTS_PER_ROUND}
        assert arm["lifespan_shutdown_completed"] is True
        assert arm["network_requests"] == 0


def test_isolated_rss_child_timeout_terminates_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = benchmark.subprocess.Popen
    processes: list[Any] = []

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(
        benchmark,
        "_isolated_child_command",
        lambda _mode: (
            benchmark.sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ),
    )
    monkeypatch.setattr(benchmark.subprocess, "Popen", recording_popen)

    with pytest.raises(benchmark._IsolatedRssProtocolError, match="hard deadline"):
        benchmark._run_isolated_rss_child("disabled", timeout_seconds=0.05)

    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_isolated_rss_child_rejects_wrong_handshake_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = benchmark.subprocess.Popen
    processes: list[Any] = []

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(
        benchmark,
        "_isolated_child_command",
        lambda _mode: (
            benchmark.sys.executable,
            "-c",
            "import sys,time;sys.stdout.write('WRONG\\n');sys.stdout.flush();time.sleep(60)",
        ),
    )
    monkeypatch.setattr(benchmark.subprocess, "Popen", recording_popen)

    with pytest.raises(benchmark._IsolatedRssProtocolError, match="handshake"):
        benchmark._run_isolated_rss_child("enabled", timeout_seconds=1.0)

    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_isolated_rss_child_environment_is_credential_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_SECRET_CANARY", "SIGNALATTICE-CANARY-NEVER-INHERIT")
    monkeypatch.setenv("NASDAQ_DATA_LINK_API_KEY", "SYNTHETIC-NOT-A-REAL-CREDENTIAL")

    environment = benchmark._isolated_child_environment()

    assert set(environment) == {
        "PATH",
        "LC_ALL",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED",
    }
    assert "SERVICE_SECRET_CANARY" not in environment
    assert "NASDAQ_DATA_LINK_API_KEY" not in environment


def test_source_hash_binds_the_complete_service_import_closure(tmp_path: Path) -> None:
    source_files = benchmark._service_source_files(ROOT)
    relative_files = {path.relative_to(ROOT).as_posix() for path in source_files}
    tracking_files = {
        relative
        for relative in relative_files
        if relative.startswith("src/quant_platform/tracking/")
    }
    service_files = {
        relative
        for relative in relative_files
        if relative.startswith("src/quant_platform/service/")
    }

    assert service_files == {
        "src/quant_platform/service/__init__.py",
        "src/quant_platform/service/__main__.py",
        "src/quant_platform/service/admission.py",
        "src/quant_platform/service/api.py",
        "src/quant_platform/service/console.py",
        "src/quant_platform/service/contracts.py",
        "src/quant_platform/service/entrypoint.py",
        "src/quant_platform/service/exporter.py",
        "src/quant_platform/service/governance_models.py",
        "src/quant_platform/service/http_protocol.py",
        "src/quant_platform/service/manifests.py",
        "src/quant_platform/service/metrics.py",
        "src/quant_platform/service/middleware.py",
        "src/quant_platform/service/models.py",
        "src/quant_platform/service/problems.py",
        "src/quant_platform/service/server.py",
        "src/quant_platform/service/telemetry.py",
        "src/quant_platform/service/telemetry_contracts.py",
    }
    assert tracking_files == {
        "src/quant_platform/tracking/__init__.py",
        "src/quant_platform/tracking/cas.py",
        "src/quant_platform/tracking/contracts.py",
        "src/quant_platform/tracking/migrations.py",
        "src/quant_platform/tracking/read_ports.py",
        "src/quant_platform/tracking/registry.py",
        "src/quant_platform/tracking/retention.py",
    }
    for source in source_files:
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    initial = benchmark._source_tree_sha256_at(tmp_path)
    retention = tmp_path / "src/quant_platform/tracking/retention.py"
    retention.write_bytes(retention.read_bytes() + b"\n# mutation regression\n")

    assert benchmark._source_tree_sha256_at(tmp_path) != initial


def test_committed_aggregate_evidence_is_integrity_bound_bounded_and_honest() -> None:
    evidence = _load_committed_evidence()

    assert evidence["schema_version"] == "1.0.0"
    assert evidence["evidence_class"] == "measured_synthetic_local_engineering"
    integrity = cast(dict[str, str], evidence["integrity"])
    assert integrity["algorithm"] == "sha256"
    assert (
        hashlib.sha256(benchmark._integrity_payload(evidence)).hexdigest()
        == integrity["canonical_payload_sha256"]
    )
    assert evidence["provenance"] == {
        "credentials_used": False,
        "market_or_model_data_used": False,
        "network_requests": 0,
        "raw_request_samples_committed": False,
        "source_tree_sha256": evidence["provenance"]["source_tree_sha256"],
    }
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["provenance"]["source_tree_sha256"])
    assert evidence["provenance"]["source_tree_sha256"] == benchmark._source_tree_sha256()
    plot._validate_isolated_rss_evidence(evidence)
    isolated_rss = evidence["telemetry_ab"]["isolated_process_rss"]
    assert isolated_rss["comparison_compatible"] is True
    assert 0 <= isolated_rss["nonnegative_delta_bytes"] <= 100 * 1024 * 1024

    profile = evidence["profile"]
    assert profile == {
        "api_burst": 40,
        "api_rate_per_second": 20.0,
        "cursor_bytes": 1024,
        "data_concurrency": 24,
        "header_bytes": 16 * 1024,
        "metric_series_strict_upper_bound": 600,
        "metrics_bytes": 256 * 1024,
        "operations_burst": 4,
        "operations_rate_per_second": 2.0,
        "query_bytes": 4 * 1024,
        "response_bytes": 2 * 1024 * 1024,
        "transport_concurrency": 32,
    }
    assert evidence["candidate_slos"]["status"] == ("aspirational_until_28_day_continuous_evidence")
    expected_names = [
        "cold_start",
        "steady_route_mix",
        "steady_route_mix",
        "maximum_projection",
        "saturation",
        "concurrent_metrics",
        "fault_injection",
    ]
    scenarios = evidence["scenarios"]
    assert [scenario["name"] for scenario in scenarios] == expected_names
    assert {
        scenario["telemetry_mode"]
        for scenario in scenarios
        if scenario["name"] == "steady_route_mix"
    } == {"disabled", "enabled"}

    for scenario in scenarios:
        sample_count = scenario["sample_count"]
        assert sample_count >= 1
        assert sum(scenario["status_counts"].values()) == sample_count
        latency = scenario["latency_ms"]
        assert latency["count"] == sample_count
        assert 0.0 <= latency["min"] <= latency["p50"] <= latency["p95"]
        assert latency["p95"] <= latency["p99"] <= latency["max"]
        assert latency["variance"] >= 0.0
        assert latency["standard_deviation"] >= 0.0
        resources = scenario["resources"]
        assert resources["wall_seconds"] > 0.0
        assert resources["cpu_seconds"] >= 0.0
        assert resources["maximum_rss_after_bytes"] > 0
        assert resources["threads_before"] >= 1
        assert resources["threads_after"] >= 1
        assert resources["processes"] == 1
        assert "raw_samples" not in scenario

    telemetry = evidence["telemetry"]
    assert telemetry["series_count"] < 600
    assert telemetry["exposition_bytes"] <= 256 * 1024
    assert telemetry["telemetry_drop_total"] >= 0.0
    assert re.fullmatch(r"[0-9a-f]{64}", telemetry["content_sha256"])
    saturation = evidence["saturation"]
    assert saturation["probe_status"] == 200
    assert saturation["rejection_count"] >= 1
    assert saturation["maximum_rejection_latency_ms"] <= 250.0
    assert {case["case"] for case in evidence["fault_cases"]} == {
        "readiness_unavailable",
        "sqlite_busy",
        "corrupt_manifest",
        "oversized_query",
        "forbidden_body",
    }
    assert all(case["status"] >= 400 for case in evidence["fault_cases"])
    assert all(bound["observed"] <= bound["limit"] for bound in evidence["bounds_observed"])

    rendered = json.dumps(evidence, sort_keys=True).lower()
    for forbidden in (
        "/users/",
        "authorization",
        "api_key",
        "synthetic-canary",
        "x-forwarded-for",
        "ticker",
        "profit guarantee",
    ):
        assert forbidden not in rendered
    limitations = " ".join(evidence["limitations"]).lower()
    for required in (
        "synthetic",
        "single local process",
        "paper-trading",
        "live-trading",
        "profitability",
        "production",
        "28-day",
    ):
        assert required in limitations


def test_plot_regeneration_is_deterministic_traceable_and_legible(tmp_path: Path) -> None:
    evidence = _load_committed_evidence()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    plot.plot_evidence(evidence, first)
    plot.plot_evidence(evidence, second)

    assert first.read_bytes() == second.read_bytes()
    image = mpimg.imread(first)
    assert image.shape[0] >= 1_500
    assert image.shape[1] >= 2_000
    assert image.shape[2] in {3, 4}
    assert float(image.max()) > float(image.min())
    assert PLOT_PATH.is_file()
    operations = OPERATIONS_PATH.read_text(encoding="utf-8")
    digest_match = re.search(r"service-operability plot SHA-256: `([0-9a-f]{64})`", operations)
    assert digest_match is not None
    assert hashlib.sha256(PLOT_PATH.read_bytes()).hexdigest() == digest_match.group(1)


def test_plot_loader_rejects_oversized_or_wrong_class_evidence(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        plot._load_evidence(oversized)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        json.dumps(
            benchmark._with_integrity(
                {
                    "schema_version": "1.0.0",
                    "evidence_class": "live",
                    "scenarios": [{}],
                    "bounds_observed": [{}],
                }
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="class is unsupported"):
        plot._load_evidence(wrong)

    alias = tmp_path / "evidence-alias.json"
    alias.symlink_to(wrong)
    with pytest.raises(ValueError, match="unable to read"):
        plot._load_evidence(alias)


def test_plot_loader_verifies_integrity_and_rejects_noncanonical_input(tmp_path: Path) -> None:
    base = {
        "schema_version": "1.0.0",
        "evidence_class": "measured_synthetic_local_engineering",
        "scenarios": [{}],
        "bounds_observed": [{}],
        "telemetry_ab": {},
    }
    missing = tmp_path / "missing-integrity.json"
    missing.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        plot._load_evidence(missing)

    tampered_document = benchmark._with_integrity(base)
    tampered_document["schema_version"] = "9.9.9"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(tampered_document), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        plot._load_evidence(tampered)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"integrity":{},"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        plot._load_evidence(nonfinite)

    with pytest.raises(ValueError, match="canonically integrity-bound"):
        plot._validate_integrity(
            {
                "integrity": {
                    "algorithm": "sha256",
                    "canonicalization": (
                        "sorted compact ASCII JSON excluding the integrity member"
                    ),
                    "canonical_payload_sha256": "0" * 64,
                },
                "not_json": {object()},
            }
        )


def test_plot_rss_schema_cross_checks_delta_workload_and_bound() -> None:
    disabled = _isolated_child_result("disabled", after=70 * 1024 * 1024)
    enabled = _isolated_child_result("enabled", after=72 * 1024 * 1024)
    measurement = benchmark._aggregate_isolated_rss_measurements(disabled, enabled)
    evidence = {
        "telemetry_ab": {"isolated_process_rss": measurement},
        "bounds_observed": [
            {
                "name": "telemetry_isolated_rss_delta_bytes",
                "display_name": "Telemetry RSS delta",
                "observed": 2 * 1024 * 1024,
                "limit": benchmark.MAX_TELEMETRY_RSS_DELTA_BYTES,
                "unit": "bytes",
            }
        ],
    }

    plot._validate_isolated_rss_evidence(evidence)

    cast(dict[str, Any], measurement)["nonnegative_delta_bytes"] = 0
    with pytest.raises(ValueError, match="delta violated"):
        plot._validate_isolated_rss_evidence(evidence)


def test_make_target_stages_candidates_without_targeting_committed_references() -> None:
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    section = makefile.split(".PHONY: benchmark-service-operability", maxsplit=1)[1].split(
        ".PHONY: verify-service-operability", maxsplit=1
    )[0]

    assert "mktemp -d build/service-operability.XXXXXX" in section
    assert '--output "$${service_evidence_run}/candidate.json"' in section
    assert '--output "$${service_evidence_run}/candidate.png"' in section
    assert "shasum -a 256" in section
    assert "docs/benchmarks/service_operability_2026-09-06_patch1.json" in section
    assert "docs/assets/service_operability_2026-09-06_patch1.png" in section
    assert "Review retained service-operability candidates" in section
    assert "--output docs/benchmarks" not in section
    assert "--output docs/assets" not in section
