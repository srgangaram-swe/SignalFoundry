"""Render Signalattice service-operability evidence from aggregate JSON.

The input contains redistribution-safe aggregate synthetic/local engineering
measurements only.  It deliberately omits raw request samples, identifiers,
market observations, credentials, and host paths.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EXPECTED_SCHEMA_VERSION = "1.0.0"
MAX_PLOT_BYTES = 32 * 1024 * 1024
MAX_TELEMETRY_RSS_DELTA_BYTES = 100 * 1024 * 1024
_PUBLICATION_CHUNK_BYTES = 64 * 1024
_PUBLICATION_MODE = 0o644
_INTEGRITY_CANONICALIZATION = "sorted compact ASCII JSON excluding the integrity member"


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _finite_number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return number


def _reject_nonfinite_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constants are forbidden")


def _exact_integer(
    value: object,
    field: str,
    *,
    minimum: int = -(2**63),
    maximum: int = 2**63 - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in its closed range")
    return value


def _canonical_integrity_payload(evidence: dict[str, Any]) -> bytes:
    without_integrity = {key: value for key, value in evidence.items() if key != "integrity"}
    try:
        return json.dumps(
            without_integrity,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("aggregate evidence cannot be canonically integrity-bound") from error


def _validate_integrity(evidence: dict[str, Any]) -> None:
    """Fail closed unless the declared digest binds the exact aggregate document."""

    integrity = _require_mapping(evidence.get("integrity"), "integrity")
    if set(integrity) != {
        "algorithm",
        "canonicalization",
        "canonical_payload_sha256",
    }:
        raise ValueError("aggregate evidence integrity fields violate the closed schema")
    digest = integrity.get("canonical_payload_sha256")
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("canonicalization") != _INTEGRITY_CANONICALIZATION
        or type(digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("aggregate evidence integrity declaration is unsupported")
    observed = hashlib.sha256(_canonical_integrity_payload(evidence)).hexdigest()
    if not hmac.compare_digest(observed, digest):
        raise ValueError("aggregate evidence integrity digest does not match its payload")


def _validate_isolated_rss_arm(
    value: object,
    *,
    expected_mode: str,
    workload_sha256: str,
) -> dict[str, Any]:
    arm = _require_mapping(value, f"isolated RSS {expected_mode} arm")
    expected_fields = {
        "protocol",
        "mode",
        "workload_sha256",
        "runtime_platform",
        "python",
        "python_implementation",
        "rss_source",
        "rss_unit",
        "rss_before_app_bytes",
        "rss_after_shutdown_peak_bytes",
        "workload_growth_bytes",
        "warmup_requests",
        "measured_requests",
        "successful_requests",
        "status_counts",
        "lifespan_shutdown_completed",
        "network_requests",
        "processes",
    }
    if set(arm) != expected_fields:
        raise ValueError("isolated RSS arm fields violate the closed schema")
    if (
        arm.get("protocol") != "signalattice-service-rss-v1"
        or arm.get("mode") != expected_mode
        or arm.get("workload_sha256") != workload_sha256
        or arm.get("rss_source") != "resource.getrusage(RUSAGE_SELF).ru_maxrss"
        or arm.get("rss_unit") != "bytes"
    ):
        raise ValueError("isolated RSS arm is incompatible")
    for field in ("runtime_platform", "python", "python_implementation"):
        text = arm.get(field)
        if type(text) is not str or not text or len(text.encode("utf-8")) > 256:
            raise ValueError(f"isolated RSS {field} is outside its bound")
    before = _exact_integer(arm.get("rss_before_app_bytes"), "isolated RSS before", minimum=1)
    after = _exact_integer(
        arm.get("rss_after_shutdown_peak_bytes"), "isolated RSS after", minimum=1
    )
    growth = _exact_integer(arm.get("workload_growth_bytes"), "isolated RSS growth", minimum=0)
    warmup = _exact_integer(arm.get("warmup_requests"), "isolated RSS warm-up", minimum=0)
    measured = _exact_integer(arm.get("measured_requests"), "isolated RSS measured", minimum=1)
    successful = _exact_integer(
        arm.get("successful_requests"), "isolated RSS successful", minimum=0
    )
    if (
        after < before
        or growth != after - before
        or warmup != 8
        or measured != 16
        or successful != measured
        or arm.get("status_counts") != {"200": measured}
        or arm.get("lifespan_shutdown_completed") is not True
        or type(arm.get("network_requests")) is not int
        or arm.get("network_requests") != 0
        or type(arm.get("processes")) is not int
        or arm.get("processes") != 1
    ):
        raise ValueError("isolated RSS arm outcome is incompatible")
    return arm


def _validate_isolated_rss_evidence(evidence: dict[str, Any]) -> None:
    """Validate the optional additive RSS schema used by newly generated references."""

    telemetry_ab = _require_mapping(evidence.get("telemetry_ab"), "telemetry A/B")
    raw_measurement = telemetry_ab.get("isolated_process_rss")
    if raw_measurement is None:
        # The unchanged 2026-08-09 reference predates this additive schema.
        return
    measurement = _require_mapping(raw_measurement, "isolated process RSS")
    expected_fields = {
        "method",
        "protocol",
        "process_order",
        "hard_timeout_seconds_per_process",
        "common_workload",
        "workload_sha256",
        "disabled",
        "enabled",
        "comparison_compatible",
        "observed_signed_delta_bytes",
        "nonnegative_delta_bytes",
        "limit_bytes",
        "within_limit",
        "negative_delta_policy",
    }
    if set(measurement) != expected_fields:
        raise ValueError("isolated process RSS fields violate the closed schema")
    if (
        measurement.get("method") != "fresh_process_peak_rss_comparison"
        or measurement.get("protocol") != "signalattice-service-rss-v1"
        or measurement.get("process_order") != ["disabled", "enabled"]
        or _finite_number(
            measurement.get("hard_timeout_seconds_per_process"),
            "isolated RSS timeout",
            minimum=1e-12,
        )
        != 20.0
    ):
        raise ValueError("isolated process RSS method is incompatible")
    workload = _require_mapping(measurement.get("common_workload"), "isolated RSS workload")
    workload_sha256 = measurement.get("workload_sha256")
    if type(workload_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", workload_sha256) is None:
        raise ValueError("isolated RSS workload identity is invalid")
    try:
        workload_payload = json.dumps(
            workload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("isolated RSS workload is not canonicalizable") from error
    if not hmac.compare_digest(hashlib.sha256(workload_payload).hexdigest(), workload_sha256):
        raise ValueError("isolated RSS workload identity does not match")
    disabled = _validate_isolated_rss_arm(
        measurement.get("disabled"),
        expected_mode="disabled",
        workload_sha256=workload_sha256,
    )
    enabled = _validate_isolated_rss_arm(
        measurement.get("enabled"),
        expected_mode="enabled",
        workload_sha256=workload_sha256,
    )
    for field in ("runtime_platform", "python", "python_implementation"):
        if disabled[field] != enabled[field]:
            raise ValueError("isolated RSS arms used incompatible runtimes")
    signed = _exact_integer(
        measurement.get("observed_signed_delta_bytes"), "isolated RSS signed delta"
    )
    nonnegative = _exact_integer(
        measurement.get("nonnegative_delta_bytes"), "isolated RSS nonnegative delta", minimum=0
    )
    limit = _exact_integer(measurement.get("limit_bytes"), "isolated RSS limit", minimum=1)
    expected_signed = int(enabled["rss_after_shutdown_peak_bytes"]) - int(
        disabled["rss_after_shutdown_peak_bytes"]
    )
    if (
        signed != expected_signed
        or nonnegative != max(0, signed)
        or limit != MAX_TELEMETRY_RSS_DELTA_BYTES
        or nonnegative > limit
        or measurement.get("comparison_compatible") is not True
        or measurement.get("within_limit") is not True
        or measurement.get("negative_delta_policy")
        != "retain the signed observation; clamp only the incremental-overhead guard to zero"
    ):
        raise ValueError("isolated RSS delta violated its fixed comparison contract")
    matching_bounds = [
        bound
        for bound in evidence.get("bounds_observed", [])
        if type(bound) is dict and bound.get("name") == "telemetry_isolated_rss_delta_bytes"
    ]
    if len(matching_bounds) != 1 or matching_bounds[0] != {
        "name": "telemetry_isolated_rss_delta_bytes",
        "display_name": "Telemetry RSS delta",
        "observed": nonnegative,
        "limit": limit,
        "unit": "bytes",
    }:
        raise ValueError("isolated RSS bound is absent or inconsistent")


def _load_evidence(path: Path) -> dict[str, Any]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 2 * 1024 * 1024:
            raise ValueError("aggregate evidence exceeds the 2 MiB regular-file bound")
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, min(64 * 1024, 2 * 1024 * 1024 + 1 - observed)):
            chunks.append(chunk)
            observed += len(chunk)
            if observed > 2 * 1024 * 1024:
                raise ValueError("aggregate evidence exceeds the 2 MiB regular-file bound")
        after = os.fstat(descriptor)
        if _file_identity(metadata) != _file_identity(after) or observed != after.st_size:
            raise ValueError("aggregate evidence changed while it was read")
        payload = b"".join(chunks)
    except OSError as error:
        raise ValueError("unable to read aggregate evidence") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("aggregate evidence exceeds the 2 MiB regular-file bound")
    try:
        document = json.loads(
            payload,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("aggregate evidence is not valid UTF-8 JSON") from error
    evidence = _require_mapping(document, "evidence")
    _validate_integrity(evidence)
    if evidence.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("aggregate evidence schema version is unsupported")
    if evidence.get("evidence_class") != "measured_synthetic_local_engineering":
        raise ValueError("aggregate evidence class is unsupported")
    scenarios = evidence.get("scenarios")
    bounds = evidence.get("bounds_observed")
    if type(scenarios) is not list or not scenarios:
        raise ValueError("aggregate evidence must contain scenarios")
    if type(bounds) is not list or not bounds:
        raise ValueError("aggregate evidence must contain observed bounds")
    _validate_isolated_rss_evidence(evidence)
    return evidence


def _read_existing_publication(destination: Path) -> bytes:
    """Read one bounded regular plot without following or blocking on its path."""

    descriptor: int | None = None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(destination, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("plot destination must be a regular file")
        if stat.S_IMODE(before.st_mode) != _PUBLICATION_MODE:
            raise ValueError("existing plot destination lacks canonical 0644 permissions")
        if before.st_size > MAX_PLOT_BYTES:
            raise ValueError("existing plot destination exceeds the 32 MiB bound")

        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_PUBLICATION_CHUNK_BYTES, MAX_PLOT_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > MAX_PLOT_BYTES:
                raise ValueError("existing plot destination exceeds the 32 MiB bound")

        after = os.fstat(descriptor)
        rebound = os.stat(destination, follow_symlinks=False)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(rebound)
            or observed != after.st_size
        ):
            raise ValueError("plot destination changed during verification")
        return b"".join(chunks)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("unable to verify existing plot destination") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    """Persist the no-replace directory entry before reporting publication success."""

    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes_no_replace(payload: bytes, destination: Path) -> None:
    """Publish a bounded plot once, or verify an identical regular destination."""

    if type(payload) is not bytes or not payload:
        raise ValueError("plot publication payload must be non-empty bytes")
    if len(payload) > MAX_PLOT_BYTES:
        raise ValueError("plot publication payload exceeds the 32 MiB bound")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor_open = False
            output.write(payload)
            output.flush()
            os.fchmod(output.fileno(), _PUBLICATION_MODE)
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            existing = _read_existing_publication(destination)
            if not hmac.compare_digest(existing, payload):
                raise ValueError("existing plot destination contains different bytes") from None
        except OSError as error:
            raise ValueError("unable to publish plot without replacement") from error
        else:
            _fsync_directory(destination.parent)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _latency_rows(evidence: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_scenario in cast(list[object], evidence["scenarios"]):
        scenario = _require_mapping(raw_scenario, "scenario")
        name = scenario.get("display_name")
        mode = scenario.get("telemetry_mode")
        if type(name) is not str or type(mode) is not str:
            raise ValueError("scenario display name and telemetry mode must be text")
        latency = _require_mapping(scenario.get("latency_ms"), f"{name}.latency_ms")
        quantiles = latency.get("empirical_percentiles")
        if type(quantiles) is not list or len(quantiles) < 5:
            raise ValueError(f"{name} must retain at least five empirical percentiles")
        previous = -1.0
        for raw_point in quantiles:
            point = _require_mapping(raw_point, f"{name}.percentile")
            percentile = _finite_number(point.get("percentile"), "percentile")
            value_ms = _finite_number(point.get("value_ms"), "value_ms")
            if percentile <= previous or percentile > 100.0:
                raise ValueError("empirical percentiles must be strictly increasing through 100")
            previous = percentile
            rows.append(
                {
                    "scenario": name,
                    "telemetry": mode,
                    "percentile": percentile,
                    "latency_ms": max(value_ms, 0.001),
                }
            )
    return pd.DataFrame(rows)


def _outcome_rows(evidence: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_scenario in cast(list[object], evidence["scenarios"]):
        scenario = _require_mapping(raw_scenario, "scenario")
        name = scenario.get("display_name")
        statuses = _require_mapping(scenario.get("status_counts"), "status_counts")
        if type(name) is not str:
            raise ValueError("scenario display name must be text")
        for raw_status, raw_count in statuses.items():
            if type(raw_status) is not str or not raw_status.isdigit():
                raise ValueError("status keys must be decimal strings")
            count = _finite_number(raw_count, "status count")
            code = int(raw_status)
            category = (
                "success"
                if 200 <= code < 300
                else (
                    "bounded rejection"
                    if code in {429, 503}
                    and scenario.get("name") in {"saturation", "concurrent_metrics"}
                    else "fault/error"
                )
            )
            rows.append({"scenario": name, "outcome": category, "count": count})
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["scenario", "outcome"], as_index=False, sort=False)["count"].sum()
        if not frame.empty
        else frame
    )


def _telemetry_rows(evidence: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_scenario in cast(list[object], evidence["scenarios"]):
        scenario = _require_mapping(raw_scenario, "scenario")
        if scenario.get("name") != "steady_route_mix":
            continue
        mode = scenario.get("telemetry_mode")
        latency = _require_mapping(scenario.get("latency_ms"), "steady latency")
        if type(mode) is not str:
            raise ValueError("telemetry mode must be text")
        for percentile, field in (("p50", "p50"), ("p95", "p95"), ("p99", "p99")):
            rows.append(
                {
                    "telemetry": mode,
                    "percentile": percentile,
                    "latency_ms": _finite_number(latency.get(field), field),
                }
            )
    frame = pd.DataFrame(rows)
    if set(frame.get("telemetry", ())) != {"disabled", "enabled"}:
        raise ValueError("steady-route evidence must contain telemetry disabled and enabled")
    return frame


def _bound_rows(evidence: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_bound in cast(list[object], evidence["bounds_observed"]):
        bound = _require_mapping(raw_bound, "observed bound")
        name = bound.get("display_name")
        unit = bound.get("unit")
        if type(name) is not str or type(unit) is not str:
            raise ValueError("bound name and unit must be text")
        observed = _finite_number(bound.get("observed"), f"{name}.observed")
        limit = _finite_number(bound.get("limit"), f"{name}.limit", minimum=1e-12)
        rows.append(
            {
                "bound": name,
                "utilization_percent": observed / limit * 100.0,
                "label": f"{observed:,.3g} / {limit:,.3g} {unit}",
                "state": "within bound" if observed <= limit else "exceeded",
            }
        )
    return pd.DataFrame(rows)


def plot_evidence(evidence: dict[str, Any], destination: Path) -> None:
    """Render one deterministic four-panel figure from validated aggregate evidence."""

    _validate_integrity(evidence)
    _validate_isolated_rss_evidence(evidence)
    if destination.is_symlink():
        raise ValueError("plot destination may not be a symlink")
    latency = _latency_rows(evidence)
    outcomes = _outcome_rows(evidence)
    telemetry = _telemetry_rows(evidence)
    bounds = _bound_rows(evidence)
    sns.set_theme(context="notebook", style="whitegrid", palette="colorblind", font_scale=0.92)
    palette = sns.color_palette("colorblind")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5))

    latency_axis = axes[0, 0]
    sns.lineplot(
        data=latency,
        x="percentile",
        y="latency_ms",
        hue="scenario",
        style="telemetry",
        markers=True,
        dashes=False,
        linewidth=2,
        ax=latency_axis,
    )
    latency_axis.axhline(100.0, color=palette[2], linestyle="--", linewidth=1.2)
    latency_axis.axhline(250.0, color=palette[3], linestyle=":", linewidth=1.2)
    latency_axis.axhline(500.0, color=palette[4], linestyle="-.", linewidth=1.2)
    latency_axis.text(1.0, 105.0, "candidate 100 ms", color=palette[2], fontsize=8)
    latency_axis.text(1.0, 263.0, "candidate 250 ms", color=palette[3], fontsize=8)
    latency_axis.text(1.0, 525.0, "candidate 500 ms", color=palette[4], fontsize=8)
    latency_axis.set_yscale("log")
    latency_axis.set_xlim(0.0, 100.0)
    latency_axis.set_title("Aggregate empirical latency percentiles")
    latency_axis.set_xlabel("Empirical percentile (%)")
    latency_axis.set_ylabel("End-to-end wall latency (ms, log scale)")
    latency_axis.legend(title="Scenario / telemetry", fontsize=7, title_fontsize=8)

    outcome_axis = axes[0, 1]
    sns.barplot(
        data=outcomes,
        x="scenario",
        y="count",
        hue="outcome",
        palette="colorblind",
        errorbar=None,
        ax=outcome_axis,
    )
    outcome_axis.set_title("Successes, bounded rejections, and injected faults")
    outcome_axis.set_xlabel("Synthetic local scenario")
    outcome_axis.set_ylabel("Response count")
    outcome_axis.tick_params(axis="x", rotation=25)
    outcome_axis.legend(title="Outcome", fontsize=8)

    telemetry_axis = axes[1, 0]
    sns.barplot(
        data=telemetry,
        x="percentile",
        y="latency_ms",
        hue="telemetry",
        palette="colorblind",
        errorbar=None,
        ax=telemetry_axis,
    )
    telemetry_axis.set_title("Steady route mix: telemetry A/B")
    telemetry_axis.set_xlabel("Aggregate percentile")
    telemetry_axis.set_ylabel("End-to-end wall latency (ms)")
    telemetry_axis.legend(title="Telemetry")
    telemetry_axis.set_ylim(bottom=0.0)

    bound_axis = axes[1, 1]
    sns.barplot(
        data=bounds,
        y="bound",
        x="utilization_percent",
        hue="state",
        palette={"within bound": palette[2], "exceeded": palette[3]},
        errorbar=None,
        dodge=False,
        ax=bound_axis,
    )
    bound_axis.axvline(100.0, color="black", linestyle="--", linewidth=1.2)
    maximum = max(110.0, float(bounds["utilization_percent"].max()) * 1.2)
    bound_axis.set_xlim(0.0, maximum)
    bound_axis.set_title("Observed safety-bound utilization (lower is better)")
    bound_axis.set_xlabel("Observed / fixed limit (%)")
    bound_axis.set_ylabel("")
    rendered_bars = [
        patch
        for patch in bound_axis.patches
        if math.isfinite(float(patch.get_width())) and float(patch.get_width()) > 0.0
    ]
    if (bounds["utilization_percent"] > 0.0).all():
        if len(rendered_bars) != len(bounds):
            raise ValueError("Seaborn did not render one bar for each observed bound")
        annotations = [
            (patch.get_width(), patch.get_y() + patch.get_height() / 2.0, label)
            for patch, label in zip(rendered_bars, bounds["label"], strict=True)
        ]
    else:
        # Seaborn deliberately renders an exact zero as a zero-width patch.
        # Annotate by categorical row so the honest zero remains visible.
        annotations = [
            (float(row.utilization_percent), float(index), str(row.label))
            for index, row in enumerate(bounds.itertuples(index=False))
        ]
    for x_value, y_value, label in annotations:
        bound_axis.annotate(
            label,
            (x_value, y_value),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    bound_axis.legend(title="Result", loc="lower right")

    workload = _require_mapping(evidence.get("workload"), "workload")
    environment = _require_mapping(evidence.get("environment"), "environment")
    integrity = _require_mapping(evidence.get("integrity"), "integrity")
    telemetry_ab = _require_mapping(evidence.get("telemetry_ab"), "telemetry A/B")
    measurement_context = (
        "Measured local ASGI mechanics; isolated RSS uses fresh child processes—"
        if telemetry_ab.get("isolated_process_rss") is not None
        else "Measured in-process ASGI mechanics only—"
    )
    figure.suptitle(
        "Signalattice bounded evidence service — synthetic local engineering evidence",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.012,
        (
            f"Seed {workload.get('seed')}; warm-up {workload.get('warmup_requests')} requests; "
            f"{environment.get('platform')} / Python {environment.get('python')}; "
            f"aggregate source {str(integrity.get('canonical_payload_sha256', ''))[:12]}. "
            f"{measurement_context}no network, market data, paper/live trading, "
            "profitability, production readiness, market-scale capacity, or proven 28-day SLO."
        ),
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.96))
    try:
        rendered = io.BytesIO()
        figure.savefig(
            rendered,
            format="png",
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "Signalattice deterministic Seaborn evidence renderer"},
        )
        _publish_bytes_no_replace(rendered.getvalue(), destination)
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot committed aggregate service-operability evidence through Seaborn."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    evidence = _load_evidence(arguments.input)
    plot_evidence(evidence, arguments.output)


if __name__ == "__main__":
    main()
