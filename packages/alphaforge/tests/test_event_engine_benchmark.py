"""Contract tests for the bounded synthetic event-engine benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "bench_event_engine.py"
    spec = importlib.util.spec_from_file_location("alphaforge_event_engine_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the event-engine benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_script()
BENCHMARK_SCHEMA_VERSION = BENCHMARK.BENCHMARK_SCHEMA_VERSION
MAX_ORDERS_PER_SAMPLE = BENCHMARK.MAX_ORDERS_PER_SAMPLE
MAX_SAMPLES = BENCHMARK.MAX_SAMPLES
MAX_TOTAL_EVENTS = BENCHMARK.MAX_TOTAL_EVENTS
MAX_WARMUPS = BENCHMARK.MAX_WARMUPS
BenchmarkConfig = BENCHMARK.BenchmarkConfig
build_synthetic_stream = BENCHMARK.build_synthetic_stream
event_type_counts = BENCHMARK.event_type_counts
run_benchmark = BENCHMARK.run_benchmark
write_report = BENCHMARK.write_report


def test_synthetic_stream_has_byte_deterministic_semantics() -> None:
    first = build_synthetic_stream("benchmark-contract", 3)
    second = build_synthetic_stream("benchmark-contract", 3)

    assert tuple(event.canonical_bytes() for event in first) == tuple(
        event.canonical_bytes() for event in second
    )
    assert event_type_counts(first) == {
        "fill_applied": 3,
        "order_accepted": 3,
        "order_submitted": 3,
        "portfolio_marked": 4,
        "signal_available": 1,
        "target_decided": 1,
    }
    assert len(first) == BenchmarkConfig(orders_per_sample=3).events_per_sample


def test_report_schema_and_semantic_counts_do_not_gate_on_wall_time() -> None:
    config = BenchmarkConfig(samples=2, warmups=1, orders_per_sample=3)
    report = run_benchmark(config)

    assert report["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert report["benchmark"] == "deterministic_event_engine_ordered_process"
    assert report["workload"] == {
        "source": "deterministic_synthetic",
        "fixture_version": "1.0.0",
        "cpu_only": True,
        "randomness": "none",
        "samples": 2,
        "warmups": 1,
        "orders_per_sample": 3,
        "events_per_sample": 15,
        "total_measured_events": 30,
        "event_type_counts_per_sample": {
            "fill_applied": 3,
            "order_accepted": 3,
            "order_submitted": 3,
            "portfolio_marked": 4,
            "signal_available": 1,
            "target_decided": 1,
        },
    }
    assert report["semantics"]["committed_events"] == 30
    assert report["semantics"]["final_filled_orders"] == 6
    assert report["semantics"]["expected_final_equity_per_sample"] == 1_000_003.0
    assert len(report["semantics"]["measured_stream_sha256"]) == 64
    assert len(report["semantics"]["journal_head_sha256_by_sample"]) == 2
    latency = report["measurements"]["per_event_latency_ns"]
    assert set(latency) == {"samples", "min", "p50", "p95", "p99", "max", "mean"}
    assert latency["samples"] == 30
    assert "pass" not in report["measurements"]
    assert "threshold" not in report["measurements"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("samples", 0),
        ("samples", MAX_SAMPLES + 1),
        ("samples", True),
        ("warmups", -1),
        ("warmups", MAX_WARMUPS + 1),
        ("orders_per_sample", 0),
        ("orders_per_sample", MAX_ORDERS_PER_SAMPLE + 1),
    ],
)
def test_benchmark_resource_bounds_fail_closed(field: str, value: object) -> None:
    settings: dict[str, object] = {
        "samples": 1,
        "warmups": 0,
        "orders_per_sample": 1,
    }
    settings[field] = value
    with pytest.raises(ValueError, match=field):
        BenchmarkConfig(**settings)


def test_combined_workload_respects_total_event_budget() -> None:
    with pytest.raises(ValueError, match=f"total event budget exceeds {MAX_TOTAL_EVENTS}"):
        BenchmarkConfig(
            samples=MAX_SAMPLES,
            warmups=MAX_WARMUPS,
            orders_per_sample=MAX_ORDERS_PER_SAMPLE,
        )


def test_report_writer_refuses_to_overwrite_an_artifact(tmp_path) -> None:
    target = tmp_path / "benchmark.json"
    report = {"schema_version": BENCHMARK_SCHEMA_VERSION}

    write_report(report, target)
    with pytest.raises(FileExistsError):
        write_report(report, target)

    assert json.loads(target.read_text()) == report
