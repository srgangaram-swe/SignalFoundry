"""Contract tests for the bounded synthetic execution-friction benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_benchmark() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_execution_frictions.py"
    spec = importlib.util.spec_from_file_location(
        "alphaforge_execution_friction_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the execution-friction benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()
BENCHMARK_SCHEMA_VERSION = BENCHMARK.BENCHMARK_SCHEMA_VERSION
MAX_ORDERS_PER_SAMPLE = BENCHMARK.MAX_ORDERS_PER_SAMPLE
MAX_SAMPLES = BENCHMARK.MAX_SAMPLES
MAX_TOTAL_EVALUATIONS = BENCHMARK.MAX_TOTAL_EVALUATIONS
MAX_WARMUPS = BENCHMARK.MAX_WARMUPS
BenchmarkConfig = BENCHMARK.BenchmarkConfig
build_execution_model = BENCHMARK.build_execution_model
build_synthetic_workload = BENCHMARK.build_synthetic_workload
run_benchmark = BENCHMARK.run_benchmark
workload_digest = BENCHMARK.workload_digest
write_report = BENCHMARK.write_report


def _clock(values: Iterable[int]) -> Callable[[], int]:
    iterator = iter(values)
    return lambda: next(iterator)


def _run_small_benchmark() -> dict[str, Any]:
    return run_benchmark(
        BenchmarkConfig(samples=2, warmups=1, orders_per_sample=8),
        wall_clock_ns=_clock((100, 2_100, 3_000, 7_000)),
        cpu_clock_ns=_clock((50, 1_050, 2_000, 4_000)),
        peak_memory_probe=lambda _cases, _model: 4_096,
    )


def test_synthetic_fixture_and_model_identities_are_deterministic() -> None:
    first_cases = build_synthetic_workload(8)
    second_cases = build_synthetic_workload(8)
    first_model = build_execution_model()
    second_model = build_execution_model()

    assert first_cases == second_cases
    assert workload_digest(first_cases) == workload_digest(second_cases)
    assert first_model.costs == second_model.costs
    assert first_model.policy == second_model.policy

    first = _run_small_benchmark()
    second = _run_small_benchmark()
    assert first["model"] == second["model"]
    assert first["workload"]["fixture_sha256"] == second["workload"]["fixture_sha256"]
    assert (
        first["semantics"]["semantic_sha256_by_sample"]
        == second["semantics"]["semantic_sha256_by_sample"]
    )


def test_report_schema_counts_hashes_and_measurement_boundaries() -> None:
    report = _run_small_benchmark()

    assert report["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert report["benchmark"] == "deterministic_execution_friction_evaluation"
    assert report["workload"] == {
        "source": "deterministic_synthetic",
        "fixture_version": "1.0.0",
        "fixture_sha256": report["workload"]["fixture_sha256"],
        "cpu_only": True,
        "network_calls": 0,
        "randomness": "none",
        "samples": 2,
        "warmups": 1,
        "orders_per_sample": 8,
        "total_measured_order_evaluations": 16,
    }
    assert len(report["workload"]["fixture_sha256"]) == 64
    assert all(
        len(report["model"][field]) == 64
        for field in (
            "cost_model_sha256",
            "cost_declaration_sha256",
            "execution_policy_sha256",
            "execution_policy_declaration_sha256",
            "combined_execution_model_sha256",
        )
    )

    semantics = report["semantics"]
    assert semantics["status_counts_per_sample"] == {
        "filled": 4,
        "partial": 2,
        "rejected": 2,
    }
    assert semantics["logical_event_counts_per_sample"] == {
        "fill_applied": 6,
        "order_accepted": 6,
        "order_cancelled": 2,
        "order_rejected": 2,
        "order_submitted": 8,
    }
    assert semantics["total_logical_event_equivalents"] == 48
    assert semantics["gross_filled_shares_per_sample"] > 0.0
    assert semantics["total_modeled_cost_usd_per_sample"] > 0.0
    assert len(semantics["semantic_sha256_by_sample"]) == 2
    assert len(set(semantics["semantic_sha256_by_sample"])) == 1
    assert len(semantics["semantic_sha256_by_sample"][0]) == 64

    measurements = report["measurements"]
    assert measurements["timed_region"] == ("BarExecutionModel.execute calls over prebuilt inputs")
    assert measurements["per_sample_wall_elapsed_ns"]["samples"] == 2
    assert measurements["per_sample_process_cpu_ns"]["samples"] == 2
    assert measurements["raw_wall_elapsed_ns"] == [2_000, 4_000]
    assert measurements["raw_process_cpu_ns"] == [1_000, 2_000]
    assert measurements["peak_python_allocated_bytes"] == 4_096
    assert measurements["peak_memory_method"] == "injected_probe"
    assert measurements["throughput_order_evaluations_per_second"] > 0.0
    assert "pass" not in measurements
    assert "threshold" not in measurements
    assert set(report["code_identity"]) == {
        "benchmark_source_sha256",
        "cost_model_source_sha256",
        "execution_model_source_sha256",
        "uv_lock_sha256",
    }
    assert all(len(value) == 64 for value in report["code_identity"].values())
    assert any("not broker or exchange latency" in item for item in report["limitations"])
    assert any("not trading-capacity" in item for item in report["limitations"])
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
        ("orders_per_sample", False),
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


def test_combined_workload_respects_total_evaluation_budget() -> None:
    with pytest.raises(
        ValueError,
        match=f"total evaluation budget exceeds {MAX_TOTAL_EVALUATIONS}",
    ):
        BenchmarkConfig(
            samples=MAX_SAMPLES,
            warmups=MAX_WARMUPS,
            orders_per_sample=MAX_ORDERS_PER_SAMPLE,
        )


def test_non_positive_wall_time_and_invalid_memory_probe_fail_closed() -> None:
    config = BenchmarkConfig(samples=1, warmups=0, orders_per_sample=1)
    with pytest.raises(RuntimeError, match="non-positive elapsed time"):
        run_benchmark(
            config,
            wall_clock_ns=_clock((100, 100)),
            cpu_clock_ns=_clock((50, 60)),
            peak_memory_probe=lambda _cases, _model: 0,
        )

    with pytest.raises(RuntimeError, match="peak-memory probe"):
        run_benchmark(
            config,
            wall_clock_ns=_clock((100, 200)),
            cpu_clock_ns=_clock((50, 60)),
            peak_memory_probe=lambda _cases, _model: -1,
        )


def test_report_writer_refuses_to_overwrite_an_artifact(tmp_path: Path) -> None:
    target = tmp_path / "benchmark.json"
    report = _run_small_benchmark()

    write_report(report, target)
    with pytest.raises(FileExistsError):
        write_report(report, target)

    assert json.loads(target.read_text(encoding="utf-8")) == report
