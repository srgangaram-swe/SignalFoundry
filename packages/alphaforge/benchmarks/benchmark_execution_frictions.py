"""Benchmark bounded execution-friction evaluation on synthetic daily-bar inputs.

The timed region contains only repeated :meth:`BarExecutionModel.execute`
calls over an already constructed, deterministic workload.  Fixture creation,
semantic hashing, JSON serialization, and the optional peak-memory pass are
outside that region.  Results describe local Python simulation overhead only;
they are not broker or exchange latency, trading-capacity evidence, observed
execution quality, or a strategy result.

Usage::

    python benchmarks/benchmark_execution_frictions.py --samples 7 --orders 256
    python benchmarks/benchmark_execution_frictions.py \
        --output runs/execution-frictions-benchmark.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from alphaforge.execution.costs import CostModel
from alphaforge.execution.models import BarExecutionModel, ExecutionPolicy, Fill, Order

BENCHMARK_SCHEMA_VERSION: Final = "1.0.0"
FIXTURE_VERSION: Final = "1.0.0"
DEFAULT_SAMPLES: Final = 7
DEFAULT_WARMUPS: Final = 2
DEFAULT_ORDERS_PER_SAMPLE: Final = 256
MAX_SAMPLES: Final = 100
MAX_WARMUPS: Final = 100
MAX_ORDERS_PER_SAMPLE: Final = 10_000
MAX_TOTAL_EVALUATIONS: Final = 1_000_000
MAX_IDENTITY_FILE_BYTES: Final = 20_000_000
INITIAL_EQUITY_USD: Final = 1_000_000.0

Clock = Callable[[], int]
PeakMemoryProbe = Callable[[tuple["ExecutionCase", ...], BarExecutionModel], int | None]


def _file_sha256(path: Path) -> str:
    """Hash one bounded regular repository file used by the benchmark."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"benchmark identity input must be a regular file: {path.name}")
    size = path.stat().st_size
    if size > MAX_IDENTITY_FILE_BYTES:
        raise RuntimeError(
            f"benchmark identity input exceeds {MAX_IDENTITY_FILE_BYTES} bytes: {path.name}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity() -> dict[str, str]:
    """Return content identities for the harness, implementation, and lock."""

    root = Path(__file__).resolve().parents[1]
    inputs = {
        "benchmark_source_sha256": root / "benchmarks" / "benchmark_execution_frictions.py",
        "cost_model_source_sha256": root / "alphaforge" / "execution" / "costs.py",
        "execution_model_source_sha256": root / "alphaforge" / "execution" / "models.py",
        "uv_lock_sha256": root / "uv.lock",
    }
    return {name: _file_sha256(path) for name, path in inputs.items()}


def _bounded_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Resource bounds for one local execution-friction benchmark."""

    samples: int = DEFAULT_SAMPLES
    warmups: int = DEFAULT_WARMUPS
    orders_per_sample: int = DEFAULT_ORDERS_PER_SAMPLE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples",
            _bounded_integer(
                self.samples,
                name="samples",
                minimum=1,
                maximum=MAX_SAMPLES,
            ),
        )
        object.__setattr__(
            self,
            "warmups",
            _bounded_integer(
                self.warmups,
                name="warmups",
                minimum=0,
                maximum=MAX_WARMUPS,
            ),
        )
        object.__setattr__(
            self,
            "orders_per_sample",
            _bounded_integer(
                self.orders_per_sample,
                name="orders_per_sample",
                minimum=1,
                maximum=MAX_ORDERS_PER_SAMPLE,
            ),
        )
        total = (self.samples + self.warmups) * self.orders_per_sample
        if total > MAX_TOTAL_EVALUATIONS:
            raise ValueError(
                f"total evaluation budget exceeds {MAX_TOTAL_EVALUATIONS}; "
                "reduce samples, warmups, or orders"
            )

    @property
    def measured_evaluations(self) -> int:
        """Return the exact number of timed model evaluations."""

        return self.samples * self.orders_per_sample


@dataclass(frozen=True, slots=True)
class ExecutionCase:
    """One immutable, redistribution-safe synthetic execution input."""

    order: Order
    reference_price: float
    lagged_adv_shares: float
    lagged_volatility: float


def build_execution_model() -> BarExecutionModel:
    """Return the frozen composite model exercised by the benchmark.

    Every coefficient is a predeclared synthetic sensitivity.  None is an
    estimate of current broker fees, spreads, market impact, or liquidity.
    """

    costs = CostModel(
        commission_bps=1.25,
        half_spread_bps=2.50,
        slippage_bps=1.75,
        commission_per_share_usd=0.0020,
        minimum_commission_usd=0.05,
        exchange_fee_bps=0.15,
        exchange_fee_per_share_usd=0.0005,
        spread_slippage_multiplier=0.25,
        participation_slippage_bps=4.0,
        participation_slippage_exponent=0.75,
        volatility_slippage_bps_per_1pct=0.50,
        calibration_provenance=(
            "predeclared deterministic synthetic benchmark sensitivity; "
            "not calibrated from strategy outcomes or market observations"
        ),
    )
    policy = ExecutionPolicy(
        adv_lookback=20,
        volatility_lookback=20,
        max_participation_rate=0.05,
        impact_coefficient=0.08,
        impact_exponent=0.50,
        missing_price_policy="raise",
    )
    return BarExecutionModel(costs, policy)


def build_synthetic_workload(orders_per_sample: int) -> tuple[ExecutionCase, ...]:
    """Build a deterministic mix of full, partial, and rejected orders."""

    order_count = _bounded_integer(
        orders_per_sample,
        name="orders_per_sample",
        minimum=1,
        maximum=MAX_ORDERS_PER_SAMPLE,
    )
    decision_date = pd.Timestamp("2025-01-02")
    fill_date = pd.Timestamp("2025-01-03")
    cases: list[ExecutionCase] = []
    for index in range(order_count):
        lagged_adv = float(10_000 + 500 * (index % 13))
        case_kind = index % 4
        if case_kind == 0:
            requested_abs = lagged_adv * 0.010
        elif case_kind == 1:
            requested_abs = lagged_adv * 0.025
        elif case_kind == 2:
            # The frozen 5% participation ceiling makes this a partial fill.
            requested_abs = lagged_adv * 0.080
        else:
            requested_abs = 0.0
        side = 1.0 if index % 2 == 0 else -1.0
        requested_shares = side * requested_abs
        cases.append(
            ExecutionCase(
                order=Order(
                    order_id=index + 1,
                    symbol=f"SYNTH{index % 16:02d}",
                    decision_date=decision_date,
                    fill_date=fill_date,
                    requested_shares=requested_shares,
                    target_weight=0.05 * side if requested_abs > 0.0 else 0.0,
                    pretrade_equity=INITIAL_EQUITY_USD,
                ),
                reference_price=float(50.0 + 0.25 * (index % 17)),
                lagged_adv_shares=lagged_adv,
                lagged_volatility=float(0.010 + 0.002 * (index % 7)),
            )
        )
    return tuple(cases)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object, *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(_canonical_bytes(value))
    return digest.hexdigest()


def _case_record(case: ExecutionCase) -> dict[str, object]:
    return {
        "order_id": case.order.order_id,
        "symbol": case.order.symbol,
        "decision_date": case.order.decision_date.date().isoformat(),
        "fill_date": case.order.fill_date.date().isoformat(),
        "requested_shares": case.order.requested_shares,
        "target_weight": case.order.target_weight,
        "pretrade_equity": case.order.pretrade_equity,
        "reference_price": case.reference_price,
        "lagged_adv_shares": case.lagged_adv_shares,
        "lagged_volatility": case.lagged_volatility,
    }


def workload_digest(cases: tuple[ExecutionCase, ...]) -> str:
    """Return the content identity of the untimed synthetic fixture."""

    return _digest(
        [_case_record(case) for case in cases],
        domain="alphaforge.execution-friction-benchmark-fixture.v1",
    )


def _model_identity(model: BarExecutionModel) -> dict[str, str]:
    cost_digest = model.costs.configuration_digest
    policy_digest = model.policy.configuration_digest
    return {
        "cost_model_sha256": cost_digest,
        "cost_declaration_sha256": model.costs.declaration.digest,
        "execution_policy_sha256": policy_digest,
        "execution_policy_declaration_sha256": model.policy.declaration.digest,
        "combined_execution_model_sha256": _digest(
            {
                "cost_model_sha256": cost_digest,
                "execution_policy_sha256": policy_digest,
            },
            domain="alphaforge.bar-execution-model.v1",
        ),
    }


def _execute_workload(
    cases: tuple[ExecutionCase, ...],
    model: BarExecutionModel,
) -> tuple[Fill, ...]:
    return tuple(
        model.execute(
            case.order,
            reference_price=case.reference_price,
            lagged_adv_shares=case.lagged_adv_shares,
            lagged_volatility=case.lagged_volatility,
        )
        for case in cases
    )


def _fill_record(fill: Fill) -> dict[str, object]:
    components = fill.cost_breakdown.components() if fill.cost_breakdown is not None else ()
    return {
        "order_id": fill.order_id,
        "symbol": fill.symbol,
        "status": fill.status,
        "requested_shares": fill.requested_shares,
        "filled_shares": fill.filled_shares,
        "residual_shares": fill.residual_shares,
        "reference_price": fill.reference_price,
        "fill_price": fill.fill_price,
        "participation_rate": fill.participation_rate,
        "total_cost": fill.total_cost,
        "rejection_reason": fill.rejection_reason,
        "components": [
            {
                "name": name,
                "accounting_path": accounting_path,
                "cost_usd": cost_usd,
                "rate_bps": rate_bps,
            }
            for name, accounting_path, cost_usd, rate_bps in components
        ],
    }


def _semantic_summary(fills: tuple[Fill, ...]) -> dict[str, object]:
    statuses = Counter(fill.status for fill in fills)
    accepted = statuses["filled"] + statuses["partial"]
    event_counts = {
        "fill_applied": accepted,
        "order_accepted": accepted,
        "order_cancelled": statuses["partial"],
        "order_rejected": statuses["rejected"],
        "order_submitted": len(fills),
    }
    records = [_fill_record(fill) for fill in fills]
    return {
        "status_counts": {
            "filled": statuses["filled"],
            "partial": statuses["partial"],
            "rejected": statuses["rejected"],
        },
        # These are the deterministic downstream event equivalents implied by
        # each fill result; the event engine itself is not in the timed region.
        "logical_event_counts": event_counts,
        "logical_event_count": sum(event_counts.values()),
        "gross_filled_shares": math.fsum(abs(fill.filled_shares) for fill in fills),
        "total_modeled_cost_usd": math.fsum(fill.total_cost for fill in fills),
        "semantic_sha256": _digest(
            {"fills": records, "logical_event_counts": event_counts},
            domain="alphaforge.execution-friction-benchmark-semantics.v1",
        ),
    }


def _percentile(sorted_values: tuple[int, ...], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("timing distribution requires at least one observation")
    if any(value < 0 for value in values):
        raise ValueError("timing observations must be non-negative")
    ordered = tuple(sorted(values))
    return {
        "samples": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
    }


def _timed_sample(
    cases: tuple[ExecutionCase, ...],
    model: BarExecutionModel,
    *,
    wall_clock_ns: Clock,
    cpu_clock_ns: Clock,
) -> tuple[int, int, tuple[Fill, ...]]:
    cpu_start = cpu_clock_ns()
    wall_start = wall_clock_ns()
    fills = _execute_workload(cases, model)
    wall_elapsed = wall_clock_ns() - wall_start
    cpu_elapsed = cpu_clock_ns() - cpu_start
    if wall_elapsed <= 0:
        raise RuntimeError("monotonic wall clock reported non-positive elapsed time")
    if cpu_elapsed < 0:
        raise RuntimeError("process CPU clock moved backward")
    return wall_elapsed, cpu_elapsed, fills


def _default_peak_memory_probe(
    cases: tuple[ExecutionCase, ...],
    model: BarExecutionModel,
) -> int | None:
    """Measure Python allocator peak in a separate, untimed execution pass."""

    if tracemalloc.is_tracing():
        return None
    tracemalloc.start()
    try:
        _execute_workload(cases, model)
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def run_benchmark(
    config: BenchmarkConfig,
    *,
    wall_clock_ns: Clock = time.perf_counter_ns,
    cpu_clock_ns: Clock = time.process_time_ns,
    peak_memory_probe: PeakMemoryProbe | None = None,
) -> dict[str, Any]:
    """Run bounded measurements and return a JSON-serializable report.

    Timing is descriptive evidence and is never evaluated against a pass/fail
    threshold.  Injected clocks and a memory probe support contract tests; CLI
    users should retain the production defaults.
    """

    model = build_execution_model()
    cases = build_synthetic_workload(config.orders_per_sample)
    for _ in range(config.warmups):
        _execute_workload(cases, model)

    wall_samples: list[int] = []
    cpu_samples: list[int] = []
    semantic_summaries: list[dict[str, object]] = []
    for _ in range(config.samples):
        wall_elapsed, cpu_elapsed, fills = _timed_sample(
            cases,
            model,
            wall_clock_ns=wall_clock_ns,
            cpu_clock_ns=cpu_clock_ns,
        )
        wall_samples.append(wall_elapsed)
        cpu_samples.append(cpu_elapsed)
        semantic_summaries.append(_semantic_summary(fills))

    first_semantics = semantic_summaries[0]
    semantic_digests = [str(item["semantic_sha256"]) for item in semantic_summaries]
    if len(set(semantic_digests)) != 1:
        raise RuntimeError("identical synthetic samples produced different semantics")
    invariant_fields = (
        "status_counts",
        "logical_event_counts",
        "logical_event_count",
        "gross_filled_shares",
        "total_modeled_cost_usd",
    )
    if any(
        any(item[field] != first_semantics[field] for field in invariant_fields)
        for item in semantic_summaries[1:]
    ):
        raise RuntimeError("identical synthetic samples produced inconsistent aggregate results")

    probe = peak_memory_probe or _default_peak_memory_probe
    peak_memory = probe(cases, model)
    if peak_memory is not None and (
        not isinstance(peak_memory, int) or isinstance(peak_memory, bool) or peak_memory < 0
    ):
        raise RuntimeError("peak-memory probe must return a non-negative integer or None")

    total_wall_ns = sum(wall_samples)
    logical_events_value = first_semantics["logical_event_count"]
    if not isinstance(logical_events_value, int) or isinstance(logical_events_value, bool):
        raise RuntimeError("logical event count must be an integer")
    logical_events_per_sample = logical_events_value
    report: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "deterministic_execution_friction_evaluation",
        "environment": {
            "cpu_logical_count": os.cpu_count(),
            "machine": platform.machine() or "unknown",
            "numpy_version": np.__version__,
            "operating_system": platform.system() or "unknown",
            "operating_system_release": platform.release() or "unknown",
            "pandas_version": pd.__version__,
            "processor": platform.processor() or "unknown",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "code_identity": _code_identity(),
        "model": {
            **_model_identity(model),
            "calibration_provenance": model.costs.calibration_provenance,
        },
        "workload": {
            "source": "deterministic_synthetic",
            "fixture_version": FIXTURE_VERSION,
            "fixture_sha256": workload_digest(cases),
            "cpu_only": True,
            "network_calls": 0,
            "randomness": "none",
            "samples": config.samples,
            "warmups": config.warmups,
            "orders_per_sample": config.orders_per_sample,
            "total_measured_order_evaluations": config.measured_evaluations,
        },
        "measurements": {
            "timed_region": "BarExecutionModel.execute calls over prebuilt inputs",
            "total_wall_elapsed_ns": total_wall_ns,
            "throughput_order_evaluations_per_second": config.measured_evaluations
            / (total_wall_ns / 1_000_000_000.0),
            "per_sample_wall_elapsed_ns": _distribution(wall_samples),
            "per_sample_process_cpu_ns": _distribution(cpu_samples),
            "raw_wall_elapsed_ns": wall_samples,
            "raw_process_cpu_ns": cpu_samples,
            "peak_python_allocated_bytes": peak_memory,
            "peak_memory_method": (
                "tracemalloc_separate_untimed_pass"
                if peak_memory_probe is None
                else "injected_probe"
            ),
        },
        "semantics": {
            "status_counts_per_sample": first_semantics["status_counts"],
            "logical_event_counts_per_sample": first_semantics["logical_event_counts"],
            "total_logical_event_equivalents": config.samples * logical_events_per_sample,
            "gross_filled_shares_per_sample": first_semantics["gross_filled_shares"],
            "total_modeled_cost_usd_per_sample": first_semantics["total_modeled_cost_usd"],
            "semantic_sha256_by_sample": semantic_digests,
        },
        "limitations": [
            (
                "Local synthetic Python simulation-overhead measurement only; "
                "it is not broker or exchange latency."
            ),
            (
                "This is not trading-capacity, fill-probability, market-impact-calibration, "
                "strategy, paper-trading, or live-trading evidence."
            ),
            (
                "Logical event counts are deterministic downstream equivalents inferred from "
                "fill statuses; the event engine is outside the timed region."
            ),
            "Timing is descriptive and has no pass/fail performance threshold.",
            "The separate tracemalloc pass is not included in the reported timing samples.",
        ],
    }
    json.dumps(report, allow_nan=False)
    return report


def write_report(report: dict[str, Any], destination: str | Path) -> None:
    """Write canonical JSON to stdout or atomically create a new artifact.

    File output uses exclusive same-directory publication and therefore never
    overwrites an existing report.  Repository runs belong under ignored
    ``runs/`` storage.
    """

    document = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if str(destination) == "-":
        sys.stdout.write(document)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure local Python execution-friction simulation overhead on synthetic inputs."
        ),
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--orders", type=int, default=DEFAULT_ORDERS_PER_SAMPLE)
    parser.add_argument(
        "--output",
        default="-",
        help="new JSON file to create, or '-' for stdout (default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate CLI inputs, run once, and publish machine-readable evidence."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(
            samples=args.samples,
            warmups=args.warmups,
            orders_per_sample=args.orders,
        )
        report = run_benchmark(config)
        write_report(report, args.output)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
