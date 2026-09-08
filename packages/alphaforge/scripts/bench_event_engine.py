"""Benchmark deterministic event reduction with a bounded synthetic workload.

The benchmark measures ordered ``DeterministicEventEngine.process`` calls,
including canonical in-memory journal appends and portfolio accounting.  Event
construction and engine setup happen outside the timed region.  Per-event
latency sampling includes the cost of ``perf_counter_ns`` and therefore must be
interpreted as Python-observed call latency, not exchange or broker latency.

Usage::

    python scripts/bench_event_engine.py --samples 7 --warmups 2 --orders 256
    python scripts/bench_event_engine.py --output runs/event-engine-benchmark.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from alphaforge.backtesting.event_engine import DeterministicEventEngine, OrderLifecycle
from alphaforge.execution.events import (
    EventCoordinate,
    EventPayload,
    EventPhase,
    ExecutionEvent,
    FillApplied,
    OrderAccepted,
    OrderSubmitted,
    PortfolioMarked,
    SignalAvailable,
    TargetDecided,
)

BENCHMARK_SCHEMA_VERSION: Final = "1.0.0"
FIXTURE_VERSION: Final = "1.0.0"
DEFAULT_SAMPLES: Final = 7
DEFAULT_WARMUPS: Final = 2
DEFAULT_ORDERS_PER_SAMPLE: Final = 256
MAX_SAMPLES: Final = 100
MAX_WARMUPS: Final = 100
MAX_ORDERS_PER_SAMPLE: Final = 10_000
MAX_TOTAL_EVENTS: Final = 1_000_000
INITIAL_CASH: Final = 1_000_000.0
OPEN_PRICE: Final = 100.0
CLOSE_PRICE: Final = 101.0
_DIGEST_A: Final = "a" * 64
_DIGEST_B: Final = "b" * 64
_DIGEST_C: Final = "c" * 64
_FIRST_SESSION: Final = date(2025, 1, 2)
_SECOND_SESSION: Final = date(2025, 1, 3)


def _integer_in_range(
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
    """Validated bounds for one local CPU benchmark invocation."""

    samples: int = DEFAULT_SAMPLES
    warmups: int = DEFAULT_WARMUPS
    orders_per_sample: int = DEFAULT_ORDERS_PER_SAMPLE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples",
            _integer_in_range(
                self.samples,
                name="samples",
                minimum=1,
                maximum=MAX_SAMPLES,
            ),
        )
        object.__setattr__(
            self,
            "warmups",
            _integer_in_range(
                self.warmups,
                name="warmups",
                minimum=0,
                maximum=MAX_WARMUPS,
            ),
        )
        object.__setattr__(
            self,
            "orders_per_sample",
            _integer_in_range(
                self.orders_per_sample,
                name="orders_per_sample",
                minimum=1,
                maximum=MAX_ORDERS_PER_SAMPLE,
            ),
        )
        total_events = (self.samples + self.warmups) * self.events_per_sample
        if total_events > MAX_TOTAL_EVENTS:
            raise ValueError(
                f"total event budget exceeds {MAX_TOTAL_EVENTS}; reduce samples, warmups, or orders"
            )

    @property
    def events_per_sample(self) -> int:
        """Return the exact event count in each deterministic stream."""
        return 6 + 3 * self.orders_per_sample


def _event(
    *,
    run_id: str,
    payload: EventPayload,
    session: date,
    bar_index: int,
    phase: EventPhase,
    ordinal: int = 0,
    correlation_id: str = "synthetic-chain",
    entity_id: str,
    causation_id: str | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        run_id=run_id,
        correlation_id=correlation_id,
        entity_id=entity_id,
        coordinate=EventCoordinate(
            session=session,
            bar_index=bar_index,
            phase=phase,
            ordinal=ordinal,
        ),
        payload=payload,
        causation_id=causation_id,
    )


def _mark(
    *,
    run_id: str,
    mark_id: str,
    mark_type: Literal["open", "close"],
    session: date,
    bar_index: int,
    price: float,
    cash: float,
    holdings_value: float,
    equity: float,
) -> ExecutionEvent:
    phase = EventPhase.OPEN_MARK if mark_type == "open" else EventPhase.CLOSE_MARK
    return _event(
        run_id=run_id,
        payload=PortfolioMarked(
            mark_id=mark_id,
            mark_type=mark_type,
            prices=(("SYNTH", price),),
            cash=cash,
            holdings_value=holdings_value,
            accrued_charges=0.0,
            equity=equity,
        ),
        session=session,
        bar_index=bar_index,
        phase=phase,
        correlation_id="mark-chain",
        entity_id=mark_id,
    )


def build_synthetic_stream(run_id: str, orders_per_sample: int) -> tuple[ExecutionEvent, ...]:
    """Build a byte-deterministic two-session, all-synthetic event stream.

    The workload intentionally contains no random generator, network call,
    market observation, strategy candidate, or wall-clock timestamp.  One
    close-time target becomes eligible on the following session and causes a
    bounded set of fully filled buy orders.  The final mark independently
    reconciles cash, holdings, and equity.
    """
    order_count = _integer_in_range(
        orders_per_sample,
        name="orders_per_sample",
        minimum=1,
        maximum=MAX_ORDERS_PER_SAMPLE,
    )
    events: list[ExecutionEvent] = [
        _mark(
            run_id=run_id,
            mark_id="mark-open-000000",
            mark_type="open",
            session=_FIRST_SESSION,
            bar_index=0,
            price=OPEN_PRICE,
            cash=INITIAL_CASH,
            holdings_value=0.0,
            equity=INITIAL_CASH,
        ),
        _mark(
            run_id=run_id,
            mark_id="mark-close-000000",
            mark_type="close",
            session=_FIRST_SESSION,
            bar_index=0,
            price=OPEN_PRICE,
            cash=INITIAL_CASH,
            holdings_value=0.0,
            equity=INITIAL_CASH,
        ),
    ]
    signal = _event(
        run_id=run_id,
        payload=SignalAvailable(
            signal_id="signal-000000",
            model_id="synthetic-noncandidate",
            signal_digest=_DIGEST_A,
        ),
        session=_FIRST_SESSION,
        bar_index=0,
        phase=EventPhase.SIGNAL,
        entity_id="signal-000000",
    )
    events.append(signal)
    target_asset_weight = order_count * OPEN_PRICE / INITIAL_CASH
    target = _event(
        run_id=run_id,
        payload=TargetDecided(
            target_id="target-000000",
            portfolio_id="synthetic-benchmark",
            solver_id="no-solver",
            eligible_session=_SECOND_SESSION,
            cash_weight=1.0 - target_asset_weight,
            weights=(("SYNTH", target_asset_weight),),
            configuration_digest=_DIGEST_A,
            data_digest=_DIGEST_B,
            problem_digest=_DIGEST_C,
        ),
        session=_FIRST_SESSION,
        bar_index=0,
        phase=EventPhase.TARGET_DECISION,
        entity_id="target-000000",
        causation_id=signal.event_id,
    )
    events.extend(
        (
            target,
            _mark(
                run_id=run_id,
                mark_id="mark-open-000001",
                mark_type="open",
                session=_SECOND_SESSION,
                bar_index=1,
                price=OPEN_PRICE,
                cash=INITIAL_CASH,
                holdings_value=0.0,
                equity=INITIAL_CASH,
            ),
        )
    )

    submissions: list[ExecutionEvent] = []
    executions: list[ExecutionEvent] = []
    for index in range(order_count):
        identifier = f"{index:06d}"
        order_id = f"order-{identifier}"
        submitted = _event(
            run_id=run_id,
            payload=OrderSubmitted(
                order_id=order_id,
                symbol="SYNTH",
                side="buy",
                quantity=1.0,
            ),
            session=_SECOND_SESSION,
            bar_index=1,
            phase=EventPhase.ORDER_SUBMISSION,
            ordinal=index,
            entity_id=order_id,
            causation_id=target.event_id,
        )
        accepted = _event(
            run_id=run_id,
            payload=OrderAccepted(order_id=order_id, accepted_quantity=1.0),
            session=_SECOND_SESSION,
            bar_index=1,
            phase=EventPhase.EXECUTION,
            ordinal=2 * index,
            entity_id=order_id,
            causation_id=submitted.event_id,
        )
        fill_id = f"fill-{identifier}"
        fill = _event(
            run_id=run_id,
            payload=FillApplied(
                fill_id=fill_id,
                order_id=order_id,
                symbol="SYNTH",
                side="buy",
                quantity=1.0,
                reference_price=OPEN_PRICE,
                price=OPEN_PRICE,
            ),
            session=_SECOND_SESSION,
            bar_index=1,
            phase=EventPhase.EXECUTION,
            ordinal=2 * index + 1,
            entity_id=fill_id,
            causation_id=accepted.event_id,
        )
        submissions.append(submitted)
        executions.extend((accepted, fill))

    ending_cash = INITIAL_CASH - order_count * OPEN_PRICE
    ending_holdings = order_count * CLOSE_PRICE
    events.extend(submissions)
    events.extend(executions)
    events.append(
        _mark(
            run_id=run_id,
            mark_id="mark-close-000001",
            mark_type="close",
            session=_SECOND_SESSION,
            bar_index=1,
            price=CLOSE_PRICE,
            cash=ending_cash,
            holdings_value=ending_holdings,
            equity=ending_cash + ending_holdings,
        )
    )
    return tuple(events)


def event_type_counts(events: Sequence[ExecutionEvent]) -> dict[str, int]:
    """Return stable, lexicographically ordered semantic event counts."""
    counts = Counter(event.event_type for event in events)
    return dict(sorted(counts.items()))


class _ByteDigest(Protocol):
    def update(self, value: bytes) -> None:
        """Consume more bytes into the digest state."""


def _update_stream_digest(digest: _ByteDigest, stream: Sequence[ExecutionEvent]) -> None:
    """Length-prefix one stream into an incremental SHA-256 digest."""
    for event in stream:
        encoded = event.canonical_bytes()
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)


def _percentile(sorted_values: Sequence[int], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower]))


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("latency distribution requires at least one observation")
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _run_stream(
    stream: Sequence[ExecutionEvent],
    *,
    clock_ns: Callable[[], int] | None,
) -> tuple[int, tuple[int, ...], str]:
    order_count = sum(isinstance(event.payload, OrderSubmitted) for event in stream)
    engine = DeterministicEventEngine(
        stream[0].run_id,
        calendar=(_FIRST_SESSION, _SECOND_SESSION),
        initial_cash=INITIAL_CASH,
        max_events=len(stream),
        max_orders=order_count,
        max_open_orders=order_count,
    )
    try:
        observed_latencies: list[int] = []
        start = time.perf_counter_ns()
        if clock_ns is None:
            for event in stream:
                engine.process(event)
        else:
            for event in stream:
                event_start = clock_ns()
                engine.process(event)
                latency = clock_ns() - event_start
                if latency < 0:
                    raise RuntimeError("per-event benchmark clock moved backward")
                observed_latencies.append(latency)
        elapsed = time.perf_counter_ns() - start

        state = engine.snapshot()
        if state.processed_events != len(stream) or engine.journal.count != len(stream):
            raise RuntimeError("event engine did not commit the complete synthetic stream")
        if len(state.orders) != order_count or any(
            order.status is not OrderLifecycle.FILLED for order in state.orders.values()
        ):
            raise RuntimeError("synthetic order lifecycle did not terminate in FILLED state")
        expected_equity = INITIAL_CASH + order_count * (CLOSE_PRICE - OPEN_PRICE)
        if state.portfolio is None or not math.isclose(
            state.portfolio.equity,
            expected_equity,
            rel_tol=0.0,
            abs_tol=math.ulp(expected_equity) * 8.0,
        ):
            raise RuntimeError("synthetic final portfolio mark did not reconcile")
        return elapsed, tuple(observed_latencies), engine.journal.head_hash.hex()
    finally:
        engine.close()


def run_benchmark(
    config: BenchmarkConfig,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run the benchmark and return one JSON-serializable evidence document.

    Timing values are observations, never pass/fail gates.  The injectable
    clock supports instrumentation tests; production callers should retain the
    default monotonic high-resolution clock.
    """
    for index in range(config.warmups):
        warmup = build_synthetic_stream(
            f"event-bench-warmup-{index:03d}",
            config.orders_per_sample,
        )
        _run_stream(warmup, clock_ns=None)

    sample_elapsed: list[int] = []
    event_latencies: list[int] = []
    head_hashes: list[str] = []
    stream_digest = hashlib.sha256()
    first_counts: dict[str, int] | None = None
    for index in range(config.samples):
        stream = build_synthetic_stream(
            f"event-bench-sample-{index:03d}",
            config.orders_per_sample,
        )
        _update_stream_digest(stream_digest, stream)
        if first_counts is None:
            first_counts = event_type_counts(stream)
        elapsed, latencies, head_hash = _run_stream(stream, clock_ns=clock_ns)
        sample_elapsed.append(elapsed)
        event_latencies.extend(latencies)
        head_hashes.append(head_hash)

    total_events = config.samples * config.events_per_sample
    total_elapsed = sum(sample_elapsed)
    if total_elapsed <= 0:
        raise RuntimeError("monotonic benchmark clock reported non-positive elapsed time")
    if first_counts is None:  # BenchmarkConfig requires at least one sample.
        raise RuntimeError("benchmark produced no measured stream")
    report: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "deterministic_event_engine_ordered_process",
        "environment": {
            "cpu_logical_count": os.cpu_count(),
            "machine": platform.machine() or "unknown",
            "operating_system": platform.system() or "unknown",
            "operating_system_release": platform.release() or "unknown",
            "processor": platform.processor() or "unknown",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "workload": {
            "source": "deterministic_synthetic",
            "fixture_version": FIXTURE_VERSION,
            "cpu_only": True,
            "randomness": "none",
            "samples": config.samples,
            "warmups": config.warmups,
            "orders_per_sample": config.orders_per_sample,
            "events_per_sample": config.events_per_sample,
            "total_measured_events": total_events,
            "event_type_counts_per_sample": first_counts,
        },
        "measurements": {
            "total_elapsed_ns": total_elapsed,
            "throughput_events_per_second": total_events / (total_elapsed / 1_000_000_000.0),
            "per_event_latency_ns": _distribution(event_latencies),
            "per_sample_elapsed_ns": _distribution(sample_elapsed),
        },
        "semantics": {
            "measured_stream_sha256": stream_digest.hexdigest(),
            "journal_head_sha256_by_sample": head_hashes,
            "committed_events": total_events,
            "final_filled_orders": config.samples * config.orders_per_sample,
            "expected_final_equity_per_sample": INITIAL_CASH
            + config.orders_per_sample * (CLOSE_PRICE - OPEN_PRICE),
        },
        "limitations": [
            "Synthetic reducer benchmark; it is not strategy, market, paper, or live evidence.",
            "Python-observed per-event latency includes perf_counter_ns instrumentation overhead.",
            "In-memory journaling excludes SQLite durability, network, broker, and exchange costs.",
            "Timing is descriptive and has no pass/fail performance threshold.",
        ],
    }
    # Fail before publication if a platform ever contributes a non-finite value.
    json.dumps(report, allow_nan=False)
    return report


def write_report(report: dict[str, Any], destination: str | Path) -> None:
    """Write canonicalized JSON to stdout or create a new local artifact.

    File output uses exclusive creation so an existing evidence artifact is
    never silently overwritten.  Callers choose a fresh ignored ``runs/`` path
    for each measurement environment.
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
        # A same-directory hard link atomically publishes without the overwrite
        # behavior of os.replace on POSIX filesystems.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure deterministic CPU event reduction with synthetic inputs.",
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
    """Validate CLI arguments, run once, and publish machine-readable JSON."""
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
