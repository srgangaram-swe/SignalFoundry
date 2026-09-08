# Execution Engine (C++ Core)

## Why it exists

The daily-bar research pipeline does not need nanosecond matching, and this
document says so up front. The native core exists for two honest reasons:

1. **Controlled microstructure experiments.** The depth-aware `simulate_fill`
   function walks a synthetic book, so tests can study how order size consumes
   visible liquidity. It is not used as evidence of historical fill quality.
2. **Systems engineering demonstration.** Matching engines are the canonical
   trading-infrastructure exercise: data-structure choice under latency
   constraints, integer determinism, FFI, and cross-implementation testing.

## Design

`cpp/include/alphaforge/order_book.hpp` — header-only, C++17, no dependencies.

- **Price-time priority.** Bids and asks are `std::map<price, Level>`
  (descending / ascending). Each `Level` holds a FIFO `std::list<Order>` and
  an aggregate quantity.
- **O(1) cancel.** An `unordered_map<order_id, (side, price, list iterator)>`
  index; `std::list` iterators stay valid under erasure, so cancels never scan.
- **Integer ticks and quantities.** No floating point in the matching path:
  behaviour is exactly reproducible across platforms and bit-identical to the
  Python reference.
- **Matching semantics.** Limit orders fill at the resting (maker) price while
  crossed; remainders rest. Market orders walk the opposite side; unfilled
  remainders are dropped. Fills are recorded as
  `(maker_id, taker_id, price, qty)` in match order. No self-match prevention
  (single-strategy simulation does not need it).

`alphaforge/execution/orderbook_py.py` is a pure-Python implementation of the
same contract; `alphaforge/execution/native.py` dispatches to whichever is
available. `tests/test_orderbook.py` drives both with identical random flow
(4,000 mixed operations) and requires identical order ids, fills, depth,
volumes, and best quotes.

The extension exposes `__abi_version__ = "1"` and the loader requires the same
`NATIVE_ABI_VERSION`. This identifier versions the Python/native execution
boundary independently of the AlphaForge package release. An absent or
incompatible extension fails closed to the tested Python implementation.

## Boundary with daily-bar execution

Historical backtests and paper replay use `alphaforge.execution.models` plus
the canonical event reducer, tamper-evident journal, and self-financing
cost-basis ledger—not a fabricated order-book snapshot. Their fills occur at a
future open and use lagged ADV/volatility sensitivities, explicit participation
caps, frozen-calendar eligibility, DAY lifecycle events, and reconciled
implementation shortfall. This is the more defensible model for OHLCV inputs
because the data does not contain queue or L2 state. See
[`event_driven_backtesting.md`](event_driven_backtesting.md).

The native book becomes a candidate execution backend only if a future data
source supplies point-in-time L2 events and the fill model is calibrated and
validated. Until then, native benchmark figures demonstrate implementation
performance—not achievable strategy latency or trading capacity.

## Benchmarks

Two benchmarks, because they answer different questions:

- `make bench-native` — pure C++ (`cpp/benchmarks/bench_main.cpp`): true
  in-process cost per operation with sampled latency percentiles.
- `make bench` — drives both engines from Python: what a Python caller
  actually observes, pybind11 overhead included.

Representative numbers (Apple M-series, 2M ops, 55% add / 30% cancel /
15% market):

| engine | throughput | p50 | p99 |
|---|---|---|---|
| C++ (in-process) | ~6.2M ops/s | ~125 ns | ~583 ns |
| C++ via pybind11 | ~1.8M ops/s | — | — |
| pure Python | ~1.1M ops/s | — | — |

The pybind11 row is the honest caveat: per-call FFI overhead dominates when
you drive a nanosecond-scale engine one op at a time from Python. Real systems
batch at the boundary or keep the hot loop native.

## Build

```bash
make native        # one-command build via scripts/build_native.py (needs pybind11)
# or the full CMake project:
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

Everything degrades gracefully: without the extension, the Python reference
implementation serves the same API and all non-parity tests still pass.
