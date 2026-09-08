"""Order-book throughput benchmark: C++ engine vs pure-Python reference.

Drives both implementations with the same seeded random flow
(55% add_limit / 30% cancel / 15% market). Numbers for the native engine
measured from Python include pybind11 call overhead; run the CMake
`bench_orderbook` binary for true in-process latency percentiles.

Usage: python scripts/bench_orderbook.py [--ops 200000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from alphaforge.execution import BUY, NATIVE_AVAILABLE, SELL, make_order_book


def run_flow(book, n_ops: int, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    ops = rng.integers(0, 100, size=n_ops)
    offsets = rng.integers(-50, 51, size=n_ops)
    qtys = rng.integers(1, 501, size=n_ops)
    mid = 100_000
    live: list[int] = []

    for i in range(2_000):  # warm-up depth
        live.append(book.add_limit(BUY, mid - 1 - (i % 50), 100))
        live.append(book.add_limit(SELL, mid + 1 + (i % 50), 100))
    book.take_fills()

    start = time.perf_counter()
    for i in range(n_ops):
        op = int(ops[i])
        if op < 55:
            side = BUY if op % 2 else SELL
            live.append(book.add_limit(side, mid + int(offsets[i]), int(qtys[i])))
        elif op < 85 and live:
            j = op % len(live)
            book.cancel(live[j])
            live[j] = live[-1]
            live.pop()
        else:
            book.add_market(BUY if op % 2 else SELL, int(qtys[i]))
        if i % 65_536 == 0:
            book.take_fills()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", type=int, default=200_000)
    parser.add_argument("--out", default="runs/bench_orderbook.json")
    args = parser.parse_args()

    results = {}
    py_elapsed = run_flow(make_order_book(prefer_native=False), args.ops)
    results["python"] = {
        "ops": args.ops,
        "elapsed_s": py_elapsed,
        "ops_per_sec": args.ops / py_elapsed,
    }
    print(f"pure python : {args.ops / py_elapsed:>12,.0f} ops/s ({py_elapsed:.2f}s)")

    if NATIVE_AVAILABLE:
        cpp_elapsed = run_flow(make_order_book(prefer_native=True), args.ops)
        results["native_via_python"] = {
            "ops": args.ops,
            "elapsed_s": cpp_elapsed,
            "ops_per_sec": args.ops / cpp_elapsed,
        }
        print(f"C++ (pybind): {args.ops / cpp_elapsed:>12,.0f} ops/s ({cpp_elapsed:.2f}s)")
        print(f"speedup     : {py_elapsed / cpp_elapsed:.1f}x (binding overhead included)")
    else:
        print("native module not built — run `make native`")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
