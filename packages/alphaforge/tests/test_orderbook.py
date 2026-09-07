"""Order-book semantics + C++/Python parity tests.

The pure-Python book defines the contract; the C++ engine must be
indistinguishable under identical random flow. Native tests are skipped
automatically when the extension has not been built (`make native`).
"""

from __future__ import annotations

import numpy as np
import pytest

import alphaforge.execution.native as native_loader
from alphaforge.execution import (
    BUY,
    NATIVE_ABI_VERSION,
    NATIVE_AVAILABLE,
    SELL,
    make_order_book,
    simulate_fill,
)
from alphaforge.execution.orderbook_py import PyOrderBook

needs_native = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native module not built")


def fills_as_tuples(fills):
    return [f.as_tuple() if hasattr(f, "as_tuple") else tuple(f) for f in fills]


@needs_native
def test_native_abi_version_matches_loader_contract() -> None:
    import alphaforge.alphaforge_native as native_module

    assert native_module.__abi_version__ == NATIVE_ABI_VERSION


@pytest.mark.parametrize("observed_abi", [None, "0", "2", 1])
def test_native_loader_rejects_missing_or_incompatible_abi(observed_abi: object) -> None:
    class FakeNative:
        pass

    fake_native = FakeNative()
    if observed_abi is not None:
        fake_native.__abi_version__ = observed_abi  # type: ignore[attr-defined]

    assert not native_loader._has_compatible_native_abi(fake_native)


def test_price_time_priority():
    book = PyOrderBook()
    first = book.add_limit(BUY, 100, 10)
    second = book.add_limit(BUY, 100, 10)
    book.add_limit(SELL, 100, 15)
    fills = fills_as_tuples(book.take_fills())
    assert fills[0][0] == first and fills[0][3] == 10
    assert fills[1][0] == second and fills[1][3] == 5


def test_partial_fill_rests_remainder():
    book = PyOrderBook()
    book.add_limit(SELL, 101, 5)
    bid_id = book.add_limit(BUY, 101, 8)
    assert book.best_bid() == 101
    assert book.bid_depth(1) == [(101, 3)]
    assert book.best_ask() is None
    assert book.cancel(bid_id)
    assert book.best_bid() is None


def test_market_order_walks_the_book():
    book = PyOrderBook()
    book.add_limit(SELL, 100, 5)
    book.add_limit(SELL, 101, 5)
    book.add_limit(SELL, 102, 5)
    filled = book.add_market(BUY, 12)
    assert filled == 12
    fills = [(f.price, f.qty) for f in book.take_fills()]
    assert fills == [(100, 5), (101, 5), (102, 2)]
    assert book.ask_depth(1) == [(102, 3)]


def test_clear_resets_book_state_and_order_ids():
    book = PyOrderBook()
    book.add_limit(BUY, 100, 10)
    book.add_limit(SELL, 100, 4)

    book.clear()

    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.open_orders() == 0
    assert book.take_fills() == []
    assert book.add_limit(BUY, 99, 1) == 1


def test_slippage_increases_with_order_size():
    small_px, small_fill = simulate_fill(BUY, 500, mid=10_000, half_spread=5, prefer_native=False)
    large_px, large_fill = simulate_fill(BUY, 5_000, mid=10_000, half_spread=5, prefer_native=False)
    assert small_fill == 500 and large_fill == 5_000
    assert large_px > small_px > 10_000


def _random_flow(book, n_ops: int, seed: int):
    rng = np.random.default_rng(seed)
    live: list[int] = []
    log: list[tuple[str, int] | tuple[str, int, bool]] = []
    for _ in range(n_ops):
        op = int(rng.integers(0, 100))
        if op < 55:
            side = BUY if op % 2 else SELL
            oid = book.add_limit(
                side, int(10_000 + rng.integers(-40, 41)), int(rng.integers(1, 200))
            )
            live.append(oid)
            log.append(("limit", oid))
        elif op < 85 and live:
            j = int(rng.integers(0, len(live)))
            log.append(("cancel", live[j], book.cancel(live[j])))
            live[j] = live[-1]
            live.pop()
        else:
            filled = book.add_market(BUY if op % 2 else SELL, int(rng.integers(1, 400)))
            log.append(("market", filled))
    return log, fills_as_tuples(book.take_fills())


@needs_native
def test_native_matches_python_reference_exactly():
    py_book = make_order_book(prefer_native=False)
    cpp_book = make_order_book(prefer_native=True)
    py_log, py_fills = _random_flow(py_book, 4_000, seed=99)
    cpp_log, cpp_fills = _random_flow(cpp_book, 4_000, seed=99)

    assert py_log == cpp_log, "order ids / cancel results / market fills diverged"
    assert py_fills == cpp_fills, "fill streams diverged"
    assert py_book.best_bid() == cpp_book.best_bid()
    assert py_book.best_ask() == cpp_book.best_ask()
    assert py_book.bid_depth(10) == cpp_book.bid_depth(10)
    assert py_book.ask_depth(10) == cpp_book.ask_depth(10)
    assert py_book.open_orders() == cpp_book.open_orders()
    assert py_book.bid_volume() == cpp_book.bid_volume()
    assert py_book.ask_volume() == cpp_book.ask_volume()


@needs_native
def test_native_simulate_fill_matches_python():
    for side in (BUY, SELL):
        for qty in (100, 2_500, 9_999):
            py = simulate_fill(side, qty, mid=10_000, half_spread=5, prefer_native=False)
            cpp = simulate_fill(side, qty, mid=10_000, half_spread=5, prefer_native=True)
            assert py[1] == cpp[1]
            assert py[0] == pytest.approx(cpp[0], rel=1e-12)
