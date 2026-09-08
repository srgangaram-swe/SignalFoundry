"""Loader for the optional C++ execution core with pure-Python fallback.

The native module is built by ``make native`` (scripts/build_native.py) or the
CMake project in cpp/. When absent, everything still works via the reference
implementation in orderbook_py — the two are parity-tested against each other.
"""

from __future__ import annotations

from alphaforge.execution import orderbook_py

NATIVE_ABI_VERSION = "1"


def _has_compatible_native_abi(module: object) -> bool:
    """Return whether an imported extension implements the governed ABI."""

    return getattr(module, "__abi_version__", None) == NATIVE_ABI_VERSION


try:
    import alphaforge.alphaforge_native as _native
except ImportError:
    NATIVE_AVAILABLE = False
else:
    NATIVE_AVAILABLE = _has_compatible_native_abi(_native)

BUY = orderbook_py.BUY
SELL = orderbook_py.SELL


def native_side(side: int):
    """Map integer side to the native enum (native module required)."""
    if not NATIVE_AVAILABLE:
        raise RuntimeError("native module not built — run `make native`")
    return _native.Side.BUY if side == BUY else _native.Side.SELL


def make_order_book(prefer_native: bool = True):
    """Return an order book instance: native if built, otherwise Python."""
    if prefer_native and NATIVE_AVAILABLE:
        return _NativeBookAdapter()
    return orderbook_py.PyOrderBook()


def simulate_fill(
    side: int,
    qty: int,
    mid: int,
    half_spread: int,
    tick: int = 1,
    n_levels: int = 10,
    qty_per_level: int = 1000,
    prefer_native: bool = True,
) -> tuple[float, int]:
    """Depth-aware fill simulation; dispatches to C++ when available."""
    if prefer_native and NATIVE_AVAILABLE:
        return _native.simulate_fill(
            native_side(side), qty, mid, half_spread, tick, n_levels, qty_per_level
        )
    return orderbook_py.simulate_fill(side, qty, mid, half_spread, tick, n_levels, qty_per_level)


class _NativeBookAdapter:
    """Thin adapter giving the native book the same int-side API as PyOrderBook."""

    def __init__(self) -> None:
        self._book = _native.OrderBook()

    def add_limit(self, side: int, price: int, qty: int) -> int:
        return self._book.add_limit(native_side(side), price, qty)

    def add_market(self, side: int, qty: int) -> int:
        return self._book.add_market(native_side(side), qty)

    def cancel(self, order_id: int) -> bool:
        return self._book.cancel(order_id)

    def best_bid(self):
        return self._book.best_bid()

    def best_ask(self):
        return self._book.best_ask()

    def bid_depth(self, levels: int = 10):
        return [tuple(x) for x in self._book.bid_depth(levels)]

    def ask_depth(self, levels: int = 10):
        return [tuple(x) for x in self._book.ask_depth(levels)]

    def bid_volume(self) -> int:
        return self._book.bid_volume()

    def ask_volume(self) -> int:
        return self._book.ask_volume()

    def open_orders(self) -> int:
        return self._book.open_orders()

    def take_fills(self):
        return self._book.take_fills()

    def clear(self) -> None:
        self._book.clear()
