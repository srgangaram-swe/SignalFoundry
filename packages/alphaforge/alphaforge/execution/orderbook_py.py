"""Pure-Python reference limit order book.

Semantically identical to the C++ engine in cpp/include/alphaforge/order_book.hpp:
price-time priority, maker-price fills, integer ticks/quantities. The parity
tests in tests/test_orderbook.py drive both implementations with the same
random order flow and require identical fills and book state.

This implementation is the always-available fallback; the native module is
an optional accelerator (see alphaforge/execution/native.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

BUY = 0
SELL = 1


@dataclass
class Fill:
    maker_id: int
    taker_id: int
    price: int
    qty: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.maker_id, self.taker_id, self.price, self.qty)


class PyOrderBook:
    """Price-time-priority book on integer ticks."""

    def __init__(self) -> None:
        # price -> deque of [order_id, qty]; separate aggregate per level
        self._bids: dict[int, deque] = {}
        self._asks: dict[int, deque] = {}
        self._level_total: dict[tuple[int, int], int] = {}
        self._index: dict[int, tuple[int, int]] = {}  # order_id -> (side, price)
        self._fills: list[Fill] = []
        self._next_id = 1
        # sorted price caches (lazily maintained)
        self._bid_prices: list[int] = []
        self._ask_prices: list[int] = []

    # -- public API (mirrors the C++ OrderBook) --

    def add_limit(self, side: int, price: int, qty: int) -> int:
        order_id = self._next_id
        self._next_id += 1
        if qty <= 0:
            return order_id
        if side == BUY:
            remaining = self._match(SELL, order_id, qty, lambda best: best <= price)
        else:
            remaining = self._match(BUY, order_id, qty, lambda best: best >= price)
        if remaining > 0:
            self._rest(side, order_id, price, remaining)
        return order_id

    def add_market(self, side: int, qty: int) -> int:
        order_id = self._next_id
        self._next_id += 1
        if qty <= 0:
            return 0
        opposite = SELL if side == BUY else BUY
        remaining = self._match(opposite, order_id, qty, lambda best: True)
        return qty - remaining

    def cancel(self, order_id: int) -> bool:
        loc = self._index.pop(order_id, None)
        if loc is None:
            return False
        side, price = loc
        book = self._bids if side == BUY else self._asks
        level = book[price]
        for i, entry in enumerate(level):
            if entry[0] == order_id:
                self._level_total[(side, price)] -= entry[1]
                del level[i]
                break
        if not level:
            self._drop_level(side, price)
        return True

    def best_bid(self) -> int | None:
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self) -> int | None:
        return self._ask_prices[0] if self._ask_prices else None

    def bid_depth(self, levels: int = 10) -> list[tuple[int, int]]:
        prices = list(reversed(self._bid_prices[-levels:]))
        return [(p, self._level_total[(BUY, p)]) for p in prices]

    def ask_depth(self, levels: int = 10) -> list[tuple[int, int]]:
        return [(p, self._level_total[(SELL, p)]) for p in self._ask_prices[:levels]]

    def bid_volume(self) -> int:
        return sum(self._level_total[(BUY, p)] for p in self._bid_prices)

    def ask_volume(self) -> int:
        return sum(self._level_total[(SELL, p)] for p in self._ask_prices)

    def open_orders(self) -> int:
        return len(self._index)

    def take_fills(self) -> list[Fill]:
        out, self._fills = self._fills, []
        return out

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._level_total.clear()
        self._index.clear()
        self._fills.clear()
        self._next_id = 1
        self._bid_prices.clear()
        self._ask_prices.clear()

    # -- internals --

    def _match(self, book_side: int, taker_id: int, qty: int, crossed) -> int:
        book = self._bids if book_side == BUY else self._asks
        prices = self._bid_prices if book_side == BUY else self._ask_prices
        remaining = qty
        while remaining > 0 and prices:
            best = prices[-1] if book_side == BUY else prices[0]
            if not crossed(best):
                break
            level = book[best]
            while remaining > 0 and level:
                entry = level[0]  # [order_id, qty]
                traded = min(entry[1], remaining)
                self._fills.append(Fill(entry[0], taker_id, best, traded))
                entry[1] -= traded
                self._level_total[(book_side, best)] -= traded
                remaining -= traded
                if entry[1] == 0:
                    self._index.pop(entry[0], None)
                    level.popleft()
            if not level:
                self._drop_level(book_side, best)
        return remaining

    def _rest(self, side: int, order_id: int, price: int, qty: int) -> None:
        book = self._bids if side == BUY else self._asks
        if price not in book:
            book[price] = deque()
            self._level_total[(side, price)] = 0
            self._insert_price(side, price)
        book[price].append([order_id, qty])
        self._level_total[(side, price)] += qty
        self._index[order_id] = (side, price)

    def _insert_price(self, side: int, price: int) -> None:
        import bisect

        prices = self._bid_prices if side == BUY else self._ask_prices
        bisect.insort(prices, price)

    def _drop_level(self, side: int, price: int) -> None:
        book = self._bids if side == BUY else self._asks
        book.pop(price, None)
        self._level_total.pop((side, price), None)
        prices = self._bid_prices if side == BUY else self._ask_prices
        import bisect

        i = bisect.bisect_left(prices, price)
        if i < len(prices) and prices[i] == price:
            del prices[i]


def simulate_fill(
    side: int,
    qty: int,
    mid: int,
    half_spread: int,
    tick: int = 1,
    n_levels: int = 10,
    qty_per_level: int = 1000,
) -> tuple[float, int]:
    """Depth-aware market-order fill against a synthetic book.

    Mirrors alphaforge::simulate_fill. Returns (avg_fill_price_ticks, filled_qty).
    """
    if qty <= 0 or qty_per_level <= 0 or n_levels <= 0:
        return (0.0, 0)
    book = PyOrderBook()
    for i in range(n_levels):
        book.add_limit(SELL, mid + half_spread + i * tick, qty_per_level)
        book.add_limit(BUY, mid - half_spread - i * tick, qty_per_level)
    filled = book.add_market(side, qty)
    if filled == 0:
        return (0.0, 0)
    notional = sum(f.price * f.qty for f in book.take_fills())
    return (notional / filled, filled)
