"""Deterministic vendor fixture; never usable as a production transport."""

from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from signal_foundry.boundary import FoundryError
from signal_foundry.trading.models import PaperConfig

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 8, 15, 0, tzinfo=UTC)


class FixedTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)


def config(**changes):
    value = {
        "enabled": True,
        "account_digest": hashlib.sha256(b"fixture-account").hexdigest(),
        "maximum_order_notional": "500",
        "maximum_position_notional": "1500",
        "plan": {
            "symbols": ["AAA", "BBB", "CCC"],
            "feed": "iex",
            "start": "2026-08-01T00:00:00Z",
            "selection_end": "2026-08-15T00:00:00Z",
            "end": "2026-09-01T00:00:00Z",
            "window": 3,
        },
        **changes,
    }
    return PaperConfig.model_validate(value)


class Vendor:
    def __init__(self):
        self.calls = []
        self.orders = {}
        self.cash = Decimal("10000")
        self.positions = {}
        self.submit_failure = False
        self.status = "filled"
        self.clock_open = True
        self.quote_at = NOW.isoformat()
        self.quote_bid = 101.0
        self.quote_ask = 101.01
        self.page = None
        self.calendar = [{"date": "2026-09-08", "open": "09:30", "close": "16:00"}]

    def request(self, host, method, path, *, query=None, body=None):
        self.calls.append((host, method, path, query, copy.deepcopy(body)))
        if path == "/v2/account":
            market = sum(self.positions.values(), Decimal(0)) * Decimal(
                str(self.quote_ask)
            )
            return {
                "id": "fixture-account",
                "status": "ACTIVE",
                "currency": "USD",
                "cash": str(self.cash),
                "equity": str(self.cash + market),
                "buying_power": str(self.cash),
                "trading_blocked": False,
                "account_blocked": False,
                "trade_suspended_by_user": False,
            }
        if path == "/v2/positions":
            return [
                {
                    "symbol": key,
                    "qty": str(value),
                    "market_value": str(value * Decimal(str(self.quote_ask))),
                }
                for key, value in self.positions.items()
                if value
            ]
        if path == "/v2/clock":
            return {
                "timestamp": NOW.isoformat(),
                "is_open": self.clock_open,
                "next_open": "2026-09-09T13:30:00Z",
                "next_close": "2026-09-08T20:00:00Z",
            }
        if path == "/v2/calendar":
            return self.calendar
        if path.endswith("quotes/latest"):
            return {
                "quote": {
                    "t": self.quote_at,
                    "bp": self.quote_bid,
                    "ap": self.quote_ask,
                    "bs": 10,
                    "as": 10,
                }
            }
        if path.endswith("/bars"):
            if self.page is not None:
                return copy.deepcopy(self.page)
            start = datetime.fromisoformat(query["start"])
            end = datetime.fromisoformat(query["end"])
            return {
                "bars": [
                    {
                        "t": (start + timedelta(minutes=i)).isoformat(),
                        "o": 99 + i,
                        "h": 100 + i,
                        "l": 98 + i,
                        "c": 100 + i,
                        "v": 1000,
                    }
                    for i in range(min(3, int((end - start).total_seconds() // 60) + 1))
                ],
                "next_page_token": None,
            }
        if path == "/v2/orders:by_client_order_id":
            return copy.deepcopy(self.orders.get(query["client_order_id"]))
        if path == "/v2/orders" and method == "GET":
            return [
                copy.deepcopy(o)
                for o in self.orders.values()
                if o["status"] not in {"filled", "canceled", "replaced"}
            ]
        if path == "/v2/orders" and method == "POST":
            value = {
                **body,
                "id": "11111111-1111-1111-1111-111111111111",
                "status": self.status,
                "filled_qty": body["qty"] if self.status == "filled" else "0",
                "filled_avg_price": (
                    body["limit_price"] if self.status == "filled" else None
                ),
            }
            self.orders[body["client_order_id"]] = value
            if self.status == "filled":
                quantity = Decimal(body["qty"]) * (1 if body["side"] == "buy" else -1)
                self.cash -= quantity * Decimal(body["limit_price"])
                self.positions[body["symbol"]] = (
                    self.positions.get(body["symbol"], Decimal(0)) + quantity
                )
            if self.submit_failure:
                raise FoundryError(
                    "provider_uncertain", "Injected post-acceptance disconnect."
                )
            return copy.deepcopy(value)
        if path.startswith("/v2/orders/") and method == "DELETE":
            for value in self.orders.values():
                if value["id"] == path.rsplit("/", 1)[1]:
                    value["status"] = "canceled"
            return None
        if path.startswith("/v2/orders/") and method == "PATCH":
            old = next(iter(self.orders.values()))
            return {
                **old,
                **body,
                "filled_qty": "0",
                "filled_avg_price": None,
                "status": "new",
            }
        raise AssertionError((host, method, path))
