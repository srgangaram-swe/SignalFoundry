"""Concrete Alpaca paper/data REST adapter with no selectable trading host.

Run only in the bounded paper worker. The parent imposes a 30-second process
deadline covering DNS, TLS and slow headers; socket and body budgets also apply.
Requests are never retried here, especially when submission outcome is unknown.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol
from urllib.parse import urlencode

from signal_foundry.boundary import FoundryError, decode, encode
from signal_foundry.trading.models import (
    Account,
    Bar,
    Clock,
    Intent,
    Order,
    Position,
    Quote,
)
from signal_foundry.trading.store import Journal

SERVICE = "com.signal-foundry.alpaca-paper"
HOSTS = {"paper": "paper-api.alpaca.markets", "data": "data.alpaca.markets"}
FORBIDDEN = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "BROKER_API_KEY",
    "BROKER_SECRET_KEY",
)


@dataclass(frozen=True, repr=False)
class Credentials:
    key: str = field(repr=False)
    secret: str = field(repr=False)


def credentials() -> Credentials:
    """Read two allowlisted Keychain account labels; never inherit secret env vars."""
    if sys.platform != "darwin" or any(key in os.environ for key in FORBIDDEN):
        raise FoundryError(
            "credential_policy",
            "Use macOS Keychain, not credential environment variables.",
        )
    values: list[str] = []
    for account in ("api-key-id", "api-secret-key"):
        try:
            result = subprocess.run(
                [
                    "/usr/bin/security",
                    "find-generic-password",
                    "-s",
                    SERVICE,
                    "-a",
                    account,
                    "-w",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FoundryError(
                "credentials_unavailable", "Paper Keychain lookup failed.", 503
            ) from exc
        if result.returncode or not 1 <= len(result.stdout) <= 512:
            raise FoundryError(
                "credentials_unavailable",
                "Provision both approved paper Keychain items.",
                503,
            )
        try:
            value = result.stdout.decode("ascii").strip()
        except UnicodeError as exc:
            raise FoundryError(
                "credential_policy", "Invalid paper credential encoding."
            ) from exc
        if not value or not value.isalnum():
            raise FoundryError("credential_policy", "Invalid paper credential format.")
        values.append(value)
    return Credentials(*values)


class Transport(Protocol):
    def request(
        self,
        host: Literal["paper", "data"],
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, object] | None = None,
    ) -> Any: ...


class HTTPS:
    """Fixed TLS hosts, no proxy/redirect/cookie handling, 32 requests per worker."""

    def __init__(self, journal: Journal, keys: Credentials) -> None:
        self.journal = journal
        self.keys = keys
        self.deadline = time.monotonic() + 25
        self.remaining = 32

    def request(
        self,
        host: Literal["paper", "data"],
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, object] | None = None,
    ) -> Any:
        if host not in HOSTS or method not in {"GET", "POST", "PATCH", "DELETE"}:
            raise FoundryError("transport_policy", "Unsupported provider operation.")
        if not path.startswith("/v2/") or any(c in path for c in "?#\\\r\n"):
            raise FoundryError("transport_policy", "Invalid provider path.")
        if host == "data" and method != "GET":
            raise FoundryError("transport_policy", "Market data is read-only.")
        remaining = self.deadline - time.monotonic()
        if self.remaining <= 0 or remaining <= 0:
            raise FoundryError(
                "provider_budget", "Provider operation budget exhausted.", 429
            )
        self.remaining -= 1
        self.journal.request(time.time())
        connection = http.client.HTTPSConnection(
            HOSTS[host], timeout=min(5, remaining), context=ssl.create_default_context()
        )
        headers = {
            "APCA-API-KEY-ID": self.keys.key,
            "APCA-API-SECRET-KEY": self.keys.secret,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        target = path + ("?" + urlencode(query) if query else "")
        try:
            if (
                host == "paper"
                and method in {"POST", "PATCH"}
                and self.journal.get("stopped")
            ):
                raise FoundryError(
                    "paper_disabled", "Stop reached the transport before dispatch.", 409
                )
            connection.request(
                method,
                target,
                body=encode(body) if body is not None else None,
                headers=headers,
            )
            response = connection.getresponse()
            if (
                response.status == 404
                and method == "GET"
                and path == "/v2/orders:by_client_order_id"
            ):
                return None
            if not 200 <= response.status < 300:
                code = (
                    "provider_auth"
                    if response.status in {401, 403}
                    else "provider_response"
                )
                raise FoundryError(
                    code, "Provider refused the operation; no automatic retry.", 502
                )
            if response.status == 204:
                return None
            if response.getheader("Content-Encoding", "identity") != "identity":
                raise FoundryError(
                    "provider_encoding", "Compressed provider response refused.", 502
                )
            payload = bytearray()
            while True:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise FoundryError(
                        "provider_timeout", "Provider body deadline exceeded.", 504
                    )
                if connection.sock is not None:
                    connection.sock.settimeout(min(5, remaining))
                chunk = response.read(min(65_536, (4 << 20) + 1 - len(payload)))
                payload.extend(chunk)
                if len(payload) > 4 << 20:
                    raise FoundryError(
                        "provider_size", "Provider body exceeds its bound.", 502
                    )
                if not chunk:
                    break
            return decode(bytes(payload), 4 << 20)
        except (OSError, http.client.HTTPException) as exc:
            raise FoundryError(
                "provider_uncertain",
                "Provider communication failed; reconcile before submission.",
                502,
            ) from exc
        finally:
            connection.close()


def order(value: Any) -> Order:
    """Normalize simple DAY limits, retaining fractional partial-fill residuals."""
    if (
        value["type"] != "limit"
        or value["time_in_force"] != "day"
        or value.get("order_class", "simple") not in {"simple", ""}
    ):
        raise FoundryError(
            "unsupported_order",
            "An unsupported broker order requires operator reconciliation.",
        )
    return Order.model_validate(
        {
            "broker_id": value["id"],
            "intent": {
                "client_order_id": value["client_order_id"],
                "symbol": value["symbol"],
                "side": value["side"],
                "quantity": value["qty"],
                "limit_price": value["limit_price"],
            },
            "state": value["status"],
            "filled_quantity": value["filled_qty"],
            "average_price": value["filled_avg_price"],
        }
    )


class Alpaca:
    """Broker-neutral normalized records over the selected concrete REST API."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    def clock(self) -> Clock:
        value = self.transport.request("paper", "GET", "/v2/clock")
        return Clock.model_validate(
            {
                "at": value["timestamp"],
                "is_open": value["is_open"],
                "next_open": value["next_open"],
                "next_close": value["next_close"],
            }
        )

    def account(self) -> Account:
        observed_at = datetime.now(UTC)
        value = self.transport.request("paper", "GET", "/v2/account")
        rows = self.transport.request("paper", "GET", "/v2/positions")
        if not isinstance(rows, list) or len(rows) > 100 or value["currency"] != "USD":
            raise FoundryError(
                "account_contract", "Unsupported account/position boundary."
            )
        if not isinstance(value["id"], str) or not 1 <= len(value["id"]) <= 128:
            raise FoundryError("account_contract", "Invalid account identity.")
        if any(
            type(value[key]) is not bool
            for key in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
        ):
            raise FoundryError("account_contract", "Invalid account restriction flags.")
        return Account(
            digest=hashlib.sha256(value["id"].encode()).hexdigest(),
            cash=value["cash"],
            equity=value["equity"],
            buying_power=value["buying_power"],
            blocked=value["status"] != "ACTIVE"
            or any(
                value[key]
                for key in (
                    "trading_blocked",
                    "account_blocked",
                    "trade_suspended_by_user",
                )
            ),
            positions=tuple(
                Position(
                    symbol=p["symbol"],
                    quantity=p["qty"],
                    market_value=p["market_value"],
                )
                for p in rows
            ),
            observed_at=observed_at,
        )

    def quote(self, symbol: str, feed: str) -> Quote:
        value = self.transport.request(
            "data", "GET", f"/v2/stocks/{symbol}/quotes/latest", query={"feed": feed}
        )["quote"]
        return Quote.model_validate(
            {
                "at": value["t"],
                "bid": str(value["bp"]),
                "ask": str(value["ap"]),
                "bid_size": str(value["bs"]),
                "ask_size": str(value["as"]),
            }
        )

    def open_orders(self) -> tuple[Order, ...]:
        rows = self.transport.request(
            "paper",
            "GET",
            "/v2/orders",
            query={"status": "open", "limit": "101", "nested": "false"},
        )
        if not isinstance(rows, list) or len(rows) > 100:
            raise FoundryError("order_capacity", "Too many broker orders to reconcile.")
        return tuple(order(value) for value in rows)

    def lookup(self, client_id: str) -> Order | None:
        value = self.transport.request(
            "paper",
            "GET",
            "/v2/orders:by_client_order_id",
            query={"client_order_id": client_id},
        )
        return None if value is None else order(value)

    def submit(self, intent: Intent) -> Order:
        body = {
            "client_order_id": intent.client_order_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "qty": str(intent.quantity),
            "type": "limit",
            "time_in_force": "day",
            "limit_price": str(intent.limit_price),
            "extended_hours": False,
        }
        return order(self.transport.request("paper", "POST", "/v2/orders", body=body))

    def cancel(self, value: Order) -> None:
        self.transport.request("paper", "DELETE", f"/v2/orders/{value.broker_id}")

    def replace(self, value: Order, replacement: Intent) -> Order:
        """Replacement needs its own durable intent and fresh admission upstream."""
        if (
            replacement.symbol != value.intent.symbol
            or replacement.side != value.intent.side
        ):
            raise FoundryError(
                "replacement_contract", "Replacement cannot change symbol or side."
            )
        return order(
            self.transport.request(
                "paper",
                "PATCH",
                f"/v2/orders/{value.broker_id}",
                body={
                    "qty": str(replacement.quantity),
                    "limit_price": str(replacement.limit_price),
                    "client_order_id": replacement.client_order_id,
                },
            )
        )

    def bars(
        self, symbol: str, feed: str, start: datetime, end: datetime
    ) -> tuple[tuple[Bar, ...], tuple[object, ...]]:
        """At most five pages/20,000 raw one-minute bars; never truncate silently."""
        # Provider end is inclusive; our causal contract is [start, end).
        # Disable implicit current-day symbol remapping; lineage flags remain false.
        query = {
            "timeframe": "1Min",
            "start": start.isoformat(),
            "end": (end - timedelta(microseconds=1)).isoformat(),
            "feed": feed,
            "adjustment": "raw",
            "asof": "-",
            "sort": "asc",
            "limit": "5000",
        }
        bars: list[Bar] = []
        pages: list[object] = []
        tokens: set[str] = set()
        for _ in range(5):
            page = self.transport.request(
                "data", "GET", f"/v2/stocks/{symbol}/bars", query=query
            )
            rows = page["bars"]
            if not isinstance(rows, list) or len(rows) + len(bars) > 20000:
                raise FoundryError(
                    "data_capacity", "Intraday acquisition exceeds its row budget."
                )
            pages.append(page)
            for row in rows:
                bars.append(
                    Bar.model_validate(
                        {
                            "at": row["t"],
                            "open": str(row["o"]),
                            "high": str(row["h"]),
                            "low": str(row["l"]),
                            "close": str(row["c"]),
                            "volume": str(row["v"]),
                        }
                    )
                )
            token = page.get("next_page_token")
            if token is None:
                if (
                    not bars
                    or any(not start <= b.at < end for b in bars)
                    or any(a.at >= b.at for a, b in zip(bars, bars[1:], strict=False))
                ):
                    raise FoundryError(
                        "data_order",
                        "Empty, duplicate, out-of-range or unordered bars.",
                    )
                return tuple(bars), tuple(pages)
            if (
                not isinstance(token, str)
                or not 1 <= len(token) <= 2048
                or token in tokens
            ):
                raise FoundryError(
                    "data_pagination", "Invalid or repeated provider page token."
                )
            tokens.add(token)
            query = {**query, "page_token": token}
        raise FoundryError(
            "data_capacity", "Provider pagination exceeds its request budget."
        )
