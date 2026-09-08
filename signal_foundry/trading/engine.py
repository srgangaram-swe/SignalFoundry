"""Shared paper lifecycle and admission policy for the CLI and Nexus.

An operation holds the process lock, but never a database transaction during
network I/O. Ambiguous dispatch is persisted before transport and can only be
resolved by querying the broker; a 404 after uncertainty never causes resubmission.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from signal_foundry.boundary import (
    FoundryError,
    code_identity,
    encode,
)
from signal_foundry.trading.alpaca import HTTPS, Alpaca, credentials
from signal_foundry.trading.campaign import campaign, record_session
from signal_foundry.trading.data import acquire, research, session
from signal_foundry.trading.models import (
    TERMINAL,
    Account,
    Intent,
    Order,
    PaperConfig,
    PaperStatus,
)
from signal_foundry.trading.qualification import qualification
from signal_foundry.trading.research import target
from signal_foundry.trading.store import Journal


class Engine:
    """One explicit operation on an immutable configuration and private journal."""

    def __init__(
        self,
        root: Path,
        journal: Journal,
        config: PaperConfig,
        broker: Alpaca | None = None,
    ) -> None:
        self.root = root
        self.journal = journal
        self.config = config
        self._broker = broker

    @property
    def broker(self) -> Alpaca:
        if self._broker is None:
            self._broker = Alpaca(HTTPS(self.journal, credentials()))
        return self._broker

    def initialize(self) -> None:
        saved = self.journal.get("config")
        if saved is not None:
            self.bound()
            return
        self.journal.set("config", self.config.wire(), kind="initialize")
        self.journal.set("code", code_identity(self.root), kind="code_identity")
        self.journal.set("created", datetime.now(UTC).isoformat(), kind="created")
        self.journal.set("state", "initialized", kind="state")

    def bound(self) -> None:
        if self.journal.get("config") != self.config.wire() or self.journal.get(
            "code"
        ) != code_identity(self.root):
            raise FoundryError(
                "paper_identity",
                "Code/configuration changed or state is uninitialized. Use a new root.",
                409,
            )

    def status(self) -> PaperStatus:
        snapshot = self.journal.get("account")
        account = Account.model_validate(snapshot) if snapshot else None
        stopped = bool(self.journal.get("stopped"))
        state = self.journal.get("state") or "uninitialized"
        blockers = ["Live capability is absent."]
        if not self.config.enabled:
            blockers.append(
                "Paper order submission is disabled in the local configuration."
            )
        if stopped:
            blockers.append(
                "Persistent stop is engaged; this run cannot be re-enabled."
            )
        if state != "ready":
            blockers.append("A qualified, reconciled paper session is required.")
        if self.journal.get("last_error"):
            blockers.append("Last operation failed: " + self.journal.get("last_error"))
        if not (self.journal.root / "qualification.json").exists():
            blockers.append("A verified source qualification dossier is missing.")
        return PaperStatus(
            configured=True,
            enabled=self.config.enabled,
            stopped=stopped,
            config_identity=self.config.identity,
            state=state,
            symbols=self.config.plan.symbols,
            feed=self.config.plan.feed,
            maximum_order_notional=str(self.config.maximum_order_notional),
            maximum_position_notional=str(self.config.maximum_position_notional),
            maximum_session_loss=str(self.config.maximum_session_loss),
            orders=len(self.journal.order_keys()),
            events=self.journal.event_count(),
            blockers=tuple(blockers),
            last_action=self.journal.get("last_action") or "none",
            paper_sessions=len(self.journal.get("observed_dates") or []),
            cash=str(account.cash) if account else None,
            equity=str(account.equity) if account else None,
            account_observed_at=account.observed_at.isoformat() if account else None,
            positions=account.positions if account else (),
        )

    def qualification(self) -> dict[str, Any]:
        return qualification(self.root, self.journal, self.config, datetime.now(UTC))

    def account(self) -> Account:
        account = self.broker.account()
        if account.digest != self.config.account_digest or account.blocked:
            raise FoundryError(
                "account_identity", "Wrong or restricted paper account.", 409
            )
        return account

    def start(self) -> None:
        self.admission_enabled()
        self.qualification()
        if self.journal.get("baseline") is not None:
            self.reconcile()
            return
        account = self.account()
        if account.positions or self.broker.open_orders():
            raise FoundryError(
                "paper_baseline",
                "Start requires a dedicated flat paper account without open orders.",
            )
        self.journal.set("baseline", account.wire(), kind="baseline")
        self.journal.set("account", account.wire(), kind="baseline_snapshot")
        self.journal.set("state", "ready", kind="state")

    def admission_enabled(self) -> None:
        self.bound()
        if not self.config.enabled or self.journal.get("stopped"):
            raise FoundryError(
                "paper_disabled",
                "Paper submission is disabled or permanently stopped.",
                409,
            )

    def reconcile(self) -> Account:
        """Rebuild cash/quantities from cumulative fills, never apply a fill twice.

        A non-atomic broker snapshot may conservatively report a break; a later
        explicit reconciliation can resolve it. Fees/external activity not
        represented by fills cause refusal, never an unexplained cash adjustment.
        """
        baseline_value = self.journal.get("baseline")
        if baseline_value is None:
            raise FoundryError(
                "paper_baseline", "Start a qualified paper session first."
            )
        baseline = Account.model_validate(baseline_value)
        self.account()  # Reject an account switch before accepting order observations.
        expected_cash = baseline.cash
        quantities: dict[str, Decimal] = {}
        known: set[str] = set()
        self.journal.set("state", "reconciliation_required", kind="state")
        for key in self.journal.order_keys():
            row = self.journal.get(key)
            intent = Intent.model_validate(row["intent"])
            known.add(intent.client_order_id)
            old = (
                Order.model_validate(row["order"]) if row["order"] is not None else None
            )
            latest = (
                old
                if old is not None and old.state in TERMINAL
                else self.broker.lookup(intent.client_order_id)
            )
            if latest is None:
                raise FoundryError(
                    "submission_unknown",
                    "Persisted intent absent at broker; do not resubmit or erase it.",
                    409,
                )
            if latest.intent != intent or (
                old and latest.filled_quantity < old.filled_quantity
            ):
                raise FoundryError(
                    "order_divergence", "Broker intent/fill history diverged.", 409
                )
            if old is None or latest != old:
                self.journal.set(
                    key,
                    {"intent": intent.wire(), "order": latest.wire()},
                    kind="order_observation",
                )
            sign = Decimal(1) if intent.side == "buy" else Decimal(-1)
            quantities[intent.symbol] = (
                quantities.get(intent.symbol, Decimal(0))
                + sign * latest.filled_quantity
            )
            expected_cash -= (
                sign * latest.filled_quantity * (latest.average_price or Decimal(0))
            )
        if any(
            o.intent.client_order_id not in known for o in self.broker.open_orders()
        ):
            raise FoundryError(
                "external_order", "An unowned paper order blocks admission.", 409
            )
        account = self.account()
        expected = {
            symbol: quantity for symbol, quantity in quantities.items() if quantity
        }
        actual = {p.symbol: p.quantity for p in account.positions if p.quantity}
        if expected != actual or abs(account.cash - expected_cash) > Decimal("0.01"):
            raise FoundryError(
                "account_divergence",
                "Cash/position break; inspect fees, external activity and fills.",
                409,
            )
        if baseline.equity - account.equity >= self.config.maximum_session_loss:
            self.journal.stop()
            raise FoundryError(
                "loss_limit",
                "Paper loss ceiling reached; persistent stop engaged.",
                409,
            )
        self.journal.set("account", account.wire(), kind="reconciliation")
        self.journal.set("state", "ready", kind="state")
        return account

    def cycle(self, symbol: str) -> None:
        """One bounded symbol decision; repeat explicitly, never a hidden daemon."""
        self.admission_enabled()
        self.qualification()
        if symbol not in self.config.plan.symbols:
            raise FoundryError("paper_symbol", "Symbol is outside the frozen universe.")
        account = self.reconcile()
        clock = self.broker.clock()
        now = datetime.now(UTC)
        if abs((now - clock.at).total_seconds()) > 5 or not clock.is_open:
            raise FoundryError(
                "market_session", "Market is closed or the broker clock is stale.", 409
            )
        quote = self.broker.quote(symbol, self.config.plan.feed)
        if not 0 <= (now - quote.at).total_seconds() <= self.config.quote_age_seconds:
            raise FoundryError("stale_quote", "Quote is future-dated or stale.", 409)
        if (quote.ask - quote.bid) / quote.bid * 10000 > self.config.maximum_spread_bps:
            raise FoundryError(
                "spread_limit", "Quoted spread exceeds the configured ceiling.", 409
            )
        end = now.replace(second=0, microsecond=0)
        bars, _ = self.broker.bars(
            symbol,
            self.config.plan.feed,
            end - timedelta(minutes=self.config.plan.window),
            end,
        )
        local_day = now.astimezone(ZoneInfo("America/New_York")).date()
        opening, closing = self.session(local_day.isoformat())
        if any(b.at < opening for b in bars) or now >= closing:
            raise FoundryError(
                "feature_session",
                "Feature window crosses the regular-session boundary.",
                409,
            )
        if (
            len(bars) != self.config.plan.window
            or bars[-1].at + timedelta(minutes=1) != end
            or any(
                b.at.astimezone(ZoneInfo("America/New_York")).date() != local_day
                for b in bars
            )
        ):
            raise FoundryError(
                "feature_freshness",
                "A complete current-session causal feature window is unavailable.",
                409,
            )
        if any(
            b.at - a.at != timedelta(minutes=1)
            for a, b in zip(bars, bars[1:], strict=False)
        ):
            raise FoundryError(
                "feature_gap", "A missing minute blocks the decision.", 409
            )
        decision_key = f"decision/{symbol}"
        if self.journal.get(decision_key) == end.isoformat():
            return
        for key in self.journal.order_keys():
            row = self.journal.get(key)
            if row["order"] is None or row["order"]["state"] not in TERMINAL:
                raise FoundryError(
                    "working_order",
                    "Reconcile or cancel working orders before another decision.",
                    409,
                )
        proposed = target(
            tuple(b.close for b in bars),
            candidate=self.config.candidate,
            threshold_bps=self.config.plan.threshold_bps,
        )
        if clock.next_close - now <= timedelta(minutes=10):
            proposed = 0
        current = next(
            (p.quantity for p in account.positions if p.symbol == symbol), Decimal(0)
        )
        desired = self.config.quantity * proposed
        delta = desired - current
        self.journal.set(decision_key, end.isoformat(), kind="decision")
        if not delta:
            self.journal.append("no_order", {"symbol": symbol, "at": now.isoformat()})
            return
        price = (quote.ask if delta > 0 else quote.bid).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )
        if price <= 0:
            raise FoundryError("price_tick", "Unsupported price tick.")
        intent = Intent(
            client_order_id="sf_"
            + hashlib.sha256(
                encode([self.config.identity, symbol, end.isoformat()])
            ).hexdigest()[:40],
            symbol=symbol,
            side="buy" if delta > 0 else "sell",
            quantity=abs(delta),
            limit_price=price,
        )
        notional = intent.quantity * price
        gross = (
            sum((p.market_value for p in account.positions), Decimal(0))
            + max(delta, Decimal(0)) * price
        )
        if (
            notional > self.config.maximum_order_notional
            or gross > self.config.maximum_position_notional
            or (delta > 0 and notional > min(account.cash, account.buying_power))
        ):
            raise FoundryError(
                "notional_limit",
                "Order, position or available-cash limit blocks admission.",
                409,
            )
        if intent.side == "sell" and intent.quantity > current:
            raise FoundryError(
                "short_sale", "This paper strategy cannot sell short.", 409
            )
        if len(self.journal.order_keys()) >= self.config.max_orders:
            raise FoundryError("order_limit", "Run order budget exhausted.", 409)
        self.admission_enabled()
        key = f"order/{intent.client_order_id}"
        dispatch_at = datetime.now(UTC)
        if (
            not 0
            <= (dispatch_at - quote.at).total_seconds()
            <= self.config.quote_age_seconds
            or not 0 <= (dispatch_at - account.observed_at).total_seconds() <= 15
            or dispatch_at >= closing
        ):
            raise FoundryError(
                "admission_expired",
                "Data/account admission expired before dispatch.",
                409,
            )
        if self.journal.get(key) is not None:
            raise FoundryError(
                "duplicate_intent", "Existing intent requires reconciliation.", 409
            )
        # This transaction must commit before any submission crosses the network.
        self.journal.set(key, {"intent": intent.wire(), "order": None}, kind="intent")
        self.journal.set("state", "reconciliation_required", kind="state")
        if self.journal.get("stopped"):
            raise FoundryError(
                "paper_disabled",
                "Stop arrived before dispatch; retain intent for reconciliation.",
                409,
            )
        result = self.broker.submit(intent)
        if result.intent != intent:
            raise FoundryError(
                "order_divergence",
                "Submission response changed the persisted intent.",
                409,
            )
        self.journal.set(
            key,
            {"intent": intent.wire(), "order": result.wire()},
            kind="order_observation",
        )
        self.reconcile()

    def cancel(self) -> None:
        """Stop future admission first, then cancel owned paper orders only."""
        self.journal.stop()
        self.account()
        for key in self.journal.order_keys():
            cached = self.journal.get(key)["order"]
            if cached is not None and cached["state"] in TERMINAL:
                continue
            value = self.broker.lookup(key.removeprefix("order/"))
            if value is None:
                raise FoundryError(
                    "submission_unknown",
                    "Owned intent cannot yet be located for cancellation.",
                    409,
                )
            if value.state not in TERMINAL:
                self.broker.cancel(value)
        self.journal.append("cancel_requested", {"positions_flattened": False})

    def acquire(self, symbol: str) -> None:
        acquire(
            self.journal, self.config, lambda: self.broker, symbol, datetime.now(UTC)
        )

    def research(self) -> None:
        research(self.journal, self.config)

    def record_session(self) -> None:
        record_session(self.journal, self.broker, self.reconcile(), datetime.now(UTC))

    def session(self, day: str) -> tuple[datetime, datetime]:
        return session(self.broker, day)

    def campaign(self) -> str:
        return campaign(self.journal, self.config, self.broker, datetime.now(UTC))
