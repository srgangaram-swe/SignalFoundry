"""Immutable intraday acquisition, calendar validation and diagnostic composition."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from signal_foundry.boundary import FoundryError
from signal_foundry.trading.alpaca import Alpaca
from signal_foundry.trading.models import Bar, PaperConfig
from signal_foundry.trading.research import evaluate
from signal_foundry.trading.store import Journal


def acquire(
    journal: Journal,
    config: PaperConfig,
    broker_factory: Callable[[], Alpaca],
    symbol: str,
    now: datetime,
) -> None:
    if symbol not in config.plan.symbols:
        raise FoundryError("paper_symbol", "Symbol is outside the frozen universe.")
    key = f"dataset/{symbol}"
    identity = journal.get(key)
    if identity is not None:
        journal.read_artifact(identity)
        return
    plan = config.plan
    if plan.end > now:
        raise FoundryError(
            "future_data", "The frozen historical interval has not finished."
        )
    broker = broker_factory()
    bars, pages = broker.bars(symbol, plan.feed, plan.start, plan.end)
    # Provider calendar records holidays and early closes; never infer them
    # from weekdays or assume the ordinary 16:00 close on every date.
    calendar_id = journal.get("calendar")
    if calendar_id is None:
        calendar = broker.transport.request(
            "paper",
            "GET",
            "/v2/calendar",
            query={
                "start": plan.start.date().isoformat(),
                "end": plan.end.date().isoformat(),
            },
        )
        if not isinstance(calendar, list) or not 1 <= len(calendar) <= 366:
            raise FoundryError("calendar_contract", "Invalid exchange calendar.")
        calendar_id = journal.artifact(calendar)
        journal.set("calendar", calendar_id, kind="calendar")
    sessions = journal.read_artifact(calendar_id)
    boundaries: dict[str, tuple[datetime, datetime]] = {}
    zone = ZoneInfo("America/New_York")
    for session in sessions:
        opening = (
            datetime.fromisoformat(f"{session['date']}T{session['open']}")
            .replace(tzinfo=zone)
            .astimezone(UTC)
        )
        closing = (
            datetime.fromisoformat(f"{session['date']}T{session['close']}")
            .replace(tzinfo=zone)
            .astimezone(UTC)
        )
        if (
            not opening < closing
            or closing - opening > timedelta(hours=8)
            or session["date"] in boundaries
        ):
            raise FoundryError(
                "calendar_contract", "Invalid or duplicate exchange session."
            )
        boundaries[session["date"]] = (opening, closing)
    regular = tuple(
        b
        for b in bars
        if (bounds := boundaries.get(b.at.astimezone(zone).date().isoformat()))
        is not None
        and bounds[0] <= b.at < bounds[1]
    )
    if not regular:
        raise FoundryError("data_session", "No regular-session observations.")
    raw = [journal.artifact(page) for page in pages]
    value = {
        "schema_version": "intraday-data-1",
        "evidence_kind": "dataset",
        "evidence_class": "measured",
        "symbol": symbol,
        "plan_identity": plan.identity,
        "config_identity": config.identity,
        "retrieved_at": now.isoformat(),
        "feed": plan.feed,
        "adjustment": "raw",
        "bar_seconds": 60,
        "calendar_identity": calendar_id,
        "raw_pages": raw,
        "bars": [b.wire() for b in regular],
        "point_in_time_universe_complete": False,
        "historical_revisions_complete": False,
        "corporate_actions_complete": False,
    }
    identity = journal.artifact(value)
    journal.set(key, identity, kind="dataset")


def research(journal: Journal, config: PaperConfig) -> None:
    reports = {}
    for symbol in config.plan.symbols:
        identity = journal.get(f"dataset/{symbol}")
        if identity is None:
            raise FoundryError(
                "data_missing", "Acquire every frozen universe symbol first."
            )
        value = journal.read_artifact(identity)
        bars = tuple(Bar.model_validate(row) for row in value["bars"])
        reports[symbol] = evaluate(bars, config.plan)
    identity = journal.artifact(
        {
            "plan_identity": config.plan.identity,
            "reports": reports,
            "decision": "NO_GO",
        }
    )
    journal.set("research", identity, kind="research")


def session(broker: Alpaca, day: str) -> tuple[datetime, datetime]:
    rows = broker.transport.request(
        "paper", "GET", "/v2/calendar", query={"start": day, "end": day}
    )
    if not isinstance(rows, list) or len(rows) != 1 or rows[0]["date"] != day:
        raise FoundryError(
            "market_session",
            "No unique official market session for this date.",
            409,
        )
    zone = ZoneInfo("America/New_York")
    opening = (
        datetime.fromisoformat(f"{day}T{rows[0]['open']}")
        .replace(tzinfo=zone)
        .astimezone(UTC)
    )
    closing = (
        datetime.fromisoformat(f"{day}T{rows[0]['close']}")
        .replace(tzinfo=zone)
        .astimezone(UTC)
    )
    if not opening < closing or closing - opening > timedelta(hours=8):
        raise FoundryError("calendar_contract", "Invalid session interval.")
    return opening, closing
