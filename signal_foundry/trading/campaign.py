"""Prospective paper observations and scheduled-date accounting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from signal_foundry.boundary import FoundryError
from signal_foundry.trading.alpaca import Alpaca
from signal_foundry.trading.data import session
from signal_foundry.trading.models import Account, PaperConfig, timestamp
from signal_foundry.trading.store import Journal


def record_session(
    journal: Journal, broker: Alpaca, account: Account, now: datetime
) -> None:
    """Record a real current broker date once, only after clean reconciliation."""
    clock = broker.clock()
    if abs((now - clock.at).total_seconds()) > 5:
        raise FoundryError(
            "market_clock", "Cannot record evidence against a stale clock."
        )
    day = now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    _, closing = session(broker, day)
    if now < closing + timedelta(minutes=1):
        raise FoundryError(
            "session_incomplete",
            "Record the session after the official close and reconciliation.",
            409,
        )
    if broker.open_orders():
        raise FoundryError(
            "session_incomplete",
            "Working orders prevent a completed session observation.",
            409,
        )
    dates = journal.get("observed_dates") or []
    if day in dates:
        raise FoundryError(
            "duplicate_session",
            "This date already has an immutable observation.",
            409,
        )
    if len(dates) >= 366:
        raise FoundryError("campaign_capacity", "Campaign date budget exhausted.")
    journal.set(
        f"session/{day}",
        {
            "date": day,
            "equity": str(account.equity),
            "at": now.isoformat(),
            "source": "alpaca-paper",
            "flat": not account.positions,
        },
        kind="session",
    )
    journal.set("observed_dates", [*dates, day], kind="session_dates")


def campaign(
    journal: Journal, config: PaperConfig, broker: Alpaca, now: datetime
) -> str:
    """Audit elapsed time and missing scheduled dates; never synthesize sessions."""
    created = timestamp(journal.get("created"))
    zone = ZoneInfo("America/New_York")
    elapsed = (now - created).days
    if not 0 <= elapsed <= 366:
        raise FoundryError(
            "campaign_clock", "Campaign is future-dated or exceeds one year."
        )
    rows = broker.transport.request(
        "paper",
        "GET",
        "/v2/calendar",
        query={
            "start": created.astimezone(zone).date().isoformat(),
            "end": now.astimezone(zone).date().isoformat(),
        },
    )
    if not isinstance(rows, list) or len(rows) > 366:
        raise FoundryError("calendar_contract", "Invalid campaign calendar.")
    scheduled = []
    for row in rows:
        closing = (
            datetime.fromisoformat(f"{row['date']}T{row['close']}")
            .replace(tzinfo=zone)
            .astimezone(UTC)
        )
        if created <= closing <= now:
            scheduled.append(row["date"])
    if len(set(scheduled)) != len(scheduled):
        raise FoundryError("calendar_contract", "Duplicate scheduled session.")
    records = [journal.get(f"session/{day}") for day in scheduled]
    missing = [
        day for day, record in zip(scheduled, records, strict=True) if record is None
    ]
    complete = [record for record in records if record is not None and record["flat"]]
    blockers = [
        "Independent review, baseline and capital-checklist evidence remain required.",
        "Live capability is absent.",
    ]
    if elapsed < 42 or len(scheduled) < 30 or len(complete) < 20:
        blockers.append("Minimum elapsed-time/session/sample gates are unmet.")
    if missing or len(complete) != len(records):
        blockers.append("Scheduled sessions are missing or retain overnight exposure.")
    value = {
        "schema_version": "paper-campaign-1",
        "config_identity": config.identity,
        "plan_identity": config.plan.identity,
        "created_at": created.isoformat(),
        "reported_at": now.isoformat(),
        "elapsed_days": elapsed,
        "scheduled_dates": scheduled,
        "missing_dates": missing,
        "observations": records,
        "decision": "NO_GO",
        "blockers": blockers,
    }
    identity = journal.artifact(value)
    journal.set("campaign", identity, kind="campaign_report")
    return identity
