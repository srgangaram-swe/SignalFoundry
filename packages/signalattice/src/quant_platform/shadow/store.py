"""Durable, idempotent persistence for shadow campaigns.

SF-S5-SL-MR4. Sealing a batch is the only operation here that must be
all-or-nothing: a half-written batch would present a partial universe as though
it were the model's complete opinion. It runs inside one ``BEGIN IMMEDIATE``
transaction, so a crash mid-seal leaves no batch rather than a truncated one.

**Re-sealing an identical batch is a success, not a conflict.** The batch
identity is content-derived, so a retry after an ambiguous failure presents the
same identity; treating that as an error would push callers toward the exact
blind-retry behaviour that creates duplicates. Re-sealing *different* forecasts
under the same campaign and as-of is a conflict and is refused.

**Outcomes only ever append.** A correction inserts a new revision; the database
triggers make overwriting impossible even from a raw connection. Restart
recovery therefore needs no repair path: whatever is on disk is what was
committed, and nothing can have been partially rewritten.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from quant_platform.shadow.contracts import (
    CampaignState,
    OutcomeStatus,
    ProbabilityVector,
    SealedBatch,
    SealedCampaignError,
    ShadowForecast,
    ShadowOutcome,
    ShadowValidationError,
    assert_campaign_transition,
    utc_instant,
    validate_campaign_name,
)
from quant_platform.tracking.contracts import ConflictError, NotFoundError

#: Bounded so a malformed request cannot ask the store to materialise an
#: unbounded result set.
MAX_QUERY_ROWS: Final = 100_000

#: SQLite busy timeout. A contended writer waits rather than failing instantly,
#: but the wait is bounded so a deadlocked caller surfaces as an error.
BUSY_TIMEOUT_MS: Final = 5_000

_TIMESTAMP_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%fZ"


def _encode_instant(value: datetime) -> str:
    """Return the canonical 27-character UTC form the schema constrains."""
    return value.astimezone(UTC).strftime(_TIMESTAMP_FORMAT)


def _decode_instant(value: str) -> datetime:
    """Parse the canonical form back to an aware UTC datetime.

    Raises:
        ShadowValidationError: If the stored text is not canonical. A row that
            cannot be parsed is corruption, not a value to guess at.
    """
    try:
        return datetime.strptime(value, _TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except (TypeError, ValueError) as error:
        raise ShadowValidationError(
            f"stored timestamp {value!r} is not canonical: {error}"
        ) from error


class ShadowStore:
    """Append-only campaign storage over the shared registry database.

    The database must already carry schema version 2; this class does not
    migrate. Migration stays in the registry's checksum-verified chain so a
    drifted database fails there rather than being silently upgraded here.
    """

    def __init__(self, database: str | Path) -> None:
        self._path = Path(database).expanduser()

    def _connect(self) -> sqlite3.Connection:
        """Open a bounded connection with foreign keys and WAL enforced."""
        connection = sqlite3.connect(
            self._path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        return connection

    def create_campaign(
        self,
        campaign: str,
        *,
        horizon_days: int,
        class_labels: Sequence[str],
        model_identity: str,
        now: datetime,
    ) -> None:
        """Register a campaign in ``DRAFT``.

        Raises:
            ConflictError: If the campaign already exists.
            ShadowValidationError: On malformed inputs.
        """
        name = validate_campaign_name(campaign)
        labels = tuple(class_labels)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ShadowValidationError("class_labels must be at least two unique labels")
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
            raise ShadowValidationError("horizon_days must be an int")
        moment = utc_instant(now, field_name="now")
        with closing(self._connect()) as connection:
            try:
                connection.execute(
                    "INSERT INTO sl_shadow_campaigns "
                    "(campaign, state, horizon_days, class_labels, created_at, model_identity) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        name,
                        CampaignState.DRAFT.value,
                        horizon_days,
                        json.dumps(list(labels), sort_keys=False),
                        _encode_instant(moment),
                        model_identity,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ConflictError(f"campaign {name!r} already exists") from error

    def campaign_state(self, campaign: str) -> CampaignState:
        """Return the campaign's current state.

        Raises:
            NotFoundError: If the campaign does not exist.
        """
        name = validate_campaign_name(campaign)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state FROM sl_shadow_campaigns WHERE campaign = ?", (name,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"campaign {name!r} does not exist")
        return CampaignState(row["state"])

    def transition(self, campaign: str, target: CampaignState, *, now: datetime) -> None:
        """Move a campaign forward, recording the transition.

        Raises:
            ShadowValidationError: If the transition is not permitted.
            NotFoundError: If the campaign does not exist.
        """
        current = self.campaign_state(campaign)
        assert_campaign_transition(current, target)
        moment = _encode_instant(utc_instant(now, field_name="now"))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "UPDATE sl_shadow_campaigns SET state = ? WHERE campaign = ?",
                    (target.value, campaign),
                )
                connection.execute(
                    "INSERT INTO sl_shadow_campaign_transitions "
                    "(campaign, from_state, to_state, occurred_at) VALUES (?, ?, ?, ?)",
                    (campaign, current.value, target.value, moment),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def seal_batch(self, batch: SealedBatch) -> str:
        """Persist a complete batch atomically and return its identity.

        Re-sealing the identical batch returns the same identity without writing
        again. Re-sealing a *different* batch for the same campaign and as-of is
        a conflict: one decision instant has one opinion.

        Raises:
            SealedCampaignError: If the campaign is not ``ACTIVE``.
            ConflictError: If a different batch already exists for this as-of.
        """
        if not isinstance(batch, SealedBatch):
            raise ShadowValidationError("batch must be a SealedBatch")
        state = self.campaign_state(batch.campaign)
        if state is not CampaignState.ACTIVE:
            raise SealedCampaignError(
                f"campaign {batch.campaign!r} is {state.value!r}; forecasts may only be "
                "sealed while it is active"
            )
        as_of = _encode_instant(batch.as_of)
        with closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT batch_id FROM sl_shadow_batches WHERE campaign = ? AND as_of = ?",
                (batch.campaign, as_of),
            ).fetchone()
            if existing is not None:
                if existing["batch_id"] == batch.batch_id:
                    return batch.batch_id
                raise ConflictError(
                    f"a different batch is already sealed for {batch.campaign!r} at "
                    f"{as_of}: stored {existing['batch_id'][:12]}, offered "
                    f"{batch.batch_id[:12]}. One decision instant has one opinion."
                )
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO sl_shadow_batches "
                    "(batch_id, campaign, as_of, sealed_at, expected_universe, forecast_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        batch.batch_id,
                        batch.campaign,
                        as_of,
                        _encode_instant(batch.sealed_at),
                        json.dumps(list(batch.expected_universe)),
                        len(batch.forecasts),
                    ),
                )
                connection.executemany(
                    "INSERT INTO sl_shadow_forecasts "
                    "(forecast_id, batch_id, campaign, symbol, as_of, target_instant, "
                    "horizon_days, distribution, feature_prefix_digest, model_identity) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            item.forecast_id,
                            batch.batch_id,
                            item.campaign,
                            item.symbol,
                            _encode_instant(item.as_of),
                            _encode_instant(item.target_instant),
                            item.horizon_days,
                            json.dumps(item.distribution.to_dict(), sort_keys=True),
                            item.feature_prefix_digest,
                            item.model_identity,
                        )
                        for item in batch.forecasts
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return batch.batch_id

    def append_outcome(self, outcome: ShadowOutcome) -> int:
        """Append an outcome revision and return the revision recorded.

        Raises:
            NotFoundError: If the forecast is unknown.
            ConflictError: If that revision already exists with different content.
        """
        if not isinstance(outcome, ShadowOutcome):
            raise ShadowValidationError("outcome must be a ShadowOutcome")
        with closing(self._connect()) as connection:
            forecast = connection.execute(
                "SELECT as_of FROM sl_shadow_forecasts WHERE forecast_id = ?",
                (outcome.forecast_id,),
            ).fetchone()
            if forecast is None:
                raise NotFoundError(
                    f"forecast {outcome.forecast_id[:12]} is unknown; an outcome cannot "
                    "precede the forecast it scores"
                )
            existing = connection.execute(
                "SELECT realized_label, status, observed_at FROM sl_shadow_outcomes "
                "WHERE forecast_id = ? AND revision = ?",
                (outcome.forecast_id, outcome.revision),
            ).fetchone()
            if existing is not None:
                same = (
                    existing["realized_label"] == outcome.realized_label
                    and existing["status"] == outcome.status.value
                    and existing["observed_at"] == _encode_instant(outcome.observed_at)
                )
                if same:
                    return outcome.revision
                raise ConflictError(
                    f"revision {outcome.revision} already exists for "
                    f"{outcome.forecast_id[:12]} with different content; a correction "
                    "appends a new revision rather than replacing one"
                )
            connection.execute(
                "INSERT INTO sl_shadow_outcomes "
                "(forecast_id, revision, realized_label, status, observed_at, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    outcome.forecast_id,
                    outcome.revision,
                    outcome.realized_label,
                    outcome.status.value,
                    _encode_instant(outcome.observed_at),
                    _encode_instant(outcome.recorded_at),
                ),
            )
        return outcome.revision

    def load_forecasts(self, campaign: str) -> tuple[ShadowForecast, ...]:
        """Return every sealed forecast for a campaign in deterministic order.

        Raises:
            ShadowValidationError: If a stored row cannot be decoded.
        """
        name = validate_campaign_name(campaign)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT symbol, as_of, target_instant, horizon_days, distribution, "
                "feature_prefix_digest, model_identity FROM sl_shadow_forecasts "
                "WHERE campaign = ? ORDER BY as_of, symbol LIMIT ?",
                (name, MAX_QUERY_ROWS),
            ).fetchall()
        forecasts = []
        for row in rows:
            payload = json.loads(row["distribution"])
            forecasts.append(
                ShadowForecast(
                    campaign=name,
                    symbol=row["symbol"],
                    as_of=_decode_instant(row["as_of"]),
                    target_instant=_decode_instant(row["target_instant"]),
                    horizon_days=row["horizon_days"],
                    distribution=ProbabilityVector(
                        labels=tuple(payload["labels"]),
                        probabilities=tuple(payload["probabilities"]),
                    ),
                    feature_prefix_digest=row["feature_prefix_digest"],
                    model_identity=row["model_identity"],
                )
            )
        return tuple(forecasts)

    def load_outcomes(self, campaign: str) -> dict[str, tuple[ShadowOutcome, ...]]:
        """Return every outcome revision keyed by forecast identity."""
        name = validate_campaign_name(campaign)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT o.forecast_id, o.revision, o.realized_label, o.status, "
                "o.observed_at, o.recorded_at FROM sl_shadow_outcomes o "
                "JOIN sl_shadow_forecasts f ON f.forecast_id = o.forecast_id "
                "WHERE f.campaign = ? ORDER BY o.forecast_id, o.revision LIMIT ?",
                (name, MAX_QUERY_ROWS),
            ).fetchall()
        grouped: dict[str, list[ShadowOutcome]] = {}
        for row in rows:
            grouped.setdefault(row["forecast_id"], []).append(
                ShadowOutcome(
                    forecast_id=row["forecast_id"],
                    realized_label=row["realized_label"],
                    observed_at=_decode_instant(row["observed_at"]),
                    recorded_at=_decode_instant(row["recorded_at"]),
                    revision=row["revision"],
                    status=OutcomeStatus(row["status"]),
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}

    def summary(self, campaign: str) -> dict[str, Any]:
        """Return bounded counts for operational inspection."""
        name = validate_campaign_name(campaign)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT (SELECT count(*) FROM sl_shadow_batches WHERE campaign = ?) AS batches, "
                "(SELECT count(*) FROM sl_shadow_forecasts WHERE campaign = ?) AS forecasts, "
                "(SELECT count(*) FROM sl_shadow_outcomes o JOIN sl_shadow_forecasts f "
                "ON f.forecast_id = o.forecast_id WHERE f.campaign = ?) AS outcomes",
                (name, name, name),
            ).fetchone()
        return {
            "campaign": name,
            "state": self.campaign_state(name).value,
            "batches": row["batches"],
            "forecasts": row["forecasts"],
            "outcome_revisions": row["outcomes"],
        }


__all__ = ["BUSY_TIMEOUT_MS", "MAX_QUERY_ROWS", "ShadowStore"]
