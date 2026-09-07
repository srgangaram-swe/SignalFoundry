"""Fail-closed consumer for the Signal Foundry market-data contract.

This module intentionally does not import Signalattice. The repositories
communicate through a versioned manifest and content-addressed Parquet files,
and AlphaForge independently verifies every trust-boundary invariant before
mapping observations into its canonical adjusted-OHLCV representation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.data.schemas import validate_panel

CONTRACT_NAME = "signal-foundry-market-data"
SCHEMA_VERSION = "1.1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0", SCHEMA_VERSION})
MANIFEST_NAME = "manifest.json"
CONTRACT_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "instrument_id",
    "currency",
    "exchange_calendar",
    "adjustment_state",
    "source",
    "source_table",
)
SEMANTIC_MANIFEST_FIELDS: tuple[str, ...] = (
    "contract",
    "schema_version",
    "source_snapshot_hash",
    "source_manifest_sha256",
    "producer_git_sha",
    "files",
    "columns",
    "rows",
    "date_min",
    "date_max",
    "tickers",
    "temporal_contract",
    "point_in_time_limits",
    "license",
    "source_provenance",
)
V1_1_SEMANTIC_MANIFEST_FIELDS: tuple[str, ...] = (
    *SEMANTIC_MANIFEST_FIELDS,
    "universe_files",
    "universe_columns",
    "universe_rows",
    "corporate_action_files",
    "corporate_action_columns",
    "corporate_action_rows",
)
UNIVERSE_COLUMNS: tuple[str, ...] = (
    "membership_id",
    "universe_id",
    "instrument_id",
    "ticker",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "is_member",
    "reason",
    "source",
    "source_table",
)
CORPORATE_ACTION_COLUMNS: tuple[str, ...] = (
    "action_id",
    "instrument_id",
    "ticker",
    "action_type",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "cash_amount",
    "split_ratio",
    "currency",
    "old_ticker",
    "new_ticker",
    "adjustment_state",
    "source",
    "source_table",
)
ACTION_TYPES = frozenset(
    {"cash_dividend", "delisting", "merger", "spinoff", "split", "symbol_change"}
)
ADJUSTED_CLOSE_STATES = frozenset(
    {
        "provider_adjusted_close_unadjusted_ohlc",
        "synthetic_fixture",
        "synthetic_benchmark",
    }
)
UNADJUSTED_STATES = frozenset({"provider_unadjusted"})
SUPPORTED_CALENDARS = frozenset({"XNYS"})
SUPPORTED_CURRENCIES = frozenset({"USD"})


class SignalFoundryDataError(ValueError):
    """Raised when a producer bundle cannot be trusted or mapped safely."""


@dataclass(frozen=True)
class PointInTimeDiagnostics:
    """Immutable summary of evidence present at the research-data boundary.

    The flags report what the producer attests and what this consumer can
    independently inspect. They are not a claim that a backtest is unbiased.
    """

    schema_version: str
    historical_revisions_complete: bool
    universe_membership_point_in_time: bool
    corporate_actions_complete: bool
    has_universe_records: bool
    has_corporate_action_records: bool
    adjustment_states: tuple[str, ...]
    survivorship_risk: bool
    corporate_action_risk: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SignalFoundryDataset:
    """Verified source observations and their canonical AlphaForge view.

    Auxiliary event frames are evidence only. AlphaForge never mutates price
    bars from corporate-action records at load time because the price contract
    already declares its adjustment policy.
    """

    bundle_dir: Path
    manifest: dict[str, Any]
    source_panel: pd.DataFrame
    panel: pd.DataFrame
    decision_panel: pd.DataFrame | None = None
    universe_records: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(UNIVERSE_COLUMNS))
    )
    corporate_actions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(CORPORATE_ACTION_COLUMNS))
    )
    point_in_time_diagnostics: PointInTimeDiagnostics | None = None
    as_of: pd.Timestamp | None = None

    @property
    def bundle_id(self) -> str:
        """Return the immutable producer identity."""
        return str(self.manifest["bundle_id"])

    def universe_membership_as_of(
        self,
        universe_id: str,
        decision_at: str | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Reconstruct the latest visible state for every known instrument.

        Returned rows include explicit exits (``is_member=False``), making
        delistings and removals observable instead of silently dropping them.
        The operation fails closed unless schema 1.1 records and a complete
        point-in-time membership attestation are present.
        """
        diagnostics = self.point_in_time_diagnostics
        if diagnostics is None or diagnostics.schema_version != SCHEMA_VERSION:
            raise SignalFoundryDataError(
                "historical universe reconstruction requires schema 1.1 records"
            )
        if not diagnostics.universe_membership_point_in_time:
            raise SignalFoundryDataError(
                "bundle does not attest point-in-time universe membership completeness"
            )
        if not isinstance(universe_id, str) or not universe_id.strip():
            raise SignalFoundryDataError("universe_id must be a non-empty string")
        cutoff = _decision_timestamp(decision_at)
        if self.as_of is not None and cutoff != self.as_of:
            raise SignalFoundryDataError(
                "requested universe timestamp must equal the dataset as-of boundary"
            )
        visible = _visible_revisions(
            self.universe_records.loc[self.universe_records["universe_id"].eq(universe_id)],
            as_of=cutoff,
            identity_column="membership_id",
        )
        if visible.empty:
            raise SignalFoundryDataError(
                f"no visible membership evidence for universe {universe_id!r}"
            )
        ordered = visible.sort_values(
            [
                "instrument_id",
                "effective_at",
                "available_at",
                "observed_at",
                "membership_id",
            ],
            kind="stable",
        )
        return ordered.drop_duplicates("instrument_id", keep="last").reset_index(drop=True)

    def active_universe_as_of(
        self,
        universe_id: str,
        decision_at: str | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return active members after reconstructing explicit entry/exit state."""
        membership = self.universe_membership_as_of(universe_id, decision_at)
        return membership.loc[membership["is_member"]].reset_index(drop=True)

    def corporate_actions_as_of(
        self,
        decision_at: str | datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Return the latest visible revision of each effective action event."""
        cutoff = _decision_timestamp(decision_at)
        if self.as_of is not None and cutoff != self.as_of:
            raise SignalFoundryDataError(
                "requested corporate-action timestamp must equal the dataset as-of boundary"
            )
        return _visible_revisions(
            self.corporate_actions,
            as_of=cutoff,
            identity_column="action_id",
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _decision_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SignalFoundryDataError("decision timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise SignalFoundryDataError("decision timestamp must include an explicit timezone")
    return timestamp.tz_convert(UTC)


def _safe_file(bundle_dir: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SignalFoundryDataError(f"unsafe bundle path: {relative!r}")
    root = bundle_dir.resolve()
    resolved = (bundle_dir / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise SignalFoundryDataError(f"bundle path escapes root: {relative!r}")
    return resolved


def _read_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignalFoundryDataError(f"cannot read canonical manifest at {path}") from exc
    if not isinstance(value, dict):
        raise SignalFoundryDataError("bundle manifest must be a JSON object")
    if value.get("contract") != CONTRACT_NAME:
        raise SignalFoundryDataError("unsupported Signal Foundry contract")
    schema_version = value.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SignalFoundryDataError(
            f"unsupported schema version {value.get('schema_version')!r}; "
            f"supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    bundle_id = value.get("bundle_id")
    if not _is_sha256(bundle_id) or bundle_dir.name != bundle_id:
        raise SignalFoundryDataError("bundle identity does not match its directory")
    if value.get("columns") != list(CONTRACT_COLUMNS):
        raise SignalFoundryDataError("bundle manifest declares an unsupported column schema")
    if schema_version == SCHEMA_VERSION:
        if value.get("universe_columns") != list(UNIVERSE_COLUMNS):
            raise SignalFoundryDataError("bundle declares an unsupported universe schema")
        if value.get("corporate_action_columns") != list(CORPORATE_ACTION_COLUMNS):
            raise SignalFoundryDataError("bundle declares an unsupported corporate-action schema")
    return value


def _validate_manifest_policy(manifest: dict[str, Any]) -> None:
    license_policy = manifest.get("license")
    if not isinstance(license_policy, dict):
        raise SignalFoundryDataError("bundle manifest lacks license policy")
    redistributable = license_policy.get("observations_redistributable")
    must_remain_local = license_policy.get("bundle_must_remain_local")
    aggregate_only = license_policy.get("public_evidence_must_be_aggregate_or_synthetic")
    policy_values = (redistributable, must_remain_local, aggregate_only)
    if not all(isinstance(value, bool) for value in policy_values):
        raise SignalFoundryDataError("bundle license policy must contain explicit booleans")
    if must_remain_local is redistributable or aggregate_only is redistributable:
        raise SignalFoundryDataError("bundle license policy is internally inconsistent")

    limits = manifest.get("point_in_time_limits")
    if not isinstance(limits, dict):
        raise SignalFoundryDataError("bundle manifest lacks point-in-time limitations")
    required_limits = {
        "historical_revisions_complete",
        "universe_membership_point_in_time",
        "corporate_actions_complete",
    }
    if set(limits) != required_limits or not all(isinstance(limits[key], bool) for key in limits):
        raise SignalFoundryDataError(
            "bundle point-in-time limitations must explicitly cover revisions, universe, "
            "and corporate actions"
        )

    temporal = manifest.get("temporal_contract")
    if not isinstance(temporal, dict) or temporal.get("as_of_rule") != (
        "available_at <= decision timestamp"
    ):
        raise SignalFoundryDataError("bundle temporal contract is missing the supported as-of rule")


def _strict_utc_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    family: str,
) -> None:
    for column in columns:
        raw = frame[column]
        try:
            parsed = pd.to_datetime(raw, errors="raise", utc=False)
        except (TypeError, ValueError) as exc:
            raise SignalFoundryDataError(
                f"{family}.{column} contains an invalid timestamp"
            ) from exc
        if column == "provider_updated_at" and parsed.isna().all():
            frame[column] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
            continue
        if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise SignalFoundryDataError(
                f"{family}.{column} must contain explicit timezone-aware timestamps"
            )
        frame[column] = parsed.dt.tz_convert("UTC")


def _validate_auxiliary_temporal_order(frame: pd.DataFrame, family: str) -> None:
    required = ["effective_at", "available_at", "observed_at"]
    if frame[required].isna().any().any():
        raise SignalFoundryDataError(f"{family} contains missing required timestamps")
    if (frame["observed_at"] < frame["available_at"]).any():
        raise SignalFoundryDataError(f"{family}.observed_at precedes available_at")
    provider_time = frame["provider_updated_at"]
    if (provider_time.notna() & provider_time.gt(frame["observed_at"])).any():
        raise SignalFoundryDataError(f"{family}.provider_updated_at follows observed_at")


def _validate_text_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    family: str,
) -> None:
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise SignalFoundryDataError(f"{family}.{column} contains an empty value")
        frame[column] = frame[column].astype(str)


def _coerce_universe_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))
    if list(records.columns) != list(UNIVERSE_COLUMNS):
        raise SignalFoundryDataError("universe columns do not match schema 1.1.0")
    frame = records.copy()
    _strict_utc_columns(
        frame,
        ("effective_at", "available_at", "observed_at", "provider_updated_at"),
        family="universe",
    )
    _validate_auxiliary_temporal_order(frame, "universe")
    _validate_text_columns(
        frame,
        (
            "membership_id",
            "universe_id",
            "instrument_id",
            "ticker",
            "reason",
            "source",
            "source_table",
        ),
        family="universe",
    )
    if not frame["is_member"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise SignalFoundryDataError("universe.is_member must be boolean")
    frame["is_member"] = frame["is_member"].astype(bool)
    if frame.duplicated(["membership_id", "available_at"]).any():
        raise SignalFoundryDataError(
            "universe contains a duplicate (membership_id, available_at) revision"
        )
    stable_identity = frame.groupby("membership_id", sort=False)[
        ["universe_id", "instrument_id"]
    ].nunique()
    if stable_identity.gt(1).any().any():
        raise SignalFoundryDataError("universe membership identity changes across revisions")
    state_key = ["universe_id", "instrument_id", "effective_at", "available_at"]
    if frame.duplicated(state_key).any():
        raise SignalFoundryDataError("universe contains ambiguous simultaneous membership states")
    return frame.sort_values(
        ["universe_id", "instrument_id", "effective_at", "available_at"], kind="stable"
    ).reset_index(drop=True)


def _coerce_corporate_action_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=list(CORPORATE_ACTION_COLUMNS))
    if list(records.columns) != list(CORPORATE_ACTION_COLUMNS):
        raise SignalFoundryDataError("corporate-action columns do not match schema 1.1.0")
    frame = records.copy()
    _strict_utc_columns(
        frame,
        ("effective_at", "available_at", "observed_at", "provider_updated_at"),
        family="corporate_actions",
    )
    _validate_auxiliary_temporal_order(frame, "corporate_actions")
    _validate_text_columns(
        frame,
        (
            "action_id",
            "instrument_id",
            "ticker",
            "action_type",
            "currency",
            "adjustment_state",
            "source",
            "source_table",
        ),
        family="corporate_actions",
    )
    for column in ("old_ticker", "new_ticker"):
        frame[column] = frame[column].fillna("").astype(str)
    unknown_actions = sorted(set(frame["action_type"]).difference(ACTION_TYPES))
    if unknown_actions:
        raise SignalFoundryDataError(
            f"corporate_actions has unsupported action types: {unknown_actions}"
        )
    unknown_currencies = sorted(set(frame["currency"]).difference(SUPPORTED_CURRENCIES))
    if unknown_currencies:
        raise SignalFoundryDataError(
            f"corporate_actions contains unsupported currencies: {unknown_currencies}"
        )
    for column in ("cash_amount", "split_ratio"):
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any():
            raise SignalFoundryDataError(f"corporate_actions.{column} contains an invalid value")
        if (numeric.dropna() <= 0).any():
            raise SignalFoundryDataError(
                f"corporate_actions.{column} must be positive when present"
            )
        frame[column] = numeric
    splits = frame["action_type"].eq("split")
    if frame.loc[splits, "split_ratio"].isna().any():
        raise SignalFoundryDataError("split actions require split_ratio")
    dividends = frame["action_type"].eq("cash_dividend")
    if frame.loc[dividends, "cash_amount"].isna().any():
        raise SignalFoundryDataError("cash-dividend actions require cash_amount")
    symbol_changes = frame["action_type"].eq("symbol_change")
    if frame.loc[symbol_changes, "new_ticker"].str.strip().eq("").any():
        raise SignalFoundryDataError("symbol-change actions require new_ticker")
    if frame.duplicated(["action_id", "available_at"]).any():
        raise SignalFoundryDataError(
            "corporate_actions contains a duplicate (action_id, available_at) revision"
        )
    stable_identity = frame.groupby("action_id", sort=False)[
        ["instrument_id", "action_type"]
    ].nunique()
    if stable_identity.gt(1).any().any():
        raise SignalFoundryDataError("corporate-action identity changes across revisions")
    return frame.sort_values(
        ["instrument_id", "effective_at", "available_at"], kind="stable"
    ).reset_index(drop=True)


def _visible_revisions(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    identity_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    visible = frame.loc[
        frame["effective_at"].le(as_of) & frame["available_at"].le(as_of)
    ].sort_values([identity_column, "available_at", "observed_at"], kind="stable")
    return visible.drop_duplicates(identity_column, keep="last").reset_index(drop=True)


def _coerce_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != list(CONTRACT_COLUMNS):
        raise SignalFoundryDataError("bundle partition column order does not match contract")
    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="raise")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise SignalFoundryDataError("bundle market dates must be timezone-naive session labels")
    if not dates.eq(dates.dt.normalize()).all():
        raise SignalFoundryDataError("bundle market dates must not contain intraday timestamps")
    out["date"] = dates
    for column in (
        "effective_at",
        "available_at",
        "observed_at",
        "provider_updated_at",
    ):
        raw = out[column]
        if raw.notna().any():
            try:
                aware = raw.dropna().map(lambda value: pd.Timestamp(value).tzinfo is not None)
            except (TypeError, ValueError) as exc:
                raise SignalFoundryDataError(
                    f"bundle contains invalid temporal values in {column!r}"
                ) from exc
            if not aware.all():
                raise SignalFoundryDataError(
                    f"bundle temporal column {column!r} must be timezone-aware"
                )
        out[column] = pd.to_datetime(raw, errors="coerce", utc=True)
    if out[["effective_at", "available_at", "observed_at"]].isna().any().any():
        raise SignalFoundryDataError("bundle contains invalid required temporal values")
    if (out["available_at"] < out["effective_at"]).any():
        raise SignalFoundryDataError("bundle contains availability before effective time")
    if (out["observed_at"] < out["effective_at"]).any():
        raise SignalFoundryDataError("bundle contains observation before effective time")
    if (out["observed_at"] < out["available_at"]).any():
        raise SignalFoundryDataError("bundle contains observation before availability time")
    provider_time = out["provider_updated_at"]
    if (provider_time.notna() & provider_time.gt(out["observed_at"])).any():
        raise SignalFoundryDataError("bundle contains provider update after observation time")
    if out.duplicated(["date", "ticker"]).any():
        raise SignalFoundryDataError("bundle contains duplicate (date, ticker) rows")

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if not np.isfinite(out[numeric_columns].to_numpy(dtype=float)).all():
        raise SignalFoundryDataError("bundle contains missing or non-finite market values")
    if (out[["open", "high", "low", "close", "adj_close"]] <= 0).any().any():
        raise SignalFoundryDataError("bundle contains non-positive prices")
    if (out["volume"] < 0).any():
        raise SignalFoundryDataError("bundle contains negative volume")
    if (
        (out["high"] < out["low"])
        | (out["open"] > out["high"])
        | (out["open"] < out["low"])
        | (out["close"] > out["high"])
        | (out["close"] < out["low"])
    ).any():
        raise SignalFoundryDataError("bundle violates OHLC bounds")

    text_columns = [
        "ticker",
        "instrument_id",
        "currency",
        "exchange_calendar",
        "adjustment_state",
        "source",
        "source_table",
    ]
    for column in text_columns:
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise SignalFoundryDataError(f"bundle column {column!r} contains empty values")
        out[column] = out[column].astype(str)
    unknown_calendars = sorted(set(out["exchange_calendar"]) - SUPPORTED_CALENDARS)
    if unknown_calendars:
        raise SignalFoundryDataError(
            f"bundle contains unsupported exchange calendars: {unknown_calendars}"
        )
    unknown_currencies = sorted(set(out["currency"]) - SUPPORTED_CURRENCIES)
    if unknown_currencies:
        raise SignalFoundryDataError(
            f"bundle contains unsupported currencies: {unknown_currencies}"
        )
    if not out["ticker"].eq(out["instrument_id"]).all():
        raise SignalFoundryDataError("bundle ticker and instrument identity disagree")
    return out.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def _to_alphaforge_panel(source: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for adjustment_state, group in source.groupby("adjustment_state", sort=True):
        block = group.copy()
        if adjustment_state in ADJUSTED_CLOSE_STATES:
            factor = block["adj_close"] / block["close"]
            if not np.isfinite(factor).all() or (factor <= 0).any():
                raise SignalFoundryDataError("bundle contains an invalid adjustment factor")
            for column in ("open", "high", "low", "close"):
                block[column] = block[column] * factor
        elif adjustment_state not in UNADJUSTED_STATES:
            raise SignalFoundryDataError(
                f"unsupported adjustment state {adjustment_state!r}; mapping must be explicit"
            )
        blocks.append(block)
    adjusted = pd.concat(blocks, ignore_index=True)
    panel = adjusted.rename(columns={"ticker": "symbol"})[
        ["date", "symbol", "open", "high", "low", "close", "volume"]
    ]
    return validate_panel(panel)


def _to_decision_panel(source: pd.DataFrame) -> pd.DataFrame:
    """Date bars at the first session close where they were actually available."""
    session_closes = (
        source.groupby("date", as_index=False)["effective_at"]
        .max()
        .sort_values("effective_at", kind="stable")
    )
    close_ns = session_closes["effective_at"].astype("int64").to_numpy()
    available_ns = source["available_at"].astype("int64").to_numpy()
    positions = np.searchsorted(close_ns, available_ns, side="left")
    eligible = positions < len(session_closes)
    if not eligible.any():
        raise SignalFoundryDataError(
            "bundle contains no observations available by a represented decision session"
        )
    decision_source = source.loc[eligible].copy()
    decision_source["date"] = session_closes["date"].to_numpy()[positions[eligible]]
    if decision_source.duplicated(["date", "ticker"]).any():
        raise SignalFoundryDataError(
            "availability mapping produces duplicate decision-session observations"
        )
    return _to_alphaforge_panel(decision_source)


def _read_record_set(
    root: Path,
    *,
    entries: object,
    columns: tuple[str, ...],
    expected_rows: object,
    family: str,
    seen_paths: set[str],
    require_files: bool,
) -> pd.DataFrame:
    if not isinstance(entries, list) or (require_files and not entries):
        raise SignalFoundryDataError(f"{family} manifest must contain a file list")
    if type(expected_rows) is not int or expected_rows < 0:
        raise SignalFoundryDataError(f"{family} row count must be a non-negative integer")
    frames: list[pd.DataFrame] = []
    total_rows = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise SignalFoundryDataError(f"{family} file entry must be an object")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        rows = entry.get("rows")
        if (
            not isinstance(relative, str)
            or not _is_sha256(expected_hash)
            or type(rows) is not int
            or rows < 0
        ):
            raise SignalFoundryDataError(f"{family} file entry is incomplete")
        if relative in seen_paths:
            raise SignalFoundryDataError(f"duplicate declared data path: {relative}")
        seen_paths.add(relative)
        path = _safe_file(root, relative)
        if not path.is_file():
            raise SignalFoundryDataError(f"{family} data file is missing: {relative}")
        if _sha256_file(path) != expected_hash:
            raise SignalFoundryDataError(f"{family} data hash mismatch: {relative}")
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise SignalFoundryDataError(f"{family} data is unreadable: {relative}") from exc
        if list(frame.columns) != list(columns):
            raise SignalFoundryDataError(f"{family} schema mismatch: {relative}")
        if len(frame) != rows:
            raise SignalFoundryDataError(f"{family} row-count mismatch: {relative}")
        total_rows += len(frame)
        frames.append(frame)
    if total_rows != expected_rows:
        raise SignalFoundryDataError(f"{family} aggregate row-count mismatch")
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


def _build_diagnostics(
    manifest: dict[str, Any],
    source: pd.DataFrame,
    universe: pd.DataFrame,
    corporate_actions: pd.DataFrame,
) -> PointInTimeDiagnostics:
    limits = manifest["point_in_time_limits"]
    schema_version = str(manifest["schema_version"])
    warnings: list[str] = []
    if schema_version != SCHEMA_VERSION:
        warnings.append(
            "schema 1.0 has no independently inspectable universe or corporate-action families"
        )
    if not limits["historical_revisions_complete"]:
        warnings.append("historical provider revisions are incomplete")
    if not limits["universe_membership_point_in_time"]:
        warnings.append(
            "point-in-time universe membership is incomplete; survivorship risk remains"
        )
    if not limits["corporate_actions_complete"]:
        warnings.append("corporate-action history is incomplete; adjustment risk remains")
    if schema_version == SCHEMA_VERSION and universe.empty:
        warnings.append("schema 1.1 bundle contains no universe-membership records")
    if schema_version == SCHEMA_VERSION and corporate_actions.empty:
        warnings.append("schema 1.1 bundle contains no corporate-action records")
    return PointInTimeDiagnostics(
        schema_version=schema_version,
        historical_revisions_complete=limits["historical_revisions_complete"],
        universe_membership_point_in_time=limits["universe_membership_point_in_time"],
        corporate_actions_complete=limits["corporate_actions_complete"],
        has_universe_records=not universe.empty,
        has_corporate_action_records=not corporate_actions.empty,
        adjustment_states=tuple(sorted(source["adjustment_state"].unique().tolist())),
        survivorship_risk=(
            schema_version != SCHEMA_VERSION or not limits["universe_membership_point_in_time"]
        ),
        corporate_action_risk=(
            schema_version != SCHEMA_VERSION or not limits["corporate_actions_complete"]
        ),
        warnings=tuple(warnings),
    )


def load_signal_foundry_dataset(
    bundle_dir: str | Path,
    *,
    as_of: str | datetime | pd.Timestamp | None = None,
) -> SignalFoundryDataset:
    """Verify and load one immutable Signal Foundry bundle.

    ``as_of`` applies both ``effective_at <= decision timestamp`` and the
    producer rule ``available_at <= decision timestamp`` to every record
    family. It must include an explicit timezone. Omit it only when a
    downstream temporal split will enforce decision-time eligibility row by
    row.
    """
    root = Path(bundle_dir)
    manifest = _read_manifest(root)
    _validate_manifest_policy(manifest)
    semantic_fields = (
        V1_1_SEMANTIC_MANIFEST_FIELDS
        if manifest["schema_version"] == SCHEMA_VERSION
        else SEMANTIC_MANIFEST_FIELDS
    )
    try:
        semantic = {key: manifest[key] for key in semantic_fields}
    except KeyError as exc:
        raise SignalFoundryDataError(
            f"bundle manifest is missing identity field: {exc.args[0]}"
        ) from exc
    if _sha256_bytes(_canonical_json(semantic)) != manifest["bundle_id"]:
        raise SignalFoundryDataError("bundle semantic identity mismatch")

    seen_paths: set[str] = set()
    source = _coerce_source_frame(
        _read_record_set(
            root,
            entries=manifest.get("files"),
            columns=CONTRACT_COLUMNS,
            expected_rows=manifest.get("rows"),
            family="prices",
            seen_paths=seen_paths,
            require_files=True,
        )
    )
    if str(source["date"].min().date()) != manifest.get("date_min"):
        raise SignalFoundryDataError("bundle minimum date mismatch")
    if str(source["date"].max().date()) != manifest.get("date_max"):
        raise SignalFoundryDataError("bundle maximum date mismatch")
    if sorted(source["ticker"].unique().tolist()) != manifest.get("tickers"):
        raise SignalFoundryDataError("bundle ticker universe mismatch")

    universe = pd.DataFrame(columns=list(UNIVERSE_COLUMNS))
    corporate_actions = pd.DataFrame(columns=list(CORPORATE_ACTION_COLUMNS))
    if manifest["schema_version"] == SCHEMA_VERSION:
        universe = _coerce_universe_records(
            _read_record_set(
                root,
                entries=manifest.get("universe_files"),
                columns=UNIVERSE_COLUMNS,
                expected_rows=manifest.get("universe_rows"),
                family="universe",
                seen_paths=seen_paths,
                require_files=False,
            )
        )
        corporate_actions = _coerce_corporate_action_records(
            _read_record_set(
                root,
                entries=manifest.get("corporate_action_files"),
                columns=CORPORATE_ACTION_COLUMNS,
                expected_rows=manifest.get("corporate_action_rows"),
                family="corporate_actions",
                seen_paths=seen_paths,
                require_files=False,
            )
        )
    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*.parquet") if path.is_file()
    }
    if actual_paths != seen_paths:
        raise SignalFoundryDataError("bundle contains missing or undeclared parquet files")
    diagnostics = _build_diagnostics(manifest, source, universe, corporate_actions)

    cutoff: pd.Timestamp | None = None
    if as_of is not None:
        cutoff = _decision_timestamp(as_of)
        source = source.loc[
            source["effective_at"].le(cutoff) & source["available_at"].le(cutoff)
        ].reset_index(drop=True)
        if source.empty:
            raise SignalFoundryDataError("as-of cutoff excludes every bundle observation")
        universe = _visible_revisions(
            universe,
            as_of=cutoff,
            identity_column="membership_id",
        )
        corporate_actions = _visible_revisions(
            corporate_actions,
            as_of=cutoff,
            identity_column="action_id",
        )
    panel = _to_alphaforge_panel(source)
    decision_panel = _to_decision_panel(source)
    return SignalFoundryDataset(
        bundle_dir=root,
        manifest=manifest,
        source_panel=source,
        panel=panel,
        decision_panel=decision_panel,
        universe_records=universe,
        corporate_actions=corporate_actions,
        point_in_time_diagnostics=diagnostics,
        as_of=cutoff,
    )
