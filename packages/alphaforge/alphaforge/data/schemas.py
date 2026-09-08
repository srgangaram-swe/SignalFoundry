"""Canonical market data schema and validation.

Every module in AlphaForge consumes the same long-format panel:

    date (datetime64) | symbol (str) | open | high | low | close | volume

`close` is the *adjusted* close (splits/dividends) whenever the source
provides one; unadjusted OHLC are rescaled by the same adjustment factor so
bar geometry stays consistent.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


class SchemaError(ValueError):
    """Raised when a data panel violates the canonical schema."""


def validate_panel(df: pd.DataFrame, *, allow_na_volume: bool = False) -> pd.DataFrame:
    """Validate and normalize a long-format OHLCV panel.

    Returns a sorted copy with canonical dtypes. Raises SchemaError on
    structural problems rather than silently coercing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    dupes = out.duplicated(subset=["date", "symbol"])
    if dupes.any():
        raise SchemaError(f"{int(dupes.sum())} duplicate (date, symbol) rows")

    if out["close"].isna().all():
        raise SchemaError("close is entirely NaN")
    if (out["close"].dropna() <= 0).any():
        raise SchemaError("non-positive close prices found")
    if not allow_na_volume and out["volume"].isna().all():
        raise SchemaError("volume is entirely NaN")

    bad_hl = (out["high"] < out["low"]).fillna(False)
    if bad_hl.any():
        raise SchemaError(f"{int(bad_hl.sum())} rows with high < low")

    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def to_wide(panel: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Pivot a long panel to a dates x symbols matrix."""
    return panel.pivot(index="date", columns="symbol", values=column).sort_index()
