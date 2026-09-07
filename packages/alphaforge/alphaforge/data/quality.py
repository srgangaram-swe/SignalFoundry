"""Data quality report: per-symbol coverage, gaps, outliers, suspect rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def data_quality_report(panel: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    """Compute a per-symbol data quality table; optionally write markdown.

    Flags: missing sessions vs. the union calendar, NaN closes, zero/NaN
    volume, and extreme single-day returns (|r| > 40%) that often indicate
    unadjusted splits or bad prints.
    """
    calendar = pd.Index(sorted(panel["date"].unique()))
    rows = []
    for symbol, g in panel.groupby("symbol"):
        g = g.sort_values("date")
        rets = g["close"].pct_change()
        first, last = g["date"].min(), g["date"].max()
        expected = calendar[(calendar >= first) & (calendar <= last)]
        rows.append(
            {
                "symbol": symbol,
                "start": first.date(),
                "end": last.date(),
                "rows": len(g),
                "missing_sessions": len(expected) - len(g),
                "nan_close": int(g["close"].isna().sum()),
                "zero_volume_days": int((g["volume"].fillna(0) == 0).sum()),
                "extreme_returns": int((rets.abs() > 0.40).sum()),
                "max_abs_return": float(rets.abs().max()) if len(g) > 1 else np.nan,
            }
        )
    report = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Data Quality Report",
            "",
            f"- Symbols: **{report.shape[0]}**",
            f"- Calendar: **{calendar.min().date()} → {calendar.max().date()}** "
            f"({len(calendar)} sessions)",
            f"- Symbols with missing sessions: **{int((report['missing_sessions'] > 0).sum())}**",
            f"- Symbols with extreme (>40%) daily moves: "
            f"**{int((report['extreme_returns'] > 0).sum())}**",
            "",
            "```",
            report.to_string(index=False),
            "```",
            "",
        ]
        output_path.write_text("\n".join(lines))
    return report


def fill_missing(panel: pd.DataFrame, max_ffill: int = 5) -> pd.DataFrame:
    """Forward-fill small gaps per symbol (prices only, never volume).

    Forward-filling uses only past values, so it cannot introduce lookahead.
    Gaps longer than ``max_ffill`` sessions are left as NaN and dropped by
    downstream feature construction.
    """

    def _fill(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        for col in ["open", "high", "low", "close"]:
            g[col] = g[col].ffill(limit=max_ffill)
        g["volume"] = g["volume"].fillna(0)
        return g

    return (
        panel.groupby("symbol", group_keys=False)[panel.columns].apply(_fill).reset_index(drop=True)
    )
