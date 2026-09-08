"""Publish deterministic, obviously synthetic bars through the real producer.

Executed only by the Signalattice interpreter in integration tests. The fixture
does not acquire network data or use any provider credential.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from quant_platform.data.signal_foundry_contract import export_signal_foundry_bundle


def main() -> None:
    dates = pd.bdate_range("2020-01-02", periods=500)
    rng = np.random.default_rng(42)
    constant_volume = "--constant-volume" in sys.argv[2:]
    frames = []
    for symbol in ("BENCH", "AAA", "BBB", "CCC", "DDD"):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, len(dates))))
        effective = dates.tz_localize("UTC") + pd.Timedelta(hours=21)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": symbol,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": (
                        1_000_000.0
                        if constant_volume
                        else rng.integers(800_000, 1_200_000, len(dates)).astype(float)
                    ),
                    "effective_at": effective,
                    "available_at": effective + pd.Timedelta(hours=8),
                    "observed_at": pd.Timestamp("2022-01-01T00:00:00Z"),
                    "provider_updated_at": pd.Timestamp("2022-01-01T00:00:00Z"),
                    "instrument_id": symbol,
                    "currency": "USD",
                    "exchange_calendar": "XNYS",
                    "adjustment_state": "provider_adjusted_close_unadjusted_ohlc",
                    "source": "synthetic",
                    "source_table": "SYNTHETIC/TEST",
                }
            )
        )
    manifest = {
        "provider": "synthetic",
        "request": {"table": "SYNTHETIC/TEST"},
        "request_hash": "a" * 64,
        "snapshot_hash": "b" * 64,
        "retrieved_at": "2022-01-01T00:00:00Z",
        "contains_api_key": False,
        "observations_redistributable": True,
        "point_in_time_limits": {
            "historical_revisions_complete": False,
            "universe_membership_point_in_time": False,
            "corporate_actions_complete": False,
        },
    }
    result = export_signal_foundry_bundle(
        pd.concat(frames, ignore_index=True),
        sys.argv[1],
        source_manifest=manifest,
        producer_git_sha="c" * 40,
    )
    print(result.name)


if __name__ == "__main__":
    main()
