"""Data loaders: yfinance download with local cache, CSV directory, synthetic.

Design rules:
- Never depend on paid APIs. yfinance is optional; CSVs and the synthetic
  generator always work offline.
- Cache raw downloads to parquet so research is reproducible and polite to
  the data source.
- Everything returns the canonical long-format panel (see schemas.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.schemas import validate_panel
from alphaforge.data.synthetic import SyntheticMarketConfig, generate_synthetic_market
from alphaforge.utils import get_logger

logger = get_logger(__name__)


def _cache_path(cache_dir: Path, symbol: str, start: str, end: str, interval: str) -> Path:
    return cache_dir / f"yf_{symbol}_{start}_{end}_{interval}.parquet"


def download_yfinance(
    symbols: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Download adjusted OHLCV bars via yfinance, with a local parquet cache.

    Uses ``auto_adjust=True`` so close is split/dividend-adjusted and OHLC are
    rescaled consistently.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is not installed. `pip install 'alphaforge[data]'` or use "
            "source: csv / synthetic in configs/data.yaml."
        ) from e

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    frames = []
    for symbol in symbols:
        path = _cache_path(cache_dir, symbol, start, end, interval)
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        logger.info("downloading %s (%s → %s, %s)", symbol, start, end, interval)
        raw = yf.download(
            symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False
        )
        if raw is None or raw.empty:
            logger.warning("no data returned for %s — skipping", symbol)
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = (
            raw.reset_index()
            .rename(
                columns={
                    "Date": "date",
                    "Datetime": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )[["date", "open", "high", "low", "close", "volume"]]
            .assign(symbol=symbol)
        )
        df.to_parquet(path, index=False)
        frames.append(df)

    if not frames:
        raise RuntimeError("no data downloaded for any requested symbol")
    return validate_panel(pd.concat(frames, ignore_index=True))


def load_csv_dir(csv_dir: str | Path, symbols: list[str] | None = None) -> pd.DataFrame:
    """Load user-provided per-symbol CSVs (``SYMBOL.csv``) from a directory.

    Expected columns (case-insensitive): date, open, high, low, close, volume.
    An ``adj_close`` / ``adj close`` column, if present, replaces close and
    rescales OHLC by the adjustment factor.
    """
    csv_dir = Path(csv_dir)
    paths = sorted(csv_dir.glob("*.csv"))
    if symbols:
        wanted = {s.upper() for s in symbols}
        paths = [p for p in paths if p.stem.upper() in wanted]
    if not paths:
        raise FileNotFoundError(f"no CSV files found in {csv_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "adj_close" in df.columns:
            factor = df["adj_close"] / df["close"]
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * factor
        df["symbol"] = path.stem.upper()
        frames.append(df[["date", "symbol", "open", "high", "low", "close", "volume"]])
    return validate_panel(pd.concat(frames, ignore_index=True))


def load_prices(config: dict) -> tuple[pd.DataFrame, str]:
    """Load the full universe + benchmark per a data.yaml-style config dict.

    Returns (panel, benchmark_symbol). The benchmark rows are part of the
    panel and are excluded from the tradable universe downstream.
    """
    source = config.get("source", "synthetic")
    benchmark = config.get("benchmark", "BENCH")

    if source == "synthetic":
        syn = config.get("synthetic", {})
        cfg = SyntheticMarketConfig(
            n_symbols=syn.get("n_symbols", 20),
            n_days=syn.get("n_days", 2500),
            seed=syn.get("seed", 42),
        )
        return generate_synthetic_market(cfg), cfg.benchmark_symbol

    if source == "csv":
        symbols = list(config.get("symbols", [])) + [benchmark]
        return load_csv_dir(config.get("csv_dir", "data/csv"), symbols), benchmark

    if source == "signal_foundry":
        raise ValueError(
            "signal_foundry data requires the governed dual-panel workflow; "
            "use scripts/run_signal_foundry_research.py"
        )

    if source == "yfinance":
        symbols = list(dict.fromkeys(list(config.get("symbols", [])) + [benchmark]))
        panel = download_yfinance(
            symbols,
            start=config.get("start", "2010-01-01"),
            end=config.get("end"),
            interval=config.get("interval", "1d"),
            cache_dir=config.get("cache_dir", "data/cache"),
        )
        return panel, benchmark

    raise ValueError(f"unknown data source: {source!r}")
