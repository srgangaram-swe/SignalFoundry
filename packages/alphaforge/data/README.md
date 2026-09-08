# data/

Local data storage. **Nothing in this directory is committed to git** except this README.

| Path | Contents |
|---|---|
| `raw/` | As-downloaded vendor data (one parquet per symbol) |
| `cache/` | Download cache keyed by (source, symbol, range, interval) |
| `csv/` | User-provided CSVs (`SYMBOL.csv` with date, open, high, low, close, volume) |
| `processed/` | Feature/label panels produced by `make build-features` |
| `signal-foundry/` | Licensed, immutable Signalattice bundles; local only |

Never commit licensed or restricted observations. Signal Foundry bundles are
independently hash-verified by AlphaForge before use and retain their producer
license and point-in-time limitations. See
[`docs/signal_foundry.md`](../docs/signal_foundry.md) for the governed workflow.
