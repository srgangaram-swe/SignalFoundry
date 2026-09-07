# Data Design

Supported sources:

- Synthetic regime-switching factor data for tests and offline demos.
- User CSV files with one file per symbol.
- yfinance downloads with local caching when optional data dependencies are installed.

Data quality checks include duplicate `(date, symbol)` rows, non-positive prices, high/low consistency, missing sessions, NaN closes, zero volume, and extreme daily returns.

Market data artifacts and run outputs are ignored by git. Do not commit restricted, proprietary, or sensitive data.
