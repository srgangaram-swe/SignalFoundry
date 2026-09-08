# Portfolio Summary

## Suggested GitHub description

`Regime-aware quant ML research platform: leakage-safe walk-forward + purged CV, deflated Sharpe / PBO, causal HMM regimes, a self-financing future-open ledger, capacity sensitivities, and a separately scoped C++17 order-book core.`

## Suggested topics

`quantitative-finance`, `machine-learning`, `backtesting`, `time-series`,
`alpha-research`, `portfolio-construction`, `risk-management`, `python`,
`cpp`, `pybind11`, `order-book`, `scikit-learn`, `hidden-markov-model`

## Portfolio website blurb

AlphaForge is a regime-aware quantitative ML research platform built to
demonstrate the engineering discipline behind credible trading research.
Every default in the ML toolbox silently cheats on financial time series —
random splits leak, overlapping labels leak, full-sample scalers leak, and
"best of N models" is selection bias. AlphaForge treats each failure mode as
an engineering requirement with a test: CI mutates future market data and
asserts past features are bit-identical; walk-forward and purged/combinatorial
CV splitters enforce embargoes; every run reports deflated Sharpe ratios and
the probability of backtest overfitting; and the backtest costs the turnover
of its own risk controls. A from-scratch Gaussian HMM provides causal regime
awareness. A frozen-calendar, hash-journaled event reducer makes target,
order, rejection, partial-fill, cancellation, and restart state explicit. Its
chronological cost-basis ledger prevents pre-fill gap capture, tracks drift and
lagged-ADV partial fills, and reconciles realized/unrealized P&L and categorized
costs. A
separately scoped C++17 limit-order-book core (~6M ops/s, ~125 ns median
latency, parity-tested against a Python reference) demonstrates systems work
without masquerading as calibrated historical microstructure. Reproducible offline via a synthetic regime-switching
market generator; shipped with tests, CI, Docker, a Streamlit dashboard, and
a FastAPI service.

## One-liner

Research platform that shows *why most backtests are wrong* — and ships the
tests, statistics, and infrastructure that catch it.
