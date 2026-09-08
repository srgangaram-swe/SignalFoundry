# LinkedIn Post

I built AlphaForge, a quantitative ML research platform focused on the engineering details that make trading research credible — and then I rebuilt the parts that most projects hand-wave.

The hard part in quant ML is not fitting a model. It is that every default in the ML toolbox silently cheats on time-series data: random splits leak, overlapping labels leak, full-sample scalers leak, smoothed regime models leak, and "we tried 20 models and report the best one" is a leak of its own kind. AlphaForge treats each of those as an engineering requirement with a test.

What's inside:

- Leakage-safe pipeline: canonical OHLCV panel, causal features, multi-horizon forward labels, walk-forward validation with an embargo at least as long as the label horizon. CI literally mutates future bars and asserts past features are bit-identical.
- Validation science: Purged K-Fold and Combinatorial Purged CV, Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV), and Newey-West IC t-stats — computed automatically for every run, so the report tells you when the "winner" is probably selection bias.
- Regime awareness that doesn't cheat: a from-scratch 2-state Gaussian HMM fit by EM, used with expanding refits and *filtered* (not smoothed) probabilities, feeding regime-gated exposure and regime-conditional attribution.
- Honest execution: out-of-sample predictions only; close-time decisions fill at a future open through a self-financing cash/share ledger. Holdings drift, lagged ADV can create partial DAY fills, and commission/spread/impact plus symbol P&L reconcile exactly—including risk-overlay rebalances.
- A C++17 limit order book with pybind11 bindings: price-time priority, O(1) cancels, ~6M ops/s at ~125 ns median latency, parity-tested against a pure-Python reference. The daily-bar pipeline does not need nanoseconds, so I keep this uncalibrated systems benchmark separate from historical execution claims.

Everything is reproducible offline: a synthetic regime-switching market generator with a deliberately faint embedded edge lets the whole stack run end-to-end in CI with no market data and no network.

It's an educational research platform, not a money machine — the README says so in bold. But it's the difference between "I trained a model on stock prices" and "I can explain why most backtests are wrong and show you the tests that catch it."

Code: github.com/srgangaram-swe/AlphaForge
