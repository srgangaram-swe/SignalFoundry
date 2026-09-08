# Model Card

## Intended Use

Educational research into return prediction, walk-forward validation, and
realistic backtesting.

## Not Intended For

Live trading, financial advice, guaranteed return claims, or automated capital
allocation.

## Inputs

Causal technical, cross-sectional, benchmark-relative, and market-regime
features (including a causally-applied Gaussian HMM stress probability) built
from public or synthetic OHLCV bars.

## Outputs

Expected forward-return predictions by `(date, symbol)` for a configured
horizon; derived signals, target weights, and simulated (paper-only) orders.

## Evaluation

Models are evaluated on out-of-sample walk-forward windows using MSE, MAE, R2,
directional accuracy, Pearson IC, Spearman rank IC, IC decay, quantile return
analysis, and Newey-West IC t-statistics. Run-level results are stress-checked
with the Probabilistic/Deflated Sharpe Ratios and the Probability of Backtest
Overfitting (CSCV); purged K-Fold and CPCV splitters are available for
overlap-safe cross-validation.

## Risks

Data quality issues, leakage bugs, overfitting (mitigated but never
eliminated by the statistics above), transaction cost underestimation, and
market regime instability. See docs/limitations.md.
