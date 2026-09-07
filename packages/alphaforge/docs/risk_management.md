# Risk Management

## Position-level controls

- Maximum per-asset weight; maximum gross and net exposure.
- Inverse-volatility sizing; optional turnover caps.
- Regime filter: exposure scaled by (1 − HMM stress probability) and cut to
  zero above a stress threshold — driven by a causal feature, so no lookahead.

## Portfolio-level controls (costed in the backtest)

- Causal volatility targeting from trailing realized strategy returns.
- Drawdown deleveraging from the previous day's trailing drawdown.

## Analytics

- Drawdown, volatility, Sharpe, Sortino, Calmar, hit rate, holding period.
- VaR and expected shortfall (historical, 95%).
- Beta / alpha vs. benchmark; concentration (HHI, effective positions).
- **Regime-conditional performance**: returns split by calm/stress regime —
  a strategy that only works in one regime is a different proposition.
- **Beta-aware stress tests**: market shocks propagated through per-name
  rolling betas, reporting portfolio beta and first-order scenario PnL.

Risk metrics are diagnostics. They do not certify that a strategy is safe or
profitable.
