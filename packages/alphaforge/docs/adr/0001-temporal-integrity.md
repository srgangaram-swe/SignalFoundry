# ADR 0001: Close decisions fill at a future open through a self-financing ledger

- Status: Accepted
- Date: 2026-07-10
- Owners: AlphaForge research platform

## Context

The original vectorized simulator shifted target weights by one row and then
multiplied them by the same row's close-to-close return. A target decided with
close(t) information could therefore receive the close(t)→open(t+1) gap even
though its stated fill occurred after that gap. Persisting target weights also
implied free rebalancing because holdings did not drift with relative returns.

Both behaviors can materially inflate or distort returns, turnover, costs, and
risk. Adding more models or a constrained optimizer before fixing this boundary
would compound a scientific-validity problem.

## Decision

- `target_weights.date` is a close-time decision timestamp.
- `execution_lag=1` schedules a DAY order at the next trading session open.
- Existing shares own the prior-close→open move; post-fill shares own the
  open→close move.
- The source of truth is a ledger of cash and signed shares, not a matrix of
  assumed held weights.
- Orders are sized against one pre-trade open NAV and current holdings.
- Commission is a cash debit; spread, slippage, and impact are embedded in the
  fill price.
- Liquidity and volatility inputs are shifted before the open. Participation
  caps create explicit partial fills and residual quantities.
- Every daily and symbol-level P&L path must reconcile or the run fails.
- Historical daily-bar execution and the uncalibrated C++ order-book benchmark
  remain separate models with separate claims.

## Consequences

Backtest P&L now respects ownership across overnight and intraday intervals.
Weights drift and repeated targets create observable turnover. Results can be
lower or simply different from the original implementation; this is expected
and is a correction, not a regression.

Daily-bar data still cannot identify queue position, intraday path, borrow
availability, auction dynamics, or actual market impact. Capacity outputs are
sensitivity analyses. More detailed execution claims require point-in-time
order-level data and calibration.

## Alternatives rejected

- **Shift returns another row while retaining weights.** Fixes the immediate
  gap leak but retains free rebalancing and opaque cash/cost accounting.
- **Assume close fills.** A close-time signal cannot generally trade at the
  same close without an earlier information cutoff or an auction model.
- **Use the synthetic C++ book for all fills.** It demonstrates data structures
  and parity, but without calibrated L2 history it cannot establish historical
  execution realism.
