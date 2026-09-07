# Ranking portfolios and uncertainty-aware sizing (SF-S4-MR1)

Six allocation policies over a cross-section of predicted scores, two orthogonal
sizing layers, an explicit constraint set with a true feasible projection, and a
net-of-cost comparison across folds, regimes, and capacity levels.

> **No strategy claim.** Every number here comes from a synthetic panel with a
> deliberately strong planted signal, used to exercise the machinery. The Sharpe
> ratios below are not plausible market results and must not be read as any.

---

## 1. The constraint set, and why the projection matters

Most portfolio bugs are not in the optimizer. They are in the step afterwards.
The common pattern — clip to a position cap, then renormalize to a gross target —
is **not a projection**: renormalizing re-inflates exactly the names the cap just
pulled down, so the result satisfies neither limit while still looking like a set
of weights.

`PortfolioConstraints` states every limit in one place:

| limit | meaning |
|---|---|
| `max_position` | largest absolute weight in any one name |
| `max_gross` | ceiling on Σ\|w\| |
| `max_net` | ceiling on \|Σw\| |
| `max_turnover` | ceiling on Σ\|w − w_prev\| per rebalance |
| `max_leverage` | gross per unit equity — a solvency limit, deliberately separate from the `max_gross` policy target |
| `cash_buffer` | capital deliberately uninvested; gross is measured against `1 − cash_buffer` |
| `long_only` | forbid negative weights |

`project_to_feasible` alternates projections onto each convex set until all hold
simultaneously, then **asserts its post-conditions and raises** rather than
returning an infeasible book. Three behaviours are worth calling out:

**Feasibility is checked before iterating.** `max_position × n_assets` below the
gross target, liquidity caps that cannot fund the book, or a long-only mandate
with `max_net` below gross are all refused up front with an actionable message.
On an empty intersection the iteration would otherwise wander and return
something plausible.

**Freed capital is redistributed.** Clipping to a position cap releases capital;
without waterfilling the book silently under-deploys — satisfying every limit
while holding unintended cash, which looks conservative and is an accounting bug.

**An unavoidable shortfall is reported, not hidden.** Five names at a 0.15 cap
cannot reach gross 1.0. The result carries `deployed_fraction`, `gross_shortfall`
and a `shortfall_reason` naming the arithmetic.

**A turnover budget can conflict with the exposure limits.** If the previous book
is further outside the gross/net limits than the trade budget can correct, the
set is empty; that is now detected exactly and reported as such rather than
surfacing as a confusing non-convergence.

Cash is explicit throughout — the issue's non-goal is hidden cash assumptions, so
`AllocationResult` carries `cash_weight` rather than leaving it implied by
whatever the weights failed to sum to.

---

## 2. The six policies

They differ in **how much of the score they trust**, which is the axis that
matters when scores are noisy:

| policy | trusts | notes |
|---|---|---|
| `top_k` | ordering, at the top only | `equal` / `score` / `rank` weighting within the sleeve |
| `long_short_spread` | ordering, at both tails | each sleeve sized to half gross, so the raw book is dollar-neutral by construction |
| `quantile` | ordering, proportionally | sleeve size tracks the universe, so it does not concentrate when the universe shrinks |
| `score_weighted` | magnitudes | correct when the score is a calibrated expected return, wrong when it is arbitrary strength |
| `rank_weighted` | full ordering, not magnitudes | one extreme score cannot dominate |
| `inverse_volatility` | direction only | volatility is far more estimable than expected return |

Two invariants hold for all of them:

**Determinism under ties.** Ranking breaks ties on the symbol name, so an
equal-score pair produces the same book every run. Without it the backtest is
intermittently irreproducible — the worst failure mode, because it passes most of
the time.

**Nothing is returned unprojected.** Every policy ends in the projection, so a
returned book has satisfied every limit or the call raised.

Two deliberate refusals, both cases where a plausible default is the dangerous
choice:

* A name with zero, missing, or non-finite **volatility is excluded**, not given
  the median. An unmeasurable risk is not a small risk.
* A name with zero or missing **ADV gets a zero liquidity cap** — untradeable
  rather than unconstrained. Defaulting an unknown ADV to "no limit" is how an
  illiquid name acquires a full-size position.

---

## 3. Sizing layers

**Uncertainty sizing** tilts positions by `(median uncertainty / uncertainty) ^
strength`, normalized to preserve gross. It therefore changes *concentration*,
not leverage. The tilt is **capped** at `MAX_CONFIDENCE_MULTIPLE`, because an
apparently very-low-uncertainty name is usually an estimation artefact and
uncapped confidence sizing converts that artefact into concentration. Fractional
and bounded by construction — explicitly **not full Kelly**, which the issue
names as a non-goal and which is a route to ruin on estimated moments.

**Volatility targeting** scales the book so ex-ante `sqrt(w'Σw)` annualized meets
a target, **capped by `max_leverage`**. That cap is the entire reason it is safe
to run: in a quiet regime the unconstrained factor grows without bound, and a
volatility target without a leverage ceiling is precisely the mechanism that
turns a calm market into a catastrophic one when the estimate is stale. The
diagnostics report ex-ante volatility before and after, the raw and applied
factors, whether leverage bound, and whether the target was actually met — so a
book that silently failed to reach its target is visible.

---

## 4. Measured comparison

Synthetic panel: 25 names, 320 days, planted score/return correlation, ADV
$200M, 10bps on turnover, `max_position` 0.15, `max_gross` 1.0, `max_net` 0.15,
`max_turnover` 0.5, 3 chronological folds.

**Net Sharpe by capital — the capacity gradient is the point:**

| policy | $100M | $200M | $240M | $400M |
|---|---|---|---|---|
| score_weighted | 16.31 | 13.87 | 12.14 | infeasible |
| rank_weighted | 16.10 | 14.03 | 12.09 | infeasible |
| long_short_spread | 14.56 | 17.20 | **17.68** | infeasible |
| top_k_equal / top_k_rank | 13.15 | 13.15 | 13.15 | infeasible |
| inverse_volatility | 11.99 | 12.27 | 12.39 | infeasible |

**The ranking inverts with capital.** `score_weighted` is best at $100M and
fourth by $240M; `long_short_spread` is third at $100M and best by $240M. Reading
only the smallest capital level would have selected the policy that degrades
fastest. At $400M every policy is infeasible — liquidity caps can no longer fund
the gross target — and that is recorded as `feasible_fraction = 0`, not dropped
from the sample.

**Costs at $100M, averaged over folds:**

| policy | gross | net | cost drag | turnover | max drawdown |
|---|---|---|---|---|---|
| score_weighted | 0.824 | 0.698 | 0.126 | 0.500 | −0.0034 |
| long_short_spread | 0.802 | 0.676 | 0.126 | 0.500 | −0.0044 |
| rank_weighted | 0.782 | 0.656 | 0.126 | 0.500 | −0.0033 |
| top_k_equal | 0.344 | 0.282 | 0.061 | 0.244 | −0.0026 |

The dense policies run at the turnover cap and pay double the cost drag of
`top_k`. Comparing gross would have flattered exactly the policies that pay most
to execute.

**By regime**, `inverse_volatility` is the only policy that scores materially
*better* in stress (11.99 → 15.21) — consistent with sizing by risk rather than
by signal magnitude when dispersion rises.

---

## 5. Holdout discipline

`compare_allocation_policies` runs on development folds only. The final holdout
is not one of its arguments; it is scored once, afterwards, by
`score_final_holdout` on an already-frozen policy. **A function that cannot see
the holdout cannot select on it** — the discipline is structural, not a
convention someone has to remember.

Comparison rows sort by `(policy, capital, fold, regime)`, never by performance,
because ranking a comparison table by its own metric invites reading the top row
as a decision. The rejection rule belongs to the frozen qualification decision
(SF-S4-MR9).

---

## 6. Evidence

`tests/test_ranking_portfolios.py` — 62 tests. The central assertion is that
every returned book satisfies every declared limit, checked for all six policies
under both long-only and long-short mandates. Named by the acceptance criteria:
ties, shuffled input order, missing and non-finite scores, duplicate symbols,
zero and missing volatility, changing universes charging exit turnover, extreme
scores, flat cross-sections, turnover caps binding, zero-turnover freeze,
infeasible position caps, infeasible liquidity, illiquid exclusion, out-of-range
`k` and quantile, uncertainty-tilt cap and no-op, volatility-target leverage
bound and partial covariance, fold ordering, regime causality under future
mutation, cost accounting, capacity exhaustion recorded rather than dropped, and
holdout separation.

Coverage: `allocation.py` 92%, `evidence.py` 90%, `contracts.py` 84% (branch).
Repository total 84.55% against an unchanged 78% floor.

---

## 7. Residual limitations

* **Synthetic evidence only**, with a planted signal. The Sharpe levels are an
  artefact of that construction and carry no market claim.
* **The ranking objective uses a linear turnover penalty.** Any selected target
  must still pass through the separate event-driven friction model for
  component-level slippage, impact, carry, and logical latency. The objective
  penalty is not a substitute for that full causal rerun and can still favour a
  dense, high-turnover book.
* **Capacity is modelled through participation caps only.** No borrow
  availability, no short fees, no crowding.
* **`apply_volatility_target` needs a covariance the caller supplies.** A
  shrinkage/factor estimator is issue #7; a sample covariance on a short window
  will understate risk.
* **The turnover feasibility check is a necessary condition, not a sufficient
  one.** It catches gross/net conflicts exactly; a pathological combination of
  per-name liquidity caps and a tiny budget could still fail to converge, which
  raises rather than returning an infeasible book.
* **No optimizer.** These are rule-based allocations; constrained Markowitz is
  SF-S4-MR2.
* **Simulation only**, per the issue's security scope — no broker, no live
  endpoint, no capital at risk.
