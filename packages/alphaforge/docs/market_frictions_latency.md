# Market frictions and logical latency

AlphaForge models market frictions as bounded, deterministic research
sensitivities inside the event-driven backtest. The implementation makes the
accounting path, causal input, units, calibration provenance, and limitations
visible for every component. It is an offline daily-bar simulator, not a
broker, order-book reconstruction, execution-quality estimate, or trading
recommendation.

The architectural decision is recorded in
[ADR 0011](adr/0011-market-frictions-and-logical-latency.md). This guide
describes the public contracts and the evidence returned by `run_backtest`.

## Causal session sequence

One modeled trading session follows this logical order:

```text
open mark
  -> eligible DAY orders
  -> accepted/rejected/partial/full fills
  -> DAY residual cancellation
  -> financing and borrow cash charges
  -> close mark
  -> signal availability and target decision when scheduled
```

The latency model separately records how one target origin becomes available:

```text
origin <= data <= feature <= signal <= submission < fill
```

All stages are positions on the validated, frozen trading calendar. They are
not elapsed wall-clock durations. The schedule uses cumulative offsets:

```text
data       = d_data
feature    = d_data + d_feature
signal     = feature + d_inference
submission = signal + d_submission
fill       = submission + execution_lag + d_fill
```

`execution_lag` is a positive, irreducible future-open boundary. A delay can
never move a signal or fill earlier. The backtest emits `SignalAvailable` and
`TargetDecided` on the resolved signal session, then processes the DAY order
lifecycle at the eligible fill-session open. `submission_session` is an
audited readiness milestone in the daily-bar schedule; it is not an invented
intraday broker timestamp.

## Public configuration boundary

`run_backtest` accepts model objects or strict mappings through these optional
arguments:

| Argument | Contract | Default behavior |
| --- | --- | --- |
| `costs` | `CostModel` | Existing commission, spread, and slippage defaults |
| `execution` | `ExecutionPolicy` | Next-open daily-bar policy; no liquidity cap or impact |
| `latency` | `LatencyModel` | Zero added stage delay |
| `carry` | `CarryCostModel` | Zero financing and borrow rates; 252 sessions/year |
| `stress_profile` | `ExecutionStressProfile` | Canonical `baseline` |

Mappings reject unknown keys. Values must have the declared numeric types;
booleans are not accepted as integers or rates. All numbers must be finite and
inside their model bounds. Provenance strings are required, bounded, and
included in each declaration digest.

The input panel and target table retain the existing `run_backtest` data
contract. ADV and volatility are computed causally from prior observations and
shifted by one session before a fill may consume them. The friction calculators
perform no network, calendar-provider, storage, or broker I/O. A
caller-supplied durable event journal remains the separate, explicit storage
boundary described by ADR 0010.

## Fill-cost model

### Units and formulas

For absolute filled shares `Q`, reference price `P` in USD/share, reference
notional `N = Q * P` in USD, participation `p` as a ratio, and lagged return
volatility `sigma` as a decimal:

| Component | Formula | Accounting path |
| --- | --- | --- |
| Commission | `max(minimum, N * bps / 10_000 + Q * USD/share)` | `cash_fee` |
| Exchange fee | `N * bps / 10_000 + Q * USD/share` | `cash_fee` |
| Half-spread | `N * half_spread_bps / 10_000` | `fill_price` |
| Fixed slippage | `N * slippage_bps / 10_000` | `fill_price` |
| Spread slippage | `N * (half_spread_bps * multiplier) / 10_000` | `fill_price` |
| Participation slippage | `N * (coefficient * p ** exponent) / 10_000` | `fill_price` |
| Volatility slippage | `N * (coefficient * sigma * 100) / 10_000` | `fill_price` |
| Market impact | `N * impact_bps / 10_000` | `fill_price` |

The execution policy calculates:

```text
impact_bps =
    impact_coefficient * sigma * 10_000 * p ** impact_exponent
```

All six `fill_price` rates are summed before the fill price is calculated:

```text
buy_fill_price  = P * (1 + embedded_shortfall_bps / 10_000)
sell_fill_price = P * (1 - embedded_shortfall_bps / 10_000)
```

Commission and exchange fees are the only fill costs debited from cash as
`FillApplied.fees`. The fill-price components are already paid through the
worse execution price. `FillCostBreakdown.total_cost` combines both paths for
attribution and reconciliation only; debiting that total would double-count
the embedded components.

### Causal input requirements

- Lagged ADV is required when a participation cap, participation slippage, or
  impact coefficient is active.
- Lagged volatility is required when volatility slippage or impact is active.
- The reference open must be finite and positive.
- A participation cap is applied to absolute requested shares without rounding
  upward. The unfilled residual expires as a DAY cancellation.
- Missing required inputs produce stable rejection evidence rather than an
  assumption of infinite liquidity or zero volatility.
- The model rejects a non-positive or non-finite all-in fill price and verifies
  the embedded USD shortfall against the price difference with a scale-aware
  ULP tolerance.
- Malformed cost inputs, arithmetic overflow, and resource-ceiling failures
  propagate and halt result publication; they are never converted into a
  zero-cost rejected order.

`CostModel.rate` remains a compatibility approximation containing constant
basis-point terms only. It intentionally excludes per-share and minimum fees,
participation, volatility, and impact, all of which require fill context.

## Financing and short borrow

`CarryCostModel` evaluates the post-fill portfolio at that session's positive
open marks. It charges only negative cash and short positions:

```text
financing_charge =
    max(-cash_usd, 0) * cash_financing_bps_annual / 10_000 / sessions_per_year

borrow_charge(symbol) =
    abs(short_shares) * open_price_usd
      * short_borrow_bps_annual / 10_000 / sessions_per_year
```

The engine evaluates carry after the DAY order lifecycle is terminal and
before the close mark. A non-zero financing total and non-zero aggregate
borrow total are each applied exactly once as `CashChargeAccrued`. Attribution
retains financing under `__CASH__` and borrow by short symbol. Exact
`math.fsum` totals connect the per-symbol records to the ledger debit.

Carry is USD only. All supplied positions and prices are validated even when a
value would otherwise be unused, and every short requires a positive open
price. The record fixes `borrow_availability` and `locate_status` to
`not_modeled`. A configured annual borrow rate does not establish availability,
a locate, a recall policy, security-specific terms, or an executable short.

## Standard adverse stress grid

`standard_stress_profiles()` returns profiles in this fixed order:

| Profile | Perturbation |
| --- | --- |
| `baseline` | No perturbation |
| `doubled_costs` | Multiply fill costs, impact, and carry rates by 2 |
| `tripled_costs` | Multiply fill costs, impact, and carry rates by 3 |
| `adverse_spread` | Multiply half-spread by 3 |
| `reduced_liquidity` | Multiply lagged ADV by 0.5 |
| `delayed_signals` | Add one inference-delay session |
| `partial_fill_pressure` | Multiply the participation limit by 0.5 |
| `capacity_scaling` | Multiply initial capital and target demand by 5 |

The cost multiplier scales configured commission, exchange, slippage,
per-share, minimum, impact, financing, and borrow terms. The adverse-spread
case also affects any slippage derived from half-spread. If no baseline
participation limit exists, partial-fill pressure establishes a 50% limit.

Call `run_backtest` once for each profile. Every call rebuilds schedules,
orders, fills, charges, positions, and returns under that profile. Do not apply
these multipliers to an already completed return series. The grid is a
deterministic sensitivity surface, not a probability distribution, execution
forecast, or validated capacity estimate.

## Evidence tables

### `friction_model_manifest`

The manifest has one row for each of `fill_cost`, `execution_policy`,
`carry_cost`, `latency`, and `stress_profile`. It records:

- model identifier and version;
- declaration and configuration digests;
- canonical parameters, units, and parameter bounds;
- calibration provenance and domain;
- evaluation timestamp semantics and failure behavior;
- explicit limitations; and
- the active stress-profile name.

Changing parameters or provenance changes the corresponding identity. Digests
are content-integrity identifiers, not signatures or proof of market truth.

### `friction_attribution`

Each row records `date`, optional `order_id`, `symbol`, component, accounting
path, `amount_usd`, applicable rate and rate unit, `basis_usd`, model and record
digests, stress profile, status, and detail. The stable fill components are:

```text
commission
exchange_fee
spread
fixed_slippage
spread_slippage
participation_slippage
volatility_slippage
market_impact
```

Rejected execution produces an `execution_rejection` row with no cost.
Financing uses `cash_financing`; borrow uses `short_borrow`. The engine requires
the per-session sum of `amount_usd` to equal the equity curve's
`trading_cost`. Fill-component model identities point to the cost declaration,
except `market_impact`, which points to the execution-policy identity; rejected
fills use a composite execution-model identity. Every fill row hashes its full
fill context and normalized attribution content, while carry rows retain the
content digest of their exact `CarryAccrual` source record.

### `latency_schedule`

Each resolved target records the origin, data, feature, signal, submission, and
fill sessions; the irreducible execution lag; model and schedule digests; and
stress profile. A serialized schedule is accepted only if every stage belongs
to the same frozen calendar and preserves the declared order.

The legacy fill table remains version 1. New component detail lives in the
normalized evidence tables so existing reporting integrations are not
silently widened.

## Reconciliation and replay

The integration checks four independent accounting relationships:

```text
open_equity - post_fill_open_equity = fill_cost
open_equity - post_charge_open_equity = fill_cost + carry_cost
net_pnl = overnight_pnl + intraday_pnl - fill_cost - carry_cost
friction_attribution.groupby(date).amount_usd = equity_curve.trading_cost
```

The event journal contains the fills, cash fees, and categorized carry charges
that actually mutated the ledger. Its verified replay reconstructs recorded
portfolio state and proves event ordering and idempotency. The manifest and
schedule tables explain which assumptions produced those events. Replay does
not infer omitted market state, recalibrate a parameter, or prove that a
simulated fill was obtainable.

On reconciliation failure, invalid arithmetic, malformed configuration,
missing causal data, a schedule outside the calendar, or an uncertain durable
journal append, the run fails closed and must not publish evidence. Recovery
uses the journal's verified committed prefix; it does not retry from uncertain
in-memory state.

## Bounds and complexity

Important resource ceilings include:

| Resource | Bound |
| --- | ---: |
| Frozen calendar | 1,000,000 sessions |
| One latency stage | 2,520 sessions |
| Resolved latency | 5,040 sessions |
| Annual carry rate | 1,000,000 bps |
| Sessions per year | 366 |
| Modeled money | USD `1e18` |
| Absolute position | `1e15` shares |
| Carry mark | USD `1e12` per share |
| Carry positions or prices | 10,000 symbols per mapping |
| Fill cost rate | 1,000,000 bps |

Schedule generation is linear in target decisions, subject to the bounded
calendar. Fill attribution is linear in modeled fill components; carry
attribution is linear in open short positions. The standard grid intentionally
costs one complete simulation per profile. This is a research-evidence choice,
not a low-latency gateway design.

`make bench-execution-frictions` runs the bounded, deterministic, CPU-only
synthetic benchmark outside the event engine. Its JSON retains every raw wall
and process-CPU timing sample, distribution summaries, the synthetic fixture
and semantic hashes, peak Python allocation from a separate untimed pass, and
SHA-256 identities for the benchmark source, cost/execution implementation,
and `uv.lock`. Timings are descriptive local overhead measurements with no
pass/fail threshold; they are not broker latency, market calibration, capacity,
or strategy evidence.

## Reproducibility and data handling

- Store exact input-data, target, run-configuration, model, stress-profile,
  latency-schedule, and accrual identities with run evidence.
- Preserve the frozen calendar and declared calibration provenance.
- Keep licensed prices, raw data, and durable journals in ignored,
  owner-controlled storage. Commit only synthetic or redistribution-safe
  evidence.
- Never place provider credentials, broker secrets, or private data in a
  configuration, manifest, event, log, plot, issue, or pull request.
- Label all results as synthetic, simulated, backtested, paper-traded, or live
  according to the evidence actually produced.

The current committed validation boundary is deterministic and offline. It
does not model exchange or broker latency, intraday paths, queue position,
auction behavior, venue routing, taxes, rebates, borrow availability, locates,
recalls, restrictions, forced buy-ins, FX, or live controls. It makes no claim
of strategy edge, profit, production readiness, or permission to trade.

## Rollback and compatibility

The compatibility baseline uses zero added latency, zero carry rates, the
baseline stress profile, and the existing fill-cost defaults. A rollback may
remove additive MR4 orchestration and evidence publication, but it must never:

- rewrite an existing journal, manifest, schedule, or accrual;
- reinterpret fill-price shortfall as a cash fee or silently discard a charge;
- accept an unsupported digest or schema as if it were current; or
- describe an old run as having used a different configuration.

Unsupported historical evidence fails closed. The version-1 fill, order,
equity, accounting, and event views remain the compatibility boundary.
