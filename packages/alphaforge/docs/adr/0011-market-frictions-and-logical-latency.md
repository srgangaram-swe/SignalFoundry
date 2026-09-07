# ADR 0011: Market frictions, logical latency, and adverse execution stress

- Status: Accepted
- Date: 2026-08-01
- Owners: AlphaForge research platform
- Builds on: ADR 0001 and ADR 0010

## Context

ADR 0001 established a causal daily-bar timeline and future-open execution.
ADR 0010 added a deterministic event journal and reconciled portfolio ledger,
but deliberately accepted supplied fill fees and cash charges rather than
calculating them. A useful research backtest also needs explicit assumptions
for commission, spread, slippage, impact, financing, short borrow, and the
logical delay between data availability and a possible fill.

These quantities are not observable from daily OHLCV alone. Treating a proxy
as a broker quote, reconstructing an intraday queue from a bar, or scaling a
completed return series would manufacture evidence. The platform instead needs
bounded sensitivity models whose units, timing, provenance, limitations, and
accounting path remain auditable through a complete event-simulation rerun.

This decision governs offline research simulation. It does not establish
borrow availability, a locate, an executable price, broker connectivity,
paper- or live-trading readiness, strategy viability, or expected profit.

## Decision

### Immutable model contracts and identities

Fill cost, execution policy, carry cost, latency, and execution-stress models
publish immutable `ModelDeclaration` records. Each declaration names a model
and version and records units, parameter bounds, calibration provenance,
domain, evaluation time, fail-closed behavior, and explicit limitations.
Canonical, domain-separated SHA-256 digests identify declarations,
configurations, resolved latency schedules, carry accruals, and stress
profiles. Exact floating-point configuration values use canonical encodings.

Constructors reject unknown keys, boolean-as-number coercion, non-finite
values, invalid types, out-of-range values, non-canonical identifiers,
unsorted or duplicate metadata, overflow, and resource-unbounded inputs. A
configuration is never silently repaired, clamped, or moved to an earlier
session. Provenance is part of the declaration identity; changing an
assumption's source changes its digest.

`run_backtest` binds the resolved models and stress profile into the run
configuration identity. It returns a five-row friction model manifest with
the declarations, exact parameters, declaration and configuration digests,
provenance, bounds, timing, failure behavior, and limitations. The manifest is
interpretive evidence for the run; it is not a claim that the assumptions
were market-calibrated.

### Fill-cost accounting paths

For filled absolute shares `Q`, positive reference price `P`, reference
notional `N = Q * P`, lagged participation `p`, and lagged decimal volatility
`sigma`, the cost model calculates:

```text
commission = max(
    minimum_commission_usd,
    N * commission_bps / 10_000
      + Q * commission_per_share_usd,
)

exchange_fee =
    N * exchange_fee_bps / 10_000
      + Q * exchange_fee_per_share_usd

spread_slippage_bps = half_spread_bps * spread_slippage_multiplier
participation_slippage_bps =
    participation_slippage_bps_parameter * p ** participation_slippage_exponent
volatility_slippage_bps =
    volatility_slippage_bps_per_1pct * (sigma * 100)
```

The execution policy supplies market-impact sensitivity:

```text
impact_bps =
    impact_coefficient * sigma * 10_000 * p ** impact_exponent
```

Half-spread, fixed slippage, spread-dependent slippage,
participation-dependent slippage, volatility-dependent slippage, and impact
are embedded once in the fill price:

```text
embedded_shortfall_bps = sum(all six price components)
fill_price = P * (1 + side_sign * embedded_shortfall_bps / 10_000)
```

Commission and exchange fees take the separate `cash_fee` path and are debited
only through `FillApplied`. Price-embedded shortfall is never debited again.
`total_cost` is the sum of both paths for reporting and reconciliation, not a
third ledger charge. The engine verifies that the fill-price difference and
the component-level embedded USD cost agree within a scale-aware ULP bound.

ADV and volatility are lagged before the fill. If any configured
participation, impact, or volatility term requires a causal input and that
input is unavailable, the fill is rejected with a stable reason. A
participation limit is a hard quantity cap; any residual is a DAY residual and
is cancelled rather than carried forward implicitly. Invalid prices,
and non-positive all-in sell prices reject the order. Malformed cost inputs,
arithmetic overflow, and resource-ceiling failures propagate and stop result
publication rather than becoming a zero-cost market rejection.

### Logical-session latency

Latency is a sequence of readiness stages on the caller-supplied, frozen,
strictly increasing trading calendar:

```text
origin <= data <= feature <= signal <= submission < fill
```

For stage delays `d_data`, `d_feature`, `d_inference`, `d_submission`, and
`d_fill`, plus the positive legacy execution lag `L`, cumulative offsets from
the origin are:

```text
data       = d_data
feature    = data + d_feature
signal     = feature + d_inference
submission = signal + d_submission
fill       = submission + L + d_fill
```

`L` remains the irreducible close-decision/future-open boundary. A modeled
delay can only preserve or postpone a stage. The engine resolves every
baseline-eligible target before the event run, emits the signal and target
decision on the resolved signal session, and makes the target executable only
on the resolved fill session. The intermediate submission session is an
audited readiness milestone; the daily-bar order lifecycle is processed at
the eligible fill-session open.

A schedule fails if its origin is absent, its calendar is malformed, its fill
is not strictly after submission, or any stage would extend beyond the frozen
calendar. No exchange, network, broker, auction, queue, wall-clock, or
intraday timestamp is inferred.

### Financing and short-borrow sensitivity

Carry is a USD-only per-session sensitivity evaluated after open fills and
DAY terminal cancellation and before the close mark. For post-fill cash `C`,
short shares `q_s < 0`, positive open price `P_s`, annual financing rate
`r_f`, annual borrow rate `r_b`, and declared sessions per year `Y`:

```text
financing_basis = max(-C, 0)
financing_charge = financing_basis * r_f / 10_000 / Y
short_market_value_s = abs(q_s) * P_s
borrow_charge_s = short_market_value_s * r_b / 10_000 / Y
```

Only non-zero financing and aggregate borrow amounts become
`CashChargeAccrued` events. Borrow attribution remains per symbol, while the
ledger event debits the exact aggregate. `CarryAccrual` requires exact
`math.fsum` totals and records configuration identity and provenance.

All supplied positions and prices are validated, including extra values. A
positive USD price is required for every short symbol. The model never invents
a mark or rate and explicitly records borrow availability and locate status as
`not_modeled`. It does not model recalls, security-specific rates,
restrictions, forced buy-ins, taxes, or FX conversion.

### Adverse stress grid and complete reruns

The canonical grid has deterministic order and reserved settings:

1. baseline;
2. doubled costs;
3. tripled costs;
4. three-times spread;
5. half available liquidity;
6. one additional signal-delay session;
7. half participation limit; and
8. five-times capital demand.

Stress profiles are structurally adverse: cost, spread, and capacity
multipliers cannot be below one; liquidity and participation multipliers
cannot exceed one; signal delay cannot be negative. Reserved names cannot be
redefined with different values.

Each profile requires a complete `run_backtest` invocation. The profile
creates immutable stressed model variants before the event simulation: costs
and carry rates change, spread may widen, lagged ADV may fall, participation
may tighten, signal latency may increase, or initial capital may rise. Orders,
partial fills, costs, positions, and returns are then recomputed causally. A
completed return series is never multiplied, shifted, or post-scaled to
manufacture a stress result. These cases are sensitivities, not forecasts of
realized costs, fill probability, or strategy capacity.

### Evidence, replay, and reconciliation

`BacktestResult` preserves the version-1 fill table and adds three normalized
evidence tables:

- `friction_model_manifest` identifies the five resolved model contracts;
- `friction_attribution` records every fill component, rejection, financing
  charge, and per-symbol borrow charge with accounting path, USD amount,
  applicable rate and basis, model identity, stress profile, and status; and
- `latency_schedule` records every stage, model digest, and resolved schedule
  digest.

The event journal remains the authority for applied fills and cash charges.
Verified replay reconstructs those recorded ledger transitions; it does not
recalibrate a model or assert that a proxy assumption was executable. At every
session the engine proves:

```text
open_equity - post_fill_open_equity = fill_cost
open_equity - post_charge_open_equity = fill_cost + carry_cost
net_pnl = overnight_pnl + intraday_pnl - fill_cost - carry_cost
sum(component attribution for session) = reported trading_cost
```

Event, data, target, configuration, model, schedule, and accrual identities
make a rerun auditable. Canonical digests detect changed content; they are not
digital signatures and do not replace access control or externally anchored
provenance.

## Bounds and failure policy

- Per-stage latency is at most 2,520 logical sessions; resolved latency is at
  most 5,040 sessions; a frozen calendar is limited to 1,000,000 sessions.
- Annual carry rates are at most 1,000,000 basis points and the annualization
  divisor is an integer from 1 through 366.
- Modeled money is bounded at USD `1e18`, absolute quantity at `1e15` shares,
  and carry price at USD `1e12` per share.
- Carry positions and price mappings are each bounded at 10,000 symbols before
  iteration or copying.
- Fill basis-point rates are bounded at 1,000,000; participation and lagged
  volatility are bounded inputs; metadata and collections have explicit size
  ceilings.
- Arithmetic overflow, malformed calendars, missing causal inputs, unknown
  settings, and reconciliation failure stop publication. No fallback may turn
  an invalid run into apparently valid evidence.

## Consequences

- A reviewer can trace each reported cost to one assumption, accounting path,
  causal basis, model digest, and event transition without widening the legacy
  fill-table schema.
- Default zero latency and zero carry preserve the prior causal timeline,
  while the default fill-cost assumptions retain their prior behavior.
- Resolving schedules constructs one bounded calendar-position map and is then
  linear in target decisions. Component publication is linear in fills plus
  open short positions. A full stress grid costs approximately one complete
  simulation per profile by design.
- Daily-bar evidence remains an offline sensitivity study. Committed tests and
  examples may use synthetic data; licensed or sensitive prices and journals
  stay in ignored, owner-controlled storage.

## Rollback

Rollback selects the baseline stress profile, zero logical stage delays, zero
carry rates, and the established fill-cost defaults, or removes the MR4
orchestration and additive evidence tables behind the compatibility facade.
Existing event journals, manifests, schedule records, and accrual records are
never rewritten to match a different model.

A rolled-back reader must either support the recorded schema and digest domain
or fail closed. It may preserve the legacy result tables, but it must not
silently drop a cash charge, reinterpret price-embedded shortfall as a fee, or
claim that an old event history was generated under a new configuration.

## Alternatives rejected

- **Subtract every component from cash.** Spread, slippage, and impact already
  alter the fill price; a second debit double-counts implementation shortfall.
- **Deduct costs from a completed return series.** This cannot change order
  size, participation, partial fills, positions, bankruptcy, or later costs.
- **Use timestamps synthesized from daily bars.** A daily bar cannot support
  claims about exchange time, queue position, auction state, or network delay.
- **Infer borrowability from a borrow-rate scalar.** A rate assumption neither
  establishes availability nor proves a locate.
- **Calibrate friction parameters on strategy test outcomes.** This invites
  selection bias and turns execution assumptions into another hidden tuning
  surface.
- **Expand the version-1 fill table in place.** Additive normalized evidence
  preserves reporting compatibility and makes component semantics explicit.
