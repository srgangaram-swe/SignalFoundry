# ADR 0010: Deterministic event sourcing and reconciled portfolio accounting

- Status: Accepted
- Date: 2026-08-01
- Owners: AlphaForge research platform
- Builds on: ADR 0001

## Context

ADR 0001 corrected the economic timeline by making close-time decisions fill no
earlier than a future open and by replacing assumed held weights with signed
shares and cash. The original daily-bar loop nevertheless encoded order
transitions only implicitly. It could report orders and fills, but it did not
have a durable state machine for acceptance, rejection, cancellation,
idempotency, restart, categorized cash charges, average cost, or realized and
unrealized P&L.

That gap makes recovery and fault evidence weaker than the accounting itself.
A restart must not infer state from whichever output tables happen to exist,
and a duplicate or time-traveling event must not silently apply twice.

Sprint 3 advanced no strategy candidate. This decision therefore governs a
simulation and accounting capability exercised with deterministic synthetic
targets; it does not qualify a strategy or authorize paper or live execution.

## Decision

### Logical event time

Daily OHLCV data cannot support truthful exchange timestamps or queue claims.
Events use a logical coordinate consisting of trading session, bounded bar
index, phase, and ordinal. Within a session, the canonical phase order is:

1. open mark;
2. order submission;
3. acceptance, rejection, and fills;
4. DAY residual cancellation;
5. separately charged cash accruals;
6. close mark;
7. signal availability;
8. target decision; and
9. terminal engine control.

This preserves ADR 0001: a target created after the close mark cannot own the
preceding overnight or intraday return. Equal logical coordinates use the
content-derived event identifier as a deterministic tie-breaker.

Every reducer instance also receives the complete frozen run calendar. It
requires a unique, strictly increasing sequence of date-only sessions and
checks that every event's zero-based bar index names exactly its declared
session. Replay requires the same external calendar. A later journal-schema
revision may bind a calendar digest in a run-start event; schema v1 does not
claim the event rows alone reconstruct omitted run provenance.

### Event and identity contract

Events are frozen, bounded, typed records for signal availability, target
decisions, submitted/accepted/rejected/cancelled orders, fills, cash charges,
portfolio marks, and terminal halts. Every envelope carries a schema version,
run/correlation/entity identity, coordinate, optional causation identity, and a
SHA-256 identifier derived from canonical JSON bytes.

Canonical records reject unknown fields, non-finite numbers, booleans presented
as numbers, duplicate or unsorted symbols/categories, unsupported versions,
invalid phase/payload combinations, oversized payloads, and inconsistent
identities. Exact duplicates are idempotent. Reusing a semantic identity or
event identifier with different content is an integrity error.

Target events contain only bounded weights and generic configuration, data,
problem, and solver identities. Runtime portfolio marks contain the prices
needed for independent restart reconciliation. They can therefore contain
licensed or sensitive runtime information and belong only in ignored,
owner-controlled run storage; committed evidence uses synthetic fixtures.

### Reducer and accounting contract

The reducer enforces this order state machine:

```text
SUBMITTED -> ACCEPTED | REJECTED | CANCELLED
ACCEPTED -> PARTIALLY_FILLED | FILLED | CANCELLED
PARTIALLY_FILLED -> PARTIALLY_FILLED | FILLED | CANCELLED
FILLED | REJECTED | CANCELLED -> terminal
```

Fills must follow acceptance, match order symbol and side, remain within the
accepted residual, and have unique fill identities. A DAY residual is cancelled
explicitly. Causation must name an earlier compatible event from the same
correlation chain. An order must occur on its target's exact eligible session,
its symbol must exist in that target book, every DAY transition remains on the
submission bar, and fill reference prices match that bar's open mark. Causal
fills use increasing explicit ordinals rather than hash tie order. Events
earlier than the committed cursor fail as time travel.

For old signed quantity `q`, average cost `a`, signed fill `dq`, and fill price
`p`, closing P&L is:

`min(abs(q), abs(dq)) * (p - a) * sign(q)`.

Same-side additions use absolute-quantity weighted average cost. A position
crossing zero takes the fill price as the residual side's new cost basis. Cash
changes by:

`cash_after = cash_before - dq * p - separately_charged_fill_fees`.

Spread, slippage, and impact embedded in `p` are not subtracted again. Financing,
borrow, and other supplied charges enter as separate categorized USD cash
events. MR3 rejects non-USD charges; multi-currency accounting and FX conversion
require a separate explicit contract.
At every mark:

```text
equity = cash + sum(quantity * mark)
equity = initial_cash + realized_pnl + unrealized_pnl - total_charges
```

Both identities use `math.fsum` and scale/ULP-aware tolerances without a
currency-unit absolute floor. Non-positive equity requires an explicit
bankruptcy halt before any further mutation.

Gross exposure is `sum(abs(quantity * mark))`; net exposure is the signed sum
used in the equity identity. Fills and charges publish a fresh reconciled
snapshot at the current open marks, while order-only transitions retain the
latest snapshot. Thus every committed event after the initial open mark exposes
one internally reconciled cash, P&L, charge, gross/net exposure, and equity
state. A bankruptcy halt cites the exact accounting event that first made
equity non-positive.

### Journal and recovery boundary

The default journal is bounded and in memory. Durable runs may use the standard
library SQLite journal under an explicit caller-owned path. SQLite uses a
versioned schema, WAL, `synchronous=FULL`, fail-fast single-writer locking,
parameterized statements, append-only triggers, and one transaction for the
event row and head update. Database files are regular non-symlink files created
with owner-only permissions.

Every row binds its sequence, previous hash, canonical event bytes, and record
hash. Open, export, and restart verify schema, sequence, hashes, resource limits,
and canonical event identities before semantic replay. Hash chaining detects
accidental or uncoordinated mutation; it is not a digital signature against an
attacker able to rewrite the database and every receipt.

Ledger transitions validate on an independent candidate before append and the
candidate becomes authoritative only after durability succeeds. If a journal
append fails or has an uncertain outcome, that engine becomes non-publishable.
The only supported recovery is to discard it and reconstruct from the journal's
verified committed prefix. Retrying on the uncertain instance is forbidden.

## Consequences

- Backtests gain deterministic event exports, explicit order failures and DAY
  expiry, exact cost-basis accounting, restart verification, and auditable halt
  behavior while preserving existing public result tables.
- Queue insertion is `O(log Q)`; order-only transitions are `O(1)`; current
  fill/charge candidates and marks are `O(P + R)` because the ledger rechecks
  cumulative position and realized-P&L invariants; full restart verification
  is `O(E)` plus reducer work.
- Durable replay adds storage and fsync cost. This is appropriate for evidence
  and recovery; it is not a low-latency exchange gateway.
- MR3 accounts supplied fees, financing, and borrow charges. MR4 owns causal
  calculators and latency models; MR5 owns borrow availability, locates,
  liquidity constraints, forced buy-ins, and capacity policy.
- Daily bars still do not establish exchange time, queue position, fill
  probability, market impact calibration, or operational trading readiness.

## Rollback

The compatibility backtest facade can stop emitting the new audit tables while
retaining ADR 0001's ledger and result tables. Existing journal history is never
rewritten or downgraded. A reader that does not support the exact schema fails
closed; rollback does not silently convert or discard durable events.

## Alternatives rejected

- **Infer events from output CSV files.** Those tables are reporting views, not
  an atomic state transition log, and cannot prove idempotent restart.
- **Use wall-clock timestamps synthesized from daily bars.** This would add
  false precision and could conceal same-bar causality errors.
- **Serialize Python objects with pickle.** Executable deserialization is an
  unnecessary trust boundary; canonical JSON is sufficient and reviewable.
- **Retry after an uncertain commit.** The in-memory mutation may no longer
  match the durable prefix. Verified replay is the only defensible recovery.
- **Treat the hash chain as a signature.** Local integrity is useful, but
  adversarial provenance needs externally anchored receipts and access control.
