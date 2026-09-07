# Deterministic event-driven backtesting

AlphaForge's daily-bar backtester records simulation decisions and accounting
transitions as a typed, append-only event stream. The stream makes order state,
cash, positions, charges, and restart behavior inspectable without inventing
exchange timestamps or queue position that daily OHLCV data cannot provide.

This is a deterministic simulation and research control, not an execution
gateway. Sprint 3 advanced no strategy candidate (`Advance = 0`, `NOT_READY`).
The Sprint 4 MR3 examples therefore use synthetic, explicitly non-candidate
targets. They do not unlock the held-out interval, establish paper-trading
readiness, authorize a live trade, or provide evidence of profit.

## Logical event model

An event is an immutable payload inside an envelope containing:

- schema, run, correlation, and entity identities;
- a logical session, zero-based bar index, phase, and deterministic ordinal;
- an optional causation identifier naming an earlier compatible event; and
- a SHA-256 event identifier derived from strict canonical JSON bytes.

Daily-bar phases have one causal order:

1. open portfolio mark;
2. order submission;
3. acceptance or rejection and zero or more fills;
4. cancellation of a DAY order's residual;
5. separately accrued cash charges;
6. close portfolio mark;
7. signal availability;
8. target decision, eligible strictly after its decision session; and
9. terminal engine control.

Equal coordinates use the content-derived event identifier as the queue
tie-breaker. The reducer rejects time travel, missing or incompatible causes,
cross-correlation causes, reused semantic identities, excess fills, illegal
terminal transitions, and work beyond configured event, pending-event, order,
or open-order ceilings. Exact duplicates are idempotent.

The constructor requires the frozen run calendar as a unique, strictly
increasing sequence of date-only sessions. Every event's zero-based bar index
must match that calendar exactly, and replay requires the same calendar. The
current journal schema does not pretend it can reconstruct an omitted calendar;
callers must preserve that run input with the journal's trusted provenance.

The order state machine is:

```text
SUBMITTED -> ACCEPTED | REJECTED | CANCELLED
ACCEPTED -> PARTIALLY_FILLED | FILLED | CANCELLED
PARTIALLY_FILLED -> PARTIALLY_FILLED | FILLED | CANCELLED
FILLED | REJECTED | CANCELLED -> terminal
```

Every unfilled DAY residual is a recorded cancellation rather than an inferred
absence. A non-positive marked equity requires an explicit terminal bankruptcy
event before the reducer accepts any further event.

## Accounting invariants

Let `q` be signed position quantity, `a` its average cost, `dq` a signed fill
quantity, and `p` the fill price. The realized P&L on the closing portion is:

```text
closing_quantity = min(abs(q), abs(dq))
realized_delta = closing_quantity * (p - a) * sign(q)
```

A same-side addition uses the absolute-quantity weighted average cost. If a
fill crosses zero, the remaining opposite-side position takes the fill price
as its cost basis. Cash changes exactly once:

```text
cash_after = cash_before - dq * p - categorized_fill_fees
```

Spread, slippage, and impact already embedded in `p` are not charged a second
time. Financing, borrow, and other supplied USD charges use separate
categorized cash events. The MR3 ledger is deliberately single-currency:
non-USD cash charges fail schema validation, and FX conversion remains outside
this event contract. At every portfolio mark the reducer independently checks:

```text
equity = cash + sum(quantity[symbol] * mark[symbol])
equity = initial_cash + realized_pnl + unrealized_pnl - total_charges
```

Gross exposure is the sum of absolute signed market values; net exposure is
their signed sum. Both derive from the same immutable per-symbol market values
used in equity reconciliation. After every fill or cash charge, the reducer
publishes a fresh open-price snapshot; state-only events retain the latest
reconciled snapshot until the next mark. Any accounting event that produces
non-positive equity must be followed immediately by an exactly caused
bankruptcy halt.

The implementation uses accurate summation and scale/ULP-aware reconciliation;
it does not grant a one-currency-unit absolute error floor. A mark must include
every held instrument, and its declared cash, holdings value, charges, and
equity must agree with reducer-owned state.

## Journal, replay, and trust boundary

The default bounded in-memory journal is appropriate for deterministic tests.
The SQLite adapter is a local durable option with append-only schema controls,
full synchronization, single-writer locking, canonical event bytes, and a hash
chain over sequence and prior state. Restart first validates the complete byte
chain and then replays semantic transitions.

The hash chain detects accidental or uncoordinated modification; it is not a
digital signature against an attacker who can rewrite the database and every
receipt. Accounting mutations are first validated on an independent candidate
ledger and become authoritative only after append. An append failure can still
have an uncertain durable outcome, so the engine is poisoned. Discard it and
replay the verified committed prefix; never retry on the uncertain instance.

Portfolio-mark events contain the prices required for independent replay.
Those prices may be licensed Nasdaq Data Link observations or otherwise
restricted data. Runtime SQLite files, canonical exports, logs, and derived
run artifacts must remain in ignored owner-controlled storage. Never commit a
runtime journal, an API credential, or licensed raw data. Committed tests and
benchmark fixtures use redistribution-safe synthetic values only.

## Complexity and resource behavior

For `Q` queued events, `P` open positions, `R` symbols with accumulated
realized P&L, `B` bounded canonical bytes per event, and `E` committed events:

| operation | time | bounded resource behavior |
|---|---:|---|
| ordered order transition and in-memory append | `O(B)` | maximum events, orders, open orders, and bytes |
| ordered fill or charge and in-memory append | `O(P + R + B)` | validates a cloned candidate ledger before commit |
| queue insertion | `O(B + log Q)` | maximum pending events and total events |
| portfolio mark and append | `O(P + R + B)` | bounded canonical price snapshot |
| verified replay | `O(E * B + reducer work)` | validates bytes before semantic reconstruction |

These are algorithmic bounds, not latency promises. Durable SQLite append adds
`O(log E)` B-tree index maintenance to the reducer and canonical-byte work,
plus filesystem and synchronization latency for the configured full-durability
transaction. Python object allocation, canonical JSON, hashing, timer
instrumentation, and the host runtime remain visible costs.
The reference benchmark holds only one synthetic symbol, so it measures steady
event mechanics rather than position-count scaling.

## Reproducible CPU benchmark

Run the bounded synthetic harness from the repository root:

```bash
uv run python scripts/bench_event_engine.py \
  --samples 7 \
  --warmups 2 \
  --orders 256 \
  --output runs/event-engine-benchmark.json
```

Use a new ignored output path for every environment; the command refuses to
overwrite an existing artifact. Omitting `--output` writes JSON to standard
output. The document records the Python and operating-system environment,
logical CPU count, sample and warm-up counts, deterministic event composition,
committed event and filled-order counts, stream and journal hashes, total
throughput, and observed p50/p95/p99 latency distributions.

The timed region is ordered `DeterministicEventEngine.process` work, including
canonical in-memory journal appends and accounting. Event construction and
engine setup are excluded. Individual latency observations include the
`perf_counter_ns` instrumentation cost. The harness is CPU-only, performs no
network or file read, uses no random generator, and supplies no favorable
performance threshold. Compare raw distributions across pinned environments;
do not turn noisy laptop timing into a correctness gate or a trading claim.

## Execution-model boundary

MR3 established event ordering, lifecycle, accounting, idempotency, integrity,
replay, and bounded failure behavior. MR4 now supplies the causal calculators
for commissions, exchange fees, spread, slippage, market impact, logical
latency, financing, and borrow accrual. Component provenance and separate
fill-price, fill-fee, and cash-charge paths prevent double charging. See
[ADR 0011](adr/0011-market-frictions-and-logical-latency.md) and the
[market-friction guide](market_frictions_latency.md).

MR5 owns point-in-time borrow availability and locates, liquidity and
  participation constraints, forced buy-ins, aggregate capacity budgets, and
  capacity-frontier evidence.

Daily bars still cannot establish exchange time, queue priority, fill
probability, current borrow, or market-impact calibration. Those limitations
remain explicit even when deterministic replay is perfect.
