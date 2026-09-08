# Point-in-time borrow, liquidity, and capacity (SF-S4-MR5)

One causal policy, enforced at both the decision and execution boundaries, over
versioned immutable records for borrow availability, simulated locates, lagged
liquidity, and conserved capacity budgets.

> **No capacity claim.** Every figure here comes from synthetic, redistribution-safe
> fixtures. Nothing in this module establishes a deployable AUM, broker capacity,
> paper-trading readiness, or expected profit. Rationale and alternatives are in
> [ADR 0012](adr/0012-point-in-time-borrow-and-capacity-policy.md).

---

## 1. What this closes

MR4 models the borrow **rate**. A rate is a price, and a price is not permission:
it says what a short would cost, never that the security was borrowable or that a
locate existed. A backtest with no borrow data therefore shorts freely, because
absence of a constraint reads as absence of a limit.

MR5 supplies the missing evidence and enforces it consistently at decision, order,
fill, and replay.

## 2. The records

| record | states | key temporal fields |
|---|---|---|
| `BorrowAvailability` | can this symbol be shorted, and up to what size | `as_of_session`, `effective_session`, `expiry_session` |
| `LocateRecord` | a bounded grant authorizing new shorts | `granted_session`, `expiry_session` |
| `LiquidityObservation` | lagged ADV in shares and notional | `as_of_session`, `lookback_sessions` |
| `CapacityPolicyDeclaration` | the frozen limits everything is judged against | — |

**Observation time is separate from effective time.** `as_of_session` says when a
row was *observed*; `effective_session` says what it *describes*. Only rows
observed strictly earlier than the decision session are admitted. Collapsing the
two is how a backtest silently learns tomorrow's borrow book while looking
perfectly causal.

Every record publishes a content-derived SHA-256, and `book_digest` binds a whole
resolved book into one identity that is **order-independent** — two runs consuming
the same rows in different order produce the same identity, so identity tracks
content rather than file layout. Digests are identity and integrity evidence, not
signatures.

## 3. The rules that carry the weight

**Absence is never permission.** Missing, stale, expired, unknown, duplicated, or
conflicting → **zero** new-short capacity. There is no path from "no data" to
"unconstrained". Long-only is reachable only by explicit declaration
(`allow_new_shorts=False`), never by failure.

**Stale is named as stale.** Both a missing and a stale record give zero, but the
denial reason distinguishes them, because "we had no feed" and "our feed went
stale" call for different operational fixes.

**Duplicates and conflicts are refused, not resolved.** Two rows for the same
`(symbol, effective_session)` are a data error. Silently taking the last would let
record ordering decide how much a strategy may short.

**A cover is never blocked by borrow.** Opening a short needs positive evidence on
every axis; covering needs none of it. Blocking a risk-reducing trade because new
borrow is unavailable would trap the book in exactly the position the restriction
was warning about. Covers still consume liquidity, participation, and budget.

**A non-available status cannot carry capacity.** A `restricted` row with a
positive `shortable_quantity` is a contradiction and is refused at construction.

## 4. Conserved capacity

Capacity is a ledger, not a check. After every mutation:

```
reserved_outstanding + consumed + released + rejected == requested
```

This is asserted in the production path, not only in tests — a leak that manifests
after thousands of events would otherwise surface as a slightly wrong backtest
rather than an error.

* **Reservations and releases are idempotent by identifier.** Journal replay
  re-presents the same identifiers; a second charge would leak capacity and a
  second credit would manufacture it.
* **Partial fills return the remainder.** Consume shrinks the outstanding claim so
  a later release credits only what was genuinely unused.
* **A fill larger than its reservation raises** rather than being clipped, because
  clipping would let the engine overtrade its budget silently.
* **Shares and notional are tracked separately.** Participation is a share
  constraint; the book budget is a notional one. Conflating them lets a
  high-priced symbol consume a low-priced symbol's allowance.
* **An unknown symbol has zero capacity**, never unlimited.

## 5. Forced buy-ins

A recall, restriction, or *disappearance* of a held short schedules a buy-in at the
next causally eligible session. A vanished record counts as `unavailable`: silence
is not evidence the borrow survived.

A buy-in is an ordinary trade — it consumes participation and budget, pays the same
frictions, and can partially fill. Modelling it as an instant costless exit would
hide the risk that makes recalls dangerous.

**An unresolved residual halts publication.** When the bounded window expires with
shares outstanding, the run raises `ForcedBuyInHalt` and no artifact is published.
It is never carried, never dropped. A surviving unauthorized short makes every
downstream P&L, risk, and capacity figure a description of an impossible book.

## 6. The capacity frontier

Every AUM scenario **reruns the complete simulation**. Scaling a completed return
series by a capital ratio produces a smooth, attractive, and entirely fictitious
frontier: it assumes the same trades happened at every size, which is precisely the
assumption capacity analysis exists to test.

* Scenarios are **candidate-order isolated** — a result must not depend on which
  scenario ran before it, and that is asserted.
* Infeasible scenarios **stay in the table** as refusals. Dropping the hard
  scenarios is how a capacity curve grows an optimistic tail.
* `verify_row_aggregation` recomputes the table from the raw results, so nothing
  was smoothed, rescaled, or reordered between them.
* `frontier_summary` reports where constraints *begin to bind* and carries an
  explicit statement that no figure is a deployable AUM, broker capacity, paper
  readiness, or expected profit — so the caveat travels with the data.

## 7. Evidence

`tests/test_capacity_policy.py` — 76 tests, grouped by acceptance criterion:
contract validation and fail-closed rejection (malformed symbols, non-finite and
boolean quantities, overflow, unsupported schema, datetimes-as-sessions, inverted
and acausal sessions, duplicates and conflicts); causality (records published on or
after the decision are invisible; a future-mutation test asserts an earlier
authorization is bit-identical after rewriting all later rows); absence and
staleness giving zero; expired locates; cover-never-blocked across all three
non-available statuses; monotonicity under reduced ADV, availability, and budget;
conservation under partial fills, cancellation, replay, over-request, and
cross-symbol book budget; forced buy-in resolution, idempotence, halt on expiry and
on unresolved run end; frontier rerun-per-scenario, row aggregation, infeasible
retention, order isolation, and grid validation.

Coverage: `budgets.py` 95%, `contracts.py` 92%, `buyin.py` 91%, `policy.py` 90%,
`frontier.py` 87% (branch). Repository total 83.94% against an unchanged 78% floor.

## 8. Residual limitations

* **Synthetic fixtures only.** Locate identifiers are simulation artifacts; real
  broker locate identifiers must never be committed.
* **Fail-closed defaults reduce measured short capacity** relative to earlier
  Sprint 4 numbers. That is the correction, not a regression, but comparisons
  across the boundary are not like-for-like.
* **A cover blocked by stale liquidity can force a halt** during a data outage.
  Deliberate: sizing with no volume estimate is guesswork.
* **Nothing is inferred from OHLCV.** Borrowability, locates, queue position, fill
  probability, intraday liquidity, recalls, and broker behaviour are inputs, never
  derived — inferring them would manufacture the evidence under test.
* **No broker connectivity, routing, paper, or live trading.** Those remain Sprint
  5 and 6 work behind explicit owner-approved gates.
