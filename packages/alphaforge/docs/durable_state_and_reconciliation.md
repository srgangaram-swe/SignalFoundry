# Durable session state and broker reconciliation (SF-S5-MR4)

Transactional, hash-chained persistence of decision and order identity, positions,
cash, and versions; restart recovery that refuses anything it cannot verify; and a
reconciliation pass that halts on divergence and never repairs.

> Simulation only. No live order, no capital, no broker connection. The paper
> session this protects is still unopenable — it requires a `QUALIFIED_FOR_PAPER`
> verdict and Sprint 4's verdict is `REJECTED`.

Design rationale: [ADR 0017](adr/0017-durable-session-state-and-halt-on-divergence.md).
Requirements implemented: [broker connectivity requirements §3](broker_connectivity_requirements.md).
Builds on: [broker contract and paper adapter](broker_contract_and_paper_adapter.md).

---

## 1. The gap this closes

MR3 made a replayed decision cycle a no-op: resubmitting a known `client_order_id`
returned the existing order instead of creating a second one. That idempotency
lived in a dictionary inside the adapter.

**A process restart empties that dictionary.** The next cycle would resubmit every
order it had already sent, and the broker's own dedupe window — measured in
minutes to hours — is not long enough to be relied on across an operator restarting
a service the following morning.

So the durable record is what makes idempotency real:

```
decision made ──▶ intent persisted ──▶ broker contacted ──▶ status persisted
                        ▲                                          │
                        └──────── restart recovers from here ◀─────┘
```

**The intent is written before the broker is contacted.** A crash in the window
between "decided" and "acknowledged" then leaves evidence that the order *may*
exist. Recording only after acknowledgement would make that window invisible — and
it is precisely the window in which duplicates are born.

## 2. Atomic or absent

Each snapshot is one row written inside a single `BEGIN IMMEDIATE` transaction
under WAL with `synchronous = FULL`. There is no intermediate state in which the
head is half-written: a crash before commit leaves the previous snapshot as head,
a crash after leaves the new one.

`test_a_failed_write_leaves_the_previous_head_intact` injects an I/O failure at
the `INSERT` and asserts the recovered head is byte-identical to the pre-failure
one, with the chain still verifying at length 1.

## 3. The hash chain

Every snapshot carries `content_hash` over all of its own fields and
`previous_hash` pointing at its predecessor. This detects more than modification:

| Attack or fault | Detected by |
| --- | --- |
| A field edited in place | `content_hash` mismatch |
| A record removed from the middle | sequence gap |
| A record removed from the end | chain still valid — see limitations |
| Records reordered | `previous_hash` mismatch |
| A truncated payload | missing-key refusal, before any field is read |

The truncation case matters on its own: a payload missing `positions` is refused
as *not partially usable*, rather than loaded as a session that happens to hold
nothing.

## 4. Recovery refuses rather than guesses

Every one of these raises a distinct exception type:

| Condition | Exception | Why refusing beats loading |
| --- | --- | --- |
| Hash mismatch, chain break, sequence gap, truncation | `SnapshotIntegrityError` | The record changed since it was written; no part is trustworthy |
| Different `SCHEMA_VERSION` | `SnapshotIncompatibleError` | A silent migration is how a field acquires a new meaning without anyone deciding it should |
| Different strategy | `SnapshotIncompatibleError` | Recovering another strategy's positions would attribute them to this one |
| Different config identity | `SnapshotIncompatibleError` | The configuration that produced these positions is not the one about to act on them |
| Older than `max_age` (default 12h) | `SnapshotStaleError` | A stale view of the book cannot authorize an order |
| `now` precedes the snapshot | `ClockRollbackError` | Staleness is unjudgeable when the clock moved backwards |
| Duplicate sequence numbers | `DurableStateError` | There is no single head to recover from |

The alternative — loading the best available interpretation — resumes trading
against a position book nobody has verified.

**Clock rollback is checked on write as well as read.** Wall clocks move backwards
on NTP correction, and "latest by timestamp" silently picks the wrong record when
they do. The sequence is monotonic and independent of wall time; the timestamp is
validated against it rather than trusted.

## 5. Reconciliation halts and never repairs

The store records what the system *believes*. The broker reports what it *did*.
`reconcile` compares them and, on any material difference, halts.

**It never repairs.** If the local book says 100 shares and the broker says 150,
the difference is one of: an unrecorded fill, a duplicate submission, a manual
intervention, or a bug. Those four have different correct responses. Writing 150
into the local book picks one interpretation and destroys the evidence needed to
distinguish it from the others.

**It never liquidates.** An automatic flatten is a market order sized from an
unverified position — the one action guaranteed to be wrong when the position is
what you are unsure about.

Eight divergence kinds are reported separately, because collapsing them into
"mismatch" discards what a human needs:

`position_quantity`, `position_only_local`, `position_only_broker`, `cash`,
`order_only_local`, `order_only_broker`, `order_state`, `unrecorded_fill`.

Each carries a plain-language interpretation. `order_only_local` says the
submission may have been lost in flight and resubmitting without confirming would
risk a duplicate — which is the operational instruction, not a description.

The cash tolerance defaults to `0.01` and is deliberately tight: it is a tolerance
for decimal representation, not for disagreement.

## 6. Out-of-order and duplicate fills

`apply_fills_idempotently` folds a fill sequence into a position book keyed by
`fill_id`, so:

- a fill delivered twice counts once;
- a sequence delivered out of order produces the same book as the in-order one;
- a symbol reaching zero net quantity is **removed**, because a flat symbol left
  in the book reconciles as a spurious `position_only_local` divergence.

That convergence is what makes a dropped-and-redelivered acknowledgement safe to
process without first proving it is new. A reused `fill_id` carrying *different*
content is refused: identity is what makes deduplication safe, and reuse makes it
unsafe.

## 7. Trust boundary and path hardening

The store file is treated as adversarial, matching the existing journal posture:
symlinks, non-regular files, hard-linked aliases, foreign ownership, and
group/other-readable permissions are all refused before the database is opened,
and the file is chmod'd to `0600`.

This is a local research persistence boundary. It is tamper-**evident**, not
tamper-proof: an attacker who can rewrite the database and every trusted reference
to it is out of scope, exactly as stated for the event journal.

## 8. Runbook

**Normal restart:** call `recover()` with the current time, expected strategy, and
expected config identity. A `None` return means an empty store — a first run.
Then reconcile against the broker before submitting anything.

**On any recovery exception:** do not retry, do not delete the store. Preserve it
and escalate; the exception type names the fault class.

**On a reconciliation halt:** stop submitting, preserve state, require operator
review. Determine which of the four causes applies, correct the underlying record
deliberately, and re-run reconciliation. Do not resume on an unexplained
divergence — that is the issue's explicit non-goal.

**Rollback:** the modules are additive and nothing in a production path imports
them. Reverting the commit removes them; the store file is inert data.

## 9. Residual limitations

- **Truncation at the tail is not detectable from the chain alone.** Removing the
  most recent snapshot leaves a valid chain ending earlier. Detecting that needs
  an external high-water mark, which is out of scope here; reconciliation against
  the broker is the compensating control.
- **Single-writer.** The store is opened on one thread and SQLite serializes
  writers, but there is no cross-process lease. Two processes on one store is
  not a supported configuration and is not currently detected.
- **`max_age` is a judgement.** Twelve hours suits a daily-cadence strategy and
  would be far too generous at higher frequency.
- **Reconciliation compares what the broker reports.** A broker that reports
  stale state reconciles cleanly against a stale local view.
- **No automatic recovery action exists by design.** Every halt requires a human.
  That is a deliberate cost, and at higher cadence it would be a real operational
  burden.

## 10. Evidence

`tests/test_broker_durable_state.py` — 47 tests: restart recovering already-sent
orders; intent recorded before contact; open-versus-terminal intent filtering;
duplicate-ID refusal; chain construction and verification; tampered payload;
truncated payload; removed middle record; injected write failure leaving the head
intact; empty-store recovery; staleness; clock rollback on both read and write;
foreign strategy; foreign configuration; incompatible schema; symlink, hard-link,
and permission refusals; float-money refusal; unknown state and side; oversized
identifier; matching reconciliation; quantity, cash, position-only-local,
position-only-broker, order-only-local, order-only-broker, and order-state
divergences; the reconciled-while-diverged contradiction; float tolerance
refusal; duplicate fills; out-of-order convergence; reused fill identity;
closing fills; opening books; and JSON serializability of both records.
