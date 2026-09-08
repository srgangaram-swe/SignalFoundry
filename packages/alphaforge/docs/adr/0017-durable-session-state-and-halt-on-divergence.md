# ADR 0017 — Durable session state and halt-on-divergence reconciliation

- **Status:** Accepted
- **Date:** 2026-08-06
- **Work item:** SF-S5-MR4 (#46), Signal Foundry Sprint 5
- **Builds on:** ADR 0010 (deterministic event sourcing), ADR 0015 (broker selection), ADR 0016 (deny-by-default broker authorization)

## Context

MR3 made a replayed decision cycle a no-op by remembering submitted client order
IDs in a dictionary. That dictionary lives in a process. A restart empties it, and
the next cycle resubmits every order already sent.

Relying on the broker's own deduplication does not close this. Alpaca's dedupe
window is bounded and far shorter than the interval between an evening crash and a
morning restart. Beyond the window, the broker will happily accept the same
`client_order_id` as a new order — which is correct behaviour on its part and a
duplicate position on ours.

The second problem is what to do when local and broker state disagree after a
restart. The tempting design is to treat the broker as authoritative and overwrite
the local book. It is tempting because it always produces a runnable state.

## Decision

**Intent is persisted before the broker is contacted.** The window between
"decided" and "acknowledged" is where duplicates are born; recording only after
acknowledgement makes that window invisible.

**Snapshots are atomic.** One row, one `BEGIN IMMEDIATE` transaction,
`synchronous = FULL`. A crash before commit leaves the previous head; after,
the new one. No half-written head exists.

**Snapshots are hash-chained.** `content_hash` over the record and
`previous_hash` to its predecessor, so removal from the middle and reordering are
detectable, not only modification.

**Recovery refuses six distinct conditions** with six distinct exception types:
integrity failure, schema incompatibility, foreign strategy, foreign
configuration, staleness, and clock rollback. Each is a stop, not a warning.

**Clock rollback is checked on write and on read.** Sequence is monotonic and
independent of wall time; timestamps are validated against it rather than trusted.

**Reconciliation halts and never repairs.** A quantity difference may be an
unrecorded fill, a duplicate submission, a manual intervention, or a bug — four
causes with four different correct responses. Overwriting picks one and destroys
the evidence distinguishing it from the others. Divergences are reported in eight
labelled kinds with plain-language interpretations.

**Reconciliation never liquidates.** An automatic flatten is a market order sized
from an unverified position.

**Fills fold by identity.** `apply_fills_idempotently` is keyed on `fill_id`, so
duplicate delivery and out-of-order arrival converge to the same book. A reused
`fill_id` with different content is refused.

**Money persists as Decimal text**, so a restart reads back exactly what was
written.

**The store file is hardened** — symlink, hard-link, ownership, and permission
checks before open, then `0600` — matching the existing event-journal posture.

## Consequences

**Accepted costs.**

- Every halt requires a human. At daily cadence that is a few interventions a
  year; at higher frequency it would be a real operational burden, and that
  tradeoff would have to be revisited rather than automated away.
- A write per decision cycle with `synchronous = FULL` costs an fsync. Correct at
  this cadence; it would need batching at higher frequency.
- Refusing a stale snapshot means an operator who returns after a long outage
  must reconcile before resuming, rather than picking up where the process left
  off. That is the intended friction.
- Schema version bumps require deliberate migration work rather than a silent
  upgrade path.

**What this does not buy.**

- Tail truncation — deleting the most recent snapshot — leaves a valid chain
  ending earlier. Detecting it needs an external high-water mark; reconciliation
  against the broker is the compensating control.
- No cross-process lease. Two writers on one store is unsupported and undetected.
- Tamper-evident, not tamper-proof. An attacker who can rewrite the database and
  every trusted reference to it is out of scope, as for the event journal.

## Alternatives considered

**Trust the broker's deduplication window.** Rejected: the window is bounded and
shorter than realistic restart intervals. Beyond it the broker correctly accepts a
repeated ID as a new order.

**Treat the broker as authoritative and overwrite local state on divergence.**
Rejected as the motivating failure. It always produces a runnable state, which is
exactly why it is dangerous — it converts an unexplained disagreement into a
confident-looking book and deletes the evidence.

**Automatically flatten positions on divergence.** Rejected: sizing a market order
from the position you are unsure about is the one action guaranteed to be wrong.

**Append-only event log instead of snapshots.** Considered seriously — the
repository already has `SQLiteJournal` for exactly that, and it is the better
model for replay. Rejected here because recovery needs the *current* book quickly
and cheaply, and folding a long event history on every restart makes recovery cost
grow without bound. The two coexist: the journal is the replay record, this is the
recovery point.

**Store money as float.** Rejected. It would not read back as written, and the
drift surfaces as a reconciliation failure that is correct while the ledger is
wrong.

**Recover the "best available" snapshot when the head fails verification.**
Rejected: that resumes trading against a book nobody verified, which is the
outcome the whole module exists to prevent.
