# ADR 0012 — Point-in-time borrow, locate, and capacity policy

- **Status:** Accepted
- **Date:** 2026-07-27
- **Work item:** SF-S4-MR5 (#9), Signal Foundry Sprint 4
- **Builds on:** ADR 0010 (event sourcing and accounting), ADR 0011 (frictions and latency)

## Context

ADR 0011 gave the backtest a borrow *rate*. A rate is a price, and a price is not
permission: it states what a short would cost, never that the security was
borrowable or that a locate existed. Meanwhile MR1/MR2 enforced allocation caps
and MR4 enforced fill-level participation, but nothing proved the *same finite
capacity* was enforced consistently at the decision, order, fill, and replay
boundaries.

The concrete failure this closes: a backtest with no borrow data shorts freely,
because absence of a constraint reads as absence of a limit. Every capacity
number downstream is then describing a book that could not have existed.

## Decision

**Absence is never permission.** A missing, stale, expired, unknown, duplicated,
or conflicting record yields **zero** new-short capacity. There is no code path
from "no data" to "unconstrained". The rollback posture — long-only — is
reachable only by explicit declaration (`allow_new_shorts=False`), never by
failure.

**Observation time is separate from effective time.** Each record carries
`as_of_session` (when it was observed) distinctly from `effective_session` (what
it describes). Only rows observed *strictly earlier* than the decision session are
admitted. Collapsing the two is how a backtest silently learns tomorrow's borrow
book while looking perfectly causal.

**Authorization is asymmetric by direction.** Opening a short requires positive
evidence on every axis: active availability, an unexpired locate with remaining
quantity, fresh liquidity, and remaining participation and book budget. Covering
requires none of the borrow evidence — blocking a risk-reducing trade because new
borrow is unavailable would trap the book in exactly the position the restriction
was warning about. Covers remain subject to liquidity, participation, cost, and
accounting.

**Capacity is a conserved ledger, not a check.** `reserved + consumed + released +
rejected == requested` is asserted after every mutation, not only in tests: a leak
that manifests after thousands of events would otherwise surface as a slightly
wrong backtest rather than an error. Reservations and releases are idempotent by
identifier so journal replay cannot double-charge or double-credit.

**Participation (shares) and book budget (notional) are tracked separately.**
Conflating them lets a high-priced symbol silently consume a low-priced symbol's
allowance.

**An unresolved forced buy-in halts publication.** When a bounded resolution
window expires with shares outstanding, the run raises rather than carrying or
dropping the residual. A surviving unauthorized short makes every downstream P&L,
risk, and capacity figure a description of an impossible book.

**The capacity frontier is complete reruns only.** Each AUM scenario re-runs the
whole simulation. Scaling a completed return series by a capital ratio assumes the
same trades happened at every size, which is precisely the assumption capacity
analysis exists to test.

## Alternatives considered

**Infer borrowability from price/volume.** Rejected and listed as an explicit
non-goal: daily OHLCV contains no information about borrow supply, and a model
that pretends otherwise manufactures the evidence it is supposed to test.

**Enforce capacity only at fill time.** Simpler, but it lets the optimizer build
targets it can never execute, so the reported target book and the achievable book
diverge silently. The policy is applied at both boundaries with the fill session's
own resolved view, which is what catches a locate that expired overnight.

**Treat a stale record as its last known value.** Rejected. Staleness is
information: it means the operator lost the feed, and continuing to short on a
week-old book is the behaviour that turns a data outage into a position.

## Consequences

**Positive.** No dependency added. Every refusal carries a closed-vocabulary
reason code, so denial evidence aggregates across runs. Content-derived digests
tie a run to the exact book it consumed. Conservation is enforced continuously.

**Negative.** Fail-closed defaults make the simulation *less* permissive than a
naive backtest, so measured short capacity — and therefore returns from short
books — will fall relative to earlier Sprint 4 numbers. That is the correction,
not a regression, but comparisons across the boundary are not like-for-like and
should not be presented as such.

A cover blocked by stale *liquidity* can force a halt during a data outage. That
is deliberate: sizing a trade with no volume estimate is guesswork, and the halt
surfaces an operational problem that would otherwise be absorbed silently.

**Rollback.** Select `allow_new_shorts=False` for a long-only posture and remove
the additive `alphaforge/capacity/` orchestration. It must never fall back from a
missing policy to unconstrained shorting, rewrite prior journals, discard an
unresolved buy-in, or reinterpret a historical run under different assumptions.

## Residual risk

Every fixture is synthetic and redistribution-safe. Locate identifiers are
simulation artifacts; real broker locate identifiers must never be committed.
Digests are identity and integrity evidence, not signatures, and carry no external
provenance guarantee. Nothing here establishes broker behaviour, queue position,
intraday liquidity, or a deployable AUM.
