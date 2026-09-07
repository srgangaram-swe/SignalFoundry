# ADR 0015 — Broker selection for paper and live trading

- **Status:** Accepted
- **Date:** 2026-08-04
- **Work item:** SF-S5-MR2 (#99), Signal Foundry Sprint 5
- **Builds on:** ADR 0010 (event sourcing and accounting), ADR 0011 (frictions and latency), ADR 0014 (frozen perturbation and qualification gate)
- **Supersedes:** nothing

## Context

Sprint 4 ended with a working qualification gate and no qualified strategy. Broker
connectivity is on the critical path regardless: account approval, market-data
entitlement, and paper-environment access all have lead times measured in weeks,
and none of them depend on research outcomes. Serializing procurement behind a
qualified candidate would idle the calendar for no gain.

What forced this ADR is a narrower problem. Issues #45 (broker-neutral contract
and paper adapter) and #46 (idempotency, restart recovery, reconciliation) both
specify behaviour that only a *concrete* broker's semantics can pin down. "The
adapter must be idempotent" is not implementable without knowing whether the
broker accepts client-supplied order IDs, how long it deduplicates them, and what
it returns on a replayed submission. Writing those two integrations against an
imagined broker would produce an abstraction shaped by nothing.

**Criteria and weights below were fixed before any candidate was scored.** The
ordering matters: a rubric written after seeing the scores is a justification, not
a decision. This is the same discipline ADR 0014 applies to strategy
qualification, and it is applied here for the same reason.

## Decision

**Alpaca is selected as the first broker integration**, with Interactive Brokers
recorded as the deferred alternative for capabilities Alpaca cannot serve.

### Criteria and weights (frozen before scoring)

| # | Criterion | Weight | Why it carries this weight |
| --- | --- | --- | --- |
| 1 | Paper-environment fidelity | 20% | The entire Sprint 5 evidence chain runs on paper. A sandbox that diverges from production semantics invalidates every conclusion drawn from it. |
| 2 | Idempotency and reconciliation support | 20% | #46 is unimplementable without client order IDs and queryable authoritative state. A broker that cannot answer "what do you think my position is?" cannot be reconciled against. |
| 3 | API completeness and stability | 15% | Order lifecycle, versioning discipline, and breaking-change policy determine long-run maintenance cost. |
| 4 | Documented rate limits | 10% | Undocumented limits force empirical discovery against a live account — the worst possible place to learn them. |
| 5 | Market-data entitlement cost | 10% | Data cost dominates total cost at small account sizes and is frequently the binding constraint. |
| 6 | Order types and TIF coverage | 10% | Must cover what the strategy family needs; excess coverage is not a benefit. |
| 7 | Fee schedule and account minimum | 5% | Matters at the intended capital scale but is not a differentiator among zero-commission venues. |
| 8 | Short locate and fractional semantics | 5% | Relevant only if the qualified strategy shorts; unknown at selection time, so weighted low rather than assumed. |
| 9 | Uptime and incident transparency | 5% | A public status history is evidence; its absence is itself a signal. |

Regional eligibility is a **gate, not a weight**: a broker unavailable to the
owner's jurisdiction scores zero regardless of everything else.

### Scoring (public documentation only, 2026-08-04)

| Criterion | Weight | Alpaca | IBKR |
| --- | --- | --- | --- |
| Paper fidelity | 20% | 5 — dedicated paper endpoint, same API surface as live, separate credentials | 4 — paper account available; some order-type behaviour differs from live |
| Idempotency / reconciliation | 20% | 5 — `client_order_id` accepted and enforced unique; positions, orders, and account queryable | 4 — order IDs supported; reconciliation possible but the API is materially more complex |
| API completeness / stability | 15% | 4 — versioned REST plus streaming, narrow and well documented | 5 — far broader coverage, long-lived and stable |
| Documented rate limits | 10% | 5 — published per-minute limits | 3 — pacing rules documented but distributed across several places |
| Market-data cost | 10% | 5 — free IEX tier sufficient for daily-cadence research and paper | 3 — most useful entitlements are paid subscriptions |
| Order types / TIF | 10% | 4 — market, limit, stop, stop-limit, trailing; day/GTC/OPG/CLS/IOC/FOK | 5 — substantially broader, including algos |
| Fees / minimum | 5% | 5 — zero commission on US equities, no minimum | 4 — low but non-zero; tiered structure |
| Short locate / fractional | 5% | 4 — fractional supported; easy-to-borrow list exposed | 5 — deeper borrow inventory and locate detail |
| Uptime / incident transparency | 5% | 4 — public status page with incident history | 4 — public status reporting |
| **Weighted total** | | **4.60** | **4.05** |

Both clear the eligibility gate. Alpaca wins on the two heaviest criteria —
exactly the two that #45 and #46 depend on — and on data cost, which is the
binding constraint at the intended account size. IBKR is the stronger platform on
breadth, and that breadth is not what this sprint needs.

### What this decision does not do

Naming a broker in an ADR is a design record. It does not open an account, create
an entitlement, or authorize a trade. No credential, endpoint, or account
identifier enters the repository as a result of this decision.

### Revisit conditions

This decision is revisited, with a superseding ADR rather than an edit, when any
of the following becomes true:

1. The qualified strategy requires an asset class, venue, or order type Alpaca
   does not serve — most plausibly options, futures, or non-US equities.
2. The strategy requires short locates that Alpaca's easy-to-borrow inventory
   cannot supply at the intended size.
3. Alpaca's paper environment is observed to diverge from production semantics in
   a way that invalidates paper evidence. Divergence is grounds for revisiting
   selection, not for quietly discounting the evidence.
4. Market-data requirements exceed the free tier and the paid differential
   changes the cost ranking.
5. Regional eligibility, account terms, or the fee schedule changes materially.
6. A sustained reliability problem appears in the incident record.

## Consequences

**Accepted costs.**

- Narrower asset-class coverage than IBKR. Accepted because the current strategy
  family is US equities only; criterion 6 was weighted at 10% for this reason and
  not retrofitted afterward.
- Alpaca's borrow inventory is thinner. If the qualified candidate turns out to
  short meaningfully, revisit condition 2 fires — which is why that criterion is
  present at all despite the strategy being unknown.
- A single-broker integration risks coupling. Mitigated by #45's broker-neutral
  contract: the adapter implements an interface that does not name a vendor, and
  a second implementation is additive.

**What this buys.**

- #45 and #46 can be written against concrete, documented semantics rather than
  an imagined API.
- The free data tier means the paper evidence period costs nothing, so its
  duration is set by evidence quality rather than by budget.

## Alternatives considered

**Interactive Brokers first.** Rejected for this sprint. The stronger platform,
but the integration surface is materially larger and the data entitlements cost
money before any evidence exists. Recorded as the deferred alternative precisely
because several revisit conditions point at it.

**Defer broker selection until a strategy qualifies.** Rejected. Account approval
and entitlement lead times are weeks and are independent of research outcomes.
Serializing them behind qualification adds latency and buys nothing.

**Build a vendor-neutral adapter with no chosen broker.** Rejected as the
motivating failure. An abstraction with no concrete implementation behind it is
shaped by imagination; idempotency and reconciliation semantics in particular
cannot be specified without a real API's dedupe window and error taxonomy.

**Write the rubric after scoring candidates.** Never on the table, and recorded
here because the failure is common and invisible after the fact: a rubric fitted
to a preferred answer is indistinguishable from a real one in the final document.
The weights above were fixed first.
