# Broker connectivity requirements (SF-S5-MR2)

What a broker must provide before AlphaForge can paper-trade against it, and the
objectively checkable gate that stands between paper evidence and a live order.

> **No connection exists.** This document adds no broker client, credential,
> endpoint, or network call. It states requirements so that [#45](https://github.com/srgangaram-swe/AlphaForge/issues/45)
> (contract and paper adapter) and [#46](https://github.com/srgangaram-swe/AlphaForge/issues/46)
> (idempotency, restart recovery, reconciliation) can be implemented without
> further interpretation. The broker selection and its frozen rubric are in
> [ADR 0015](adr/0015-broker-selection-for-paper-and-live-trading.md).

---

## 1. Capability requirements matrix

What the **current** strategy family actually needs — daily-cadence, long/short
US equities, rebalanced on a scheduled decision — against what each candidate
supplies. "Needed" reflects the strategy as it exists today, not a strategy
someone might write later.

| Capability | Needed | Alpaca | IBKR | Gap status |
| --- | --- | --- | --- | --- |
| Market order | Yes | Yes | Yes | None |
| Limit order | Yes | Yes | Yes | None |
| Stop / stop-limit | No — risk limits act on positions, not resting stops | Yes | Yes | None (surplus) |
| Trailing stop | No | Yes | Yes | None (surplus) |
| TIF: `day` | Yes | Yes | Yes | None |
| TIF: `gtc` | Yes | Yes | Yes | None |
| TIF: `opg` / `cls` | Preferred — decisions are made on close, executed on open | Yes | Yes | None |
| TIF: `ioc` / `fok` | No | Yes | Yes | None (surplus) |
| Fractional shares | Preferred — improves weight fidelity at small account sizes | Yes | Yes | None |
| Short selling | Conditional — unknown until a candidate qualifies | Easy-to-borrow list exposed | Deeper inventory | **Workaround-available**: restrict to easy-to-borrow, or fall back to long-only. Escalates to blocking only if a qualified candidate requires hard-to-borrow names. |
| Locate request | No — hard-to-borrow shorting is out of scope | Not offered | Offered | **Non-blocking by scope.** If scope changes, this is blocking and ADR 0015 revisit condition 2 fires. |
| Position query | **Yes — required** | Yes | Yes | None |
| Order-status query | **Yes — required** | Yes | Yes | None |
| Account/equity query | **Yes — required** | Yes | Yes | None |
| Activity / fill history | **Yes — required for reconciliation** | Yes | Yes | None |
| Client-supplied order ID | **Yes — required for idempotency** | `client_order_id` | Yes | None |
| Market clock / calendar | **Yes — required** | Yes | Yes | None |
| Corporate-action notifications | Preferred | Partial | Yes | **Workaround-available**: reconcile position deltas against Signalattice corporate-action records rather than trusting broker notifications. |
| Streaming order updates | Preferred, not required | Yes | Yes | None. Polling is acceptable at daily cadence; streaming is an optimization. |

### What the system must NOT require

Stated explicitly so an over-specified integration is not built:

- **Intraday routing control, venue selection, or queue position.** Decisions are
  daily; sub-second placement is not part of the thesis and modelling it would be
  fiction at this cadence.
- **Sub-second market data.** Daily bars plus a quote at decision time suffice.
- **Options, futures, crypto, or non-US equities.** Out of scope; adding them
  fires ADR 0015 revisit condition 1.
- **Margin beyond Reg-T overnight.** Leverage is bounded by portfolio policy well
  below any broker-imposed limit.
- **Algorithmic order types** (VWAP, TWAP, implementation shortfall). Order sizes
  at the intended capital scale are far below the threshold where they matter.

## 2. Non-code prerequisites

Every item is owner-executed; none can be automated, and several have lead times
that dominate the critical path.

| # | Prerequisite | Owner | Depends on | Lead time |
| --- | --- | --- | --- | --- |
| P1 | Brokerage account application and identity verification | Owner | — | 1–5 business days |
| P2 | Account type decision: cash vs margin | Owner | P1 | Same day, at application |
| P3 | Paper-environment credentials | Owner | P1 | Immediate after P1 |
| P4 | Market-data entitlement for research/paper (free tier) | Owner | P1 | Immediate |
| P5 | Market-data entitlement for live, if the free tier proves insufficient | Owner | P4 + measured need | 1–3 business days |
| P6 | Keychain entries for paper credentials | Owner | P3 | Minutes |
| P7 | Funding for live evaluation | Owner | P1, and the §5 gate | 1–5 business days |
| P8 | Tax-reporting readiness for a taxable account | Owner | P1 | Before first live fill |

**Critical path: P1 → P3 → P6.** Everything else parallelizes. Paper trading can
begin once P6 completes; P7 and P8 are live-only and blocked behind §5 regardless
of when they finish.

### Account type: cash vs margin

The choice is consequential and belongs at application time.

- A **cash account** avoids pattern-day-trader (PDT) rules entirely but settles
  T+1, so proceeds are not immediately redeployable. For a daily-rebalanced
  strategy this can leave capital idle for a day after each sell.
- A **margin account** under $25,000 is subject to PDT: four or more day trades in
  five business days triggers restriction. A daily-cadence strategy holding
  overnight does **not** normally day-trade, but a same-day entry and exit — which
  a stop or a reversal signal can produce — counts.

**Requirement:** whichever is chosen, the system must track day-trade count
against the PDT threshold and refuse an order that would breach it. Discovering
the restriction from a broker rejection is not acceptable; the constraint is
knowable in advance and must be enforced locally.

### Market-data entitlements by environment

| Environment | Requirement | Justification |
| --- | --- | --- |
| Research | Historical daily bars | Signalattice supplies these; the broker is not the research data source |
| Paper | Delayed or IEX-tier real-time quotes | Sufficient at daily cadence; free |
| Live | Same as paper unless slippage measurement shows the quote source materially misstates fills | Upgrade only on evidence, not on principle |

### Credential custody

- Every secret lives in **macOS Keychain** under a `com.signal-foundry.*` service
  name, retrieved only for the lifetime of the bounded process that needs it.
- **Nothing enters the repository**: no API key, secret key, token, account
  number, account identifier, or endpoint URL containing an identifier. This
  includes environment templates, example configs, test fixtures, CI
  configuration, logs, error messages, issues, and PR descriptions.
- Paper and live credentials are **separate entries** and never interchangeable.
  A configuration that could reach live with paper credentials, or the reverse,
  is a defect.
- **Rotation:** immediately on suspected exposure; otherwise every 90 days while
  live trading is enabled. Rotation is manual and recorded.
- **Redaction:** any log line, exception, or diagnostic that could carry a
  credential or account identifier is redacted at the boundary that produces it,
  not filtered downstream — a filter that must be remembered will eventually be
  forgotten.

This restates and does not weaken the existing policy in
[release security](release_security.md).

## 3. Operational and failure requirements

Required behaviour for #45 and #46. Every row is a requirement on the system, not
a description of the broker.

| Condition | Required system behaviour |
| --- | --- |
| **Rate-limit exhaustion** | Respect published limits proactively via a local token bucket. On a 429, back off exponentially with jitter, bounded by a maximum attempt count and total wall-clock budget. Exhausting the budget is a **terminal** failure that halts the decision cycle — never an unbounded retry. |
| **Partial fill** | Treat as a first-class terminal state, not an anomaly. Record filled quantity and average price; reconcile the residual against intent. Never assume the remainder will fill. |
| **Rejected order** | Classify as retryable (transient venue condition) or terminal (insufficient buying power, unshortable, PDT breach, malformed). Terminal rejections halt the cycle and surface the broker's reason verbatim. Never silently re-submit a terminal rejection. |
| **Duplicate submission** | Every order carries a deterministic client order ID derived from `(strategy_id, decision_timestamp, symbol, side, sequence)`. Re-submitting an identical intent must be a no-op at the broker. A duplicate acknowledgement is a **success**, not an error. |
| **Connection loss mid-order** | Assume the order **may** have been accepted. Never blind-retry. On reconnect, query by client order ID to establish authoritative state before any further action. |
| **Stale market data** | Every quote carries an observation timestamp. Data older than a declared staleness bound blocks order submission. Absence of a timestamp is treated as stale — an unknown age is not a fresh one. |
| **Clock skew** | Compare local clock against broker server time each cycle. Skew beyond a declared bound halts trading: order timestamps, TIF, and market-hours logic all become unreliable. |
| **Broker outage** | Halt and hold. Never queue orders locally for later replay — a queued order that fires after an outage executes against a market that has moved. On recovery, reconcile before resuming. |
| **Market closed** | Consult the broker's calendar and clock rather than inferring from local time; holidays and early closes are not derivable from a weekday check. |
| **Authentication failure** | Terminal. Halt immediately, redact the credential from all diagnostics, and require operator intervention. Never retry with an alternate credential. |

### Reconciliation requirements

| What | Against | Cadence | Divergence action |
| --- | --- | --- | --- |
| Position quantity per symbol | Local ledger vs broker | Before and after every decision cycle | **Halt.** Any non-zero difference. |
| Cash / buying power | Local ledger vs broker | Before and after every cycle | **Halt** beyond a declared rounding tolerance. |
| Open order set | Local ledger vs broker | Before every cycle | **Halt** on any order known to one side only. |
| Fill history | Local ledger vs broker activity | Daily | **Halt** on any fill absent locally. |
| Account equity | Local computation vs broker | Daily | **Warn** beyond tolerance, **halt** beyond a wider bound; equity legitimately differs by accrual timing. |

**Halt means halt**: stop submitting, preserve state for inspection, require
explicit operator action to resume. It does not mean retry, and it does not mean
liquidate — an automatic liquidation on a reconciliation error would act on
exactly the state that is known to be wrong.

### Idempotency requirements (feeding #46)

1. Client order IDs are **deterministic** and derived from decision content, so
   the same decision replayed produces the same ID.
2. The broker must **reject or return the existing order** for a duplicate ID
   rather than creating a second order.
3. The system must tolerate the broker's dedupe window being **shorter** than its
   own retention: after the window, the local ledger is authoritative for
   "was this already sent?" and must be consulted before resubmission.
4. Restart recovery must reconstruct in-flight order state from persisted local
   records plus a broker query — never from memory alone, and never from the
   broker alone.

## 4. Trust boundary

The broker is an **external, adversarial-until-validated** service, consistent
with the workspace security posture. Every response is untrusted input: field
presence, types, enum values, numeric finiteness, quantity signs, and timestamp
sanity are validated before any value reaches accounting. A well-formed JSON body
is not a semantically valid one, and a broker that reports a position AlphaForge
never opened is a reconciliation halt, not a position.

## 5. Capital-authorization gate

Every item is objectively verifiable. **Absence of any single item blocks live
enablement.** This is a gate, not guidance, and it has no weighted score and no
override.

| # | Item | Verification | Status (2026-08-04) |
| --- | --- | --- | --- |
| G1 | A strategy holds a `QUALIFIED_FOR_PAPER` verdict under a frozen rubric | Qualification dossier with rubric identity and evidence hashes | ❌ **Sprint 4 verdict is REJECTED** (7 of 8 criteria blocking) |
| G2 | Paper trading has run for a declared minimum duration | Paper run records covering the interval, no gaps | ❌ Not started |
| G3 | Paper results reconcile against broker state with zero unexplained divergence | Reconciliation log across the paper period | ❌ Not started |
| G4 | An operational rehearsal has been performed, including a deliberate failure drill | Rehearsal record with injected failure and observed recovery | ❌ Not started |
| G5 | The kill switch has been verified to stop submission within a declared bound | Kill-switch test record with measured latency | ❌ Not started |
| G6 | A capital-at-risk cap is set and enforced in code | Configuration plus the test proving breach is refused | ❌ Not implemented |
| G7 | Position, notional, and loss limits are set and enforced in code | Configuration plus enforcement tests | ❌ Not implemented |
| G8 | A rollback procedure exists and has been exercised | Runbook plus an execution record | ❌ Not written |
| G9 | Recorded owner approval, naming the cap and the date | Written approval referencing G1–G8 | ❌ Not given |

**Current state: 0 of 9 satisfied. Live trading is blocked, and G1 is blocked by
research outcomes rather than by engineering.**

The gate is deliberately ordered so G1 cannot be satisfied by effort alone. No
amount of infrastructure produces a qualified strategy, and the sprint that builds
the infrastructure must not be able to talk itself into believing otherwise.

## 6. Residual risks

- **Paper fidelity is an assumption until measured.** A paper environment models
  fills; it does not experience queue position, partial-fill dynamics under real
  contention, or borrow scarcity. Paper evidence bounds *operational* readiness,
  not executable performance — a distinction #45's non-goals state directly.
- **Selection rests on public documentation.** Documented behaviour and observed
  behaviour diverge. First contact with the paper environment is itself evidence
  and may fire a revisit condition.
- **Single-broker coupling.** Mitigated by the broker-neutral contract in #45, but
  a contract shaped by one implementation carries that implementation's
  assumptions. A second adapter is the only real test of neutrality.
- **Free market data has a quality floor.** IEX-tier quotes cover a fraction of
  consolidated volume; at daily cadence this is acceptable, and if slippage
  measurement contradicts that, the entitlement decision reopens.

## 7. Related work

- [ADR 0015](adr/0015-broker-selection-for-paper-and-live-trading.md) — selection rubric, scoring, and revisit conditions
- [#45](https://github.com/srgangaram-swe/AlphaForge/issues/45) — broker-neutral contract and paper adapter (SF-S5-MR3)
- [#46](https://github.com/srgangaram-swe/AlphaForge/issues/46) — idempotency, restart recovery, reconciliation (SF-S5-MR4)
- [#49](https://github.com/srgangaram-swe/AlphaForge/issues/49) — live-readiness framework (SF-S5-MR10)
- [Perturbation and qualification](perturbation_and_qualification.md) — the gate that must produce G1
- [Release security](release_security.md) — credential and artifact policy this document restates
