# ADR 0005: Gate champion-challenger promotion behind absolute rules and human authority

- **Status:** Accepted
- **Date:** 2026-08-11
- **Issue:** [#22](https://github.com/srgangaram-swe/Signalattice/issues/22)

## Context

ADR 0001 committed this repository to probabilistic forecasts scored by strictly proper rules, and
SF-S5-SL-MR4 added delayed-outcome shadow campaigns so a model can accumulate an out-of-sample
record before it is trusted. That record is the input to a decision nobody has yet specified: when
does a challenger replace the champion?

Answering it badly is the failure mode this ADR exists to prevent. The specific ways a promotion
decision goes wrong are well understood and all of them produce a number that looks fine:

- the two arms are scored on different questions, so the difference describes the sample rather
  than the models;
- one arm is missing on exactly the days it would have scored badly, so every retained pair is
  individually valid and the aggregate is still biased;
- forecasts made on the same day are treated as independent, inflating the effective sample size
  and narrowing every interval until noise clears significance;
- several metrics are tested and the best one is reported, so the nominal alpha is not the actual
  false-positive rate;
- a non-significant result is read as evidence of equivalence;
- a threshold is adjusted after the results are seen;
- the system that computes the recommendation is also the system that applies it.

This work is local and offline. It is not a trading system, it does not place orders, and a
promotion here changes which model a local lane designates as champion. Nothing in this ADR is
evidence of profitability.

## Decision

### A lane is the unit of assignment, and its identity is derived

`quant_platform.governance.lane` keys a `GovernanceLane` by purpose, target,
horizon, frequency, universe contract, decision policy, and environment. The identity is the
SHA-256 of that canonical key rather than an assigned name, so two lanes differing in any component
are different lanes and cannot silently share a champion.

**Model state is never a mutable global.** One revision may be champion in one lane and a rejected
challenger in another, so "is this the champion?" is only answerable relative to a lane. A
`champion` flag on the revision would make the question look answerable when it is not.

Two state machines are declared as explicit transition tables rather than scattered conditionals,
and a test enumerates every ordered pair against the table rather than sampling remembered paths:

- `LaneState` is `UNASSIGNED`, `ACTIVE`, or `FROZEN`. The only permitted transitions are
  `UNASSIGNED → ACTIVE`, `ACTIVE → ACTIVE` (a new champion), `ACTIVE → FROZEN`, and
  `FROZEN → ACTIVE`. There is no automatic activation, failover, or unfreeze.
- `RequestState` runs `AWAITING_APPROVAL → APPROVED → APPLIED` with `REJECTED`, `WITHDRAWN`,
  `EXPIRED`, `STALE`, and `REVOKED` as exits. **Every terminal state is terminal**: a rejected
  request is never reopened and an applied one is never re-applied.

### History is append-only and the head is a projection

`quant_platform.governance.store` writes to `sl_governance_events`, which the database itself
refuses to update or delete via `BEFORE UPDATE`/`BEFORE DELETE` triggers. Each event carries its
payload digest, its predecessor's chain digest, and a chain digest over both plus the sequence and
kind — so an event cannot be moved, relabelled, or altered while keeping its links intact.
`verify_chain` reports the first sequence at which the chain fails to reproduce.

`sl_governance_lane_head` is a **cache**. `rebuild_head` verifies the chain, replays the events, and
compares the result against the stored row; a disagreement raises rather than resolving to either
value. The events are the authority, and a projection rebuilt from a broken chain would launder the
break into a plausible-looking head, so verification comes first.

A head update and its event are written in **one** `BEGIN IMMEDIATE` transaction. A head that moved
without recording why, and an event describing an assignment that never took effect, are both
states no reader could reconcile.

Idempotency keys are stored only as one-way digests, so replaying a request resolves to the
existing event while a reader of the table cannot reconstruct the key. Reusing a key for different
content is a conflict, not a silent no-op — the alternative discards the new payload.

### Freezing is automatic; clearing a freeze is not

Automation may freeze a lane: on a hard integrity breach immediately, or after a second consecutive
non-overlapping soft-breach window. Freezing preserves the champion, so a reader can still see
which model the lane was running when it stopped, and the trigger is recorded because it determines
what is required to clear it.

There is no `unfreeze` method. A frozen lane returns to `ACTIVE` only through a newly approved
assignment under compare-and-swap, which means the same human authority path as any promotion. A
freeze may *recommend* rollback; it cannot apply one.

### Comparison is exact, and every exclusion is reported

`quant_platform.governance.comparison` pairs forecasts only when campaign, symbol, as-of instant,
and horizon match exactly. Duplicate keys are refused rather than resolved, because a silent
last-writer-wins pick makes the pairing ambiguous. Unmatched rows are counted into
`champion_only`, `challenger_only`, `champion_unscored`, and `challenger_unscored` and travel with
the cohort, so a reader can see what fraction of the universe the comparison actually covers rather
than inferring it from an inner join.

Two further checks run at pairing time:

- **Outcome reconciliation.** If the two arms recorded different realised labels for the same
  question, that is a reconciliation fault, not a scoring difference, and it raises rather than
  resolving to either arm's view. One question has one answer.
- **Leakage re-verification.** Every paired forecast is re-checked to precede its own outcome on
  both arms. The shadow contracts already enforce this at construction; a cohort assembled from two
  independently sealed campaigns must not take that on trust, and a violation raises the distinct
  `LeakageError` because it invalidates the arm rather than reducing its coverage.

**Asymmetric missingness makes a cohort incomparable.** When per-arm missingness differs by more
than `MAX_MISSINGNESS_ASYMMETRY` (0.05), the cohort reports `comparable=False` with a stated
reason, and inference refuses to run on it. This is the case where every individual pair is
legitimate and the sample is still not answering the question asked.

### Inference is dependence-aware, one-sided where the question is, and family-corrected

`quant_platform.governance.inference` resamples **whole days**, not individual forecasts. Forecasts
made on the same day share market conditions; treating them as independent is what turns a
coin-flip challenger into a significant result. Fewer than `MIN_BLOCKS` (10) distinct days or
`MIN_EFFECTIVE_OBSERVATIONS` (50) pairs returns an `UNDERPOWERED` verdict with no p-value at all,
rather than a p-value that invites reading non-significance as equivalence.

The scheme is a **circular moving-block** bootstrap over dates with 10,000 replicates. *Moving
blocks* because the day-level series is itself serially dependent — a regime lasting a week
correlates consecutive days, and resampling days independently destroys exactly that structure.
*Circular* because a non-circular scheme can only start a block within the first `n - L + 1` dates,
under-sampling the end of the window, which is the most recent evidence and the part a promotion
leans on hardest.

Block length is `max(horizon, ceil(n ** (1/3)))` and is **derived, never supplied**: the horizon
because overlapping forecasts stay dependent at least that long, the cube root because that is the
rate at which block length must grow for the bootstrap to remain consistent. A caller who could
choose the length could choose the one producing the narrowest interval, so no function in the
module accepts it. A cohort mixing horizons is refused rather than averaged.

Superiority and non-inferiority are kept as separate questions with separate functions.
Non-inferiority requires a `Margin` declared in the frozen policy before the comparison runs; a
zero margin is refused because it is a superiority test wearing the wrong name. The one-sided
non-inferiority interval reports its unbounded end as `None` rather than an infinity, since
infinities are not JSON-representable and the decision record must serialise for its identity.

Bootstrap p-values are computed as `(exceedances + 1) / (replicates + 1)`, so they can never be
reported as exactly zero. A p-value of zero asserts impossibility, which a finite resample cannot
establish.

Multiplicity is corrected across the **complete** family with Holm-Bonferroni, which controls the
familywise error rate under arbitrary dependence. That property is required rather than
convenient: Brier and log score on the same cohort are strongly correlated, and an
independence-assuming correction would be anticonservative exactly when it matters. Correcting a
subset chosen after seeing results is not a correction, so an empty or partial family raises.

### Gates are absolute; there is no override

`FrozenPolicy` records alpha, minimum days, minimum pairs, minimum coverage, and the margin, and
carries a content identity. `recommend` accepts an `expected_policy_identity` and refuses when the
running policy does not match it, which is how a threshold edited after seeing results is detected.

The preregistered operational floor is a set of module constants, not parameters with permissive
defaults — a floor a caller can lower at the call site is not a floor. A policy may declare a
stricter value and its identity then records that it did. The floor is 28 consecutive calendar
days, 20 resolved target dates, 200 exactly paired rows, 50 observations in the least-observed
class, 0.99 coverage, and 0.80 preregistered power.

Calendar span and distinct target dates are gated **separately**, because they answer different
questions: 20 forecasts on 20 consecutive days and 20 spread across a year both yield 20 distinct
dates, but only one describes a single regime.

**Clearing the floor never establishes superiority.** It means the evidence is sufficient to ask the
question, not that the answer is favourable.

Gates are evaluated independently and all results are returned, so a reader sees every failure
rather than only the first. There is no weighted score: a challenger that wins decisively on Brier
but ran for six days does not average its way to a recommendation. A `PROMOTE` recommendation
carrying an unsatisfied gate is rejected in `Decision.__post_init__`, so the invalid state cannot
be constructed at all.

**An unevaluated gate is not a satisfied gate.** Omitting the class counts or the power figure fails
those gates rather than skipping them, so a caller cannot obtain a favourable result by supplying
less.

**There is no `force`, `waive`, `skip`, `override`, `bypass`, or `unsafe` parameter anywhere in the
governance modules, and a test proves it by parsing the module AST rather than by reading the
code.** Retiring a gate requires publishing a new frozen policy version, which leaves a record.

Failures are distinguished rather than collapsed: an incomparable cohort yields `INVALID`, while
insufficient duration, pairs, coverage, or power yields `INSUFFICIENT_EVIDENCE`. Neither is
`RETAIN_CHAMPION`, because "we could not tell" and "we checked and the champion is better" are
different findings.

### Only a human can approve, and only once, and only for this decision

`recommend` produces a `Decision` and cannot do anything else — it has no write path, and a test
asserts the absence of applying verbs in its source. Applying requires an `Approval` carrying a
named approver bound to `Decision.identity`, which prevents recycling a sign-off onto a later
comparison that produced a different result. Approvals expire after `APPROVAL_VALIDITY` (7 days),
because evidence moves and a month-old sign-off approved a comparison that no longer describes the
lane.

`authorize_apply` additionally performs a compare-and-swap against the lane head. Two approvals
racing to promote different challengers cannot both succeed: the second sees a head that no longer
matches what it was approved against and must be re-evaluated against the new champion. Every
refusal raises `NotAuthorizedError`, a distinct type so generic retry handling cannot loop past it.

A recommendation is not an authorization. Nothing in this package can approve, apply, roll back, or
unfreeze.

## Consequences

**Accepted.** Promotion decisions are reproducible from their inputs: policy identity, cohort
identity, and a seeded bootstrap fix the result. A decision record states every gate, every test,
the correction applied, and what was excluded, so a reviewer can disagree with the conclusion on
the evidence rather than on trust. Governance history is append-only and chain-verified, so a lane
can be audited after the fact and a divergence is reported with the sequence at which it occurred.

The absolute-gate rule means some genuinely better challengers wait longer than they need to, which
is the intended trade: the cost of a delayed promotion is bounded and the cost of an unjustified one
is not. Holm is more conservative than dependence-exploiting alternatives; its guarantee holds
without assumptions we cannot verify about the correlation between metrics.

**Costs.** A 10,000-replicate circular block bootstrap makes a decision take seconds rather than
milliseconds; decisions are rare and reproducibility is worth more than latency. Requiring symmetric
missingness will occasionally refuse a cohort that is in fact unbiased, and the honest remedy is to
fix collection rather than relax the bound. The 28-day floor means Sprint 5's deterministic replay
proves the mechanics while honestly returning `INSUFFICIENT_EVIDENCE`; the wall-clock campaign is
tracked separately in #63.

## Residual risk

These are the things this design does **not** protect against, stated so nobody reads the chain as
more than it is.

- **The hash chain is not externally tamper-proof.** Anyone who can rewrite a row can recompute the
  rest of the chain. It detects accidental divergence, partial writes, and careless edits. External
  anchoring and signing belong to the release work in #23.
- **The approver field is an owner assertion, not an identity.** There is no cryptographic
  authentication, no independent validation, and no separation of duties. A single local operator
  is both the requester and the approver, and the record says who *claimed* to approve.
- **Storage rollback is only partly visible.** Restoring an older database file wholesale leaves a
  self-consistent chain that verifies. What is detected is divergence between the projection and the
  events, and breaks *within* a chain — not the substitution of an older intact one.
- **Clock trust.** Approval validity is checked against a supplied instant. A rolled-back clock is
  refused when it falls outside the approval window in either direction, but the store does not have
  an independent time source.
- **The operational floor is preregistered, not derived from this system's own power.** The 0.80
  power figure is supplied by the caller and gated, not computed here; a caller that supplies a
  wrong figure gets a gate that passes on a wrong premise.
- **Local scope only.** There is no remote authorization, no multi-user identity, and no HTTP
  mutation route. Nothing in this package places an order, allocates capital, or authorizes paper or
  live trading, and none of it is evidence of profitability or production readiness.

## Rollback and containment

Freezing the lane through a new event disables further application. All policies, comparisons,
approvals, assignments, and monitoring evidence are retained — governance records are
retention-ineligible, because expiring them destroys the audit trail the lane exists to produce.

Restoring a prior verified champion is a **new approved assignment** under compare-and-swap, never a
deletion, rewrite, down-migration, or silent repair of history. There is no broker, position, order,
or capital state to unwind.

## Not addressed

Durable persistence of the request and approval *lifecycle* as first-class rows (they are currently
recorded as chain events rather than queryable request tables), multi-arm comparison beyond a single
champion-challenger pair, and the wall-clock shadow campaign that would let the 28-day floor be
cleared with real elapsed evidence.
