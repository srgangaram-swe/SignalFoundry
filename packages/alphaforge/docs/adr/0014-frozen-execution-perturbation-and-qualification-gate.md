# ADR 0014 — Frozen execution perturbation and the qualification gate

- **Status:** Accepted
- **Date:** 2026-08-02
- **Work items:** SF-S4-MR8 (#43), SF-S4-MR9 (#44), Signal Foundry Sprint 4
- **Builds on:** ADR 0010 (event sourcing and accounting), ADR 0012 (point-in-time borrow and capacity), ADR 0013 (causal regime labels and dependence-aware uncertainty)

## Context

Sprint 4 built portfolio construction, optimization, capacity, and robustness
machinery. Two gaps remained before a candidate could be considered for paper
trading.

First, every result rested on **one realized execution path**. Orders cleared in
one order, at one set of fills, with no rejections and no missed bars. Nothing
established that the result survived ordinary operational imperfection.

Second, there was **no decision procedure**. Without one, qualification happens
implicitly and after the fact: a reviewer looks at the numbers, forms an
impression, and reconstructs a standard the numbers happen to meet. That is not a
gate; it is a narrative.

## Decision

**Perturbation touches mechanics, never the P&L.** Order sequence, fill price,
signal timing, missed and delayed trades, size, cost, liquidity, and partial
fills are perturbed. Returns are never shocked directly, because that
manufactures the answer instead of stressing it.

**Every perturbation family is bounded and frozen** under a content-derived
SHA-256, verified before the study runs. Widening a distribution after seeing the
tails changes the digest.

**Seed streams are derived, not drawn.** `SeedSequence([root_seed, sha256(kind),
index])` makes any single path replayable from its coordinates and stops two arms
from sharing draws.

**Failures stay in the denominator.** Insolvent, no-trade, and unreconciled paths
are counted. Equity stops at zero on insolvency rather than going negative.

**Monte Carlo frequency is reported as model frequency**, never as a probability
of loss in the market, and the caveat travels on the record.

**The qualification rubric is frozen and versioned before scoring**, with every
criterion requiring an evidence link that pins a full content hash.

**Qualification fails closed and totally.** Any failed criterion, missing or
non-finite observation, missing or partial evidence, or ledger reconciliation
failure forces `REJECTED`. No weighted score, no override.

**The only verdicts are `QUALIFIED_FOR_PAPER` and `REJECTED`**, and the former
authorizes zero-capital paper evaluation only.

## Consequences

**Accepted costs.**

- An arm whose metric is mathematically invariant (`trade_order` on compounded
  return) must be documented and excluded by name from the two-sidedness guard,
  rather than quietly passing it on floating-point noise.
- Requiring content hashes on evidence links makes the API tedious to call by
  hand. That is deliberate: a link that does not pin contents lets the evidence
  change after the claim.
- Rejection is the common path. Sprint 4's own candidate is `REJECTED` with 7 of
  8 criteria blocking, and callers must handle that as the normal outcome.
- Each arm is stressed independently, so interaction effects are not modelled.

**What this does not buy.**

- The perturbation family is an assumption. A failure mode it does not describe
  is not covered by any number the study produces.
- A frozen rubric prevents moving the bar; it does not make the bar correct.
- Passing every criterion establishes that a candidate was not obviously broken.

## Alternatives considered

**Bootstrap the return series directly.** Rejected: resampling returns tests
sampling variability, not execution risk, and it silently perturbs the very
quantity the strategy is being credited with.

**Blend a fraction of each bar into the next for timing jitter.** Implemented
first, then rejected on evidence. Blending is a smoothing operator; smoothing a
fixed-sum path always lowers variance drag, so compounded return improved on
every replicate and the arm reported a 0.000 frequency of being at or below
baseline. A perturbation that can only help is not a stress. Replaced with
stale-signal substitution, which preserves scale and is genuinely two-sided.
`test_timing_jitter_can_hurt_as_well_as_help` is the regression guard.

**Drop the reordering arm once its net-return invariance was understood.**
Rejected: the arm is the single largest source of drawdown risk in the published
evidence (−52.3% worst versus −35.1% unperturbed). Removing it would have deleted
a real finding because one of its metrics was uninformative.

**A weighted qualification score with a passing threshold.** Rejected: a weighted
score lets a strong showing on cheap criteria offset a failure on an expensive
one, and the weights become the new place to put a thumb.

**An `override` or `force` parameter on `qualify`.** Rejected outright. Every
override mechanism is a mechanism for talking oneself into a trade.

**Let a missing observation default to passing.** Rejected: silence is not
evidence. An unmeasured criterion is recorded as unmeasured and blocks.
