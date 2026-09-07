# ADR 0013 — Causal regime labels and dependence-aware uncertainty

- **Status:** Accepted
- **Date:** 2026-08-01
- **Work item:** SF-S4-MR7 (#42), Signal Foundry Sprint 4
- **Builds on:** ADR 0007 (fail-closed decision eligibility), ADR 0012 (point-in-time borrow and capacity policy)

## Context

SF-S4-MR6 froze the parameter grid and supplied negative controls. That answers
"is the effect robust to how it was configured?" It does not answer "is the
effect robust to *when* it was measured, *under what conditions*, and *on which
securities*?" — and those three questions have their own well-documented ways of
being answered dishonestly without anyone intending to.

Three concrete failures motivated this ADR:

1. **Circular regimes.** Labelling a stretch a "crisis" because the candidate
   lost money there, then reporting that the candidate is robust outside crises,
   explains an outcome with a label derived from that outcome.
2. **Survivorship.** A membership table with no dates asserts that today's
   constituents were always the constituents, so the backtest earns returns on
   securities it could not have held.
3. **Overconfident error bars.** Returns are autocorrelated. An i.i.d. standard
   error counts each overlapping bar as fresh evidence and produces an interval
   narrow enough to make an ordinary stretch of luck look decisive.

Each was addressable by convention. None of them stays addressed that way.

## Decision

**Regime labels are causal, and cut points are expanding rather than
full-sample.** `label_regimes` computes each bar's quantile thresholds from
strictly-prior statistic values. A trailing statistic under a full-sample
threshold *looks* causal while every early label depends on how the series ended;
we treat that as a leak, not a rounding of one.

**The labeller has no parameter through which the candidate can arrive**, and
`assert_conditioning_is_independent` refuses a conditioning series that is a
near-duplicate of the candidate's returns. Structural separation first, runtime
check as the backstop for dynamically built series.

That runtime check **abstains when it cannot judge** — too few shared
observations, or a constant series on either side — rather than refusing. This
is the one place in the module that does not fail closed, and it is deliberate:
those cases carry no evidence of circularity rather than evidence of its
absence, a constant conditioning series cannot be a disguised copy of a varying
candidate, and refusing would make the guard unusable on short samples. The
enforcement that matters is structural, so the screen is permitted to be a
screen. Tests pin each abstention so it stays deliberate.

**Calendar intervals and regime definitions carry content-derived identities**
and are verified against the identity frozen before evaluation. A post-hoc recut
changes the digest and fails.

**Intervals are half-open `[start, end)`.** A boundary observation joins the
interval that starts there — counted exactly once, never dropped.

**Universe membership is dated, with an exclusive expiry session.** Overlapping
intervals for one symbol are refused. Delistings are recorded distinctly from
index removals because their consequences differ.

**Ablations declare whether they are hindsight-based.** `drop_top_contributors`
is a fragility probe and says so on its own record; sector, liquidity, and
inactive-security exclusions are genuine estimates because their criteria are
knowable in advance.

**Uncertainty is reported by three estimators side by side** — i.i.d.,
Newey-West HAC, and moving-block bootstrap — with the width-inflation ratio
between them. `block_length` is required, not defaulted: a block shorter than the
true correlation horizon reproduces the i.i.d. interval's overconfidence while
appearing to have corrected for something.

**Every declared period is reported**, including losing, sparse, and empty ones.
A period too short for a dependence-aware interval receives **no** interval
rather than a misleadingly tight i.i.d. one.

**Portfolio-level dependence claims require a `QualifiedCandidate`**, which
demands a named decision, the frozen plan's full SHA-256, and a corrected p-value
at or below alpha — and refuses construction otherwise. Absent one,
`portfolio_dependence_evidence` raises.

## Consequences

**Accepted costs.**

- Expanding quantiles cost a warm-up window at the start of every conditioning
  series, and those bars carry no regime label at all. Reported as unlabelled,
  never filled.
- Requiring `block_length` makes the API less convenient. This is deliberate: the
  convenient default is the one that silently under-corrects.
- The unqualified-candidate refusal is the *expected* path for all of Sprint 4.
  Callers must handle an exception on the normal route until SF-S4-MR9 issues a
  qualification.
- Three intervals per estimate is more output than one. The width-inflation ratio
  is the reason — a single number hides how much of its own precision was assumed.

**What this does not buy.**

- A frozen calendar does not make a sample representative. A decade that held one
  long bull market cannot speak to the regime it never contained.
- The bootstrap accounts for dependence up to the declared block length and no
  further.
- Membership dates govern constituents, not prices. A survivorship-free
  constituent list paired with a silently backfilled price panel is still
  contaminated; that boundary belongs to the data platform.

## Alternatives considered

**Fit a regime model on the evaluation window** (HMM, changepoint — both already
exist in `alphaforge/regimes/`). Rejected here: a model estimated on the window
places its boundaries where they best explain that window, which is the exact
circularity this ADR exists to prevent. Those models remain available for
*forecasting* regimes; they are not admissible as *evaluation* partitions.

**Full-sample quantile cut points.** Rejected: prefix labels shift when later
data arrives. A test demonstrates the shift so the choice cannot be quietly
reverted as a simplification.

**Report only the dependence-aware interval.** Rejected: without the i.i.d.
baseline beside it, a reader cannot see how much of the usual interval's
tightness was an assumption.

**A boolean `qualified=True` flag.** Rejected: a boolean is trivially passed by a
caller who has not qualified anything. Requiring a decision identifier, plan
hash, and corrected p-value makes the claim expensive to fabricate and auditable
after the fact.

**Drop sparse periods.** Rejected: dropping short intervals is how a strategy's
worst stretch leaves the record. They are reported with their counts and without
an interval estimate.
