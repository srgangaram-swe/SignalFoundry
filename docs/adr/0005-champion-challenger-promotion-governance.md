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

Gates are evaluated independently and all results are returned, so a reader sees every failure
rather than only the first. There is no weighted score: a challenger that wins decisively on Brier
but ran for six days does not average its way to a recommendation. A `PROMOTE` recommendation
carrying an unsatisfied gate is rejected in `Decision.__post_init__`, so the invalid state cannot
be constructed at all.

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
the evidence rather than on trust. The absolute-gate rule means some genuinely better challengers
wait longer than they need to, which is the intended trade: the cost of a delayed promotion is
bounded and the cost of an unjustified one is not.

Holm is more conservative than dependence-exploiting alternatives. That is deliberate — its
guarantee holds without assumptions we cannot verify about the correlation between metrics.

**Costs.** A 10,000-replicate block bootstrap makes a decision take seconds rather than
milliseconds; decisions are rare and reproducibility is worth more than latency here. Requiring
symmetric missingness will occasionally refuse a cohort that is in fact unbiased, and the honest
remedy is to fix collection rather than to relax the bound.

**Not addressed.** Rollback mechanics, the durable persistence of decisions and approvals, and the
lane-head record itself are out of scope for this ADR; this module verifies the head it is given.
Multi-arm comparison beyond a single champion-challenger pair is not supported.
