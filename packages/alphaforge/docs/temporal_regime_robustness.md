# Temporal, regime, sector, and universe robustness (SF-S4-MR7)

Frozen calendar intervals, causal regime labels, point-in-time universe
membership, and evidence whose error bars account for the fact that returns are
not independent.

> **No qualification claim.** This MR supplies the machinery and its correctness
> evidence. Whether any candidate survives it is SF-S4-MR9's frozen decision.
> Simulation only — nothing here authorizes paper or live trading.

The design rationale and the alternatives that were rejected are in [ADR
0013](adr/0013-causal-regime-labels-and-dependence-aware-uncertainty.md).

---

## 1. The four ways a robustness check quietly becomes a fitting procedure

Each of the modules in this MR closes one of them.

| Failure | What it looks like | Where it is closed |
| --- | --- | --- |
| Choosing periods after the fact | "Excluding 2020, the strategy is stable" | `periods.py` — frozen identities |
| Defining a regime from its outcomes | "It only struggles in crisis regimes" (crisis = where it struggled) | `periods.py` — causal labels, independence check |
| Survivorship in the universe | Backtesting today's constituents over the last decade | `universe.py` — dated membership |
| Error bars that assume independence | A t-statistic of 3 on autocorrelated returns | `temporal_evidence.py` — block bootstrap, HAC |

## 2. Periods are frozen, and the freeze is checkable

`FrozenPeriodSet` publishes a **content-derived SHA-256** over its intervals, and
`verify_frozen_periods` refuses a set that differs from the identity recorded
before evaluation. Dropping one bad year from a calendar-year partition changes
the digest, so a post-hoc recut fails loudly rather than passing as "the
analysis".

`calendar_years` is the default cut for a reason: calendar years are declared by
the calendar rather than by anyone's judgement about what happened in them, so
they are the one partition that cannot be accused of hindsight.

**Boundaries are half-open `[start, end)`.** An observation on 2020-01-01 belongs
to 2020, not to both 2019 and 2020 and not to neither. This is the difference
between an evidence table that reconciles and one that double-counts a day at
every seam. A test asserts that boundary observations are counted exactly once,
and another asserts that per-interval counts plus unlabelled counts equal the
whole sample.

Overlap is refused by default. `allow_overlap=True` exists for a deliberate
overlay — a "high volatility" stress window sitting on top of a calendar-year
partition — and has to be asked for, because an accidental overlap silently
inflates the apparent sample.

## 3. A regime cannot be defined from the outcomes it explains

This is the issue's governing rule, and it is enforced in three places rather
than documented as a norm.

**The labeller cannot see the candidate.** `label_regimes(conditioning,
definition)` has one data parameter, and it is the conditioning series — a
benchmark, a volatility index, a macro series. There is no argument through which
the candidate's own returns could arrive.

**Near-duplicates are caught anyway.** A caller who builds the conditioning
series dynamically can still hand over something that is really the strategy's
P&L in disguise. `assert_conditioning_is_independent` refuses a series whose
correlation with the candidate's returns is indistinguishable from perfect. A
scaled and shifted copy of the returns is refused as readily as the returns
themselves.

**Cut points are expanding, not full-sample.** This is the subtle one. A trailing
volatility statistic *looks* causal, and it is — but taking its median over the
whole sample to split "calm" from "stormy" means every early label depends on how
the series ended. `label_regimes` computes each bar's cut points from the
strictly-prior statistic values only.

The test for this is a prefix-invariance check: labels for the first 800 bars
must be identical whether or not the remaining 400 exist. A companion test
demonstrates that a full-sample threshold **fails** that same check, so the
expanding-quantile choice cannot be "simplified" away later without a red test.

Warm-up bars carry no label. `warmup_policy` accepts only `"unknown"` — forward
filling a regime label into a window whose statistic is not yet defined invents
information, and the constructor refuses it.

`standard_regime_definitions()` supplies the regimes the work item names — bull,
bear, sideways, high/low volatility, crisis, recovery — as a **set** of three
frozen definitions rather than one. Supplying them together is the point: a
single definition is a choice, and offering three forces the sensitivity sweep in
§8 rather than allowing the one that reads best to be reported alone.

## 4. Families are compared on identical windows

Two families evaluated over different spans cannot be compared — the one that
happened to cover a calmer stretch looks better for a reason that has nothing to
do with the family. `matched_family_evidence` restricts every family to the
timestamps they *all* share and evaluates them on the same frozen intervals.

The match itself is reported: `coverage_sacrificed` names how many observations
each family gave up to reach the shared span, because a family that loses most of
its history to matching is being compared on a fragment and the reader should see
that rather than infer it. Families with no shared timestamps are refused
outright.

## 5. Universe membership is resolved as of a session

A membership table without dates silently asserts that today's constituents were
always the constituents. That deletes every delisting and every company that
failed, and the backtest then earns returns on securities it could not have held.

`PointInTimeUniverse` stores `MembershipRecord`s with an effective session and an
**exclusive** expiry session, so:

- a symbol is not investable before it lists,
- a delisted symbol stops being investable from its exit session forward,
- a symbol can leave and rejoin without its two intervals overlapping,
- and two overlapping intervals for one symbol are refused outright, because a
  name cannot be a member twice at once without being double-counted.

`assert_no_future_membership` is the check that catches survivorship at the point
it enters: hand it a panel built from a later constituent list and it names the
securities that had not listed yet, rather than letting them contribute returns.

Delistings are recorded separately from index removals because the consequences
differ — a delisted position must be liquidated at whatever the terminal price
was, while an index removal leaves a still-tradeable security.

## 6. Ablations: which ones are estimates and which are only probes

This distinction is load-bearing, so it travels with every result as
`AblationResult.hindsight_based` rather than living in prose.

**`drop_top_contributors` is a fragility probe, not a performance estimate.** You
can only know which names won after the fact, so the resulting number answers
"how much of this rested on a handful of names?" — never "what could have been
earned". Reporting a winners-removed return as an achievable return would be a
straightforwardly false claim, and the flag plus the `interpretation` string on
the record exist so a reader cannot mistake one for the other.

**`drop_sector`, `apply_liquidity_floor`, and `exclude_inactive` are genuine
estimates.** Sector membership, liquidity, and point-in-time membership are all
knowable in advance, so excluding on them describes a portfolio someone could
actually have run.

Two smaller decisions worth stating:

- A name with **missing** liquidity is removed, not retained. Treating an unknown
  as passing the floor is how an untradeable name keeps its contribution in an
  investability study.
- An **empty sector** returns an unchanged total rather than raising. A sector
  may legitimately have no members on a session, and refusing would make sector
  sweeps fail on exactly the sparse sessions worth examining.

Accounting reconciles exactly: removed plus retained equals the whole, and a test
asserts it. Ties in the top-contributor ranking break on symbol name so the
removal set does not depend on dictionary ordering.

`concentration_profile` reports the Herfindahl index and top-1/3/5 shares,
because a result carried by three names and one spread across the book have
identical point estimates and are not the same proposition.

## 7. Error bars that survive contact with financial data

The textbook standard error assumes independent observations. Returns are not
independent — volatility clusters, so adjacent bars carry overlapping
information, and an i.i.d. interval counts each one as fresh evidence. The
interval comes out too narrow, and an ordinary stretch of luck looks decisive.

Two dependence-aware estimators are provided, and `compare_uncertainty` reports
both **alongside** the naive one with the width-inflation ratio between them. The
ratio is the point of the exercise: it says how much of the naive interval's
tightness came from assuming independence rather than from the data. On an AR(1)
series with ρ = 0.85, both dependence-aware intervals come out more than 1.5×
wider; on genuinely independent returns all three agree, and tests assert both
directions.

- **`block_bootstrap_interval`** resamples contiguous blocks, so dependence
  inside a block survives the resample. `block_length` is a **required**
  argument. A block shorter than the true correlation horizon reproduces the
  i.i.d. interval's overconfidence while looking like it corrected for something,
  so defaulting it would be worse than asking.
- **`newey_west_standard_error`** widens the error analytically using
  autocovariances out to a declared lag, with Bartlett weights so the long-run
  variance cannot come out negative. The default lag is the standard
  `floor(4·(n/100)^(2/9))` rule, declared once rather than tuned per result.

Every `UncertaintyInterval` carries the `assumption` that produced it, because an
interval is only as good as the dependence structure it accounts for and a reader
comparing two of them needs to know which one assumed independence.

## 8. Failure periods are reported, never trimmed

`period_evidence` returns **every** declared interval:

- intervals where the candidate lost money, collected under `failure_periods`;
- intervals too sparse for an interval estimate, marked `sparse`, keeping their
  observation count, and carrying a note saying why no interval is given;
- intervals with **no** observations at all, reported as empty rather than
  omitted.

A short period declines an interval instead of receiving a misleadingly tight
i.i.d. one. Dropping short intervals is precisely how a strategy's worst stretch
leaves the record, and an evidence set containing only the good periods is not
evidence.

`regime_definition_sensitivity` runs the same reporting across several **frozen**
alternative definitions and flags whether the conclusion moved between them. A
conclusion that holds under one window and threshold set but not another is a
property of that choice, not of the market. All definitions are declared up
front; this function reports the spread, it does not pick the definition that
reads best.

## 9. No portfolio-level claim without a qualified candidate

Asset, sector, factor, concentration, inactive-security, universe, liquidity, and
top-contributor dependence are all statements *about a portfolio that is claimed
to work*. Emitting them for an unqualified candidate dresses an unproven result
in the apparatus of a proven one.

`portfolio_dependence_evidence` therefore raises `UnqualifiedCandidateError` when
`qualified is None`, rather than returning a hedged report. Sprint 4 has no
qualified candidate — qualification is SF-S4-MR9's decision — so **the refusal is
the expected path today**, and it is structural rather than a matter of
remembering not to make the claim.

`QualifiedCandidate` is deliberately harder to construct than a boolean flag. It
requires a named decision, the full SHA-256 of the frozen research plan the
decision was made under, and the corrected p-value it rested on — and it refuses
construction when that p-value exceeds alpha. Multiple-testing correction itself
stays where it belongs, in `alphaforge/research/governance.py`.

## 10. Evidence

`tests/test_temporal_regime_robustness.py` — 99 tests: calendar tiling and
boundary assignment; unlabelled observations not absorbed; overlap refusal and
declared overlays; recut detection and order-independent identities; regime-label
prefix invariance plus the companion test proving a full-sample threshold breaks
it, run against every definition in the standard set; warm-up labelling;
circular- and near-duplicate-conditioning refusals; point-in-time membership
across listing, delisting, and rejoin; the future-composition refusal;
overlapping-membership refusal; ablation accounting reconciliation, deterministic
tie-breaking, empty-sector and unknown-liquidity handling; concentration profiles
including the undefined-at-zero case; interval widening on autocorrelated returns
and agreement on independent ones; HAC monotonicity in persistence,
non-negativity under alternating signs, and reduction to the i.i.d. error at lag
0; bootstrap reproducibility and seed-sensitivity; short-sample, oversized-block,
replicate-ceiling, and lag-ceiling refusals; empty, sparse, and losing period
reporting; empty-regime handling; disjoint-index refusal; matched-window
evaluation, sacrificed-coverage accounting, gap handling, and the single-family
and disjoint-family refusals; missing-price accounting and an all-missing period
reported as empty rather than as a zero return; a bounded-cost guard on the
bootstrap; the unqualified-candidate refusal and the qualified-candidate unlock;
plan-hash and alpha validation; report determinism and JSON serializability; and
the contract refusals across every module.

Coverage: `universe.py` 92%, `periods.py` 91%, `temporal_evidence.py` 91%
(branch). Repository total 84.30% against an unchanged 78% floor, 1779 tests
passing.

One behaviour is pinned as a **documented abstention** rather than a refusal:
`assert_conditioning_is_independent` returns without raising when it cannot judge
— fewer than three shared observations, fewer than three after dropping missing
values, or a constant series on either side. Those cases carry no evidence of
circularity rather than evidence of its absence, and a constant conditioning
series cannot be a disguised copy of a varying candidate anyway. The screen
catches the dynamically-built accident; the structural separation in
`label_regimes`, which has no parameter for candidate returns, is what actually
enforces the rule.

## 11. Residual limitations

- **A frozen calendar does not make the sample representative.** Calendar years
  are unarguable as a partition, but a decade that contained one long bull market
  cannot tell you about the regime it never held.
- **Block length is a judgement.** The bootstrap accounts for dependence up to
  the declared block length and no further. A slow-moving dependence longer than
  the block is still uncounted, which is why the assumption is printed on every
  interval.
- **Regime sensitivity spans the definitions supplied.** Declaring three similar
  definitions and finding agreement is weaker evidence than it looks.
- **Sector labels are as good as their vendor.** A reclassification recorded on
  the wrong date reintroduces exactly the lookahead the membership dates prevent.
- **The universe module governs membership, not prices.** A survivorship-free
  constituent list paired with a price panel that silently backfills adjusted
  prices is still contaminated; that boundary belongs to the data platform.
