# Execution perturbation and strategy qualification (SF-S4-MR8/MR9)

The last two slices of Sprint 4: stress the execution path a strategy does not
control, then decide — against a rubric frozen beforehand — whether the candidate
has earned **paper** trading.

> **Nothing here is qualified.** Sprint 4's synthetic candidate is `REJECTED`,
> and that is the correct outcome. Simulation only; no capital has been at risk.

The design rationale and rejected alternatives are in [ADR
0014](adr/0014-frozen-execution-perturbation-and-qualification-gate.md).

---

## 1. Why perturb the path at all

A backtest reports one history. That history is a single draw from a process that
could plausibly have gone otherwise: the same signal reaching the desk a bar
late, fills landing slightly worse, one order rejected, a day of prices missing.
A result that survives only the exact sequence that happened is a description of
that sequence, not of a strategy.

`alphaforge/robustness/perturbation.py` perturbs the mechanics and reports the
**distribution** of outcomes.

## 2. What may be perturbed, and what may not

Perturbation applies to execution mechanics only — order sequence, fill price,
signal timing, missed and delayed trades, size, cost, liquidity, partial fills.
It never touches the returns directly. Perturbing the P&L would manufacture the
answer rather than stress it.

Every family is **bounded and declared before the run**, published under a
content-derived SHA-256. `verify_frozen_perturbations` refuses a grid that
differs from the frozen one, so widening a distribution after seeing the tails is
detectable rather than silent.

## 3. Three modelling defects this MR found and fixed

These are recorded because each one produced a plausible-looking number that was
wrong, and the tests that now pin them are the reason they cannot come back.

**A timing arm that could only help.** The first model blended a fraction of each
bar's return into the next. That is a *smoothing* operator, and smoothing a
fixed-sum path always lowers variance drag — so compounded return improved on
every single replicate, and the arm reported a 0.000 frequency of being at or
below the baseline. A perturbation that cannot hurt is not a stress. The arm is
now **stale-signal substitution**: a mistimed bar earns the *previous* bar's
outcome at full scale, which is genuinely two-sided.
`test_timing_jitter_can_hurt_as_well_as_help` and
`test_no_perturbation_arm_is_silently_one_directional` fail if any arm becomes
one-directional again.

**A boundary leak.** The same blend dropped the final bar's carried fraction off
the end of the array, silently deleting that return from every replicate and
biasing the study opposite the last bar's sign. On a losing path it made the
perturbation look free.

**A vacuous arm.** `trade_order` permutes realized bar outcomes, and compounded
return is *invariant* under permutation — multiplication commutes. The arm was
reporting a net-return distribution that was floating-point noise. It is retained
because it stresses **path-dependent** quantities, and drawdown is strongly
order-dependent even when the total is not: on the published evidence, reordering
alone deepens worst drawdown from −35.1% to −52.3%. The invariance is now pinned
by `test_reordering_conserves_compounded_return_exactly` and the arm is excluded
by name from the two-sidedness guard rather than passing it by luck.

## 4. Failures stay in the denominator

Insolvent, no-trade, and unreconciled paths are **counted, not dropped**. A study
that discards the paths where the strategy blew up reports the distribution
conditional on survival, which is the number that flatters.

Equity compounds from 1.0 and an insolvent path stops at zero — a strategy cannot
lose more than everything and then recover, and letting equity go negative would
let a blown-up path contribute a positive average. Insolvency requires a bar
worse than −100%, which is reachable with leverage; a −90% bar is a catastrophe,
not a wipe-out, and the two are not conflated.

## 5. Frequency is not probability

A Monte Carlo frequency states how often *this declared perturbation family*
produced an outcome at least this bad. It is not the probability of that outcome
in the market, because the family is an assumption. The caveat travels on every
`PerturbationOutcome` record rather than living in prose.

## 6. Exact replay

Seed streams are derived from `(root_seed, sha256(kind), replicate_index)` rather
than drawn from a shared advancing generator. Any single path in a 2,000-path
study replays exactly from its coordinates, without re-running the study or
knowing what any other arm drew, and no two arms can share draws —
`assert_perturbation_streams_isolated` refuses a grid where they would.

## 7. The qualification gate

`alphaforge/research/qualification.py` exists to prevent the most expensive
failure in quantitative research: deciding what "good enough" means *after*
seeing how good the result was.

**The rubric is frozen and versioned before scoring.** Lowering a threshold to
admit a candidate changes the digest, and `qualify` refuses to score against a
rubric that has moved.

**Every claim carries an evidence link.** A criterion without an `EvidenceLink`
pinning a full content hash cannot pass. An unevidenced metric is a *failure*,
not a weaker pass — and partial evidence is not a partial pass.

**Failure is closed and total.** Any failed criterion, missing observation,
non-finite value, missing evidence, or reconciliation failure forces `REJECTED`.
There is no weighted score, no "mostly passed", and no override parameter,
because each of those is a mechanism for talking oneself into a trade. Seven of
eight criteria satisfied is a rejection.

Ledger reconciliation failure rejects **regardless of every other number**: an
account that does not balance makes all of them meaningless.

## 8. The two verdicts

`QUALIFIED_FOR_PAPER` authorizes **zero-capital paper evaluation only**. It is
not authorization for live capital, not broker access, and not a statement that
the strategy will be profitable. `REJECTED` authorizes nothing.

The dossier renders failures **before** passes, so a reader who stops after the
first screen sees what is wrong with the candidate rather than what is right.

## 9. Sprint 4's actual result

The committed evidence bundle lives in
[`docs/evidence/signal_foundry_sprint_4/mr8_mr9_closeout/`](evidence/signal_foundry_sprint_4/mr8_mr9_closeout/)
and is regenerated by `scripts/publish_sprint_4_evidence.py`.

**Verdict: `REJECTED`, with 7 of 8 criteria blocking.** The synthetic development
path lost 19.7% net over 756 bars, and four criteria — multiple-testing
correction, capacity utilization, top-name concentration, and dependence-aware
uncertainty — have **no qualifying evidence at all** in Sprint 4, so they are
recorded as unmeasured and block.

That is the gate working. A synthetic development draw with no trial ledger and
no capacity measurement has not earned paper trading, and the honest report says
so. The seed was fixed before the run and has not been changed; picking a seed
whose path happened to make money would be exactly the cherry-picking the
evidence standard forbids.

The figure shows the adverse outcomes beside the favorable ones: the perturbation
downside envelope, the drawdown amplification, the frequency of underperforming
the unperturbed path, and the seven blocking failures.

## 10. Evidence

- `tests/test_perturbation_study.py` — 50 tests: grid freezing and identity;
  duplicate, unsupported, oversized, and under-replicated refusals; stream
  isolation and exact single-path replay against what the study recorded;
  per-arm mechanics including cost one-sidedness, fill attenuation, stale-signal
  substitution, and bounded draws; the reordering invariance and its drawdown
  effect; the one-directionality guard across every arm; insolvency, the
  severe-but-survivable distinction, no-trade, and reconciliation; failure
  ordering; determinism; JSON serializability; bounded retention and cost.
- `tests/test_qualification_decision.py` — 37 tests: rubric freezing and
  threshold-lowering detection; evidence-hash and evidence-kind validation;
  full-pass qualification; rejection on missing observation, non-finite
  observation, missing evidence, partial evidence, a single failed criterion,
  insolvent stress paths, and reconciliation failure; inclusive boundaries; both
  threshold directions; dossier ordering, identities, limitations, and the
  authorization language in both verdicts.
- `tests/test_sprint_4_evidence.py` — 8 tests: deterministic synthetic path,
  the `REJECTED` verdict and which criteria block it, complete artifact set,
  non-overwriting publication, reproducibility across runs, and the limitations
  text.

## 11. Residual limitations

- **The perturbation family is an assumption**, chosen to represent ordinary
  operational imperfection. A market that fails in a way the family does not
  describe is not covered by any of these numbers.
- **A frozen rubric prevents moving the bar; it does not make the bar correct.**
  The thresholds are judgements, recorded so they can be argued with.
- **Passing every criterion establishes that a candidate was not obviously
  broken**, not that it will earn money.
- **The stress operates on a realized return and cost path**, not on the order
  book. Interaction effects between perturbations — a missed trade *and* a
  liquidity shock in the same bar — are not modelled; each arm is stressed
  independently.
- **Synthetic development data only.** No licensed point-in-time data, no
  survivorship-free universe, and no live or paper execution.
