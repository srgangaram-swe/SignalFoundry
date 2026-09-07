# Sprint 1 — Research foundation and temporal integrity

Sprint 1 established an evidence chain from a validated Signalattice bundle to
causal features and labels, interval-aware validation, governed model
selection, reconciled historical execution, adversarial evaluation, and a
fail-closed zero-capital paper decision.

## Outcome

The final cached WIKI engineering run selected ridge from four pre-registered
candidates. On the untouched 2017-01-03 through 2018-03-27 interval it returned
`NOT_READY`. The net historical result was -1.98% total and -1.63% annualized,
versus 16.94% annualized for the sample's declared AAPL benchmark. Deflated
Sharpe probability was 6.72%. Benchmark excess, Deflated Sharpe,
point-in-time evidence, and adversarial stress gates failed.

This is the useful result of an honest gate: the current candidate and stale
dataset do not justify paper or live trading. No parameter was retuned after
the holdout result.

## Evidence chain

- [Financial-label diagnostics](evidence/label_diagnostics/manifest.json) use
  deterministic synthetic data to expose overlap, effective evidence, class
  balance, chronological stability, and predeclared sensitivity.
- [Temporal-validation evidence](evidence/temporal_validation/manifest.json)
  shows train, validation, purge, embargo, test, overlap, and inaccessible
  final-holdout roles.
- [Governed WIKI evidence](evidence/signal_foundry_sprint_1/README.md) publishes
  aggregate readiness, scenario, capacity, and performance evidence only.

The sprint-close Seaborn plots are:

- [label dependence](evidence/label_diagnostics/plots/dependence.png),
  [class balance](evidence/label_diagnostics/plots/class_balance.png),
  [temporal stability](evidence/label_diagnostics/plots/temporal_stability.png),
  and [parameter sensitivity](evidence/label_diagnostics/plots/parameter_sensitivity.png);
- [temporal fold roles](evidence/temporal_validation/temporal_folds.png); and
- [readiness gates](evidence/signal_foundry_sprint_1/plots/readiness_gates.png),
  [adversarial scenario returns](evidence/signal_foundry_sprint_1/plots/scenario_returns.png),
  and [capacity sensitivity](evidence/signal_foundry_sprint_1/plots/capacity_sensitivity.png).

All plots were generated through Seaborn from machine-readable evidence and
visually inspected. The WIKI plots contain only non-reconstructive aggregates.

## Reproducibility and performance

The successful run identity is
`ec76e4129e8b9a4a26c6bdc9435444e4bdd575b57b36613467deb9bd73462a49`
on AlphaForge code
`2530bf06a81fcb61b13b2b4c8b94cc95dd70b840`. It replayed the verified
13,169-row, ten-instrument, six-partition bundle entirely from immutable local
cache and made zero provider requests.

The single local macOS/arm64 CPU profile took 13.44 seconds wall time, 17.38
seconds user CPU, 2.09 seconds system CPU, and reported a maximum resident set
of 585,334,784 bytes. This is a bounded engineering profile, not a latency
distribution or capacity service-level objective.

## Safety and limitations

The public evidence contains no credential, provider response, licensed
observation, ticker list, prediction, order, fill, position, or date-level
return. The source ends in 2018, is current-vintage, lacks complete historical
revisions, point-in-time universe membership, and corporate actions, and uses
one constituent rather than a diversified benchmark. Capacity is an ex-post
daily-bar sensitivity, not a deployable-AUM forecast.

The offline paper audit proves idempotency, stale-data, exposure-limit, and
one-way kill-switch halts. It has no broker adapter and emits no executable
order. Any future paper milestone requires licensed point-in-time current data,
a diversified investable benchmark, a new untouched interval, continued
negative controls, a bounded observation period, and separate owner approval.
