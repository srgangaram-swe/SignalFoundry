# Sprint 2 — Baselines and statistical governance

Sprint 2 closes with a governed **REJECT_PAPER_ADVANCEMENT** decision. That is
the scientifically useful outcome: the platform executed the complete
pre-registered family, preserved every result, corrected for selection, opened
the final holdout once for the development-selected candidate, and rejected
evidence that did not clear the declared gates.

## Scope delivered

- conventional causal features and seven deterministic CPU baselines;
- eight identical embargoed expanding walk-forward folds and one frozen final
  holdout;
- model convergence and bounded resource contracts;
- OOF regression-calibration and dependence-aware uncertainty tools;
- unified predictive, economic, cost, risk, capacity, and readiness metrics;
- immutable experiment identity and append-only, receipted trial governance;
- exact complete-family Holm-Bonferroni correction and predeclared kill
  criteria; and
- aggregate-only machine-readable evidence plus four inspected Seaborn plots.

The closing study used producer commit
`881779f6cc7b1b0ce3261cc8d3819a6ab87e6fd7`, plan/study identity
`30426f30f03a465085951044a29caec4d3569dad5d5ab25e526b8979f869a725`,
and research-run identity
`f50fde355139317a9d93685c68c148e35f954e3b4cafea18d29900d127a88a9b`.
Its verified local source was cached Signalattice bundle
`63bc9af39ed5199cf4027355163c95767c8530b31af9d986a9e7a18006c0e26f`.
The replay made **zero provider requests** and did not read an API credential.

## Evidence and decision

All seven pre-registered candidates completed eight identical development
folds, for 56 model/fold fits. No backend failed. Linear regression ranked
first on mean development rank IC and was the only candidate admitted to the
frozen final holdout.

Linear's development raw one-sided p-value against the naive momentum reference
was approximately 0.0120, but the complete-family Holm-adjusted p-value was
approximately 0.0841. No candidate survived the predeclared 0.05 family-wise
threshold. The linear development backtest was only approximately 0.56% net
annualized after declared costs; momentum and every nonlinear model also failed
at least one incremental-IC or net-return kill rule.

On the untouched historical holdout, linear regression produced:

- gross annualized return: approximately 0.61%;
- net annualized return after declared costs: approximately -0.88%;
- maximum drawdown: approximately -3.97%;
- probability-of-backtest-overfitting estimate: approximately 0.315; and
- 95% circular moving-block annual-return interval: approximately
  [-1.60%, 1.27%].

The readiness rubric returned `NOT_READY`. Failed gates were benchmark excess,
deflated Sharpe, stress scenarios, and complete point-in-time evidence.
Incomplete historical revisions, universe membership, delistings, and
corporate actions independently prohibit paper or live advancement.

This is a stale, current-vintage Nasdaq WIKI engineering study through
2018-03-27. It is not current market evidence, paper trading, live trading, a
forecast, or a profit claim.

## Resource and reproducibility evidence

The local seven-candidate run recorded 62.39 seconds wall time, 60.77 seconds
user CPU, 0.89 seconds system CPU, 634,765,312 bytes maximum resident set, and
413,861,472 bytes peak memory footprint on the recorded macOS environment.
Every tree/boosting backend was bounded to one worker. The evidence records 56
model fits, 29,106 OOS prediction rows, 3,149 reported estimator iterations,
and model-level warning counts.

The complete ledger contains 23 verified records. Its local head receipt binds
the chain head and ledger byte hash. This detects local mutation/truncation
relative to the receipt but is not a signature; externally anchored immutable
receipts remain future operational work.

The reference publication is
[Signal Foundry Sprint 2 evidence](evidence/signal_foundry_sprint_2/README.md).
The four regenerated and visually inspected plots show:

1. every matched-fold rank IC, including negative observations;
2. gross versus net development economics under one declared cost policy;
3. all seven Holm-adjusted p-values and the frozen alpha threshold; and
4. bounded estimator-iteration accounting and recorded warning state.

Only non-reconstructive fold/model aggregates and resource totals are tracked.
The licensed bundle, final holdout, observations, targets, predictions, return
series, orders, fills, positions, and the complete local run remain ignored.

## Validation

The implementation passed the unchanged repository gate:

- `uv lock --check` and strict validation of every committed configuration;
- all repository policy hooks, Ruff, Black, and mypy across 140 source files;
- 484 offline tests with 83.13% branch-aware coverage;
- focused negative, exact-family, ledger, publication-boundary, and plot tests;
- wheel and source-distribution builds; and
- staged diff, generated-file, private-key, and secret hygiene.

The test run emitted 176 existing joblib/NumPy deprecation warnings and seven
Seaborn/Matplotlib deprecation warnings. They did not alter results or weaken a
gate, but dependency compatibility remains an explicit maintenance concern.

## Residual gates

No capital should be placed at risk from this result. Before paper evaluation,
rerun the full evidence chain on licensed point-in-time prices, corporate
actions, delistings, symbol history, and universe membership without tuning on
the final holdout. Any later paper phase must add current data, broker/exchange
failure rehearsal, reconciliation, kill-switch drills, and a time-bounded
zero-capital shadow period. Live use still requires a separate explicit owner
approval and bounded capital-at-risk gate.
