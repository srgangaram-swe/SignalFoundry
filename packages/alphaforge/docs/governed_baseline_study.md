# Governed seven-candidate baseline study

Sprint 2 closes with one pre-registered comparison of a naive momentum rule,
ordinary linear regression, Random Forest, LightGBM, XGBoost, CatBoost, and a
bounded scikit-learn MLP. The study is designed to reject weak evidence, not to
manufacture a favorable equity curve.

## Frozen boundary

`configs/signal_foundry_sprint_2_study.yaml` fixes, before execution:

- the ordered seven-candidate family and deterministic root seed;
- the feature, label, embargoed expanding-window, and 2017-01-03 final-holdout
  boundaries;
- single-worker CPU limits and bounded estimator iterations;
- one portfolio, execution, commission, spread, slippage, capacity, borrow, and
  financing policy;
- the selected-candidate-only final-holdout rule; and
- the all-gates paper-readiness rubric.

`configs/research_governance.yaml` separately fixes the complete family size,
Holm-Bonferroni family-wise correction, alpha, failed-trial treatment, and kill
criteria. The runner injects the seed before hashing the plan. Every non-naive
candidate records the naive momentum trial as its parent lineage.

The immutable plan is the first record in a bounded single-writer
`ResearchLedger`. Each candidate is registered, started, and terminal before
the family may be evaluated. A failed backend remains in the family with
p-value 1.0. Missing, duplicate, mutated, truncated, partially evaluated, or
re-evaluated evidence fails closed.

## Statistical interpretation

Each candidate receives the identical development folds. The primary
development statistic is fold-level Spearman rank IC:

- momentum is tested against zero;
- every other candidate is tested on matched-fold rank-IC differences from
  momentum; and
- the one-sided Student-t p-values are corrected together with
  Holm-Bonferroni.

The t-test uses the small set of walk-forward folds as the analysis unit. It is
transparent and matched, but it has low power and depends on the fold-level
sampling assumptions. The published distribution therefore shows every fold,
including negative values. A corrected p-value is not evidence of economic
value: each candidate must also survive positive incremental IC, positive net
annual return after the common cost model, and maximum-drawdown criteria.

OOF regression calibration reports the intercept, slope, and RMSE from
realized target on prediction. These are descriptive continuous-forecast
diagnostics, not probability calibration. The selected candidate alone enters
the frozen final holdout, where the existing governed runner adds reconciled
costed accounting, stress scenarios, capacity, moving-block return intervals,
selection-overfitting diagnostics, and paper-readiness gates.

## Local execution

The proven cached Signalattice WIKI bundle is replayed without a provider
request or API credential:

```bash
/usr/bin/time -l -o runs/sprint-2-time.txt \
  .venv/bin/python scripts/run_sprint_2_study.py \
  ../Signalattice/data/signal-foundry-bundles/<bundle-id> \
  --config configs/signal_foundry_sprint_2_study.yaml \
  --governance-config configs/research_governance.yaml
```

The command prints the immutable study and research-run directories. Publish a
new aggregate-only directory with:

```bash
.venv/bin/python scripts/publish_sprint_2_evidence.py \
  <study-dir> <run-dir> \
  ../Signalattice/data/signal-foundry-bundles/<bundle-id> \
  runs/sprint-2-time.txt \
  --config configs/signal_foundry_sprint_2_study.yaml \
  --output docs/evidence/signal_foundry_sprint_2
```

The publisher verifies the plan, ledger chain and receipt, complete family,
matched folds, run/bundle identities, and aggregate-only license policy. It
allows only fold/model aggregates and local resource totals across the public
boundary. Dates, symbols, observations, targets, predictions, final-holdout
rows, returns, orders, fills, and positions remain under ignored `runs/` and
Signalattice `data/` paths.

## Decision and rollback

`ADVANCE_TO_PAPER_EVALUATION` requires both a selected candidate with no
statistical/economic kill reason and the existing all-gates
`READY_FOR_PAPER` result. Any other state is `REJECT_PAPER_ADVANCEMENT`.
Neither decision authorizes live trading.

Rollback removes this opt-in runner, profile, publisher, and aggregate evidence.
It does not alter the stable feature, model, backtest, data-bundle, or final
holdout contracts. Local immutable run artifacts may be retained for audit or
removed deliberately outside Git after their published hashes are no longer
needed.
