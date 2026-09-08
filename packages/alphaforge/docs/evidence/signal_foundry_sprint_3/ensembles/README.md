# Governed ensemble reference evidence

This directory contains deterministic **synthetic engineering evidence** for
SF-S3-MR9. It verifies the temporal-OOF contracts, six combination policies,
fallback paths, aggregate diagnostics, and evidence publisher. It is not market
evidence, a backtest on licensed observations, paper trading, or live trading.

Reproduce into a new directory:

```bash
make ensemble-evidence OUTPUT=/absolute/path/to/new-evidence-directory
```

The frozen input is
[`configs/ensemble_benchmark.yaml`](../../../../configs/ensemble_benchmark.yaml).
The publisher refuses overwrite and writes no row predictions, targets, fitted
state, tensors, or model artifacts.

Artifacts:

- [`model_summary.csv`](model_summary.csv) — holdout error, within-date rank IC,
  moving-block variance/intervals, fallback, heuristic dispersion,
  turnover/cost, and paired best-single comparisons;
- [`prediction_error_correlations.csv`](prediction_error_correlations.csv) —
  pairwise prediction and error correlations with explicit undefined states;
- [`regime_overlap.csv`](regime_overlap.csv) — pairwise signed overlap and error
  correlation for calm, uncertain, and stress groups;
- [`marginal_contribution.csv`](marginal_contribution.csv) — drop-one expert
  ablations;
- [`turnover_costs.csv`](turnover_costs.csv) — transparent configured synthetic
  turnover-cost diagnostic with its complete moving-block policy;
- [`uncertainty_diagnostics.csv`](uncertainty_diagnostics.csv) — explicitly
  heuristic interval coverage and dispersion/error association;
- [`ensemble_evidence.png`](ensemble_evidence.png) — inspected Seaborn summary;
  and
- [`summary.json`](summary.json) — resolved config, generator/bootstrap seed
  map, resampling assumptions, runtime, identities, resource counts, artifact
  integrity hashes, and limitations.

See the [engineering report](../../../sprint_3_ensemble_report.md) and
[governed ensemble contract](../../../governed_ensembles.md) for interpretation.
