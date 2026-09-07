# Governed temporal-OOF ensembles

AlphaForge's advanced ensemble layer is a second-stage research boundary. It
does not fit experts and it does not accept a feature matrix. It consumes only
complete temporal out-of-fold predictions whose fold provenance and final
holdout cutoff have already been frozen.

> The committed reference is deterministic synthetic engineering evidence. It
> demonstrates contracts, algorithms, failures, and reporting. It is not
> evidence of market predictability, paper readiness, or live profitability.

## Contracts and timing

`TrainingOOFPanel` is immutable and content-addressed. Each expert prediction
contains:

| field | invariant |
|---|---|
| `date`, `symbol` | canonical row identity |
| `fold_id` | declared temporal validation fold |
| `expert` | member of the canonical expected-expert set |
| `prediction` | finite scalar created outside that expert's training fold |
| `uncertainty` | optional finite non-negative standard deviation |
| `regime_probability` | optional causal probability in `[0, 1]`, identical across experts for a row |

One `TrainingOOFTarget` is aligned to every `(date, symbol, fold_id)`. A
`TemporalOOFFold` records `training_end`, `validation_start`,
`validation_end`, and a minimum number of intervening calendar embargo days.
Validation rejects partial expert sets, duplicates, false embargo attestations,
interleaved folds, dates outside validation, and any record at or after
`holdout_start`.

`InferenceBatch` is a distinct exact-schema type with no target field. A caller
cannot pass a final-holdout outcome through this interface. It also carries the
predeclared expected `(date, symbol)` key set, so loss of every expert row for a
symbol cannot silently shrink a rank universe. Missing expert rows produce
explicit abstention records; unexpected keys, duplicates, and malformed values
fail immediately.

```python
from alphaforge.models import (
    GovernedEnsembleConfig,
    InferenceBatch,
    TrainingOOFPanel,
    fit_governed_ensemble,
)

panel = TrainingOOFPanel.from_frame(
    training_oof_frame,
    folds=folds,
    expected_experts=("defensive", "stable", "trend"),
    holdout_start="2026-01-02",
    source_id="validated-oof-run-17",
)
state = fit_governed_ensemble(
    panel,
    GovernedEnsembleConfig(
        method="stacking",
        experts=("defensive", "stable", "trend"),
        ridge_penalty=1.0,
    ),
)
decisions = state.predict(
    InferenceBatch.from_frame(
        target_free_holdout_predictions,
        expected_keys=predeclared_holdout_date_symbol_keys,
    )
)
```

Every decision is immutable, has a stable content identity, and contains the
state identity, effective weights, heuristic dispersion, status, and fallback
reason. The dispersion combines supplied expert estimates, training-OOF
residual MSE, and expert disagreement; it is not a calibrated predictive
standard deviation. Consumers must check `status`; a numeric fallback
prediction is not permission to trade.

## Policies

### Static blend

Weights must be finite, non-negative, complete, and non-degenerate. They are
normalized once. Missing experts abstain; available weights are never silently
renormalized.

### Rank/vote

Each expert is ranked across the predeclared complete symbol set on a date.
Centered ranks are averaged and mapped back to training-target scale. If any
expert or whole symbol is missing, the entire date abstains because partial
ranks change every other symbol's score. A one-symbol cross-section is
undefined and also abstains.

### Temporal OOF stacking

The meta-learner uses ridge regression over expert OOF predictions. Scaling is
fit on training OOF rows only. Its audit trail includes one diagnostic fit per
fold: fold `k` can name only folds `< k` as input; the first fold records
`no_prior_meta_fold`. The final state trains on all OOF rows and is frozen
before the target-free final holdout is opened.

### Bayesian averaging

OOF residual variances define relative log evidence. Shifted, clipped softmax
prevents overflow. A feasible pre-registered post-normalization floor
\(f \leq 1/n\) uses \(w_j=f+(1-nf)\operatorname{softmax}_j(\ell)\), so the
declared lower bound actually holds. This is an engineering approximation, not
a claim that expert errors are independent or that the prior is economically
calibrated. Prediction and error correlations are always published beside it.

### Causal dynamic weighting

Date `t` is predicted with the weights that existed before its targets. Only
after all OOF targets for `t` are observed does the EWMA error state update.
Every transition records its before and after weights and effective date.
Final-holdout inference is side-effect free and does not adapt.

### Regime gate

Separate calm/stress weights use only confident training-OOF rows. Gate
probabilities are assumed causal and must agree across experts. Insufficient
training support creates a fallback state. Missing/inconsistent probabilities
or a probability inside the uncertainty band abstain at inference. With a zero
confidence margin, equality at the threshold belongs only to stress in both
training and inference.

## Failure and resource behavior

- Unknown fields, unsafe identifiers, non-finite values, invalid
  uncertainties/probabilities, and mismatched targets fail before fitting.
- Training requires complete expert coverage; inference reports missing
  experts as abstentions.
- Degenerate learned targets produce a frozen fallback state.
- Arithmetic overflow and non-finite intermediate loss/inference results are
  translated into structured contract errors.
- Member exceptions in the legacy registry adapter propagate; they are not
  translated into zero IC.
- Prediction records, distinct dates, experts, serialized bytes, and
  stacking/dynamic audit transitions have hard ceilings enforced before
  over-budget fitting.
- Canonical JSON round trips only after exact-field, version, digest-integrity,
  semantic, and resource validation. The embedded unkeyed digest is not an
  authenticity proof; provenance-sensitive callers compare it with an
  independently trusted expected identity. Pickle/joblib is not accepted.

## Reproducible evidence

Run the frozen offline study into a new directory:

```bash
make ensemble-evidence OUTPUT=/absolute/path/to/new-evidence-directory
```

`configs/ensemble_benchmark.yaml` is exact-schema: unknown keys, inconsistent
fold dimensions, insufficient resource budgets, unsafe ranges, and any
interpretation other than `synthetic_engineering_only` fail before generation.
The runner refuses overwrite.

Published outputs are aggregate:

- `model_summary.csv`: error, within-date rank IC, moving-block
  variance/intervals, fallback, heuristic dispersion, turnover, cost, and
  paired best-single deltas;
- `prediction_error_correlations.csv`: full prediction/error dependence matrix
  with explicit undefined states;
- `regime_overlap.csv`: pairwise signed overlap and error correlation by regime
  with explicit undefined states;
- `marginal_contribution.csv`: drop-one-expert MSE changes;
- `turnover_costs.csv`: gross/net simplified cost diagnostic with complete
  moving-block resampling policy;
- `uncertainty_diagnostics.csv`: heuristic interval coverage and
  dispersion/error association with explicit undefined states;
- `ensemble_evidence.png`: inspected Seaborn summary; and
- `summary.json`: resolved config, named generator/bootstrap seed map,
  resampling assumptions, environment, identities, artifact hashes, counts,
  and limitations.

No row prediction, target, fitted state, model artifact, credential, or licensed
observation is written. See [the Sprint 3 ensemble report](sprint_3_ensemble_report.md)
and [ADR 0006](adr/0006-governed-temporal-oof-ensembles.md).

## Residual limitations

The boundary verifies declared temporal structure, not the honesty of an
external prediction producer. Bayesian weights use an approximate likelihood.
Regime probabilities may be miscalibrated or drift. The synthetic generator
contains deliberately complementary experts. The cost calculation is a
transparent turnover diagnostic, not AlphaForge's event-driven fill engine.
Licensed point-in-time evaluation, frozen multiplicity policy, capacity
analysis, and independent review remain mandatory before any advancement.
