# Governed benchmark models (SF-S2-MR5)

AlphaForge exposes a bounded family of tabular regression models through the
same [`AlphaModel`](../alphaforge/models/base.py) interface. These models are
research candidates: they estimate forward returns inside temporal development
folds, publish termination evidence, and must compete against naive baselines.
They do not constitute a trading strategy or establish a persistent edge.

## Contract and preprocessing boundary

Every factory validates its complete public hyperparameter surface before
allocating an estimator. Tree count is capped at 2,000, iterative solvers at
their declared finite limits, tree depth at 64, MLP width at 256 per layer,
MLP depth at two layers, and worker count at 64. `n_jobs=-1` is rejected because
it delegates resource policy to the host and is not a reproducible bound.

Each estimator owns a scikit-learn `Pipeline`:

1. median imputation is fitted on the current training fold;
2. linear, robust, and MLP families fit standardization on that same fold;
3. the estimator fits after those learned transforms; and
4. prediction reuses the frozen transform state.

No statistic from a validation, test, final-holdout, or live observation enters
the fitted pipeline. A new model is instantiated for every walk-forward window.
The outer optional feature transform follows the same train-fold-only rule.

## Families and mathematical role

Let \(X \in \mathbb{R}^{n \times p}\) be one fold's transformed feature matrix,
\(y\) its forward-return target, and \(\beta\) the coefficient vector.

| Registry name | Governed role |
|---|---|
| `linear` | Ordinary least squares, \(\min_\beta \lVert y-X\beta\rVert_2^2\), as an independent low-variance reference. |
| `ridge` | L2-regularized least squares, adding \(\alpha\lVert\beta\rVert_2^2\). |
| `lasso` | L1-regularized least squares, adding \(\alpha\lVert\beta\rVert_1\), for sparse selection pressure. |
| `elastic_net` | Convex L1/L2 mixture controlled by `l1_ratio`. |
| `huber` | Robust M-estimation with quadratic residual loss near zero and linear growth beyond `epsilon`; this bounds outlier influence rather than assuming Gaussian tails. |
| `random_forest` | Seeded bagged decision trees with bounded depth, leaf size, feature sampling, tree count, and worker count. |
| `extra_trees` | Seeded extremely randomized trees, separated from Random Forest so random split thresholds are an explicit modeling decision. |
| `gradient_boosting` | Backward-compatible explicit selector for bounded scikit-learn histogram boosting or LightGBM. It never changes backend because a package happens to be installed. |
| `lightgbm` | Primary optional tabular benchmark: deterministic CPU histogram boosting with column-wise construction and one worker by default. |
| `xgboost` | Optional CPU histogram booster. `min_child_weight` retains XGBoost's Hessian-mass semantics; it is not mislabeled as a sample-count leaf bound. |
| `catboost` | Optional CPU depthwise booster with bootstrap and random score noise disabled, no file-writing side effect, an explicit seed, and bounded threads. |
| `small_mlp` | Controlled one- or two-hidden-layer ReLU MLP with fixed row order, bounded width/batch/iterations, train-fold scaling, and an explicit seed. |

Linear-model parity is tested against an independent NumPy least-squares
solution. Huber's reduced sensitivity is tested on predeclared contaminated
synthetic targets. These tests establish implementation behavior, not market
predictiveness.

## Termination and reproducibility evidence

After every fit, `training_diagnostics()` returns an immutable record:

- exact backend;
- `converged`, `completed`, `max_iterations`, or `warning` status;
- observed and maximum iteration counts when exposed;
- injected seed; and
- captured warning messages.

Walk-forward `model_metrics.csv` rows carry the backend, status, iteration
count, budget, seed, and warning count. Algorithms without a numerical stopping
criterion report `completed`, not a manufactured convergence claim. An
exhausted iterative budget reports `max_iterations`; warnings are never
silently discarded.

The root seed is injected by model name and does not depend on candidate order.
Explicit seeds are preserved. Reproducibility tests fit independent instances
and require equal predictions, exercise trusted persistence round trips, and
run on every core Python version in CI. CPU execution and one worker are the
reference settings. Platform or dependency changes remain part of the
experiment identity, so equality across a different environment is not
silently assumed.

## Optional dependencies

The locked `ml` extra contains:

- LightGBM (MIT), the primary tabular boosting benchmark;
- XGBoost (Apache-2.0), an independent boosted-tree comparison;
- CatBoost (Apache-2.0), an ordered-boosting ecosystem comparison; and
- Graphviz (MIT Python package), a CatBoost transitive runtime dependency.

Factories import an optional ecosystem only when its registry name is selected.
A missing library raises an actionable `alphaforge[ml]` installation error.
There is no silent fallback. CI installs the locked extra and fits every
optional backend twice; import-only checks would not prove estimator behavior.
Each optional contract test runs in a fresh interpreter. This is a deliberate
native-runtime boundary: restoring an XGBoost joblib object in a long-lived
macOS process after several unrelated native/OpenMP runtimes have been loaded
has produced a native crash. Optional artifacts must therefore be restored in a
clean, single-backend worker, never inside a mixed-backend service process.

```bash
uv sync --locked --extra dev --extra data --extra ml
uv run pytest tests/test_governed_models.py tests/test_optional_boosters.py
```

The default `configs/models.yaml` remains runnable without optional ML
libraries. A governed LightGBM comparison is enabled explicitly:

```yaml
- name: lightgbm
  params:
    n_estimators: 300
    max_depth: 4
    learning_rate: 0.05
    min_samples_leaf: 20
    l2_regularization: 1.0
    n_jobs: 1
    random_state: 42
```

## Security, rollback, and trading limitations

Sklearn and optional model persistence uses joblib. Joblib is a pickle-family
format and can execute code, so `AlphaModel.load` refuses it unless the caller
passes `trusted=True` after verifying provenance and integrity. Model artifacts,
licensed observations, credentials, and generated runs remain untracked.
Optional backends run on CPU, receive no network capability from AlphaForge,
and CatBoost is configured not to write training files. The trusted flag does
not make native deserialization memory-safe; the clean single-backend worker
boundary above is also required for optional artifacts.

Rollback is registry-local: remove the affected explicit registry entry and
configuration surface while retaining the common model contract. Never replace
a failed backend with another estimator under the same name, because that
would invalidate experiment identity.

The model contract is now `1.1.0`; `1.0.0` JSON artifacts fail closed and must
be regenerated from recorded configuration and immutable inputs. Editing an
old artifact's version marker is not a supported migration.

This slice makes model comparison more rigorous; it does not make AlphaForge
paper- or live-ready. Before capital is considered, candidates still need
licensed point-in-time data, untouched-holdout evidence, correction for all
attempted variants, realistic execution and financing costs, regime and
capacity stability, paper-trading duration, monitoring, reconciliation, risk
limits, kill switches, operational rehearsal, and explicit owner approval.
No historical metric or fitted model guarantees profit.
