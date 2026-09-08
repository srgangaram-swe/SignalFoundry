# Leakage-safe latent representations

AlphaForge's representation layer compares dimensionality reduction and
self-supervised feature learning behind one target-free, temporally aligned
contract. It is a research component. It does not select a production model,
authorize a trade, or establish that a learned representation is useful on real
markets.

## Public contract

`RepresentationBatch` owns a finite, read-only feature matrix and exact
`(date, symbol)` row identities. Dates must be monotonically non-decreasing,
identities and feature names must be unique, and a target cannot be passed through
this API. `RepresentationConfig` validates the algorithm and all resource and
optimization limits. `create_representation` returns a `BaseRepresentation` with:

- `fit(batch)`, which learns state from one training batch;
- `transform(batch)`, which preserves row identities and returns immutable
  embeddings; and
- `reconstruct(batch)`, when the representation has a decoder or inverse map.

Inference before fit, changed feature order, excess rows or columns, non-finite
values, unsupported reconstruction, missing Torch, non-finite optimization, and
embedding collapse raise typed errors. The output never silently changes row
alignment.

Each fitted state records the config and state identities, exact fit interval and
row count, inner-validation rows, termination reason, embedding variance,
reconstruction error where defined, parameter count and bytes, CPU/wall time, and
device. Timing is evidence, not part of the state identity.

## Candidate semantics

| Candidate | Mechanism | Important invariant |
|---|---|---|
| `raw` | Train-standardized feature control | Establishes the no-compression baseline |
| `pca` | Full-SVD PCA | Sign-canonical components and rotation-invariant subspace identity |
| `incremental_pca` | Chronological contiguous partial fits | No shuffled or future batch |
| `robust_pca` | Median/IQR scaling followed by full-SVD PCA | Robust marginal scaling; not sparse low-rank decomposition |
| `dense_autoencoder` | Bounded dense encoder/decoder | Inner-train normalization and validation-best restore |
| `sequence_autoencoder` | GRU encoder with dense current-row decoder | Per-symbol causal left-padded windows |
| `denoising_autoencoder` | Dense autoencoder with Gaussian input corruption | Corruption occurs during fit only |
| `variational_autoencoder` | Gaussian latent encoder with KL regularization | Posterior mean makes inference deterministic |
| `contrastive_timeseries` | Causal GRU encoder and NT-Xent projection head | Fit-only meaning-preserving paired views; no decoder |

All neural candidates use AdamW, gradient clipping, named random streams, bounded
epochs and patience, deterministic CPU algorithms, and a hard parameter ceiling.
An epoch-ceiling stop remains explicitly `converged=false`; the best finite
validation state may still be used for engineering evaluation.

## Leakage and causality boundary

The reference study fixes complete-date train, validation, and test partitions
before fitting:

```text
outer train -> representation fit
             -> neural inner-train normalization and optimization
             -> neural inner-validation early stopping

outer train embeddings -> fixed downstream estimators
outer validation       -> representation-family selection
outer test             -> one post-selection summary
```

Transforming a later batch never updates representation state. A sequence row can
observe only the same symbol's current and earlier features. Positive scaling,
jitter, and past-step masking are allowed only while fitting the contrastive
candidate; time reversal and cross-symbol mixing are invalid because they change
temporal meaning.

The study derives independent seeds by semantic name rather than consuming one
global stream. Reversing candidate execution order therefore cannot change fitted
state or deterministic numeric evidence. Tests also mutate held-out targets and
future feature rows to prove that they cannot alter fitted identities, validation
selection, or earlier causal windows.

## Evaluation axes

Every candidate uses the same four-dimensional latent budget except the
ten-dimensional raw control. The deterministic reference contains 432 synthetic
rows, six symbols, 72 dates, ten observed features, planted nonlinear factors,
two regimes, and planted anomalies.

The study reports:

- validation and untouched-test prediction MSE and mean daily rank IC from a
  fixed ridge evaluator;
- per-date rank-IC standard error and contributing date count;
- fixed logistic-regime accuracy;
- nearest-training-neighbor regime recall as a similarity diagnostic;
- distance-based anomaly AUROC;
- effective rank and mean absolute embedding correlation as diversification
  diagnostics;
- original-unit reconstruction MSE where defined;
- validation-only selection and deltas against the raw control; and
- iterations, convergence, parameters, bytes, CPU time, and wall time.

The daily standard error is descriptive variation across observed dates. It is
not a confidence interval and does not correct for serial dependence. These
transfer tasks test mechanics against deliberately planted structure; they do not
constitute market or economic evidence.

## Reproduce the reference

Install the optional Torch dependency, then publish into a new directory:

```bash
make install-all
make latent-representation-evidence \
  OUTPUT=/tmp/alphaforge-latent-representation-evidence
```

The exact-field profile is
`configs/latent_representation_benchmark.yaml`. The command is offline and
CPU-capable. It refuses to overwrite an existing path and atomically publishes:

- `candidate_summary.csv`;
- `summary.json`;
- `manifest.json`; and
- `plots/representation_comparison.png`.

The manifest hashes every generated evidence artifact and explicitly records that
raw rows, targets, embeddings, predictions, weights, and credentials were not
published. A rerun should reproduce identities and numeric metrics; wall/CPU time
and the plot's compute bars are environment-dependent measurements.

## Residual limitations

The reference is a single small synthetic split with planted structure. It is
cleaner than real markets, has no transaction costs or investable baseline, and
does not estimate capacity, turnover, stability under regime drift, or
multiple-testing risk from future studies. Several bounded neural candidates
reached their epoch ceiling rather than a patience convergence condition.

A market-facing follow-up requires licensed point-in-time data, train-fold-only
feature construction, multiple predeclared temporal folds, dependence-aware
uncertainty, a complete trial family, realistic costs, and an untouched final
holdout. No result from this layer can bypass AlphaForge's existing research,
paper-readiness, or live-capital gates.
