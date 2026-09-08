# ADR 0005: Bound latent representations behind a leakage-safe contract

- Status: Accepted
- Date: 2026-07-26
- Decision owners: AlphaForge maintainers
- Depends on: ADR 0001 and the temporal-validation contract

## Context

AlphaForge needs to compare linear and nonlinear feature representations without
letting labels, future rows, candidate order, or the final evaluation interval
affect learned preprocessing. A representation is an upstream feature mechanism,
not a prediction model: its fit boundary must therefore be narrower than the
supervised-model boundary.

PCA also has non-identifiability that ordinary model hashes obscure. Component
signs are arbitrary, and any orthogonal rotation within a repeated-eigenvalue
subspace is equivalent. Neural encoders introduce different risks: nondeterministic
initialization, temporal augmentation that changes financial meaning, unbounded
allocation, silent embedding collapse, and accidental publication of weights or
row-level embeddings.

## Decision

Add a typed `RepresentationBatch -> RepresentationOutput` contract under
`alphaforge/representations`. The input contains finite feature values and unique,
chronologically ordered `(date, symbol)` identities. It deliberately has no target
field. Every implementation:

1. fits normalization and learned state on the declared training batch only;
2. validates an exact feature schema at inference;
3. publishes immutable config, fit-batch, and fitted-state identities;
4. enforces sample, feature, tensor-byte, parameter, and iteration ceilings;
5. rejects non-finite state and materially collapsed embeddings; and
6. emits CPU resource and termination evidence without publishing learned weights.

The frozen family is:

- a standardized raw-feature control;
- full-SVD PCA;
- chronological incremental PCA;
- median/IQR-scaled full-SVD PCA, named `robust_pca` for API stability but
  explicitly not principal-component pursuit;
- dense, sequence, and denoising autoencoders;
- a variational autoencoder whose inference output is the posterior mean; and
- a causal time-series contrastive encoder.

Torch remains an optional, lazily imported backend. Linear representations and
configuration validation do not require it. The neural reference path is CPU-only,
uses named random streams and deterministic Torch algorithms, restores the
validation-best state, clips gradients, and records whether patience or the epoch
ceiling stopped fitting.

Sequence windows are built independently per symbol, ordered by time, and left
padded. They contain the current and earlier feature rows only. Contrastive
augmentation is fit-only and limited to positive amplitude scaling, jitter, and
masking of past steps. Time reversal, future access, cross-symbol mixing, and
sample permutation as a semantic transform are prohibited.

## Identity and mathematical invariants

The config identity hashes canonical JSON. The fit-batch identity also binds
feature values, ordered dates, symbols, and feature names. A fitted-state identity
binds the config, fit batch, normalization statistics, and learned numeric state;
wall-clock timing is intentionally excluded.

PCA components are sign-canonicalized by making each row's largest-magnitude
loading positive. A separate subspace identity hashes the rounded orthogonal
projector

\[
P = W^\top (W W^\top)^\dagger W,
\]

with signed zero normalized before serialization. This identity is invariant to
component sign and orthogonal basis rotation. Reconstruction is measured in the
original feature units.

The synthetic reference splits complete dates into train, validation, and test
before any fit. Neural early stopping uses only an inner tail of the train
partition. A fixed downstream ridge model and diagnostic classifiers fit on the
outer train embeddings. Outer validation alone selects the representation; test
metrics are summarized only after selection is fixed. Stable named seeds make the
result independent of candidate iteration order.

## Security, privacy, and operational boundary

Arrays, temporal identities, schemas, finite values, dimensions, and resource
limits are validated before semantic use or large allocation. Unknown or missing
configuration fields fail closed. The boundary accepts no pickle, executable model
artifact, network input, credential, or broker authority.

The public publisher refuses an existing or symbolic-link destination, stages
within the destination parent, fsyncs JSON, and atomically renames a complete
directory. It publishes aggregate metrics, hashes, environment metadata, and a
Seaborn plot only. Raw feature rows, targets, embeddings, predictions, tensors,
weights, caches, and credentials remain absent.

## Consequences and alternatives

The contract makes leakage and resource assumptions inspectable and permits the
same candidate family to be evaluated later on licensed point-in-time data. It
also makes failure visible: reaching `max_epochs` is not reported as convergence,
and the reference result may reject neural capacity.

The reference does not implement robust principal-component pursuit, GPU
execution, checkpoint persistence, a hyperparameter sweep, or a trading strategy.
Those would add materially different estimators, trust boundaries, and selection
risk. A generic sklearn `Pipeline` was rejected as the public boundary because it
cannot express temporal identities, causal windows, reconstruction capability,
state identities, or the publication policy.

Rollback removes the representation package, frozen study, config, evidence, and
documentation. It does not alter the canonical market-data contract, feature
registry, prediction panels, backtester, or execution controls.
