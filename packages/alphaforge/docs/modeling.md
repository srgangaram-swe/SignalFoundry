# Modeling

## Models

- Zero, historical mean, and momentum baselines — every ML model must beat these.
- Linear regression, ridge, lasso, elastic net, and robust Huber regression
  (imputation + scaling
  embedded in the pipeline, so statistics are always train-window-only).
- Random Forest, Extra Trees, bounded histogram gradient boosting, and a
  controlled small MLP. Every stochastic backend receives an order-independent
  seed and every worker, depth, tree, layer, and iteration setting is bounded.
- LightGBM is the primary optional tabular benchmark; XGBoost and CatBoost are
  explicit independent comparisons. Installing an optional package never
  silently changes another registry name's estimator semantics. See
  [Governed benchmark models](governed_benchmark_models.md) for mathematics,
  dependency boundaries, termination evidence, and trading limitations.
- Optional PyTorch MLP, GRU, and temporal CNN preserve the original temporal
  research interfaces. The separate
  [controlled deep-sequence family](deep_sequence_benchmarks.md) compares a
  causal CNN, TCN, masked LSTM, masked GRU, and causal Transformer under one
  bounded training, validation, resource, persistence, and costed-OOS policy
  (ADR 0003).
- **TemporalAlphaNet** (`alphaforge/models/temporal.py`, ADR 0002): the
  flagship neural model — dilated causal TCN encoder with attention pooling,
  optional multi-task horizon heads, and a composite Huber + cross-sectional
  IC loss on date-batched mini-batches. Trained by a real loop (AdamW +
  cosine decay, gradient clipping, early stopping on validation rank IC,
  best-checkpoint restore, persisted per-epoch history) via
  `scripts/train_model.py` (`make train`), and available in walk-forward
  comparisons as `temporal_alpha`.
- **Latent representations** (`alphaforge/representations/`, ADR 0005):
  standardized raw features, full/incremental/robust-scaling PCA, dense,
  sequence, denoising, and variational autoencoders, and a causal contrastive
  encoder behind a target-free fit/transform/reconstruct contract. Every
  learned transform fits inside the training interval, publishes stable
  identities and resource/termination evidence, and is compared with fixed
  prediction, regime, similarity, anomaly, diversification, and reconstruction
  diagnostics. See [Leakage-safe latent representations](latent_representations.md).
- **Governed temporal-OOF ensembles**: immutable complete-date expert
  predictions feed static blend, rank/vote, genuine temporal-OOF ridge
  stacking, Bayesian averaging, causal dynamic weighting, and regime-gated
  policies. Final-holdout inference is target-free; missing experts and
  uncertain regimes produce typed abstentions. Stable state/audit identities
  and aggregate correlation, marginal-contribution, turnover/cost, and
  uncertainty evidence are described in
  [Governed temporal-OOF ensembles](governed_ensembles.md) and ADR 0006.
  The historic `ensemble` registry entry remains an equal/IC compatibility
  adapter and is not the advanced-policy training boundary.
- **Regime and change-point contracts** (`alphaforge/regimes/`, SF-S3-MR4):
  rule-based, GMM, K-state HMM, CUSUM, and Bayesian online change-point
  models behind one causal contract, plus retrospective segmenters that are
  deliberately not usable as features, and an incremental-value harness that
  measures a regime against a no-regime baseline. See
  [Regime contracts](regimes.md).
- **Gaussian HMM regime model** (`alphaforge/models/regime.py`): 2-state
  Baum-Welch EM from scratch. Used causally — expanding parameter refits and
  filtered (never smoothed) state probabilities — as a feature
  (`hmm_stress_prob`) and for regime-gated exposure.

The registry is config-driven. Walk-forward validation instantiates a fresh
model per window, fits only on training rows, and emits predictions only for
test rows. Its metrics also publish backend, convergence/completion status,
iteration budget, seed, and warning count for governed estimators.

## Validation

- **Walk-forward** (primary): expanding or rolling windows with an embargo
  at least as long as the longest label horizon.
- **Interval-aware development plan**: explicit train, validation, test,
  purge, embargo, overlap, and final-holdout roles. Exact label-event ends must
  precede the next protected boundary; crossing samples are excluded. See
  [Temporal validation contract](temporal_validation.md).
- **Purged K-Fold / CPCV** (`alphaforge/training/purged_cv.py`): purging
  removes train dates whose exact label intervals overlap each contiguous test
  block; the embargo limits trailing-window dependence. CPCV evaluates every
  C(n, k) test-group combination, producing many OOS paths.

## Overfitting statistics (alphaforge/evaluation/overfitting.py)

- **PSR** — P(true Sharpe > benchmark), adjusted for sample length, skew, kurtosis.
- **DSR** — PSR against the expected best-of-N unskilled Sharpe; `n_trials`
  counts every model variant that competed for selection.
- **PBO (CSCV)** — probability the in-sample winner underperforms the median
  out-of-sample, computed from per-date rank-IC panels across models.
- **Newey-West t-stats** — IC series are serially correlated under
  overlapping multi-day labels; HAC errors keep significance honest.
