# ADR 0002: A neural temporal alpha model with an explicit training protocol

- Status: Accepted
- Date: 2026-07-11
- Owners: AlphaForge research platform

## Context

The model zoo (baselines, regularized linear, trees, small torch prototypes)
treats each row as an independent tabular observation. The lagged/rolling
features encode *some* history, but no model consumed the raw temporal
structure of the feature sequence, none optimized the objective the portfolio
actually monetizes (cross-sectional ranking), and there was no inspectable
training loop — the torch prototypes buried a fixed-epoch loop with no
checkpointing, no history, and no evaluation artifacts. That gap blocked
credible claims about "ML for time series" and made hyperparameter work
unreviewable.

## Decision

`alphaforge/models/temporal.py` adds **TemporalAlphaNet**:

- **Encoder**: stack of pre-norm dilated causal Conv1d residual blocks
  (dilations 1, 2, 4, 8 → ~61-day receptive field) over per-symbol sequences
  of daily feature vectors. Causality is enforced by left-only padding — a
  structural property, not a convention.
- **Pooling**: learned attention over time steps instead of last-step or
  mean pooling.
- **Heads**: primary target head plus optional auxiliary horizon heads
  (multi-task) sharing the encoder.
- **Loss**: Huber on standardized returns + a differentiable cross-sectional
  IC penalty (negative per-date Pearson correlation), computed on
  date-batched mini-batches so every batch contains complete cross-sections.
- **Training loop**: chronological train/validation split by date, AdamW with
  cosine decay, gradient clipping, early stopping on validation rank IC,
  best-checkpoint restore, per-epoch history persisted for plots, and
  deterministic seeding. Checkpoints round-trip via `save()`/`load()`.
- **Integration**: the walk-forward and purged-CV drivers dispatch sequence
  construction on a `needs_sequence_index` model attribute (replacing
  name-prefix matching). `scripts/train_model.py` is the standalone protocol:
  train+val | embargo | single-touch test, with artifacts and plots.
- **Evaluation plots**: `alphaforge/visualization` renders training curves,
  IC time series, IC decay, quantile returns, model comparison, and
  prediction-vs-realized density into `runs/<id>/plots/`, embedded in the
  markdown report.

Architecture alternatives considered: GRU/LSTM (sequential training, weaker
long-range gradients at equal budget), transformer encoder (quadratic
attention cost and data appetite unjustified at ~10^5 daily samples;
attention pooling captures most of the benefit at a fraction of the
parameters). The TCN choice is revisitable behind the same interface.

## Consequences

- Torch stays an optional extra; the registry raises an actionable error and
  tests skip when it is absent. CI installs it to run the full suite.
- A single chronological split (train_model.py) is a *development* tool; the
  walk-forward driver remains the statistically serious evaluation, and the
  temporal model participates in it unchanged.
- Synthetic-market results remain engineering verification, not market
  evidence; the same entry points run on real (yfinance/CSV) data via
  configs/data.yaml.
