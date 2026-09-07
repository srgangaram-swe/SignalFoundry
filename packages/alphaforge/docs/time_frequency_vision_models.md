# Gated time-frequency vision models (SF-S3-MR7)

This capability consumes the causal tensor contract produced by Signalattice
SF-S3-MR2 and evaluates image models under a fail-closed capacity progression.
It is disabled unless called explicitly and does not modify prior feature,
model, backtest, or evidence paths.

## Boundary and alignment

`TimeFrequencyBatch` accepts:

- `values`: `[sample, channel, frequency, within-window time]`, float32/float64;
- `observed_mask`: `[sample, channel]`, boolean;
- one date, symbol, and finite target per sample;
- conventional time-feature and spectral-descriptor matrices with identical
  row order;
- unique channel names, a strictly ascending finite frequency grid, and the
  representation name; and
- an explicit tensor-byte ceiling.

Observed surfaces must be finite. A masked surface may contain NaN, but the
study conservatively selects only rows with all requested channels observed.
Duplicate `(date, symbol)` rows fail. Tensor order is never inferred from a
join, filename, or dictionary.

The adapter intentionally does not duplicate Signalattice's STFT, wavelet,
normalization, content-addressing, or object-store logic. A licensed-data run
constructs the batch from a verified local Signalattice object plus its exact
alignment index and keeps that object local.

## Progression and gates

The fixed candidate order is:

1. LightGBM conventional time features;
2. LightGBM spectral descriptors;
3. small CNN;
4. ResNet; and
5. Vision Transformer.

The small CNN always runs first. Its validation metrics are compared with the
spectral LightGBM baseline under `small_cnn_to_resnet`. Only a passing signed
record can construct the ResNet. The same rule applies from ResNet to ViT.

For candidate \(c\) and baseline \(b\), every configured check must pass:

\[
n_c \ge n_{\min}, \quad d_c \ge d_{\min}, \quad
\mathrm{RIC}_c \ge r_{\min}, \quad
\mathrm{RIC}_c-\mathrm{RIC}_b \ge \Delta r_{\min}, \quad
\frac{\mathrm{RMSE}_c}{\mathrm{RMSE}_b} \le q_{\max}.
\]

All quantities come from the matched chronological validation interval. Test
metrics are structurally rejected by the gate API. The gate embeds its policy
SHA-256, metrics, checks, result, and evidence SHA-256. Mutating any field makes
verification fail.

## Training controls

Every neural model uses:

- training-only per-cell normalization and target scaling;
- deterministic CPU execution unless a specific available accelerator is
  requested;
- Huber loss, AdamW, gradient clipping, early stopping, and best-state restore;
- the same seed, batch, epoch, patience, tensor-byte, and parameter policies;
- measured wall/process CPU time, parameter count/bytes, loss history, best
  epoch, and termination state.

The small CNN and ResNet use bounded convolutions over the unmodified
frequency/time axes. The ViT uses bounded non-overlapping patches and can exist
only after both earlier stages earn it.

Augmentation is train-only. Positive log-amplitude rescaling tests power-scale
robustness. A contiguous frequency mask tests sensitivity to a missing band.
Neither operation changes sample, channel, time, frequency, or label order.

## Reproduction

Install all optional research dependencies, then run:

```bash
make time-frequency-evidence OUTPUT=/absolute/path/to/new/evidence
```

The committed profile
`configs/time_frequency_vision_benchmark.yaml` freezes data geometry, random
seed, chronological fractions, both LightGBM and neural budgets, cost
diagnostic, augmentation, and both gate policies. Publication is atomic and
refuses an existing or stale destination.

The deterministic reference is synthetic by design. It tests a known
frequency-localized response with 80 dates and 12 symbols. It is useful for
checking mechanics and rejection behavior, not for estimating a market edge.

## Evidence and observed decision

The reference evidence is in
`docs/evidence/signal_foundry_sprint_3/time_frequency_vision/`.

- Training/validation/test contain 576/192/192 observations across
  48/16/16 dates.
- The spectral LightGBM validation rank IC was 0.5988 with RMSE 0.00464.
- The mandatory small CNN validation rank IC was 0.0769 with RMSE 0.00895.
- The small-CNN gate failed incremental rank IC and RMSE-ratio checks.
- ResNet and ViT were therefore not trained; their report rows retain the
  exact blocked reason and no metric.
- On the untouched synthetic test, spectral LightGBM rank IC was 0.6665
  (daily-rank-IC SE 0.0466), while small CNN was 0.2024 (SE 0.1171).

This is a rejection, not a failed pipeline. The controls prevented two
unjustified experiments after a simple descriptor model explained the
synthetic response more effectively.

## Residual limitations

- Synthetic evidence is not financial evidence.
- The reference has 16 test dates; its standard errors are descriptive and
  rely on limited daily observations.
- The costed long-short statistic is a diagnostic, not event-driven execution.
- Saliency is not a causal explanation and is intentionally absent.
- A later licensed point-in-time study must pre-register its batch identity,
  folds, hypotheses, costs, multiple-testing family, and untouched holdout.
- No result here qualifies paper or live trading.
