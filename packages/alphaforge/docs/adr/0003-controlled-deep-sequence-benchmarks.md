# ADR 0003: Controlled deep-sequence benchmark family

- Status: Accepted
- Date: 2026-07-26
- Decision owners: AlphaForge maintainers
- Supersedes: the benchmark-family exclusion in ADR 0002; ADR 0002 remains
  authoritative for `TemporalAlphaNet`

## Context

ADR 0002 selected a causal TCN with attention pooling as AlphaForge's focused
temporal research model. Sprint 3 requires a different question: under the
same data, validation, preprocessing, optimization, resource, cost, and
reporting decisions, do basic sequence architectures add value relative to
the frozen LightGBM benchmark?

Ad hoc implementations cannot answer that question. Different window
boundaries, validation fractions, parameter counts, or hardware would
confound architecture with experimental policy. Conventional padded recurrent
layers can also update hidden state during padded steps, and symmetric
convolutions can silently violate causality.

## Decision

Add a separate, opt-in controlled family:

- `sequence_cnn`: one causal convolutional block;
- `sequence_tcn`: a bounded stack of exponentially dilated causal residual
  blocks;
- `sequence_lstm` and `sequence_gru`: masked recurrent cells that leave state
  unchanged during left padding;
- `sequence_transformer`: a bounded Transformer encoder with an explicit
  upper-triangular attention mask and padding mask.

All variants use `SequenceBenchmarkConfig`, per-symbol windows from
`build_causal_windows`, a chronological inner train/validation split,
inner-train-only standardization and target scaling, Huber loss, AdamW,
gradient clipping, deterministic seed streams, identical early-stopping
semantics, best-checkpoint restoration, and explicit limits on windows,
parameters, sequence length, epochs, and batch size. `auto` resolves to CPU;
accelerators are opt-in and unavailable explicit devices fail closed.

Each fit publishes parameter and byte counts, wall and process CPU time,
accelerator peak allocation when available, convergence history, best epoch,
and validation loss. The walk-forward boundary carries these fields beside
the existing predictive metrics. `evaluate_oos_predictions` applies the same
descriptive regression calibration and simple costed long-short diagnostic to
each candidate and LightGBM.

The original `torch_gru`, `torch_tcn`, and `TemporalAlphaNet` interfaces remain
available. This decision adds a governed comparison surface; it does not
delete, rename, or reinterpret earlier evidence.

## Mathematical and causal invariants

For asset \(s\) and decision time \(t\), a window contains only
\(\{x_{s,u}: u \le t\}\). No tensor contains another asset. Short histories
are left-padded and accompanied by a validity mask \(m\); padding never updates
recurrent state and is zeroed after convolutional blocks. Transformer
attention from position \(i\) to \(j > i\) is prohibited.

Given training dates \(D\), the last declared fraction forms inner validation
\(D_v\). Scaling statistics are fitted only where \(d < \min D_v\).
Validation selects an epoch but never changes the transform. An outer
walk-forward test fold remains untouched until prediction.

The common loss is Huber loss on training-standardized targets. It bounds the
influence of large residuals without claiming a Gaussian return distribution.
Architecture comparison is valid only on matched outer folds and with the
same cost and selection policies. Parameter count and compute remain reported
confounders, not proof that the largest model is best.

## Security, persistence, and operational boundary

Inputs are treated as adversarial: shape, index uniqueness, numeric
convertibility, infinities, target alignment, window count, and every public
resource setting are validated before semantic use or allocation. Tensor
allocations are estimated and rejected against an explicit byte budget before
window arrays are created. Model
artifacts use the existing binary AlphaModel persistence boundary and therefore
require `trusted=True`; callers must verify provenance and integrity before
deserialization. No dataset, credential, model checkpoint, row-level
prediction, or training cache is committed.

These models have no broker, paper-execution, or live-order capability. A
favorable historical result cannot authorize capital. Point-in-time data,
multiple-testing correction, final-holdout discipline, realistic execution,
stability, paper duration, operational controls, and owner approval remain
independent gates.

## Consequences

The family creates an apples-to-apples architecture benchmark and makes
resource cost visible. CPU training is slower than opportunistic accelerator
selection, but it is reproducible and prevents hardware availability from
silently changing experiment identity. Left-padding and explicit masks add
implementation complexity that is justified by testable boundary invariants.

Rollback removes the five new registry entries and the standalone module.
Earlier temporal code, configurations, checkpoints, and evidence remain
unchanged.
