# ADR 0004: Gate time-frequency vision-model capacity

- Status: Accepted
- Date: 2026-07-26
- Decision owners: AlphaForge maintainers
- Depends on: Signalattice SF-S3-MR2 and AlphaForge ADR 0003

## Context

Signalattice now produces causal spectrogram/scalogram tensors with shape
`[sample, channel, frequency, within-window time]`, explicit missingness,
train-fit normalization provenance, immutable identities, and bounded storage.
AlphaForge needs to test whether their spatial structure adds value without
letting model novelty, capacity, or test-set inspection determine the result.

Running a ResNet or Vision Transformer simply because the tensor looks like an
image would be selection bias. It would also spend more compute before a small
image baseline establishes that the representation is usable.

## Decision

Add an opt-in, downstream consumer with this fixed progression:

1. LightGBM on conventional time features;
2. LightGBM on tensor-derived spectral descriptors;
3. mandatory small CNN;
4. ResNet only after a passing small-CNN validation gate; and
5. ViT only after a passing ResNet validation gate.

`ProgressionGateEvidence` binds the candidate, matched baseline, metrics,
partition, threshold-policy identity, individual check results, and overall
decision under a canonical SHA-256 digest. A gate rejects test-partition
metrics, a changed policy, a failed predecessor, an unexpected predecessor, a
tampered record, or an inconsistent summary. There is no boolean bypass.

The chronological test partition is accessed only after the progression has
finished. A blocked architecture appears in the aggregate report with its
reason and no fabricated metric.

## Mathematical and data invariants

For sample \(i\), the input is
\(X_i \in \mathbb{R}^{C \times F \times T}\), where frequency and within-window
time retain Signalattice's order and units. The model never rebuilds or
reorders the upstream transform. A row is usable only when every requested
channel is observed; masked values are never silently imputed.

Per-bin mean and standard deviation and target location/scale are fitted on
training rows only. Validation chooses a checkpoint and architecture
progression. The chronological test interval chooses neither.

Train-only perturbations are limited to:

- positive log-normal channel-amplitude rescaling, which preserves sign,
  channel, frequency, time, sample, and label order; and
- one bounded contiguous frequency-band mask, which models an unavailable band
  without reversing, translating, or permuting temporal content.

Time reversal, sample mixing, arbitrary crops, rotations, image flips, and
channel permutation are prohibited because they change financial semantics.

All neural candidates use the same Huber loss, AdamW optimizer, gradient
clipping, deterministic seed, best-checkpoint restoration, explicit CPU
default, tensor-byte ceiling, parameter ceiling, epoch limit, patience, and
batch limit. The ResNet uses bounded residual blocks. The ViT uses bounded
non-overlapping patches and a bounded encoder; it has no privileged exception
to the gate.

## Security, privacy, and operational boundary

The consumer validates dimensions, dtypes, finiteness, frequency ordering,
unique channel names, one-to-one date/symbol/label alignment, duplicate sample
identities, masks, feature matrices, and allocation size before semantic use.
No pickle or model artifact is accepted by this boundary.

Licensed tensor cells, rows, targets, credentials, checkpoints, caches, and
predictions remain local. Public evidence contains aggregate metrics, gate
records, resource data, limitations, and a Seaborn plot only.

These models cannot access a broker or authorize paper/live execution. A gate
authorizes one research architecture, not capital.

## Consequences

The architecture path is auditable and cheap failures stop early. This can
produce a less visually impressive result—as the reference run did—but it
prevents unsupported capacity and test-set tuning. Rollback removes the
standalone consumer, study, config, and evidence without changing Signalattice
tensors, AlphaForge's tabular/sequence models, or earlier research evidence.
