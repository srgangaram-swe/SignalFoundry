# ADR 0008: Preserve evidence context in the Sprint 3 decision

- Status: Accepted
- Date: 2026-07-26
- Decision owners: AlphaForge maintainers
- Depends on: ADRs 0001–0007 and the governed Sprint 2 baseline study
- Tracks: AlphaForge issue #37

## Context

Signal Foundry Sprint 3 produced several legitimate but non-comparable evidence
boundaries. The conventional and deep-sequence studies use the stale Nasdaq
WIKI engineering panel. The vision, latent, ensemble, abstention, and
cross-repository state-space references use different deterministic synthetic
generators. Regime and adaptive-decomposition work establishes causal
mechanisms and failure behavior without completing the matched economic
experiments described in their operator documentation.

Putting their rank correlations or returns in one leaderboard would erase
dataset, split, target, cost, selection, and uncertainty differences. Selecting
the largest number would also introduce a new post-holdout trial that was never
pre-registered. Missing randomized controls, feature ablations, or year
stability cannot be converted into completed evidence by declaring them
“not applicable.”

The final synthesis also sits near an operational boundary. It must not turn
aggregate research claims into paper authority, executable orders, or capital
deployment.

## Decision

The Sprint 3 publisher evaluates an exact, ordered ten-family inventory:

1. conventional baselines;
2. spectral descriptors;
3. adaptive decomposition;
4. regime and change-point models;
5. state-space models;
6. deep-sequence models;
7. time-frequency vision models;
8. latent representations;
9. governed ensembles; and
10. the abstention policy.

Each family retains one of three evidence contexts: historical engineering,
synthetic engineering, or unsupported. Performance scalars are not copied into
the synthesis table. Instead, the table records a disposition of the current
evidence—advance, reject, or defer—and coverage of nine governed evidence
categories:

- out-of-sample evaluation;
- dependence-aware uncertainty;
- net economics;
- selection correction;
- feature ablation;
- randomized control;
- regime stability;
- year stability; and
- compute accounting.

A positive gate requires a content-addressed source plus a machine-checkable
JSON pointer, CSV column, or exact Markdown heading. Missing evidence remains a
zero in the matrix. An “advance” disposition is structurally invalid unless all
nine gates have support. The reference decision advances no family.

The final YAML is frozen after the constituent experiments and before aggregate
synthesis publication. It fixes the complete family, source hashes, gate
classifications, dispositions, protocol status, readiness facts, and evidence
policy for this final synthesis. It does not retroactively preregister or freeze
an already-completed constituent experiment; only that experiment's original
configuration, ledger, or source artifact can establish pre-execution intent.
Protocol dimensions that were not completed are recorded as `deferred` with an
explicit reason and follow-up scope; they are not self-attested as frozen.
Feature and parameter ablations, feature permutation, randomized-label controls,
and representation placebos continue under Sprint 4 issue #41. Matched temporal
and year/regime robustness with dependence-aware uncertainty continues under
issue #42.

Cross-repository Signalattice claims use a strict receipt pinned to repository,
origin, commit, Git blob, byte length, and SHA-256. A local verifier reads the
pinned Git object database without fetching, checking out, or trusting current
worktree contents. Without a sibling checkout, AlphaForge CI loads the committed
receipt network-free and validates its schema, expected repository/origin, hash
formats, declared size bounds, and pinned receipt regression hash. CI does not
claim that this validates the external blobs. Exact Signalattice commit/blob,
byte-length, and SHA-256 verification remains an explicit local/manual step
through `scripts/verify_sprint_3_cross_repository_sources.py`.

The synthesis decision is unconditionally `NOT_READY`. Even a hypothetical
configuration with every research and operational boolean true cannot grant
paper or order authority through this component. A later paper-readiness
decision requires its own reviewed contract, current point-in-time evidence,
operational rehearsal, and human approval.

Architecturally, the publisher is an evidence-only leaf from the research
governance boundary. It reads the strict plan and content-addressed source
artifacts and writes only the bounded aggregate allowlist. It neither imports
nor invokes signal generation, portfolio construction, paper controls,
execution, broker connectivity, credential handling, or capital state. There
is intentionally no supported path from a synthesis disposition to an order.

## Security, data, and reproducibility

- Sources and the plan are regular repository-relative files; symlinks, parent
  traversal, conflicting hashes, unsafe identifiers, oversized files, excessive
  source counts, and unbounded aggregate bytes fail closed.
- The publisher re-loads the on-disk content-addressed plan instead of trusting
  an arbitrary in-memory object.
- Publication is atomic, refuses overwrite, cleans staging after faults, and is
  deterministic byte for byte.
- Public artifacts contain aggregate claims and hashes only. They contain no
  licensed observations, row targets, predictions, tensors, fitted weights,
  credentials, orders, positions, or capital instructions.
- The evidence plot is a Seaborn gate-coverage heatmap. Its cells mean
  “reported with a verified locator” or “missing”; they are not performance
  scores.

## Consequences

The final result is less promotional than a cross-model leaderboard, but its
meaning survives expert review. Strong synthetic results remain useful
engineering evidence without becoming market claims. Negative results and
blocked progressions remain visible. Unfinished controls have named follow-up
work instead of disappearing.

Rollback removes the standalone plan, publisher, receipt verifier, aggregate
decision evidence, and report. It does not mutate constituent Sprint 2/3
evidence, model implementations, execution contracts, or local vendor data.
The released interpretation and reproduction procedure are recorded in the
[final Sprint 3 report](../sprint_3_report.md).
