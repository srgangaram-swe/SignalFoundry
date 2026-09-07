# Signal Foundry Sprint 3 — final evidence synthesis

## Executive outcome

Sprint 3 closes with an evidence-quality decision, not a model leaderboard:

| Decision field | Result |
| --- | ---: |
| Families evaluated | 10 |
| Reject current advancement evidence | 5 |
| Defer pending a matched experiment | 5 |
| Advance | 0 |
| Paper/live readiness | **NOT_READY** |
| Orders emitted / capital deployed | 0 / 0 |

`NOT_READY` is the only decision this synthesis component can emit. It grants
no authority to paper trade, route an order, access a broker, deploy capital, or
claim a profitable edge. “Reject” applies to advancement under the current
evidence; it does not discard a sound implementation. “Defer” records that the
available context cannot answer the required matched research question.

The [machine-readable summary](evidence/signal_foundry_sprint_3/decision/summary.json)
is the release decision source of truth; the
[evidence index](evidence/signal_foundry_sprint_3/decision/README.md) links its
content-addressed inputs and outputs. The
[governed decision narrative](multi_representation_decision.md) explains the
evidence gates and interpretation, while
[ADR 0008](adr/0008-context-preserving-sprint-3-synthesis.md) records why the
families must retain their original contexts.

## Ten-family context inventory

The constituent studies differ in dataset, generator, target, split, cost,
selection, uncertainty, and holdout boundaries. Their performance scalars are
therefore not commensurate and are intentionally absent from the final matrix.

| Family | Preserved evidence context | Disposition |
| --- | --- | --- |
| Conventional baselines | Historical Nasdaq WIKI engineering study | Reject |
| Spectral descriptors | Synthetic representation comparison | Defer |
| Adaptive decomposition | Synthetic mechanism and compute study | Defer |
| Regime and change-point models | Planted synthetic regimes | Defer |
| State-space models | Synthetic recovery and interval study | Defer |
| Deep-sequence models | Historical WIKI development fold | Reject |
| Time-frequency vision models | Synthetic chronological holdout | Reject |
| Latent representations | Synthetic chronological holdout | Reject |
| Governed ensembles | Synthetic chronological holdout | Defer |
| Abstention policy | Synthetic policy-mechanics study | Reject |

The [family evidence table](evidence/signal_foundry_sprint_3/decision/family_evidence.csv)
retains these dispositions, source hashes, and limitations. The
[gate matrix](evidence/signal_foundry_sprint_3/decision/gate_matrix.csv) and
[semantic-review receipt](evidence/signal_foundry_sprint_3/decision/semantic_review.json)
bind every positive gate to reviewed facts and machine-checkable locators.
Missing support remains false; “not applicable” cannot silently become
completed evidence.

## What the sprint established

- Historical and synthetic research paths now expose causal contracts,
  explicit contexts, bounded failure behavior, and reproducible aggregate
  evidence instead of notebook-only claims.
- Time-frequency progression failed closed when the mandatory small CNN missed
  its frozen validation gate; larger architectures remained blocked.
- Validation selected PCA in the latent-representation study, but the selected
  representation did not improve synthetic test prediction over the raw
  control.
- The governed ensemble study demonstrated deterministic recovery and
  abstention behavior in a deliberately complementary synthetic generator; it
  did not establish a persistent market edge.
- The standalone abstention policy reduced conditional synthetic loss
  frequency, turnover, and capacity demand, while the always-trade baseline
  retained higher total mean synthetic net value. The policy is a guard, not an
  alpha source.
- The final publisher verifies a strict plan and source identities, preserves
  negative and incomplete results, publishes only safe aggregates, and cannot
  emit orders or deploy capital.

These results demonstrate engineering and research-governance behavior. They
do not demonstrate current market predictability, investability, or expected
profit.

## Explicit re-scope

Two incomplete protocol dimensions remain visible rather than being
retroactively declared complete:

- AlphaForge issue #41 owns matched feature/parameter ablations plus
  feature-permutation, randomized-label, and representation-placebo controls.
- AlphaForge issue #42 owns matched year and temporal/regime stability with
  dependence-aware uncertainty, including portfolio/universe checks when a
  candidate exists.

Both issues belong to later work. Their absence is a readiness failure in this
release, not a reason to weaken the final gate.

## Cross-repository provenance

The committed
[Signalattice receipt](evidence/signal_foundry_sprint_3/cross_repository_provenance.json)
pins:

- repository `srgangaram-swe/Signalattice`;
- origin `https://github.com/srgangaram-swe/Signalattice.git`;
- commit `000ae12de3b409e5f409b53fb191aa003b105318`; and
- nine source records totaling 142,578 bytes, each with its Git blob identity,
  byte length, SHA-256 digest, family, and bounded claim scope.

AlphaForge CI validates the committed receipt’s strict schema, expected
repository/origin, pinned regression identity, formats, and resource bounds
without a sibling checkout or network request. That does not prove the external
blobs are present. Given an existing local Signalattice object database, verify
the exact pinned blobs separately and without fetching:

```bash
uv run python scripts/verify_sprint_3_cross_repository_sources.py \
  --signalattice /absolute/path/to/Signalattice
```

Signalattice issues #14–#16—hierarchical Bayesian inference, temporal graphs,
and posterior scenarios—remain separate, open Signalattice work at the pinned
commit. They are not completed AlphaForge families and are not hidden inside
this sprint’s result.

## Reproduction and release validation

The strict plan is
[`configs/sprint_3_decision.yaml`](../configs/sprint_3_decision.yaml). Its
SHA-256 is the plan identity. Loading it verifies exact fields, bounded resource
limits, the ten-family order, every AlphaForge source digest, and every positive
gate’s semantic locator before publication.

From a locked supported environment:

```bash
uv sync --locked --extra dev --extra data
uv run make config-check
uv run make check
uv run make sprint-3-decision-evidence \
  OUTPUT=runs/sprint-3-decision-replay
diff -r \
  docs/evidence/signal_foundry_sprint_3/decision \
  runs/sprint-3-decision-replay
```

The publisher accepts only a new repository-local destination, re-loads the
on-disk plan, re-verifies every source, bounds files, references, diagnostics,
document structure, artifact count, and total bytes, and publishes atomically.
Its [manifest](evidence/signal_foundry_sprint_3/decision/manifest.json) records
the plan and exact SHA-256/byte-length inventory for the six payload artifacts.
A locked replay must be byte-identical; any difference requires investigation,
not a golden-file refresh.

Before release, inspect the committed
[Seaborn evidence-coverage heatmap](evidence/signal_foundry_sprint_3/decision/plots/evidence_coverage.png)
at full resolution. Confirm that all ten families and nine governed gates are
legible, cells match `gate_matrix.csv`, the title defines `1` as reported
evidence and `0` as a missing gate, and nothing suggests that color or row
totals are performance scores. Also inspect each constituent Sprint 3 plot
linked from its source report; plot generation success alone is insufficient.

## Security and publication boundary

The committed synthesis contains aggregate CSV/JSON, explanatory Markdown,
hashes, and one Seaborn plot. It contains no licensed observations, row-level
targets or predictions, tensors, fitted state, model weights, credentials,
orders, positions, or capital instructions. Repository-relative regular files
are mandatory; symlinks, traversal, duplicate or conflicting identities,
oversized inputs, malformed documents, unverified locators, and overwrite
attempts fail closed.

The publisher is an evidence-only leaf. It has no dependency path into signal
generation, portfolio construction, paper controls, execution, broker
connectivity, or credential handling.

## Residual limitations and next gates

- The historical WIKI panel ends on 2018-03-27 and lacks complete revisions,
  delistings, symbol history, corporate actions, and point-in-time universe
  membership.
- Most Sprint 3 evidence is deterministic synthetic engineering evidence;
  planted structure cannot establish market persistence or capacity.
- Heterogeneous contexts prevent a defensible cross-family performance ranking.
- The issue #41 randomized/placebo/ablation controls and issue #42 matched
  year/regime stability evidence remain incomplete.
- Current licensed point-in-time data, calibrated execution costs, borrow and
  liquidity constraints, multiple-comparison control for the eventual complete
  family, and an untouched final interval are still required.
- Paper readiness additionally requires a bounded shadow period,
  reconciliation and recovery rehearsals, stale-data and kill-switch exercises,
  capital-at-risk limits, independent review, and explicit owner approval.

No Sprint 3 artifact should be used as a live-trading input or a claim of
future profit.
