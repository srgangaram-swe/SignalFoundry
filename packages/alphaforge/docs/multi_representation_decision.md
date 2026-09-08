# Governed multi-representation decision

SF-S3-MR11 closes AlphaForge Sprint 3 with an evidence-quality synthesis, not a
model leaderboard. The constituent studies do not share one dataset, target,
fold family, cost boundary, or protected holdout. Their performance values must
be interpreted in their source reports and must not be ranked across contexts.

The reference synthesis reaches one operational conclusion:

> **NOT_READY.** No Sprint 3 result authorizes paper trading, live trading,
> executable orders, capital deployment, or a profitability claim.

“Reject” below means reject advancement under the current evidence. It does not
mean that the implementation or research mechanism should be deleted. “Defer”
means the required matched experiment does not exist yet or its context cannot
answer the advancement question.

## Evidence-gate contract

The final matrix reports whether a family provides each category through a
content-addressed source and a verified locator:

| Gate | Required meaning |
| --- | --- |
| Out-of-sample | A chronologically separated evaluation not used to fit the reported state |
| Uncertainty | Variability or interval evidence appropriate to the reported metric and temporal dependence |
| Net economics | A cost-aware result with units and an explicit limitation boundary |
| Selection correction | Complete-family correction or another predeclared control for model selection |
| Feature ablation | A matched feature/representation removal experiment, not merely a model drop-one diagnostic |
| Randomized control | A predeclared placebo, permutation, or randomized-label/control experiment |
| Regime stability | Matched evidence across regimes rather than a planted-regime recovery demonstration alone |
| Year stability | Matched evidence across calendar years or comparable temporal blocks |
| Compute accounting | Bounded compute/resource evidence for the evaluated boundary |

A gate with no qualifying source remains false. A synthetic result can satisfy a
mechanical gate, but it cannot become historical market evidence. The synthesis
copies no “best” rank IC, return, or Sharpe value because those quantities are
not commensurate across the studies.

This matrix is a retrospective synthesis of already completed constituent
experiments. The final inventory, source hashes, gate classifications,
dispositions, and readiness policy are frozen before synthesis publication.
That freeze is not retroactive preregistration of any constituent experiment;
only the original constituent configuration, ledger, or source artifact can
establish what was fixed before that experiment ran.

## Frozen synthesis decision

Gate columns are abbreviated in the order OOS, uncertainty, net economics,
selection correction, feature ablation, randomized control, regime stability,
year stability, and compute accounting.

| Family | Context | Current-evidence disposition | OOS | U | Net | Sel | Abl | Rand | Reg | Year | Compute |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conventional baselines | Historical WIKI engineering | Reject paper advancement | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| Spectral descriptors | Synthetic representation comparison | Defer matched incremental-value claim | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| Adaptive decomposition | Synthetic mechanism and compute only | Defer; economic experiment not run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Regime/change points | Planted synthetic regimes | Defer matched economic and stability study | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| State space | Synthetic recovery and interval study | Defer market evaluation | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Deep sequence | Historical WIKI development fold | Reject current advancement evidence | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Time-frequency vision | Synthetic chronological holdout | Reject progression; small-CNN gate failed | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| Latent representations | Synthetic chronological holdout | Reject learned-representation promotion | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Governed ensembles | Synthetic chronological holdout | Defer market evaluation | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Abstention policy | Synthetic policy mechanics | Reject current thresholds as an improvement | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |

The detailed interpretation is deliberately conservative:

- The governed Sprint 2 historical study rejected paper advancement after
  Holm–Bonferroni correction.
- The spectral synthetic descriptor result does not replace the unrun
  predeclared Signalattice incremental-value study.
- Adaptive decomposition has reconstruction, stability, and compute contracts,
  but no matched costed predictive study.
- Regime work proves causal state contracts and planted-regime recovery; it
  does not provide the predeclared out-of-sample incremental-value experiment.
- State-space evidence demonstrates synthetic recovery and interval behavior,
  not alpha or market robustness.
- Deep-sequence diagnostics use one development fold, lack a complete
  dependence-aware, multiplicity-corrected study, and omit required LightGBM
  resource fields; their compute-accounting gate therefore remains false.
- The small CNN failed its validation progression gate, so ResNet and ViT were
  correctly blocked.
- PCA was selected on validation, tied the raw control on synthetic-test rank
  IC, and worsened prediction MSE. A different representation’s post-selection
  test value cannot revise that decision.
- Ensemble experts were deliberately complementary in a synthetic generator.
  Their favorable regime-gated result is not evidence of a persistent market
  edge.
- The abstention policy lowered conditional loss frequency, turnover, and
  capacity demand in its generator, but the always-trade baseline produced
  higher total mean synthetic net value. Abstention is a guard, not an alpha
  source or guarantee against loss.

## Protocol re-scope

The final synthesis plan content-addresses the exact ten-family inventory,
source evidence, gate decisions, dispositions, protocol status, and readiness
thresholds before aggregate publication. It preserves constituent
configurations, folds, costs, and compute evidence where the original sources
actually recorded them; it does not claim that the final plan preregistered
experiments that had already completed. Two planned dimensions were not
completed and are therefore explicitly re-scoped rather than claimed:

- matched feature/parameter ablations and feature-permutation,
  randomized-label, and representation-placebo controls continue under Sprint
  4 issue #41;
- matched year and temporal/regime stability with dependence-aware uncertainty
  continues under Sprint 4 issue #42.

These gaps remain visible in the final matrix and are readiness failures.

Signalattice’s hierarchical Bayesian, temporal-graph, and posterior-scenario
issues (#14–#16 in that repository) are also not implemented at the pinned
cross-repository evidence commit. They are not represented as completed
AlphaForge families or hidden inside another result.

## Data and readiness boundary

The only historical evidence uses the ten-symbol Nasdaq WIKI engineering panel,
which ends on 2018-03-27. It is stale, current-vintage data with incomplete
corporate actions, revisions, delistings, symbol history, and point-in-time
universe membership. The synthetic studies use deliberately constructed
signals and cannot demonstrate market predictability.

Paper advancement additionally lacks current licensed point-in-time data,
complete corporate actions and universe history, calibrated execution costs,
matched year/regime robustness, a completed paper shadow period, and broker
failure/reconciliation rehearsal. The publisher contains no broker adapter and
cannot emit an order.

## Reproducibility and provenance

The final plan is `configs/sprint_3_decision.yaml`. Its SHA-256 is the plan
identity. Loading the plan verifies every AlphaForge source hash and every
positive gate locator. The Signalattice receipt at
`docs/evidence/signal_foundry_sprint_3/cross_repository_provenance.json` pins
repository `srgangaram-swe/Signalattice`, commit
`000ae12de3b409e5f409b53fb191aa003b105318`, Git blob identities, byte lengths,
and SHA-256 digests. AlphaForge CI loads that committed receipt without a
sibling checkout and validates its schema, expected repository/origin, hash
formats, and declared size bounds. CI does not claim to have read the external
Signalattice blobs. Verify those exact blobs explicitly against an existing
local Signalattice object database, without a network request:

```bash
python scripts/verify_sprint_3_cross_repository_sources.py \
  --signalattice /absolute/path/to/Signalattice
```

Publish the aggregate decision into a new directory:

```bash
python scripts/publish_sprint_3_decision.py \
  --config configs/sprint_3_decision.yaml \
  --output runs/sprint-3-decision-replay
```

Publication is atomic and refuses overwrite. The committed reference contains
only aggregate CSV/JSON, a manifest, explanatory Markdown, and a Seaborn
evidence-coverage plot. Licensed rows, targets, predictions, fitted state,
weights, tensors, credentials, orders, and positions remain absent.

## Published reference

The [final Sprint 3 report](sprint_3_report.md) records the release outcome,
validation procedure, security boundary, and residual gates. The
[decision-evidence index](evidence/signal_foundry_sprint_3/decision/README.md)
links the exact aggregate payload:

- [family dispositions and source identities](evidence/signal_foundry_sprint_3/decision/family_evidence.csv);
- [gate coverage and semantic locators](evidence/signal_foundry_sprint_3/decision/gate_matrix.csv);
- [semantic-review receipt](evidence/signal_foundry_sprint_3/decision/semantic_review.json);
- [machine-readable outcome](evidence/signal_foundry_sprint_3/decision/summary.json); and
- [content manifest](evidence/signal_foundry_sprint_3/decision/manifest.json).

![Sprint 3 evidence coverage](evidence/signal_foundry_sprint_3/decision/plots/evidence_coverage.png)
