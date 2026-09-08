# Changelog

Notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
semantic versioning. AlphaForge is pre-1.0; research and artifact contracts may
still evolve between minor releases.

## [Unreleased]

### Added

- Immutable, content-addressed fill-cost, execution-policy, carry-cost,
  logical-session latency, and adverse stress-profile contracts with strict
  units, bounds, calibration provenance, failure behavior, and limitations.
- Event-ledger financing and per-symbol short-borrow accrual, categorized
  commission and exchange fees, component-level spread/slippage/impact
  attribution, causal latency schedules, and exact cross-table reconciliation.
- A deterministic CPU-only execution-friction benchmark plus full-rerun
  baseline, doubled/tripled-cost, adverse-spread, reduced-liquidity,
  delayed-signal, partial-fill-pressure, and capacity-scaling profiles.

### Changed

- Governed research now charges borrow and financing exactly once through
  replayable cash-charge events instead of applying a post-hoc return proxy.
- Backtest run artifacts now publish friction model manifests, normalized
  component attribution, and logical latency schedules while preserving the
  version-1 order and fill table schemas.

## [0.3.0] - 2026-07-26

### Added

- A bounded, content-addressed Sprint 3 synthesis across ten heterogeneous
  research families, with verified semantic evidence locators, explicit missing
  gates, a reproducible Seaborn coverage heatmap, and a conservative outcome of
  five rejected, five deferred, zero advanced, and `NOT_READY`.
- A strict cross-repository provenance receipt pinned to Signalattice commit
  `000ae12de3b409e5f409b53fb191aa003b105318`, including exact Git blobs, byte
  lengths, SHA-256 digests, bounded network-free CI validation, and an explicit
  local Git-object verification command.
- A final Sprint 3 report that preserves historical, synthetic, and unsupported
  evidence contexts without constructing a misleading cross-context
  leaderboard or reopening a protected holdout.
- A pure, immutable cost/uncertainty decision-policy boundary with stable
  reason codes and identities, fail-closed freshness/disagreement/regime/drift
  gates, strict configuration, deterministic always/never-trade aggregate
  evidence, and no order-routing authority.
- Schema-1.0 financial-label contracts for regression, classification,
  threshold, quantile, triple-barrier, volatility-scaled, and meta-label
  targets, with deterministic identities, normalized future-event intervals,
  holdout protection, statistical diagnostics, and reproducible synthetic
  Seaborn evidence.
- A versioned semantic feature registry, deterministic content-addressed
  feature cache with verified lineage, and train-fold-only fitted-transform
  state for walk-forward and governed final-holdout workflows.
- Independent Signal Foundry schema 1.1 universe-membership and
  corporate-action validation, decision-time revision views, explicit
  entry/exit reconstruction, and point-in-time risk diagnostics.
- Versioned experiment manifests with canonical configuration identities, full
  Git provenance, immutable dataset fingerprints, named deterministic seed
  streams, exact dependency versions, safe runtime metadata, redacted
  invocation arguments, and hashed artifact inventories.
- A committed universal `uv.lock` plus locked CI and container installation for
  reproducible dependency resolution across supported Python versions.
- A leakage-safe latent-representation layer with raw/PCA controls, bounded
  optional autoencoder and causal contrastive families, stable fitted-state and
  PCA-subspace identities, validation-only selection, downstream transfer
  diagnostics, and aggregate synthetic Seaborn evidence.

### Changed

- Signal Foundry as-of loading now applies both economic-effective and
  information-availability cutoffs to every record family while preserving
  schema 1.0 compatibility.
- Governed Signal Foundry run manifests now use the versioned `2.0.0` wrapper
  and separate semantic experiment identity from execution and result metadata.
- Added strict frozen schemas and cross-field validation for every supported
  YAML configuration, including model-parameter allowlists and a repository
  configuration gate.
- Replaced executable pickle pipeline interchange with atomic, resource-bounded
  versioned JSON Table Schema artifacts.
- Migrated evaluation graphics to Seaborn's plotting and theme APIs with a
  colorblind palette and explicit sprint visual-evidence guidance.
- Added locked CI import coverage for the data, ML, and application extras.
- Aligned package, API, lock, wheel, and container version assertions at
  `0.3.0`.

## [0.2.1] - 2026-07-23

### Added

- Independent, fail-closed consumption of the Signal Foundry v1 market-data
  contract, including semantic identity, partition hash, schema, license,
  temporal, and point-in-time validation.
- Separate market and decision-eligible panels so after-close publication
  delays cannot leak a bar into a same-close feature or order decision.
- Pre-registered development selection with a purged pre-holdout embargo and
  one immutable final-holdout evaluation per bundle/code/config identity.
- Hash-chained trial ledger, cost/liquidity/latency/placebo stresses, and a
  machine-readable `READY_FOR_PAPER` or `NOT_READY` dossier.
- Offline paper-decision controls for idempotency, stale data, exposure,
  position, turnover, notional, drawdown, daily loss, and a one-way kill switch.
- Typed close-decision/future-open order and fill contracts.
- Self-financing signed-share/cash ledger with daily and symbol-level P&L reconciliation.
- Lagged-ADV participation caps, partial DAY fills, and square-root impact sensitivity.
- Auditable orders, fills, holdings, P&L attribution, and capacity-scenario artifacts.
- CI type checking, wheel smoke installation, Python 3.12–3.14 matrix, and branch coverage gate.

### Changed

- Backtesting and paper replay now share the same causal daily-bar execution policy.
- Target weights drift between explicit rebalances; restoring a target generates costed trades.
- Research CLI runs explicitly liquidate after the final OOS target stream.
- Package metadata and API version are aligned at `0.2.1` with an SPDX MIT license.
- CI installs every dependency needed by the supported test matrix instead of
  silently skipping temporal-model coverage.

### Fixed

- Prevented close-time decisions from receiving an overnight return that occurred before fill.
- Removed implicit free rebalancing from persisted target-weight matrices.
- Included the first active session in compounded total return.

## [0.1.0] - 2026-07-07

### Added

- Initial research pipeline, validation science, API/notebooks, and Python/C++ order-book parity.
