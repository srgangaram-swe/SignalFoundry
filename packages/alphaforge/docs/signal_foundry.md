# Signal Foundry governed research

AlphaForge consumes Signalattice output through a narrow, versioned data
contract. The repositories do not import each other and do not share runtime
state. This boundary makes provenance, licensing, temporal semantics, and
failure behavior independently reviewable.

This workflow supports research and zero-capital shadow evaluation. It has no
broker adapter, order-routing interface, credential path, or live-trading
authorization. A favorable backtest is not evidence of guaranteed profit.

## Trust boundary

`alphaforge.data.load_signal_foundry_dataset` accepts one content-addressed
bundle directory and fails closed unless all of these checks pass:

- contract name and supported schema version `1.0.0` or `1.1.0`;
- bundle directory name and semantic manifest SHA-256 identity;
- safe relative paths, unique partitions, partition SHA-256 hashes, row counts,
  complete declared Parquet inventory, aggregate date bounds, and ticker
  universe;
- exact column order, finite positive prices, nonnegative volume, valid OHLC
  bounds, unique `(date, ticker)` keys, and nonempty identifiers;
- timezone-normalized `effective_at`, `available_at`, and `observed_at`, with no
  ambiguous timestamps or invalid observation/availability ordering;
- explicit source adjustment state, license booleans, and point-in-time
  limitations for historical revisions, universe membership, and corporate
  actions.

Schema `1.1.0` adds independently verified bitemporal universe-membership and
corporate-action record families. The consumer checks their manifest schemas,
paths, hashes, row counts, strict UTC timestamps, stable record identities,
revision uniqueness, supported event types, and action-specific payloads.
Unknown or ambiguous records fail the entire load before any frame is returned.
Schema `1.0.0` remains readable for compatibility, but it has no independently
inspectable auxiliary record families and cannot be used to reconstruct a
historical universe.

The optional timezone-explicit `as_of` argument enforces both
`effective_at <= decision timestamp` and
`available_at <= decision timestamp` across prices, membership, and action
events. Versioned auxiliary records collapse to the latest visible revision by
stable identity. Provider-adjusted close data is mapped into AlphaForge's
adjusted-OHLCV representation using the declared adjustment state. Unknown
adjustment semantics are rejected instead of guessed.

The loader preserves two distinct views. The market panel retains the actual
session dates used for labels, prices, and fills. The decision panel dates each
bar at the first represented session close at which its `available_at`
timestamp made it usable. Features use the decision panel; labels and execution
use the market panel. The generic one-panel loader therefore rejects Signal
Foundry input and directs operators to this governed workflow.

## Historical universe and action controls

`SignalFoundryDataset.universe_membership_as_of` reconstructs the latest
visible state of every known instrument in a named universe. Its result retains
explicit exits (`is_member=False`) so removals and delistings stay visible.
`active_universe_as_of` filters that reconstructed state only after the exit
evidence has been applied. Both operations fail closed unless the bundle uses
schema `1.1.0` and explicitly attests complete point-in-time universe
membership.

`corporate_actions_as_of` returns the latest visible revision of each effective
action event. These records are evidence and diagnostics only: the loader
never applies dividends, splits, merger consideration, symbol changes, or
delisting returns to prices. Silent application would risk double adjustment
because each price row already declares its provider adjustment state. A later
execution/accounting slice must define any economic treatment explicitly and
test it against the declared price policy.

`PointInTimeDiagnostics` reports the schema, producer completeness flags,
presence of auxiliary record families, observed price-adjustment states,
survivorship risk, corporate-action risk, and plain-language warnings. It
preserves uncertainty rather than upgrading an empty or incomplete record
family into evidence.

Licensed bundles must remain outside Git. The repository ignores
`data/signal-foundry/`, `data/raw/`, caches, processed panels, and all run
artifacts. Docker context exclusions provide a second boundary. Only synthetic
contract fixtures are committed.

The free WIKI bootstrap bundle is schema `1.1.0` but honestly contains empty
universe and corporate-action families and declares the corresponding
point-in-time completeness flags `false`. It remains useful for validating
pipeline mechanics and historical experiments, but historical-universe
reconstruction rejects it and the default readiness rubric remains
`NOT_READY`. No API credential, licensed observation, or provider response is
stored by AlphaForge.

Its dedicated pre-registered policy is
`configs/signal_foundry_wiki_bootstrap.yaml`. The 2017-01-03 final-holdout
boundary, four-candidate trial family, execution assumptions, stress scenarios,
and readiness thresholds are fixed before the final interval is inspected.
It uses AAPL only because the public Signalattice sample declares AAPL as its
available benchmark; a single constituent is not a diversified market proxy.
This profile is not a substitute for the default current-data policy.

## Pre-registered evaluation

The default policy lives in
`configs/signal_foundry_research.yaml`. Change it only before starting a new
immutable evaluation. The run identity hashes:

- the producer bundle;
- the AlphaForge Git commit;
- model candidates and hyperparameters;
- features and labels;
- walk-forward and embargo policy;
- backtest, cost, execution, portfolio, and risk settings; and
- the final paper-readiness rubric.

The workflow:

1. builds causal features and multi-horizon forward labels;
2. ends development before the final holdout and purges an embargo at least as
   long as the maximum label horizon;
3. compares every registered candidate using development-only walk-forward
   evidence;
4. records each candidate and result in an ordered SHA-256 hash chain;
5. fits the selected candidate on development data and touches the final
   holdout once for that immutable run identity;
6. backtests future-open execution with component-level fees, spread/slippage,
   power-law impact, native financing/borrow accrual, logical-session latency,
   participation limits, drifted positions, and reconciled cash/share accounting;
7. repeats the complete event simulation under doubled and tripled costs,
   adverse spread, reduced liquidity, delayed signal, partial-fill pressure,
   five-times capital demand, and perturbed selection breadth, then compares
   against a deterministic permuted-signal placebo;
8. proves idempotency, stale-data, risk-limit, and one-way kill-switch behavior
   through deterministic offline proposed decisions with no broker interface;
9. produces circular moving-block uncertainty intervals, year/regime
   stability, drawdown duration, missing-price halt evidence, and auditable AUM
   capacity sensitivities; and
10. publishes the run atomically with hashes and a machine-readable dossier.

Existing run identities cannot be overwritten or repeated. A failed run
removes its staging directory; a stale staging directory fails closed for
operator inspection.

## Run locally

Install the data dependencies, then provide the absolute path to an immutable
Signalattice bundle:

```bash
python -m pip install -e ".[dev,data]"
make signal-foundry \
  BUNDLE=/absolute/path/to/signalattice/data/processed/signal-foundry/<bundle-id>
```

The command prints the run identity, selected candidate, decision, and dossier
path. It never prints or needs the Nasdaq Data Link API key; acquisition occurs
in Signalattice.

For the verified cached WIKI bundle, opt into the engineering profile
explicitly:

```bash
make signal-foundry \
  BUNDLE=/absolute/path/to/<wiki-bundle-id> \
  SIGNAL_FOUNDRY_CONFIG=configs/signal_foundry_wiki_bootstrap.yaml
```

After the ignored local run finishes, publish only the aggregate allowlist:

```bash
make signal-foundry-evidence \
  RUN=/absolute/path/to/runs/signal-foundry/<run-id> \
  BUNDLE=/absolute/path/to/<wiki-bundle-id> \
  SIGNAL_FOUNDRY_CONFIG=configs/signal_foundry_wiki_bootstrap.yaml \
  PROFILE=/absolute/path/to/macos-time-profile.txt \
  OUTPUT=/absolute/path/to/new-public-evidence
```

The publisher verifies bundle/run identity and the producer license policy,
then emits aggregate gate, scenario, and capacity tables plus reproducible
Seaborn plots. It never copies ticker identities, observations, predictions,
orders, fills, positions, or date-level returns. Existing destinations and
symlinked inputs fail closed.
When supplied, `PROFILE` must be bounded output from macOS `/usr/bin/time -l`;
the publisher extracts only duration and memory aggregates and labels the
single-machine result as a profile rather than a service-level claim.

## Readiness decision

The dossier reports exactly `READY_FOR_PAPER` or `NOT_READY`. Every configured
gate must pass:

- sufficient untouched holdout sessions;
- Deflated Sharpe probability after accounting for attempted candidates;
- bounded Probability of Backtest Overfitting;
- bounded drawdown and turnover;
- nonnegative annual return relative to the investable benchmark;
- complete declared point-in-time evidence;
- reconciled accounting; and
- cost, latency, liquidity, and placebo stress success.

`READY_FOR_PAPER` permits only a time-bounded, zero-capital offline or shadow
exercise under the documented controls. It does not authorize live trading,
capital deployment, broker connectivity, or a risk increase. Nasdaq source
bundles that honestly declare incomplete revision, universe-membership, or
corporate-action history must receive `NOT_READY` under the default rubric,
even when their return statistics look attractive.

## Paper-control boundary

`alphaforge.paper.PaperControlState` evaluates proposed shadow-state
transitions without creating executable orders. It enforces unique decision
identities, data freshness, gross/net/position/turnover/notional bounds,
drawdown and daily-loss limits, and a one-way manual kill switch. Any breach
produces a machine-readable halt reason. Invalid or non-finite input raises
before policy evaluation.

Every governed dossier includes a deterministic `paper_control_evidence.json`
audit. The audit proves one bounded proposal is allowed while duplicate,
stale-data, exposure-limit, and manual-kill-switch paths halt. It explicitly
records that no broker adapter exists and no executable order was emitted.

Moving beyond zero-capital shadow evaluation requires a separate milestone,
threat model, broker-specific contract, reconciliation and recovery design,
operational rehearsal, explicit capital-at-risk cap, legal and tax review, and
the owner's affirmative approval. None of those capabilities are present in
this release.

Those requirements are now written down rather than merely asserted. The
[broker connectivity requirements](broker_connectivity_requirements.md) specify
the capability matrix, non-code prerequisites and their lead times, credential
custody, failure-mode behaviour, reconciliation cadence, and the nine-item
capital-authorization gate that stands between paper evidence and a live order.
**Zero of the nine are currently satisfied**, and the first — a strategy holding
a `QUALIFIED_FOR_PAPER` verdict — is blocked by research outcomes rather than by
engineering. The broker selection and its pre-frozen rubric are recorded in
[ADR 0015](adr/0015-broker-selection-for-paper-and-live-trading.md). No broker
client, credential, endpoint, or connection exists in this release.
