# Reproducibility and experiment provenance

AlphaForge treats a research result as an evidence bundle, not as a notebook
state or an unversioned serialized object. Governed Signal Foundry runs publish
atomically into `runs/signal-foundry/<experiment-id>/` and never overwrite an
existing run.

## Experiment identity

`alphaforge.research.ExperimentManifest` defines schema `1.0.0`. Its
`experiment_id` is the SHA-256 digest of canonical JSON containing:

- full Git commit and dirty-worktree state;
- immutable dataset bundle identity, contract version, license boundary, and
  point-in-time limitations;
- sorted universe and research date bounds;
- feature, label, model, validation, and backtest-cost configuration;
- stable named seed streams; and
- exact runtime, platform, and dependency versions.

Canonical JSON uses sorted keys, compact separators, and ASCII encoding.
Execution timestamps, invocation arguments, result summaries, and artifact
hashes are recorded but do not change the semantic experiment identity. A
different dependency, platform, dataset, code state, or governed setting
therefore receives a different identity.

The outer `run_manifest.json` schema is `2.0.0`. It contains the experiment
manifest plus result pointers such as the selected candidate and trial-ledger
head. Consumers must validate the declared versions rather than infer a schema
from filenames.

## Randomness

One non-negative root seed is expanded into order-independent, named 32-bit
streams for Python, NumPy, scikit-learn, optional model backends, data loading,
and hyperparameter search. Adding or reordering a candidate cannot perturb an
existing stream. The current execution boundary seeds Python, NumPy, PyTorch,
and TensorFlow when installed and injects the pre-registered root seed into
applicable scikit-learn, LightGBM, and torch model specifications without
overwriting an explicit model seed. Named streams reserved for future data
workers and search systems remain visible in the manifest; those systems must
accept their assigned stream before they may claim deterministic support.

## Environment and invocation safety

Environment capture is allowlisted. It records only reproducibility controls
such as thread counts and deterministic-runtime switches; it never copies the
process environment wholesale. CLI options containing credential-like names
(`api-key`, `password`, `secret`, or `token`) retain the option name and replace
the value with `[REDACTED]`.

Do not pass secrets on a command line: process listings and shell history exist
outside the manifest boundary. Market-data credentials belong in Keychain or
another approved secret store, and licensed observations remain local.

## Artifact integrity

Each artifact record contains a run-root-relative path, byte length, and
SHA-256 digest. Absolute paths, parent traversal, duplicate paths, invalid
hashes, and backward execution timestamps fail validation. The manifest itself
is excluded from its artifact inventory to avoid a recursive hash.

## Locked environments

`uv.lock` is the committed universal resolution for Python 3.12–3.14 and all
declared extras. CI pins the `setup-uv` action by full commit SHA and uv itself
to `0.11.28`; the container uses the same uv version and lockfile. CI exercises
the base environment on every supported Python version, the native and torch
boundaries, and separate locked import checks for the data, ML, and application
extras.

```bash
make install
uv run make check
uv run make demo
```

`make install` refuses to update the lock. Intentionally changing a dependency
requires regenerating `uv.lock`, reviewing the entire resolution diff, and
running the complete validation matrix. Network-requiring market-data tests are
separate from the deterministic offline suite.

## Typed configuration boundary

Every supported YAML surface has a dedicated frozen Pydantic schema under
`alphaforge.config`. Unknown keys, untyped model parameters, unsafe configured
paths, invalid ranges, and cross-field contradictions fail before network,
artifact, or model work begins. `make config-check` validates every canonical
configuration plus the pre-registered WIKI bootstrap profile. The generic
permissive YAML loader has been retired;
adding a new YAML surface requires a named schema and negative tests.

The canonical `configs/calibration.yaml` surface fixes the probability
calibrator, reliability bins, bootstrap resamples/block length/seed, conformal
miscoverage and minimum blocks, and quantile-regression bounds. Calibration and
conformal artifacts are atomic, versioned, bounded JSON records containing only
numeric state and fit provenance; their readers never deserialize executable
Python objects. See [Calibration and uncertainty
contracts](calibration_uncertainty.md).

The canonical `configs/metrics.yaml` surface freezes the unified metric
minimum samples, reliability bins, benchmark identity, and moving-block
resample count, block length, confidence level, circular policy, and seed.
Every published scalar carries a `MetricContract`; every defined uncertainty
record includes its variance and complete resampling policy. Annualization uses
the actual elapsed calendar span, so irregular calendars do not silently
inherit a 252-session assumption. See [Metric governance and time-series
distributions](metric_governance.md).

The canonical `configs/research_governance.yaml` surface freezes the eligible
family size, multiple-testing method and assumptions, failed-trial treatment,
candidate kill criteria, and ledger resource bounds. The plan and every ledger
event use canonical JSON and SHA-256 chaining; an independently updated head
receipt detects mutation and truncation before any supported append. Missing,
extra, failed, interrupted, or nonterminal candidates cannot be silently
excluded from family correction. See
[Append-only research governance](research_governance.md).

The canonical `configs/ensemble_benchmark.yaml` surface freezes the synthetic
reference dimensions, temporal folds, candidate-independent seed, turnover
cost, policy hyperparameters, fallback prediction, and record/audit ceilings.
Its exact-schema loader rejects unknown fields and misleading interpretation
labels. Training panels and fitted states use canonical sorted-key JSON and
SHA-256 identities; final inference is a distinct target-free type. See
[Governed temporal-OOF ensembles](governed_ensembles.md).

The canonical `configs/sprint_3_decision.yaml` surface is the strict,
content-addressed Sprint 3 synthesis plan. Its SHA-256 is the plan identity; it
fixes exact resource ceilings, the ordered ten-family inventory, source
digests, reviewed semantic locators, dispositions, protocol status, and the
unconditional `NOT_READY` boundary. `make config-check` fails before
publication on unknown fields, a changed family order, an unsupported positive
gate, an unverified source, an unsafe path, a mismatched limit, or a claim of
paper/order authority.

## Safe tabular artifacts

Pipeline tables use the `1.0.0` `*.table.json` contract in
`alphaforge.research.artifacts`. The envelope contains an explicit format and
schema version around JSON Table Schema data. Readers validate the exact
envelope, field identity, resource bound, path suffix, and symlink boundary
before deserializing. Writers reject arbitrary Python objects and publish by an
atomic same-filesystem replace.

Pickle is not a supported research interchange format: loading it can execute
code and its implicit Python-object contract is unsuitable for durable evidence.
The walk-forward, feature, training, backtest, paper, visualization, and API
paths all consume the versioned non-executable format. Existing local pickle
runs are legacy artifacts and must be regenerated; they are never migrated by
loading and reserializing untrusted pickle bytes.

## Sprint visual evidence

Sprint-close plots are generated through Seaborn's plotting API and theme
system with a colorblind palette; Matplotlib supplies only the non-interactive
rendering, layout, annotation, and export backend. Plots must be regenerated
from machine-readable run evidence, visually inspected, linked from the sprint
report, and labeled honestly as synthetic, historical backtest, paper, or live
evidence. Restricted market observations remain local; only licensed-safe
aggregates, synthetic fixtures, and their reproducible generation instructions
may be committed.

The label reference workflow uses `make label-evidence OUTPUT=<new-directory>`.
It records the complete semantic label contract, deterministic synthetic seed
and dimensions, predeclared sensitivity scales, aggregate CSV evidence, plot
hashes, and explicit non-market limitations. Publication refuses to overwrite
an existing evidence root. See [Financial label contracts and
diagnostics](label_design.md).

The governed ensemble reference uses
`make ensemble-evidence OUTPUT=<new-directory>`. It evaluates all six policies
offline, refuses overwrite, records SHA-256 integrity hashes for each aggregate
artifact, and commits no row prediction, target, fitted state, licensed
observation, or model binary. The resolved config and runtime environment are
recorded together with distinct generator/bootstrap seeds, derived named child
streams, and the complete moving-block policy. Its Seaborn summary and
prediction/error correlations, regime overlap, drop-one marginal
contributions, turnover/cost, heuristic dispersion, and moving-block
variability tables are linked from the
[Sprint 3 ensemble report](sprint_3_ensemble_report.md).

The final Sprint 3 decision uses
`make sprint-3-decision-evidence OUTPUT=<new-repository-local-directory>`.
The publisher re-loads the on-disk strict plan, verifies each AlphaForge source
digest and positive semantic locator, and rejects symlinks, traversal,
conflicting identities, resource-limit violations, malformed documents,
unbounded diagnostics, and overwrite attempts. It publishes exactly six
payload artifacts plus `manifest.json` through an atomic directory replace:
aggregate family and gate CSV files, the semantic-review receipt, summary,
README, and Seaborn coverage heatmap. The manifest records the plan plus the
exact SHA-256 and byte length of every payload.

Reproduce into ignored local storage and compare every byte with the committed
reference:

```bash
uv run make config-check
uv run make sprint-3-decision-evidence \
  OUTPUT=runs/sprint-3-decision-replay
diff -r \
  docs/evidence/signal_foundry_sprint_3/decision \
  runs/sprint-3-decision-replay
```

Any byte difference under the locked environment is a failed replay requiring
investigation; do not refresh the reference blindly. The committed
`cross_repository_provenance.json` receipt can be schema-validated without a
sibling checkout, but external-blob verification is a separate local step.
Given an existing Signalattice Git object database, verify the pinned commit,
Git blobs, lengths, and SHA-256 values without fetching:

```bash
uv run python scripts/verify_sprint_3_cross_repository_sources.py \
  --signalattice /absolute/path/to/Signalattice
```

Finally, visually inspect
`docs/evidence/signal_foundry_sprint_3/decision/plots/evidence_coverage.png` at
full resolution. Confirm all ten family labels and nine gate labels are
legible, every cell matches `gate_matrix.csv`, and the title makes clear that
`1` means reported evidence and `0` means a missing gate—not model quality or
performance. See the [final Sprint 3 report](sprint_3_report.md).

The governed real-data workflow uses
`make signal-foundry-evidence RUN=<run> BUNDLE=<bundle> OUTPUT=<new-directory>`.
Its publication boundary independently matches the bundle identity against the
run and dossier, requires the aggregate-only license flag, copies only an
explicit field allowlist, hashes every output, and refuses overwrite or
symlinked inputs. The committed WIKI profile and aggregate evidence can be
replayed from the verified local cache with zero provider requests. Licensed
rows and row-level predictions, orders, fills, positions, and returns remain
ignored and local.

The Sprint 2 baseline study uses
`configs/signal_foundry_sprint_2_study.yaml` together with
`configs/research_governance.yaml`. Run
`scripts/run_sprint_2_study.py` under `/usr/bin/time -l`, then pass its
immutable study/run directories and the bounded time profile to
`scripts/publish_sprint_2_evidence.py`. The plan hash includes the seeded
candidate configurations, folds, holdout, costs, correction, and kill policy.
The aggregate publisher refuses an existing destination, symlinked input,
identity mismatch, incomplete family, unequal fold set, or source license that
does not require aggregate-only publication. Full commands and assumptions are
in [Governed seven-candidate baseline study](governed_baseline_study.md).
