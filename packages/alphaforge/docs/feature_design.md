# Feature contracts, lineage, and fitted transformations

AlphaForge's feature layer turns a canonical, point-in-time market panel into a
versioned matrix keyed by `(date, symbol)`. It separates three responsibilities:

1. causal feature mathematics in `alphaforge.features.technical`;
2. semantic declarations and output validation in
   `alphaforge.features.registry`; and
3. training-fold-only learned preprocessing in
   `alphaforge.features.transform`.

This separation makes a feature's meaning reviewable without executing it and
makes a cache hit prove semantic equivalence rather than merely reuse a filename.

## Registry contract

Registry schema `1.0.0` declares each feature family's:

- stable name and exact semantic version;
- raw inputs and normalized parameters;
- maximum lookback and minimum warm-up sessions;
- ordered output columns and missing-value policy; and
- fit-period policy (`causal-through-observation` for the current deterministic
  families).

The registry rejects unknown versions, duplicate name/version pairs, overlapping
outputs, unsafe metadata, and invalid windows before feature execution. The
emitted matrix must match the complete ordered schema exactly, have unique and
non-missing identifiers, contain numeric outputs, and contain no infinity.
Warm-up `NaN` values are allowed because they are declared missing observations,
not numeric results. A learned transform later rejects any column that remains
entirely missing in its training fold.

The current families are lagged and log returns, rolling volatility,
moving-average ratios, momentum, RSI, MACD, Bollinger and mean-reversion
z-scores, volume diagnostics, drawdown, rolling Sharpe-style ratios,
benchmark-relative beta/correlation, market-regime observations, the causal HMM
stress probability, and same-date cross-sectional ranks. Adding speculative
families is outside this contract.

## Mathematics and temporal semantics

For adjusted close \(P_t\), representative definitions are:

- \(r_{t,k}=P_t/P_{t-k}-1\);
- \(\sigma_{t,w}=\sqrt{252}\,\operatorname{sd}(r_{t-w+1:t})\);
- \(m_{t,w}=P_t/\operatorname{mean}(P_{t-w+1:t})-1\); and
- rolling beta
  \(\beta_{t,w}=\operatorname{cov}(r,b)_{t,w}/
  \operatorname{var}(b)_{t,w}\).

Every value at date \(t\) uses observations available at or before \(t\).
Cross-sectional ranks use only the same date's eligible symbols. The HMM uses
expanding refits and filtered—not smoothed—state probabilities. Mutation tests
replace all future prices and volumes and require every earlier feature value
to remain unchanged.

These mechanics do not fix upstream survivorship, selection, revision, or
corporate-action incompleteness. Those properties remain explicit dataset
limitations and must travel with research results.

## Semantic identity and cache key

`materialize_feature_set` returns the matrix, registry, lineage, cache key, and
hit/miss state. Its SHA-256 key binds:

- the declared dataset reference and a deterministic content fingerprint of the
  exact input panel;
- full code revision;
- semantic feature parameters (excluding output/cache locations and downstream
  fitted-transform policy);
- date bounds and sorted universe; and
- the complete registry identity.

Changing one input observation, code revision, feature parameter, date interval,
symbol set, version, or output contract changes the key. Configuration map order
does not.

Cache entries live under the configured ignored `data/cache/features` root.
Each immutable directory contains a versioned, non-executable JSON Table Schema
artifact and a manifest that repeats the complete lineage and registry and pins
the artifact SHA-256. Publication uses a same-filesystem staging directory and
atomic rename. Loading bounds artifact size and rejects symbolic links, malformed
keys, undeclared files, corrupt JSON, digest mismatch, lineage mismatch, registry
mismatch, and incompatible frames. A corrupt hit fails closed; it is never
silently treated as a miss.

No API credential or licensed panel belongs in this cache if its storage terms
do not permit local retention. Cache roots, processed panels, and generated run
artifacts are ignored and must never be committed.

## Fitted-transform contract

The optional transform schema is `1.0.0`:

```yaml
fitted_transform:
  version: "1.0.0"
  enabled: false
  imputation: median
  standardize: true
  variance_threshold: null
  pca_components: null
  pca_whiten: false
  min_fit_rows: 64
```

It supports train-fitted mean/median imputation, z-score standardization,
variance filtering, and deterministic full-SVD PCA. For training observations
\(x_1,\ldots,x_n\), standardization uses only
\(\mu_{\mathrm{train}}\) and \(\sigma_{\mathrm{train}}\):

\[
z = \frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
\]

Each walk-forward window creates one transformer, fits it once on that window's
training rows, and applies the frozen state to its test rows. The governed
final-holdout workflow independently fits on the complete development interval
ending before the embargo and applies the frozen state once to the untouched
holdout. It records version, specification identity, ordered input/output
schemas, fit dates, fit row count, and a hash of learned statistics/components.
Rule-based baselines such as `momentum_baseline` retain raw semantic columns and
explicitly opt out.

The transform rejects inference before fit, insufficient rows, all-missing
training columns, infinity, nonnumeric input, unsafe parameters, transformations
that remove every feature, non-finite output, and any reordered/missing/extra
column. Holdout mutation tests prove that neither learned state nor transformed
training values can change after fitting.

The default remains disabled because existing estimator pipelines already fit
their own imputation/scaling inside each training call. Enabling central
preprocessing is an explicit experiment-contract change; it is not a shortcut
for fitting on the full panel.

## Reproducibility

From a clean checkout and locked environment:

```bash
make setup
make test
python scripts/build_features.py \
  --config configs/features.yaml \
  --data-config configs/data.yaml \
  --models-config configs/models.yaml
```

Feature materialization refuses a dirty Git worktree because an uncommitted
implementation cannot be identified by a commit SHA. The CLI writes the
registry and lineage beside ignored processed artifacts. Repeating an identical
clean run verifies and reuses the content-addressed entry.

Focused contract tests are in `tests/test_feature_registry.py`; causal mutation
coverage remains in `tests/test_leakage.py`. Sprint-close Seaborn evidence is
generated by the sprint reporting workflow from redistribution-safe,
machine-readable results, not from cached market observations.

## Risks, rollback, and limitations

- A registry describes and validates the current implementation; it is not a
  formal proof that every formula is economically useful.
- JSON caching favors safety and portability over Parquet throughput. Benchmark
  evidence is required before changing the format.
- Panel fingerprinting is linear in rows and intentionally paid once at the
  materialization boundary.
- Imputation and PCA can erase economically meaningful missingness or
  interpretability. They require an explicit enabled configuration and
  out-of-sample comparison.
- A cache hit establishes exact replay of inputs and semantics, not data
  licensing, point-in-time completeness, or future profitability.

Rollback disables `fitted_transform`, sets `cache_dir: null`, and retains the
last supported registry version. Cache directories are disposable local
derivatives; deleting one changes performance, not research meaning. Never
downgrade a version field or bypass validation to consume an incompatible
artifact.
