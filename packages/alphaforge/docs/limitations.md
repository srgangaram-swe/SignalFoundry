# Limitations

AlphaForge is an educational research platform, not a production trading system.

Important limitations:

- yfinance data can contain survivorship, adjustment, and availability issues.
- The no-cost Nasdaq WIKI engineering bundle ends on 2018-03-27, is
  current-vintage, and lacks complete historical revisions, point-in-time
  universe membership, and corporate-action records. It can validate pipeline
  mechanics but must not qualify current paper or live readiness. Its declared
  AAPL benchmark is one constituent, not a diversified investable market proxy.
- Synthetic data verifies engineering behavior but is not market evidence; its
  embedded edge is deliberately faint but still far cleaner than real markets,
  so demo Sharpe ratios overstate what identical code would earn live.
- The SF-S3-MR7 time-frequency vision reference is synthetic engineering
  evidence. Its small CNN failed the frozen validation gate, so ResNet and ViT
  were correctly blocked. This rejection does not establish that image models
  fail on licensed point-in-time data; it establishes only that larger models
  were unjustified on the declared reference.
- The SF-S3-MR8 latent-representation reference is one small synthetic split
  with planted factors, regimes, and conspicuous anomalies. PCA was selected on
  validation but tied the raw control on test rank IC and slightly worsened
  prediction MSE. Robust-scale PCA's higher post-selection test rank IC cannot
  be used to revise that choice. Daily standard errors do not correct for serial
  dependence, and neural epoch ceilings, synthetic transfer scores, and
  single-fit timings do not establish convergence, market value, or production
  performance.
- The SF-S3-MR9 ensemble reference is synthetic and deliberately gives experts
  complementary calm/stress error. Its favorable stacking, dynamic, and gated
  results prove recovery and failure behavior only. The OOF contract validates
  declared fold/date provenance but cannot prove an external expert producer
  was honest; Bayesian likelihood assumptions, regime calibration, online
  drift, licensed point-in-time data, multiplicity, capacity, and event-driven
  execution remain unresolved.
- Backtests are daily-bar approximations. The close-decision/next-open ledger
  prevents pre-fill gap capture and lets holdings drift, but bars cannot reveal
  queue position, auction dynamics, intraday path, or order-book state.
- Deterministic event replay proves ordering and accounting mechanics, not that
  daily-bar fills occurred or would be available. Schema-v1 replay also requires
  the caller-preserved frozen calendar; the local SHA-256 journal chain is not a
  digital signature against an actor able to replace the database and its
  trusted provenance. Runtime mark events can contain licensed prices and must
  remain in ignored owner-controlled storage.
- Commission, exchange fees, spread, slippage, power-law impact, financing,
  borrow, logical latency, participation, and capacity settings are transparent
  sensitivities, not estimates calibrated to proprietary order-level execution
  data. A scalar borrow rate proves neither availability nor a locate. Logical
  session delays do not represent wall-clock exchange or broker latency.
  Partial DAY-order residuals expire rather than following a production
  order-management lifecycle.
- The C++ order book simulates fills against a *synthetic* book shape; it is
  a systems-engineering and parity-testing module, not a historical market
  microstructure calibration or the paper replay's source of truth.
- DSR/PBO correct for the trials the platform knows about; they cannot correct
  for ideas discarded before they were coded.
- The append-only research ledger detects mutation or truncation against its
  local head receipt, but it is not digitally signed. An actor who can rewrite
  both files can manufacture another internally consistent history; durable
  operation requires externally anchored immutable receipts and access logs.
  Its lock is a fail-closed single-writer boundary, not distributed consensus.
- Probability calibration and interval coverage can fail under prevalence
  drift, regime changes, or changed model/feature policy. ECE depends on its
  declared bins; moving-block bootstrap intervals require local stationarity
  and an adequate block length; block-conformal coverage requires exchangeable
  residual blocks; and linear quantile regressions can be misspecified.
- The SF-S3-MR10 abstention reference is deterministic synthetic engineering
  evidence, not a calibrated market study. Its policy reduced synthetic
  conditional loss frequency, turnover, and capacity demand but captured less
  total net value than always-trade. Expected costs, predictive uncertainty,
  disagreement, regime support, and drift can all be jointly misspecified;
  abstention cannot guarantee against loss.
- The final Sprint 3 synthesis spans historical WIKI engineering, multiple
  unrelated synthetic generators, and unsupported economic questions. The
  datasets, targets, splits, costs, uncertainty methods, and holdout boundaries
  are heterogeneous, so gate coverage and dispositions cannot be interpreted
  as a performance score or cross-family ranking. Five families are rejected
  under current evidence, five are deferred, none advances, and the decision
  remains `NOT_READY`.
- Matched feature/parameter ablations, feature permutation, randomized-label
  and representation-placebo controls remain re-scoped to issue #41. Matched
  year and temporal/regime stability with dependence-aware uncertainty remains
  re-scoped to issue #42. The final matrix records these omissions as failed
  readiness evidence rather than treating them as not applicable.
- CI can validate the committed Signalattice receipt without network access,
  but it cannot prove the nine external blobs are present without an existing
  local Signalattice Git object database. Signalattice issues #14–#16 remain
  separate open work and are not completed by the AlphaForge synthesis.
- Unified metric intervals remain conditional on the selected calendar,
  benchmark, sample, block length, and stationarity approximation. An
  undefined metric is reported explicitly, but a defined metric can still be
  economically irrelevant, selected after many trials, unstable across
  regimes, or overwhelmed by unmodeled execution costs.
- No broker integration places live orders.
- Offline paper controls emit proposed state decisions only. They do not model
  broker acknowledgements, rejects, outages, recovery, exchange state, legal or
  tax obligations, or operational capital controls.
- Model results can overfit even with walk-forward validation, purging, and
  deflated statistics.
- Deterministic fitting, convergence evidence, and agreement with numerical
  references establish implementation behavior only. They do not establish a
  tradable signal, cross-platform bitwise identity, or persistence of an edge.
- The Sprint 2 seven-candidate study uses a small number of matched
  walk-forward folds. Its one-sided Student-t tests are low-power and rely on
  fold-level sampling assumptions; Holm correction controls the declared
  family but cannot eliminate research-selection risk outside that family.
- OOF regression calibration slope/intercept are descriptive diagnostics, not
  probability calibration or evidence of stable forecast calibration.
- The no-cost WIKI study ends in 2018 and lacks complete point-in-time universe,
  revision, delisting, and corporate-action evidence. It must reject paper/live
  advancement regardless of a favorable historical metric.
- Label diagnostics quantify overlap, autocorrelation, class balance, temporal
  drift, and parameter sensitivity but do not prove that a target is
  predictable. Daily-close triple barriers cannot establish intraday hit
  ordering, and synthetic label evidence is not market evidence.
- Real deployment would require stronger data licensing, monitoring, capital
  controls, compliance review, and independent validation.
