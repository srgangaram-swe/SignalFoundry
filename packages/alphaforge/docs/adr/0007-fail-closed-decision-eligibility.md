# ADR 0007: Put fail-closed eligibility before portfolio and order boundaries

- Status: Accepted
- Date: 2026-07-26
- Decision owners: AlphaForge maintainers
- Depends on: SF-S2-MR6 uncertainty, SF-S3-MR9 governed ensembles, and existing
  execution-cost contracts

## Context

A forecast can be finite and directionally plausible while still being
economically unusable. Costs can consume the expected return, uncertainty can
make its sign unreliable, ensemble members can disagree, a detected regime can
be unsupported, drift can invalidate fitted assumptions, or the underlying
data can be stale. Passing such a forecast directly to portfolio construction
would make downstream caps responsible for a model-evidence decision they
cannot reconstruct.

The inverse failure is also important: silently dropping a forecast loses the
reason, threshold identity, and opportunity-cost evidence needed to review an
abstention policy.

## Decision

Add an opt-in, pure `alphaforge.decision` layer after model uncertainty/regime
evidence and before portfolio construction.

The layer accepts immutable typed signals and thresholds. It computes absolute
expected return minus a conservative cost estimate and a predictive
uncertainty charge. Eligibility requires the result to strictly exceed a
required margin and every independent freshness, cost, disagreement, regime,
drift, uncertainty, finiteness, and range gate to pass.

All failing gates are retained in a fixed typed reason order. Maximum-threshold
equality passes, while margin equality abstains. A canonical SHA-256 identity
binds every policy field and every signal field. Batch evaluation rejects
duplicates, stops at a configured bound, and sorts by decision identity so
input order cannot affect replay.

The result contains eligibility, forecast direction, value components, reason
codes, and failed fields. It deliberately omits quantity, price, venue,
account, order type, broker authority, and network behavior. `trade` therefore
means only “eligible for downstream research”; it is not an order.

## Evidence decision

The committed study is synthetic engineering evidence with a frozen PCG64 seed
and no protected-holdout access. Realized evaluation labels are stored outside
the policy input type. The study compares the full policy with always-trade and
never-trade ablations and reports coverage, selective risk, net value, missed
opportunity, turnover demand, and capacity demand.

Only aggregates, non-exclusive reason counts, a JSON manifest, and a Seaborn
plot may be committed. Signal rows and realized labels remain unpublished.
Atomic new-directory publication prevents a partial reference artifact from
looking complete.

## Alternatives considered

- **Embed thresholds in portfolio construction.** Rejected because it mixes
  model-evidence policy with weight optimization and makes isolated validation
  and rollback harder.
- **Treat expected return above zero as sufficient.** Rejected because it
  ignores costs, uncertainty, disagreement, regime support, drift, and data
  availability.
- **Emit the first failure only.** Rejected because it hides concurrent
  violations and makes monitoring sensitive to gate implementation order.
- **Tune thresholds on the final holdout.** Rejected because it converts the
  holdout into selection data and invalidates its role.
- **Connect eligibility directly to an order adapter.** Rejected. This research
  slice has no authority to size or route capital.

## Consequences

The system gains a deterministic, reviewable choke point and can quantify the
economic cost of abstention. Conservative policies may reject most
opportunities, reduce total net value, or concentrate eligibility in
unrepresentative subsets. Those outcomes must be reported, not optimized away.

SHA-256 identities detect accidental configuration/input changes but are not
signatures. Synthetic evidence proves mechanics only. Production use would
still require licensed point-in-time data, calibrated costs, drift monitoring,
independent validation, portfolio/risk controls, paper rehearsal, compliance
review, and explicit owner approval.

Rollback removes the decision package, config, standalone study, evidence, and
documentation. No portfolio, execution, paper, or broker state is migrated or
corrupted because this layer owns none.
