# Append-only research governance

`alphaforge.research.governance` is the statistical and audit boundary between
a pre-registered research question and its observed trial family. It prevents
the supported execution path from silently deleting failures, changing
thresholds after seeing results, or correcting only a favorable subset.

This boundary governs research evidence. It does not authorize paper or live
trading and does not turn historical significance into expected profit.

## Frozen plan

`FrozenResearchPlan` is deeply immutable after construction. Its canonical
SHA-256 identity commits to:

- the falsifiable hypothesis and proposed mechanism;
- immutable dataset identity, feature family, and label;
- the test and temporal-validation design;
- costs and uncertainty assumptions;
- explicit rejection thresholds and per-candidate kill criteria;
- every eligible candidate configuration;
- each candidate's parent lineage; and
- the multiple-testing method, alpha, family size, assumptions, and conservative
  failed-trial treatment.

The first ledger record is exactly one `PLAN_FROZEN` event. A candidate outside
that plan cannot be registered. Changing any semantic field creates a different
plan hash and therefore a different governed experiment.

## Append-only state machine

The supported state transitions are:

```text
REGISTERED -> STARTED -> SUCCEEDED
                      -> FAILED
                      -> INTERRUPTED -> RESUMED -> SUCCEEDED
                                               -> FAILED
                                               -> INTERRUPTED
REGISTERED -> FAILED
```

Every accepted transition appends one canonical JSON line. Each record commits
to its sequence number, preceding record hash, event time, trial identity, and
validated payload. A separate atomically replaced head receipt records the
expected record count, chain head, and SHA-256 of the complete ledger bytes.
Before any append, the reader checks exact schemas, resource limits, the full
chain, the plan hash, the state machine, and the receipt.

Duplicate registration appends `DUPLICATE_REJECTED` before raising. An attempted
family evaluation while any eligible trial is missing or nonterminal appends
`FAMILY_EVALUATION_REJECTED` before raising. Failed trials remain terminal
records and enter correction with `p=1`; they are never removed from the
denominator. Interrupted and resumed attempts remain separate events.

The writer uses an exclusive local lock file and bounded record/byte limits.
A stale lock or a crash between ledger append and receipt publication stops
future work for investigation; recovery does not silently discard the
unreceipted bytes.

## Complete-family corrections

Let the frozen family contain \(m\) raw p-values
\(p_{(1)} \leq \dots \leq p_{(m)}\).
The implementation accepts a mapping only when its identities exactly equal
the eligible family and every value is finite in \([0,1]\).

Holm-Bonferroni adjusted values are:

\[
\tilde p_{(i)} =
\min\left(1,\max_{j \leq i}\{(m-j+1)p_{(j)}\}\right).
\]

This controls family-wise error under arbitrary dependence and is the committed
Sprint 2 default.

Benjamini-Hochberg adjusted values are:

\[
\tilde p_{(i)} =
\min\left(1,\min_{j \geq i}\left\{\frac{m}{j}p_{(j)}\right\}\right).
\]

BH is available only as an explicitly frozen alternative. Its false-discovery
interpretation remains conditional on the dependence assumptions recorded in
the plan.

Candidate rejection requires adjusted \(p \leq \alpha\). Statistical rejection
of a null is not sufficient for advancement: every predeclared economic or
risk kill criterion is also evaluated. A failed trial, missing/non-finite
metric, uncorrected result, or triggered criterion kills that candidate.

## Trust boundary and residual risk

The ledger detects accidental or partial mutation, reordering, truncation,
extension, invalid transitions, and mismatch against an expected plan hash. It
is not digitally signed. An actor able to rewrite both the ledger and head
receipt can create a different internally consistent chain. Production
research governance must anchor each receipt in a separately controlled
immutable store with access logs and retention policy.

The single-writer local lock is intentionally fail-closed and is not a
distributed consensus protocol. This sprint does not add a database,
multi-writer service, cloud retention layer, final-holdout data, credentials,
or broker capability.

## Reproduction

```bash
uv run python scripts/validate_configs.py
uv run pytest tests/test_research_governance.py -q
uv run make check
```

`configs/research_governance.yaml` freezes the seven-candidate Sprint 2 family
size, Holm policy, assumptions, kill criteria, and resource limits. The
subsequent governed study must construct a matching immutable plan before
reading any final-holdout result.
