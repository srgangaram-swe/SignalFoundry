# ADR 0009 — Certified sparse QP for constrained mean-variance

- **Status:** Accepted
- **Date:** 2026-08-01
- **Work items:** #7, SF-S4-MR2 (#8), and #10
- **Supersedes:** the unmerged projected-gradient draft on the MR2 work branch

## Context

MR2 needs a deterministic, point-in-time optimizer for a convex objective with
budget, gross, net, leverage, position, liquidity, turnover, target-return, and
linear exposure constraints. Linear turnover cost and both gross and turnover
limits introduce absolute values.

The initial branch draft composed FISTA, an L1-cost proximal step, and a cyclic
map that stopped at the first feasible point. Independent review produced small,
well-conditioned counterexamples where the code returned `optimal` and passed
its feasibility audit while its objective was materially worse than the true
constrained optimum. The cyclic map was not an exact Euclidean projection onto
the intersection; applying an L1 prox before it was not the prox of the combined
objective and feasible-set indicator. Consecutive-iterate movement was also not
a KKT certificate. The draft architecture and its claimed convergence guarantee
were therefore rejected before merge.

## Options considered

### Repair projected gradient with Dykstra

Dykstra corrections can provide the exact projection needed for the smooth
quadratic problem. They do not, by themselves, make a separate L1 turnover prox
and constraint projection the exact combined proximal operator. Generalized
Dykstra, ADMM, or a primal-dual method would be required, increasing the proof
and maintenance surface.

### SciPy SLSQP

SLSQP is already available transitively and is useful as an independent test
reference. It is a general nonlinear optimizer, however, and does not expose a
natural sparse-QP primal/dual certificate. Using the same algorithm for both the
production result and its differential reference would also weaken review.

### Canonical sparse QP with OSQP

Introduce epigraph variables `g >= |w|` and `t >= |w-p|`, leaving the covariance
quadratic exact and every configured limit linear. Solve the resulting canonical
convex QP with OSQP, then independently reconstruct the financial constraints,
objective, and KKT conditions. Chosen.

## Decision

### Canonical formulation

For `alpha_risk_cost`, the effective objective is:

```text
0.5 w' (lambda Sigma + 2 c2 I) w
  + (-mu - 2 c2 p)' w + c1 1't
  + c2 p'p + mandatory-exit costs
```

with linear rows enforcing `g >= w`, `g >= -w`, `t >= w-p`, and
`t >= -(w-p)`. Gross and turnover limits operate on `1'g` and `1't`.
Minimum-variance, target-return, and maximum-utility are explicit changes to
that same canonical contract; packaging independently reconstructs the exact
formulation that was solved.

Assets leaving the risk universe remain in the previous-book record. Their
liquidation L1 and squared trades are constants in the new problem, reduce the
remaining turnover allowance, enter cost, and appear in result accounting.

### Dependency and runtime surface

Pin `osqp>=1.1.3,<2` through `uv.lock`. OSQP and its Python interface are
Apache-2.0 licensed. The locked dependency uses the existing NumPy/SciPy stack
and supports the repository's Python 3.12–3.14 matrix. No optional accelerator,
MKL, CUDA, code-generation, network, or executable-input path is enabled.

The dependency is appropriate because this is a dedicated convex QP boundary,
not an attempt to hide model assumptions behind a high-level modelling system.
The code constructs every matrix and row itself, documents units and signs, and
audits results independently.

### Determinism and resource bounds

The request is capped at 512 assets, 256 configured exposures, 1,024
previous-book names, 20,000 iterations, and five solver seconds. Nested risk
diagnostics are separately bounded to 4,096 items and eight levels.
Solver logs and warm starts are disabled; scaled termination is disabled;
termination is checked every iteration; polishing uses ten refinement steps;
and adaptive-rho updates occur on a fixed 25-iteration interval. Disabling
adaptive rho was tested and exhausted the iteration ceiling on ordinary
covariance/budget scaling; the fixed update interval is retained and included
in solve identity. Exact input bytes, timestamps, formulation, OSQP version, and
all solver settings are hashed.

OSQP's effective absolute, relative, and infeasibility tolerances have a
`1e-10` numerical floor even when a tighter public tolerance is requested. A
nonzero budget at or below the float64 audit floor (`128 * eps`) is refused so
an absolute certificate floor cannot accept a zero book for a tiny requested
budget. If the largest absolute coefficient in the quadratic or linear
objective is below `1e-2`, the quadratic matrix, linear vector, and constant are
scaled together to a unit characteristic coefficient. This transformation does
not change the optimizer and is hashed into solve identity.

Wall-clock duration is measured evidence, not part of semantic equality or the
deterministic identity. A time-limit result is unusable rather than retried or
silently relaxed.

### Independent acceptance certificate

Only OSQP's exact `solved` status is eligible. The returned primal and dual
vectors are checked independently for:

- every original financial constraint on the actual weights;
- canonical QP primal feasibility;
- KKT stationarity and complementarity;
- exact formulation-specific objective reconstruction; and
- primal/dual objective agreement.

The financial-audit and KKT acceptance coefficients are fixed at `1e-7`;
objective reconstruction uses `1e-8`. The checks deliberately do not erase
their original units: canonical QP primal row violation is an absolute residual
in each submitted row's units, while stationarity, complementarity, objective
reconstruction, and primal/dual gap are normalized by their implemented vector
or objective scales. The result's maximum residual therefore spans unlike
diagnostics and is not advertised as a universal dimensionless error. A caller
may request a tighter solve tolerance within the bounded public range but cannot
loosen the certificate. Inaccurate solve/infeasibility, maximum iterations, time
limit, malformed/non-finite output, or any certificate failure returns no
eligible portfolio.

Regression tests preserve the review counterexamples, compare randomized small
problems to an independently configured SciPy reference, match analytic
unconstrained solutions only when all limits are genuinely slack, and cover
formulation objectives, timestamp/unit identities, mutation resistance,
mandatory exits, active boundaries, immutability, and non-optimal statuses.

## Consequences

### Positive

- The actual nonsmooth constrained problem is represented exactly as a convex
  QP rather than approximated through an invalid operator composition.
- Sparse linear constraints and a dedicated QP solver provide bounded,
  inspectable primal/dual state.
- Feasibility alone can no longer certify a suboptimal portfolio.
- Solver, financial, and provenance failures remain structured and fail closed.
- Rule-based MR1 portfolios remain available as an independent rollback and
  comparison surface.

### Costs

- OSQP adds a direct native dependency and a versioned numerical behavior
  surface that must remain locked, audited, and tested on all supported Python
  versions.
- The covariance block is dense even though the surrounding constraints are
  sparse; the explicit 512-asset and time/iteration ceilings are therefore part
  of the safety contract.
- First-order numerical certification is tolerance-bounded, not a symbolic
  proof, and difficult conditioning is rejected upstream rather than accepted
  with a persuasive-looking answer.
- Absolute primal-row residuals and scaled optimality residuals are intentionally
  reported together; consumers must interpret their declared units and must not
  compare the aggregate maximum across differently scaled problems as one
  dimensionless accuracy statistic.

## Rollback and residual risk

Rollback removes the optimizer from allocation selection and retains the MR1
ranking/inverse-volatility baselines. No broker, order, paper, or live interface
depends on this solver.

Numerical optimality does not validate `mu`, `Sigma`, the factor specification,
or an economic edge. Expected-return error, regime change, capacity, borrow,
execution, and selection bias can dominate optimizer precision. MR2 therefore
publishes only synthetic infrastructure evidence; later robustness and frozen
qualification work may still reject every strategy.
