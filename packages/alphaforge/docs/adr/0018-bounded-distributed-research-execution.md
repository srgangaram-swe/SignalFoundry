# ADR 0018 — Bounded distributed research execution

- **Status:** Accepted
- **Date:** 2026-08-06
- **Work item:** SF-S5-MR8 (#47), Signal Foundry Sprint 5
- **Builds on:** ADR 0014 (frozen perturbation and qualification gate)

> **Measurement-method correction (2026-08-08):** the table below is the historical,
> single-sample observation used for the original architecture decision. It is not the current
> close-out reference and must not be interpreted as a distribution or SLA. [ADR
> 0022](0022-content-addressed-sprint-evidence.md) supersedes the measurement method with bounded
> warm-ups, repeated raw samples, dispersion, parity identities, content-addressed inputs, and
> transactional publication. Dask remains the selected future framework, but the numeric
> single-sample crossover and speedup are no longer sufficient adoption evidence.

## Context

The robustness sweeps built in Sprint 4 are embarrassingly parallel: each grid
point has an isolated seed stream and no cross-point dependency. That structure
invites distribution, and the work item names the obvious trap directly —
"distributing unprofiled code" is an explicit non-goal.

So the pipeline was profiled first.

### Measurement 1 — where the time goes

A 36-point `RobustnessGrid` sweep with rolling-window feature evaluation, three
repeats, minimum taken (macOS, arm64, CPython 3.13):

| Stage | Wall time | Parallelizable |
| --- | --- | --- |
| Grid construction | 0.06 ms | no |
| Per-point evaluation | 17.17 ms | **yes** |
| Aggregation | 0.42 ms | no |
| **Total** | **17.66 ms** | |

Parallel fraction **0.9727**. Amdahl bound: **36.6×** at infinite workers,
**6.7×** at eight. Structurally, this is close to the ideal case.

### Measurement 2 — where distribution actually pays

> **Correction history (2026-08-08).** [ADR 0021](0021-reproducible-sprint-evidence.md)
> first identified that the table below came from one unwarmed sample per work size,
> but its replacement evidence and publication mechanism later failed independent
> provenance and race-safety review. [ADR 0022](0022-content-addressed-sprint-evidence.md)
> supersedes that attempt. The current source-bound repeated record observes no
> process-pool break-even at any declared size in its macOS `spawn` environment.
> Both earlier numeric intervals are historical, unsupported as current release
> evidence, and preserved only so the correction remains auditable.

The parallel fraction says what *could* be gained. It says nothing about the
fixed cost of distributing, which is what decides whether any of it is realizable.
32 tasks, 8 workers, `ProcessPoolExecutor`, same machine:

| Per-task cost | Serial | 8 workers | Speedup | Parity |
| --- | --- | --- | --- | --- |
| 0.045 ms | 1.4 ms | 101.8 ms | **0.01×** | ✓ |
| 1.97 ms | 63.1 ms | 96.8 ms | **0.65×** | ✓ |
| 19.7 ms | 630.1 ms | 227.6 ms | **2.77×** | ✓ |
| 79.5 ms | 2542.4 ms | 636.4 ms | **4.00×** | ✓ |

**The crossover sits between 2 ms and 20 ms per task.** Below it, distribution is
up to a hundred times *slower*: process startup and payload serialization
dominate work that takes microseconds.

And the Sprint 4 sweep measured above runs at **0.49 ms per point** — comfortably
*below* the crossover. A 97% parallel fraction with a sub-millisecond task cost
is exactly the configuration that looks ideal on paper and loses to a `for` loop
in practice.

## Decision

**Dask is selected as the cluster framework**, and its adoption is gated on a
measured threshold rather than taken now.

*Why Dask over Ray.* The workload is a static, embarrassingly parallel task
graph over pure functions. Ray's distinguishing strengths — stateful actors, a
distributed object store, serving, RL primitives — are all things this workload
does not have and does not want; adopting them would mean carrying an actor
runtime for a `map`. Dask's `LocalCluster` needs no external service, its
task-graph model matches the shape of a parameter sweep directly, and its
scheduler exposes per-task resource annotations that map onto `ResourceRequest`
without translation. The work item also forbids adding both, and Ray's
operational surface is the larger of the two to carry unused.

*Deployment boundary.* Workers run pure functions over JSON-serializable
payloads. No worker touches the broker, credentials, the durable state store, or
any network endpoint — the distributed layer computes, it does not act.

*Serialization.* Payloads are JSON-serializable by contract, validated at
`TaskSpec` construction rather than at submission, so a malformed payload fails
where it was written. Results must be content-hashable for the same reason:
a result with no stable bytes cannot be parity-checked.

*Scheduling.* Every task declares CPU, RAM, GPU, scratch, and expected duration.
There are no defaults on the safety-relevant fields — an under-declared task is
the one that gets a machine killed.

*Failure model.* Four named terminal outcomes: `succeeded`, `failed`,
`timed_out`, `cancelled`. Retries are bounded and refused entirely for tasks
declaring `idempotent=False`. Every non-success carries an attributed error
naming the task.

**The local in-process backend is the reference implementation**, always
available, and defines the correct answer. `ProcessPoolExecutor` is the
intermediate rung: real parallelism and real serialization with no external
service, so serialization bugs and order-dependence surface in CI.

**Results assemble by content-addressed task identity, never by completion
order.** This is the load-bearing property. A distributed run finishes tasks in a
different order each time, and any assembly that folds in arrival sequence
produces results that vary run to run while every individual task is
deterministic — the hardest class of irreproducibility to diagnose, because
nothing is obviously wrong.

**Cluster access is never required for reproducibility.** A result reproducible
only on a cluster is not reproducible.

### Historical adoption gate (numeric evidence superseded)

> ADR 0022 supersedes the numeric sufficiency claim below. The 20 ms value is a
> conservative profiling floor, not an adoption threshold, and the 2.77× sample
> is not current evidence. Adoption now requires fresh repeated measurements on
> the representative target backend and workload, exact result parity, reported
> dispersion, and a benefit that survives operational cost.

The Dask dependency is added when, and only when, a measured profile shows:

1. parallel fraction ≥ 0.5 (`MIN_USEFUL_PARALLEL_FRACTION`), **and**
2. per-task cost ≥ 20 ms — above the measured crossover, **and**
3. a batch large enough that the ≥ 2.77× observed speedup exceeds the
   operational cost of running a cluster at all.

Sprint 4's sweep satisfies (1) and fails (2) by roughly forty-fold. Adding a
distributed-computing dependency to accelerate a 17-millisecond workload would
be decorative complexity, and the dependency standard requires a reason that
this measurement does not currently supply.

## Consequences

**Accepted costs.**

- The cluster backend does not exist yet, so the "distributed" capability is
  currently local and process-pool only. That is what the measurement supports;
  claiming more would be claiming an unmeasured benefit.
- Requiring content-hashable results rules out returning fitted model objects
  directly. Callers return hashable summaries, which is better discipline anyway.
- Declaring resources on every task is verbose. The verbosity is the point.
- `ProcessPoolExecutor` requires module-scope functions. A real constraint, and
  the same one Dask and Ray impose.

**What this does not buy.**

- No GPU scheduling is exercised; `gpus` is declared and validated but no backend
  honours it yet.
- No network partition testing, because there is no network. That criterion
  cannot honestly be claimed until a cluster backend exists.
- Per-task and shared batch budgets are checked only between completed
  in-process calls. A call that never returns cannot be preempted by this
  backend; hard preemption requires an independently killable worker.

## Alternatives considered

**Adopt Ray.** Rejected: actor and object-store strengths are irrelevant to a
static task graph over pure functions, and its operational surface is larger.

**Adopt Dask now, unconditionally.** Rejected on the measurement. At 0.49 ms per
task the framework would make the sweep slower while adding a substantial
dependency tree. The gate records exactly what would change that.

**Add both and let callers choose.** Explicitly forbidden by the work item, and
rightly: two orchestration frameworks means two failure models, two
serialization contracts, and two sets of operational knowledge.

**Assemble results in completion order for throughput.** Rejected as the central
hazard. It is faster and it silently destroys reproducibility.

**Skip profiling and distribute the obviously parallel sweep.** Rejected by the
work item and vindicated by measurement 2: the obviously parallel sweep is
currently below the crossover, and distributing it would have been a measurable
regression presented as an improvement.
