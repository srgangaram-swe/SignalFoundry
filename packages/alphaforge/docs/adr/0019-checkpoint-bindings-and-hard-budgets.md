# ADR 0019 — Checkpoint bindings and hard budgets

- **Status:** Accepted
- **Date:** 2026-08-08
- **Work item:** SF-S5-MR9 (#48), Signal Foundry Sprint 5
- **Builds on:** ADR 0017 (durable session state), ADR 0018 (bounded distributed execution)

## Context

MR8 gave tasks content-addressed identity and bounded execution. Two gaps
remained, and both are the kind that only show up after something goes wrong.

**Resumption.** A checkpoint that records only "which task ids finished" lets a
run resume after the code changed, the data was re-pulled, a dependency was
upgraded, or the seed moved. The resumed run then reports a single result
assembled from two different experiments — and nothing in the artifact says so.

**Budgets.** A limit checked only while running is not a limit: by the time it
fires, the resource is spent. And a limit that trusts the declaration is not a
limit either, because the task that declared one hour and is now four hours in
declared honestly and was wrong.

## Decision

**A checkpoint binds six things, not one.** `code_hash`, `data_hash`,
`config_hash`, `dependency_hash`, `task_graph_hash`, and `root_seed`. Resumption
verifies every binding and names the one that broke, with an explanation of what
changing it would mean. Changing any makes the resumed run a different experiment
wearing the original's name.

**Checkpoint writes are atomic via `os.replace`.** Write to a temporary sibling,
fsync, then rename. A crash mid-write leaves the previous checkpoint intact
rather than a truncated one — which matters most here, because a corrupt
checkpoint destroys a resumable run and forces a full restart.

**Concurrent writers are detected, not merged.** Two processes checkpointing one
experiment would produce a manifest describing work no single run performed.
A foreign writer is refused unless a caller explicitly consents.

**Budgets are enforced twice.** Admission compares declared requirements against
the limit before any worker starts, so a hopeless batch fails at zero cost.
Runtime enforcement compares observed consumption, catching the honest
declaration that turned out wrong. Both are hard refusals; there is no `soft`,
`warn_only`, `allow_overage`, or `best_effort` parameter, and a test parses the
module AST to prove it.

**Wall time is charged as the critical path, not the serial total.** Tasks run in
parallel; charging a batch the sum of its durations would refuse work that fits
comfortably. Memory is peak-concurrent for the same reason. GPU-hours and storage
accumulate, because those are consumed in total regardless of overlap.

**Cost is denominated in abstract units, not currency.** This repository has no
billing integration, and a dollar limit would imply a price feed that does not
exist.

**Breaches produce machine-readable evidence** naming the limit, the bound, the
observed value, and the overage. "The job was cancelled" is not actionable.

## Consequences

**Accepted costs.**

- Callers must compute and supply five content hashes. That is real work, and it
  is the work that makes resumption trustworthy.
- A dependency upgrade invalidates in-flight checkpoints. Correct: the completed
  and remaining halves would otherwise run under different numerics.
- Admission needs declared durations to be roughly honest. A wildly optimistic
  declaration passes admission and is caught later by enforcement — later than
  ideal, but caught.
- Single-writer only. Multi-writer checkpointing would need a lease protocol.

**What this does not buy.**

- No automatic cost metering. `cost_units` must be supplied by the caller; the
  module enforces a limit it cannot itself measure.
- No mid-task preemption. Enforcement fires between observations, so a task can
  overshoot within one interval.
- Atomicity is per-filesystem. `os.replace` across filesystems is not atomic, and
  the store does not detect that configuration.

## Alternatives considered

**Record only completed task ids.** Rejected as the motivating failure: it
permits resuming into a different world and produces one artifact describing two
experiments.

**Migrate incompatible checkpoints automatically.** Rejected. A silent migration
is how a resumed run acquires semantics nobody chose. Schema changes require
deliberate work.

**Write checkpoints in place.** Rejected: a crash mid-write leaves a truncated
file that fails verification, turning a recoverable interruption into a full
restart.

**Soft budgets that warn and continue.** Rejected by the work item's non-goal and
on the merits — a warning nobody reads is the same as no limit, and the resource
is gone either way.

**Charge wall time as the serial total.** Rejected: it would refuse batches that
fit comfortably under real concurrency, which trains operators to inflate limits
until they stop meaning anything.
