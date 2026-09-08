# Checkpointing, resumption, and enforced budgets (SF-S5-MR9)

Checkpoints that refuse to resume into a different world, and budgets that are
admitted before they are spent.

> Research and simulation only. Nothing here places an order or touches capital.

Rationale: [ADR 0019](adr/0019-checkpoint-bindings-and-hard-budgets.md).
Builds on: [bounded distributed execution](distributed_execution.md).

---

## 1. A checkpoint binds six things

Recording "which tasks finished" is not enough. A run that resumes after the code
changed, the data was re-pulled, a dependency moved, or the seed shifted produces
**one artifact describing two different experiments**, and nothing in it says so.

`CheckpointManifest` binds all six:

| Binding | What changing it would mean |
| --- | --- |
| `code_hash` | Resumed tasks would run different logic |
| `data_hash` | Completed and remaining work saw different inputs |
| `config_hash` | The parameters are not the ones the completed work used |
| `dependency_hash` | Numerical results may differ between the two halves |
| `task_graph_hash` | Completed ids may not correspond to tasks in this graph |
| `root_seed` | Remaining tasks would draw from a different stream |

`verify_resumable` checks each and **names the one that broke**, with that
explanation attached. The task-graph hash is order-independent — reordering
submission has not changed the experiment.

## 2. Writes are atomic

Temporary sibling → `fsync` → `os.replace`. A crash mid-write leaves the previous
checkpoint intact rather than a truncated one, which matters more here than
elsewhere: a corrupt checkpoint destroys a resumable run and forces a full
restart. `test_a_failed_write_leaves_the_previous_checkpoint_intact` injects a
failure at the rename and asserts both that the previous record survives and that
no temporary file is left behind.

## 3. Concurrent writers are detected, not merged

Two processes checkpointing one experiment would produce a manifest describing
work no single run performed. The writer identity is recorded and a foreign
writer is refused, unless a caller passes `allow_foreign_writer=True` to take
over deliberately.

## 4. Budgets are enforced twice

**Admission** (`admit`) compares declared requirements against the limit *before
any worker starts*, so a batch that could never fit fails at zero cost.

**Runtime enforcement** (`enforce`) compares observed consumption. Admission
trusts the declaration; this does not. The task that declared one hour and is
four hours in declared honestly and was wrong, and only enforcement catches it.

Both are hard refusals. There is no `soft`, `warn_only`, `allow_overage`, or
`best_effort` parameter, and a test parses the module AST to prove no function
accepts one.

### How each limit is charged

| Limit | Charged as | Why |
| --- | --- | --- |
| Wall seconds | Critical path under declared concurrency | Tasks run in parallel; the serial total would refuse work that fits |
| Memory | Peak concurrent | Same reason |
| GPU-hours | Accumulated | Consumed in total regardless of overlap |
| Storage | Accumulated | Same |
| Concurrency | Simultaneous tasks | Direct |
| Cost units | Accumulated, caller-supplied | Abstract, not currency — there is no billing integration and a dollar limit would imply a price feed that does not exist |

A batch also cannot be charged less than its slowest single task, however many
workers are available.

## 5. Evidence on breach

`breach_report` emits canonical JSON naming the limit, the bound, the observed
value, and the overage. "The job was cancelled" is not something an operator can
act on.

## 6. Runbook

**Resuming:** read the checkpoint, call `verify_resumable` with the current
hashes, then run `manifest.remaining(tasks)`. Any refusal names its cause — do
not delete the checkpoint to make it go away; the mismatch is the finding.

**On a budget refusal at admission:** raise the limit deliberately or reduce the
work. The budget is not advisory.

**Rollback:** both modules are additive; nothing else imports them. Reverting the
commit removes them and checkpoint files are inert data.

## 7. Residual limitations

- **No automatic cost metering.** `cost_units` is caller-supplied; the module
  enforces a limit it cannot itself measure.
- **No mid-task preemption.** Enforcement fires between observations, so a task
  can overshoot within one interval.
- **Atomicity is per-filesystem.** `os.replace` across filesystems is not atomic,
  and the store does not detect that configuration.
- **Single-writer only.** Multi-writer checkpointing would need a lease protocol.
- **Admission needs roughly honest declarations.** A wildly optimistic duration
  passes admission and is caught later by enforcement.

## 8. Evidence

`tests/test_checkpoints_and_budgets.py` — 59 tests: every binding mismatch
including the parametrized four; the explanatory refusal message; changed task
graph, changed seed, and completed ids absent from the graph; order-independent
graph hashing; schema incompatibility; tampered manifest; duplicate completed
ids; malformed, uppercase, and short hashes; naive timestamps; atomic round trip;
injected rename failure with temp-file cleanup; truncated and field-missing
payloads; concurrent-writer detection and deliberate takeover; symlinked
directory refusal; critical-path versus serial wall time; slowest-task floor;
peak-concurrent memory; accumulated storage and GPU-hours; admission refusals on
five limits; the unexplained-refusal refusal; runtime enforcement including the
inclusive boundary; the AST assertion that no soft mode exists; machine-readable
breach evidence; and resumed-run determinism.
