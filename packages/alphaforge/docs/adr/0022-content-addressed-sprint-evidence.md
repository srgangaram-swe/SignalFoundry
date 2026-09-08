# ADR 0022 — Content-addressed and transactional sprint evidence

- **Status:** Accepted
- **Date:** 2026-08-08
- **Work item:** SF-S5-MR11 (#112), Signal Foundry Sprint 5
- **Supersedes:** ADR 0021 and corrects ADR 0018's single-sample performance table plus the first two Sprint 5 close-out bundles

## Context

The first AlphaForge Sprint 5 close-out bundle copied four wall-clock observations and six
delivery counts into Python constants. Its figure then described those values as if they had been
computed from source evidence. The performance values also differed from ADR 0018, which recorded
a separate single run. Neither set reported warm-up policy, repeated samples, or dispersion. The
publisher wrote artifacts directly into the final directory and its manifest listed filenames but
did not bind their bytes.

PR #114 and ADR 0021 made a first corrective attempt, but its benchmark record still did not bind
the exact workload, task builder, harness, executor, dependency lock, or realized task graphs. Its
delivery ledger read current working-tree files instead of frozen Git blobs. Its verifier checked
file hashes without independently re-deriving the semantic artifacts, and its check-then-
`os.replace` publication could overwrite a concurrently created destination. That attempt is
retained in history and explicitly superseded rather than silently rewritten.

Those defects do not change the distributed executor, readiness gate, or capital boundary. They do
make the close-out evidence unsuitable for release: a reviewer cannot determine which measurement
supports the current figure, reproduce its variability, or detect a one-byte modification.

## Decision

Sprint close-out evidence is treated as a bounded publication transaction over two independently
verifiable inputs:

1. A benchmark record contains the complete declared workload, at least one warm-up, at least seven
   measured repetitions, every monotonic-clock duration, deterministic backend order, output
   identity, parity result, and a privacy-safe machine/runtime description. It also binds the
   workload, task builder, harness, evidence contract, executor, task contract, dependency lock,
   and realized task-declaration graphs to exact content identities. The runner rejects callable,
   source, task-graph, and result-membership mismatches. Medians and dispersion are derived when
   evidence is consumed; no performance value is duplicated in source code.
2. A delivery inventory is derived from the frozen Git squash commits and their first-parent diffs.
   It records exact commit, tree, parent, path, blob, mode, byte size, and deterministic category.
   The plot labels these values as per-slice tracked-path changes—not unique paths, tests,
   collected cases, effort, or quality.

The delivery panel covers the six planned capability slices from SF-S5-MR2 through SF-S5-MR10.
It intentionally excludes both evidence-correction attempts: a publisher cannot include its own
future squash commit without circular provenance, and those corrections do not add a seventh
trading capability. Their commits remain independently visible in Git and the pull-request record.

The committed benchmark workload is resource-bounded by work sizes, task count, worker count,
warm-ups, repetitions, and total declared operations. A shared monotonic budget is checked between
builder advances and backend returns and is passed to each executor, but it cannot preempt arbitrary
in-process Python that never returns; that limitation is explicit rather than described as a hard
deadline. The serial implementation remains the correctness reference; every process-pool sample
must have identical result identity. Timing never becomes a CI pass/fail threshold.

The evidence publisher validates both inputs before rendering, writes the complete allowlisted
payload into a unique sibling staging directory under an exclusive cooperative-writer lock, flushes
files and directory metadata, verifies the staged artifact set, and renames the directory into a
previously absent destination through the platform's atomic no-replace primitive. There is no
check-then-rename fallback: an unsupported filesystem fails closed. Any pre-commit failure removes
only this invocation's identity-checked staging directory and lock. An existing or concurrently
created destination is never overwritten.

The manifest records the schema, a content inventory of every material generator source and locked
renderer dependency, the frozen delivery source head, input identities, bounded environments, and
SHA-256 plus byte size for every payload artifact. It intentionally does not hash itself: embedding
its own digest would change the bytes being hashed and recurse forever. The Git commit anchors the
manifest. An independent verifier validates the exact manifest schema and canonical encoding,
checks every payload byte, strict-parses both raw inputs, re-derives the CSV and readiness artifacts,
and reconciles their identities and claims.

The Seaborn figure presents raw benchmark repetitions alongside their summaries, the observed
below/above-break-even bracket when one exists, the fail-closed readiness verdict, the inert capital
cap, and the Git-derived delivery inventory. In the corrected reference run no declared size has a
median speedup above 1×, so the evidence reports that no break-even was observed. It does not claim
a universal crossover, cluster performance, a qualified strategy, paper readiness, or live
readiness.

ADR 0018's 20 ms figure remains a conservative floor, not a sufficient adoption threshold. A
cluster dependency now requires a new repeated profile of the representative workload on the
target backend and environment, exact result parity, and a benefit that survives reported
dispersion and operational cost. The historical single-sample 2.77× value cannot satisfy that gate.

## Consequences

- Current close-out claims are traceable to committed raw evidence and frozen Git objects.
- Repeated measurements expose variance and unfavorable observations instead of selecting one run.
- Publication failure cannot leave a partial final bundle, and content tampering is detectable.
- Reference timings remain local, single-machine engineering measurements. They are not an SLA and
  may differ on other hardware, operating systems, Python versions, or process-start methods. In
  this macOS `spawn` run, process construction dominates and every measured median is slower.
- Git path counts establish delivery scope, not semantic complexity, correctness, or productivity.
- Regeneration requires the frozen Git objects and the committed benchmark record. It does not need
  network access, market data, broker access, credentials, or a cluster.
- Exact PNG verification is renderer-environment-sensitive by design. A different font or renderer
  fails verification instead of silently producing bytes under the old manifest.

## Alternatives considered

**Keep the hand-transcribed table and clarify its date.** Rejected. A dated transcription still
cannot prove which raw samples produced it, show variability, or detect drift between code and plot.

**Run the benchmark during every CI job and compare timings.** Rejected. Shared-runner timing is
noisy and hardware-dependent. CI validates the schema, bounds, parity, derivation, and publisher;
reference timing is collected deliberately and reported without a numeric gate.

**Hash the manifest from within itself.** Rejected as self-referential. The manifest hashes every
other artifact, and the repository commit provides the manifest's external content identity.

**Count tests from source syntax or a hand-maintained table.** Rejected. A test function is not a
collected parameterized case or a passing result, and none is a defensible proxy for delivered
scope. The corrected figure reports exactly what Git can prove: per-slice tracked-path changes.

## Rollback

Reverting this slice stops Sprint 5 promotion. It must not restore the misleading
computed-from-modules claim or treat either historic single sample as current release evidence. A
replacement must preserve equivalent raw provenance, bounded validation, atomic publication,
artifact integrity, and honest limitations.
