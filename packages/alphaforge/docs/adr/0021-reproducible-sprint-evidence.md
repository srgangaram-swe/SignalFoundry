# ADR 0021 — Reproducible, content-addressed sprint evidence

- **Status:** Superseded by [ADR 0022](0022-content-addressed-sprint-evidence.md)
- **Date:** 2026-08-08
- **Work item:** SF-S5-MR11 (#112), Signal Foundry Sprint 5
- **Amends:** ADR 0018 (bounded distributed research execution) — see "Correction" below
- **Supersedes:** nothing
- **Superseded by:** ADR 0022 after post-merge provenance and publication-race review

> **Historical attempt; not current release evidence.** PR #114 implemented this decision,
> but a post-merge acceptance audit found that the benchmark was not bound to the exact
> workload, harness, executor, dependency lock, and realized task graphs; the delivery
> ledger read current working-tree files rather than frozen blobs; verification did not
> independently rederive semantic artifacts; and `os.replace` could overwrite a raced
> destination. The numeric crossover and test-function claims below therefore describe
> the superseded attempt and must not be used as current evidence. ADR 0022 records the
> fail-closed replacement while this document remains intact as decision history.

## Context

The Sprint 5 close-out generator carried this docstring:

> *"Every panel is generated from the modules themselves rather than from
> transcribed numbers, so a figure that disagrees with the code is impossible."*

Twelve lines below it sat two tuples of hand-typed numbers. Two of the four
panels were transcription. The claim was false, and it was repeated in the PR
body, the sprint report, and the closing comment on #49.

Three consequences followed from the same root:

1. **The transcribed crossover values disagreed with ADR 0018.** The ADR recorded
   one benchmark run; the generator recorded a second run of the same procedure.
   Both were presented as the finding.
2. **Each figure was one unwarmed sample.** No dispersion, so nothing
   distinguished a real 2.98× from noise.
3. **The bundle was written in place with a filename-only manifest.** A failure
   partway through left something that looked like a bundle, and no artifact
   could be checked for tampering afterwards.

## Correction to ADR 0018

ADR 0018's decision — select Dask, gate adoption on a measured per-task
threshold — is **unchanged and still correct**. Its supporting table is amended:
those numbers came from a single unwarmed sample per work size and are not
reproducible at the precision they implied.

The original table is preserved in ADR 0018 with an annotation pointing here. It
is not rewritten: a decision record that edits its own evidence after the fact is
worth less than one that shows the correction.

Re-measured with 1 warmup and 7 repetitions per size, medians with min–max range:

| Per-task cost | Speedup (median) | Range |
| --- | --- | --- |
| 0.052 ms | 0.02× | 0.02–0.02× |
| 2.04 ms | 0.59× | 0.56–0.66× |
| 20.84 ms | 2.74× | 2.65–3.06× |
| 85.82 ms | 3.96× | 3.85–4.32× |

The **crossover interval is 2.04–20.84 ms per task**, reported as an interval
because the samples bound where break-even lies without locating it. The
qualitative conclusion is unchanged: this repository's sweep at ~0.5 ms per task
sits well below it, so distributing it would be a regression.

## Decision

**Statistics are properties over raw samples, never stored fields.** A
`WorkSizeMeasurement` holds nanosecond samples; median, range, and speedup are
computed. No published number can disagree with the samples behind it because
there is nowhere to put a disagreeing number.

**A single sample is refused.** `MIN_REPETITIONS` is 7 and at least one warmup is
required. One timing cannot show dispersion, and dispersion is what separates a
finding from noise.

**Backend parity is proven before a ratio is reported.** A speedup between
backends that disagree is not a speedup.

**The environment is part of the measurement**, and records from different
environments cannot be combined. It excludes anything identifying — no username,
home path, hostname, or process environment.

**Delivery scope is derived from Git**, not counted by hand: frozen commit
identities plus tracked paths, with test functions counted by parsing source.
**The label states exactly what is counted** — module-level test functions, not
collected parameter cases and not passing tests. One parametrized function is one
function.

**Publication is all-or-nothing**: staging sibling → validate → fsync → rename
into a destination that must not exist. A failure removes staging and creates
nothing.

**The manifest records a SHA-256 and byte size for every other artifact**, so one
changed byte fails verification. It cannot record its own digest — writing the
digest changes the bytes it covers — and the manifest says so rather than leaving
the omission to look like an oversight.

**Regeneration is byte-identical**, including the PNG. Matplotlib's default
metadata stamps its version and a creation time; both are suppressed.

**A test parses the generator's AST** and fails on module-level float or large
integer literals, because the original failure was not a typo — it was a literal
that drifted from what it described while the docstring kept asserting otherwise.

## Consequences

**Accepted costs.**

- The benchmark takes minutes rather than seconds: 8 backend runs per size
  instead of 2. Correct — the cheap version produced numbers that were wrong.
- Raw samples are committed, adding bytes to the repository. They are the
  evidence; a summary without them is an assertion.
- Regenerating the bundle requires the raw records to exist. The generator
  refuses to substitute defaults, so a missing input is a loud failure.
- Byte-identical PNGs depend on matplotlib's metadata behaviour, which a version
  bump could change. The test would catch that.

**What this does not buy.**

- Measurements remain single-machine wall clock on a synthetic workload. More
  repetitions reduce noise; they do not make this a cluster benchmark or an SLA.
- The manifest establishes content identity, not authorship or external
  provenance. Anyone able to rewrite both bundle and manifest defeats it, exactly
  as for the event journal.
- The ledger counts what is tracked at the named paths; it cannot verify that a
  test is meaningful.

## Alternatives considered

**Fix the numbers and keep the constants.** Rejected: it addresses the symptom.
The next regeneration would drift again, and the docstring would still be lying.

**Delete the false claim and keep transcribing.** Rejected. The claim was worth
making; it was the implementation that failed to earn it.

**Import test modules to count tests via collection.** Rejected: executing every
test module's import-time code inside an evidence generator is a much larger
blast radius than parsing files, and slower.

**Report a single interpolated crossover point.** Rejected. Four sizes bound the
interval; they do not locate a crossing, and a single number would imply a
precision the samples do not support.

**Rewrite ADR 0018's table in place.** Rejected. A decision record that silently
edits its evidence is worth less than one that carries its correction visibly.
