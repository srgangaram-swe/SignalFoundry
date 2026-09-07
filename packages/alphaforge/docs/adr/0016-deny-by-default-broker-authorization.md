# ADR 0016 — Deny-by-default broker authorization

- **Status:** Accepted
- **Date:** 2026-08-04
- **Work item:** SF-S5-MR3 (#45), Signal Foundry Sprint 5
- **Builds on:** ADR 0014 (frozen perturbation and qualification gate), ADR 0015 (broker selection)

## Context

This is the first module in the repository whose defect mode is *spending real
money*. Everything before it produced numbers; a broker adapter produces orders.
The failure that matters is not an incorrect result — it is a correct-looking
system that contacts a live endpoint because three conditions lined up wrongly.

The obvious design is a `live` flag defaulting to `False`. That is a
configuration value, and configuration values get set: by a copied example, by an
environment override, by a well-meaning edit during debugging, by a test fixture
that leaks. A flag makes live trading *one assignment away* at all times.

A second pressure pushes the same direction. Issue #45's dependencies name a
Sprint 4 `QUALIFIED_FOR_PAPER` decision, and Sprint 4's verdict is `REJECTED`.
The honest options were to build nothing, or to build the mechanism such that it
structurally cannot operationalize an unqualified strategy. Building nothing
wastes a sprint on a dependency that engineering cannot resolve.

## Decision

**Live capability is absent, not disabled.** No code path in
`alphaforge.broker` constructs a live endpoint, and no parameter enables one. A
test parses the module AST and asserts no function anywhere in the package
accepts an argument named `force`, `allow_live`, `override`, `skip_checks`,
`unsafe`, or `bypass`. Adding live capability is a future change that must pass
its own review — it is not a value someone can set.

**Endpoints are allowlisted, not denylisted.** A denylist of live hosts fails
open the moment a vendor introduces a hostname nobody added. Failing open here
routes a real order, so the check refuses the unknown by construction.

**Three independent conditions, each individually sufficient to refuse:**
explicitly enabled, allowlisted paper endpoint, and a `QUALIFIED_FOR_PAPER`
decision. They are checked in that order so the qualification message — the one
that cannot be fixed by configuration — is the last a caller sees.

**The qualification argument is type-checked, not duck-typed.**
`authorize_paper_session` requires an actual `QualificationDecision` instance. An
object with `qualified = True` is refused, because the whole point is that the
verdict came from the frozen rubric rather than from the caller.

**Credentials never touch the environment.** Six known credential variables are
checked and their presence is a refusal. The message names the variable and
withholds the value.

**Account identifiers are stored only as SHA-256 digests.** `AccountSnapshot`
refuses a non-digest, so the raw identifier cannot be persisted or logged even by
accident.

**The adapter imports no networking library**, asserted by parsing its imports.

**The kill switch is one-way** with no `disengage` method.

**Money is `Decimal`; `float` is refused at the boundary.**

## Consequences

**Accepted costs.**

- Callers must construct `Decimal` values. Less convenient than floats, and the
  convenience is exactly what causes cent-level drift that fails a reconciliation
  which is working correctly.
- A paper session cannot currently be opened at all. This is the correct state
  and the tests exercise it as the primary path rather than an edge case.
- The AST tests couple to module structure and will need updating if the package
  is reorganized. Accepted: a structural assertion that must be maintained is
  stronger than a comment that cannot fail.
- Adding a live adapter later requires deliberate work rather than a flag flip.
  That is the point.

**What this does not buy.**

- The environment check is a snapshot at authorization time and cannot prevent a
  credential set afterwards.
- No persistence: restart recovery is #46. Order state is in memory.
- Simulated fills bound operational readiness only.

## Alternatives considered

**A `live: bool = False` flag.** Rejected as the motivating failure. It makes
live trading one assignment away, and the assignment can arrive from a copied
config, an environment override, or a test fixture.

**A denylist of known live hostnames.** Rejected: fails open on any hostname
nobody thought to add, and the failure routes a real order.

**Accepting any object with a `qualified` attribute.** Rejected. Duck typing here
would let a caller satisfy the most important gate in the package with a
two-line stand-in; a test asserts the stand-in is refused.

**Floats for money.** Rejected. `0.01` is not representable in binary floating
point; the drift is invisible until a reconciliation fails, and then the
reconciliation looks wrong when it is right.

**Building nothing until a strategy qualifies.** Rejected. The mechanism is
independently reviewable and its correctness does not depend on research
outcomes. Gating it structurally is strictly better than deferring it, because
the gate is what makes the mechanism safe to hold.

**A kill switch with a reset.** Rejected. A switch code can flip back is not a
kill switch. Re-enabling must construct a new session deliberately.
