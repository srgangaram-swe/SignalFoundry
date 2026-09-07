# ADR 0020 — Live-readiness gate and inert capital configuration

- **Status:** Accepted
- **Date:** 2026-08-08
- **Work item:** SF-S5-MR10 (#49), Signal Foundry Sprint 5
- **Builds on:** ADR 0014 (qualification gate), ADR 0015 (broker selection), ADR 0016 (deny-by-default broker authorization), ADR 0017 (halt on divergence)

## Context

This is the last gate before real money, and it has a property no other module in
the repository has: **the person operating it wants it to pass**. They will have
spent five sprints building the infrastructure. They will be able to argue that
the remaining items are formalities. They may be right about most of them.

That is the threat model. Not an attacker — the author, at the end of a long
project, reasoning their way toward a conclusion they already want.

A second problem is that two of the required conditions cannot be verified by
software at all. Whether an employment policy permits personal trading, and
whether the applicable legal and tax obligations have been met, are human
judgements. A framework that pretends to check them is worse than one that admits
it cannot.

## Decision

**No score, no weighting.** Every checklist item is required. A weighted score
would let sixteen strong items outvote one missing legal review, and the missing
legal review is the one that matters.

**No override anywhere.** There is no `force`, `waive`, `skip`, `override`,
`acknowledge_risk`, `bypass`, `unsafe`, or `ignore_unmet` parameter in either
module, and a test parses both ASTs to prove it. An item that genuinely should
not apply is removed by publishing a **new checklist version**, which leaves a
record of who removed it and when.

**Absence is failure, not omission.** An item with no evidence is unmet.
Readiness is demonstrated; it is never inherited by silence.

**The checklist has a content identity.** Editing it to admit a candidate changes
the digest, and evaluation against a recorded identity refuses the edited one.

**Attestations are recorded, not verified — and say so.** Policy and legal items
require a named attester, a dated statement, and a validity window. The framework
records the attestation and states plainly on the record that it cannot confirm
the review occurred or reached that conclusion. What *is* checkable — that an
attester was named, that the date is not in the future, that it is not stale — is
checked.

**Attestations expire at 180 days.** Employment policy, legal posture, and
personal circumstances change. A two-year-old sign-off is not current consent.

**Capital configuration is inert by default** and there is **no method that
raises a cap**. Not a guarded one, not a validated one — none. Increasing
exposure requires constructing a new configuration with a new authorization
naming the new cap, which leaves a record. A mutable cap behind a permission
check is one bug away from an unbounded one; an absent method is not.

**Authorization is bound to what it authorizes.** It names the cap, the checklist
identity the decision was made under, the approver, and an expiry. It cannot be
recycled onto a different checklist or a larger cap.

**An absolute ceiling exists independent of any approval.** A first deployment is
an operational test of the plumbing. Sizing it for return defeats the purpose,
and a ceiling in code cannot be argued with at 2am.

**Everything is reversible.** `deactivated()` returns an inert copy; there is no
partial teardown and no state to unwind.

## Consequences

**Accepted costs.**

- The gate is hard to pass, deliberately. Seventeen items, two requiring dated
  human attestation, all required.
- Attestations must be renewed every 180 days, which is friction on a long-running
  deployment. That friction is the point: it forces periodic reconsideration.
- Raising a cap requires a new authorization rather than an edit. Inconvenient by
  construction.
- The absolute ceiling makes the first deployment economically pointless as a
  profit exercise. It is not meant to be one.

**What this does not buy.**

- **It cannot verify the attestations.** A false attestation produces a READY
  verdict. This is a genuine hole and it is unclosable in software; the framework
  narrows it to a named person on a dated record rather than a silent boolean.
- It does not check that the evidence flags are truthful either — it checks that
  they were supplied. Each flag's underlying evidence is the responsibility of
  the module that produces it.
- No order routing, no broker interaction, no position sizing.

## Alternatives considered

**A weighted readiness score with a pass threshold.** Rejected as the motivating
failure. It is precisely the mechanism by which a missing legal review gets
outvoted by engineering work that was more fun to do.

**A `--force` flag for operators who know what they are doing.** Rejected. The
operator who knows what they are doing is exactly the person this gate is
protecting against, because they are the one with a persuasive argument.

**Treating policy and legal items as ordinary booleans.** Rejected: a boolean
does not record who decided, when, or on what basis, and those are the only
things that make a human judgement auditable.

**Attestations that never expire.** Rejected. An approval that never expires
eventually authorizes something nobody looked at.

**A mutable cap with a permission check.** Rejected in favour of no method at
all. Permission checks have bugs; missing methods do not.

**Omitting the absolute ceiling and trusting the approval.** Rejected. The
approval is written by the same person the gate protects against, at the same
moment they most want to proceed.
