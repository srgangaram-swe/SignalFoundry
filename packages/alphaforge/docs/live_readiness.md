# Live readiness and minimal-capital configuration (SF-S5-MR10)

A versioned checklist that cannot be talked into saying READY, and a capital
configuration that is inert by default and cannot expand its own risk.

> **Current verdict: `NOT_READY`, with every item unmet.** No strategy is
> qualified, no paper trading has run, and no attestation exists. That is the
> honest state of this repository at the close of Sprint 5.

Rationale: [ADR 0020](adr/0020-live-readiness-gate-and-inert-capital.md).
Requirements: [broker connectivity §5](broker_connectivity_requirements.md).

---

## 1. The threat model is the author

Every other module in this repository defends against bad data, a broker that
disagrees, or a process that crashes. This one defends against **the person
running it**, who will have spent five sprints building the infrastructure and
will be able to argue that the remaining items are formalities.

So the design removes the mechanisms by which that argument usually wins:

| Mechanism | Why it is absent |
| --- | --- |
| A weighted score | Sixteen strong items would outvote one missing legal review |
| A `--force` flag | The operator who "knows what they're doing" is who this protects against |
| Silence defaulting to pass | Readiness is demonstrated, never inherited by omission |
| An editable checklist | Content identity makes an edit to admit a candidate detectable |
| A mutable cap | Permission checks have bugs; missing methods do not |

A test parses the AST of both modules and asserts no function accepts `force`,
`waive`, `skip`, `override`, `acknowledge_risk`, `bypass`, `unsafe`, or
`ignore_unmet`.

## 2. Seventeen items, six categories, all required

| Category | Items |
| --- | --- |
| Evidence | qualified candidate, paper duration, paper stability, cost-model validation |
| Reconciliation | clean reconciliation history, current broker state |
| Operational | rehearsal, broker-failure drill, kill switch, deactivation procedure, audit/tax export |
| Security | capital cap enforced, risk limits enforced, credential custody |
| Policy | employment-policy review, owner approval |
| Legal | legal/regulatory review |

Categories exist because **remedies differ**: an evidence gap is closed by running
something; a legal gap is not. `unmet_by_category()` groups them accordingly.

## 3. What the framework cannot do

**It cannot verify an attestation.** Whether an employment policy permits personal
trading, and whether legal and tax obligations have been reviewed, are human
judgements. A false attestation produces a READY verdict.

This is a real hole and it cannot be closed in software. What the framework does
instead is narrow it: policy and legal items require a **named attester, a dated
statement, and a reference**, and the record carries the sentence *"Recorded, not
verified. Software cannot confirm that a policy or legal review actually occurred
or reached this conclusion."* A silent boolean would have hidden the same hole
without the disclosure.

What *is* checkable is checked: an anonymous attester is refused, a future-dated
attestation is refused, and one older than **180 days** is stale — policy and
personal circumstances change, and an old sign-off is not current consent.

## 4. Capital is inert until four conditions hold

`LiveCapitalConfig.inert()` is the default: disabled, zero cap, deploys nothing.
`activate()` requires all four, each alone sufficient to refuse:

1. The readiness decision is `READY_FOR_MINIMAL_CAPITAL` (and is a real
   `ReadinessDecision`, not a duck-typed stand-in).
2. The authorization names **that decision's** checklist identity.
3. The authorization is within its window.
4. The requested cap is within the approved cap.

Plus two ceilings that no approval can raise: an **absolute capital ceiling**, and
a **30-day maximum authorization span**. A first deployment is an operational test
of the plumbing; sizing it for return defeats the purpose, and a ceiling in code
cannot be argued with at 2am.

Risk-limit *fractions* are also bounded above, so a configuration cannot be
written with limits so loose they impose nothing while passing review.

## 5. Nothing can raise a cap

There is no `raise_cap`, `increase_limit`, `expand`, or `set_cap` method, and a
test asserts no such name exists on the class. Increasing exposure means
constructing a new configuration with a new authorization naming the new cap —
which leaves a record.

`deactivated()` returns an inert copy. Reversal is total: no partial teardown, no
state to unwind.

## 6. Runbook

**Evaluating readiness:** call `evaluate_readiness` with the checklist, the
evidence map, current attestations, and — importantly — `expected_identity`, so a
checklist edited since the last evaluation is refused.

**On NOT_READY:** the report leads with unmet items grouped by category. Close
them; do not edit the checklist. Editing to admit a candidate is what the identity
check exists to catch.

**On READY:** obtain a written authorization naming the cap and the checklist
identity, then `activate()`. Record the configuration identity for audit.

**Deactivating:** call `deactivated()`. It is immediate and total.

**Rollback:** the package is additive and nothing else imports it. Reverting the
commit removes it.

## 7. Residual limitations

- **Attestations are unverifiable.** §3. The largest residual risk here, and it is
  disclosed rather than papered over.
- **Evidence flags are not independently audited.** The framework checks that a
  flag was supplied, not that the underlying work happened. Each flag's evidence
  is the responsibility of the module producing it.
- **No order routing or position sizing.** A READY verdict authorizes a separately
  configured deployment; it does not build one.
- **The absolute ceiling is a judgement**, chosen to make the first deployment an
  operational test rather than an investment.

## 8. Evidence

`tests/test_live_readiness.py` — 66 tests: the empty-evidence default; **every one
of the seventeen items parametrized to prove it alone blocks**; the fully
evidenced pass; the READY-with-unmet and NOT_READY-without-unmet contradictions;
the AST assertions that no override parameter and no cap-raising method exist; the
absence of weighting; attestation requirements, staleness, future-dating, and
anonymity; the unverified-attestation disclosure; checklist identity detection of
an item removed to admit a candidate; order-independent identity; empty and
duplicate checklists; unknown and non-boolean evidence; category coverage; inert
defaults; enabled-without-authorization; activation refusals on all four
conditions plus both ceilings; the duck-typed stand-in; valid activation and its
computed position cap; total deactivation; float refusal; limits too loose or
zero; runtime breach detection on three limits individually and together; report
ordering; and configuration identity.
