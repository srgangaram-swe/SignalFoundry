# Broker contract and paper adapter (SF-S5-MR3)

Typed, vendor-neutral contracts for account, order, fill, position, clock, and
quote state, and a strictly simulated paper adapter behind a deny-by-default
authorization boundary.

> **No live capability exists.** It is *absent*, not disabled by a flag. There is
> no `force`, `allow_live`, `override`, or `skip_checks` parameter anywhere in
> the package, and a test parses the module AST to prove it. The adapter contains
> no network client — a second test parses its imports to prove that too, because
> a comment claiming "no network" is not a control.
>
> **A paper session is currently unauthorizable.** It requires a
> `QUALIFIED_FOR_PAPER` decision and Sprint 4's verdict is `REJECTED`. That is
> the intended state, not an obstacle.

Requirements this implements: [broker connectivity requirements](broker_connectivity_requirements.md).
Broker selection: [ADR 0015](adr/0015-broker-selection-for-paper-and-live-trading.md).
Design rationale: [ADR 0016](adr/0016-deny-by-default-broker-authorization.md).

---

## 1. Architecture and trust boundary

```
decision layer  ──▶  alphaforge.broker.config      (authorization: may a session exist?)
                          │
                          ▼
                     alphaforge.broker.contracts   (typed domain, no I/O)
                          │
                          ▼
                     alphaforge.broker.paper_adapter  (in-process simulation)
```

Authorization is a **separate module** from the adapter so that the decision to
enable trading cannot be made incidentally by code whose job is order routing.
`contracts` performs no I/O and names no vendor; swapping the adapter does not
change anything above it.

**The broker is adversarial until validated.** Every response field is checked
for presence, type, finiteness, sign, and mutual consistency before it reaches
accounting. A well-formed JSON body is not a semantically valid one: a reported
fill quantity exceeding what was requested is refused as a reconciliation fault
rather than recorded as a fill.

## 2. The three independent authorization conditions

Each alone is sufficient to refuse.

| Condition | Default | Failure mode it prevents |
| --- | --- | --- |
| `enabled` is `True` | `False` | A configuration that forgets to mention trading does not trade |
| Endpoint is on the paper **allowlist** | empty | Routing a real order |
| A `QUALIFIED_FOR_PAPER` decision is supplied | `None` | Operationalizing a strategy that did not clear the gate |

**Why an allowlist, not a denylist.** A denylist of live hosts fails open the
moment a vendor introduces a hostname nobody added to it — and failing open here
means a real order. The allowlist refuses the unknown by construction; a test
asserts that `https://paper-api.evil.example` is rejected despite looking like a
paper host.

A bad endpoint is validated **when it is written**, even while the session is
disabled, so a latent misconfiguration fails at authorship rather than on the day
someone flips the switch.

## 3. Credentials

- Retrieved from **macOS Keychain only**, under an allowlisted service name.
- Environment variables are **refused**: `assert_no_credentials_in_environment`
  raises when any of six known credential variables is populated, because an
  environment variable is readable by every child process and lands in crash
  dumps and process listings.
- The refusal message **names the variable and withholds the value**. A test
  asserts the secret does not appear in the exception text.
- Account identifiers are stored as **SHA-256 digests**. `AccountSnapshot`
  refuses anything that is not a 64-character digest, so the raw identifier
  cannot be persisted or logged. A digest is sufficient for the only thing the
  system needs it for: detecting that the account changed.

## 4. Time, money, and identity

**Time is UTC and explicit.** A naive datetime is refused rather than localized.
"It's local" and "it's UTC" are both wrong somewhere, and the resulting defect
surfaces at a session boundary where it is most expensive to diagnose. Aware
non-UTC datetimes are converted, not rejected.

**Money and quantity are `Decimal`.** A `float` is refused at the boundary with
an explicit message. Binary floating point cannot represent `0.01`, and an
accumulated cash balance that drifts by fractions of a cent will eventually fail
a reconciliation that is working correctly — the worst kind of failure, because
the reconciliation is right and the ledger is wrong.

**Client order IDs are content-derived**:

```
af-<sha256(strategy_id | decision_ts | symbol | side | sequence)[:32]>
```

Deterministic rather than random, so a process that restarts mid-cycle cannot
forget what it already sent. Every component changes the identifier; five
parametrized tests assert this.

## 5. Order lifecycle

`ALLOWED_ORDER_TRANSITIONS` is a closed map. Every terminal state —
`FILLED`, `CANCELED`, `REJECTED`, `EXPIRED` — maps to the empty set, and a test
asserts that for each of them.

A transition absent from the map is a **refusal, not a gap**: it means the
adapter and the broker disagree about the order's history, and applying the
update would process events out of order. A late `PARTIALLY_FILLED` for an order
already `FILLED` raises with a message saying that local and broker history
disagree and must be reconciled rather than applied.

## 6. Idempotency

Resubmitting a known `client_order_id` **returns the existing order unchanged**
and creates nothing. This is a success, not an error: after a connection loss the
caller cannot know whether the broker saw the first submission, and asking is the
only safe move. A test asserts that a duplicate submission leaves exactly one
fill and a position of 10 shares rather than 20.

## 7. The simulation is pessimistic where reality is uncertain

| Behaviour | Choice | Why |
| --- | --- | --- |
| Market order fill price | Far side of the spread | The midpoint is what you get when someone *else* pays the spread. Assuming it flatters every result. |
| Limit order | Fills only when the market is already through the limit | Never "close enough" |
| Closed market | Rests, does not fill | Simulating a closed-market fill invents liquidity that did not exist |
| Flat position | Removed from the book | A zero-quantity row must not look like a position during reconciliation |

**Paper fills are not evidence of executable performance.** There is no queue
position, no contention, no partial-fill dynamics under real depth, and no borrow
scarcity. This bounds *operational* readiness — that the plumbing works,
reconciles, and fails closed — and nothing more. The claim travels on
`to_dict()["evidence_note"]` so it cannot be separated from the numbers.

## 8. Failure classification

`RetryableBrokerError` versus `TerminalBrokerError` is the distinction that
decides whether a caller may resubmit. Getting it wrong in the safe direction
costs a missed cycle; getting it wrong in the unsafe direction duplicates an
order. **Anything unrecognized is therefore terminal** — an error the adapter has
never seen is not evidence that retrying is safe.

Terminal conditions currently enforced: stale quote, future-dated quote, clock
skew beyond bound, mismatched quote symbol, insufficient buying power, unknown
order, cancellation of a settled order, and stale account snapshot.

## 9. Kill switch

One-way for the lifetime of the session object. Every order-facing call raises
`KillSwitchEngagedError` afterwards. There is **no `disengage`**: a switch that
code can flip back is not a kill switch, and re-enabling must be a deliberate
human act that constructs a new session. Tests assert both the halt and the
absence of any reset method.

## 10. Runbook

**Enabling a paper session** (not currently possible — G1 is unsatisfied):

1. Confirm a `QUALIFIED_FOR_PAPER` decision exists for the candidate.
2. Store paper credentials in Keychain under `com.signal-foundry.alpaca-paper`.
3. Verify no credential variables are set in the environment.
4. Construct `BrokerSessionConfig(enabled=True, endpoint=..., keychain_service=...)`.
5. Call `connect()`, which runs the full authorization check.

**On a reconciliation divergence:** halt, preserve state, escalate to an
operator. Do not retry. Do not liquidate — an automatic liquidation acts on
exactly the state known to be wrong.

**On a suspected credential exposure:** engage the kill switch, rotate the
Keychain entry, and audit logs for the account digest.

**Rollback:** the package is additive and nothing imports it in a production
path. Reverting the commit removes it entirely; no state migration is involved.

## 11. Residual risks and limitations

- **Simulated fills bound operations, not performance.** §7.
- **The contract is shaped by one broker.** ADR 0015 selected Alpaca, and a
  contract written against a single implementation carries that implementation's
  assumptions. A second adapter is the only real test of neutrality.
- **No persistence yet.** Restart recovery and durable idempotency are #46; this
  MR holds order state in memory, so a process restart loses it. The
  content-derived order ID is the mechanism #46 will build on.
- **No rate limiting yet.** The configuration carries the bounds; the token
  bucket that enforces them belongs with the real HTTP client.
- **The environment-variable check is a snapshot.** It cannot prevent a
  credential being set after `connect()` returns.

## 12. Evidence

- `tests/test_broker_contracts.py` — 56 tests
- `tests/test_broker_paper_adapter.py` — 52 tests

Covering: naive-datetime refusal, float refusal, symbol and identifier
validation, deterministic ID derivation across all five components, the closed
transition table, impossible broker responses (overfill, `FILLED` with a
residual, missing average price, update-before-submission), crossed quotes,
duplicate positions, raw-account-identifier refusal, every live-endpoint form,
the not-allowlisted paper lookalike, plaintext HTTP, the AST assertion that no
override parameter exists, the AST assertion that no networking library is
imported, all six credential environment variables, the non-echoing refusal
message, the duck-typed decision stand-in, the REJECTED path, idempotent replay,
ask/bid fill sides, resting limit orders, closed-market behaviour, cash and
position reconciliation, position removal at flat, stale/future quotes, clock
skew, insufficient buying power, cancellation of a settled order, the one-way
kill switch, and determinism.
