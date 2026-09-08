# ADR 0003: Explicit paper transport outside preserved research packages

Status: accepted for the additive paper-operations implementation in #13.

## Decision and scope

Implement Alpaca paper/data HTTPS in `signal_foundry.trading`, behind one bounded
worker and one shared CLI/HTTP service. Preserve both imported package trees and
their independent locks exactly. This adds an optional paper control panel to the
Nexus boundary in ADR 0002; its research-only default remains compatible.
AlphaForge ADRs 0015, 0016 and 0020 continue to govern broker selection,
qualification and absent live authority. The original simulated adapter remains
network independent. There is no live host, live enable flag or capital override.

Separate validated Decimal records, HTTPS normalization, immutable acquisition,
causal diagnostics, source qualification, durable journal, lifecycle policy and
prospective campaign accounting. The parent bounds an operation to 30 seconds and
kills descendants. Credentials are read only inside the explicit paper worker,
from the approved macOS Keychain service. Research workers and browser requests
cannot supply credentials, URLs, quantities or qualification verdicts.

## Invariants and failure domains

- Freeze the 3–5-symbol universe, data feed, split, candidate family, configuration
  and code identity before observations. A change requires a new state root.
- Recompute the actual preserved AlphaForge rubric, including all eight criteria
  and required content-addressed evidence. A supplied pass flag has no authority.
  Evidence hashes prove identity; honest measurement and independent scientific
  review remain human responsibilities.
- Persist intent and client order identity before submission. A lost response is
  uncertain until explicit broker reconciliation; a lookup 404 never resubmits it.
- Rebuild cash and quantities from cumulative fills. Unknown orders, external
  activity, unsupported corrections, fees or accounting breaks block admission.
- Persist a one-way stop independently of the ordinary operation lock. Recheck it
  at dispatch. A stop cannot recall bytes already sent; cancel owned orders
  explicitly, then reconcile. Cancellation never silently liquidates positions.
- Record calendar dates only after the official close and reconciliation. Missing
  sessions remain missing. Campaign reports never manufacture elapsed time or
  promote a diagnostic result to live readiness.

## Alternatives, costs and containment

An SDK would add another dependency and implicit retry/configuration surface.
Fixed-host standard-library HTTPS keeps this small REST contract inspectable.
Streaming and distributed schedulers are deferred: minute polling has explicit
request/time ceilings and refuses overdue work. Non-atomic account/order REST
snapshots may conservatively refuse a valid account; operators reconcile again
without retrying orders. This is an intentional availability tradeoff.

SQLite FULL synchronous transactions, append-only triggers, hashes and fsync/link
artifact publication provide local durability, not distributed consensus or
protection against the machine owner rewriting both content and hashes. State
and source must remain trusted and private. No remote/multi-user deployment is
supported. TLS validates the two fixed hosts; redirects, environment proxies,
compressed bodies, inherited secrets and automatic retries are absent.

Rollback disables the optional configuration and stops/cancels/reconciles the
paper account before reverting the root change. Retain the private journal and
artifacts for audit. Never erase ambiguous intent to make a new run appear clean.
The [runbook](../paper-operations.md) specifies limits and residual risks.

## Official interfaces

- [Paper environment](https://docs.alpaca.markets/us/docs/paper-trading)
- [Order lifecycle and DAY limit support](https://docs.alpaca.markets/us/docs/orders-at-alpaca)
- [Historical stock bars](https://docs.alpaca.markets/us/reference/stockbars)
- [Market data entitlements](https://docs.alpaca.markets/us/docs/market-data-faq)

Interface guidance checked 2026-09-08. Software fixture evidence is separate from
an actual credentialed broker acceptance run.
