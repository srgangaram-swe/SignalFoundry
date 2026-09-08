# Security boundary

This is a local research repository, not a brokerage or production trading system.
It has no authority to place a live order. Do not put provider or broker credentials
in browser forms, committed configuration, URLs or logs. Use the operating-system
credential store through a bounded, explicitly authorized process when needed.

Assembly tooling accepts reviewed local mirrors and source-generated preservation
ledgers. Hashes establish integrity, not the trustworthiness or ownership of code.
The complete source scanner and pinned security gates remain necessary. Root Git
commands have deadlines and combined stdout/stderr ceilings, disable hooks and
replacement/lazy-fetch behavior, and do not inherit credential environment variables.
Input JSON rejects nonregular files, symlinks, oversized payloads, duplicate keys
and nonstandard numeric constants. Context publication requires a trusted local
parent directory and cooperating writers; it is not an adversarial multi-user
filesystem sandbox.

The owner's determination resolves only the four exact historical missing-license
findings named in the current provenance record. Other license, secret, raw-data,
symlink and object-corruption findings still fail closed. Historical records are
not falsified or removed.

Root CI is read-only, uses full-SHA Action pins and separate package environments.
Source credentialed publication workflows remain inert below package prefixes.
Dependency audit results are time-dependent; passing audits do not prove that all
code, Actions, native libraries or future resolutions are safe.

## Local control plane

The [control-plane boundary](docs/control-plane.md#security-boundary-and-remote-exposure)
and [ADR 0001](docs/adr/0001-isolated-research-control-plane.md) define its authority.
Browser requests cannot select executables, imports, provider URLs, filesystem
paths, credentials or broker actions. Fixed worker commands run without inherited
secrets, with bounded output, CPU/wall time, threads, descriptors and sampled RSS.
The process boundary contains a native crash; it is not an untrusted-code sandbox.

| Threat | Enforced boundary | Residual limit |
| --- | --- | --- |
| Cross-site access to a local research service | Exact Host/Origin, fetch-site checks, explicit JSON client header, no CORS | Other local processes are not authenticated |
| Malformed or excessive work | Strict finite contracts, bounded bodies/depth, worker/HTTP/queue/retention ceilings | Resource sampling can overshoot |
| Path or command injection | Operator-selected non-symlink bundle root, content IDs, fixed executable vectors, no shell | The trusted owner can replace local source/executables |
| Duplicate, cancelled or interrupted publication | Idempotency, serialized transitions, transactional SQLite artifact/hash/audit, restart failure | Local durability is not distributed consensus |
| Accidental disclosure | Aggregate-only evidence, bounded projections, no secret fields, sanitized errors and source logs | Dataset IDs and instrument metadata are visible to the local operator |

The supported launcher binds only 127.0.0.1, with one process and proxy-header trust
disabled. Do not expose it through forwarding or a reverse proxy. Remote access
needs a new reviewed authentication, authorization, TLS and deployment design;
there is no supported insecure opt-out switch. A completed job never authorizes
paper routing, live orders or a larger risk budget.

Report a suspected vulnerability privately using GitHub's private vulnerability
reporting when enabled. Never put exploit credentials, sensitive data or proprietary
material in a public issue. No security reporting channel authorizes access to
accounts, external services or live funds.
