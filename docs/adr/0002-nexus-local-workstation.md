# ADR 0002: Nexus is a bounded client, not a second research engine

Status: accepted for the additive AlphaForge #81 workstation delivery.

## Decision

Build `apps/nexus` with React, strict TypeScript and Vite. Serve its reviewed,
prebuilt assets from the existing loopback service under `/nexus`, opt-in through
`signal-foundry serve --nexus`. Keep the API-only launch compatible. There is no
development proxy, CORS exception, remote bind, broker adapter or credential form.
Both imported source trees and their dashboards remain unchanged.

Generate browser validators at build time from the same OpenAPI document that
generates TypeScript types. Ajv standalone code avoids runtime schema compilation
and `eval`; CSP never needs `unsafe-eval` or `unsafe-inline`. Backend validation
remains authoritative for registry/model-specific semantics. The catalog supplies
the actual default request. UI transformations format evidence, not fit models or
reimplement portfolio mathematics.

Separate transport, workflow transitions, form parsing, evidence projections and
presentation. An edit invalidates preflight. Submission uses a stable idempotency
key; an ambiguous network outcome offers an explicit same-request retry instead
of creating a fresh job. Bounded GET polling stops on terminal state, timeout or
unmount. Cancelling a fetch is not presented as cancelling a research job.

Render missing, partial, unavailable and negative evidence explicitly. A completed
simulation never upgrades `NOT_READY`. Server-verified hashes are labelled as
such: JavaScript serialization is not Python's canonical float/string encoding,
so the client does not pretend to independently reproduce those hashes.

## Security and resource contract

The server admits a bounded, hash-verified build manifest and holds assets as an
immutable in-memory snapshot. Exact routes, fixed MIME types, no symlinks, no
source maps, no arbitrary files and no wildcard SPA fallback limit filesystem
exposure. The browser uses same-origin requests only, bounded concurrency, byte
and chunk ceilings, deadlines, abort cleanup and schema checks before rendering.
Evidence is text, never HTML. No analytics, remote fonts or external assets load.

Tables virtualize retained rows with bounded DOM work and accessible row indices;
projection cost is linear in the already-bounded response. Charts retain missing
values, zero/reference lines, units, sampling limits and non-color encodings.
Keyboard navigation, visible focus, reduced motion, responsive layout and both
contrast-tested themes are acceptance requirements, not optional decoration.

Correctness/security gates remain independent of measured latency. Reference
budgets cover cold load, interaction, JS/CSS/total bytes, CPU, process memory and
2,048-row tables. Measurements retain samples, environment and limitations. A
local browser benchmark does not measure exchange latency or market capacity.

## Alternatives and consequences

Extending only Streamlit would preserve a smaller toolchain but would not give
the new typed multi-view client a coherent cancellation, accessibility and dense
evidence-navigation boundary. Replacing both legacy dashboards immediately would
hide capabilities not yet projected through API v1. The explicit parity matrix
therefore distinguishes new Nexus projections from retained, tested legacy paths.
In particular, proper-score calibration is not invented for regression artifacts.

The frontend is another locked dependency graph and build to qualify. Original
dashboards remain separately launched local applications, not embedded frames.
Remote or multi-user deployment requires a separate authentication, authorization,
TLS, data-entitlement and operations ADR. Local same-user processes are trusted;
this interface is neither an adversarial-code sandbox nor live-trading software.

Rollback removes the opt-in mount and additive app while preserving all private
research stores, source entry points and histories.

## References

- [Ajv standalone validation](https://ajv.js.org/standalone.html)
- [Vite guide](https://vite.dev/guide/)
- [WCAG 2.2 quick reference](https://www.w3.org/WAI/WCAG22/quickref/)
- [Control-plane ADR](0001-isolated-research-control-plane.md)
