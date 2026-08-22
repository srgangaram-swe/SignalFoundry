# ADR 0006: Ship a local-only read-only forecast-observability console

- **Status:** Accepted
- **Date:** 2026-08-20
- **Issue:** [#19](https://github.com/srgangaram-swe/Signalattice/issues/19)

## Context

ADR 0003 added a versioned read-only HTTP projection over the registry, ADR 0004 bounded its
operating envelope, and ADR 0005 added champion-challenger governance with an append-only lane
history. The evidence exists and is well specified. It is also, at this point, only readable as
JSON by someone who already knows which endpoint to ask.

That is a real gap for the thing this repository is for. Lineage, calibration, uncertainty, drift,
evidence sufficiency, and governance state are exactly the properties a reviewer needs to inspect,
and a reviewer who has to hand-assemble them from `curl` output will not inspect them carefully.

The risk of adding a browser is equally real and specific:

- a console becomes a second source of truth the moment it computes anything the service did not;
- a chart makes a weak result look like a strong one, because a rendered number carries more
  authority than a JSON field;
- "no data" and "we could not tell" collapse into the same empty panel unless they are deliberately
  kept apart;
- a browser is an execution environment, and a read-only service that grows a UI has grown a script
  host, a storage surface, and an outbound-request capability it did not have before;
- a UI is the natural place for someone to eventually add a button that approves a promotion.

## Decision

### The console is a projection, not an application

`web/` is a React single-page application served by the existing local service under `/console`. It
reads the version-1 API and renders it. It computes no metric the service did not send; where a
view needs a judgement (is this cohort large enough to support a calibration claim?), the rule is
stated in the view, applied to validated server fields, and shown to the reader alongside the
numbers.

There are exactly **seven** views: system overview, run catalog, run evidence, model comparison,
calibration and uncertainty, drift/latency/operations, and governance and readiness. The route
allowlist lives in the Python boundary as well as the front-end router, so an eighth view would
have to be added in two places, visibly. There is no administration, mutation, artifact-download,
SQL, broker, order, position, P&L, or capital-control route.

### The honest-state model

Every route and every reusable evidence panel resolves to exactly one of nine states: `LOADING`,
`READY`, `EMPTY`, `PARTIAL`, `INSUFFICIENT_EVIDENCE`, `INVALID`, `STALE`, `UNAVAILABLE`, `ERROR`.

Keeping these apart is the central design decision. `EMPTY` means the server answered and there is
nothing recorded. `INSUFFICIENT_EVIDENCE` means there is data but not enough of it to support the
claim the panel exists to make — a finding, not an absence. `INVALID` means the evidence
contradicts itself or failed verification, and is never a smaller version of insufficiency.
Collapsing any pair of these would let a reader conclude "nothing is wrong" from "we could not
tell", which is the specific mistake the whole console exists to avoid.

State is derived from validated server fields only: never from an HTTP status alone, never from a
missing value, never from a colour, and never from the absence of a reported failure.

**A state that carries data renders both the banner and the data.** An `INSUFFICIENT_EVIDENCE`
panel that showed nothing would be indistinguishable from an empty one, and a governance lane whose
chain failed verification is exactly the lane that must not disappear from the table.

### Generated types and runtime decoding are separate jobs

`openapi-typescript` generates compile-time bindings from the committed contract, and a drift check
regenerates and compares them so the types cannot silently diverge from the document.

Those types are erased at runtime and prove nothing about the bytes that arrive, so every response
is independently decoded with hand-written Zod schemas before any semantic use. The schemas are
written by hand rather than derived from the same document on purpose: a decoder generated from the
contract agrees with the contract even when the server does not.

Everything fails closed. Unknown fields, non-finite numbers, invalid timestamps, malformed digests,
oversized collections, wrong schema versions, and unrecognised enum members are rejected rather
than coerced. The decode failure is reported as incompatible evidence, and the reason names the
failing field path without echoing the received value — echoing an unvalidated server string into
the DOM is what the decoder exists to prevent.

### The transport can only do one thing

`src/api/client.ts` is the console's only network capability, and every property is a refusal:

- **GET only.** There is no method parameter, so a mutation is unrepresentable.
- **Same-origin only.** Paths are checked against an allowlist; a scheme, an authority, a
  protocol-relative prefix, or a fragment is refused. Path and query are validated separately
  because they admit different characters.
- **No credentials.** The local authority boundary is the loopback socket, not an ambient cookie.
- **Bounded.** At most four requests in flight, each with a hard ten-second abort, and a response
  ceiling enforced both from the declared length and after reading (a chunked response declares no
  length).
- **No automatic retry, and no polling.** A failed read stays failed until a person asks again.
  Retrying a saturated local service automatically is how a read-only console becomes the load that
  keeps it saturated.

### The delivery boundary is not a static file server

`quant_platform.service.console` enumerates the built bundle once at load and answers from that
manifest. A generic static handler resolves whatever path it is given, guesses content types from
the filesystem, and follows symlinks; each of those is a way for a file outside the build output to
reach a browser.

Consequently: symlinks are refused at load rather than followed at request time; content types come
from a fixed map, so an unexpected extension is a build failure rather than a guess; and route
fallback is an allowlist of the seven views rather than a wildcard, so the console cannot answer
200 for arbitrary paths.

The console document gets its own content-security policy — `script-src 'self'`, `style-src 'self'`,
`connect-src 'self'`, everything else `'none'`, no inline, no eval, no worker, no frame, no remote
image, no third-party origin. API responses keep the stricter `default-src 'none'`. `HEAD` is
admitted for console assets only; the JSON API stays GET-only.

### Accessibility is a correctness property here

Status is carried by a word, a glyph, **and** a colour, so no single channel is load-bearing. Every
visual encoding has a tabular equivalent in the same row: an interval bar prints the numbers it
draws, and a one-sided interval is labelled "unbounded" rather than closed at an arbitrary value.
There is no canvas-only evidence.

The palette is Okabe-Ito darkened for light backgrounds. The published values are chosen for hue
separation under colour vision deficiency, not for luminance: `#009e73` reaches only 3.4:1 and
`#d55e00` only 3.9:1 against white, both below the 4.5:1 that WCAG 2.2 AA requires. A unit test
computes every ratio so a regression fails there rather than in a browser audit.

## Consequences

**Accepted.** A reviewer can now see lineage, gate outcomes, uncertainty, exclusions, and lane state
without reconstructing them by hand, and can see them *with their limitations attached*. The
console is cheap to open — 94.7 KiB gzip of initial JavaScript against a 250 KiB budget — and
cheap to remove, because it is one additive module plus a static directory.

**Costs.** A second toolchain enters the repository. Node is a build and test boundary only: the
hardened service container still ships without Node, without a browser, and without the governance
package. The console adds a surface that must be kept in step with the API, which is why contract
drift is a gate rather than a convention.

**Deviation from the issue's pinned toolchain.** The issue specifies TypeScript 7. At the time of
writing no published version of the required tooling supports it: `openapi-typescript@7.13.0` peers
on `^5.x` and `typescript-eslint@8.67.0` on `>=4.8.4 <6.1.0`. Forcing TypeScript 7 with
`--legacy-peer-deps` would leave the type-aware lint rules running against an unsupported compiler,
which is a weakened gate rather than a satisfied requirement. TypeScript **5.9.3** — the newest
version the whole required toolchain supports — is pinned instead, with every strict compiler flag
enabled (`exactOptionalPropertyTypes`, `noUncheckedIndexedAccess`,
`noPropertyAccessFromIndexSignature`, and the rest). Revisiting this is follow-up work, not a
silent reinterpretation.

**Contract change.** The published OpenAPI document referenced the OpenAPI meta-schema by URL, so a
local-only offline contract could not be interpreted without a network fetch, and type generation
failed on it outright. The document now describes its own response locally and resolves entirely
offline; conformance to the meta-schema is asserted by the contract test suite instead.

## Residual risk

- **The console is only as honest as the fields it is given.** It renders `chain_verified` and
  `satisfied`; it does not re-verify a hash chain or re-run a gate in the browser. A service that
  reported a false `true` would be believed.
- **Local-only enforcement is the service's, not the console's.** The browser refuses to leave its
  origin, but the guarantee that the origin is loopback comes from the service's `Host` validation.
- **The freshness window is a stated contract, not a measurement.** Evidence older than 24 hours is
  labelled `STALE`; that threshold is a convention, and nothing detects evidence that is fresh but
  wrong.
- **Two browser tests are skipped on WebKit** because macOS Safari omits buttons and links from the
  Tab order unless Full Keyboard Access is enabled and Playwright cannot set it. The same property
  is asserted on every engine by reading the DOM focus order directly.
- **Panel 4 of the evidence figure is a static property of the sources** — which states each view
  can report — not a claim about which states were exercised.

## Rollback and containment

Pass `console=None` to `create_app` and the delivery boundary is not mounted; delete `web/` and the
front-end is gone. The read-only API, the UDS/headless container, the registry and CAS evidence, the
OpenAPI contract, and every prior service behaviour are untouched.

Rollback must not take the form of expanding CORS, exposing a remote listener, relaxing the CSP,
weakening validation or accessibility, or adding a browser mutation path.

## Non-goals

Server or API mutation, approval or governance authority, raw-row browsing, artifact download,
model execution, training, scheduling, provider acquisition, remote or public hosting,
authentication as a substitute for local-only enforcement, multi-user tenancy, analytics, broker
integration, orders, positions, P&L, paper or live trading, capital authorization, automatic
promotion, or any claim of profitability or readiness.
