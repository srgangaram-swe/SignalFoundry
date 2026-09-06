# Signal Foundry Sprint 5 — evidence dossier

**Local ForecastOps, Shadow Evidence & Governance.** Release `0.3.0`.

This document is the public record of what Sprint 5 built, what its evidence establishes, and —
more importantly — what it does not. Every claim below resolves to a committed artifact whose
digest is recorded in [`docs/benchmarks/sprint5_evidence_index.json`](benchmarks/sprint5_evidence_index.json)
and re-checkable with `make sprint5-dossier-check`.

## What this sprint does not establish

Stated first, because it is the part most easily lost:

- **No prospective wall-clock evidence exists.** Every shadow result is deterministic replay. The
  promotion floor requires 28 consecutive calendar days of real elapsed evidence, and Sprint 5 does
  not meet it. Prospective validation is tracked in [#63](https://github.com/srgangaram-swe/Signalattice/issues/63).
- **No profitability, readiness, or capacity claim is made or supported.** Artifact digests
  establish content identity. A signature establishes that the authorized process signed identified
  bytes. Neither is independent review, economic validity, or production readiness.
- **Nothing here authorizes deployment, capital, paper trading, or live trading.** The governance
  layer can recommend; only a separate local human action can approve, and no such action has been
  taken.
- **The underlying data is stale and not point-in-time.** Public-domain daily US equity prices
  through 2018-03-27, vulnerable to survivorship and selection bias, with corporate-action and
  universe-membership completeness explicitly recorded as `false`.

## What Sprint 5 shipped

| Issue | Capability | ADR |
| --- | --- | --- |
| [#18](https://github.com/srgangaram-swe/Signalattice/issues/18) | Durable append-only run and artifact registry with content-addressed storage | [0002](adr/0002-durable-local-registry.md) |
| [#17](https://github.com/srgangaram-swe/Signalattice/issues/17) | Versioned local read-only evidence API | [0003](adr/0003-local-read-only-evidence-api.md) |
| [#20](https://github.com/srgangaram-swe/Signalattice/issues/20) | Bounded, hardened service operability | [0004](adr/0004-bounded-service-operability.md) |
| [#21](https://github.com/srgangaram-swe/Signalattice/issues/21) | Append-only delayed shadow forecast evidence | — |
| [#22](https://github.com/srgangaram-swe/Signalattice/issues/22) | Champion-challenger governance with human-only approval | [0005](adr/0005-champion-challenger-promotion-governance.md) |
| [#19](https://github.com/srgangaram-swe/Signalattice/issues/19) | Local-only read-only forecast-observability console | [0006](adr/0006-local-forecast-observability-console.md) |
| [#23](https://github.com/srgangaram-swe/Signalattice/issues/23) | Reproducible release machinery, proven by a non-publishing dry run | [0007](adr/0007-reproducible-signed-release.md) |

## Architecture and trust boundaries

Licensed point-in-time data enters Signalattice with immutable raw provenance, is promoted through
a checksum-verified migration chain into a durable registry and content-addressed store, and is
projected outward through progressively narrower boundaries:

```
 raw vintages ──► registry + CAS ──► read-only API ──► console (browser)
                        │                  │
                        └──► governance ───┘
                             (lanes, gates, approval)
```

Each arrow is a narrowing. Nothing flows back:

- **Registry → API.** The API projects aggregates. It never returns raw rows, host paths, or
  authority-bearing state, and every route is `GET`.
- **API → console.** The browser can only issue same-origin `GET` requests with no credentials. Its
  transport has no method parameter, so a mutation is unrepresentable.
- **Governance → everything.** Automation may compute, recommend, and freeze. It may never approve,
  apply, roll back, or unfreeze. A test proves the absence of any override parameter by parsing the
  module ASTs.
- **Release → public.** `quant_platform.release` contains no function that publishes; publication is
  a separately authorized operation behind a protected environment on `main`.

## Evidence, by claim

Full records — source path, digest, size, collection method, environment, sample context,
limitations — are in the [machine-readable index](benchmarks/sprint5_evidence_index.json). Summary:

| Claim | Class | Artifact |
| --- | --- | --- |
| Service holds its admission, latency, and response bounds | local engineering | `service_operability_2026-09-06_patch1.json` |
| Console meets budgets, renders every state, passes accessibility | local engineering | `console_evidence_2026-08-20.json` |
| Release dry run verifies and refuses six tamper cases | local engineering | `release_dry_run_2026-08-20.json` |
| Two clean builds are byte-identical | local engineering | `release_reproducibility_2026-08-20.json` |
| Block bootstrap holds nominal alpha under dependence | deterministic synthetic | `governance_promotion_evidence.json` |
| Dataset contract round-trips a verified panel | historical replay | `signal_foundry_contract_1_1_2026-07-25.json` |
| Cache replay reproduces a panel with zero provider requests | historical replay | `nasdaq_cache_replay_2026-07-24.json` |

Evidence classes are never relabelled upward. The index refuses to build if any entry claims
`prospective_wallclock`, because Sprint 5 produced none.

## Unfavourable and insufficient outcomes

A sprint record that reports only successes has stopped recording. These are carried in the gate
matrix at [`reports/figures/sprint5_evidence.png`](../reports/figures/sprint5_evidence.png):

| Outcome | Gate | Why |
| --- | --- | --- |
| **INVALID** | Reproducible image config digest | Clean container builds produce identical filesystem inventories but differing image *config* digests. Tracked as [#67](https://github.com/srgangaram-swe/Signalattice/issues/67); the job is not a required check. |
| **INSUFFICIENT** | Wall-clock promotion evidence | The floor requires 28 consecutive days; Sprint 5 has deterministic replay only. This is a finding about the evidence, not about any model. |
| **UNAVAILABLE** | Cryptographic signing exercised | The trust policy and publication gate are built, but no signing material exists yet. |

## Statistical evidence and uncertainty

The governance layer's inference is deliberately conservative, because the failure it guards
against is approving a challenger that is not actually better:

- Resampling is a **circular moving-block bootstrap over whole dates**, not over individual
  forecasts. Same-day forecasts share market conditions, and treating them as independent is what
  turns noise into significance. Under a true null the block estimator tracks alpha (0.04–0.07)
  across dependence levels while a naive per-forecast bootstrap reaches **0.49**.
- Block length is derived as `max(horizon, ceil(n^(1/3)))` and cannot be supplied, because a caller
  who could choose it could choose the narrowest interval.
- Too few blocks returns `UNDERPOWERED` with **no p-value at all**, so a non-significant result
  cannot be read as evidence of equivalence.
- Multiplicity is corrected with Holm-Bonferroni across the complete family, chosen because it
  holds under arbitrary dependence and the metrics are strongly correlated.
- Gates are absolute. There is no weighted score and no override parameter anywhere in the package.

Exclusions are reported rather than dropped: a paired cohort carries `champion_only`,
`challenger_only`, and per-arm unscored counts, and a cohort whose per-arm missingness differs by
more than 0.05 is reported **incomparable** rather than scored.

## Service, console, and operational limits

| Surface | Limit | Observed |
| --- | --- | --- |
| Service response ceiling | 2 MiB | 79 KiB |
| Service metrics exposition | 256 KiB | 26 KiB |
| Service metric series | 600 | 296 |
| Console initial JavaScript | 250 KiB gzip | ~95 KiB |
| Console initial CSS | 50 KiB gzip | ~1.6 KiB |
| Console total assets | 1 MiB | ~335 KiB |

The service is loopback-only by `Host` validation, refuses proxy headers, and is `GET`-only. The
console adds a static delivery boundary with its own strict content-security policy and admits
`HEAD` for assets alone.

## Accessibility and frontend evidence

WCAG 2.2 AA across seven views on Chromium, Firefox, WebKit, and a 320-pixel viewport: 130 browser
tests passing, 2 skipped with recorded reasons, and axe reporting no serious or critical violation
on any route or on a failure state. Status is conveyed by word, glyph, and colour so no single
channel is required, and every chart has a tabular equivalent.

Two accessibility defects were found and fixed during the work: the palette used published
Okabe-Ito values that fall below 4.5:1 on white, and scrolling table containers were not keyboard
focusable.

## Supply chain and security

- Reproducible builds: 27 of 27 artifacts byte-identical across two clean builds of one commit.
- CycloneDX 1.6 SBOM covering 439 Python and Node components, generated from committed lockfiles.
- in-toto v1 / SLSA v1 provenance binding every subject, the source commit, the locked dependency
  digest, and a builder identity that distinguishes dry-run from publication.
- Six deliberate tamper cases — single-bit flip, removed artifact, extra artifact, rename,
  substituted source commit, and a dry-run attestation relabelled as a publication — all refused.
- Every GitHub Action pinned by full commit SHA; the release dry-run job holds `contents: read` with
  no environment, so pull-request code cannot reach publication credentials.

Residual risks are enumerated in each ADR. The most important: local hash chains detect accidental
divergence but are **not** externally tamper-proof, and the governance approver field is an owner
assertion rather than cryptographic identity or separation of duties.

## Reproducing this dossier

Network-independent, from committed fixtures and redistribution-safe aggregates only:

```bash
make sprint5-dossier          # regenerate the evidence index and figure
make sprint5-dossier-check    # recompute every digest and refuse on drift
make ci                       # the full unchanged validation suite
```

No provider request is permitted to regenerate this dossier.

## Compatibility, migration, and rollback

`0.3.0` — **not** 1.0. Reaching 1.0 is a compatibility promise, and release machinery existing is
not evidence of maturity.

- **Registry schema 3**, forward-only. Migrations apply on open; a database at a newer version is
  refused rather than down-migrated.
- **API contract v1**, versioned independently of the software so a 0.x release can serve a stable
  wire contract.
- **Supported**: Python 3.12–3.14; the console builds on Node 24 and is a build-time boundary only.
- **Rollback**: the console is removed by passing `console=None`; governance reads by omitting the
  port; the release machinery by reverting its workflow. Published tags are immutable — a defective
  release is corrected with a new version, never by moving a tag.

Full procedures: [release runbook](release_runbook.md), [ADR 0007](adr/0007-reproducible-signed-release.md).

## What comes next

[#63](https://github.com/srgangaram-swe/Signalattice/issues/63) carries the prospective wall-clock
campaign — the evidence Sprint 5 cannot produce by construction. Deferred engineering work is
tracked in [#67](https://github.com/srgangaram-swe/Signalattice/issues/67),
[#69](https://github.com/srgangaram-swe/Signalattice/issues/69),
[#71](https://github.com/srgangaram-swe/Signalattice/issues/71), and
[#73](https://github.com/srgangaram-swe/Signalattice/issues/73), each re-scoped in writing out of
Sprint 5 rather than silently carried.
