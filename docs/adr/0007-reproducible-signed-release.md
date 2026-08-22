# ADR 0007: Build releases reproducibly and gate publication behind one authorized path

- **Status:** Accepted
- **Date:** 2026-08-20
- **Issue:** [#23](https://github.com/srgangaram-swe/Signalattice/issues/23)

## Context

Sprint 5 produced a durable registry, a bounded read-only service, delayed shadow evidence,
champion-challenger governance, and a local console. None of that is distributable yet: there is no
statement of what a release *is*, no way to prove two builds of one commit produced the same bytes,
and no gate that would stop a release being cut from the wrong branch.

The failure modes are specific and all hard to detect afterwards:

- artifacts built from a dirty tree that no commit reproduces;
- a wheel, a console bundle, and release notes that each state a different version, so nobody can
  later establish which bytes were released;
- a tag moved or reused to "fix" a release, which silently invalidates every verification anyone
  already performed against the old bytes;
- a dry-run artifact acquiring release identity;
- a workflow reachable from a pull request obtaining publication credentials;
- a signature read as though it were scientific validation.

## Decision

### One canonical version, checked everywhere

`pyproject.toml` `[project].version` is the release version. `quant_platform.__version__` is derived
from installed package metadata rather than re-typed, so the two cannot drift; the console's
`package.json` tracks it, and `assert_versions_agree` refuses when any source disagrees, naming
**every** offender so one run reports all of them.

The OpenAPI `info.version` is deliberately **not** forced to match. It names the wire contract,
which is versioned independently: a 0.x release can serve a stable v1 contract, and coupling them
would either freeze the contract or misstate the software's maturity. The descriptor records both,
and `assert_contract_pinned` checks the contract the console was built against.

**Sprint 5 releases as 0.3.0, not 1.0.** Reaching 1.0 is a compatibility promise, and release
machinery existing is not evidence of maturity. `claims_stable_api` exists so that choice stays
explicit, and a 1.x descriptor listing undocumented breaking changes is refused outright.

### Reproducible by construction, verified by rebuilding

Timestamps, hash ordering, locale, timezone, and ownership are pinned, and the build's reference
instant is the **commit date** rather than the wall clock — a reproducible build must not record
when it happened to run.

Two things needed explicit work:

- setuptools honours `SOURCE_DATE_EPOCH` for most sdist members but stamps the generated entries
  (`PKG-INFO` and every directory) with the wall clock. The sdist is repacked with member order
  sorted, mtimes pinned, and ownership normalised.
- That repack initially did nothing, because setuptools writes **PAX**-format archives whose
  extended headers carry their own sub-second `mtime` that takes precedence over the `TarInfo`
  field on write. Clearing `pax_headers` per member is what actually pins the timestamp.

`make release-reproducible` builds twice from scratch and compares every file. All 27 artifacts are
byte-identical, and the check names the offending paths when they are not.

### The inventory is closed, not a minimum

Every artifact is recorded by relative path, SHA-256, and byte size. Verification rebuilds the
inventory and re-hashes every file rather than trusting the recorded digests — a verifier that reads
the producer's numbers is checking arithmetic, not content.

A file that moved is reported as one missing plus one unexpected rather than matched by digest: a
wheel published under the wrong name is not the release the inventory describes. Missing, extra,
renamed, truncated, substituted, and single-byte-modified artifacts each have a test.

### Provenance binds the build, and dry runs cannot impersonate publications

An in-toto v1 statement carrying a SLSA v1 predicate names every subject by digest, the exact source
commit, the locked dependency digest, and the builder identity. Verification rejects a statement
that omits a built subject **or** attests to one that was not built.

`BUILDER_DRY_RUN` and `BUILDER_PUBLICATION` are different URIs. That is the mechanism preventing a
dry-run attestation from being presented as a publication one, and it has its own test.

The SBOM is CycloneDX 1.6 generated from the committed lockfiles rather than an installed
environment, so it describes what the release resolves to instead of what a build machine happened
to have. Its serial number is derived from the release identity, because a random UUID would make
two reproducible builds differ.

### Publication is a separate authority

`quant_platform.release` contains **no function that publishes**. `policy.py` answers whether
publication is permitted; performing it lives outside the package, which is what stops a build step
acquiring release authority by accident. `make release-publish` deliberately does not exist — a make
target would put publication one keystroke from any developer shell.

The gate refuses: a dry-run descriptor, any branch but `main`, a dirty tree, a commit that is not
both `HEAD` and current `origin/main`, a descriptor built from a different commit, a missing
`dev → prod → main` ancestry, an existing tag, and a free-form tag name. Each refusal names the
single condition that failed.

In CI, the `dry-run` job has `contents: read` and no environment, so pull-request code cannot reach
publication credentials whatever it executes. The `publish` job is `workflow_dispatch`-only, refuses
anything but `main`, and runs inside a protected environment where the human approval lives.

## Consequences

**Accepted.** A release is now a checkable claim: one version, one commit, a closed inventory,
provenance that binds them, and a rebuild that reproduces the bytes. Six deliberate tamper cases —
a single bit flip, a removed artifact, an extra artifact, a rename, a substituted source commit, and
a dry-run attestation relabelled as a publication — are all refused, and that evidence is committed
rather than asserted.

**Costs.** Reproducibility constrains the build: no wall-clock timestamps, no random identifiers, no
unpinned ordering. The sdist repack is extra machinery that exists only because setuptools and PAX
interact badly, and it needs revisiting if either changes. Two full builds per reproducibility check
roughly doubles that job's cost.

**Not delivered here.** This MR builds and proves the machinery; it publishes nothing. No tag and no
GitHub Release exist as a result of it. Container/OCI subjects and cryptographic signing are wired
into the descriptor's vocabulary but are not exercised by the dry run — the signing identity and
credential custody belong to the publication path in #24.

## Residual risk

- **A signature proves process, not merit.** It establishes that the authorized release process
  signed identified bytes under the documented trust policy. It is not independent review, economic
  validity, production readiness, or profitability. The descriptor carries that sentence as a
  required field so it travels with the release.
- **Reproducibility is proven on one platform.** Two builds on this machine agree; cross-platform
  byte-identity is not established, and the container image is not yet a dry-run subject.
- **The tamper cases are the ones we thought of.** They cover content, membership, naming, and
  attestation identity, but an attacker who can rewrite the inventory and the provenance together
  produces a self-consistent stage. External anchoring is what would close that, and it is not here.
- **No signing material exists yet.** The trust policy is documented and the gate is built, but the
  signer, its custody, and revocation are #24's responsibility.
- **The dirty-tree refusal depends on `git status`.** A build environment that lies about its
  working tree is outside this boundary.

## Rollback and containment

Before publication: revert the additive workflow and entry points on a normal work branch. The
release package is inert without them, and audit evidence of any failure is retained.

After publication: never move, delete, reuse, or replace a tag or its assets. Stop distribution,
publish an advisory where warranted, correct on a new branch from current `dev`, repeat
`dev → prod → main`, and publish a **new version**.

## Non-goals

Publishing the Sprint 5 release, creating a tag or GitHub Release, changing application behaviour,
adding models or data, remote exposure, paper or live trading, capital authorization, proving
profitability, treating signatures as scientific validation, mutable tags, or an arbitrary 1.0.
