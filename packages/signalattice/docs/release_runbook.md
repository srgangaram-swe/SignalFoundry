# Release runbook

Operator procedure for building, verifying, and publishing a Signalattice release. The design and
its residual risks are in [ADR 0007](adr/0007-reproducible-signed-release.md).

A release proves that the authorized process built and signed identified bytes. It is not
independent review, production readiness, or an authorization to trade.

## Entry points

| Command | What it does | Can it publish? |
| --- | --- | --- |
| `make release-dry-run` | Builds the full candidate and verifies it | No |
| `make release-verify` | Independently re-verifies a staged candidate | No |
| `make release-reproducible` | Builds twice and compares every byte | No |
| Release workflow, `publish` job | The only publication path | Yes, from `main`, with approval |

There is deliberately **no** `make release-publish`. Publication runs only from the protected
release workflow.

## Before you start

- [ ] The working tree is clean. The build refuses a dirty tree, because artifacts built from
      uncommitted changes are not reproducible from any commit.
- [ ] `web/node_modules` is installed (`npm ci` in `web/`), or the console is omitted and the
      descriptor says so.
- [ ] The version is what you intend. `make release-dry-run` prints it and refuses if
      `pyproject.toml`, `web/package.json`, and installed metadata disagree.

## Dry run

```bash
make release-dry-run          # builds into build/release
make release-verify           # re-hashes everything and checks provenance
make release-reproducible     # two clean builds, byte-compared
```

Expected output: a JSON summary with `"published": false`, a verification summary with
`"verified": true`, and a reproducibility report with `"reproducible": true`.

Exit codes are stable: `0` success, `2` a release gate refused, `3` the environment is unusable
(missing tool, unreadable tree). A `2` is a release problem; a `3` is a machine problem.

## Reading a refusal

| Message | Meaning | Action |
| --- | --- | --- |
| `the working tree is dirty` | Uncommitted changes | Commit or stash, then rebuild |
| `release version disagreement` | Two sources state different versions | Fix the named source |
| `release inventory does not verify` | Artifacts changed after the build | Rebuild; do not hand-edit the stage |
| `provenance does not bind source commit` | Statement is from a different build | Rebuild; never reuse an old statement |
| `not a publication attestation` | A dry-run attestation was offered for publication | Build under the publication path |
| `tag ... already exists` | The version was already published | Correct with a **new version** |

## Publication (SF-S5-SL-MR8)

Publication happens only after the `dev → prod → main` promotions, against the exact resulting
`main` merge commit.

1. Confirm `origin/main` is at the intended merge commit.
2. Run the release workflow via `workflow_dispatch` with `publish: true` and
   `release_commit: <exact main commit>`.
3. The `publish` job refuses anything but `main` and requires approval in the protected `release`
   environment. Nothing in the repository can approve on its own behalf.
4. After upload, verify the remote tag, release, artifacts, SBOM, and provenance. A partial upload
   is not a valid release and must not be reported as one.

## Immutability and correction

The publication correction in [ADR 0008](adr/0008-release-signing-and-remote-verification.md)
builds publication-identity metadata, signs the annotated tag with an ephemeral SSH key,
and attests the complete asset set using the approved GitHub workflow identity. Downloaded
assets and the full stage ZIP are compared to their source bytes before the draft becomes
public. To authenticate a downloaded signing key, verify its GitHub attestation with the
repository, `.github/workflows/release.yml`, `refs/heads/main`, and exact source digest
pinned. Then use that key in an allowed-signers file to run `git verify-tag`. A public key
downloaded without verified provenance is not a trust anchor.

The offline `scripts/release.py publication` command grants no publishing permission;
`publication-gate --commit <sha>` checks current main and promotion ancestry. These are
workflow plumbing, not an alternate publication path. Staging refuses an existing output
directory. Download verification includes the console, schemas, evidence and license in
`release-stage.zip`; the archive verifier never extracts untrusted members.

Published tags and releases are **immutable**. A defective release is never fixed by moving a tag,
replacing an asset, or deleting a release — every one of those silently invalidates verifications
that other people already performed.

To correct a defective release:

1. Stop further distribution and publish an advisory if warranted.
2. Branch from current `dev`, fix, and merge normally.
3. Repeat `dev → prod → main`.
4. Publish a **new version** and document the supersession.

## Incident: a bad release was published

1. Do not touch the existing tag.
2. Record what was wrong and which digests are affected.
3. Follow the correction procedure above.
4. Note the superseded version in the new release notes.

## What the evidence covers

`docs/benchmarks/release_dry_run_2026-08-20.json` and
`docs/benchmarks/release_reproducibility_2026-08-20.json` are produced by real dry runs;
`reports/figures/release_supply_chain.png` renders them. Six deliberate tamper cases are exercised
and all six are refused. Reproducibility is established on one platform only; the container image is
not yet a dry-run subject, and no signing material exists until #24.
