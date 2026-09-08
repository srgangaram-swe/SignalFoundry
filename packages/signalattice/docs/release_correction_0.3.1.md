# 0.3.1: license recognition and release identity correction

Issue: [#24](https://github.com/srgangaram-swe/Signalattice/issues/24).

The signed [0.3.0 release](https://github.com/srgangaram-swe/Signalattice/releases/tag/v0.3.0)
passed artifact, signature, attestation, provenance, SBOM and clean-installation
verification. GitHub's license API nevertheless returned `NOASSERTION`, not the
intended MIT identifier. `LICENSE` contained canonical MIT text followed by an
educational/trading disclaimer. That unmet acceptance criterion prevents calling
the release closeout complete.

Keep the MIT grant, copyright and warranty text unchanged in `LICENSE`. Move the
existing disclaimer verbatim into `DISCLAIMER.md`; include both in the wheel,
source distribution and complete release stage. The README links both notices.
This is a file-layout correction, not a change to the intended license or a
removal of the trading warning. The canonical template is published by
[GitHub's Choose a License](https://choosealicense.com/licenses/mit/).

The correction uses a new patch version, 0.3.1. Never move the v0.3.0 tag or
replace its assets. The Python manifest, installed metadata, console manifest,
and both npm lockfile root identities must agree. The previous npm lockfile
roots still stated 0.1.0; new regression tests reject either root drifting,
malformed root metadata, and non-object console manifests. Dependency component
versions remain independent of the application version. No dependency is
upgraded by this patch.

Changing the package and lock metadata changes the service evidence's bound
source hash. A new dated patch-1 synthetic reference reruns the same bounded
workload; both prior references remain intact. This is not a controlled speedup
comparison. The service and sprint Seaborn figures are regenerated and inspected,
with the original INVALID/INSUFFICIENT outcomes retained.

Validation covers notice preservation, closed archive membership, both npm root
identities, malformed JSON, version agreement, current evidence hashes, wheel
and source installation, and deterministic builds. GitHub license recognition
must be re-queried after the correction reaches main; a local assertion alone
does not satisfy that external metadata requirement.

Use the same protected work-branch review, forward promotions, human-approved
release environment and independent downloaded-artifact verification. Existing
release process trust and the deferred image-metadata reproducibility issue #67
are unchanged. Signatures do not establish production readiness or profitability.
