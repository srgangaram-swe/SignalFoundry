# Assembly validation record

Scope: the exact prefixed source trees and new assembly adapters, not model
qualification or live-trading readiness. Local host: macOS, CPython 3.13.11,
Node 26.5.0; numerical package environments use their independent committed locks.

| Check | Observed local result |
| --- | --- |
| New root contracts and Git/transaction integrations | 54-test run passed; 97.33% combined statement/branch coverage, above 90% floor; two additional GitHub snapshot checks pass separately |
| Root quality | Black/Ruff clean; strict mypy clean across six modules |
| AlphaForge relocated full suite | 2,552 passed, one environment failure, eight skips; 84.56% coverage above 78% floor |
| AlphaForge failure disposition | Shell PATH omitted uv; all 51 audit tests passed after exposing the pinned executable, without source edits |
| AlphaForge native and execution checks | Native core built; 12 order-book tests and 193 execution tests passed; resolves the three native-dependent skips |
| AlphaForge source quality | Original pre-commit hooks, mypy across 287 files and all committed config/provenance checks passed |
| Signalattice relocated full suite | 1,994 passed, 13 console-dependency skips; 86.07% coverage above 80% floor |
| Signalattice skip disposition | After locked npm installation, all 13 console integration tests passed |
| Signalattice source quality | Ruff, Black and strict source typing across 107 files passed |
| Signalattice console | 172 component/unit tests; 91.82% branch coverage; types, lint, build, schema drift and resource budgets passed |
| Existing browser/accessibility matrix | 130 passed, two existing narrow-mobile skips; Chromium, Firefox, WebKit and narrow layout exercised |
| Both package distributions | Wheel/sdist builds passed; Signalattice archive allowlists passed |
| Preservation/recovery | Both self-contained source bundles restored; full Git integrity and exact refs verified; fixture clean-room clone verified |

The initial AlphaForge failure is recorded rather than hidden by a retry. It did
not expose a package relocation defect; required remote jobs install uv before
testing and build the native core before the full suite. The unchanged optional
dependency skips are not claimed exercised by that local numerical run; dedicated
remote optional-extra jobs remain required. Current PR check results are the
authority for remote status, not this local record.

The original source workflow pins, test commands, coverage floors and required-job
sets remain enforced by generated root workflows. Signalattice's previously
non-required service-image metadata reproducibility job remains visible and tied
to [source #67](https://github.com/srgangaram-swe/Signalattice/issues/67); no passing
container-reproducibility claim is made.

## Reproducible visual evidence

The [summary](evidence/assembly/summary.json) binds the exact assembly manifest.
The [artifact manifest](evidence/assembly/manifest.json) binds summary and PNG bytes.
The three-panel [Seaborn plot](evidence/assembly/preservation.png) was inspected
for readable labels, zero-based axes, source identities and honest scope. Before
and after counts are equal because independent identity checks precede plotting;
equal counts alone would not prove preservation.

```bash
uv run python -m foundry_build.evidence --output /new/empty/assembly-evidence
```

The parent must exist and the destination must not. Fixed inputs reproduce the
same summary and PNG bytes. Tests cover tampering, duplicate JSON keys, unsafe
files, byte/time limits, exact rights-resolution scope, missing ancestry, ref/object
corruption, stale/foreign package Git contexts and staged publication failures.

New namespace refs preserve every original tag object and advertised source tip;
they do not change original source repositories or add development branches. New
root artifacts contain metadata/hashes only—no licensed observations, secrets,
trained models, private bundles or local filesystem paths.
