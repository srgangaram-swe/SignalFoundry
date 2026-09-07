"""Reproducible, signed release machinery.

SF-S5-SL-MR7. Four modules, layered so each depends only on the one below it:

- :mod:`~quant_platform.release.identity` resolves the one canonical version and
  refuses when any source disagrees;
- :mod:`~quant_platform.release.inventory` records exactly which bytes were
  built and detects any change to them;
- :mod:`~quant_platform.release.provenance` binds those bytes to a commit, a
  builder, and a locked dependency set, and emits the SBOM;
- :mod:`~quant_platform.release.policy` decides whether publication is
  permitted.

**Nothing in this package publishes.** There is no function that creates a tag,
uploads an asset, or contacts a registry. The policy module answers whether
publication is permitted; performing it is a separately authorized operation
outside this package, which is what keeps a build step from acquiring release
authority by accident.

A signature here establishes that the authorized process signed identified
bytes under the documented trust policy. It does not establish independent
review, economic validity, production readiness, or profitability.
"""
