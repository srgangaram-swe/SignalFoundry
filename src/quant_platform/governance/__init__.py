"""Champion-challenger comparison, inference, and promotion authority.

SF-S5-SL-MR5. Three modules, layered so each depends only on the one below it:

- :mod:`~quant_platform.governance.comparison` builds an exactly paired cohort
  and reports every exclusion rather than dropping it;
- :mod:`~quant_platform.governance.inference` tests that cohort with
  dependence-aware resampling and a familywise correction;
- :mod:`~quant_platform.governance.gates` turns those results into a
  recommendation, and refuses to apply one without named human approval.

Nothing in this package can approve, apply, roll back, or unfreeze a promotion.
A recommendation is evidence that promotion is permissible, never that it has
happened, and a test parses these modules' ASTs to prove no override parameter
exists. See ``docs/adr/0005-champion-challenger-promotion-governance.md``.

Submodules are imported explicitly rather than re-exported here, so importing
the package does not drag the inference stack into callers that only need the
comparison contracts.
"""
