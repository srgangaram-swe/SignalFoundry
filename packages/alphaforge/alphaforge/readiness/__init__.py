"""Live-readiness framework and inert-by-default capital configuration (SF-S5-MR10).

- :mod:`~alphaforge.readiness.checklist` — a versioned, content-identified
  checklist with no score and no override. Every item is required; an item with
  no evidence is unmet, because readiness is demonstrated rather than inherited
  by omission.
- :mod:`~alphaforge.readiness.capital` — a configuration that is inert unless
  activated against a READY decision and a current, cap-bound authorization, and
  that has **no method capable of raising a cap**.

Nothing here places an order. A `READY_FOR_MINIMAL_CAPITAL` verdict authorizes a
separately configured, hard-capped, owner-approved deployment and nothing else.
"""

from alphaforge.readiness.capital import (
    ABSOLUTE_MAX_CAPITAL,
    MAX_AUTHORIZATION_DAYS,
    CapitalAuthorization,
    CapitalError,
    LiveCapitalConfig,
    NotAuthorizedError,
    activate,
    assert_within_limits,
)
from alphaforge.readiness.checklist import (
    ATTESTATION_ONLY,
    DEFAULT_ATTESTATION_VALIDITY,
    Attestation,
    Category,
    ChecklistItem,
    ItemResult,
    ReadinessChecklist,
    ReadinessDecision,
    ReadinessError,
    Verdict,
    evaluate_readiness,
    minimal_capital_checklist,
    render_readiness_report,
)

__all__ = [
    "ABSOLUTE_MAX_CAPITAL",
    "ATTESTATION_ONLY",
    "DEFAULT_ATTESTATION_VALIDITY",
    "MAX_AUTHORIZATION_DAYS",
    "Attestation",
    "CapitalAuthorization",
    "CapitalError",
    "Category",
    "ChecklistItem",
    "ItemResult",
    "LiveCapitalConfig",
    "NotAuthorizedError",
    "ReadinessChecklist",
    "ReadinessDecision",
    "ReadinessError",
    "Verdict",
    "activate",
    "assert_within_limits",
    "evaluate_readiness",
    "minimal_capital_checklist",
    "render_readiness_report",
]
