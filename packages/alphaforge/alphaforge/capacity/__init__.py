"""Point-in-time borrow, liquidity, and capacity policy (SF-S4-MR5).

One causal policy enforced at both the decision and execution boundaries:

- :mod:`~alphaforge.capacity.contracts` — versioned, bounded, immutable borrow,
  locate, liquidity, and policy records with content-derived identity. Missing,
  stale, expired, unknown, duplicated, or conflicting records fail closed.
- :mod:`~alphaforge.capacity.budgets` — conserved per-session participation and
  book-notional ledgers with an exact reserve/consume/release/reject identity and
  idempotent replay.
- :mod:`~alphaforge.capacity.policy` — causal resolution and authorization.
  Opening a short requires positive evidence; covering never does.
- :mod:`~alphaforge.capacity.buyin` — forced buy-ins on recall or restriction,
  halting publication on an unresolved residual.
- :mod:`~alphaforge.capacity.frontier` — capacity frontier by complete rerun per
  frozen AUM scenario, never by scaling a completed series.

Nothing here claims a deployable AUM, broker capacity, paper-trading readiness,
or expected profit.
"""

from alphaforge.capacity.budgets import (
    CONSERVATION_TOLERANCE,
    Reservation,
    SessionCapacityLedger,
)
from alphaforge.capacity.buyin import (
    BuyInOrder,
    ForcedBuyInBook,
    ForcedBuyInHalt,
    detect_recalls,
)
from alphaforge.capacity.contracts import (
    SCHEMA_VERSION,
    SHORTABLE_STATUSES,
    BorrowAvailability,
    CapacityContractError,
    CapacityPolicyDeclaration,
    CapacityPolicyViolation,
    LiquidityObservation,
    LocateRecord,
    book_digest,
    validate_unique_records,
)
from alphaforge.capacity.frontier import (
    MAX_SCENARIOS,
    ScenarioResult,
    capacity_frontier,
    frontier_summary,
    validate_scenarios,
    verify_row_aggregation,
)
from alphaforge.capacity.policy import (
    CapacityDecision,
    CapacityEvidence,
    CapacityPolicy,
    ResolvedCapacity,
)

__all__ = [
    "CONSERVATION_TOLERANCE",
    "MAX_SCENARIOS",
    "SCHEMA_VERSION",
    "SHORTABLE_STATUSES",
    "BorrowAvailability",
    "BuyInOrder",
    "CapacityContractError",
    "CapacityDecision",
    "CapacityEvidence",
    "CapacityPolicy",
    "CapacityPolicyDeclaration",
    "CapacityPolicyViolation",
    "ForcedBuyInBook",
    "ForcedBuyInHalt",
    "LiquidityObservation",
    "LocateRecord",
    "Reservation",
    "ResolvedCapacity",
    "ScenarioResult",
    "SessionCapacityLedger",
    "book_digest",
    "capacity_frontier",
    "detect_recalls",
    "frontier_summary",
    "validate_scenarios",
    "validate_unique_records",
    "verify_row_aggregation",
]
