"""Broker-neutral contracts and a strictly simulated paper adapter (SF-S5-MR3).

- :mod:`~alphaforge.broker.contracts` — typed account, order, fill, position,
  clock, and quote records with an explicit order-state machine, UTC-only time,
  decimal money, and a retryable-versus-terminal failure split.
- :mod:`~alphaforge.broker.config` — the deny-by-default authorization boundary:
  disabled unless enabled, an allowlisted paper endpoint, Keychain-only
  credentials, and a required ``QUALIFIED_FOR_PAPER`` decision.
- :mod:`~alphaforge.broker.paper_adapter` — an in-process simulated broker. No
  network client, no socket, no credential read.
- :mod:`~alphaforge.broker.durable` — transactional, hash-chained session state
  with restart recovery that refuses tampered, partial, stale, incompatible, or
  ambiguous snapshots.
- :mod:`~alphaforge.broker.reconciliation` — compares persisted intent against
  broker-reported state and halts on divergence. It never repairs and never
  liquidates.

**No live capability exists here.** It is absent rather than disabled by a flag,
and there is no override parameter anywhere in the package. Sprint 4's verdict is
``REJECTED``, so a paper session is currently unauthorizable — the intended state.

Paper fills are simulated and bound operational readiness only. They are not
evidence of executable performance.
"""

from alphaforge.broker.config import (
    ALLOWED_KEYCHAIN_SERVICES,
    ALLOWED_PAPER_ENDPOINTS,
    FORBIDDEN_CREDENTIAL_ENVIRONMENT,
    BrokerConfigurationError,
    BrokerSessionConfig,
    LiveTradingNotAuthorizedError,
    assert_endpoint_is_paper,
    assert_no_credentials_in_environment,
    authorize_paper_session,
    digest_account_identifier,
)
from alphaforge.broker.contracts import (
    ALLOWED_ORDER_TRANSITIONS,
    QUANTITY_EPSILON,
    TERMINAL_ORDER_STATES,
    AccountSnapshot,
    BrokerContractError,
    Fill,
    MarketClock,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    RetryableBrokerError,
    TerminalBrokerError,
    TimeInForce,
    assert_transition_allowed,
    derive_client_order_id,
    utc_timestamp,
    validate_client_order_id,
    validate_symbol,
)
from alphaforge.broker.durable import (
    DEFAULT_MAX_SNAPSHOT_AGE,
    SCHEMA_VERSION,
    ClockRollbackError,
    DurableStateError,
    OrderIntent,
    SessionSnapshot,
    SessionStateStore,
    SnapshotIncompatibleError,
    SnapshotIntegrityError,
    SnapshotStaleError,
)
from alphaforge.broker.paper_adapter import (
    MAX_TRACKED_FILLS,
    MAX_TRACKED_ORDERS,
    KillSwitchEngagedError,
    PaperBrokerAdapter,
)
from alphaforge.broker.reconciliation import (
    DEFAULT_CASH_TOLERANCE,
    Divergence,
    DivergenceKind,
    ReconciliationError,
    ReconciliationReport,
    apply_fills_idempotently,
    intents_from_orders,
    reconcile,
)

__all__ = [
    "ALLOWED_KEYCHAIN_SERVICES",
    "ALLOWED_ORDER_TRANSITIONS",
    "ALLOWED_PAPER_ENDPOINTS",
    "AccountSnapshot",
    "BrokerConfigurationError",
    "BrokerContractError",
    "BrokerSessionConfig",
    "ClockRollbackError",
    "DEFAULT_CASH_TOLERANCE",
    "DEFAULT_MAX_SNAPSHOT_AGE",
    "Divergence",
    "DivergenceKind",
    "DurableStateError",
    "FORBIDDEN_CREDENTIAL_ENVIRONMENT",
    "Fill",
    "KillSwitchEngagedError",
    "LiveTradingNotAuthorizedError",
    "MAX_TRACKED_FILLS",
    "MAX_TRACKED_ORDERS",
    "MarketClock",
    "OrderIntent",
    "OrderRequest",
    "OrderSide",
    "OrderState",
    "OrderStatus",
    "OrderType",
    "PaperBrokerAdapter",
    "Position",
    "QUANTITY_EPSILON",
    "Quote",
    "ReconciliationError",
    "ReconciliationReport",
    "RetryableBrokerError",
    "SCHEMA_VERSION",
    "SessionSnapshot",
    "SessionStateStore",
    "SnapshotIncompatibleError",
    "SnapshotIntegrityError",
    "SnapshotStaleError",
    "TERMINAL_ORDER_STATES",
    "TerminalBrokerError",
    "TimeInForce",
    "apply_fills_idempotently",
    "assert_endpoint_is_paper",
    "assert_no_credentials_in_environment",
    "assert_transition_allowed",
    "authorize_paper_session",
    "derive_client_order_id",
    "digest_account_identifier",
    "intents_from_orders",
    "reconcile",
    "utc_timestamp",
    "validate_client_order_id",
    "validate_symbol",
]
