"""Security-negative, fault-injection, and behaviour tests for the paper adapter.

Grouped by the guarantee each protects. The ones that carry the most weight:

* **The package cannot reach a live endpoint** — an allowlist, not a denylist,
  and no override parameter exists anywhere.
* **A session is unauthorizable without a qualified candidate**, which is the
  current state and therefore the path exercised most.
* **The module imports no networking library.** Asserted structurally, because a
  comment claiming "no network" is not a control.
* **Resubmitting a known order ID is a no-op**, not a second order.
* **The kill switch is one-way.**
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from alphaforge.broker import (
    ALLOWED_PAPER_ENDPOINTS,
    FORBIDDEN_CREDENTIAL_ENVIRONMENT,
    BrokerConfigurationError,
    BrokerSessionConfig,
    KillSwitchEngagedError,
    LiveTradingNotAuthorizedError,
    MarketClock,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderType,
    PaperBrokerAdapter,
    Quote,
    TerminalBrokerError,
    TimeInForce,
    assert_endpoint_is_paper,
    assert_no_credentials_in_environment,
    authorize_paper_session,
    derive_client_order_id,
    digest_account_identifier,
)
from alphaforge.broker import config as broker_config
from alphaforge.broker import paper_adapter as broker_paper_adapter
from alphaforge.research.qualification import (
    Criterion,
    EvidenceLink,
    QualificationRubric,
    qualify,
)

NOW = datetime(2026, 8, 4, 15, 0, tzinfo=UTC)
DIGEST = digest_account_identifier("PA-TEST-ACCOUNT")
PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
KEYCHAIN = "com.signal-foundry.alpaca-paper"


# ---------------------------------------------------------------------------
# Fixtures: a genuinely qualified decision, built through the real rubric
# ---------------------------------------------------------------------------


def _test_rubric() -> QualificationRubric:
    """A minimal single-criterion rubric for exercising the gate."""
    return QualificationRubric(
        version="broker-adapter-test-1",
        criteria=(
            Criterion(
                name="net_return",
                question="Did the candidate earn a positive net return?",
                threshold=0.0,
                direction="at_least",
                required_evidence=("dataset",),
            ),
        ),
    )


def _decide(net_return: float):
    """Return a decision produced by the real qualification machinery.

    Deliberately routed through ``qualify`` rather than instantiating the
    dataclass: a hand-built decision would not prove the adapter accepts what
    the qualification gate actually emits.
    """
    rubric = _test_rubric()
    return qualify(
        candidate_id="cand-test",
        rubric=rubric,
        expected_rubric_identity=rubric.identity,
        observations={"net_return": net_return},
        evidence={
            "net_return": (
                EvidenceLink(kind="dataset", identifier="run://test", content_hash="c" * 64),
            )
        },
        plan_hash="b" * 64,
        decided_at="2026-08-04T00:00:00Z",
        reconciliation_ok=True,
    )


def _qualified_decision():
    """A QUALIFIED_FOR_PAPER decision."""
    return _decide(0.12)


def _rejected_decision():
    """A REJECTED decision, mirroring Sprint 4's actual outcome."""
    return _decide(-0.197)


@pytest.fixture
def enabled_config() -> BrokerSessionConfig:
    return BrokerSessionConfig(enabled=True, endpoint=PAPER_ENDPOINT, keychain_service=KEYCHAIN)


@pytest.fixture
def adapter(enabled_config: BrokerSessionConfig) -> PaperBrokerAdapter:
    session = PaperBrokerAdapter(
        config=enabled_config,
        qualification=_qualified_decision(),
        account_id_digest=DIGEST,
        opening_cash=Decimal("100000"),
    )
    session.connect(environment={})
    return session


def _quote(bid: str = "100", ask: str = "100.10", *, at: datetime = NOW) -> Quote:
    return Quote(symbol="AAPL", bid=Decimal(bid), ask=Decimal(ask), observed_at=at)


def _open_clock(at: datetime = NOW) -> MarketClock:
    return MarketClock(is_open=True, server_time=at)


def _request(**overrides: object) -> OrderRequest:
    base: dict[str, object] = {
        "client_order_id": derive_client_order_id(
            strategy_id="s1",
            decision_timestamp=NOW,
            symbol="AAPL",
            side=OrderSide.BUY,
            sequence=0,
        ),
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": Decimal("10"),
        "order_type": OrderType.MARKET,
        "time_in_force": TimeInForce.DAY,
    }
    base.update(overrides)
    return OrderRequest(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The live endpoint is unreachable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.alpaca.markets",
        "https://api.alpaca.markets/v2",
        "https://live.example.com",
        "https://production.broker.example",
    ],
)
def test_a_live_endpoint_is_refused(endpoint: str) -> None:
    with pytest.raises(LiveTradingNotAuthorizedError):
        assert_endpoint_is_paper(endpoint)


def test_an_unknown_endpoint_is_refused_even_if_it_looks_like_paper() -> None:
    """An allowlist refuses the unknown; a denylist would have let this through."""
    with pytest.raises(LiveTradingNotAuthorizedError, match="not an allowlisted"):
        assert_endpoint_is_paper("https://paper-api.evil.example")


def test_an_unencrypted_endpoint_is_refused() -> None:
    with pytest.raises(BrokerConfigurationError, match="https"):
        assert_endpoint_is_paper("http://paper-api.alpaca.markets")


def test_every_allowlisted_endpoint_actually_passes() -> None:
    """Guards against an allowlist that is accidentally self-contradictory."""
    for endpoint in ALLOWED_PAPER_ENDPOINTS:
        assert assert_endpoint_is_paper(endpoint) == endpoint


def test_a_bad_endpoint_fails_when_written_not_when_enabled() -> None:
    """A latent bad endpoint must not wait for someone to flip the switch."""
    with pytest.raises(LiveTradingNotAuthorizedError):
        BrokerSessionConfig(enabled=False, endpoint="https://api.alpaca.markets")


def test_no_override_parameter_exists_anywhere_in_the_package() -> None:
    """Live capability is absent, not disabled by a flag someone can set."""
    forbidden = {"force", "allow_live", "override", "skip_checks", "unsafe", "bypass"}
    for module in (broker_config, broker_paper_adapter):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
                assert not (
                    names & forbidden
                ), f"{module.__name__}.{node.name} exposes {names & forbidden}"


def test_the_adapter_imports_no_networking_library() -> None:
    """A comment claiming "no network" is not a control; this is."""
    banned = {
        "socket",
        "http",
        "http.client",
        "urllib.request",
        "requests",
        "httpx",
        "aiohttp",
        "websockets",
        "ssl",
        "ftplib",
        "telnetlib",
        "smtplib",
    }
    source = Path(inspect.getfile(broker_paper_adapter)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not (imported & banned), f"adapter imports networking modules: {imported & banned}"


# ---------------------------------------------------------------------------
# Deny-by-default authorization
# ---------------------------------------------------------------------------


def test_a_default_configuration_refuses() -> None:
    """A configuration that forgets to mention trading does not trade."""
    assert BrokerSessionConfig().enabled is False
    with pytest.raises(BrokerConfigurationError, match="disabled"):
        authorize_paper_session(
            BrokerSessionConfig(), qualification=_qualified_decision(), environment={}
        )


def test_an_enabled_session_without_an_endpoint_is_refused() -> None:
    with pytest.raises(BrokerConfigurationError, match="requires an endpoint"):
        BrokerSessionConfig(enabled=True)


def test_an_enabled_session_without_a_keychain_service_is_refused() -> None:
    with pytest.raises(BrokerConfigurationError, match="keychain_service"):
        BrokerSessionConfig(enabled=True, endpoint=PAPER_ENDPOINT)


def test_a_non_allowlisted_keychain_service_is_refused() -> None:
    with pytest.raises(BrokerConfigurationError, match="not allowlisted"):
        BrokerSessionConfig(
            enabled=True, endpoint=PAPER_ENDPOINT, keychain_service="com.example.other"
        )


@pytest.mark.parametrize("variable", FORBIDDEN_CREDENTIAL_ENVIRONMENT)
def test_a_credential_in_the_environment_is_refused(variable: str) -> None:
    with pytest.raises(BrokerConfigurationError, match="environment variables"):
        assert_no_credentials_in_environment({variable: "secret-value"})


def test_the_environment_refusal_never_echoes_the_credential() -> None:
    """The error names the variable and withholds the value."""
    secret = "AKIA-NOT-A-REAL-SECRET-VALUE"
    with pytest.raises(BrokerConfigurationError) as caught:
        assert_no_credentials_in_environment({"ALPACA_API_KEY": secret})
    assert secret not in str(caught.value)
    assert "ALPACA_API_KEY" in str(caught.value)


def test_an_empty_environment_variable_is_not_treated_as_a_credential() -> None:
    assert_no_credentials_in_environment({"ALPACA_API_KEY": ""})


def test_an_unbounded_retry_budget_is_refused() -> None:
    with pytest.raises(BrokerConfigurationError, match="max_order_attempts"):
        BrokerSessionConfig(
            enabled=True,
            endpoint=PAPER_ENDPOINT,
            keychain_service=KEYCHAIN,
            max_order_attempts=50,
        )


def test_a_non_finite_staleness_bound_is_refused() -> None:
    with pytest.raises(BrokerConfigurationError, match="finite"):
        BrokerSessionConfig(
            enabled=True,
            endpoint=PAPER_ENDPOINT,
            keychain_service=KEYCHAIN,
            max_quote_age_seconds=float("inf"),
        )


# ---------------------------------------------------------------------------
# The qualification gate — currently the closed path
# ---------------------------------------------------------------------------


def test_no_session_without_a_qualification_decision(
    enabled_config: BrokerSessionConfig,
) -> None:
    with pytest.raises(BrokerConfigurationError, match="no qualification decision"):
        authorize_paper_session(enabled_config, qualification=None, environment={})


def test_a_rejected_candidate_cannot_open_a_session(
    enabled_config: BrokerSessionConfig,
) -> None:
    """Sprint 4's actual state: REJECTED, so paper evaluation is unauthorized."""
    decision = _rejected_decision()
    assert decision.verdict == "REJECTED"
    with pytest.raises(BrokerConfigurationError, match="REJECTED"):
        authorize_paper_session(enabled_config, qualification=decision, environment={})


def test_the_rejection_message_says_it_is_not_a_configuration_problem(
    enabled_config: BrokerSessionConfig,
) -> None:
    with pytest.raises(BrokerConfigurationError) as caught:
        authorize_paper_session(enabled_config, qualification=_rejected_decision(), environment={})
    assert "cannot be resolved by changing a setting" in str(caught.value)


def test_a_caller_constructed_stand_in_is_refused(
    enabled_config: BrokerSessionConfig,
) -> None:
    """A duck-typed object with `qualified=True` must not open a session."""

    class FakeDecision:
        qualified = True
        verdict = "QUALIFIED_FOR_PAPER"
        candidate_id = "fake"
        blocking_failures: tuple[str, ...] = ()

    with pytest.raises(BrokerConfigurationError, match="not a caller-constructed"):
        authorize_paper_session(
            enabled_config,
            qualification=FakeDecision(),  # type: ignore[arg-type]
            environment={},
        )


def test_an_unconnected_adapter_refuses_every_order_operation(
    enabled_config: BrokerSessionConfig,
) -> None:
    session = PaperBrokerAdapter(
        config=enabled_config, qualification=_qualified_decision(), account_id_digest=DIGEST
    )
    with pytest.raises(BrokerConfigurationError, match="not connected"):
        session.submit_order(_request(), quote=_quote(), clock=_open_clock(), now=NOW)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_resubmitting_a_known_order_id_is_a_no_op(adapter: PaperBrokerAdapter) -> None:
    """A replayed decision cycle must not double exposure."""
    request = _request()
    first = adapter.submit_order(request, quote=_quote(), clock=_open_clock(), now=NOW)
    second = adapter.submit_order(request, quote=_quote(), clock=_open_clock(), now=NOW)
    assert first == second
    assert len(adapter.list_fills()) == 1
    snapshot = adapter.account_snapshot(now=NOW)
    position = snapshot.position_for("AAPL")
    assert position is not None
    assert position.quantity == Decimal("10")


def test_a_duplicate_after_a_connection_loss_recovers_state(
    adapter: PaperBrokerAdapter,
) -> None:
    """The caller cannot know whether we saw the first submission; asking is safe."""
    request = _request()
    adapter.submit_order(request, quote=_quote(), clock=_open_clock(), now=NOW)
    recovered = adapter.get_order(request.client_order_id)
    assert recovered.state is OrderState.FILLED
    assert recovered.filled_quantity == Decimal("10")


def test_querying_an_unknown_order_is_terminal(adapter: PaperBrokerAdapter) -> None:
    with pytest.raises(TerminalBrokerError, match="unknown client_order_id"):
        adapter.get_order("af-never-sent")


# ---------------------------------------------------------------------------
# Fill semantics
# ---------------------------------------------------------------------------


def test_a_market_buy_fills_at_the_ask_not_the_midpoint(
    adapter: PaperBrokerAdapter,
) -> None:
    """The midpoint is what you get when someone else pays the spread."""
    status = adapter.submit_order(_request(), quote=_quote(), clock=_open_clock(), now=NOW)
    assert status.average_fill_price == Decimal("100.10")


def test_a_market_sell_fills_at_the_bid(adapter: PaperBrokerAdapter) -> None:
    status = adapter.submit_order(
        _request(side=OrderSide.SELL, client_order_id="af-sell-1"),
        quote=_quote(),
        clock=_open_clock(),
        now=NOW,
    )
    assert status.average_fill_price == Decimal("100")


def test_a_non_marketable_limit_order_rests_unfilled(adapter: PaperBrokerAdapter) -> None:
    """It never fills "close enough"."""
    status = adapter.submit_order(
        _request(
            client_order_id="af-limit-1",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("99"),
        ),
        quote=_quote(),
        clock=_open_clock(),
        now=NOW,
    )
    assert status.state is OrderState.ACCEPTED
    assert status.filled_quantity == Decimal("0")


def test_a_marketable_limit_order_fills(adapter: PaperBrokerAdapter) -> None:
    status = adapter.submit_order(
        _request(
            client_order_id="af-limit-2",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("101"),
        ),
        quote=_quote(),
        clock=_open_clock(),
        now=NOW,
    )
    assert status.state is OrderState.FILLED


def test_nothing_fills_while_the_market_is_closed(adapter: PaperBrokerAdapter) -> None:
    """Simulating a closed-market fill would invent liquidity that did not exist."""
    status = adapter.submit_order(
        _request(),
        quote=_quote(),
        clock=MarketClock(is_open=False, server_time=NOW),
        now=NOW,
    )
    assert status.state is OrderState.ACCEPTED
    assert adapter.list_fills() == ()


def test_cash_and_position_reconcile_after_a_fill(adapter: PaperBrokerAdapter) -> None:
    adapter.submit_order(_request(), quote=_quote(), clock=_open_clock(), now=NOW)
    snapshot = adapter.account_snapshot(now=NOW)
    assert snapshot.cash == Decimal("100000") - Decimal("10") * Decimal("100.10")
    position = snapshot.position_for("AAPL")
    assert position is not None
    assert position.quantity == Decimal("10")


def test_closing_a_position_removes_it_rather_than_leaving_a_zero(
    adapter: PaperBrokerAdapter,
) -> None:
    """A flat symbol must not look like a position during reconciliation."""
    adapter.submit_order(_request(), quote=_quote(), clock=_open_clock(), now=NOW)
    adapter.submit_order(
        _request(side=OrderSide.SELL, client_order_id="af-close-1"),
        quote=_quote(),
        clock=_open_clock(),
        now=NOW,
    )
    assert adapter.account_snapshot(now=NOW).position_for("AAPL") is None


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


def test_a_stale_quote_blocks_submission(adapter: PaperBrokerAdapter) -> None:
    stale = _quote(at=NOW - timedelta(seconds=3_600))
    with pytest.raises(TerminalBrokerError, match="stale"):
        adapter.submit_order(_request(), quote=stale, clock=_open_clock(), now=NOW)


def test_a_future_dated_quote_blocks_submission(adapter: PaperBrokerAdapter) -> None:
    """A clock fault or a fabricated observation; neither is safe to trade on."""
    future = _quote(at=NOW + timedelta(seconds=60))
    with pytest.raises(TerminalBrokerError, match="future"):
        adapter.submit_order(_request(), quote=future, clock=_open_clock(), now=NOW)


def test_clock_skew_beyond_the_bound_halts(adapter: PaperBrokerAdapter) -> None:
    skewed = MarketClock(is_open=True, server_time=NOW - timedelta(seconds=120))
    with pytest.raises(TerminalBrokerError, match="clock skew"):
        adapter.submit_order(_request(), quote=_quote(), clock=skewed, now=NOW)


def test_a_mismatched_quote_symbol_is_refused(adapter: PaperBrokerAdapter) -> None:
    other = Quote(symbol="MSFT", bid=Decimal("100"), ask=Decimal("101"), observed_at=NOW)
    with pytest.raises(TerminalBrokerError, match="quote is for"):
        adapter.submit_order(_request(), quote=other, clock=_open_clock(), now=NOW)


def test_insufficient_buying_power_is_terminal_not_retryable(
    enabled_config: BrokerSessionConfig,
) -> None:
    """Retrying an underfunded order cannot succeed."""
    session = PaperBrokerAdapter(
        config=enabled_config,
        qualification=_qualified_decision(),
        account_id_digest=DIGEST,
        opening_cash=Decimal("100"),
    )
    session.connect(environment={})
    with pytest.raises(TerminalBrokerError, match="insufficient buying power"):
        session.submit_order(_request(), quote=_quote(), clock=_open_clock(), now=NOW)


def test_a_stale_account_snapshot_is_refused(adapter: PaperBrokerAdapter) -> None:
    snapshot = adapter.account_snapshot(now=NOW - timedelta(seconds=600))
    with pytest.raises(TerminalBrokerError, match="account snapshot"):
        adapter.assert_account_fresh(snapshot, now=NOW)


# ---------------------------------------------------------------------------
# Cancellation and the kill switch
# ---------------------------------------------------------------------------


def test_a_resting_order_can_be_canceled(adapter: PaperBrokerAdapter) -> None:
    request = _request(
        client_order_id="af-limit-3", order_type=OrderType.LIMIT, limit_price=Decimal("99")
    )
    adapter.submit_order(request, quote=_quote(), clock=_open_clock(), now=NOW)
    canceled = adapter.cancel_order(request.client_order_id, now=NOW)
    assert canceled.state is OrderState.CANCELED
    assert canceled.reason == "canceled_by_client"


def test_cancelling_a_settled_order_is_refused(adapter: PaperBrokerAdapter) -> None:
    """Cancelling a filled order would rewrite history."""
    request = _request()
    adapter.submit_order(request, quote=_quote(), clock=_open_clock(), now=NOW)
    with pytest.raises(TerminalBrokerError, match="already terminal"):
        adapter.cancel_order(request.client_order_id, now=NOW)


def test_the_kill_switch_halts_every_order_operation(adapter: PaperBrokerAdapter) -> None:
    adapter.engage_kill_switch()
    assert adapter.kill_switch_engaged
    with pytest.raises(KillSwitchEngagedError):
        adapter.submit_order(_request(), quote=_quote(), clock=_open_clock(), now=NOW)
    with pytest.raises(KillSwitchEngagedError):
        adapter.account_snapshot(now=NOW)


def test_the_kill_switch_is_one_way(adapter: PaperBrokerAdapter) -> None:
    """A switch code can flip back is not a kill switch."""
    adapter.engage_kill_switch()
    assert not hasattr(adapter, "disengage_kill_switch")
    assert not hasattr(adapter, "reset_kill_switch")
    adapter.engage_kill_switch()  # idempotent
    assert adapter.kill_switch_engaged


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def test_the_session_summary_carries_no_credential(adapter: PaperBrokerAdapter) -> None:
    payload = adapter.to_dict()
    assert payload["network_capable"] is False
    assert payload["simulated"] is True
    assert "credential" not in str(payload).lower() or "no credential" in str(payload).lower()
    assert KEYCHAIN in str(payload)  # the service NAME is not a secret
    assert "operational readiness, not executable performance" in payload["evidence_note"]


def test_an_account_digest_is_stable_and_hides_the_identifier() -> None:
    first = digest_account_identifier("PA-REAL-ACCOUNT-123")
    assert first == digest_account_identifier("PA-REAL-ACCOUNT-123")
    assert "PA-REAL-ACCOUNT-123" not in first
    assert len(first) == 64


def test_the_adapter_is_deterministic(enabled_config: BrokerSessionConfig) -> None:
    def run() -> list[str]:
        session = PaperBrokerAdapter(
            config=enabled_config,
            qualification=_qualified_decision(),
            account_id_digest=DIGEST,
        )
        session.connect(environment={})
        for index in range(3):
            session.submit_order(
                _request(client_order_id=f"af-det-{index}"),
                quote=_quote(),
                clock=_open_clock(),
                now=NOW,
            )
        return [fill.to_dict()["price"] for fill in session.list_fills()]

    assert run() == run()
