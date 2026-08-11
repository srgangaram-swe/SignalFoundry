"""Tests for delayed shadow forecast evidence (SF-S5-SL-MR4).

Grouped by the invariant each protects. The ones carrying the most weight:

* **A forecast cannot consume its own outcome.** Chronology is checked at
  construction and again at scoring time.
* **Append-only is enforced by the database**, so the guarantee survives a
  caller holding a raw connection.
* **First-eligible and latest-known never merge.** They carry distinct
  identities and the delta between them stays computable.
* **Too little evidence returns a verdict, not a number.**
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quant_platform.shadow.contracts import (
    MIN_SCORED_FORECASTS,
    CampaignState,
    ChronologyError,
    OutcomeStatus,
    ProbabilityVector,
    ReportBasis,
    SealedBatch,
    SealedCampaignError,
    ShadowForecast,
    ShadowOutcome,
    ShadowValidationError,
    assert_campaign_transition,
)
from quant_platform.shadow.reports import (
    ReportVerdict,
    brier_score,
    build_report,
    log_score,
    revision_delta,
)
from quant_platform.shadow.schema import SHADOW_MIGRATION_SQL
from quant_platform.shadow.store import ShadowStore
from quant_platform.tracking.contracts import ConflictError, NotFoundError

AS_OF = datetime(2026, 8, 1, tzinfo=UTC)
DIGEST_A = "a" * 64
DIGEST_C = "c" * 64


def _vector(up: float = 0.6) -> ProbabilityVector:
    return ProbabilityVector(labels=("down", "up"), probabilities=(1.0 - up, up))


def _forecast(symbol: str = "AAPL", *, as_of: datetime = AS_OF, up: float = 0.6) -> ShadowForecast:
    return ShadowForecast(
        campaign="wiki-shadow",
        symbol=symbol,
        as_of=as_of,
        target_instant=as_of + timedelta(days=5),
        horizon_days=5,
        distribution=_vector(up),
        feature_prefix_digest=DIGEST_C,
        model_identity=DIGEST_A,
    )


def _batch(symbols: tuple[str, ...] = ("AAPL",), *, as_of: datetime = AS_OF) -> SealedBatch:
    return SealedBatch(
        campaign="wiki-shadow",
        as_of=as_of,
        expected_universe=symbols,
        forecasts=tuple(_forecast(item, as_of=as_of) for item in symbols),
        sealed_at=as_of,
    )


def _store(tmp_path: Path) -> ShadowStore:
    database = tmp_path / "registry.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(SHADOW_MIGRATION_SQL)
    connection.close()
    store = ShadowStore(database)
    store.create_campaign(
        "wiki-shadow",
        horizon_days=5,
        class_labels=("down", "up"),
        model_identity=DIGEST_A,
        now=AS_OF,
    )
    store.transition("wiki-shadow", CampaignState.ACTIVE, now=AS_OF)
    return store


# ---------------------------------------------------------------------------
# Chronology: a forecast cannot consume its own outcome
# ---------------------------------------------------------------------------


def test_a_target_at_or_before_as_of_is_refused() -> None:
    with pytest.raises(ChronologyError, match="strictly later"):
        ShadowForecast(
            campaign="wiki-shadow",
            symbol="AAPL",
            as_of=AS_OF,
            target_instant=AS_OF,
            horizon_days=5,
            distribution=_vector(),
            feature_prefix_digest=DIGEST_C,
            model_identity=DIGEST_A,
        )


def test_an_outcome_observed_at_the_as_of_instant_is_refused() -> None:
    """Equal instants are refused, not merely earlier ones."""
    with pytest.raises(ChronologyError, match="not after as_of"):
        _forecast().assert_precedes(AS_OF)


def test_an_outcome_recorded_before_it_was_observed_is_refused() -> None:
    with pytest.raises(ChronologyError, match="precedes observed_at"):
        ShadowOutcome(
            forecast_id="f" * 64,
            realized_label="up",
            observed_at=AS_OF + timedelta(days=5),
            recorded_at=AS_OF + timedelta(days=4),
            revision=0,
        )


def test_a_naive_instant_is_refused() -> None:
    with pytest.raises(ShadowValidationError, match="timezone-aware"):
        _forecast(as_of=datetime(2026, 8, 1))  # noqa: DTZ001


def test_a_batch_sealed_before_its_as_of_is_refused() -> None:
    with pytest.raises(ChronologyError, match="sealed_at"):
        SealedBatch(
            campaign="wiki-shadow",
            as_of=AS_OF,
            expected_universe=("AAPL",),
            forecasts=(_forecast(),),
            sealed_at=AS_OF - timedelta(seconds=1),
        )


# ---------------------------------------------------------------------------
# Sealing is all-or-nothing over a complete universe
# ---------------------------------------------------------------------------


def test_a_partial_universe_cannot_be_sealed() -> None:
    with pytest.raises(ShadowValidationError, match="batch is incomplete"):
        SealedBatch(
            campaign="wiki-shadow",
            as_of=AS_OF,
            expected_universe=("AAPL", "MSFT"),
            forecasts=(_forecast("AAPL"),),
            sealed_at=AS_OF,
        )


def test_a_forecast_outside_the_universe_is_refused() -> None:
    with pytest.raises(ShadowValidationError, match="absent from the expected universe"):
        SealedBatch(
            campaign="wiki-shadow",
            as_of=AS_OF,
            expected_universe=("AAPL",),
            forecasts=(_forecast("AAPL"), _forecast("MSFT")),
            sealed_at=AS_OF,
        )


def test_two_forecasts_for_one_symbol_are_refused() -> None:
    with pytest.raises(ShadowValidationError, match="same symbol"):
        SealedBatch(
            campaign="wiki-shadow",
            as_of=AS_OF,
            expected_universe=("AAPL",),
            forecasts=(_forecast("AAPL"), _forecast("AAPL", up=0.7)),
            sealed_at=AS_OF,
        )


def test_forecasts_from_a_different_instant_cannot_share_a_batch() -> None:
    """One batch is one decision instant."""
    with pytest.raises(ChronologyError, match="different as_of"):
        SealedBatch(
            campaign="wiki-shadow",
            as_of=AS_OF,
            expected_universe=("AAPL", "MSFT"),
            forecasts=(_forecast("AAPL"), _forecast("MSFT", as_of=AS_OF + timedelta(days=1))),
            sealed_at=AS_OF,
        )


# ---------------------------------------------------------------------------
# Persistence, idempotency, restart
# ---------------------------------------------------------------------------


def test_sealing_round_trips(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch = _batch(("AAPL", "MSFT"))
    store.seal_batch(batch)
    loaded = store.load_forecasts("wiki-shadow")
    assert len(loaded) == 2
    assert {item.forecast_id for item in loaded} == set(batch.forecast_ids())


def test_resealing_an_identical_batch_is_a_no_op(tmp_path: Path) -> None:
    """A retry after an ambiguous failure must not create a second batch."""
    store = _store(tmp_path)
    batch = _batch()
    first = store.seal_batch(batch)
    second = store.seal_batch(batch)
    assert first == second
    assert store.summary("wiki-shadow")["batches"] == 1
    assert store.summary("wiki-shadow")["forecasts"] == 1


def test_a_different_batch_for_the_same_instant_is_a_conflict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.seal_batch(_batch())
    divergent = SealedBatch(
        campaign="wiki-shadow",
        as_of=AS_OF,
        expected_universe=("AAPL",),
        forecasts=(_forecast("AAPL", up=0.9),),
        sealed_at=AS_OF,
    )
    with pytest.raises(ConflictError, match="one opinion"):
        store.seal_batch(divergent)


def test_sealing_into_a_non_active_campaign_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.transition("wiki-shadow", CampaignState.SEALED, now=AS_OF)
    with pytest.raises(SealedCampaignError, match="only be"):
        store.seal_batch(_batch())


def test_state_survives_a_restart(tmp_path: Path) -> None:
    """A fresh store object over the same file sees committed state."""
    store = _store(tmp_path)
    store.seal_batch(_batch())
    reopened = ShadowStore(tmp_path / "registry.sqlite3")
    assert reopened.summary("wiki-shadow")["forecasts"] == 1
    assert reopened.campaign_state("wiki-shadow") is CampaignState.ACTIVE


def test_an_outcome_for_an_unknown_forecast_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(NotFoundError, match="unknown"):
        store.append_outcome(
            ShadowOutcome(
                forecast_id="e" * 64,
                realized_label="up",
                observed_at=AS_OF + timedelta(days=5),
                recorded_at=AS_OF + timedelta(days=5),
                revision=0,
            )
        )


def test_appending_the_same_revision_twice_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    batch = _batch()
    store.seal_batch(batch)
    outcome = ShadowOutcome(
        forecast_id=batch.forecast_ids()[0],
        realized_label="up",
        observed_at=AS_OF + timedelta(days=5),
        recorded_at=AS_OF + timedelta(days=5),
        revision=0,
    )
    assert store.append_outcome(outcome) == 0
    assert store.append_outcome(outcome) == 0
    assert store.summary("wiki-shadow")["outcome_revisions"] == 1


def test_a_conflicting_revision_is_refused(tmp_path: Path) -> None:
    """A correction appends a new revision; it never rewrites one."""
    store = _store(tmp_path)
    batch = _batch()
    store.seal_batch(batch)
    forecast_id = batch.forecast_ids()[0]
    store.append_outcome(
        ShadowOutcome(
            forecast_id=forecast_id,
            realized_label="up",
            observed_at=AS_OF + timedelta(days=5),
            recorded_at=AS_OF + timedelta(days=5),
            revision=0,
        )
    )
    with pytest.raises(ConflictError, match="appends a new revision"):
        store.append_outcome(
            ShadowOutcome(
                forecast_id=forecast_id,
                realized_label="down",
                observed_at=AS_OF + timedelta(days=5),
                recorded_at=AS_OF + timedelta(days=5),
                revision=0,
            )
        )


# ---------------------------------------------------------------------------
# The database enforces append-only, not the application
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE sl_shadow_forecasts SET symbol = 'MSFT'",
        "DELETE FROM sl_shadow_forecasts",
        "UPDATE sl_shadow_outcomes SET realized_label = 'down'",
        "DELETE FROM sl_shadow_outcomes",
        "DELETE FROM sl_shadow_batches",
        "UPDATE sl_shadow_campaigns SET horizon_days = 9",
    ],
)
def test_raw_sql_cannot_rewrite_sealed_evidence(tmp_path: Path, statement: str) -> None:
    """A caller holding a connection still cannot rewrite history."""
    store = _store(tmp_path)
    batch = _batch()
    store.seal_batch(batch)
    store.append_outcome(
        ShadowOutcome(
            forecast_id=batch.forecast_ids()[0],
            realized_label="up",
            observed_at=AS_OF + timedelta(days=5),
            recorded_at=AS_OF + timedelta(days=5),
            revision=0,
        )
    )
    connection = sqlite3.connect(tmp_path / "registry.sqlite3")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only|immutable|state changes"):
            connection.execute(statement)
            connection.commit()
    finally:
        connection.close()


def test_a_terminal_campaign_cannot_be_reopened(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.transition("wiki-shadow", CampaignState.SEALED, now=AS_OF)
    store.transition("wiki-shadow", CampaignState.RECONCILING, now=AS_OF)
    store.transition("wiki-shadow", CampaignState.CLOSED, now=AS_OF)
    with pytest.raises(ShadowValidationError, match="terminal"):
        assert_campaign_transition(CampaignState.CLOSED, CampaignState.ACTIVE)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_a_perfect_forecast_scores_zero() -> None:
    certain = ProbabilityVector(labels=("down", "up"), probabilities=(0.0, 1.0))
    assert brier_score(certain, "up") == 0.0
    assert log_score(certain, "up") == pytest.approx(0.0, abs=1e-12)


def test_a_confident_miss_is_large_but_finite() -> None:
    """An unclipped -log(0) would dominate every aggregate."""
    certain = ProbabilityVector(labels=("down", "up"), probabilities=(1.0, 0.0))
    score = log_score(certain, "up")
    assert score > 30.0
    assert score < float("inf")


def test_brier_is_symmetric_between_classes() -> None:
    a = ProbabilityVector(labels=("down", "up"), probabilities=(0.3, 0.7))
    b = ProbabilityVector(labels=("down", "up"), probabilities=(0.7, 0.3))
    assert brier_score(a, "up") == pytest.approx(brier_score(b, "down"))


def test_a_realised_label_outside_the_class_space_is_refused() -> None:
    """A label mismatch is a reconciliation fault, not a score of zero."""
    with pytest.raises(ShadowValidationError, match="label space"):
        brier_score(_vector(), "sideways")


# ---------------------------------------------------------------------------
# Reports: the two bases stay distinct
# ---------------------------------------------------------------------------


def _campaign_with_revisions(count: int) -> tuple[list[ShadowForecast], dict[str, Any]]:
    """Build a campaign where later revisions flip some outcomes favourably."""
    forecasts: list[ShadowForecast] = []
    outcomes: dict[str, Any] = {}
    for index in range(count):
        as_of = AS_OF + timedelta(days=index // 5)
        forecast = _forecast(f"SYM{index:02d}", as_of=as_of, up=0.8)
        forecasts.append(forecast)
        observed = as_of + timedelta(days=5)
        history = [
            ShadowOutcome(
                forecast_id=forecast.forecast_id,
                realized_label="down" if index % 2 else "up",
                observed_at=observed,
                recorded_at=observed,
                revision=0,
            )
        ]
        if index % 2:
            # A later correction flips a miss into a hit, which is exactly the
            # revision effect the two bases must keep separable.
            history.append(
                ShadowOutcome(
                    forecast_id=forecast.forecast_id,
                    realized_label="up",
                    observed_at=observed,
                    recorded_at=observed + timedelta(days=1),
                    revision=1,
                )
            )
        outcomes[forecast.forecast_id] = tuple(history)
    return forecasts, outcomes


def test_below_the_minimum_returns_a_verdict_not_a_number() -> None:
    forecasts, outcomes = _campaign_with_revisions(4)
    report = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN)
    assert report.verdict is ReportVerdict.INSUFFICIENT_EVIDENCE
    assert report.mean_brier is None
    assert report.brier_interval is None


def test_at_the_minimum_a_score_is_reported() -> None:
    forecasts, outcomes = _campaign_with_revisions(MIN_SCORED_FORECASTS)
    report = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN)
    assert report.verdict is ReportVerdict.REPORTED
    assert report.mean_brier is not None
    assert report.scored_count == MIN_SCORED_FORECASTS


def test_the_two_bases_differ_when_revisions_exist() -> None:
    """Latest-known must not be quotable as live-available performance."""
    forecasts, outcomes = _campaign_with_revisions(40)
    first = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.FIRST_ELIGIBLE)
    latest = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN)
    assert first.identity != latest.identity
    assert first.mean_brier != latest.mean_brier
    delta = revision_delta(first, latest)
    assert delta["identities_differ"] is True
    assert delta["brier_movement"] < 0  # corrections flattered the score


def test_revision_delta_refuses_a_reversed_pairing() -> None:
    forecasts, outcomes = _campaign_with_revisions(40)
    first = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.FIRST_ELIGIBLE)
    latest = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN)
    with pytest.raises(ShadowValidationError, match="first-eligible report against"):
        revision_delta(latest, first)


def test_the_interval_is_deterministic() -> None:
    forecasts, outcomes = _campaign_with_revisions(40)
    a = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN, seed=7)
    b = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN, seed=7)
    assert a.brier_interval == b.brier_interval
    assert a.identity == b.identity


def test_a_single_day_reports_no_interval() -> None:
    """One block resampled repeatedly asserts a precision the data lacks."""
    forecasts = [_forecast(f"SYM{index:02d}", up=0.8) for index in range(MIN_SCORED_FORECASTS)]
    outcomes = {
        item.forecast_id: (
            ShadowOutcome(
                forecast_id=item.forecast_id,
                realized_label="up",
                observed_at=AS_OF + timedelta(days=5),
                recorded_at=AS_OF + timedelta(days=5),
                revision=0,
            ),
        )
        for item in forecasts
    }
    report = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN)
    assert report.distinct_days == 1
    assert report.brier_interval is None


def test_pending_and_missing_outcomes_are_counted_separately() -> None:
    forecasts = [_forecast(f"SYM{index:02d}") for index in range(3)]
    outcomes = {
        forecasts[0].forecast_id: (
            ShadowOutcome(
                forecast_id=forecasts[0].forecast_id,
                realized_label="up",
                observed_at=AS_OF + timedelta(days=5),
                recorded_at=AS_OF + timedelta(days=5),
                revision=0,
                status=OutcomeStatus.PENDING,
            ),
        )
    }
    report = build_report("wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN)
    assert report.pending_count == 1
    assert report.missing_count == 2
    assert report.scored_count == 0


def test_an_outcome_for_an_unknown_forecast_is_refused_by_the_report() -> None:
    forecasts = [_forecast()]
    with pytest.raises(ShadowValidationError, match="absent from this campaign"):
        build_report(
            "wiki-shadow",
            forecasts,
            {"d" * 64: ()},
            basis=ReportBasis.LATEST_KNOWN,
        )


def test_the_report_serializes_and_carries_its_limitations() -> None:
    forecasts, outcomes = _campaign_with_revisions(40)
    payload = build_report(
        "wiki-shadow", forecasts, outcomes, basis=ReportBasis.LATEST_KNOWN
    ).to_dict()
    assert json.loads(json.dumps(payload))
    assert "not prospective evidence" in payload["interpretation"]
    assert payload["baseline_brier"] is not None
