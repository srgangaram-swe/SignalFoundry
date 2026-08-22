"""Property-style and adversarial tests for aggregate evidence manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError as PydanticValidationError

from quant_platform.service.manifests import (
    MAX_DIAGNOSTIC_VALUES,
    MAX_FORECAST_AGGREGATES,
    MAX_MODEL_CARD_TEXT_BYTES,
    DiagnosticCategory,
    DiagnosticsManifest,
    DiagnosticStatus,
    DiagnosticValue,
    ForecastAggregate,
    ForecastSplit,
    ForecastSummaryManifest,
    HorizonUnit,
    ManifestKind,
    ManifestValidationError,
    ModelCardManifest,
    ModelCardSection,
    ModelCardSectionName,
    parse_evidence_manifest,
)

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _forecast(aggregate_id: str = "aggregate-001") -> ForecastAggregate:
    return ForecastAggregate(
        aggregate_id=aggregate_id,
        target="return-1d",
        split=ForecastSplit.TEST,
        horizon_steps=1,
        horizon_unit=HorizonUnit.BUSINESS_DAYS,
        window_start=NOW - timedelta(days=90),
        window_end=NOW - timedelta(days=1),
        sample_count=512,
        mean_prediction=0.001,
        mean_observation=0.0008,
        mean_error=0.0002,
        mean_absolute_error=0.012,
        root_mean_squared_error=0.018,
        interval_coverage=0.81,
        mean_interval_width=0.043,
    )


def _forecast_manifest() -> ForecastSummaryManifest:
    return ForecastSummaryManifest(
        run_id="run-001",
        generated_at=NOW,
        limitations=("Historical aggregate evidence; no prospective claim.",),
        aggregates=(_forecast(),),
    )


def _diagnostics_manifest() -> DiagnosticsManifest:
    return DiagnosticsManifest(
        run_id="run-001",
        generated_at=NOW,
        values=(
            DiagnosticValue(
                code="coverage-80",
                category=DiagnosticCategory.CALIBRATION,
                value=0.81,
                unit="fraction",
                status=DiagnosticStatus.PASS,
                detail="Measured on the declared untouched interval.",
            ),
            DiagnosticValue(
                code="ready-for-live",
                category=DiagnosticCategory.READINESS,
                value=False,
                status=DiagnosticStatus.FAIL,
                detail="No paper-trading duration or capital authorization.",
            ),
        ),
    )


def _model_card_sections(
    *, overview_text: str = "Aggregate research model evidence."
) -> tuple[ModelCardSection, ...]:
    text_by_name = {
        ModelCardSectionName.OVERVIEW: overview_text,
        ModelCardSectionName.INTENDED_USE: "Historical research and bounded offline evaluation.",
        ModelCardSectionName.OUT_OF_SCOPE_USE: "Live trading and guaranteed-profit claims.",
        ModelCardSectionName.DATA: "Versioned synthetic or licensed point-in-time inputs.",
        ModelCardSectionName.EVALUATION: "Walk-forward aggregate metrics with realistic costs.",
        ModelCardSectionName.LIMITATIONS: "No prospective paper-trading evidence.",
        ModelCardSectionName.MONITORING: "Re-evaluate drift and risk before any promotion.",
    }
    return tuple(
        ModelCardSection(
            name=name, title=name.value.replace("_", " ").title(), text=text_by_name[name]
        )
        for name in ModelCardSectionName
        if name in text_by_name
    )


def _model_card() -> ModelCardManifest:
    return ModelCardManifest(
        run_id="run-001",
        generated_at=NOW,
        card_id="card-001",
        model_name="baseline-forecast",
        model_version="1.0.0",
        sections=_model_card_sections(),
    )


@pytest.mark.parametrize(
    ("kind", "manifest"),
    [
        (ManifestKind.FORECAST_SUMMARY, _forecast_manifest()),
        (ManifestKind.DIAGNOSTICS, _diagnostics_manifest()),
        (ManifestKind.MODEL_CARD, _model_card()),
    ],
)
def test_canonical_manifest_round_trip_is_exact_and_immutable(
    kind: ManifestKind,
    manifest: ForecastSummaryManifest | DiagnosticsManifest | ModelCardManifest,
) -> None:
    payload = manifest.canonical_json_bytes()

    parsed = parse_evidence_manifest(payload, expected_kind=kind, expected_run_id="run-001")

    assert parsed == manifest
    assert parsed.canonical_json_bytes() == payload
    with pytest.raises(PydanticValidationError):
        parsed.run_id = "different-run"


def test_forecast_contract_makes_row_level_evidence_unrepresentable() -> None:
    payload = _forecast_manifest().model_dump(mode="json")
    payload["rows"] = [{"ticker": "AAPL", "prediction": 1.0}]
    with pytest.raises(PydanticValidationError, match="rows"):
        ForecastSummaryManifest.model_validate(payload)

    payload = _forecast_manifest().model_dump(mode="json")
    payload["row_level_available"] = True
    with pytest.raises(PydanticValidationError, match="row_level_available"):
        ForecastSummaryManifest.model_validate(payload)

    aggregate = _forecast().model_dump()
    aggregate["sample_count"] = 1
    with pytest.raises(PydanticValidationError, match="greater than or equal to 2"):
        ForecastAggregate.model_validate(aggregate)


def test_forecast_aggregate_enforces_temporal_and_mathematical_invariants() -> None:
    valid = _forecast().model_dump()
    valid["window_end"] = valid["window_start"] - timedelta(seconds=1)
    with pytest.raises(PydanticValidationError, match="may not precede"):
        ForecastAggregate.model_validate(valid)

    valid = _forecast().model_dump()
    valid["root_mean_squared_error"] = 0.01
    valid["mean_absolute_error"] = 0.02
    with pytest.raises(PydanticValidationError, match="may not be smaller"):
        ForecastAggregate.model_validate(valid)

    valid = _forecast().model_dump()
    valid["interval_coverage"] = None
    with pytest.raises(PydanticValidationError, match="present or absent together"):
        ForecastAggregate.model_validate(valid)


def test_forecast_cardinality_identity_and_order_are_bounded() -> None:
    accepted = tuple(
        _forecast(f"aggregate-{index:03d}") for index in range(MAX_FORECAST_AGGREGATES)
    )
    assert (
        len(
            ForecastSummaryManifest(
                run_id="run-1", generated_at=NOW, aggregates=accepted
            ).aggregates
        )
        == 100
    )
    with pytest.raises(PydanticValidationError, match="at most 100"):
        ForecastSummaryManifest(
            run_id="run-1",
            generated_at=NOW,
            aggregates=(*accepted, _forecast("aggregate-100")),
        )
    with pytest.raises(PydanticValidationError, match="unique"):
        ForecastSummaryManifest(
            run_id="run-1",
            generated_at=NOW,
            aggregates=(_forecast(), _forecast()),
        )
    with pytest.raises(PydanticValidationError, match="sorted"):
        ForecastSummaryManifest(
            run_id="run-1",
            generated_at=NOW,
            aggregates=(_forecast("aggregate-b"), _forecast("aggregate-a")),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_forecast_and_diagnostic_values_fail(value: float) -> None:
    aggregate = _forecast().model_dump()
    aggregate["mean_prediction"] = value
    with pytest.raises(PydanticValidationError):
        ForecastAggregate.model_validate(aggregate)
    with pytest.raises(PydanticValidationError):
        DiagnosticValue(
            code="nonfinite",
            category=DiagnosticCategory.UNCERTAINTY,
            value=value,
            status=DiagnosticStatus.FAIL,
        )


def test_forecast_metrics_do_not_coerce_integer_inputs() -> None:
    aggregate = _forecast().model_dump()
    aggregate["mean_prediction"] = 1
    with pytest.raises(PydanticValidationError, match="exact finite float"):
        ForecastAggregate.model_validate(aggregate)


def test_diagnostic_scalars_and_cardinality_are_closed_and_bounded() -> None:
    values = tuple(
        DiagnosticValue(
            code=f"metric-{index:03d}",
            category=DiagnosticCategory.CALIBRATION,
            value=float(index),
            status=DiagnosticStatus.INFORMATIONAL,
        )
        for index in range(MAX_DIAGNOSTIC_VALUES)
    )
    assert len(DiagnosticsManifest(run_id="run-1", generated_at=NOW, values=values).values) == 256
    with pytest.raises(PydanticValidationError, match="at most 256"):
        DiagnosticsManifest(
            run_id="run-1",
            generated_at=NOW,
            values=(
                *values,
                DiagnosticValue(
                    code="metric-256",
                    category=DiagnosticCategory.CALIBRATION,
                    value=256.0,
                    status=DiagnosticStatus.INFORMATIONAL,
                ),
            ),
        )
    raw = values[0].model_dump()
    raw["value"] = {"recursive": "mapping"}
    with pytest.raises(PydanticValidationError):
        DiagnosticValue.model_validate(raw)


def test_maximum_diagnostics_manifest_remains_canonically_parseable() -> None:
    values = tuple(
        DiagnosticValue(
            code=f"metric-{index:03d}",
            category=DiagnosticCategory.CALIBRATION,
            value="x" * 256,
            unit="normalized-standard-error-unit",
            status=DiagnosticStatus.INFORMATIONAL,
            detail="d" * 1_024,
        )
        for index in range(MAX_DIAGNOSTIC_VALUES)
    )
    manifest = DiagnosticsManifest(run_id="run-1", generated_at=NOW, values=values)
    payload = manifest.canonical_json_bytes()

    assert len(payload) > 256 * 1024
    assert (
        parse_evidence_manifest(
            payload,
            expected_kind=ManifestKind.DIAGNOSTICS,
            expected_run_id="run-1",
        )
        == manifest
    )


def test_model_card_requires_unique_canonical_structured_sections() -> None:
    sections = _model_card_sections()
    with pytest.raises(PydanticValidationError, match="required structured sections"):
        ModelCardManifest(
            run_id="run-1",
            generated_at=NOW,
            card_id="card-1",
            model_name="model-1",
            model_version="1.0.0",
            sections=sections[:-1],
        )
    with pytest.raises(PydanticValidationError, match="canonical section order"):
        ModelCardManifest(
            run_id="run-1",
            generated_at=NOW,
            card_id="card-1",
            model_name="model-1",
            model_version="1.0.0",
            sections=(sections[1], sections[0], *sections[2:]),
        )
    with pytest.raises(PydanticValidationError, match="unique"):
        ModelCardManifest(
            run_id="run-1",
            generated_at=NOW,
            card_id="card-1",
            model_name="model-1",
            model_version="1.0.0",
            sections=(*sections, sections[-1]),
        )


def test_model_card_enforces_cumulative_utf8_text_budget() -> None:
    base = _model_card()
    fixed_bytes = sum(
        len(value.encode("utf-8"))
        for value in (
            base.card_id,
            base.model_name,
            base.model_version,
            *(section.title for section in base.sections),
            *(section.text for section in base.sections[1:]),
        )
    )
    accepted = "x" * (MAX_MODEL_CARD_TEXT_BYTES - fixed_bytes)
    assert ModelCardManifest(
        run_id=base.run_id,
        generated_at=base.generated_at,
        card_id=base.card_id,
        model_name=base.model_name,
        model_version=base.model_version,
        sections=_model_card_sections(overview_text=accepted),
    )
    with pytest.raises(PydanticValidationError, match="cumulative model-card text"):
        ModelCardManifest(
            run_id=base.run_id,
            generated_at=base.generated_at,
            card_id=base.card_id,
            model_name=base.model_name,
            model_version=base.model_version,
            sections=_model_card_sections(overview_text=accepted + "x"),
        )


def _replace_canonical_field(payload: bytes, field: str, value: object) -> bytes:
    decoded = json.loads(payload)
    decoded[field] = value
    return json.dumps(
        decoded,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_parser_rejects_duplicate_keys_nonfinite_and_noncanonical_json() -> None:
    payload = _forecast_manifest().canonical_json_bytes()
    duplicate = payload[:-1] + b',"schema_version":1}'
    with pytest.raises(ManifestValidationError, match="canonical JSON"):
        parse_evidence_manifest(
            duplicate,
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
    nonfinite = payload.replace(b'"mean_prediction":0.001', b'"mean_prediction":NaN')
    with pytest.raises(ManifestValidationError, match="canonical JSON"):
        parse_evidence_manifest(
            nonfinite,
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
    with pytest.raises(ManifestValidationError, match="canonical JSON"):
        parse_evidence_manifest(
            payload + b"\n",
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )


def test_parser_rejects_wrong_kind_version_run_and_semantic_timestamp_spelling() -> None:
    payload = _forecast_manifest().canonical_json_bytes()
    with pytest.raises(ManifestValidationError, match="kind"):
        parse_evidence_manifest(
            payload,
            expected_kind=ManifestKind.DIAGNOSTICS,
            expected_run_id="run-001",
        )
    with pytest.raises(ManifestValidationError, match="kind"):
        parse_evidence_manifest(
            _replace_canonical_field(payload, "kind", "førecast_summary"),
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
    with pytest.raises(ManifestValidationError, match="version"):
        parse_evidence_manifest(
            _replace_canonical_field(payload, "schema_version", 2),
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
    with pytest.raises(ManifestValidationError, match="run identity"):
        parse_evidence_manifest(
            payload,
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-002",
        )
    alternate_utc = _replace_canonical_field(payload, "generated_at", "2026-08-09T12:00:00+00:00")
    with pytest.raises(ManifestValidationError, match="typed representation"):
        parse_evidence_manifest(
            alternate_utc,
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
    omitted_default = json.loads(payload)
    del omitted_default["row_level_available"]
    omitted_default_payload = json.dumps(
        omitted_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    with pytest.raises(ManifestValidationError, match="typed representation"):
        parse_evidence_manifest(
            omitted_default_payload,
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )


def test_parser_rejects_oversize_deep_and_wrong_root_payloads() -> None:
    oversized = b'{"padding":"' + (b"x" * (128 * 1024)) + b'"}'
    with pytest.raises(ManifestValidationError, match="canonical JSON"):
        parse_evidence_manifest(
            oversized,
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
    deep_value: object = 0
    for _ in range(17):
        deep_value = [deep_value]
    deep = json.dumps(deep_value, separators=(",", ":")).encode()
    with pytest.raises(ManifestValidationError, match="canonical JSON"):
        parse_evidence_manifest(
            deep,
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
    with pytest.raises(ManifestValidationError, match="root"):
        parse_evidence_manifest(
            b"[]",
            expected_kind=ManifestKind.FORECAST_SUMMARY,
            expected_run_id="run-001",
        )
