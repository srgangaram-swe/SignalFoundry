"""Exporter endpoint, payload, and injected-transport security tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from quant_platform.service.exporter import (
    ConfiguredExporter,
    ExporterConfig,
    ExporterConfigurationError,
    ExporterPayloadError,
    ExportRequest,
    encode_batch,
)
from quant_platform.service.telemetry_contracts import (
    LifecycleState,
    TelemetryChannel,
    lifecycle_record,
)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://telemetry.example/v1/records",
        "https://192.0.2.10:4318/v1/records",
        "http://127.0.0.1:4318/v1/records",
        "http://127.255.255.254:4318/v1/records",
        "http://[::1]:4318/v1/records",
    ],
)
def test_exporter_accepts_only_explicit_https_or_literal_loopback_http(endpoint: str) -> None:
    config = ExporterConfig(endpoint=endpoint)

    assert config.enabled is True
    assert config.endpoint == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://telemetry.example/v1/records",
        "http://localhost:4318/v1/records",
        "http://10.0.0.1/v1/records",
        "http://127.1/v1/records",
        "ftp://telemetry.example/v1/records",
        "https://user:password@telemetry.example/v1/records",
        "https://telemetry.example/v1/records?token=secret",
        "https://telemetry.example/v1/records#fragment",
        "https://telemetry.example./v1/records",
        "https://telemetry.example:/v1/records",
        "https://[::1]:/v1/records",
        "http://127.0.0.1:/v1/records",
        "https://telemetry.example\\redirect",
        "https://telemetry.example/\nforged",
        "https://t\N{LATIN SMALL LETTER E WITH ACUTE}lemetry.example/v1/records",
        "https://%65xample.com/v1/records",
        "",
    ],
)
def test_exporter_rejects_credential_ssrf_and_ambiguous_endpoint_forms(endpoint: str) -> None:
    with pytest.raises(ExporterConfigurationError):
        ExporterConfig(endpoint=endpoint)


def test_exporter_config_is_strict_and_bounded() -> None:
    assert ExporterConfig().enabled is False
    with pytest.raises(ExporterConfigurationError, match="cannot be disabled"):
        ExporterConfig(verify_tls=False)
    with pytest.raises(ExporterConfigurationError, match="float"):
        ExporterConfig(timeout_seconds=1)  # type: ignore[arg-type]
    with pytest.raises(ExporterConfigurationError, match="max_retries"):
        ExporterConfig(max_retries=3)
    with pytest.raises(ExporterConfigurationError, match="batch_size"):
        ExporterConfig(queue_capacity=2, batch_size=3)
    with pytest.raises(ExporterConfigurationError, match="shutdown_timeout"):
        ExporterConfig(shutdown_timeout_seconds=0.0)
    with pytest.raises(ExporterConfigurationError, match="worker_termination_timeout"):
        ExporterConfig(worker_termination_timeout_seconds=2.1)


def test_batch_encoding_is_deterministic_and_bounded() -> None:
    first = lifecycle_record(LifecycleState.STARTING, occurred_at_unix_ms=1).json_line()
    second = lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=2).json_line()

    encoded = encode_batch((first, second), maximum_records=2)

    assert encoded == b"[" + first[:-1] + b"," + second[:-1] + b"]"
    assert b"\n" not in encoded
    with pytest.raises(ExporterPayloadError, match="immutable tuple"):
        encode_batch([first], maximum_records=1)  # type: ignore[arg-type]
    with pytest.raises(ExporterPayloadError, match="record count"):
        encode_batch((first, second), maximum_records=1)
    with pytest.raises(ExporterPayloadError, match="invalid JSON line"):
        encode_batch((b'{"safe":"unsafe\nvalue"}\n',), maximum_records=1)
    with pytest.raises(ExporterPayloadError, match="non-object"):
        encode_batch((b"[]\n",), maximum_records=1)
    with pytest.raises(ExporterPayloadError, match="invalid JSON"):
        encode_batch((b'{"field":}\n',), maximum_records=1)
    with pytest.raises(ExporterPayloadError, match="non-canonical"):
        encode_batch((b'{"b":1,"a":2}\n',), maximum_records=1)
    with pytest.raises(ExporterPayloadError, match="non-canonical"):
        encode_batch((b'{"a":1,"a":2}\n',), maximum_records=1)


class _RecordingTransport:
    def __init__(self) -> None:
        self.requests: list[ExportRequest] = []

    async def send(self, request: ExportRequest) -> None:
        self.requests.append(request)


def test_configured_exporter_delegates_one_headerless_bounded_request() -> None:
    async def scenario() -> None:
        config = ExporterConfig(
            endpoint="https://telemetry.example/v1/records",
            timeout_seconds=0.25,
            queue_capacity=4,
            batch_size=2,
        )
        transport = _RecordingTransport()
        exporter = ConfiguredExporter(config, transport)
        record = lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1).json_line()

        await exporter.export(TelemetryChannel.LOG, (record,))

        assert len(transport.requests) == 1
        request = transport.requests[0]
        assert request.endpoint == config.endpoint
        assert request.channel is TelemetryChannel.LOG
        assert request.timeout_seconds == 0.25
        assert request.follow_redirects is False
        assert request.payload == b"[" + record[:-1] + b"]"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        b'[{"field":}]',
        b'[{"b":1,"a":2}]',
        b"[]",
        b"[1]",
    ],
)
def test_export_request_rejects_malformed_or_noncanonical_direct_payload(payload: bytes) -> None:
    with pytest.raises(ExporterPayloadError):
        ExportRequest(
            endpoint="https://telemetry.example/v1/records",
            channel=TelemetryChannel.LOG,
            payload=payload,
            timeout_seconds=0.25,
        )


def test_configured_exporter_requires_enabled_config_and_transport() -> None:
    transport = _RecordingTransport()
    with pytest.raises(ExporterConfigurationError, match="explicitly enabled"):
        ConfiguredExporter(ExporterConfig(), transport)
    with pytest.raises(ExporterConfigurationError, match="transport"):
        ConfiguredExporter(
            ExporterConfig(endpoint="https://telemetry.example/v1/records"),
            object(),  # type: ignore[arg-type]
        )


def test_export_request_revalidates_endpoint_and_payload() -> None:
    request = ExportRequest(
        endpoint="https://telemetry.example/v1/records",
        channel=TelemetryChannel.TRACE,
        payload=b"[{}]",
        timeout_seconds=0.5,
    )

    assert request.channel is TelemetryChannel.TRACE
    with pytest.raises(ExporterConfigurationError):
        replace(request, endpoint="http://telemetry.example/v1/records")
    with pytest.raises(ExporterPayloadError):
        replace(request, payload=b"not-json")
    with pytest.raises(ExporterPayloadError, match="redirects"):
        replace(request, follow_redirects=True)
