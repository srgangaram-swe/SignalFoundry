"""Validated, opt-in telemetry exporter boundary.

Export is disabled unless an explicit endpoint is configured.  HTTPS is
required except for literal loopback-IP HTTP, TLS verification cannot be disabled,
and URL credentials, queries, fragments, controls, and ambiguous authorities
are rejected before any transport is constructed.  The transport is injected
behind an asynchronous protocol so network I/O remains outside request tasks
and can be bounded by :mod:`quant_platform.service.telemetry`.
"""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from typing import Final, Protocol
from urllib.parse import SplitResult, urlsplit

from quant_platform.service.telemetry_contracts import (
    MAX_TELEMETRY_RECORD_BYTES,
    TelemetryChannel,
)

MAX_EXPORT_BATCH_BYTES: Final = 256 * 1024
MAX_EXPORT_ENDPOINT_BYTES: Final = 2_048


class ExporterConfigurationError(ValueError):
    """An exporter configuration violates the local service trust boundary."""

    code = "invalid_exporter_configuration"


class ExporterPayloadError(ValueError):
    """A telemetry batch violates its fixed size or JSON-line contract."""

    code = "invalid_exporter_payload"


@dataclass(frozen=True, slots=True)
class ExporterConfig:
    """Bounded remote-export policy; ``endpoint=None`` is the safe default.

    Retries are immediate and limited to exporter attempts only.  Request and
    database operations are never retried by this configuration.
    """

    endpoint: str | None = None
    verify_tls: bool = True
    timeout_seconds: float = 0.5
    max_retries: int = 0
    queue_capacity: int = 256
    batch_size: int = 32
    shutdown_timeout_seconds: float = 2.0
    worker_termination_timeout_seconds: float = 0.25

    def __post_init__(self) -> None:
        if type(self.verify_tls) is not bool or not self.verify_tls:
            raise ExporterConfigurationError("TLS verification cannot be disabled")
        if type(self.timeout_seconds) is not float or not 0.01 <= self.timeout_seconds <= 5.0:
            raise ExporterConfigurationError("timeout_seconds must be a float in [0.01, 5.0]")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 2:
            raise ExporterConfigurationError("max_retries must be an integer in [0, 2]")
        if type(self.queue_capacity) is not int or not 1 <= self.queue_capacity <= 4_096:
            raise ExporterConfigurationError("queue_capacity must be an integer in [1, 4096]")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= min(
            128, self.queue_capacity
        ):
            raise ExporterConfigurationError(
                "batch_size must be an integer no larger than 128 or queue_capacity"
            )
        if (
            type(self.shutdown_timeout_seconds) is not float
            or not 0.01 <= self.shutdown_timeout_seconds <= 10.0
        ):
            raise ExporterConfigurationError(
                "shutdown_timeout_seconds must be a float in [0.01, 10.0]"
            )
        if (
            type(self.worker_termination_timeout_seconds) is not float
            or not 0.01 <= self.worker_termination_timeout_seconds <= 2.0
        ):
            raise ExporterConfigurationError(
                "worker_termination_timeout_seconds must be a float in [0.01, 2.0]"
            )
        if self.endpoint is not None:
            _validate_endpoint(self.endpoint)

    @property
    def enabled(self) -> bool:
        """Return whether external log/span export was explicitly configured."""

        return self.endpoint is not None


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """One bounded transport request with no headers, credentials, or redirects."""

    endpoint: str
    channel: TelemetryChannel
    payload: bytes
    timeout_seconds: float
    follow_redirects: bool = False

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint)
        if type(self.channel) is not TelemetryChannel:
            raise ExporterPayloadError("channel must be an exact TelemetryChannel")
        if type(self.payload) is not bytes or not 2 <= len(self.payload) <= MAX_EXPORT_BATCH_BYTES:
            raise ExporterPayloadError("export payload exceeds its byte bound")
        if not (self.payload.startswith(b"[") and self.payload.endswith(b"]")):
            raise ExporterPayloadError("export payload must be one bounded JSON array")
        try:
            document = json.loads(self.payload)
            canonical = json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        except (RecursionError, UnicodeError, ValueError):
            raise ExporterPayloadError("export payload contains invalid JSON") from None
        if (
            type(document) is not list
            or not 1 <= len(document) <= 128
            or any(type(record) is not dict for record in document)
            or canonical != self.payload
        ):
            raise ExporterPayloadError("export payload must be a canonical nonempty object array")
        if type(self.timeout_seconds) is not float or not 0.01 <= self.timeout_seconds <= 5.0:
            raise ExporterPayloadError("export timeout is outside the configured bound")
        if type(self.follow_redirects) is not bool or self.follow_redirects:
            raise ExporterPayloadError("telemetry export redirects must remain disabled")


class ExportTransport(Protocol):
    """Asynchronous I/O adapter owned by a telemetry worker, never a request."""

    async def send(self, request: ExportRequest) -> None:
        """Send once, honor ``follow_redirects=False``, or raise without logging secrets."""


class TelemetryExporter(Protocol):
    """Narrow batch interface consumed by the telemetry runtime."""

    async def export(
        self,
        channel: TelemetryChannel,
        records: tuple[bytes, ...],
    ) -> None:
        """Export one channel-homogeneous batch."""


class ConfiguredExporter:
    """Encode validated JSON lines and delegate to an injected async transport."""

    def __init__(self, config: ExporterConfig, transport: ExportTransport) -> None:
        if type(config) is not ExporterConfig or not config.enabled or config.endpoint is None:
            raise ExporterConfigurationError(
                "ConfiguredExporter requires an explicitly enabled ExporterConfig"
            )
        if not callable(getattr(transport, "send", None)):
            raise ExporterConfigurationError("transport must implement async send")
        self._config = config
        self._transport = transport

    async def export(
        self,
        channel: TelemetryChannel,
        records: tuple[bytes, ...],
    ) -> None:
        """Send one validated batch; endpoint and failure text are never logged."""

        if type(channel) is not TelemetryChannel:
            raise ExporterPayloadError("channel must be an exact TelemetryChannel")
        payload = encode_batch(records, maximum_records=self._config.batch_size)
        endpoint = self._config.endpoint
        if endpoint is None:  # Defensive against unsupported reflection mutation.
            raise ExporterConfigurationError("exporter endpoint is disabled")
        await self._transport.send(
            ExportRequest(
                endpoint=endpoint,
                channel=channel,
                payload=payload,
                timeout_seconds=self._config.timeout_seconds,
            )
        )


def encode_batch(records: tuple[bytes, ...], *, maximum_records: int) -> bytes:
    """Return one bounded JSON array from canonical telemetry JSON lines."""

    if type(records) is not tuple:
        raise ExporterPayloadError("records must be an immutable tuple")
    if type(maximum_records) is not int or not 1 <= maximum_records <= 128:
        raise ExporterPayloadError("maximum_records must be an integer in [1, 128]")
    if not 1 <= len(records) <= maximum_records:
        raise ExporterPayloadError("export batch has an invalid record count")
    bodies: list[bytes] = []
    observed_bytes = 2
    for record in records:
        if (
            type(record) is not bytes
            or not record.endswith(b"\n")
            or not 2 <= len(record) <= MAX_TELEMETRY_RECORD_BYTES
            or any(byte < 0x20 for byte in record[:-1])
        ):
            raise ExporterPayloadError("export batch contains an invalid JSON line")
        body = record[:-1]
        if not (body.startswith(b"{") and body.endswith(b"}")):
            raise ExporterPayloadError("export batch contains a non-object JSON line")
        try:
            document = json.loads(body)
            canonical = json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        except (RecursionError, UnicodeError, ValueError):
            raise ExporterPayloadError("export batch contains invalid JSON") from None
        if type(document) is not dict or canonical != body:
            raise ExporterPayloadError("export batch contains non-canonical object JSON")
        observed_bytes += len(body) + (1 if bodies else 0)
        if observed_bytes > MAX_EXPORT_BATCH_BYTES:
            raise ExporterPayloadError("export batch exceeds the byte ceiling")
        bodies.append(body)
    return b"[" + b",".join(bodies) + b"]"


def _validate_endpoint(endpoint: str) -> SplitResult:
    if type(endpoint) is not str:
        raise ExporterConfigurationError("export endpoint must be exact text")
    try:
        encoded = endpoint.encode("ascii")
    except (MemoryError, UnicodeEncodeError):
        raise ExporterConfigurationError("export endpoint must be bounded ASCII") from None
    if not 1 <= len(encoded) <= MAX_EXPORT_ENDPOINT_BYTES or any(
        byte <= 0x20 or byte == 0x7F or byte == 0x5C for byte in encoded
    ):
        raise ExporterConfigurationError("export endpoint contains forbidden bytes")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
        host = parsed.hostname
    except ValueError:
        raise ExporterConfigurationError("export endpoint authority is malformed") from None
    if parsed.scheme not in {"https", "http"}:
        raise ExporterConfigurationError("export endpoint must use HTTPS or loopback HTTP")
    if (
        not parsed.netloc
        or host is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ExporterConfigurationError(
            "export endpoint cannot contain credentials, query text, or a fragment"
        )
    if port is not None and not 1 <= port <= 65_535:
        raise ExporterConfigurationError("export endpoint port is invalid")
    if "%" in parsed.netloc or parsed.netloc.endswith(":") or host.endswith("."):
        raise ExporterConfigurationError("export endpoint authority is ambiguous")
    if parsed.scheme == "http" and not _is_literal_loopback(host):
        raise ExporterConfigurationError("cleartext export is limited to literal loopback")
    return parsed


def _is_literal_loopback(host: str) -> bool:
    normalized = host.lower()
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False
