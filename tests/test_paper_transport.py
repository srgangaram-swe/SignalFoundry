"""Adversarial provider and credential boundaries without network or real secrets."""

from __future__ import annotations

import http.client
import subprocess

import pytest

import signal_foundry.trading.alpaca as module
from signal_foundry.boundary import FoundryError
from signal_foundry.trading.alpaca import HTTPS, Credentials, credentials
from signal_foundry.trading.store import Journal


class Response:
    def __init__(self, status=200, body=b'{"ok":true}', encoding="identity"):
        self.status = status
        self.body = body
        self.encoding = encoding

    def getheader(self, name, default=None):
        return self.encoding if name == "Content-Encoding" else default

    def read(self, limit):
        value, self.body = self.body[:limit], self.body[limit:]
        return value


class Connection:
    def __init__(self, response):
        self.response = response
        self.sock = self
        self.calls = []
        self.closed = False

    def settimeout(self, value):
        assert 0 < value <= 5

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


@pytest.fixture
def wire(tmp_path, monkeypatch):
    journal = Journal(tmp_path / "paper")
    connection = Connection(Response())
    hosts = []

    def connect(host, **kwargs):
        hosts.append(host)
        assert kwargs["context"].check_hostname
        return connection

    monkeypatch.setattr(module.http.client, "HTTPSConnection", connect)
    transport = HTTPS(journal, Credentials("testkey", "testsecret"))
    yield transport, connection, hosts
    journal.close()


def test_fixed_tls_host_secret_headers_and_no_redirect(wire):
    transport, connection, hosts = wire
    assert transport.request("paper", "GET", "/v2/account") == {"ok": True}
    assert hosts == ["paper-api.alpaca.markets"]
    assert connection.closed
    headers = connection.calls[0][1]["headers"]
    assert headers["APCA-API-KEY-ID"] == "testkey"
    assert "testkey" not in repr(transport.keys)
    assert "testsecret" not in repr(transport.keys)


@pytest.mark.parametrize(
    "status,code",
    [
        (301, "provider_response"),
        (401, "provider_auth"),
        (403, "provider_auth"),
        (429, "provider_response"),
        (500, "provider_response"),
        (404, "provider_response"),
    ],
)
def test_http_refusal_never_retries_or_reflects_body(wire, status, code):
    transport, connection, hosts = wire
    connection.response = Response(status, b"SECRET BODY")
    with pytest.raises(FoundryError, match=code) as caught:
        transport.request("paper", "POST", "/v2/orders", body={"qty": "1"})
    assert "SECRET" not in caught.value.detail and len(hosts) == 1
    assert connection.closed


@pytest.mark.parametrize(
    "host,method,path",
    [
        ("live", "GET", "/v2/account"),
        ("paper", "PUT", "/v2/account"),
        ("data", "POST", "/v2/stocks"),
        ("paper", "GET", "https://api.alpaca.markets"),
        ("paper", "GET", "/v2/account\r\nX:secret"),
    ],
)
def test_transport_policy_refuses_before_io(wire, host, method, path):
    transport, _, hosts = wire
    with pytest.raises(FoundryError, match="transport_policy"):
        transport.request(host, method, path)
    assert hosts == []


def test_not_found_lookup_cancel_and_bounds(wire):
    transport, connection, _ = wire
    connection.response = Response(404)
    assert (
        transport.request(
            "paper",
            "GET",
            "/v2/orders:by_client_order_id",
            query={"client_order_id": "one"},
        )
        is None
    )
    connection.response = Response(204)
    assert transport.request("paper", "DELETE", "/v2/orders/one") is None
    connection.response = Response(200, encoding="gzip")
    with pytest.raises(FoundryError, match="provider_encoding"):
        transport.request("paper", "GET", "/v2/account")
    connection.response = Response(200, b"x" * ((4 << 20) + 1))
    with pytest.raises(FoundryError, match="provider_size"):
        transport.request("paper", "GET", "/v2/account")
    connection.response = Response(200, b'{"a":1,"a":2}')
    with pytest.raises(FoundryError, match="invalid_json"):
        transport.request("paper", "GET", "/v2/account")
    transport.remaining = 0
    with pytest.raises(FoundryError, match="provider_budget"):
        transport.request("paper", "GET", "/v2/account")


def test_network_failure_and_body_deadline_are_explicit(wire, monkeypatch):
    transport, connection, _ = wire

    def fail():
        raise http.client.RemoteDisconnected("fixture")

    monkeypatch.setattr(connection, "getresponse", fail)
    with pytest.raises(FoundryError, match="provider_uncertain") as caught:
        transport.request("paper", "GET", "/v2/account")
    assert isinstance(caught.value.__cause__, http.client.RemoteDisconnected)
    monkeypatch.setattr(connection, "getresponse", lambda: Response())
    transport.deadline = 2
    times = iter([1, 3])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    with pytest.raises(FoundryError, match="provider_timeout"):
        transport.request("paper", "GET", "/v2/account")


def test_keychain_is_explicit_and_no_credentials_enter_arguments(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "darwin")
    for key in module.FORBIDDEN:
        monkeypatch.delenv(key, raising=False)
    calls = []

    def read(args, **kwargs):
        calls.append(args)
        assert kwargs["stderr"] is subprocess.DEVNULL and kwargs["timeout"] == 5
        return subprocess.CompletedProcess(args, 0, b"testSecret123\n")

    monkeypatch.setattr(module.subprocess, "run", read)
    result = credentials()
    assert result.key == "testSecret123"
    assert [args[-2] for args in calls] == ["api-key-id", "api-secret-key"]
    assert all("testSecret123" not in args for args in calls)
    monkeypatch.setenv("APCA_API_KEY_ID", "must-not-leak")
    with pytest.raises(FoundryError, match="credential_policy") as caught:
        credentials()
    assert "must-not-leak" not in caught.value.detail


@pytest.mark.parametrize(
    "kind", ["missing", "nonascii", "newline", "timeout", "platform"]
)
def test_keychain_failure_modes(monkeypatch, kind):
    monkeypatch.setattr(
        module.sys, "platform", "linux" if kind == "platform" else "darwin"
    )
    for key in module.FORBIDDEN:
        monkeypatch.delenv(key, raising=False)

    def read(args, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(args, 5)
        body = b"\xff" if kind == "nonascii" else b"bad\nheader"
        return subprocess.CompletedProcess(args, 1 if kind == "missing" else 0, body)

    monkeypatch.setattr(module.subprocess, "run", read)
    with pytest.raises(FoundryError):
        credentials()
