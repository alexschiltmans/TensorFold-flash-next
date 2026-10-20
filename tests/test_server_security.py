"""The endpoint's request checks: Host, Origin, API key, body size, bodies left on a kept-alive connection, and
the refusal to listen on other machines' addresses without an allowlist and a key."""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
from http.server import ThreadingHTTPServer
from typing import Any

import pytest

from tensorfold.server.guard import ServerPolicy, check_bind, is_loopback_bind, split_host_header
from tensorfold.server.http import make_handler
from tests.test_server_openai_compat import FakeApp

CHAT = {"model": "fake-model", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4}


@pytest.fixture
def serve():
    servers: list[ThreadingHTTPServer] = []

    def start(policy: ServerPolicy | None = None, app: Any = None, handler_factory: Any = make_handler):
        app = app or FakeApp()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(app, policy))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, app

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def request(server, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    sent = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=data, headers=sent)
    response = conn.getresponse()
    text = response.read().decode()
    conn.close()
    return response.status, text, response


def raw_exchange(server, payload: bytes, *, read_timeout: float = 5.0) -> bytes:
    """Send raw bytes on one connection and read until the server closes it (or the timeout passes)."""

    with socket.create_connection(("127.0.0.1", server.server_port), timeout=read_timeout) as sock:
        sock.sendall(payload)
        chunks = []
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        except TimeoutError:
            pass
    return b"".join(chunks)


def statuses(raw: bytes) -> list[int]:
    return [int(code) for code in re.findall(rb"HTTP/1\.1 (\d{3}) ", raw)]


def http_request(method: str, path: str, port: int, body: bytes = b"", extra: str = "", close: bool = False) -> bytes:
    head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{extra}"
    if body or method == "POST":
        head += f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
    if close:
        head += "Connection: close\r\n"
    return head.encode() + b"\r\n" + body


# -- normal local clients ---------------------------------------------------------------------------------------


def test_local_client_without_origin_is_served_as_before(serve) -> None:
    server, _ = serve()
    status, body, _ = request(server, "POST", "/v1/chat/completions", CHAT)
    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "Hello"
    assert request(server, "GET", "/v1/models")[0] == 200
    assert request(server, "GET", "/health")[0] == 200
    for host in ("localhost", "[::1]"):
        assert request(server, "GET", "/v1/models", headers={"Host": f"{host}:{server.server_port}"})[0] == 200


def test_placeholder_bearer_is_ignored_without_a_key(serve) -> None:
    server, _ = serve()
    status, _, _ = request(server, "POST", "/v1/chat/completions", CHAT, headers={"Authorization": "Bearer sk-x"})
    assert status == 200


# -- Host -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["evil.example", "evil.example:{port}", "127.0.0.1:1", "localhost:1",
                                  "0.0.0.0:{port}", "127.0.0.1.evil.example:{port}", "user@127.0.0.1:{port}", ""])
def test_spoofed_host_is_refused(serve, host: str) -> None:
    server, app = serve()
    status, _, _ = request(server, "POST", "/v1/chat/completions", CHAT,
                           headers={"Host": host.format(port=server.server_port)})
    assert status == 403
    assert app.messages is None
    assert request(server, "GET", "/v1/models", headers={"Host": "evil.example"})[0] == 403


def test_missing_host_is_refused(serve) -> None:
    server, _ = serve()
    raw = raw_exchange(server, b"GET /v1/models HTTP/1.1\r\nConnection: close\r\n\r\n")
    assert statuses(raw) == [403]


def test_allowed_host_is_served(serve) -> None:
    server, _ = serve(ServerPolicy.build(allowed_hosts=["model.lan", "box.lan:9000"], environ={}))
    assert request(server, "GET", "/v1/models", headers={"Host": "model.lan"})[0] == 200
    assert request(server, "GET", "/v1/models", headers={"Host": "model.lan:443"})[0] == 200
    assert request(server, "GET", "/v1/models", headers={"Host": "box.lan:9000"})[0] == 200
    assert request(server, "GET", "/v1/models", headers={"Host": "box.lan:9001"})[0] == 403


def test_split_host_header() -> None:
    assert split_host_header("LocalHost:8080") == ("localhost", 8080)
    assert split_host_header("[::1]:8080") == ("::1", 8080)
    assert split_host_header("[::1]") == ("::1", None)
    assert split_host_header("::1") is None
    assert split_host_header("host:") is None
    assert split_host_header("host:99999") is None


# -- Origin -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://127.0.0.1:3000"])
def test_foreign_origin_is_refused(serve, origin: str) -> None:
    server, app = serve()
    status, _, response = request(server, "POST", "/v1/chat/completions", CHAT, headers={"Origin": origin})
    assert status == 403
    assert app.messages is None
    assert response.getheader("Access-Control-Allow-Origin") is None


def test_allowed_origin_is_served(serve) -> None:
    server, _ = serve(ServerPolicy.build(allowed_origins=["http://localhost:3000/"], environ={}))
    assert request(server, "GET", "/v1/models", headers={"Origin": "http://localhost:3000"})[0] == 200
    assert request(server, "GET", "/v1/models", headers={"Origin": "http://localhost:3001"})[0] == 403


def test_preflight_is_not_answered_with_cors(serve) -> None:
    server, _ = serve()
    status, _, response = request(server, "OPTIONS", "/v1/chat/completions",
                                  headers={"Origin": "https://evil.example"})
    assert status == 403
    assert response.getheader("Access-Control-Allow-Origin") is None


# -- API key ----------------------------------------------------------------------------------------------------


def test_api_key_is_enforced(serve) -> None:
    server, app = serve(ServerPolicy.build(api_key="s3cret", environ={}))
    status, _, response = request(server, "POST", "/v1/chat/completions", CHAT)
    assert status == 401
    assert response.getheader("WWW-Authenticate") == "Bearer"
    assert request(server, "POST", "/v1/chat/completions", CHAT, headers={"Authorization": "Bearer wrong"})[0] == 401
    assert request(server, "POST", "/v1/chat/completions", CHAT, headers={"Authorization": "Basic s3cret"})[0] == 401
    assert request(server, "GET", "/v1/models")[0] == 401
    assert app.messages is None
    assert request(server, "GET", "/health")[0] == 200
    good = {"Authorization": "Bearer s3cret"}
    assert request(server, "POST", "/v1/chat/completions", CHAT, headers=good)[0] == 200
    assert request(server, "GET", "/v1/models", headers=good)[0] == 200


def test_api_key_from_environment() -> None:
    assert ServerPolicy.build(environ={"TENSORFOLD_API_KEY": "k"}).api_key == "k"
    assert ServerPolicy.build(api_key="flag", environ={"TENSORFOLD_API_KEY": "k"}).api_key == "flag"
    assert ServerPolicy.build(environ={"TENSORFOLD_ALLOWED_HOSTS": "a.lan, b.lan"}).allowed_hosts == {"a.lan", "b.lan"}
    assert ServerPolicy.build(environ={"TENSORFOLD_MAX_BODY_BYTES": "100"}).max_body_bytes == 100
    with pytest.raises(ValueError):
        ServerPolicy.build(allowed_origins=["localhost:3000"], environ={})


# -- bodies -----------------------------------------------------------------------------------------------------


def test_oversized_body_gets_413_before_it_is_read(serve) -> None:
    server, app = serve(ServerPolicy.build(max_body_bytes=1024, environ={}))
    # announces 10 MiB but sends a few bytes: the reply must not wait for the rest
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {10 * 1024 * 1024}\r\n\r\n{{}}").encode()
    raw = raw_exchange(server, head)
    assert statuses(raw) == [413]
    assert b"Connection: close" in raw
    assert app.messages is None
    assert request(server, "POST", "/v1/chat/completions", CHAT)[0] == 200


def test_default_body_limit_is_16_mib() -> None:
    assert ServerPolicy().max_body_bytes == 16 * 1024 * 1024


def test_chunked_body_is_refused_and_closed(serve) -> None:
    server, app = serve()
    body = json.dumps(CHAT).encode()
    chunked = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
    smuggled = http_request("GET", "/v1/models", server.server_port, close=True)
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
            "Transfer-Encoding: chunked\r\n\r\n").encode()
    raw = raw_exchange(server, head + chunked + smuggled)
    assert statuses(raw) == [411]
    assert app.messages is None


@pytest.mark.parametrize("length", ["-1", "abc", "1, 2"])
def test_invalid_content_length_is_refused_and_closed(serve, length: str) -> None:
    server, _ = serve()
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
            f"Content-Length: {length}\r\n\r\n").encode()
    assert statuses(raw_exchange(server, head)) == [400]


def test_refused_body_does_not_leak_into_the_next_request(serve) -> None:
    # each body is itself a complete request: if it stayed on the connection the server would answer it too
    server, _ = serve(ServerPolicy.build(api_key="k", environ={}))
    port = server.server_port
    body = http_request("GET", "/v1/models", port)
    cases = {
        "unknown POST route (404)": http_request("POST", "/v1/nope", port, body,
                                                 extra="Authorization: Bearer k\r\n"),
        "unknown GET route with a body (404)": http_request("GET", "/nope", port, body,
                                                            extra="Authorization: Bearer k\r\n"),
        "foreign origin (403)": http_request("POST", "/v1/chat/completions", port, body,
                                             extra="Origin: https://evil.example\r\n"),
        "missing key (401)": http_request("POST", "/v1/chat/completions", port, body),
        "bad JSON (400)": http_request("POST", "/v1/chat/completions", port, b"{" + body,
                                       extra="Authorization: Bearer k\r\n"),
    }
    expected = {"unknown POST route (404)": 404, "unknown GET route with a body (404)": 404,
                "foreign origin (403)": 403, "missing key (401)": 401, "bad JSON (400)": 400}
    for name, first in cases.items():
        got = statuses(raw_exchange(server, first + http_request("GET", "/health", port, close=True)))
        assert got == [expected[name], 200], name


def test_spoofed_host_body_does_not_leak(serve) -> None:
    server, _ = serve()
    port = server.server_port
    inner = http_request("GET", "/v1/models", port)
    first = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: evil.example\r\nContent-Length: {len(inner)}\r\n\r\n"
             ).encode() + inner
    got = statuses(raw_exchange(server, first + http_request("GET", "/health", port, close=True)))
    assert got == [403, 200]


def test_kept_alive_connection_serves_requests_in_turn(serve) -> None:
    server, _ = serve()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        conn.request("POST", "/v1/nope", body=b'{"x": 1}', headers={"Content-Type": "application/json"})
        first = conn.getresponse()
        first.read()
        assert first.status == 404
        conn.request("POST", "/v1/chat/completions", body=json.dumps(CHAT).encode(),
                     headers={"Content-Type": "application/json"})
        second = conn.getresponse()
        assert second.status == 200
        assert json.loads(second.read())["choices"][0]["message"]["content"] == "Hello"
    finally:
        conn.close()


# -- listening address ------------------------------------------------------------------------------------------


def test_loopback_bind_needs_nothing() -> None:
    for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"):
        assert is_loopback_bind(host)
        check_bind(host, ServerPolicy())


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20"])
def test_non_loopback_bind_needs_allowlist_and_key(host: str) -> None:
    assert not is_loopback_bind(host)
    with pytest.raises(ValueError, match="--allowed-host.*--api-key"):
        check_bind(host, ServerPolicy())
    with pytest.raises(ValueError, match="--api-key"):
        check_bind(host, ServerPolicy.build(allowed_hosts=["box.lan"], environ={}))
    with pytest.raises(ValueError, match="--allowed-host"):
        check_bind(host, ServerPolicy.build(api_key="k", environ={}))
    check_bind(host, ServerPolicy.build(allowed_hosts=["box.lan"], api_key="k", environ={}))


def test_cli_refuses_non_loopback_before_loading(monkeypatch) -> None:
    from tensorfold import cli

    for name in ("TENSORFOLD_API_KEY", "TENSORFOLD_ALLOWED_HOSTS"):
        monkeypatch.delenv(name, raising=False)
    args = cli.build_parser().parse_args(["serve", "/nonexistent-model-dir", "--host", "0.0.0.0", "--no-update-check"])
    with pytest.raises(ValueError, match="other machines"):
        cli.cmd_serve(args)
    args = cli.build_parser().parse_args(["serve", "/nonexistent-model-dir", "--host", "0.0.0.0",
                                          "--allowed-host", "box.lan", "--api-key", "k", "--max-body-mib", "2"])
    policy = cli._server_policy(args)
    assert policy.max_body_bytes == 2 * 1024 * 1024 and policy.api_key == "k"


# -- the CUDA server's handler shares the checks ----------------------------------------------------------------


class FakeCudaApp:
    served = "fake-model"

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def check(self, body: dict[str, Any]) -> str | None:
        return None

    def run(self, body: dict[str, Any], chat: bool, emit: Any) -> dict[str, Any]:
        self.bodies.append(body)
        return {"final": {}, "calls": None, "finish": "stop", "content": "Hello", "reasoning": "",
                "prompt_tokens": 1, "completion_tokens": 1, "stats": {}}


def test_cuda_handler_applies_the_same_checks(serve) -> None:
    from tensorfold.cuda.server import make_handler as cuda_handler

    app = FakeCudaApp()
    server, _ = serve(ServerPolicy.build(max_body_bytes=4096, environ={}), app=app, handler_factory=cuda_handler)
    assert request(server, "POST", "/v1/chat/completions", CHAT)[0] == 200
    assert request(server, "POST", "/v1/chat/completions", CHAT, headers={"Host": "evil.example"})[0] == 403
    assert request(server, "POST", "/v1/chat/completions", CHAT, headers={"Origin": "https://evil.example"})[0] == 403
    assert request(server, "POST", "/v1/chat/completions", b"x" * 5000)[0] == 413
    port = server.server_port
    inner = http_request("GET", "/v1/models", port)
    got = statuses(raw_exchange(server, http_request("POST", "/v1/nope", port, inner)
                                + http_request("GET", "/health", port, close=True)))
    assert got == [404, 200]
    assert len(app.bodies) == 1
