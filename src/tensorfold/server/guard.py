"""Who may talk to the server, and how much they may send.

``ServerPolicy`` holds the rules; ``GuardedHandler`` applies them before any route runs:

- Host: a loopback name (``127.0.0.1``, ``localhost``, ``[::1]``) with the port the server listens on, or an
  ``allowed_hosts`` entry (``name`` matches that name on any port, ``name:port`` only that port).
  A web page that rebinds its own domain to 127.0.0.1 sends its domain as Host and is refused (403).
- Origin: a request that carries an Origin header is refused (403) unless that origin is in ``allowed_origins``
  (empty by default). Clients outside a browser send no Origin and pass. No CORS headers are sent.
- API key: when ``api_key`` is set, every route but ``GET /health`` (and ``/v1/health``) needs ``Authorization: Bearer <key>`` (401).
  Without a key the Authorization header is ignored, so clients that send a placeholder key keep working.
- Body: at most ``max_body_bytes`` (413); a body must come with a Content-Length (chunked bodies get 411).

A refused request never leaves its body on the connection: a body of known length within the limit is read and
dropped, anything else closes the connection, so the next request on a kept-alive connection starts clean.

``check_bind`` refuses to listen on a non-loopback address unless both ``allowed_hosts`` and ``api_key`` are set.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import socket
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler

DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
HEALTH_ROUTES = frozenset({"/health", "/v1/health"})
_DRAIN_TIMEOUT_S = 5.0


class BodyError(Exception):
    """A request body the server will not read: ``status`` is the reply, and the connection is closed after it."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_origin(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


@dataclass(frozen=True)
class ServerPolicy:
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    allowed_origins: frozenset[str] = field(default_factory=frozenset)
    api_key: str = ""
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES

    @classmethod
    def build(
        cls,
        *,
        allowed_hosts: list[str] | None = None,
        allowed_origins: list[str] | None = None,
        api_key: str | None = None,
        max_body_bytes: int | None = None,
        environ: dict[str, str] | None = None,
    ) -> ServerPolicy:
        """A policy from command-line values, falling back to ``TENSORFOLD_ALLOWED_HOSTS``,
        ``TENSORFOLD_ALLOWED_ORIGINS`` (comma-separated), ``TENSORFOLD_API_KEY`` and ``TENSORFOLD_MAX_BODY_BYTES``."""

        env = os.environ if environ is None else environ
        hosts = list(allowed_hosts or []) or _split_list(env.get("TENSORFOLD_ALLOWED_HOSTS", ""))
        origins = list(allowed_origins or []) or _split_list(env.get("TENSORFOLD_ALLOWED_ORIGINS", ""))
        key = api_key if api_key else env.get("TENSORFOLD_API_KEY", "")
        if max_body_bytes is None:
            raw = env.get("TENSORFOLD_MAX_BODY_BYTES", "").strip()
            max_body_bytes = int(raw) if raw else DEFAULT_MAX_BODY_BYTES
        if max_body_bytes < 1:
            raise ValueError("the request body limit must be at least 1 byte")
        for origin in origins:
            if "://" not in origin and origin.lower() != "null":
                raise ValueError(f"allowed origin {origin!r} needs a scheme, such as http://{origin}")
        return cls(
            allowed_hosts=frozenset(_host_entry(h) for h in hosts),
            allowed_origins=frozenset(_normalize_origin(o) for o in origins),
            api_key=key.strip(),
            max_body_bytes=int(max_body_bytes),
        )


def _host_entry(entry: str) -> str:
    """An allowlist entry as ``name`` or ``name:port``, lower case, IPv6 without brackets when it has no port."""

    value = entry.strip().lower()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def is_loopback_bind(host: str) -> bool:
    """True when ``host`` (a listen address) only accepts connections from this machine."""

    name = host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        pass
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(name, None)}
    except OSError:
        return False
    return bool(addresses) and all(ipaddress.ip_address(a.split("%", 1)[0]).is_loopback for a in addresses)


def check_bind(host: str, policy: ServerPolicy) -> None:
    """Raise ``ValueError`` when ``host`` is reachable from other machines and the policy does not name the Host
    values it answers to and an API key."""

    if is_loopback_bind(host):
        return
    missing = []
    if not policy.allowed_hosts:
        missing.append("--allowed-host (or TENSORFOLD_ALLOWED_HOSTS)")
    if not policy.api_key:
        missing.append("--api-key (or TENSORFOLD_API_KEY)")
    if missing:
        raise ValueError(f"--host {host} accepts connections from other machines; it needs " + " and ".join(missing))


def split_host_header(value: str) -> tuple[str, int | None] | None:
    """``(name, port)`` from a Host header (``name``, ``name:port``, ``[v6]``, ``[v6]:port``); None when malformed."""

    value = value.strip().lower()
    if not value or any(c.isspace() for c in value) or "@" in value or "/" in value:
        return None
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            return None
        name, rest = value[1:end], value[end + 1:]
        if rest and not rest.startswith(":"):
            return None
        port_text = rest[1:] if rest else ""
    else:
        if value.count(":") > 1:
            return None              # an IPv6 address must be in brackets
        name, _, port_text = value.partition(":")
        if ":" in value and not port_text:
            return None
    if not name:
        return None
    if port_text:
        if not port_text.isdigit() or not 0 < int(port_text) < 65536:
            return None
        return name, int(port_text)
    return name, None


def host_allowed(value: str | None, port: int, policy: ServerPolicy) -> bool:
    if value is None:
        return False
    parsed = split_host_header(value)
    if parsed is None:
        return False
    name, given = parsed
    if name in policy.allowed_hosts:
        return True                  # a name the operator listed, on any port (a reverse proxy may drop the port)
    if name in LOOPBACK_NAMES:
        return (given if given is not None else 80) == port
    shown = f"[{name}]" if ":" in name else name
    return given is not None and f"{shown}:{given}" in policy.allowed_hosts


def origin_allowed(value: str | None, policy: ServerPolicy) -> bool:
    return value is None or _normalize_origin(value) in policy.allowed_origins


def key_accepted(header: str | None, policy: ServerPolicy) -> bool:
    if not policy.api_key:
        return True
    if not header:
        return False
    scheme, _, token = header.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(token.strip().encode("utf-8"), policy.api_key.encode("utf-8"))


class GuardedHandler(BaseHTTPRequestHandler):
    """A request handler that checks every request against ``policy`` before ``handle_get`` / ``handle_post`` run.

    Subclasses read a body only through ``read_body()``; whatever body is left unread when the handler returns is
    dropped (or the connection closed) so it is never parsed as the next request."""

    protocol_version = "HTTP/1.1"
    policy: ServerPolicy = ServerPolicy()

    def handle_get(self) -> None:
        raise NotImplementedError

    def handle_post(self) -> None:
        raise NotImplementedError

    def route(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/")

    # -- dispatch ---------------------------------------------------------------------------------------------

    def do_GET(self) -> None:
        self._dispatch(self.handle_get)

    def do_POST(self) -> None:
        self._dispatch(self.handle_post)

    def do_HEAD(self) -> None:
        self._dispatch(None)

    def do_PUT(self) -> None:
        self._dispatch(None)

    def do_DELETE(self) -> None:
        self._dispatch(None)

    def do_PATCH(self) -> None:
        self._dispatch(None)

    def do_OPTIONS(self) -> None:
        self._dispatch(None)

    def _dispatch(self, handler) -> None:  # type: ignore[no-untyped-def]
        self._body_pending = True
        try:
            refusal = self._refusal()
            if refusal is not None:
                status, message, extra = refusal
                self.send_refusal(status, message, extra)
            elif handler is None:
                self.send_refusal(405, f"method {self.command} is not supported", {"Allow": "GET, POST"})
            else:
                handler()
        finally:
            if self._body_pending:
                self._drop_body()

    def _refusal(self) -> tuple[int, str, dict[str, str]] | None:
        port = int(self.server.server_address[1])
        if not host_allowed(self.headers.get("Host"), port, self.policy):
            return 403, "this Host is not served here", {}
        if not origin_allowed(self.headers.get("Origin"), self.policy):
            return 403, "requests from this origin are not served", {}
        exempt = self.command == "GET" and self.route() in HEALTH_ROUTES
        if not exempt and not key_accepted(self.headers.get("Authorization"), self.policy):
            return 401, "a valid API key is required", {"WWW-Authenticate": "Bearer"}
        return None

    # -- bodies -----------------------------------------------------------------------------------------------

    def _content_length(self) -> int:
        """The body's length; raises BodyError for a body the server will not read."""

        if self.headers.get("Transfer-Encoding") is not None:
            raise BodyError(411, "send the request body with a Content-Length (chunked bodies are not accepted)")
        values = self.headers.get_all("Content-Length") or []
        if not values:
            return 0
        if len({v.strip() for v in values}) != 1 or not values[0].strip().isdigit():
            raise BodyError(400, "the Content-Length header is not valid")
        length = int(values[0].strip())
        if length > self.policy.max_body_bytes:
            raise BodyError(413, f"the request body is larger than {self.policy.max_body_bytes} bytes")
        return length

    def read_body(self) -> bytes:
        """The request body (empty when there is none). Raises BodyError; ``send_refusal`` answers it."""

        self._body_pending = False
        try:
            length = self._content_length()
        except BodyError:
            self.close_connection = True
            raise
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            self.close_connection = True
            raise BodyError(400, "the request body ended early")
        return data

    def _drop_body(self) -> None:
        """Read and drop an unread body within the limit; otherwise close the connection."""

        self._body_pending = False
        try:
            length = self._content_length()
        except BodyError:
            self.close_connection = True
            return
        if not length:
            return
        try:
            self.connection.settimeout(_DRAIN_TIMEOUT_S)
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
            if remaining:
                self.close_connection = True
        except OSError:
            self.close_connection = True
        finally:
            try:
                self.connection.settimeout(self.timeout)
            except OSError:
                pass

    # -- replies ----------------------------------------------------------------------------------------------

    def send_refusal(self, status: int, message: str, extra: dict[str, str] | None = None) -> None:
        """An error reply. Unless the body can be dropped, the connection closes after it."""

        if self._body_pending:
            try:
                self._content_length()
            except BodyError:
                self.close_connection = True
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error"}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            self.close_connection = True
