"""memsom.interface.serve -- Phase 10 remote transport (PLAN.md Sec3.5/3.6).

Preserves the MCP argv round-trip end to end (Matt's Q4, Sec3.6):

    Claude Code (Mac)  --stdio-->  memsom-mcp  --HTTPS/mesh-->  memsom serve (PC)
                                   (client mode)                       |
                                                                        v
                                                    memsom_cli.main(argv), stdout captured

The client shim builds the SAME argv the local MCP transport builds
(memsom.interface.mcp._tool_argv) and, in remote mode, ships it as JSON
instead of executing it -- one dispatch path for CLI, local MCP and remote
MCP. This module is the SERVER half: it owns sockets/TLS only; identity,
authorisation and dispatch live in memsom.interface.remote (Sec3.5).

`http.server.ThreadingHTTPServer` + `ssl` from the stdlib -- zero runtime
dependencies survives remote mode.
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import memsom
from memsom.interface import remote as memsom_remote
from memsom import tuning as memsom_tuning

DEFAULT_PORT = 8765
_ANY_ADDRESSES = frozenset(("0.0.0.0", "::", "0:0:0:0:0:0:0:0"))


# ---------------------------------------------------------------------------
# Point 1 (Sec3.5): bind to the mesh interface, never 0.0.0.0/::
# ---------------------------------------------------------------------------

def bind_ip_is_local(ip: str) -> bool:
    """True iff *ip* names an interface THIS machine actually has -- proven by
    attempting a real bind, not by parsing an OS-specific interface list."""
    if ip in _ANY_ADDRESSES:
        return False
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.bind((ip, 0))
        return True
    except OSError:
        return False


def assert_bindable(ip: str) -> None:
    """Refuses (SystemExit(1)) on 0.0.0.0/:: or any address this host does not
    actually own. Gate: a unit test that starts the server with 0.0.0.0 and
    asserts SystemExit (Sec3.5's own exit-gate line)."""
    if ip in _ANY_ADDRESSES:
        raise SystemExit(
            f"refused: --bind {ip!r} is an any-address bind. A remote memsom "
            f"server must bind to ONE mesh interface it actually owns, never "
            f"every interface on the box (Sec3.5 point 1)."
        )
    if not bind_ip_is_local(ip):
        raise SystemExit(
            f"refused: {ip!r} is not a local interface address on this machine."
        )


def discover_mesh_ip() -> str:
    """Best-effort pick of a non-loopback IPv4 address to self-check against.
    Falls back to loopback on a box with no other interface (a sandboxed CI
    runner) -- selfcheck's job is proving the bind path works, not asserting
    a mesh exists."""
    candidates = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.append(info[4][0])
    except socket.gaierror:
        pass
    for ip in candidates:
        if not ip.startswith("127.") and bind_ip_is_local(ip):
            return ip
    # UDP connect trick: no packet is sent, but the OS picks the outbound
    # interface, which is usually the mesh/LAN one even offline.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        if bind_ip_is_local(ip):
            return ip
    except OSError:
        pass
    return "127.0.0.1"


# ---------------------------------------------------------------------------
# HTTP handler -- one endpoint, POST /rpc {"tool":..., "arguments":{...}}
# ---------------------------------------------------------------------------

class RemoteHandler(BaseHTTPRequestHandler):
    server_version = "memsom-remote/1"

    def log_message(self, fmt, *args):  # noqa: A003 -- stdlib override signature
        print(f"[memsom-serve] {self.address_string()} " + (fmt % args), file=sys.stderr)

    def _token(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    def do_POST(self):  # noqa: N802 -- stdlib method name
        if self.path != "/rpc":
            self.send_error(404, "unknown endpoint")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "malformed JSON body")
            return
        tool = body.get("tool", "")
        arguments = body.get("arguments") or {}

        conn = memsom.get_connection()
        try:
            result = memsom_remote.handle_request(conn, self._token(), tool, arguments)
        finally:
            conn.close()

        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == "/features":
            import memsom.interface.features as memsom_features
            conn = memsom.get_connection(read_only=True)
            try:
                statuses = memsom_features.all_statuses(conn)
            finally:
                conn.close()
            payload = json.dumps(statuses).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404, "unknown endpoint")


def _maybe_wrap_tls(sock, server):
    """Optional self-signed TLS (Sec3.5 point 5: 'no custom crypto handshake
    ... Optionally a self-signed cert with a pinned SHA-256'). The mesh
    already encrypts the link; this is belt-and-suspenders and off by
    default."""
    cert = memsom_tuning.resolve("remote.tls_cert")
    key = memsom_tuning.resolve("remote.tls_key")
    if not cert or not key:
        return sock
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert, keyfile=key)
    return ctx.wrap_socket(sock, server_side=True)


def build_server(bind_ip: str, port: int) -> ThreadingHTTPServer:
    assert_bindable(bind_ip)
    server = ThreadingHTTPServer((bind_ip, port), RemoteHandler)
    server.socket = _maybe_wrap_tls(server.socket, server)
    return server


# ---------------------------------------------------------------------------
# --selfcheck: bind the mesh IP, assert it, close, exit 0/1
# ---------------------------------------------------------------------------

def selfcheck() -> None:
    ip = discover_mesh_ip()
    try:
        server = build_server(ip, 0)  # port 0 = OS-assigned ephemeral port
    except SystemExit as exc:
        print(f"[serve --selfcheck] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    bound_ip, bound_port = server.server_address[:2]
    server.server_close()
    assert bind_ip_is_local(ip), "discover_mesh_ip returned a non-local address"
    print(f"[serve --selfcheck] OK: bound {bound_ip}:{bound_port} "
          f"(mesh/local interface confirmed)")
    sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_serve(args):
    if args.selfcheck:
        selfcheck()
        return
    ip = args.bind or discover_mesh_ip()
    server = build_server(ip, args.port)
    print(f"[memsom-serve] listening on {ip}:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def register(sub) -> None:
    p = sub.add_parser("serve", help="run the remote memsom server (Sec3.5/3.6)")
    p.add_argument("--bind", default=None,
                   help="mesh interface IP to bind (default: auto-discovered)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--selfcheck", action="store_true",
                   help="bind the mesh IP, assert it, exit 0/1 (no serving)")
    p.set_defaults(func=cmd_serve)
