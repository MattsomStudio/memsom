"""The connector — it must dial what it vetted, and never ask the OS to resolve.

Runs against a real loopback HTTP server rather than a mocked socket, because the
two properties under test are both about what the socket layer actually does:
that `getaddrinfo` is never consulted, and that the response object handed back is
a genuine `http.client.HTTPResponse` whose streaming contract is untouched.
"""

import http.server
import ipaddress
import socket
import threading
import urllib.error

import pytest

from memsom.providers.net import connect, dns, policy


# ---------------------------------------------------------------------------
# a loopback server
# ---------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):                                        # noqa: N802
        if self.path == "/lines":
            body = b"".join(b"line-%d\n" % i for i in range(5))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/redirect-loopback":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/panel")
            self.end_headers()
        elif self.path == "/redirect-file":
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.end_headers()
        elif self.path == "/redirect-ftp":
            self.send_response(302)
            self.send_header("Location", "ftp://example.com/x")
            self.end_headers()
        elif self.path == "/teapot":
            self.send_response(418)
            self.end_headers()
            self.wfile.write(b"nope")
        else:
            body = b"hello"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):                               # silence the server
        pass


@pytest.fixture(scope="module")
def server():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


class _FakeResolver:
    """Answers from a dict. Never opens a socket, never asks the OS."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []
        self.deadlines = []

    def resolve(self, host, want_v6=None, deadline_s=None):
        self.calls.append(host)
        self.deadlines.append(deadline_s)
        try:
            return [ipaddress.ip_address(a) for a in self.mapping[host]]
        except KeyError:
            raise dns.DnsError(f"no address for {host}")


def _open(url, **kw):
    kw.setdefault("resolver", _FakeResolver({"127.0.0.1": ["127.0.0.1"]}))
    kw.setdefault("guard", False)
    kw.setdefault("timeout", 10)
    return connect.urlopen(url, **kw)


# ---------------------------------------------------------------------------
# the load-bearing one
# ---------------------------------------------------------------------------

def test_the_connector_never_touches_the_os_resolver(server, monkeypatch):
    """If anyone ever "simplifies" `_dial` back to `socket.create_connection`,
    every other test in this repo still passes and only this one notices. That is
    the whole reason it exists — it reintroduces the exact bug the subpackage was
    built to remove."""
    def exploded(*a, **k):
        raise AssertionError("the OS resolver was consulted")

    monkeypatch.setattr(connect.socket, "getaddrinfo", exploded)
    with _open(server + "/ok") as resp:
        assert resp.status == 200
        assert resp.read() == b"hello"


def test_the_address_dialled_is_the_address_that_was_vetted(server):
    resolver = _FakeResolver({"127.0.0.1": ["127.0.0.1"]})
    opener = connect.build_opener(resolver=resolver, guard=False)
    with opener.open(server + "/ok", timeout=10) as resp:
        assert resp.status == 200
    assert resolver.calls == ["127.0.0.1"]


def test_the_socket_keeps_tcp_nodelay(server):
    """Replacing a base method silently drops what it did.

    `http.client.HTTPConnection.connect` sets `TCP_NODELAY`; `connect()` here
    overrides that method wholesale, so for a while every request in the repo ran
    with Nagle back on. `http.client` writes headers and body as separate sends,
    and Nagle makes the second wait for the first's ACK — one extra round trip on
    anything with a body. Measured 2026-07-25 on an established connection to a
    DoH endpoint: **563ms with Nagle, 313ms without**, RTT ~250ms.
    """
    conn = connect.PinnedHTTPConnection(
        "127.0.0.1", int(server.rsplit(":", 1)[1]), timeout=10,
        resolver=_FakeResolver({"127.0.0.1": ["127.0.0.1"]}), guard=False)
    conn.connect()
    try:
        assert conn.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# the streaming contract every provider depends on
# ---------------------------------------------------------------------------

def test_the_response_is_a_real_http_response_not_a_wrapper(server):
    """`net.urlopen` returns the unwrapped object on purpose — chunked framing
    lives in `http.client`, and wrapping it would put every provider's
    `for line in resp:` at risk for no gain."""
    import http.client
    with _open(server + "/lines") as resp:
        assert isinstance(resp, http.client.HTTPResponse)


def test_line_iteration_is_preserved(server):
    """`with resp:` + `for raw in resp:` is the shape ollama.py, claude.py and
    oai.py all stream with."""
    with _open(server + "/lines") as resp:
        lines = [raw.strip() for raw in resp]
    assert lines == [b"line-0", b"line-1", b"line-2", b"line-3", b"line-4"]


def test_a_non_2xx_response_still_raises_httperror_for_the_caller_to_catch(server):
    """builtins.py treats HTTPError as a RESPONSE rather than a failure. That
    only works if the handler chain still produces one."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        _open(server + "/teapot")
    assert caught.value.code == 418


# ---------------------------------------------------------------------------
# the gauntlet, at the point of connection
# ---------------------------------------------------------------------------

def test_a_name_that_resolves_to_loopback_is_refused_at_the_socket(server):
    """The rebinding case. scope.py can only hold an opinion; this is the gate."""
    resolver = _FakeResolver({"rebind.test": ["127.0.0.1"]})
    with pytest.raises(connect.NetRefused) as caught:
        connect.urlopen("http://rebind.test/x", resolver=resolver, timeout=5)
    assert "127.0.0.0/8" in str(caught.value)


def test_one_bad_address_in_the_answer_condemns_the_whole_set():
    resolver = _FakeResolver({"mixed.test": ["93.184.216.34", "127.0.0.1"]})
    with pytest.raises(connect.NetRefused):
        connect.urlopen("http://mixed.test/x", resolver=resolver, timeout=5)


def test_naming_a_target_in_scope_hosts_waives_the_seatbelt(server):
    """A seatbelt, not a wall — the user can unbuckle it on purpose, per target."""
    port = server.rsplit(":", 1)[1]
    resolver = _FakeResolver({"127.0.0.1": ["127.0.0.1"]})
    with connect.urlopen(f"http://127.0.0.1:{port}/ok", resolver=resolver,
                         timeout=10, waivers=["127.0.0.1"]) as resp:
        assert resp.status == 200


def test_a_refusal_message_is_not_buried_in_two_layers_of_urlerror():
    """NetRefused is deliberately not an OSError: do_open wraps those, and the
    reason would reach the model as `<urlopen error <urlopen error ...>>`."""
    resolver = _FakeResolver({"rebind.test": ["127.0.0.1"]})
    with pytest.raises(connect.NetRefused) as caught:
        connect.urlopen("http://rebind.test/x", resolver=resolver, timeout=5)
    assert "urlopen error" not in str(caught.value)


def test_an_unresolvable_name_says_so_rather_than_failing_generically():
    resolver = _FakeResolver({})
    with pytest.raises(connect.NetRefused) as caught:
        connect.urlopen("http://nowhere.test/x", resolver=resolver, timeout=5)
    assert "nowhere.test" in str(caught.value)


# ---------------------------------------------------------------------------
# redirects
# ---------------------------------------------------------------------------

def test_a_redirect_to_loopback_is_refused(server):
    """The hole this closes: a public URL that 302s into the panel's own control
    plane on loopback, where an agent can start runs and approve its own gates."""
    resolver = _FakeResolver({"127.0.0.1": ["127.0.0.1"]})
    with pytest.raises(connect.NetRefused):
        connect.urlopen(server + "/redirect-loopback", resolver=resolver,
                        timeout=10, waivers=[server.split("//")[1].split(":")[0]])


def test_a_redirect_to_ftp_is_refused_by_our_check(server):
    """urllib's own scheme guard permits `ftp` — it only blocks everything
    outside (http, https, ftp, ''). So ftp is exactly the hop that reaches our
    handler, and exactly the one a stdlib-only reading would leave open."""
    with pytest.raises(connect.NetRefused):
        _open(server + "/redirect-ftp")


def test_a_redirect_to_file_is_refused_before_it_reaches_us(server):
    """Refused either way — urllib raises first, so this documents the layering
    rather than pretending our handler is what stops it. A hop to file:// is how
    a fetch tool becomes a local file reader."""
    with pytest.raises((connect.NetRefused, urllib.error.HTTPError)):
        _open(server + "/redirect-file")


def test_the_per_hop_callback_can_refuse(server):
    """http_fetch uses this to re-run scope.check on the new host."""
    seen = []

    def refuse(url):
        seen.append(url)
        return "not in this run's scope"

    with pytest.raises(connect.NetRefused):
        _open(server + "/redirect-loopback", on_hop=refuse)
    assert seen, "the callback was never consulted"


def test_the_redirect_cap_is_three():
    assert connect.CheckedRedirects.max_redirections == 3


# ---------------------------------------------------------------------------
# budget, proxies, kill switch
# ---------------------------------------------------------------------------

def test_resolution_time_comes_out_of_the_callers_timeout(server):
    """`_infer_with_deadline` shrinks `params["timeout"]` to enforce a run
    budget. A resolver that spent a fixed 5s on top would silently break it."""
    resolver = _FakeResolver({"127.0.0.1": ["127.0.0.1"]})
    with connect.urlopen(server + "/ok", resolver=resolver, timeout=3,
                         guard=False) as resp:
        assert resp.status == 200
    assert resolver.deadlines == [3.0]


def test_no_proxy_handler_is_installed():
    """urllib adds one by default and it honours HTTP(S)_PROXY. On this path we
    would vet the proxy rather than the target."""
    names = [type(h).__name__ for h in connect.build_opener().handlers]
    assert not any("Proxy" in n for n in names), names


def test_a_proxy_tunnel_is_refused():
    conn = connect.PinnedHTTPConnection("example.com", 443,
                                        resolver=_FakeResolver({}))
    conn._tunnel_host = "example.com"
    with pytest.raises(connect.NetRefused):
        conn.connect()


def test_the_kill_switch_hands_everything_back_to_urllib(monkeypatch):
    """One env var, not a config edit and a restart. When someone we shipped to
    hits a name we cannot resolve, this is the escape hatch."""
    called = {}

    def fake(url, data=None, timeout=None):
        called["url"] = url
        return "stdlib"

    monkeypatch.setattr(connect.urllib.request, "urlopen", fake)
    off = policy.NetPolicy(enabled=False)
    assert connect.urlopen("http://anything.test/x", policy=off) == "stdlib"
    assert called["url"] == "http://anything.test/x"


def test_the_env_switch_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv(policy.ENV_SWITCH, "off")
    assert policy.from_env().enabled is False
    monkeypatch.setenv(policy.ENV_SWITCH, "on")
    assert policy.from_env().enabled is True


def test_the_shared_resolver_is_shared():
    """One cache per process, not one per request."""
    connect.reset_shared_resolver()
    try:
        assert connect.shared_resolver() is connect.shared_resolver()
    finally:
        connect.reset_shared_resolver()


def test_ipv4_is_dialled_first_when_there_is_no_route_to_ipv6(monkeypatch):
    """The ULA trap, at the socket layer: a hosts-file pin can still hand us an
    AAAA the resolver never asked for."""
    monkeypatch.setattr(connect.probe, "has_global_ipv6", lambda: False)
    ordered = connect._ordered([ipaddress.ip_address("2606:4700::1"),
                                ipaddress.ip_address("93.184.216.34")])
    assert ordered[0].version == 4
