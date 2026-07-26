"""DoH on the public-fallback path.

Three properties carry the security value here, and each has a test whose failure
would be silent otherwise:

* the request is a real RFC 8484 exchange and its answer goes through the **same
  parser and the same refusals** as the UDP path;
* the fallback **cannot recurse** into the resolver that is currently inside its
  own fallback, and cannot be pointed at a hostname it would have to resolve
  first;
* it **does not downgrade** to cleartext UDP when DoH fails, because an on-path
  attacker can cause that failure on purpose.

Entirely offline — the opener is injected. `test_provider_net_tls.py` is where a
real certificate gets checked.
"""

import ipaddress
import struct

import pytest

from memsom.providers.net import dns, doh
from memsom.providers.net import policy as _policy


# ---------------------------------------------------------------------------
# a fake endpoint
# ---------------------------------------------------------------------------

class _Resp:
    #: `http.client` attributes the channel reads to decide whether the
    #: connection can be kept.
    will_close = False

    def __init__(self, body, status=200, ctype=doh.MEDIA_TYPE):
        self._body = body
        self.status = status
        self.headers = {"content-type": ctype}
        self._drained = False
        #: Every limit `read` was called with. `None` means unbounded, which is
        #: the thing `test_the_body_read_is_bounded` exists to catch.
        self.read_limits = []

    def read(self, limit=None):
        self.read_limits.append(limit)
        self._drained = limit is None or limit >= len(self._body)
        return self._body if limit is None else self._body[:limit]

    def isclosed(self):
        return self._drained

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _answer(name, qtype=dns.TYPE_A, addresses=("93.184.216.34",), ttl=300,
            txid=doh._TXID, rcode=dns.RCODE_NOERROR, qname=None):
    """A wire-format response, built by hand so a test can corrupt one field."""
    question = dns.encode_name(qname or name) + struct.pack(">HH", qtype,
                                                            dns.CLASS_IN)
    body = b""
    for text in addresses:
        raw = ipaddress.ip_address(text).packed
        body += (b"\xc0\x0c" + struct.pack(">HHIH", qtype, dns.CLASS_IN, ttl,
                                           len(raw)) + raw)
    header = struct.pack(">HHHHHH", txid, 0x8180 | rcode, 1, len(addresses), 0, 0)
    return header + question + body


class _Endpoint:
    """Records what was asked and replies with whatever it was told to."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def __call__(self, request, body=None, timeout=None):
        url = getattr(request, "full_url", request)
        data = body if body is not None else getattr(request, "data", None)
        self.calls.append((url, data, timeout))
        if self.error is not None:
            raise self.error
        reply = self.reply(data) if callable(self.reply) else self.reply
        return reply


def _client(reply=None, error=None, pol=None, endpoints=None):
    endpoint = _Endpoint(reply, error)
    client = doh.DohClient(pol or _policy.NetPolicy(), endpoints,
                           opener=endpoint)
    return client, endpoint


# ---------------------------------------------------------------------------
# the exchange
# ---------------------------------------------------------------------------

def test_a_normal_lookup_returns_the_address():
    client, endpoint = _client(_Resp(_answer("example.com")))
    rcode, found, ttl = client.query("example.com", dns.TYPE_A, 5)
    assert rcode == dns.RCODE_NOERROR
    assert [str(a) for a in found] == ["93.184.216.34"]
    assert ttl == 300


def test_the_request_is_a_wire_format_post_per_rfc_8484():
    """POST + `application/dns-message` + the query as the body. The GET form
    exists too but needs base64url and buys nothing on a path we control."""
    client, endpoint = _client(_Resp(_answer("example.com")),
                               endpoints=("https://1.1.1.1/dns-query",))
    client.query("example.com", dns.TYPE_A, 5)
    url, body, _ = endpoint.calls[0]
    assert url == "https://1.1.1.1/dns-query"
    assert body == dns.build_query("example.com", dns.TYPE_A, doh._TXID)
    assert doh._HEADERS == {"Content-Type": doh.MEDIA_TYPE,
                            "Accept": doh.MEDIA_TYPE}
    assert doh._Channel("https://1.1.1.1/dns-query", None).path == "/dns-query"


def test_the_transaction_id_is_zero_on_purpose():
    """Over UDP the TXID is entropy an off-path forger must beat. Over an
    authenticated TLS channel it buys nothing, and RFC 8484 recommends 0 so
    HTTP caches can share responses."""
    assert doh._TXID == 0
    query = dns.build_query("example.com", dns.TYPE_A, doh._TXID)
    assert struct.unpack(">H", query[:2])[0] == 0


def test_an_answer_to_a_different_question_is_refused():
    """The parser check that still earns its keep here — not against a forger,
    who cannot reach inside TLS, but against a resolver that answers the wrong
    question. Accepting it would be a cache-poisoning primitive we handed
    ourselves."""
    client, _ = _client(_Resp(_answer("example.com", qname="evil.example")))
    with pytest.raises(dns.DnsError):
        client.query("example.com", dns.TYPE_A, 5)


def test_a_captive_portal_login_page_is_not_mistaken_for_dns():
    """A 200 with HTML is what airport wifi hands you.

    Note what this asserts: not that the page is *refused* — the parser rejects
    it either way, as garbage — but that it is refused **with the right reason**.
    Without the content-type check the user is told "transaction id mismatch",
    which sends them hunting a DNS bug instead of clicking "accept terms". A
    diagnostic that misdirects is worse than none."""
    page = _Resp(b"<html>sign in</html>", ctype="text/html")
    client, _ = _client(page)
    with pytest.raises(doh.DohError) as caught:
        client.query("example.com", dns.TYPE_A, 5)
    assert "not a DNS message" in str(caught.value)
    assert page.read_limits == [], "the body was parsed before being classified"


def test_a_non_200_is_refused():
    client, _ = _client(_Resp(b"", status=503))
    with pytest.raises(doh.DohError):
        client.query("example.com", dns.TYPE_A, 5)


def test_the_body_read_is_bounded():
    """`read()` with no argument on an endpoint that keeps sending is an
    unbounded allocation — the refusal downstream arrives too late to matter,
    because the memory is already gone. So this asserts the LIMIT, not the
    refusal: a DNS message cannot exceed 65535 bytes, so nothing past that ever
    needs to be in RAM."""
    body = _Resp(b"\x00" * (doh._MAX_MESSAGE + 10))
    client, _ = _client(body, endpoints=(doh.DEFAULT_ENDPOINTS[0],))
    with pytest.raises(doh.DohError) as caught:
        client.query("example.com", dns.TYPE_A, 5)
    assert body.read_limits == [doh._MAX_MESSAGE + 1], "the read was unbounded"
    assert "larger than a DNS message" in str(caught.value)


def test_the_second_endpoint_is_tried_when_the_first_fails():
    seen = []

    class _Two(doh.DohClient):
        def _exchange(self, url, name, qtype, timeout):
            seen.append(url)
            if len(seen) == 1:
                raise doh.DohError("first is down")
            return dns.RCODE_NOERROR, [ipaddress.ip_address("1.2.3.4")], 60, False

    rcode, found, _ = _Two(_policy.NetPolicy()).query("example.com",
                                                      dns.TYPE_A, 5)
    assert len(seen) == 2 and str(found[0]) == "1.2.3.4"


def test_every_endpoint_failing_raises_with_the_reason():
    client, _ = _client(error=OSError("connection reset"))
    with pytest.raises(dns.DnsError) as caught:
        client.query("example.com", dns.TYPE_A, 5)
    assert "connection reset" in str(caught.value)


# ---------------------------------------------------------------------------
# the keep-alive channel — where DoH stops being expensive
# ---------------------------------------------------------------------------

class _FakeConn:
    """Enough of `http.client.HTTPSConnection` for the channel to drive."""

    sock = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.timeout = None
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, headers))

    def getresponse(self):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        self.closed = True


def _channel(*conns):
    channel = doh._Channel("https://1.1.1.1/dns-query", _policy.NetPolicy())
    made = list(conns)
    opened = []

    def _open(timeout):
        conn = made.pop(0)
        opened.append(conn)
        return conn

    channel._open = _open
    return channel, opened


def test_the_connection_is_reused_across_queries():
    """The measurement that motivated this: a fresh connection per query cost
    **1375ms** against 1.1.1.1 (≈600ms of it the TLS handshake), while a reused
    one cost **~270ms** — level with cleartext UDP's 250ms. Encryption was never
    the expense; reconnecting was, and `urllib` reconnects every time. Lose this
    and DoH becomes the slow option someone eventually turns off."""
    conn = _FakeConn([_Resp(_answer("a.example")), _Resp(_answer("b.example"))])
    channel, opened = _channel(conn)
    for name in ("a.example", "b.example"):
        channel.exchange(b"x", 5, lambda r: r.read(99))
    assert len(opened) == 1, "a second connection was opened"
    assert len(conn.requests) == 2
    assert not conn.closed


def test_a_connection_the_far_end_closed_is_retried_once_on_a_fresh_one():
    """Keep-alive's cost: the peer may have closed an idle socket, and we only
    find out by using it. One silent retry; a second failure is real."""
    dead = _FakeConn([OSError("connection reset by peer")])
    live = _FakeConn([_Resp(_answer("example.com"))])
    channel, opened = _channel(dead, live)
    channel.exchange(b"x", 5, lambda r: r.read(99))
    assert len(opened) == 2 and dead.closed


def test_two_failures_in_a_row_are_reported_not_retried_forever():
    channel, _ = _channel(_FakeConn([OSError("down")]), _FakeConn([OSError("down")]))
    with pytest.raises(doh.DohError) as caught:
        channel.exchange(b"x", 5, lambda r: r.read(99))
    assert "down" in str(caught.value)


def test_an_undrained_response_drops_the_connection():
    """Keep-alive needs the body fully read. After a truncated oversize read the
    socket still holds the rest of that body, so reusing it would splice one
    response onto the next."""
    conn = _FakeConn([_Resp(b"\x00" * 500)])
    channel, _ = _channel(conn)
    channel.exchange(b"x", 5, lambda r: r.read(10))     # deliberately short
    assert conn.closed, "the connection was kept with a half-read body"


def test_an_idle_connection_is_dropped_before_it_is_used():
    conn = _FakeConn([_Resp(_answer("example.com"))])
    fresh = _FakeConn([_Resp(_answer("example.com"))])
    channel, opened = _channel(conn, fresh)
    channel.exchange(b"x", 5, lambda r: r.read(99))
    channel._used -= doh._IDLE_S + 1                    # pretend time passed
    channel.exchange(b"x", 5, lambda r: r.read(99))
    assert conn.closed and len(opened) == 2


# ---------------------------------------------------------------------------
# bootstrap and recursion — the structural guards
# ---------------------------------------------------------------------------

def test_a_hostname_endpoint_is_refused_not_resolved():
    """Resolving `cloudflare-dns.com` would mean asking the very servers that
    just failed. The endpoints are IP literals because their certificates carry
    IP SANs (verified against the live certs 2026-07-25), which is what removes
    the bootstrap problem entirely."""
    client, endpoint = _client(_Resp(_answer("example.com")),
                               endpoints=("https://cloudflare-dns.com/dns-query",))
    with pytest.raises(doh.DohError) as caught:
        client.query("example.com", dns.TYPE_A, 5)
    assert "IP literal" in str(caught.value)
    assert endpoint.calls == [], "it must refuse before dialling"


def test_the_default_endpoints_are_all_literals():
    for url in doh.DEFAULT_ENDPOINTS:
        assert doh._endpoint_address(url) is not None, url


def test_the_doh_request_cannot_re_enter_the_resolver():
    """The recursion guard, as a type rather than a comment: the client dials
    with a resolver that can only echo literals, so a fallback lookup can never
    loop back into the resolver that is already inside its own fallback."""
    literal = doh._LiteralOnly()
    assert [str(a) for a in literal.resolve("1.1.1.1")] == ["1.1.1.1"]
    with pytest.raises(doh.DohError):
        literal.resolve("cloudflare-dns.com")


# ---------------------------------------------------------------------------
# integration with the stub resolver — where it actually runs
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append(message)


class _FakeDoh:
    def __init__(self, addresses=("93.184.216.34",), error=None):
        self.addresses = addresses
        self.error = error
        self.queries = []

    def query(self, name, qtype, timeout=None):
        self.queries.append((name, qtype))
        if self.error is not None:
            raise self.error
        if qtype == dns.TYPE_AAAA:
            raise doh.DohError("no AAAA")
        return (dns.RCODE_NOERROR,
                [ipaddress.ip_address(a) for a in self.addresses], 120)


def _stub(pol, servers=()):
    resolver = dns.StubResolver(pol, servers=list(servers), hosts={})
    return resolver


def test_the_public_fallback_goes_over_doh():
    log = _Recorder()
    pol = _policy.NetPolicy(log=log)
    resolver = _stub(pol)                       # no configured servers at all
    resolver._doh = _FakeDoh()
    found = resolver.resolve("example.com", want_v6=False)
    assert [str(a) for a in found] == ["93.184.216.34"]
    assert resolver._doh.queries == [("example.com", dns.TYPE_A)]
    assert any("DoH" in line for line in log.lines)


def test_doh_failure_does_not_downgrade_to_cleartext():
    """THE load-bearing one. An on-path attacker who wants a forgeable exchange
    only has to drop our 443 traffic; retrying in cleartext would hand them
    exactly what they blocked us for. This must fail closed."""
    log = _Recorder()
    resolver = _stub(_policy.NetPolicy(log=log))
    resolver._doh = _FakeDoh(error=doh.DohError("443 blocked"))

    asked = []
    resolver._ask = lambda *a, **k: asked.append(a) or ([], 15.0, False)

    with pytest.raises(dns.DnsError):
        resolver.resolve("example.com", want_v6=False)
    assert asked == [], "it fell back to cleartext UDP after DoH failed"
    assert any("DoH fallback failed" in line for line in log.lines)


def test_the_downgrade_is_available_but_must_be_asked_for_and_is_logged():
    """A network that genuinely blocks 443 to public resolvers can opt in. It is
    an explicit choice with a loud line in the log, not a silent retry."""
    log = _Recorder()
    pol = _policy.NetPolicy(log=log, plaintext_public_fallback=True)
    resolver = _stub(pol)
    resolver._doh = _FakeDoh(error=doh.DohError("443 blocked"))

    asked = []

    def _fake_ask(servers, name, qtype, deadline):
        asked.append(list(servers))
        return [ipaddress.ip_address("93.184.216.34")], 60.0, False

    resolver._ask = _fake_ask
    found = resolver.resolve("example.com", want_v6=False)
    assert [str(a) for a in found] == ["93.184.216.34"]
    assert asked and asked[0] == list(_policy.PUBLIC_FALLBACK)
    assert any("DOWNGRADING" in line for line in log.lines)


def test_turning_doh_off_uses_the_cleartext_path_directly():
    pol = _policy.NetPolicy(doh=False)
    resolver = _stub(pol)
    asked = []

    def _fake_ask(servers, name, qtype, deadline):
        asked.append(list(servers))
        return [ipaddress.ip_address("93.184.216.34")], 60.0, False

    resolver._ask = _fake_ask
    resolver.resolve("example.com", want_v6=False)
    assert asked == [list(_policy.PUBLIC_FALLBACK)]


def test_a_private_suffix_is_never_sent_to_a_public_resolver_over_doh_either():
    """DoH changes who can *read* the fallback query, not whether it should
    happen. `nas.corp` still must not leave the building."""
    resolver = _stub(_policy.NetPolicy(), servers=["192.0.2.1"])
    resolver._doh = _FakeDoh()
    resolver._ask = lambda *a, **k: ([], 15.0, True)      # NXDOMAIN
    with pytest.raises(dns.DnsError):
        resolver.resolve("nas.corp", want_v6=False)
    assert resolver._doh.queries == []


# ---------------------------------------------------------------------------
# the standalone resolver
# ---------------------------------------------------------------------------

def test_the_doh_resolver_matches_the_stub_resolver_signature():
    """Swappable: `connect.urlopen(url, resolver=DohResolver())` must work."""
    resolver = doh.DohResolver(_policy.NetPolicy(), client=_FakeDoh())
    found = resolver.resolve("example.com", want_v6=False, deadline_s=2.0)
    assert [str(a) for a in found] == ["93.184.216.34"]


def test_the_doh_resolver_short_circuits_a_literal():
    client = _FakeDoh()
    resolver = doh.DohResolver(_policy.NetPolicy(), client=client)
    assert [str(a) for a in resolver.resolve("127.0.0.1")] == ["127.0.0.1"]
    assert client.queries == [], "a literal needs no lookup"


def test_the_caller_deadline_bounds_the_doh_lookup():
    """Resolution time comes OUT of the caller's budget, never on top of it —
    the same contract `_infer_with_deadline` relies on."""
    client = _FakeDoh()
    seen = []

    class _Timed(_FakeDoh):
        def query(self, name, qtype, timeout=None):
            seen.append(timeout)
            return super().query(name, qtype, timeout)

    doh.DohResolver(_policy.NetPolicy(total_deadline_s=5.0),
                    client=_Timed()).resolve("example.com", want_v6=False,
                                             deadline_s=0.75)
    assert seen == [0.75]
