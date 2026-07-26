"""DNS over HTTPS (RFC 8484) — for the one path where we talk to a stranger.

**Scoped deliberately.** This does not replace the resolver. The configured
nameservers stay plaintext UDP, because a home router's dnsmasq does not speak
DoH and demanding it would break every LAN name on day one. What runs over DoH
is the **public fallback** — the single place where, by design, we hand a name we
are trying to resolve to a third party we do not control, over a path we do not
control. Doing that in cleartext leaks every fallback name to anyone on the wire
*and* trusts a UDP packet nobody signed.

Three things make this small, and all three are properties of where it sits:

* **No bootstrap problem.** The endpoints are IP literals, and their certificates
  carry IP SANs — measured 2026-07-25: `1.1.1.1`'s cert lists `1.1.1.1` and
  `1.0.0.1`, `8.8.8.8`'s lists `8.8.8.8` and `8.8.4.4`. So there is no name to
  resolve before we can resolve names. Hostname endpoints are **refused** rather
  than resolved, because resolving one would mean asking the very servers that
  just failed — see `_endpoint_address`.
* **No recursion.** The request goes out through `connect` with a literal-only
  resolver wired in, so a future edit cannot accidentally route a DoH lookup back
  through the resolver that is currently in its fallback path.
* **The transaction ID is 0, on purpose.** Over UDP a random TXID and a random
  source port are the only thing an off-path forger has to beat, and `dns.py`
  spends real effort on both. Here the channel is authenticated and encrypted by
  TLS, so that entropy buys nothing, and RFC 8484 §4.1 recommends 0 because it
  makes responses cacheable by HTTP intermediaries. The question-section check
  still runs — it now guards against a confused server rather than a hostile one.

**What DoH does and does not buy.** It authenticates *the resolver*, not *the
answer*: Cloudflare could still hand us a wrong address, and without DNSSEC we
could not tell. What it removes is everyone between us and Cloudflare — the
router, the ISP, the coffee-shop AP — as parties who can read or rewrite the
exchange. Against the attacker who actually motivated this (someone on the path,
watching a fetch tool), that is the whole distance worth travelling. TLS on the
fetch itself is still what protects the payload.
"""

from __future__ import annotations

import http.client
import threading
import time
import urllib.parse

from memsom.providers.net import addrs, dns
from memsom.providers.net import policy as _policy
from memsom.providers.net import probe

#: RFC 8484 wire-format endpoints. IP literals with IP SANs — see the module
#: docstring for why that is a requirement and not a shortcut.
DEFAULT_ENDPOINTS = (
    "https://1.1.1.1/dns-query",
    "https://8.8.8.8/dns-query",
)

MEDIA_TYPE = "application/dns-message"

#: A DNS message cannot exceed 65535 bytes over TCP framing, so anything larger
#: is not a DNS message. Bounded because `read()` on a hostile endpoint is
#: otherwise an unbounded allocation.
_MAX_MESSAGE = 65535

#: RFC 8484 §4.1 — 0 is recommended, and here it costs nothing. See the docstring.
_TXID = 0

_HEADERS = {"Content-Type": MEDIA_TYPE, "Accept": MEDIA_TYPE}

#: Drop an idle connection rather than discover it dead on the next query. The
#: retry below would cover it either way; this just keeps the common case quiet.
_IDLE_S = 30.0


class DohError(dns.DnsError):
    """A DoH exchange failed. A `DnsError` so existing handling still catches it."""


class _LiteralOnly:
    """A resolver that can only answer with what it was already given.

    This is the recursion guard, expressed as a type rather than a comment. The
    DoH client dials through `connect`, and `connect` takes a resolver; wiring
    the real one in would mean a fallback lookup could re-enter the resolver that
    is *currently inside* its own fallback path. With this, that mistake cannot
    compile into a loop — it fails immediately and says why.
    """

    def resolve(self, host, want_v6=None, deadline_s=None):
        address = addrs.as_ip(host)
        if address is None:
            raise DohError(
                f"DoH endpoint {host!r} is a name, not an address — it cannot be "
                "resolved without the resolver that is already failing")
        return [address]


def _endpoint_address(url: str):
    """The literal an endpoint dials, or None if it names a host.

    Enforced at use, not at config load, so a bad entry disables one endpoint
    instead of breaking the process at import.
    """
    return addrs.as_ip(urllib.parse.urlsplit(url).hostname or "")


class _Channel:
    """One keep-alive HTTPS connection to one endpoint.

    **This is what makes DoH affordable.** Measured 2026-07-25 against
    `1.1.1.1`: a fresh connection per query costs **1375ms**, of which ~600ms is
    the TLS handshake; on a reused connection the same query is **~270ms** —
    level with the 250ms the cleartext UDP path costs. So encryption is not what
    made DoH slow, reconnecting was, and `urllib` reconnects every time. A
    defence that costs 5x gets switched off; one that costs nothing stays on.

    Serialized by a lock rather than pooled. The panel runs agents from a thread
    pool, but the public fallback only fires when the configured servers have
    failed, so contention here is rare and a pool would be complexity bought for
    a path that should be empty.
    """

    def __init__(self, url, pol):
        parts = urllib.parse.urlsplit(url)
        self.host = parts.hostname or ""
        self.port = parts.port or 443
        self.path = (parts.path or "/dns-query") + (
            "?" + parts.query if parts.query else "")
        self.policy = pol
        self._lock = threading.Lock()
        self._conn = None
        self._used = 0.0

    def _open(self, timeout):
        # Late import: `connect` imports `dns`, which imports this module.
        from memsom.providers.net import connect
        return connect.PinnedHTTPSConnection(
            self.host, self.port, timeout=timeout,
            resolver=_LiteralOnly(), policy=self.policy)

    def close(self):
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def exchange(self, body, timeout, consume):
        """Send *body*, hand the live response to *consume*, return its result.

        *consume* runs while the connection is still held, which is what lets the
        caller keep its checks in order — status, then content-type, then a
        **bounded** read — rather than being handed an already-slurped body.
        """
        with self._lock:
            if self._conn is not None and time.monotonic() - self._used > _IDLE_S:
                self.close()
            last = None
            for attempt in (0, 1):
                try:
                    if self._conn is None:
                        self._conn = self._open(timeout)
                    conn = self._conn
                    conn.timeout = timeout
                    if conn.sock is not None:
                        conn.sock.settimeout(timeout)
                    conn.request("POST", self.path, body=body, headers=_HEADERS)
                    resp = conn.getresponse()
                except Exception as exc:
                    if isinstance(exc, dns.DnsError):
                        raise                  # a refusal, not a broken socket
                    # A reused connection the far end closed looks exactly like
                    # this. Retry once on a fresh one; a second failure is real.
                    self.close()
                    last = exc
                    continue
                try:
                    return consume(resp)
                finally:
                    self._used = time.monotonic()
                    # Keep-alive requires the body to have been fully drained. If
                    # the caller stopped short — a truncated oversize read — the
                    # connection is no longer reusable.
                    if resp.will_close or not resp.isclosed():
                        self.close()
            raise DohError(f"{self.host}: {last}")


class DohClient:
    """One authenticated exchange with a public resolver. No cache, no policy."""

    def __init__(self, pol=None, endpoints=None, opener=None):
        self.policy = pol or _policy.from_env()
        self.endpoints = tuple(endpoints if endpoints is not None
                               else self.policy.doh_endpoints)
        #: Injected in tests. Signature matches `connect.urlopen`.
        self._opener = opener
        self._channels: dict = {}
        self._channels_lock = threading.Lock()

    # -- transport ---------------------------------------------------------

    def channel(self, url):
        with self._channels_lock:
            if url not in self._channels:
                self._channels[url] = _Channel(url, self.policy)
            return self._channels[url]

    def close(self):
        """Drop every kept-alive connection. Tests, and an explicit reset."""
        with self._channels_lock:
            channels, self._channels = list(self._channels.values()), {}
        for channel in channels:
            channel.close()

    def _consume(self, url, resp):
        """Status, then type, then a bounded read — in that order, deliberately.

        Classifying before reading is why a captive portal is reported as one
        instead of as "transaction id mismatch", and bounding the read is why a
        hostile endpoint cannot make us allocate until we die.
        """
        status = getattr(resp, "status", None) or getattr(resp, "code", 0)
        if status != 200:
            raise DohError(f"{url}: HTTP {status}")
        ctype = str(resp.headers.get("content-type", "")).split(";")[0]
        if ctype.strip().lower() != MEDIA_TYPE:
            raise DohError(f"{url}: not a DNS message ({ctype or 'no type'})")
        data = resp.read(_MAX_MESSAGE + 1)
        if len(data) > _MAX_MESSAGE:
            raise DohError(f"{url}: response larger than a DNS message can be")
        return data

    def _exchange(self, url, name, qtype, timeout):
        if _endpoint_address(url) is None:
            raise DohError(f"DoH endpoint must be an IP literal: {url}")
        body = dns.build_query(name, qtype, _TXID)
        consume = lambda resp: self._consume(url, resp)          # noqa: E731
        try:
            if self._opener is not None:
                with self._opener(url, body, timeout) as resp:
                    data = consume(resp)
            else:
                data = self.channel(url).exchange(body, timeout, consume)
        except dns.DnsError:
            raise
        except Exception as exc:                 # URLError, NetRefused, socket
            raise DohError(f"{url}: {exc}") from exc
        # Same parser as the UDP path, same refusals. The question-section check
        # is what still earns its keep here.
        return dns.parse_response(data, _TXID, name, qtype)

    def query(self, name, qtype, timeout=None):
        """`(rcode, addresses, ttl)` from the first endpoint that answers."""
        patience = timeout if timeout is not None else self.policy.server_timeout_s
        last = None
        for url in self.endpoints:
            try:
                rcode, found, ttl, _ = self._exchange(url, name, qtype, patience)
            except dns.DnsError as exc:
                last = exc
                _policy.note(self.policy, f"doh: {exc}")
                continue
            return rcode, found, ttl
        raise DohError(last if last is not None else "no DoH endpoint configured")


class DohResolver:
    """`resolve()` over DoH alone — the same signature as `StubResolver`.

    Usable directly (`connect.urlopen(url, resolver=DohResolver())`) for a
    deployment that wants *every* lookup authenticated and has no LAN names to
    lose. Uncached on purpose: `StubResolver` is the caching front, and two
    caches with two TTL policies is how they disagree.
    """

    def __init__(self, pol=None, endpoints=None, client=None):
        self.policy = pol or _policy.from_env()
        self.client = client or DohClient(self.policy, endpoints)

    def resolve(self, host, want_v6=None, deadline_s=None):
        name = str(host or "").strip().rstrip(".").lower()
        if not name:
            raise DohError("empty hostname")
        literal = addrs.as_ip(name)
        if literal is not None:
            return [literal]
        if want_v6 is None:
            want_v6 = probe.has_global_ipv6()

        budget = self.policy.total_deadline_s
        if isinstance(deadline_s, (int, float)) and not isinstance(deadline_s, bool):
            budget = max(0.05, min(budget, float(deadline_s)))

        _, found, _ = self.client.query(name, dns.TYPE_A, budget)
        if want_v6:
            try:
                _, v6, _ = self.client.query(name, dns.TYPE_AAAA, budget)
                found = list(found) + list(v6)
            except dns.DnsError:
                pass                             # A alone is a usable answer
        if not found:
            raise DohError(f"no address for {name}")
        return found
