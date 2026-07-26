"""Dial an address we have already judged, and never let the OS pick a different one.

This is where the security property actually lives. `providers/scope.py` can only
ever hold an *opinion* — it resolves a name, decides, and then hands the name to
an HTTP client that resolves it again and connects to whatever it gets the second
time. That gap is DNS rebinding, and the old docstring conceded it could not be
closed. It closes here, by construction: `connect()` resolves, vets the whole
answer set, and opens a socket to **one of the addresses it just vetted**. There
is no second resolution to disagree with the first.

Two implementation details carry most of the weight:

* **`socket.create_connection` is never used.** It calls `getaddrinfo` on the
  host it is given, which would put the OS resolver back in the path — the exact
  thing this subpackage exists to remove. Sockets are built by hand from a vetted
  `ipaddress` object. `test_the_connector_never_touches_the_os_resolver` pins
  this by making `getaddrinfo` raise; if anyone ever "simplifies" this back, that
  is the only test that notices.
* **TLS is verified against the NAME, not the address.** `wrap_socket` gets
  `server_hostname=self.host`, so SNI and certificate validation behave exactly
  as they would have. Connecting by IP without this silently breaks HTTPS, which
  is the classic way a pinning connector gets abandoned.

Redirects re-run the whole gauntlet because each hop is a fresh connection and
therefore a fresh vet — plus an optional per-hop callback so the caller can apply
its own policy (`http_fetch` uses it to re-run `scope.check`, closing a hole
where a public URL could 302 into the panel's own loopback control plane).
"""

from __future__ import annotations

import http.client
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from memsom.providers.net import addrs, dns
from memsom.providers.net import policy as _policy
from memsom.providers.net import probe

#: http.client's own sentinel for "use the module default".
_DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT

#: urllib's default is 10 hops and any scheme the target names. Three is plenty
#: for legitimate traffic, and a hop off http(s) is how a fetch tool becomes a
#: local file reader.
MAX_REDIRECTS = 3

_shared_lock = threading.Lock()
_shared_resolver = None


class NetRefused(Exception):
    """A target was refused by policy, or could not be resolved at all.

    **Deliberately not an `OSError`.** `urllib.request.AbstractHTTPHandler.
    do_open` wraps any `OSError` escaping `connect()` in a `URLError`, so an
    `OSError` subclass here reaches the model as
    `<urlopen error <urlopen error the reason>>` — the reason survives, buried in
    two layers of noise. This message is a user-facing explanation of *why a
    target was refused*; it is worth one extra except-clause at each call site to
    keep it legible.
    """

    def __str__(self):
        return " ".join(str(a) for a in self.args) or "refused"


def shared_resolver(pol=None):
    """One resolver, so one cache, across every caller in the process."""
    global _shared_resolver
    with _shared_lock:
        if _shared_resolver is None:
            _shared_resolver = dns.StubResolver(pol or _policy.from_env())
        return _shared_resolver


def reset_shared_resolver():
    """Drop the process-wide resolver. Tests, and an explicit re-read of config."""
    global _shared_resolver
    with _shared_lock:
        _shared_resolver = None


def _ordered(addresses):
    """Addresses in dial order: a family we can actually route comes first.

    Belt to the resolver's braces. The resolver already declines to ASK for AAAA
    without a v6 route, but a hosts-file pin can still hand us one.
    """
    if probe.has_global_ipv6():
        return list(addresses)
    v4 = [a for a in addresses if a.version == 4]
    return v4 + [a for a in addresses if a.version != 4]


class PinnedHTTPConnection(http.client.HTTPConnection):
    """An `HTTPConnection` that resolves and vets before it dials."""

    def __init__(self, host, port=None, timeout=_DEFAULT_TIMEOUT,
                 source_address=None, blocksize=8192, *,
                 resolver=None, policy=None, extra_denied=(),
                 guard=True, waivers=()):
        super().__init__(host, port, timeout=timeout,
                         source_address=source_address, blocksize=blocksize)
        self._policy = policy or _policy.from_env()
        self._resolver = resolver or shared_resolver(self._policy)
        self._extra_denied = tuple(extra_denied or ())
        #: Apply the address gauntlet? True for anything the MODEL can aim.
        #: False for an operator-configured base URL — `ollama` lives on
        #: 127.0.0.1 and denying loopback there would break every local model
        #: call while protecting nobody, since nothing attacker-influenced
        #: chose that address. Same reasoning `scope.py` uses to bound
        #: destinations rather than queries.
        self._guard = bool(guard)
        #: `scope.hosts` entries. A matching entry waives the seatbelt for that
        #: target — naming a target is how you take responsibility for it.
        self._waivers = tuple(waivers or ())
        #: Set on the instance so a caller can report what was actually dialled.
        self.dialled_address = None

    # -- resolution + vetting ---------------------------------------------

    def _budget(self):
        if isinstance(self.timeout, (int, float)) and not isinstance(
                self.timeout, bool) and self.timeout > 0:
            return float(self.timeout)
        return None

    def _resolve_and_vet(self, deadline_s):
        try:
            found = self._resolver.resolve(self.host, deadline_s=deadline_s)
        except dns.DnsError as exc:
            raise NetRefused(str(exc)) from exc
        if not found:
            raise NetRefused(f"no usable address for {self.host}")
        if self._guard and not self._waived(found):
            refusal = addrs.vet(found, self.host, self._extra_denied)
            if refusal:
                raise NetRefused(refusal)
        return _ordered(found)

    def _waived(self, found) -> bool:
        """Did the operator explicitly name this target in `scope.hosts`?"""
        return any(addrs.entry_matches(entry, self.host, found)
                   for entry in self._waivers)

    def connect(self):
        if getattr(self, "_tunnel_host", None):
            # A CONNECT tunnel's target is never address-checked, so a proxy is
            # an SSRF bypass by construction. Refuse rather than pretend.
            raise NetRefused(
                "an HTTP proxy tunnel cannot be address-checked and is refused")

        started = time.monotonic()
        budget = self._budget()
        addresses = self._resolve_and_vet(budget)

        last = None
        for address in addresses:
            remaining = None
            if budget is not None:
                remaining = budget - (time.monotonic() - started)
                if remaining <= 0:
                    break
            try:
                self.sock = self._dial(address, remaining)
                self.dialled_address = address
                return
            except OSError as exc:
                last = exc                      # unreachable family, refused port
                continue

        raise NetRefused(
            f"could not connect to {self.host}: "
            f"{last if last is not None else 'no address was reachable'}")

    def _dial(self, address, timeout):
        """A socket to one vetted address. Deliberately not `create_connection`."""
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            if timeout is not None:
                sock.settimeout(timeout)
            if self.source_address:
                sock.bind(self.source_address)
            sock.connect((str(address), self.port))
            return sock
        except OSError:
            sock.close()
            raise


class PinnedHTTPSConnection(PinnedHTTPConnection):
    """The same, plus TLS verified against the hostname rather than the address."""

    default_port = http.client.HTTPS_PORT

    def __init__(self, *args, context=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._context = context or ssl.create_default_context()

    def connect(self):
        super().connect()
        # server_hostname is the whole trick: we dialled an IP, but SNI and
        # certificate validation must still see the NAME the caller asked for.
        self.sock = self._context.wrap_socket(self.sock,
                                              server_hostname=self.host)


# ---------------------------------------------------------------------------
# urllib wiring
# ---------------------------------------------------------------------------

class PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, factory, debuglevel=0):
        super().__init__(debuglevel=debuglevel)
        self._factory = factory

    def http_open(self, req):
        return self.do_open(self._factory, req)


class PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, factory, debuglevel=0):
        super().__init__(debuglevel=debuglevel)
        self._factory = factory

    def https_open(self, req):
        return self.do_open(self._factory, req)


class CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Cap the hops, refuse a scheme change, and let the caller vet each host.

    The connector re-vets every hop for free, since each is a new connection. The
    callback exists for policy the connector does not own — `http_fetch` uses it
    to re-run `scope.check`, which is what stops a public URL 302-ing into
    `http://127.0.0.1:7788/api/agents/run`.
    """

    max_redirections = MAX_REDIRECTS

    def __init__(self, on_hop=None):
        self.on_hop = on_hop

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme not in ("http", "https"):
            raise NetRefused(f"redirect to non-http(s) url refused: {newurl}")
        if self.on_hop is not None:
            refusal = self.on_hop(newurl)
            if refusal:
                raise NetRefused(f"redirect refused: {refusal}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener(*, policy=None, resolver=None, on_hop=None, context=None,
                 extra_denied=(), guard=True, waivers=(), extra_handlers=()):
    """An opener that resolves through us and has **no `ProxyHandler`**.

    urllib installs one by default and it honours `HTTP(S)_PROXY` from the
    environment. On this path that would mean vetting the proxy instead of the
    target — so it is left out deliberately rather than by omission.
    """
    import functools

    pol = policy or _policy.from_env()
    res = resolver or shared_resolver(pol)
    common = {"resolver": res, "policy": pol,
              "extra_denied": tuple(extra_denied),
              "guard": guard, "waivers": tuple(waivers)}
    http_factory = functools.partial(PinnedHTTPConnection, **common)
    https_factory = functools.partial(PinnedHTTPSConnection, context=context,
                                      **common)

    handlers = [
        PinnedHTTPHandler(http_factory),
        PinnedHTTPSHandler(https_factory),
        CheckedRedirects(on_hop),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
        urllib.request.UnknownHandler(),
        *extra_handlers,
    ]
    opener = urllib.request.OpenerDirector()
    for handler in handlers:
        opener.add_handler(handler)
    return opener


def urlopen(url, data=None, timeout=_DEFAULT_TIMEOUT, *, policy=None,
            resolver=None, on_hop=None, context=None, extra_denied=(),
            guard=True, waivers=()):
    """Drop-in for `urllib.request.urlopen`, returning the real response object.

    Returning `http.client.HTTPResponse` **unwrapped** is what keeps the
    migration to one-line edits: every provider in this repo streams with
    `with resp:` and `for line in resp:`, and chunked framing lives inside
    `http.client`. Wrapping the response in anything would put that contract at
    risk for no gain.

    `guard=False` skips the address gauntlet for an operator-configured base
    URL (a provider on 127.0.0.1); it does NOT put the OS resolver back in the
    path. `waivers` carries `scope.hosts` entries, which unbuckle the seatbelt
    per named target.

    With the kill switch off this is literally `urllib.request.urlopen`.
    """
    pol = policy or _policy.from_env()
    if not pol.enabled:
        return urllib.request.urlopen(url, data, timeout)
    opener = build_opener(policy=pol, resolver=resolver, on_hop=on_hop,
                          context=context, extra_denied=extra_denied,
                          guard=guard, waivers=waivers)
    return opener.open(url, data, timeout)


def open_configured(url, data=None, timeout=_DEFAULT_TIMEOUT, **kwargs):
    """`urlopen` for an endpoint the OPERATOR configured, not one a model chose.

    A separate name rather than `guard=False` sprinkled through thirteen call
    sites, because the distinction is the whole point and a bare boolean hides
    it. `ollama` lives on `127.0.0.1`; running the SSRF gauntlet against it would
    refuse every local model call while protecting nobody, since nothing
    attacker-influenced picked that address. Same reasoning `scope.py` uses when
    it bounds destinations rather than queries.

    What these calls still get — and the reason they were migrated at all — is
    independence from the OS resolver. `claude.py` talks to `api.anthropic.com`
    and would have died in exactly the way `http_fetch` did.

    `urlopen` keeps `guard=True` as its default deliberately: a tool added later
    that forwards a model-supplied URL and forgets the flag should fail closed.

    **Refusals are re-raised as `URLError` on this path only.** `NetRefused` is
    deliberately not an `OSError` so the guarded tool path can surface a clean
    one-line reason to the model; but every provider in this repo catches
    `(URLError, HTTPError, OSError, TimeoutError, JSONDecodeError)` and turns it
    into a `ProviderError` with a tidy message. Leaking a new exception type
    through that seam would make an unreachable local model server crash a run
    instead of reporting "ollama is down" — so here we look exactly like urllib.
    """
    kwargs.setdefault("guard", False)
    try:
        return urlopen(url, data, timeout, **kwargs)
    except NetRefused as exc:
        raise urllib.error.URLError(str(exc)) from exc
