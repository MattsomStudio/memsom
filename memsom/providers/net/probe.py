"""Does this machine actually have IPv6, or does it merely have an IPv6 address?

`AI_ADDRCONFIG` answers the second question and reports it as the first, and that
is precisely the bug this subpackage was built for. Measured 2026-07-25: a
router-advertised **ULA** prefix (`fda5:3116:1d09::/48`, i.e. `fd00::/8` — private,
not globally routable) was enough to convince Windows it was IPv6-capable, so it
accepted an AAAA-only answer for a host and every connection attempt died at
`WinError 10051, network unreachable`. The address existed; the route never did.

So we ask the question that matters — *is there a route* — by opening a UDP
socket and connecting it. A `SOCK_DGRAM` connect sends nothing; it only forces
the kernel to select a source address and route, which fails immediately and
locally when there is none. No packet leaves the machine, so this is safe to run
on a network you are only allowed to read.

The verdict is cached: a per-request probe would be a syscall on every fetch, and
the answer changes on the timescale of plugging in a cable.
"""

from __future__ import annotations

import socket
import threading
import time

#: One of Cloudflare's public resolvers. Never contacted — it is a routing
#: target, not a correspondent. Any globally-routable address would do.
_GLOBAL_V6 = "2606:4700:4700::1111"
_GLOBAL_V4 = "1.1.1.1"

_TTL_S = 60.0
_lock = threading.Lock()
_cache: dict = {}


def _route_exists(family, target) -> bool:
    if not socket.has_ipv6 and family == socket.AF_INET6:
        return False
    sock = None
    try:
        sock = socket.socket(family, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.connect((target, 53))
        return True
    except OSError:
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _cached(key, compute, ttl=_TTL_S):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value = compute()          # computed outside the lock — never hold across I/O
    with _lock:
        _cache[key] = (now + ttl, value)
    return value


def has_global_ipv6(ttl=_TTL_S) -> bool:
    """Is there a route to the global IPv6 internet? Cached."""
    return _cached("v6", lambda: _route_exists(socket.AF_INET6, _GLOBAL_V6), ttl)


def has_global_ipv4(ttl=_TTL_S) -> bool:
    """Is there a route to the global IPv4 internet? Cached."""
    return _cached("v4", lambda: _route_exists(socket.AF_INET, _GLOBAL_V4), ttl)


def ipv6_addresses_present() -> bool:
    """Does the box hold a non-loopback, non-link-local IPv6 address?

    Only interesting next to `has_global_ipv6`: **present and unroutable is the
    ULA trap**, and it is the single most useful thing to show a user whose
    network is quietly broken.
    """
    if not socket.has_ipv6:
        return False
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None,
                                   socket.AF_INET6, socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return False
    for info in infos:
        text = str(info[4][0]).split("%", 1)[0]
        if text.startswith(("::1", "fe80")):
            continue
        if text:
            return True
    return False


def diagnosis() -> dict:
    """A snapshot for `--diagnose` and for the panel's environment check."""
    present = ipv6_addresses_present()
    routable = has_global_ipv6()
    return {
        "ipv6_addresses_present": present,
        "ipv6_global_route": routable,
        "ipv4_global_route": has_global_ipv4(),
        # The named fault, so callers do not have to re-derive it.
        "ipv6_advertised_but_unroutable": bool(present and not routable),
    }


def reset_cache() -> None:
    """Drop the cached verdicts. For tests and for an explicit re-check."""
    with _lock:
        _cache.clear()
