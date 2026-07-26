"""Turn whatever the attacker wrote into a real address, then judge it.

This is the "parse to binary before checking" rule, which is the difference
between a gauntlet and a string-matching ritual. `127.0.0.1` is the form nobody
attacking you will use; `2130706433`, `0x7f000001`, `0177.0.0.1`, `127.1` and
`::ffff:127.0.0.1` all reach the same socket and all sail past a check written
against the text.

Two properties worth stating because they are easy to get subtly wrong:

* **Unwrap before judging.** An IPv6 literal can carry an IPv4 address inside it
  — mapped, 6to4, Teredo, NAT64, or the deprecated compatible form. Each one is
  a way to spell loopback that a v6-only check never sees.
* **Reject the whole answer set.** If a name resolves to five addresses and one
  is denied, the name is denied. Approving the other four means an attacker with
  round-robin DNS wins by retrying, which is not a control at all.

Pure functions, no I/O — every rule here is unit-testable without a network, and
`test_provider_net_addrs.py` walks the full bypass catalog.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import socket

#: Denied unless a scope entry explicitly names the target.
#:
#: Moved here from `providers/scope.py` unchanged in meaning, with ONE addition:
#: `0.0.0.0/8`. `socket.inet_aton("0")` is `0.0.0.0`, and connecting there
#: reaches a local listener — so leaving it out left a spelling of loopback open
#: while the loopback range itself was closed. The rest of the reasoning is
#: unchanged and still lives with the scope module: the panel's own control plane
#: is on loopback, link-local is the cloud-metadata SSRF address, and the home
#: LAN is deliberately NOT here because `http_fetch` is a read-only GET and the
#: standing rule permits those.
DENIED = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
)

#: NAT64. `ipaddress` knows mapped/6to4/Teredo but not this one.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def as_ip(host: str):
    """*host* as an address, or None if it is a name.

    Tries the canonical parse first, then the legacy IPv4 forms via
    `inet_aton` — which implements exactly the decimal/octal/hex/short-form
    semantics an attacker reaches for, and which rejects anything hostname-shaped
    (`example.com`, `1.2.3.4.example.com`, `12345.com` all raise). Using the
    libc-equivalent rather than a hand-rolled parser matters: the bypass only
    works if our parser and the connect path agree, and the connect path is libc.
    """
    text = str(host or "").strip().strip("[]")
    if not text:
        return None
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        pass
    # Legacy IPv4 spellings. Guarded to digits/dots/hex so a hostname can never
    # reach inet_aton's looser acceptance (it tolerates a trailing space).
    if not text or any(c.isspace() for c in text):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(text))
    except (OSError, ValueError):
        return None


def unwrap(address):
    """Every address *address* is also a way of spelling.

    Returns a list starting with the address itself and including any IPv4
    address embedded in it. Judging only the outer form is how `::ffff:127.0.0.1`
    gets to loopback through a check that only knows `127.0.0.0/8`.
    """
    out = [address]
    if not isinstance(address, ipaddress.IPv6Address):
        return out

    for candidate in (address.ipv4_mapped, address.sixtofour):
        if candidate is not None:
            out.append(candidate)
    teredo = address.teredo
    if teredo:
        # (server, client) — the client is the interesting one, but a server
        # pointed at loopback is equally a way in. Judge both.
        out.extend(t for t in teredo if t is not None)
    if address in _NAT64:
        out.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    # ::a.b.c.d, deprecated but still routable text.
    packed = int(address)
    if 0 < packed <= 0xFFFFFFFF and address not in (
            ipaddress.IPv6Address("::1"),):
        out.append(ipaddress.IPv4Address(packed))
    return out


def denied_by(address, extra=()):
    """The network *address* is refused by, or None. Unwraps first."""
    networks = list(DENIED) + [
        n if isinstance(n, (ipaddress.IPv4Network, ipaddress.IPv6Network))
        else ipaddress.ip_network(str(n), strict=False) for n in (extra or ())
    ]
    for candidate in unwrap(address):
        for network in networks:
            if candidate.version == network.version and candidate in network:
                return network
    return None


def vet(addresses, host="", extra=()) -> str:
    """Why this whole answer set is refused, or ``""`` to allow it.

    All-or-nothing on purpose: one denied address in the set condemns the set.
    Anything else is defeated by a round-robin record and a retry loop.
    """
    if not addresses:
        return f"no usable address for {host or 'the target'}"
    for address in addresses:
        hit = denied_by(address, extra)
        if hit is not None:
            return (f"{host or address} resolves to {address} in {hit} — "
                    f"refused by default. Name it in the trigger's scope.hosts "
                    f"to allow it deliberately.")
    return ""


def seatbelt_hit(host: str, ips: list):
    """The denied network *host* lands in, or None. Literal form plus resolved."""
    for address in [as_ip(host)] + list(ips or []):
        if address is None:
            continue
        hit = denied_by(address)
        if hit is not None:
            return hit
    return None


def entry_matches(entry: str, host: str, ips: list) -> bool:
    """Does one scope entry cover *host*? CIDR by network, else glob by name."""
    entry = str(entry).strip()
    if not entry:
        return False
    try:
        network = ipaddress.ip_network(entry, strict=False)
    except ValueError:
        # a name pattern: match the literal host, case-insensitively
        return fnmatch.fnmatch(host.lower(), entry.lower())
    # A CIDR entry matches by NETWORK, never by string prefix — "10.0.0.0/24"
    # must not admit "10.0.0.50" because the text happens to start the same way.
    return any(address in network
               for address in ([as_ip(host)] + list(ips or []))
               if address is not None and address.version == network.version)
