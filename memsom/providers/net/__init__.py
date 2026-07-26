"""App-owned name resolution and HTTP, so a broken OS resolver cannot stop us.

The motivating failure, measured 2026-07-25: the Windows DNS client cached
AAAA-only records for a host while the configured nameserver answered A records
correctly when asked directly over UDP. `getaddrinfo(AF_UNSPEC)` returned
`WSANO_DATA`, so every `urllib` call to that host failed — while the box's own
router would have answered in 17ms. The box held a router-advertised ULA IPv6
address with no route to the global IPv6 internet, which is enough to convince
`AI_ADDRCONFIG` that IPv6 works and to make an AAAA-only answer look usable.

Owning resolution buys two things, and the second is the more valuable one:

1. **The app stops depending on the OS resolver's cache.** We read *which*
   nameserver to ask from OS config, then ask it ourselves.
2. **The address actually connected to is the address that was policy-checked.**
   `providers/scope.py` used to resolve a name, decide, and throw the addresses
   away; `urllib` then resolved again and connected to whatever it got the second
   time. That gap is DNS rebinding. The connector closes it by vetting on every
   resolution and dialling a vetted address.

**What this does NOT claim.** There is no DNSSEC, no DoT and no DoH here — this
is plaintext UDP to whatever DHCP handed us, so against an *on-path* attacker it
is no better than the OS resolver. The honest claim is: immune to a broken or
locally-poisoned OS resolver cache, and the connected address is policy-checked.
Not "immune to DNS poisoning". DoH over the connector is the obvious next step
and would let the stronger claim be made honestly.
"""

from __future__ import annotations

from memsom.providers.net.addrs import (
    DENIED,
    as_ip,
    denied_by,
    entry_matches,
    seatbelt_hit,
    unwrap,
    vet,
)
from memsom.providers.net.connect import (
    MAX_REDIRECTS,
    NetRefused,
    build_opener,
    reset_shared_resolver,
    shared_resolver,
    urlopen,
)
from memsom.providers.net.dns import DnsError, StubResolver
from memsom.providers.net.policy import (
    ENV_SWITCH,
    INTERNAL_SUFFIXES,
    PRIVATE_SUFFIXES,
    PUBLIC_FALLBACK,
    NetPolicy,
    from_env,
    note,
)

__all__ = [
    "DENIED",
    "ENV_SWITCH",
    "INTERNAL_SUFFIXES",
    "MAX_REDIRECTS",
    "PRIVATE_SUFFIXES",
    "PUBLIC_FALLBACK",
    "DnsError",
    "NetPolicy",
    "NetRefused",
    "StubResolver",
    "as_ip",
    "build_opener",
    "denied_by",
    "entry_matches",
    "from_env",
    "note",
    "reset_shared_resolver",
    "seatbelt_hit",
    "shared_resolver",
    "unwrap",
    "urlopen",
    "vet",
]
