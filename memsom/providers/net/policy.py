"""Knobs for the app-owned resolver, and the one switch that turns it all off.

Everything configurable about `memsom.providers.net` lives here so there is a
single place to audit what can change behaviour, and a single kill switch that
does not require editing a config file and restarting a server.

**The kill switch is the point of this module.** Owning DNS resolution means
owning a class of failure the OS used to absorb for us — a hosts entry we did
not reimplement, an mDNS name, a corporate split-horizon suffix. When that
happens on someone else's machine, the fix has to be one environment variable
they can set from the shell they are already in, not a support conversation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

#: Set to "off" to route every call back through `urllib.request.urlopen` and
#: `socket.getaddrinfo`, exactly as before this subpackage existed.
ENV_SWITCH = "MEMSOM_NET_STUB"

#: Queried only when every configured nameserver FAILS to answer — and, for
#: NXDOMAIN, only for public-suffix-shaped names. See `dns.py` for that split;
#: the short version is that an internal name's NXDOMAIN is the truth and a
#: public name's NXDOMAIN from a router that has lost its upstream is not.
PUBLIC_FALLBACK = ("1.1.1.1", "8.8.8.8")

#: The same two resolvers, reached over RFC 8484 instead of cleartext UDP. See
#: `doh.py` — these are IP literals because their certificates carry IP SANs,
#: which is what removes the bootstrap problem.
DOH_ENDPOINTS = ("https://1.1.1.1/dns-query", "https://8.8.8.8/dns-query")

#: Names a unicast DNS stub structurally CANNOT answer. mDNS lives on multicast
#: 5353, NetBIOS and LLMNR elsewhere again; a single-label name depends on a
#: search list we may not have read correctly. These fall through to the OS
#: resolver **with the address gauntlet still applied to whatever comes back** —
#: a documented exception rather than one discovered in the field.
#:
#: Deliberately short. `.lan` and `.corp` are NOT here: a home router's dnsmasq
#: serves `.lan` over ordinary unicast DNS, so we can and should ask it.
INTERNAL_SUFFIXES = (".local", ".localhost", ".home.arpa")

#: Names whose "does not exist" we must BELIEVE, rather than retrying against a
#: public resolver. A different question from INTERNAL_SUFFIXES above: these are
#: answerable over unicast, just never by the public internet.
#:
#: Getting this list too short is the expensive direction. `.corp`, `.home` and
#: `.mail` are the never-delegated ICANN high-risk strings, `.test`/`.example`/
#: `.invalid`/`.localhost` are reserved by RFC 6761, and `.lan`/`.intranet`/
#: `.private`/`.localdomain` are what home and office kit ships with. Retrying
#: any of them against Cloudflare both leaks an internal hostname and — under
#: split horizon — can return a real, wrong, public address.
PRIVATE_SUFFIXES = INTERNAL_SUFFIXES + (
    ".lan", ".internal", ".intranet", ".private", ".localdomain",
    ".corp", ".home", ".mail", ".test", ".example", ".invalid",
)


@dataclass(frozen=True)
class NetPolicy:
    """How to resolve and what to refuse. Immutable; derive with `replace`."""

    #: False routes everything back to the stdlib. The kill switch.
    enabled: bool = True
    #: Explicit nameservers. Empty means discover them from OS config.
    nameservers: tuple = ()
    #: Consult a public resolver when the configured servers do not answer.
    public_fallback: bool = True
    #: Reach that public resolver over DoH rather than cleartext UDP.
    doh: bool = True
    #: Where. IP literals only — see `doh.py`.
    doh_endpoints: tuple = DOH_ENDPOINTS
    #: May the public fallback DOWNGRADE to cleartext UDP when DoH fails?
    #:
    #: Off, and that is the security decision this flag exists to record. An
    #: on-path attacker who wants a forgeable exchange only has to drop our 443
    #: traffic to the resolver; if we then retried in cleartext they would get
    #: exactly what they blocked us for. The availability argument for allowing
    #: it is thin — the realistic case is a network where both ports are dead,
    #: and there the retry fails anyway. Turn it on with
    #: MEMSOM_NET_PLAINTEXT_FALLBACK=1 for a network that genuinely blocks 443
    #: to public resolvers; every use is logged.
    plaintext_public_fallback: bool = False
    #: Extra CIDRs to deny, on top of the built-in gauntlet.
    extra_denied: tuple = ()
    #: Per-server UDP wait on the SECOND pass.
    server_timeout_s: float = 2.0
    #: Per-server wait on the first pass across the server list.
    #:
    #: Measured on this box 2026-07-25: with a VPN up, three of the five
    #: configured nameservers were unreachable for port 53 and a cold lookup cost
    #: **4047ms** — three serial 2s timeouts before the live server answered in
    #: 266ms. A short first pass finds the live server quickly; anything that
    #: genuinely needs longer than this still gets the full `server_timeout_s` on
    #: the second pass, so a merely-slow resolver is not abandoned.
    first_pass_timeout_s: float = 0.6
    #: Whole-resolution ceiling, including retries and the TCP re-ask. Unlike
    #: `getaddrinfo` — which has no cap and sits outside every run budget — this
    #: is bounded, and callers subtract it from their own deadline.
    total_deadline_s: float = 5.0
    #: How long a negative answer is trusted. Deliberately short: on a
    #: multi-homed box the server list is a UNION across interfaces rather than a
    #: coherent set, so an NXDOMAIN may just mean "asked the wrong one".
    negative_ttl_s: float = 15.0
    #: Ceiling on a positive answer's TTL, so a hostile TTL cannot pin us.
    max_ttl_s: float = 600.0
    #: Suffixes handed back to the OS resolver. See INTERNAL_SUFFIXES.
    internal_suffixes: tuple = INTERNAL_SUFFIXES
    #: Suffixes whose NXDOMAIN is believed rather than retried publicly.
    private_suffixes: tuple = PRIVATE_SUFFIXES
    #: Sink for one-line operational notes (fallback used, OS handback taken).
    #: Injected so tests can assert on it without capturing logging config.
    log: object = field(default=None, compare=False)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "off", "false", "no", "")


def from_env(base: NetPolicy | None = None) -> NetPolicy:
    """*base* with any `MEMSOM_NET_*` overrides applied.

    Read at call time rather than import time so a test — or a user mid-session —
    can flip the switch without reimporting the package.
    """
    policy = base or NetPolicy()
    servers = os.environ.get("MEMSOM_NET_NAMESERVERS", "").strip()
    endpoints = os.environ.get("MEMSOM_NET_DOH_ENDPOINTS", "").strip()
    return replace(
        policy,
        enabled=_env_flag(ENV_SWITCH, policy.enabled),
        public_fallback=_env_flag("MEMSOM_NET_PUBLIC_FALLBACK",
                                  policy.public_fallback),
        doh=_env_flag("MEMSOM_NET_DOH", policy.doh),
        plaintext_public_fallback=_env_flag("MEMSOM_NET_PLAINTEXT_FALLBACK",
                                            policy.plaintext_public_fallback),
        nameservers=tuple(s.strip() for s in servers.split(",") if s.strip())
        or policy.nameservers,
        doh_endpoints=tuple(e.strip() for e in endpoints.split(",") if e.strip())
        or policy.doh_endpoints,
    )


def note(policy: NetPolicy, message: str) -> None:
    """One operational line. Never silent, never fatal.

    A fallback to public DNS, or a handback to the OS resolver, is exactly the
    kind of thing that must not happen quietly — it changes who answered the
    question.
    """
    sink = getattr(policy, "log", None)
    if sink is None:
        return
    try:
        sink(message)
    except Exception:  # a broken log sink must never break a fetch
        pass
