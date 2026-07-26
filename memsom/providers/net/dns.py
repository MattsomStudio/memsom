"""A DNS stub resolver — the wire protocol, and the parts that must not be wrong.

Writing a resolver is easy. Writing one that cannot be lied to is the work, and
almost all of it is in *refusing* packets rather than in reading them. A resolver
that accepts whatever arrives on the right port is a cache-poisoning target with
extra steps, so every check below exists to throw something away:

* **Random transaction ID and a fresh ephemeral source port per query.** An
  off-path attacker has to guess both to land a forgery. Reusing one socket for
  every lookup would throw away the port half of that.
* **The responder must be the server we asked**, on the port we asked. Checked
  with `recvfrom`, and a mismatch is *dropped and ignored* rather than treated as
  a failure — otherwise anyone who can spray a packet at us can turn a working
  lookup into an error.
* **The question section must come back identical.** The classic forgery is a
  reply carrying an answer for a name nobody asked about.
* **Compression pointers may only point backwards, never forwards, never twice
  to the same offset, and never more than `_MAX_JUMPS` times.** A pointer loop is
  the textbook parser hang, and it arrives as a well-formed packet.
* **Every length is bounded before it is used** — label ≤63, name ≤255, RDLENGTH
  against what is actually left in the buffer.

**What this deliberately is not.** No DNSSEC, no DoT, no DoH: this is plaintext
UDP to whatever DHCP handed us, so an *on-path* attacker is no worse off than
they were against the OS resolver. What it buys is independence from the OS
resolver's cache — which is the thing that actually broke — and a resolution
result we can hand straight to the connector so policy is applied to the address
we truly dial. DoH over `net/connect.py` is the natural next step and is what
would earn the stronger claim.
"""

from __future__ import annotations

import ipaddress
import random
import socket
import struct
import threading
import time

from memsom.providers.net import addrs, nameservers, policy as _policy, probe

#: Record types we ask for. Everything else is parsed past, never interpreted.
TYPE_A = 1
TYPE_AAAA = 28
CLASS_IN = 1

RCODE_NOERROR = 0
RCODE_SERVFAIL = 2
RCODE_NXDOMAIN = 3
RCODE_REFUSED = 5

#: A legitimate name needs a handful of pointer follows at most. Anything past
#: this is a packet built to make us spin.
_MAX_JUMPS = 16
_MAX_NAME = 255
_MAX_LABEL = 63
_UDP_READ = 4096

_rng = random.SystemRandom()


class DnsError(Exception):
    """A query could not be answered. Carries a human-usable reason."""


# ---------------------------------------------------------------------------
# wire codec
# ---------------------------------------------------------------------------

def encode_name(name: str) -> bytes:
    out = bytearray()
    for label in str(name).rstrip(".").split("."):
        raw = label.encode("idna") if any(ord(c) > 127 for c in label) \
            else label.encode("ascii", "strict")
        if not 0 < len(raw) <= _MAX_LABEL:
            raise DnsError(f"bad label in {name!r}")
        out.append(len(raw))
        out += raw
    out.append(0)
    if len(out) > _MAX_NAME:
        raise DnsError(f"name too long: {name!r}")
    return bytes(out)


def decode_name(buf: bytes, offset: int):
    """`(name, offset_after)` — the hardened one.

    `offset_after` is the position following the name *as it appeared at
    `offset`*, which is not where parsing ended if a pointer was followed. Every
    caller needs the former to keep walking the packet; conflating the two is how
    a compressed record silently desynchronises the whole parse.
    """
    labels: list = []
    visited: set = set()
    jumps = 0
    total = 0
    pos = offset
    after = None

    while True:
        if pos >= len(buf):
            raise DnsError("name runs past the end of the packet")
        length = buf[pos]

        if length == 0:
            pos += 1
            if after is None:
                after = pos
            break

        if length & 0xC0 == 0xC0:
            if pos + 1 >= len(buf):
                raise DnsError("truncated compression pointer")
            target = ((length & 0x3F) << 8) | buf[pos + 1]
            if after is None:
                after = pos + 2
            # Backwards-only plus a visited set makes a loop unrepresentable;
            # the jump cap bounds even a legal chain.
            if target >= pos:
                raise DnsError("forward compression pointer")
            if target in visited:
                raise DnsError("compression pointer loop")
            visited.add(target)
            jumps += 1
            if jumps > _MAX_JUMPS:
                raise DnsError("too many compression pointers")
            pos = target
            continue

        if length > _MAX_LABEL:
            raise DnsError("oversize label")
        pos += 1
        if pos + length > len(buf):
            raise DnsError("label runs past the end of the packet")
        total += length + 1
        if total > _MAX_NAME:
            raise DnsError("name too long")
        labels.append(buf[pos:pos + length])
        pos += length

    name = b".".join(labels).decode("ascii", "replace").lower()
    return name, after


def build_query(name: str, qtype: int, txid: int) -> bytes:
    header = struct.pack(">HHHHHH", txid, 0x0100, 1, 0, 0, 0)  # RD=1
    return header + encode_name(name) + struct.pack(">HH", qtype, CLASS_IN)


def parse_response(data: bytes, txid: int, name: str, qtype: int):
    """`(rcode, addresses, min_ttl, truncated)`, or raise `DnsError`.

    Raising means "this packet is not an answer to my question" — the caller
    keeps waiting rather than giving up, because giving up on an unsolicited
    packet is a denial-of-service anyone on the path can trigger.
    """
    if len(data) < 12:
        raise DnsError("short header")
    rid, flags, qdcount, ancount, _, _ = struct.unpack(">HHHHHH", data[:12])
    if rid != txid:
        raise DnsError("transaction id mismatch")
    if not flags & 0x8000:
        raise DnsError("not a response")
    rcode = flags & 0x000F
    truncated = bool(flags & 0x0200)

    pos = 12
    wanted = str(name).rstrip(".").lower()
    if qdcount != 1:
        raise DnsError("unexpected question count")
    qname, pos = decode_name(data, pos)
    if pos + 4 > len(data):
        raise DnsError("truncated question")
    qtype_seen, qclass_seen = struct.unpack(">HH", data[pos:pos + 4])
    pos += 4
    # The forgery this blocks: a reply that answers a question we never asked.
    if qname != wanted or qtype_seen != qtype or qclass_seen != CLASS_IN:
        raise DnsError("question section does not match the query")

    found: list = []
    min_ttl = None
    for _ in range(ancount):
        try:
            _, pos = decode_name(data, pos)
            if pos + 10 > len(data):
                break
            rtype, rclass, ttl, rdlength = struct.unpack(">HHIH",
                                                        data[pos:pos + 10])
            pos += 10
            if pos + rdlength > len(data):
                break                      # bounded: never read past the buffer
            rdata = data[pos:pos + rdlength]
            pos += rdlength
        except DnsError:
            break                          # a malformed record ends the walk
        if rclass != CLASS_IN:
            continue
        if rtype == TYPE_A and rdlength == 4:
            found.append(ipaddress.IPv4Address(rdata))
        elif rtype == TYPE_AAAA and rdlength == 16:
            found.append(ipaddress.IPv6Address(rdata))
        else:
            continue
        min_ttl = ttl if min_ttl is None else min(min_ttl, ttl)

    return rcode, found, min_ttl, truncated


# ---------------------------------------------------------------------------
# the resolver
# ---------------------------------------------------------------------------

def _family_of(server: str):
    address = addrs.as_ip(server)
    if address is None:
        raise DnsError(f"nameserver {server!r} is not an address")
    return (socket.AF_INET6 if address.version == 6 else socket.AF_INET)


def looks_public(name: str, internal_suffixes=()) -> bool:
    """Is this a name the public DNS could plausibly answer?

    Gates the one dangerous fallback. An NXDOMAIN for `nas.lan` is the truth and
    must be believed; an NXDOMAIN for `github.com` from a router that has lost
    its upstream is not, and asking a public resolver is the right move. Getting
    this backwards either leaks internal names or reproduces the outage.
    """
    text = str(name or "").strip().rstrip(".").lower()
    if not text or "." not in text:
        return False                       # single label: never public
    if any(text.endswith(s) for s in internal_suffixes):
        return False
    tld = text.rsplit(".", 1)[-1]
    return len(tld) >= 2 and tld.isalpha()


class StubResolver:
    """Name -> addresses, asking the configured servers ourselves.

    Thread-safe: the panel serves agent runs from a thread pool, and the cache is
    shared. One lock, never held across I/O.
    """

    def __init__(self, pol=None, servers=None, hosts=None):
        self.policy = pol or _policy.from_env()
        self._servers = list(servers) if servers is not None else None
        self._hosts = hosts
        self._cache: dict = {}
        self._lock = threading.Lock()
        #: The last server that answered anything. See `_preferred_order`.
        self._preferred = None

    # -- configuration -----------------------------------------------------

    def servers(self) -> list:
        if self._servers is not None:
            return list(self._servers)
        configured = list(self.policy.nameservers) or nameservers.system_nameservers()
        return configured

    def hosts(self) -> dict:
        if self._hosts is not None:
            return self._hosts
        return nameservers.hosts_entries()

    # -- cache -------------------------------------------------------------

    def _cached(self, key):
        with self._lock:
            hit = self._cache.get(key)
        if hit is None or hit[0] <= time.monotonic():
            return None
        return hit[1]

    def _store(self, key, value, ttl):
        ttl = max(0.0, min(float(ttl), self.policy.max_ttl_s))
        with self._lock:
            self._cache[key] = (time.monotonic() + ttl, value)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    # -- one exchange ------------------------------------------------------

    def _exchange_udp(self, server, name, qtype, deadline, wait=None):
        txid = _rng.randrange(0, 65536)
        query = build_query(name, qtype, txid)
        # A fresh socket per query means a fresh ephemeral source port, which is
        # half the entropy an off-path forger has to guess.
        sock = socket.socket(_family_of(server), socket.SOCK_DGRAM)
        patience = wait if wait is not None else self.policy.server_timeout_s
        try:
            sock.sendto(query, (server, 53))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DnsError(f"{server} timed out")
                sock.settimeout(min(remaining, patience))
                try:
                    data, peer = sock.recvfrom(_UDP_READ)
                except socket.timeout as exc:
                    raise DnsError(f"{server} timed out") from exc
                # Wrong sender: drop it and keep waiting. Treating it as failure
                # would let anyone who can spray one packet break our lookups.
                if str(peer[0]).split("%", 1)[0] != str(server) or peer[1] != 53:
                    continue
                try:
                    return parse_response(data, txid, name, qtype)
                except DnsError:
                    continue               # not our answer; keep waiting
        finally:
            sock.close()

    def _exchange_tcp(self, server, name, qtype, deadline):
        """Used when a UDP reply sets TC. Same checks, framed differently."""
        txid = _rng.randrange(0, 65536)
        query = build_query(name, qtype, txid)
        remaining = max(0.01, deadline - time.monotonic())
        sock = socket.create_connection((server, 53), timeout=remaining)
        try:
            sock.sendall(struct.pack(">H", len(query)) + query)
            header = _recv_exactly(sock, 2)
            (length,) = struct.unpack(">H", header)
            return parse_response(_recv_exactly(sock, length), txid, name, qtype)
        finally:
            sock.close()

    def _preferred_order(self, servers):
        """Whoever answered last, first.

        The server list is a union across every interface, and on a machine with
        a VPN or a second NIC most of it is usually dead — measured here, three of
        five. Without this, every cold lookup re-walks the graveyard.
        """
        first = self._preferred
        if first and first in servers:
            return [first] + [s for s in servers if s != first]
        return list(servers)

    def _ask(self, servers, name, qtype, deadline):
        """`(addresses, ttl, saw_nxdomain)` from the first server that answers.

        Two passes over the list: a short-patience sweep to find a live server
        quickly, then a full-timeout sweep so a merely-slow-but-working resolver
        is never abandoned. A dead server costs `first_pass_timeout_s`, not
        `server_timeout_s`, which is the difference between a 4s and a sub-second
        cold lookup on a multi-homed box.
        """
        nxdomain = False
        last = None
        waits = (self.policy.first_pass_timeout_s, self.policy.server_timeout_s)
        for attempt, wait in enumerate(waits):
            for server in self._preferred_order(servers):
                if time.monotonic() >= deadline:
                    break
                try:
                    rcode, found, ttl, truncated = self._exchange_udp(
                        server, name, qtype, deadline, wait=wait)
                    if truncated:
                        rcode, found, ttl, _ = self._exchange_tcp(
                            server, name, qtype, deadline)
                except (DnsError, OSError) as exc:
                    last = exc
                    continue               # no answer: try the next server
                if rcode == RCODE_NOERROR:
                    self._preferred = server
                    return found, (ttl if ttl is not None else
                                   self.policy.negative_ttl_s), False
                if rcode == RCODE_NXDOMAIN:
                    # Weaker evidence than an answer: on a multi-homed box our
                    # server list is a union across interfaces, so this may just
                    # mean we asked one that is not authoritative for this suffix.
                    self._preferred = server
                    nxdomain = True
                    continue
                last = DnsError(f"{server} returned rcode {rcode}")
            if nxdomain:
                break                      # an answer, just not the one we wanted
        if last is not None and not nxdomain:
            _policy.note(self.policy, f"dns: no server answered for {name}: {last}")
        return [], self.policy.negative_ttl_s, nxdomain

    # -- public API --------------------------------------------------------

    def resolve(self, host: str, want_v6=None, deadline_s=None) -> list:
        """Every address *host* resolves to, best first. Never `getaddrinfo`.

        Raises `DnsError` when nothing could be learned, so the caller can say
        what actually went wrong instead of reporting a generic failure.
        """
        name = str(host or "").strip().rstrip(".").lower()
        if not name:
            raise DnsError("empty hostname")

        literal = addrs.as_ip(name)
        if literal is not None:
            return [literal]

        pinned = self.hosts().get(name)
        if pinned:
            # A hosts entry is a deliberate local decision and outranks DNS,
            # exactly as it does for the OS.
            out = [a for a in (addrs.as_ip(p) for p in pinned) if a is not None]
            if out:
                return out

        if want_v6 is None:
            # The fix, stated plainly: AAAA answers are only candidates if there
            # is a ROUTE to the v6 internet. An address is not a route.
            want_v6 = probe.has_global_ipv6()

        cache_key = (name, bool(want_v6))
        hit = self._cached(cache_key)
        if hit is not None:
            if not hit:
                raise DnsError(f"no address for {name} (cached)")
            return list(hit)

        if self._should_hand_back(name):
            out = self._os_handback(name)
            self._store(cache_key, out, self.policy.negative_ttl_s
                        if not out else self.policy.max_ttl_s)
            if not out:
                raise DnsError(f"no address for {name}")
            return out

        # The caller's timeout is a ceiling, never a floor: resolution time comes
        # OUT of their budget rather than being added to it. `_infer_with_deadline`
        # hands adapters a shrinking `params["timeout"]` to enforce a run budget,
        # and a resolver that spent a fixed 5s on top would quietly break it.
        budget = self.policy.total_deadline_s
        if isinstance(deadline_s, (int, float)) and not isinstance(deadline_s, bool):
            budget = max(0.05, min(budget, float(deadline_s)))

        servers = self.servers()
        deadline = time.monotonic() + budget
        found: list = []
        ttl = self.policy.negative_ttl_s
        nxdomain = False

        if servers:
            found, ttl, nxdomain = self._ask(servers, name, TYPE_A, deadline)
            if want_v6:
                v6, ttl6, _ = self._ask(servers, name, TYPE_AAAA, deadline)
                found = found + v6
                ttl = min(ttl, ttl6) if v6 else ttl

        if not found and self._may_fall_back(name, bool(servers), nxdomain):
            _policy.note(self.policy,
                         f"dns: falling back to public resolvers for {name}")
            fallback = list(_policy.PUBLIC_FALLBACK)
            deadline = time.monotonic() + budget
            found, ttl, _ = self._ask(fallback, name, TYPE_A, deadline)
            if want_v6:
                v6, _, _ = self._ask(fallback, name, TYPE_AAAA, deadline)
                found = found + v6

        self._store(cache_key, found,
                    ttl if found else self.policy.negative_ttl_s)
        if not found:
            raise DnsError(
                f"no address for {name}"
                + (" (all nameservers said it does not exist)" if nxdomain
                   else " (no nameserver answered)"))
        return found

    def _should_hand_back(self, name: str) -> bool:
        """Names a unicast stub cannot answer — mDNS, NetBIOS, LLMNR, search."""
        if "." not in name:
            return True
        return any(name.endswith(s) for s in self.policy.internal_suffixes)

    def _os_handback(self, name: str) -> list:
        """Ask the OS, then judge the answer exactly as if we had resolved it.

        A documented exception rather than a hole: the threat model is a poisoned
        *unicast* cache, and these names never came from unicast DNS anyway.
        """
        _policy.note(self.policy, f"dns: handing {name} to the OS resolver")
        try:
            infos = socket.getaddrinfo(name, None, 0, socket.SOCK_STREAM)
        except (OSError, UnicodeError):
            return []
        out: list = []
        for info in infos:
            address = addrs.as_ip(str(info[4][0]).split("%", 1)[0])
            if address is not None and address not in out:
                out.append(address)
        return out

    def _may_fall_back(self, name, had_servers, nxdomain) -> bool:
        if not self.policy.public_fallback:
            return False
        if not had_servers:
            return True                    # nothing configured: nothing to lose
        if nxdomain:
            # Only second-guess a "does not exist" for a name the public DNS
            # could plausibly answer. Doing it for internal names both leaks them
            # and returns the wrong address under split horizon. Note this uses
            # the WIDER private list, not the handback list: `.corp` is perfectly
            # answerable over unicast DNS, it is just never answerable publicly.
            return looks_public(name, self.policy.private_suffixes)
        return True                        # nobody answered at all


def _recv_exactly(sock, count: int) -> bytes:
    chunks, remaining = [], count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise DnsError("connection closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
