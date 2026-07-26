"""The DNS parser — mostly tests that it throws the right things away.

Every case here is a packet a hostile or broken server can send. The parser's job
is to refuse it without hanging, without reading past the buffer, and without
turning an unsolicited packet into a failed lookup.
"""

import ipaddress
import struct

import pytest

from memsom.providers.net import dns, policy


# ---------------------------------------------------------------------------
# packet builders
# ---------------------------------------------------------------------------

def _question(name="example.com", qtype=dns.TYPE_A):
    return dns.encode_name(name) + struct.pack(">HH", qtype, dns.CLASS_IN)


def _a_record(ip="93.184.216.34", ttl=300, name_ptr=0xC00C):
    packed = ipaddress.IPv4Address(ip).packed
    return (struct.pack(">H", name_ptr)
            + struct.pack(">HHIH", dns.TYPE_A, dns.CLASS_IN, ttl, len(packed))
            + packed)


def _aaaa_record(ip="2606:4700::1", ttl=300):
    packed = ipaddress.IPv6Address(ip).packed
    return (struct.pack(">H", 0xC00C)
            + struct.pack(">HHIH", dns.TYPE_AAAA, dns.CLASS_IN, ttl, len(packed))
            + packed)


def _response(txid=0x1234, flags=0x8180, question=None, answers=b"", ancount=None):
    question = _question() if question is None else question
    count = ancount if ancount is not None else (1 if answers else 0)
    return (struct.pack(">HHHHHH", txid, flags, 1, count, 0, 0)
            + question + answers)


# ---------------------------------------------------------------------------
# the happy path, so the refusals below mean something
# ---------------------------------------------------------------------------

def test_a_well_formed_answer_parses():
    rcode, found, ttl, truncated = dns.parse_response(
        _response(answers=_a_record()), 0x1234, "example.com", dns.TYPE_A)
    assert rcode == dns.RCODE_NOERROR
    assert found == [ipaddress.IPv4Address("93.184.216.34")]
    assert ttl == 300 and truncated is False


def test_an_aaaa_answer_parses():
    _, found, _, _ = dns.parse_response(
        _response(question=_question(qtype=dns.TYPE_AAAA),
                  answers=_aaaa_record()),
        0x1234, "example.com", dns.TYPE_AAAA)
    assert found == [ipaddress.IPv6Address("2606:4700::1")]


def test_nxdomain_is_reported_not_raised():
    """A "does not exist" is an answer. The resolver decides what to do with it."""
    rcode, found, _, _ = dns.parse_response(
        _response(flags=0x8183), 0x1234, "example.com", dns.TYPE_A)
    assert rcode == dns.RCODE_NXDOMAIN and found == []


# ---------------------------------------------------------------------------
# forgery — every one of these must be thrown away
# ---------------------------------------------------------------------------

def test_a_reply_with_the_wrong_transaction_id_is_refused():
    with pytest.raises(dns.DnsError):
        dns.parse_response(_response(txid=0x9999, answers=_a_record()),
                           0x1234, "example.com", dns.TYPE_A)


def test_a_packet_that_is_not_a_response_is_refused():
    """QR clear. A query arriving on our socket is not an answer to it."""
    with pytest.raises(dns.DnsError):
        dns.parse_response(_response(flags=0x0100, answers=_a_record()),
                           0x1234, "example.com", dns.TYPE_A)


def test_an_answer_to_a_different_name_is_refused():
    """The classic forgery: a reply carrying an answer for a name nobody asked
    about, hoping it gets cached anyway."""
    with pytest.raises(dns.DnsError):
        dns.parse_response(
            _response(question=_question("evil.example"), answers=_a_record()),
            0x1234, "example.com", dns.TYPE_A)


def test_an_answer_to_a_different_type_is_refused():
    with pytest.raises(dns.DnsError):
        dns.parse_response(
            _response(question=_question(qtype=dns.TYPE_AAAA)),
            0x1234, "example.com", dns.TYPE_A)


def test_a_short_header_is_refused():
    with pytest.raises(dns.DnsError):
        dns.parse_response(b"\x12\x34", 0x1234, "example.com", dns.TYPE_A)


# ---------------------------------------------------------------------------
# the parser DoS surface — compression pointers
# ---------------------------------------------------------------------------

def test_a_forward_compression_pointer_is_refused():
    """Backwards-only is what makes a pointer loop unrepresentable rather than
    merely unlikely."""
    buf = b"\x00" * 12 + struct.pack(">H", 0xC000 | 40) + b"\x00" * 40
    with pytest.raises(dns.DnsError):
        dns.decode_name(buf, 12)


def test_a_self_referential_compression_pointer_is_refused():
    buf = b"\x00" * 12 + struct.pack(">H", 0xC000 | 12)
    with pytest.raises(dns.DnsError):
        dns.decode_name(buf, 12)


def test_a_long_chain_of_backward_pointers_terminates():
    """Even legal backward pointers are capped, so a packet built entirely of
    them cannot make the parser spin."""
    buf = bytearray(b"\x00\x00")
    offsets = [0]
    for _ in range(dns._MAX_JUMPS + 4):
        offsets.append(len(buf))
        buf += struct.pack(">H", 0xC000 | offsets[-2])
    with pytest.raises(dns.DnsError):
        dns.decode_name(bytes(buf), offsets[-1])


def test_a_name_that_runs_past_the_end_of_the_packet_is_refused():
    with pytest.raises(dns.DnsError):
        dns.decode_name(b"\x05abc", 0)          # claims 5 bytes, supplies 3


def test_an_oversize_label_is_refused():
    with pytest.raises(dns.DnsError):
        dns.decode_name(bytes([0x7F]) + b"x" * 0x7F + b"\x00", 0)


def test_the_offset_returned_is_after_the_pointer_not_after_the_target():
    """Conflating these desynchronises the whole answer walk — the record after a
    compressed name would be parsed from the wrong place."""
    buf = b"\x03abc\x00" + struct.pack(">H", 0xC000 | 0) + b"TAIL"
    name, after = dns.decode_name(buf, 5)
    assert name == "abc"
    assert buf[after:] == b"TAIL"


# ---------------------------------------------------------------------------
# bounded reads in the answer section
# ---------------------------------------------------------------------------

def test_an_rdlength_that_overruns_the_buffer_truncates_cleanly():
    """A record claiming a VALID A-record length with the bytes missing.

    The hostile-looking case (rdlength=4000) proves nothing — it fails the
    `rdlength == 4` type check before the bound is ever consulted. This is the
    one that bites: without the bound, the slice comes back short and
    `IPv4Address(2 bytes)` raises `AddressValueError`, which is not a `DnsError`
    and so escapes `parse_response` entirely as an unexpected exception type.
    """
    bad = (struct.pack(">H", 0xC00C)
           + struct.pack(">HHIH", dns.TYPE_A, dns.CLASS_IN, 300, 4)
           + b"\x01\x02")                   # claims 4 bytes, supplies 2
    rcode, found, _, _ = dns.parse_response(
        _response(answers=bad), 0x1234, "example.com", dns.TYPE_A)
    assert rcode == dns.RCODE_NOERROR and found == []


def test_an_absurd_rdlength_also_truncates_cleanly():
    bad = (struct.pack(">H", 0xC00C)
           + struct.pack(">HHIH", dns.TYPE_A, dns.CLASS_IN, 300, 40000)
           + b"\x01\x02")
    _, found, _, _ = dns.parse_response(
        _response(answers=bad), 0x1234, "example.com", dns.TYPE_A)
    assert found == []


def test_an_answer_count_larger_than_the_records_present_is_survivable():
    rcode, found, _, _ = dns.parse_response(
        _response(answers=_a_record(), ancount=50),
        0x1234, "example.com", dns.TYPE_A)
    assert found == [ipaddress.IPv4Address("93.184.216.34")]


def test_record_types_we_did_not_ask_for_are_skipped_not_misread():
    cname = (struct.pack(">H", 0xC00C)
             + struct.pack(">HHIH", 5, dns.CLASS_IN, 300, 2) + b"\xc0\x0c")
    _, found, _, _ = dns.parse_response(
        _response(answers=cname + _a_record(), ancount=2),
        0x1234, "example.com", dns.TYPE_A)
    assert found == [ipaddress.IPv4Address("93.184.216.34")]


def test_the_truncated_bit_is_reported_so_the_caller_can_retry_over_tcp():
    _, _, _, truncated = dns.parse_response(
        _response(flags=0x8380), 0x1234, "example.com", dns.TYPE_A)
    assert truncated is True


# ---------------------------------------------------------------------------
# name encoding
# ---------------------------------------------------------------------------

def test_a_name_longer_than_the_protocol_allows_is_refused():
    with pytest.raises(dns.DnsError):
        dns.encode_name(".".join(["label"] * 60))


def test_a_trailing_dot_is_accepted():
    assert dns.encode_name("example.com.") == dns.encode_name("example.com")


# ---------------------------------------------------------------------------
# looks_public — which NXDOMAIN we are allowed to second-guess
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["github.com", "api.example.co.uk", "a.b.io"])
def test_public_shaped_names_may_be_retried_against_public_resolvers(name):
    assert dns.looks_public(name, policy.PRIVATE_SUFFIXES) is True


@pytest.mark.parametrize("name", [
    "nas", "printer.local", "box.home.arpa", "server.lan", "thing.internal",
    "nas.corp", "www.home", "host.localdomain", "rebind.test", "a.invalid",
    "10.0.0.5",
])
def test_internal_names_are_believed_when_they_say_they_do_not_exist(name):
    """Second-guessing these both leaks the name to Cloudflare and returns the
    wrong address under split horizon."""
    assert dns.looks_public(name, policy.PRIVATE_SUFFIXES) is False


# ---------------------------------------------------------------------------
# StubResolver — no sockets, every exchange faked
# ---------------------------------------------------------------------------

def _resolver(monkeypatch, answers, hosts=None, **kw):
    """A resolver whose wire exchange is a dict lookup. `answers` maps
    (server, qtype) -> (rcode, [addresses]); anything missing times out."""
    asked = []
    pol = policy.NetPolicy(**kw)
    res = dns.StubResolver(pol, servers=["10.9.9.9"], hosts=hosts or {})

    def fake(server, name, qtype, deadline, wait=None):
        asked.append((server, name, qtype))
        try:
            rcode, found = answers[(server, qtype)]
        except KeyError:
            raise dns.DnsError(f"{server} timed out")
        return rcode, [ipaddress.ip_address(a) for a in found], 300, False

    monkeypatch.setattr(res, "_exchange_udp", fake)
    return res, asked


def test_a_hosts_entry_wins_without_any_query(monkeypatch):
    """A mesh/VPN host pinned in the hosts file. A resolver that only speaks
    DNS would silently lose it -- and on a homelab box that is a daily-driver
    name, not an edge case."""
    res, asked = _resolver(monkeypatch, {},
                           hosts={"mesh.example": ["192.0.2.20"]})
    assert res.resolve("mesh.example") == [
        ipaddress.IPv4Address("192.0.2.20")]
    assert asked == [], "a pinned name must not hit the network"


def test_a_literal_address_is_returned_as_is(monkeypatch):
    res, asked = _resolver(monkeypatch, {})
    assert res.resolve("93.184.216.34") == [ipaddress.IPv4Address("93.184.216.34")]
    assert asked == []


def test_aaaa_is_not_even_asked_for_without_a_route_to_the_v6_internet(monkeypatch):
    """The ULA fix. Asking for AAAA on a box that cannot reach v6 is how an
    AAAA-only answer becomes the only candidate and everything fails."""
    res, asked = _resolver(monkeypatch, {
        ("10.9.9.9", dns.TYPE_A): (dns.RCODE_NOERROR, ["93.184.216.34"])})
    got = res.resolve("example.com", want_v6=False)
    assert got == [ipaddress.IPv4Address("93.184.216.34")]
    assert [q[2] for q in asked] == [dns.TYPE_A]


def test_aaaa_is_asked_for_when_v6_actually_works(monkeypatch):
    res, asked = _resolver(monkeypatch, {
        ("10.9.9.9", dns.TYPE_A): (dns.RCODE_NOERROR, ["93.184.216.34"]),
        ("10.9.9.9", dns.TYPE_AAAA): (dns.RCODE_NOERROR, ["2606:4700::1"])})
    got = res.resolve("example.com", want_v6=True)
    assert ipaddress.IPv6Address("2606:4700::1") in got
    assert sorted(q[2] for q in asked) == [dns.TYPE_A, dns.TYPE_AAAA]


def test_an_answer_is_cached_rather_than_asked_twice(monkeypatch):
    res, asked = _resolver(monkeypatch, {
        ("10.9.9.9", dns.TYPE_A): (dns.RCODE_NOERROR, ["93.184.216.34"])})
    res.resolve("example.com", want_v6=False)
    res.resolve("example.com", want_v6=False)
    assert len(asked) == 1


def test_a_single_label_name_is_handed_to_the_os_resolver(monkeypatch):
    """mDNS, NetBIOS and search suffixes are out of reach for a unicast stub.
    Handing back is a documented exception, not a hole -- the gauntlet still
    judges whatever comes back."""
    res, asked = _resolver(monkeypatch, {})
    monkeypatch.setattr(res, "_os_handback",
                        lambda name: [ipaddress.IPv4Address("10.0.0.7")])
    assert res.resolve("nas") == [ipaddress.IPv4Address("10.0.0.7")]
    assert asked == []


def test_a_dot_local_name_is_handed_to_the_os_resolver(monkeypatch):
    res, asked = _resolver(monkeypatch, {})
    monkeypatch.setattr(res, "_os_handback",
                        lambda name: [ipaddress.IPv4Address("10.0.0.8")])
    assert res.resolve("printer.local") == [ipaddress.IPv4Address("10.0.0.8")]
    assert asked == []


def test_nxdomain_for_an_internal_name_is_not_second_guessed(monkeypatch):
    """Falling back to a public resolver here would leak the name AND return the
    wrong address under split horizon."""
    res, asked = _resolver(monkeypatch, {
        ("10.9.9.9", dns.TYPE_A): (dns.RCODE_NXDOMAIN, [])})
    with pytest.raises(dns.DnsError):
        res.resolve("nas.corp", want_v6=False)
    assert all(server == "10.9.9.9" for server, _, _ in asked), \
        "public resolvers must not be consulted for an internal name"


def test_a_server_that_does_not_answer_falls_back_to_public_resolvers(monkeypatch):
    """The 2026-07-08 failure mode: the router's dnsmasq stopped answering while
    its forwards were already correct."""
    res, asked = _resolver(monkeypatch, {
        ("1.1.1.1", dns.TYPE_A): (dns.RCODE_NOERROR, ["93.184.216.34"])})
    assert res.resolve("example.com", want_v6=False) == [
        ipaddress.IPv4Address("93.184.216.34")]
    assert "1.1.1.1" in [server for server, _, _ in asked]


def test_the_public_fallback_can_be_switched_off(monkeypatch):
    res, asked = _resolver(monkeypatch, {}, public_fallback=False)
    with pytest.raises(dns.DnsError):
        res.resolve("example.com", want_v6=False)
    assert all(server == "10.9.9.9" for server, _, _ in asked)


def test_a_dead_server_costs_the_short_timeout_not_the_long_one(monkeypatch):
    """Measured on this box 2026-07-25: with a VPN up, three of five configured
    nameservers were unreachable for port 53 and a cold lookup cost 4047ms —
    three serial 2s timeouts before the live server answered in 266ms. The first
    pass finds a live server fast; a merely-slow one still gets the full wait on
    the second pass."""
    waits = []
    pol = policy.NetPolicy(first_pass_timeout_s=0.6, server_timeout_s=2.0)
    res = dns.StubResolver(pol, servers=["10.0.0.1", "10.0.0.2"], hosts={})

    def fake(server, name, qtype, deadline, wait=None):
        waits.append((server, wait))
        if server == "10.0.0.2":
            return dns.RCODE_NOERROR, [ipaddress.ip_address("93.184.216.34")], 300, False
        raise dns.DnsError("timed out")

    monkeypatch.setattr(res, "_exchange_udp", fake)
    res.resolve("example.com", want_v6=False)
    assert waits[0] == ("10.0.0.1", 0.6), waits
    assert all(w == 0.6 for _, w in waits), "the dead server must not cost 2s"


def test_the_server_that_answered_is_asked_first_next_time(monkeypatch):
    """Without this, every cold lookup re-walks the graveyard of dead servers a
    multi-homed box accumulates."""
    order = []
    res = dns.StubResolver(policy.NetPolicy(),
                           servers=["10.0.0.1", "10.0.0.2"], hosts={})

    def fake(server, name, qtype, deadline, wait=None):
        order.append(server)
        if server == "10.0.0.2":
            return dns.RCODE_NOERROR, [ipaddress.ip_address("93.184.216.34")], 300, False
        raise dns.DnsError("timed out")

    monkeypatch.setattr(res, "_exchange_udp", fake)
    res.resolve("first.example", want_v6=False)
    order.clear()
    res.resolve("second.example", want_v6=False)
    assert order[0] == "10.0.0.2", order


def test_the_fallback_is_logged_rather_than_silent(monkeypatch):
    """Who answered the question changed. That must never happen quietly."""
    lines = []
    res, _ = _resolver(monkeypatch, {
        ("1.1.1.1", dns.TYPE_A): (dns.RCODE_NOERROR, ["93.184.216.34"])},
        log=lines.append)
    res.resolve("example.com", want_v6=False)
    assert any("falling back" in line for line in lines)
