"""The address gauntlet — every way of spelling a forbidden address.

`127.0.0.1` is the form nobody attacking you will type. These are the forms they
will, and each one reaches the same socket. A check written against the text
rather than against the parsed address passes all of them.
"""

import ipaddress

import pytest

from memsom.providers.net import addrs


# --------------------------------------------------------------------------
# as_ip — legacy IPv4 spellings decode; hostnames stay hostnames
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "2130706433",     # decimal
    "0x7f000001",     # hex
    "0177.0.0.1",     # octal
    "127.1",          # short form
    "127.0.0.1",      # the boring one
])
def test_every_legacy_spelling_of_loopback_decodes_to_loopback(text):
    """The bypass catalog. If any of these returns None the caller treats it as
    a hostname, resolves it, gets nothing back, and lets the call through."""
    assert addrs.as_ip(text) == ipaddress.IPv4Address("127.0.0.1")


@pytest.mark.parametrize("text", [
    "example.com",
    "1.2.3.4.example.com",
    "12345.com",
    "rebind.test",
    "",
    "   ",
])
def test_a_hostname_is_never_mistaken_for_an_address(text):
    """The other half of the contract: over-eager parsing would make every
    numeric-ish hostname unresolvable."""
    assert addrs.as_ip(text) is None


def test_a_bracketed_ipv6_literal_from_a_url_parses():
    """urlsplit hands back `[::1]` with the brackets on."""
    assert addrs.as_ip("[::1]") == ipaddress.IPv6Address("::1")


# --------------------------------------------------------------------------
# unwrap — an IPv6 literal can carry an IPv4 address inside it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,reason", [
    ("::ffff:127.0.0.1", "IPv4-mapped"),
    ("2002:7f00:1::", "6to4"),
    ("64:ff9b::7f00:1", "NAT64"),
    ("::7f00:1", "deprecated IPv4-compatible"),
])
def test_loopback_hidden_inside_an_ipv6_address_is_found(text, reason):
    """Judging only the outer form is how each of these reaches loopback through
    a check that knows 127.0.0.0/8 perfectly well."""
    hit = addrs.denied_by(addrs.as_ip(text))
    assert hit is not None, f"{reason} ({text}) was not unwrapped"


def test_a_teredo_address_is_judged_on_both_embedded_addresses():
    """Teredo carries a server and a client v4 address. A server pointed at
    loopback is as much a way in as a client one."""
    embedded = addrs.unwrap(ipaddress.IPv6Address(
        "2001:0:4136:e378:8000:63bf:3fff:fdd2"))
    assert ipaddress.IPv4Address("192.0.2.45") in embedded


def test_an_ordinary_public_ipv6_address_is_not_unwrapped_into_nonsense():
    """Guard against the unwrapper inventing an embedded v4 for every address —
    that would deny most of the v6 internet."""
    assert addrs.denied_by(ipaddress.IPv6Address("2606:4700::6810:85e5")) is None


# --------------------------------------------------------------------------
# the deny list
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "127.0.0.1", "127.9.9.9", "::1",
    "169.254.169.254",           # cloud metadata — the Capital One address
    "fe80::1",
    "0.0.0.0", "0", "::",        # unspecified: reaches a local listener
])
def test_the_denied_ranges_are_refused(text):
    assert addrs.denied_by(addrs.as_ip(text)) is not None


@pytest.mark.parametrize("text", [
    "93.184.216.34",             # public
    "192.168.1.10",              # home LAN — deliberately permitted
    "10.0.0.5",
    "2606:4700::6810:85e5",
])
def test_public_and_home_lan_addresses_are_allowed(text):
    """The home LAN is NOT denied on purpose: http_fetch is a read-only GET and
    the standing rule permits those. Denying it wholesale is how a seatbelt gets
    switched off for good."""
    assert addrs.denied_by(addrs.as_ip(text)) is None


def test_an_extra_denied_cidr_is_honoured():
    """The operator's NAS belongs on the list, and its address is theirs to
    supply rather than this module's to guess."""
    hit = addrs.denied_by(addrs.as_ip("192.168.1.50"),
                          extra=["192.168.1.50/32"])
    assert hit is not None


# --------------------------------------------------------------------------
# vet — all or nothing
# --------------------------------------------------------------------------

def test_one_denied_address_condemns_the_whole_answer_set():
    """Round-robin DNS smuggling: approve the good four and an attacker wins by
    retrying until the bad one is served."""
    answer = [addrs.as_ip("93.184.216.34"), addrs.as_ip("127.0.0.1")]
    refusal = addrs.vet(answer, host="rebind.test")
    assert refusal and "127.0.0.1" in refusal


def test_a_clean_answer_set_is_allowed():
    answer = [addrs.as_ip("93.184.216.34"), addrs.as_ip("104.20.23.154")]
    assert addrs.vet(answer, host="example.com") == ""


def test_an_empty_answer_set_is_refused_not_allowed():
    """Fail-closed at the connector. Nothing to dial is not permission to dial —
    this is the opposite posture from scope.check, deliberately."""
    assert addrs.vet([], host="nowhere.test") != ""


# --------------------------------------------------------------------------
# entry_matches — moved from scope.py, behaviour must be identical
# --------------------------------------------------------------------------

def test_a_cidr_entry_matches_by_network_not_by_string_prefix():
    """The bug this pins is a string-prefix match. "10.0.0.0/24" admits
    10.0.0.5 AND 10.0.0.50 — both are genuinely in the network — but must refuse
    10.0.1.5, which starts with the same text and is outside it."""
    assert addrs.entry_matches("10.0.0.0/24", "10.0.0.5", []) is True
    assert addrs.entry_matches("10.0.0.0/24", "10.0.0.50", []) is True
    assert addrs.entry_matches("10.0.0.0/24", "10.0.1.5", []) is False


def test_a_name_glob_matches_case_insensitively():
    assert addrs.entry_matches("*.Example.COM", "api.example.com", []) is True


def test_a_cidr_entry_is_judged_against_resolved_addresses_too():
    ips = [addrs.as_ip("10.0.0.5")]
    assert addrs.entry_matches("10.0.0.0/24", "whatever.test", ips) is True


def test_an_empty_entry_matches_nothing():
    assert addrs.entry_matches("   ", "example.com", []) is False
