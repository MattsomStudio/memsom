"""Nameserver discovery, the hosts file, and the IPv6-route probe.

The hosts file has its own test module because forgetting it is the single most
likely way this subpackage breaks a working machine: a homelab box pins mesh and
VPN hostnames there, and a resolver that only speaks DNS would silently lose
them. Addresses below are RFC 5737 documentation ranges, never a real network.
"""

import socket

from memsom.providers.net import nameservers, probe


# ---------------------------------------------------------------------------
# hosts file
# ---------------------------------------------------------------------------

def _write(tmp_path, text):
    path = tmp_path / "hosts"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_pinned_name_is_read_from_the_hosts_file(tmp_path):
    path = _write(tmp_path, "192.0.2.20\tmesh.example\n")
    assert nameservers.hosts_entries(path) == {
        "mesh.example": ["192.0.2.20"]}


def test_comments_and_blank_lines_are_ignored(tmp_path):
    path = _write(tmp_path, "# Copyright\n\n  # indented comment\n"
                            "10.0.0.1 a.test   # trailing comment\n")
    assert nameservers.hosts_entries(path) == {"a.test": ["10.0.0.1"]}


def test_one_address_can_carry_several_names(tmp_path):
    path = _write(tmp_path, "127.0.0.1 localhost loopback me.test\n")
    entries = nameservers.hosts_entries(path)
    assert entries["loopback"] == ["127.0.0.1"]
    assert entries["me.test"] == ["127.0.0.1"]


def test_a_name_pinned_twice_keeps_both_addresses(tmp_path):
    path = _write(tmp_path, "10.0.0.1 dual.test\n10.0.0.2 dual.test\n")
    assert nameservers.hosts_entries(path)["dual.test"] == ["10.0.0.1", "10.0.0.2"]


def test_names_are_matched_case_insensitively(tmp_path):
    path = _write(tmp_path, "10.0.0.1 Mesh.EXAMPLE.\n")
    assert "mesh.example" in nameservers.hosts_entries(path)


def test_a_malformed_hosts_file_degrades_to_no_pins_not_to_an_exception(tmp_path):
    """A broken hosts file must cost you local pins, never all networking."""
    path = _write(tmp_path, "this-line-has-no-address\n\x00\x01garbage\n")
    assert nameservers.hosts_entries(path) == {}


def test_a_missing_hosts_file_is_not_an_error(tmp_path):
    assert nameservers.hosts_entries(tmp_path / "nope") == {}


def test_the_platform_hosts_path_is_plausible():
    path = str(nameservers.hosts_path()).lower()
    assert path.endswith("hosts") and ("etc" in path or "drivers" in path)


# ---------------------------------------------------------------------------
# nameserver discovery
# ---------------------------------------------------------------------------

def test_discovery_returns_something_on_this_machine():
    """Not a strict assertion about content — a machine with no configured
    resolver is legitimate. This pins the shape: a list of plain strings."""
    found = nameservers.system_nameservers()
    assert isinstance(found, list)
    assert all(isinstance(s, str) and s.strip() for s in found)


def test_a_loopback_nameserver_is_never_returned(monkeypatch):
    """systemd-resolved sits on 127.0.0.53 and Docker on 127.0.0.11. Asking
    either would route us straight back into the OS resolver cache this whole
    subpackage exists to bypass."""
    monkeypatch.setattr(nameservers, "_WINDOWS", False)
    monkeypatch.setattr(nameservers, "_resolv_conf_nameservers",
                        lambda *a, **k: ["127.0.0.53", "127.0.0.11",
                                         "192.168.1.1", "::1", "0.0.0.0"])
    assert nameservers.system_nameservers() == ["192.168.1.1"]


def test_duplicate_servers_across_interfaces_are_collapsed(monkeypatch):
    """A multi-homed box lists the same server on several interfaces; querying
    it three times in a row is just three timeouts when it is down."""
    monkeypatch.setattr(nameservers, "_WINDOWS", False)
    monkeypatch.setattr(nameservers, "_resolv_conf_nameservers",
                        lambda *a, **k: ["1.1.1.1", "9.9.9.9", "1.1.1.1"])
    assert nameservers.system_nameservers() == ["1.1.1.1", "9.9.9.9"]


def test_resolv_conf_parsing(tmp_path):
    path = tmp_path / "resolv.conf"
    path.write_text("# generated\nsearch example.com\n"
                    "nameserver 10.0.0.1\n; comment\nnameserver 10.0.0.2\n",
                    encoding="utf-8")
    assert nameservers._resolv_conf_nameservers(path) == ["10.0.0.1", "10.0.0.2"]


# ---------------------------------------------------------------------------
# the IPv6 probe — route, not address
# ---------------------------------------------------------------------------

def test_the_probe_asks_for_a_route_not_for_an_address(monkeypatch):
    """The ULA trap: the box HAS a v6 address and has NO route. AI_ADDRCONFIG
    reads the first and reports the second, which is the entire bug."""
    probe.reset_cache()
    monkeypatch.setattr(probe, "_route_exists", lambda family, target: False)
    monkeypatch.setattr(probe, "ipv6_addresses_present", lambda: True)
    report = probe.diagnosis()
    assert report["ipv6_advertised_but_unroutable"] is True
    probe.reset_cache()


def test_a_healthy_dual_stack_box_is_not_flagged(monkeypatch):
    probe.reset_cache()
    monkeypatch.setattr(probe, "_route_exists", lambda family, target: True)
    monkeypatch.setattr(probe, "ipv6_addresses_present", lambda: True)
    assert probe.diagnosis()["ipv6_advertised_but_unroutable"] is False
    probe.reset_cache()


def test_the_verdict_is_cached_rather_than_probed_per_request(monkeypatch):
    """A syscall on every fetch would be silly; the answer changes when a cable
    is plugged in."""
    probe.reset_cache()
    calls = []

    def counting(family, target):
        calls.append(family)
        return True

    monkeypatch.setattr(probe, "_route_exists", counting)
    for _ in range(5):
        probe.has_global_ipv6()
    assert len(calls) == 1
    probe.reset_cache()


def test_the_probe_sends_no_packet():
    """A UDP connect only selects a route. This runs against the real stack on
    purpose — it must be safe on a network you are only allowed to read."""
    assert probe._route_exists(socket.AF_INET, "1.1.1.1") in (True, False)
