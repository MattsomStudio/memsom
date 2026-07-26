"""Where to ask, learned from OS *config* rather than the OS *resolver*.

This distinction is the whole reason the subpackage can work. Reading
`HKLM\\...\\Tcpip\\Parameters\\Interfaces\\*\\DhcpNameServer` tells us *which
server to ask*; it does not go anywhere near `getaddrinfo` or the DNS client
cache that broke. Same for `/etc/resolv.conf` and `scutil --dns`.

**The hosts file is not optional.** A stub resolver that skips it silently breaks
every locally-pinned name — on a homelab box that means mesh/VPN hostnames and
the `*.docker.internal` names, i.e. things people use every day. Losing those to
a DNS "improvement" is exactly the kind of regression that makes people switch
the whole thing off.

What a unicast stub structurally cannot do — mDNS on multicast 5353, NetBIOS,
LLMNR — is handled by handing those names back to the OS resolver, with the
address gauntlet still applied to whatever it returns. See `policy.INTERNAL_SUFFIXES`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_WINDOWS = sys.platform.startswith("win")

#: Splits the Windows registry's multi-server strings, which use space in the
#: DHCP case and comma in the static case, inconsistently across versions.
_SPLIT = re.compile(r"[,\s]+")


def hosts_path() -> Path:
    """The platform's hosts file."""
    if _WINDOWS:
        root = os.environ.get("SystemRoot", r"C:\Windows")
        return Path(root) / "System32" / "drivers" / "etc" / "hosts"
    return Path("/etc/hosts")


def hosts_entries(path=None) -> dict:
    """`{lowercased name: [address strings]}` from the hosts file.

    Consulted before any query, because that is what the OS does and because a
    name pinned here is pinned deliberately. Unparseable lines are skipped rather
    than raising: a malformed hosts file must degrade to "no local pins", never
    to "no networking".
    """
    target = Path(path) if path is not None else hosts_path()
    out: dict = {}
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return out
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        address, names = parts[0], parts[1:]
        for name in names:
            out.setdefault(name.strip().lower().rstrip("."), []).append(address)
    return out


def _windows_nameservers() -> list:
    """Registry only — no `getaddrinfo`, no WMI, no subprocess."""
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return []
    found: list = []
    roots = [
        r"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters",
        r"SYSTEM\CurrentControlSet\Services\Tcpip6\Parameters",
    ]
    for root in list(roots):
        interfaces = root + r"\Interfaces"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, interfaces) as key:
                index = 0
                while True:
                    try:
                        roots.append(interfaces + "\\" + winreg.EnumKey(key, index))
                    except OSError:
                        break
                    index += 1
        except OSError:
            continue

    for path in roots:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                # Static first: an operator who typed a server meant it.
                for value in ("NameServer", "DhcpNameServer"):
                    try:
                        raw, _ = winreg.QueryValueEx(key, value)
                    except OSError:
                        continue
                    found.extend(s for s in _SPLIT.split(str(raw or "")) if s)
        except OSError:
            continue
    return found


def _resolv_conf_nameservers(path="/etc/resolv.conf") -> list:
    found: list = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return found
    for line in text.splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        if line.lower().startswith("nameserver"):
            parts = line.split()
            if len(parts) >= 2:
                found.append(parts[1])
    return found


def _scutil_nameservers() -> list:
    """macOS fallback when resolv.conf is absent or stale.

    Plain `subprocess` rather than the house `run_no_window` helper: this branch
    only ever executes on macOS, where the hidden-console problem that helper
    exists for does not apply, and importing `providers.base` here would couple
    the resolver to the provider base class for no gain.
    """
    try:
        out = subprocess.run(["scutil", "--dns"], capture_output=True,
                             text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return []
    return re.findall(r"nameserver\[\d+\]\s*:\s*(\S+)", out.stdout or "")


def system_nameservers() -> list:
    """Every nameserver this machine is configured with, in preference order.

    A **union across interfaces**, deduplicated, which is a deliberate deviation
    from what a strict stub would do. On a multi-homed box — and this one runs a
    Nebula mesh alongside Ethernet — there is no stdlib way to learn which
    interface is authoritative for which suffix, so the honest options are "ask
    them all" or "guess". Asking them all is why `dns.py` treats an NXDOMAIN as
    weaker evidence than an answer.
    """
    if _WINDOWS:
        found = _windows_nameservers()
    else:
        found = _resolv_conf_nameservers()
        if not found and sys.platform == "darwin":
            found = _scutil_nameservers()

    seen, ordered = set(), []
    for server in found:
        server = str(server).strip().strip("[]")
        # A nameserver pointing at ourselves is the OS resolver we are routing
        # around (systemd-resolved on 127.0.0.53, Docker on 127.0.0.11). Asking
        # it would reintroduce the cache this subpackage exists to bypass.
        if not server or server.startswith("127.") or server in ("::1", "0.0.0.0"):
            continue
        if server not in seen:
            seen.add(server)
            ordered.append(server)
    return ordered


def search_suffixes() -> list:
    """Suffixes to append to a single-label name, best effort."""
    if not _WINDOWS:
        try:
            text = Path("/etc/resolv.conf").read_text(encoding="utf-8",
                                                      errors="replace")
        except (OSError, ValueError):
            return []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.lower().startswith(("search", "domain")):
                return line.split()[1:]
        return []
    try:
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters") as key:
            for value in ("SearchList", "Domain"):
                try:
                    raw, _ = winreg.QueryValueEx(key, value)
                except OSError:
                    continue
                items = [s for s in _SPLIT.split(str(raw or "")) if s]
                if items:
                    return items
    except (ImportError, OSError):
        pass
    return []
