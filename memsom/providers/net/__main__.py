"""`python -m memsom.providers.net --diagnose` — what this machine looks like to us.

Shipped before anything depends on it, on purpose. When someone's fetch fails,
the first question is "is this the app or is this your network", and the only
cheap way to answer it is a read-only dump of what we discovered: which
nameservers, which local pins, whether IPv6 is real. Read-only — it never
reconfigures anything.
"""

from __future__ import annotations

import argparse
import json
import sys

from memsom.providers.net import nameservers, policy, probe


def collect(name: str | None = None) -> dict:
    pol = policy.from_env()
    hosts = nameservers.hosts_entries()
    out = {
        "stub_enabled": pol.enabled,
        "nameservers_configured": list(pol.nameservers)
                                  or nameservers.system_nameservers(),
        "public_fallback": pol.public_fallback,
        "public_fallback_servers": list(policy.PUBLIC_FALLBACK),
        "search_suffixes": nameservers.search_suffixes(),
        "hosts_file": str(nameservers.hosts_path()),
        "hosts_entries": {k: v for k, v in sorted(hosts.items())
                          if not k.startswith("localhost")},
        "internal_suffixes": list(pol.internal_suffixes),
        **probe.diagnosis(),
    }
    if name:
        try:
            from memsom.providers.net import dns
        except ImportError:
            out["resolve"] = {"error": "resolver not available"}
            return out
        try:
            answer = dns.StubResolver(pol).resolve(name)
            out["resolve"] = {"name": name,
                              "addresses": [str(a) for a in answer]}
        except Exception as exc:                       # diagnostic: report, never raise
            out["resolve"] = {"name": name,
                              "error": f"{type(exc).__name__}: {exc}"}
    return out


def _human(report: dict) -> str:
    lines = []
    fault = report.get("ipv6_advertised_but_unroutable")
    lines.append(f"stub resolver     : {'ON' if report['stub_enabled'] else 'OFF'}")
    lines.append("nameservers       : "
                 + (", ".join(report["nameservers_configured"]) or "NONE FOUND"))
    lines.append(f"public fallback   : {'on' if report['public_fallback'] else 'off'}"
                 f" ({', '.join(report['public_fallback_servers'])})")
    lines.append("search suffixes   : "
                 + (", ".join(report["search_suffixes"]) or "-"))
    lines.append(f"ipv4 route        : {report['ipv4_global_route']}")
    lines.append(f"ipv6 addresses    : {report['ipv6_addresses_present']}")
    lines.append(f"ipv6 route        : {report['ipv6_global_route']}")
    if fault:
        lines.append("")
        lines.append("  ** IPv6 is advertised on this machine but has no route to the")
        lines.append("     internet. This is what makes AAAA-only DNS answers look")
        lines.append("     usable and then fail to connect. Disable IPv6 on the")
        lines.append("     adapter, or stop the router advertising a ULA prefix.")
        lines.append("")
    lines.append(f"hosts file        : {report['hosts_file']}")
    for name, addresses in report["hosts_entries"].items():
        lines.append(f"  {name:<34} {', '.join(addresses)}")
    resolved = report.get("resolve")
    if resolved:
        lines.append("")
        if "error" in resolved:
            lines.append(f"resolve {resolved['name']}: ERROR {resolved['error']}")
        else:
            lines.append(f"resolve {resolved['name']}: "
                         + ", ".join(resolved["addresses"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m memsom.providers.net")
    parser.add_argument("--diagnose", action="store_true",
                        help="dump discovered network configuration")
    parser.add_argument("--resolve", metavar="NAME",
                        help="also resolve NAME through the stub resolver")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    report = collect(args.resolve)
    print(json.dumps(report, indent=2) if args.json else _human(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
