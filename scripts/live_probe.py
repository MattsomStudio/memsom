#!/usr/bin/env python3
"""live_probe -- one-line "which memsom is live" (PLAN.md Phase 0, A9.7).

Phase 1 changes behaviour on the live EDITABLE tree, so the escape hatch
(the known-good wheel built in this phase, see WHEEL.md) has to be provable
to actually be in effect before it is trusted. Reads
`importlib.metadata`'s `direct_url.json` for the installed `memsom`
distribution: `editable: true` means the working tree is live; anything
else (or no `direct_url.json` at all, the shape a wheel install leaves)
means a frozen wheel is live.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata


def main() -> int:
    try:
        dist = metadata.distribution("memsom")
    except metadata.PackageNotFoundError:
        print("memsom is NOT INSTALLED in this interpreter")
        return 1

    version = dist.version
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        editable = direct_url.get("dir_info", {}).get("editable", False)
        url = direct_url.get("url", "?")
        if editable:
            print(f"LIVE: editable install, version {version}, source {url}")
            return 0
        print(f"LIVE: wheel/sdist install, version {version}, source {url}")
        return 0

    print(f"LIVE: version {version} (no direct_url.json -- installed from an "
          f"opaque wheel, e.g. via `pip install <wheel path>`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
