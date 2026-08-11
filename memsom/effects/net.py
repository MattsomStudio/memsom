"""memsom.effects.net -- outbound network calls.

fetch_external moved out of memsom/__init__.py (Phase 2, the core split).
PLAN.md's fuller destination for this file (every module's urllib/requests
call, one timeout audit) is later work; only the frozen-core fetch lands here
in Phase 2.
"""

import sys
import urllib.request
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent  # memsom/ package root (ships with the wheel)
EXT_URL = "https://raw.githubusercontent.com/sqlite/sqlite/master/README.md"
FALLBACK = HOME / "external_fallback.txt"


def fetch_external(offline):
    if not offline:
        try:
            req = urllib.request.Request(EXT_URL, headers={"User-Agent": "memsom/0.1"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read().decode("utf-8", "replace"), f"{EXT_URL} (fetched, stored)"
        except Exception as err:
            print(f"[memsom] live fetch failed ({err}); using stored fallback", file=sys.stderr)
    return FALLBACK.read_text(encoding="utf-8", errors="replace"), f"{EXT_URL} (local snapshot)"
