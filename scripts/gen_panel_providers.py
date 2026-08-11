#!/usr/bin/env python3
"""gen_panel_providers -- one-shot generator, tuning registry -> panel provider stubs.

PLAN.md Sec2.2: "a one-shot generator script that emits panel provider entries
from `tuning list --json`." Run by hand, once, when the panel repo (a separate
package -- memsom-agentic-os) is ready to adopt the registry (Matt's Q7: "The
panel adopts it on its own schedule. No cross-repo coupling, no cross-repo exit
gate"). This script has ZERO cross-repo import: it reads `tuning list --json`
(stdin or a live in-process call) and writes a plain Python source file the
panel repo can copy in and hand-edit -- it is a starting point, not a live sync.

Usage:
    memsom tuning list --json | python scripts/gen_panel_providers.py > providers.py
    python scripts/gen_panel_providers.py --live > providers.py   # in-process, no CLI hop
"""

from __future__ import annotations

import argparse
import json
import sys


_HEADER = '''"""Auto-generated {date} by scripts/gen_panel_providers.py -- STARTING POINT,
not a live sync (Matt's Q7: the panel adopts this on its own schedule). Hand-edit
freely; re-running this script does not merge, it overwrites.

Each entry is {{key, source, doc}} for a memsom.tuning knob. `source` starting
with "env:" means memsom itself is the read-only side of truth for that value --
the panel's set-line provider owns writing the underlying file/env, per
PLAN.md Sec2.2 ("env knobs are read-only through the API").
"""

PROVIDERS = [
'''


def _entry(knob: dict) -> str:
    return (
        "    {\n"
        f"        \"key\": {knob['key']!r},\n"
        f"        \"source\": {knob['source']!r},\n"
        f"        \"doc\": {knob['doc']!r},\n"
        f"        \"feature\": {knob.get('feature')!r},\n"
        "    },\n"
    )


def generate(knobs: dict) -> str:
    import datetime
    out = [_HEADER.format(date=datetime.date.today().isoformat())]
    for key in sorted(knobs):
        out.append(_entry(knobs[key]))
    out.append("]\n")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                     help="call memsom.tuning in-process instead of reading stdin")
    args = ap.parse_args()

    if args.live:
        sys.path.insert(0, ".")
        from memsom import tuning
        knobs = tuning.as_json()
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            print("no input on stdin -- pipe `memsom tuning list --json` or use --live",
                  file=sys.stderr)
            return 1
        knobs = json.loads(raw)

    sys.stdout.write(generate(knobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
