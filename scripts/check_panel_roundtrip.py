#!/usr/bin/env python3
"""check_panel_roundtrip -- every canonical tuning knob round-trips.

Phase 8 exit gate: `memsom tuning list --json | python scripts/check_panel_roundtrip.py`.
Reads a JSON array of knob objects from stdin, each expected to carry at least
`name`, `type` and `default`, and asserts `parse(format(default)) == default`
for every knob -- the round-trip a panel control (a text box, a toggle) relies
on to not silently coerce a value on save.

`memsom tuning` does not exist before Phase 8. Empty/absent stdin is reported
as "0 knobs checked" and exits 0 rather than crashing a pipeline that has
nothing to feed it yet.
"""

from __future__ import annotations

import json
import sys


def _roundtrips(knob: dict) -> bool:
    default = knob.get("default")
    kind = knob.get("type", "str")
    text = str(default)
    try:
        if kind == "int":
            back = int(text)
        elif kind == "float":
            back = float(text)
        elif kind == "bool":
            back = text.strip().lower() in ("1", "true", "yes", "on")
        else:
            back = text
    except (ValueError, TypeError):
        return False
    return back == default


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        print("0 knobs on stdin -- `memsom tuning list --json` does not exist before "
              "Phase 8")
        return 0
    try:
        knobs = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"invalid JSON on stdin: {exc}")
        return 1
    if isinstance(knobs, dict):
        knobs = list(knobs.values())

    failures = [k.get("name", "?") for k in knobs if not _roundtrips(k)]
    for name in failures:
        print(f"  ROUND-TRIP FAILED: {name}")
    print(f"knobs checked: {len(knobs)}, failed: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
