#!/usr/bin/env python3
"""shadow_falsepos -- false-positive rate in Gate #3's shadow log (Phase 9).

A "would-have-denied" decision the operator later marked `outcome: "ok"` (the
action was legitimate) is a false positive -- the reason shadow mode exists at
all (PLAN.md Phase 9: "a gate that wrongly blocks a Bash call mid-build gets
ripped out the same day"). This reports would-deny count, confirmed-bad count,
and the resulting rate per rule, so a rule can be tuned or dropped BEFORE it is
flipped to enforcing.

Same graceful-empty behaviour as shadow_summary.py: no log yet is 0 decisions,
not an error.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(Path.home() / ".claude" / "gate3_shadow.jsonl"))
    ap.add_argument("--check", action="store_true",
                     help="exit 1 if any rule's false-positive rate exceeds --max-rate")
    ap.add_argument("--max-rate", type=float, default=0.05)
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"{path} does not exist -- 0 decisions (Gate #3 ships in Phase 9)")
        return 0

    denies = defaultdict(int)
    false_pos = defaultdict(int)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("decision") != "deny":
            continue
        rule = row.get("rule", "?")
        denies[rule] += 1
        if row.get("outcome") == "ok":
            false_pos[rule] += 1

    worst = 0.0
    for rule in sorted(denies):
        rate = false_pos[rule] / denies[rule]
        worst = max(worst, rate)
        print(f"  rule={rule}: {false_pos[rule]}/{denies[rule]} false positives ({rate:.1%})")

    if not denies:
        print("0 would-deny decisions logged")
        return 0

    if args.check and worst > args.max_rate:
        print(f"REGRESSION: worst false-positive rate {worst:.1%} exceeds {args.max_rate:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
