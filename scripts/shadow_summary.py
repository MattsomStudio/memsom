#!/usr/bin/env python3
"""shadow_summary -- summarize Gate #3's shadow-mode decision log (Phase 9).

Gate #3 (broker/hook-pre/hook-post/capgate/policy) ships dark: it logs every
decision it WOULD have made and enforces nothing (PLAN.md Phase 9, Matt's Q2).
This reads that JSONL log (one decision per line: at least `action`, `decision`
in {"allow", "deny"}, `rule`) and prints allow/deny counts per rule -- the
input to "read the log, tune the policy, then flip to enforcing per action".

The log does not exist before Phase 9 builds the gate. A missing path is
reported as "0 decisions" and exits 0, not an error -- there is nothing to
summarize yet, which is a true and expected state, not a broken one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(Path.home() / ".claude" / "gate3_shadow.jsonl"))
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"{path} does not exist -- 0 decisions (Gate #3 ships in Phase 9)")
        return 0

    by_rule = Counter()
    by_decision = Counter()
    total = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        by_rule[row.get("rule", "?")] += 1
        by_decision[row.get("decision", "?")] += 1

    print(f"total shadow decisions: {total}")
    for decision, n in sorted(by_decision.items()):
        print(f"  decision={decision}: {n}")
    for rule, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"  rule={rule}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
