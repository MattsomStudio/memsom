#!/usr/bin/env python3
"""env_ratchet -- os.environ reads outside memsom/interface/tuning.py.

Phase 8 (A2.2) centralizes every tunable knob's env lookup in `tuning.py`; a
site anywhere else reading `os.environ`/`os.getenv` directly is a knob the
registry does not know about. `tuning.py` does not exist yet -- the exclusion
is a no-op today, and the baseline below is "every os.environ read in the repo
right now", which is exactly the number Phase 8's own work has to absorb.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_OWNER = "interface/tuning.py"


def find_reads() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        rel = path.relative_to(g.SRC).as_posix()
        if rel == _OWNER:
            continue
        tree = g.parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("environ",):
                if isinstance(node.value, ast.Attribute) or isinstance(node.value, ast.Name):
                    hits.append(f"{rel}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "getenv":
                hits.append(f"{rel}:{node.lineno}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    hits = find_reads()
    for h in hits:
        print(f"  {h}")
    count = len(hits)
    print(f"os.environ reads outside {_OWNER}: {count}")

    if args.check:
        return g.check("env_ratchet.reads_outside_tuning", count)
    g.record("env_ratchet.reads_outside_tuning", count,
              "python scripts/env_ratchet.py",
              "tuning.py does not exist before Phase 8; this baseline is the full-repo "
              "count today, not yet a real violation total",
              0,
              note="Phase 8 builds interface/tuning.py; the target of 0 only becomes "
                   "meaningful once every knob has migrated there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
