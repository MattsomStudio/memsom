#!/usr/bin/env python3
"""gate_writeowner -- how many call sites mint a node outside the one write path.

Phase 4 (A1.4) makes `ingest` the single stamping write path for new nodes: every
caller goes through it instead of `insert_node` directly, so `MEMDAG_CHANNEL_CEILING`
and the Biba floor cannot be bypassed by a caller that skips the wrapper (MS-20).

This counts distinct MODULES (not call sites -- one module calling insert_node
five times is one owner decision, not five) that call `insert_node(` directly,
outside `memsom/__init__.py` (which defines it) and outside the eventual
`ingest` module itself. Phase 4's exit gate expects exactly 1 (the ingest
module's own internal call).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

# Both paths below were Phase-0 guesses, written before the concrete moves
# landed. `insert_node`/`derive_node` moved to integrity/dag.py in Phase 2
# (the core split); the write path landed at integrity/ingest.py in Phase 4
# (PLAN.md Sec1.4/Sec1.5), not the placeholder `kernel/ingest.py` -- ingest's
# write-path logic is not pure, so kernel/ was never the right layer for it.
_DEFINER = "integrity/dag.py"          # defines insert_node; derive_node's own internal call lives here too
_FUTURE_OWNER = "integrity/ingest.py"  # the one write path (PLAN.md Sec1.4)


def find_direct_callers() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in g.iter_py_files():
        rel = path.relative_to(g.SRC).as_posix()
        if rel in (_DEFINER, _FUTURE_OWNER):
            continue
        tree = g.parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, (ast.Name, ast.Attribute))
                    and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) == "insert_node"):
                found.setdefault(rel, []).append(node.lineno)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    found = find_direct_callers()
    count = len(found)  # distinct modules bypassing the single write path
    for mod, lines in sorted(found.items()):
        print(f"  {mod}: {lines}")
    print(f"modules calling insert_node directly (outside the write path): {count}")

    if args.check:
        return g.check("gate_writeowner.direct_callers", count)
    g.record("gate_writeowner.direct_callers", count,
              source="python scripts/gate_writeowner.py",
              control="control-tested red: baseline > 1 at Phase 0 (bridge_import.py, "
                      "obsidian.py and others each call insert_node directly; MS-20)",
              target=1,
              note="Phase 4 builds the `ingest` single write path (A1.4) and this should "
                   "collapse to 1: the ingest module's own call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
