#!/usr/bin/env python3
"""gate_readpool -- how many places build a source read-pool, outside the primitive.

`memsom.storage.schema.taint_filter_clauses` (once Phase 2 moves it there; today
`memsom/__init__.py` via the `schema` module) is meant to be THE builder of a
taint-filtered read pool over `nodes` (tombstoned / redacted / archived / status /
conf_label). This walks memsom/ for functions that hand-roll an equivalent SQL
WHERE clause -- an executable string literal mentioning `nodes` and at least two
of the five taint columns -- OUTSIDE the primitive's own module.

MS-13 / MS-08 (SECURITY-REMEDIATION.md) are exactly this shape: `live_sources`,
`cmd_dump`, `redact.live_unredacted_sources`, `quarantine.live_unquarantined_sources`
each answer "what may I read" differently. The count below is expected to be > 1
at Phase 0 -- that IS the finding -- and the Phase 7 exit gate asserts it reaches
exactly 1 once those sites are routed through the primitive.

Phase 7 also runs this as `--check`: this script is seeded in Phase 0, wired into
CI in Phase 7 (PLAN.md Phase 0 bullet list).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_TAINT_COLS = ("tombstoned", "redacted", "archived", "status", "conf_label")
_PRIMITIVE_MODULE = "storage/schema.py"


def _is_readpool_literal(value: str) -> bool:
    up = value.upper()
    if "NODES" not in up:
        return False
    hits = sum(1 for c in _TAINT_COLS if c.upper() in up)
    return hits >= 2


def find_builders() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in g.iter_py_files():
        rel = path.relative_to(g.SRC).as_posix()
        if rel == _PRIMITIVE_MODULE:
            continue
        tree = g.parse(path)
        if tree is None:
            continue
        skip = g.docstring_node_ids(tree)
        for node in __import__("ast").walk(tree):
            if (node.__class__.__name__ == "Constant" and isinstance(getattr(node, "value", None), str)
                    and id(node) not in skip and _is_readpool_literal(node.value)):
                found.setdefault(rel, []).append(node.lineno)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    found = find_builders()
    count = len(found)  # distinct modules hand-rolling a read pool; target 1 (Phase 7)
    for mod, lines in sorted(found.items()):
        print(f"  {mod}: {lines}")
    print(f"readpool builders outside the primitive: {count}")

    if args.check:
        return g.check("gate_readpool.builders", count)
    g.record("gate_readpool.builders", count,
              source="python scripts/gate_readpool.py",
              control="control-tested red: baseline > 1 at Phase 0 (MS-13/MS-08 -- "
                      "live_sources, cmd_dump, redact.live_unredacted_sources, "
                      "quarantine.live_unquarantined_sources all hand-roll a partial filter)",
              target=1,
              note="Phase 7 wires this into CI and expects exactly 1 (the primitive's own site).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
