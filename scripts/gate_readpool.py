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

# Phase 7 refinement, control-tested against the 10 modules the unrefined
# substring scan flagged at that phase's start: a raw "NODES + >=2 taint
# columns" match also fires on shapes that are NOT a read-pool builder and
# were never meant to route through taint_filter_clauses --
#
#   - a single-row fetch by primary key ("WHERE id = ?" / "WHERE uuid = ?"):
#     a column LIST for one already-known row, not a pool.
#   - an INSERT statement: a column list for a write, not a read pool.
#   - an edges-table parent/ancestor walk (this codebase's "e.child"/"e.parent"
#     aliasing convention): a DAG traversal for one node, not a pool scan.
#   - an OR-joined predicate with no AND joining two taint columns: this is
#     an INCLUSION test ("is this node tainted") -- e.g. reflex.py's MS-01
#     backstop and federation.py's changeset export, which must ship dead
#     rows for revocation to propagate (DECISIONS-AND-DEVIATIONS.md C-1).
#     The primitive builds an AND-joined EXCLUSION clause; these are its
#     logical inverse, not a duplicate of it.
#   - a positive "status = 'quarantined'" listing: an audit view that must
#     SHOW quarantined rows -- the opposite of the primitive's own
#     "status != 'quarantined'" exclusion clause, which never appears
#     outside storage/schema.py.
#
# Measured against the Phase-7 baseline: applying these rules leaves exactly
# the genuine duplicate-pool sites flagged (which were then routed through
# the primitive, or deleted as dead partial-filter helpers per the MS-39
# precedent), matching the same text-classifier-vs-AST correction C-1
# already made for federation.py.
import re as _re
_SINGLE_ROW_RE = _re.compile(r"WHERE\s+(id|uuid)\s*=\s*\?", _re.IGNORECASE)
_POS_QUARANTINED_RE = _re.compile(r"STATUS\s*=\s*'QUARANTINED'", _re.IGNORECASE)
_NEG_QUARANTINED_RE = _re.compile(r"STATUS\s*!=\s*'QUARANTINED'", _re.IGNORECASE)


def _is_readpool_literal(value: str) -> bool:
    up = value.upper()
    if "NODES" not in up:
        return False
    hits = sum(1 for c in _TAINT_COLS if c.upper() in up)
    if hits < 2:
        return False
    if up.lstrip().startswith("INSERT"):
        return False
    if _SINGLE_ROW_RE.search(value):
        return False
    if "E.CHILD" in up or "E.PARENT" in up:
        return False
    if _POS_QUARANTINED_RE.search(value) and not _NEG_QUARANTINED_RE.search(value):
        return False
    if " OR " in up and " AND " not in up:
        return False
    return True


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
                      "quarantine.live_unquarantined_sources all hand-roll a partial filter). "
                      "Phase 7 (layers land): the 10 modules the unrefined classifier flagged "
                      "were adjudicated one by one -- distill.py's export_training and two dead "
                      "partial-filter helpers (confid.sources_for_clearance, "
                      "quarantine.live_source_ids, zero production consumers) were the only "
                      "genuine duplicates, now fixed/deleted; the rest were false positives "
                      "(single-row-by-key fetches, edges-table DAG walks, OR-joined inclusion "
                      "predicates, positive quarantined-status listings) and the classifier was "
                      "refined to stop flagging them, matching C-1's federation.py precedent.",
              target=0,
              note="The scan excludes storage/schema.py (the primitive's own module) by "
                   "construction, so 0 outside it is the fully-consolidated state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
