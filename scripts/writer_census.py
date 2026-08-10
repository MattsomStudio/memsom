#!/usr/bin/env python3
"""writer_census -- every read-modify-write sequence is guarded or annotated.

"The one to not skip" (PLAN.md Phase 0): the entire mechanism behind "the
concurrency census is owned". A function counts as RMW if it issues a SELECT
against `nodes`/`edges` and, later in the same function, an INSERT/UPDATE/
DELETE against the same tables -- the shape MS-06 exploited at redact.py
(cascade_set's read, then the write, with no transaction between them).

An RMW function is ADJUDICATED if either:
  * the function body contains the literal "BEGIN IMMEDIATE" (it opens its own
    guarded transaction), or
  * the line immediately above `def` is `# RMW-OK: <reason>` (a single INSERT
    needs no guard; the annotation is where that decision is recorded).

Everything else is unadjudicated. Target: 0 (Phase 6). A bare single INSERT
with no preceding read is not RMW at all and is correctly not counted --
`writer_census.py`'s whole point is the read-then-write race, not every write.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_TABLES = ("nodes", "edges")
_ANNOTATION = "# RMW-OK:"


def _sql_const(node: ast.Call) -> str | None:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "execute"):
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    return node.args[0].value


def _touches(sql: str) -> bool:
    up = sql.upper()
    return any(t.upper() in up for t in _TABLES)


def _classify(sql: str) -> str | None:
    up = sql.strip().upper()
    if up.startswith("SELECT"):
        return "read"
    if up.startswith(("INSERT", "UPDATE", "DELETE")):
        return "write"
    return None


def find_unadjudicated() -> list[str]:
    hits: list[str] = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        src_lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(g.SRC).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            seen_read = False
            is_rmw = False
            has_begin_immediate = False
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and "BEGIN IMMEDIATE" in node.value.upper():
                    has_begin_immediate = True
                if isinstance(node, ast.Call):
                    sql = _sql_const(node)
                    if sql and _touches(sql):
                        kind = _classify(sql)
                        if kind == "read":
                            seen_read = True
                        elif kind == "write" and seen_read:
                            is_rmw = True
            if not is_rmw:
                continue
            annotated = has_begin_immediate
            if not annotated:
                lineno = fn.lineno
                # walk back past decorators to the line above the def/decorators
                start = lineno - 1
                for dec in fn.decorator_list:
                    start = min(start, dec.lineno - 1)
                prev = src_lines[start - 1].strip() if start >= 1 else ""
                annotated = prev.startswith(_ANNOTATION)
            if not annotated:
                hits.append(f"{rel}:{fn.lineno} {fn.name}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    hits = find_unadjudicated()
    for h in hits:
        print(f"  {h}")
    count = len(hits)
    print(f"unadjudicated RMW writers: {count}")

    if args.check:
        return g.check("writer_census.unadjudicated", count)
    g.record("writer_census.unadjudicated", count,
              "python scripts/writer_census.py",
              "control-tested: redact.py's cascade_set-then-write (MS-06, fixed Phase 1 "
              "via BEGIN IMMEDIATE) is the shape this detects; reverting that fix raises "
              "the count by 1",
              0,
              note="MEASURED in 00-chat-findings.md A4: 26 modules issue INSERT/UPDATE/"
                   "DELETE, 11 use BEGIN IMMEDIATE -- this script is the walk that was "
                   "missing for the rest, per Phase 6's writer-census item.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
