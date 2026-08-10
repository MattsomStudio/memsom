#!/usr/bin/env python3
"""fanin -- logic fan-in of a module: how much of it is real code vs re-export.

Phase 2 exit gate: `python scripts/fanin.py memsom/__init__.py` must print 0 once
the split lands and `__init__.py` becomes a pure facade. A def/class counts as
"real" unless its entire body is import statements, `pass`, a docstring, or a
single return/expression that just forwards to another name (`return mod.f(*a,
**kw)` / `x = mod.x`) -- the shapes a re-export facade actually takes.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _is_reexport_body(body: list[ast.stmt]) -> bool:
    stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if not stmts:
        return True
    if len(stmts) > 2:
        return False
    for s in stmts:
        if isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(s, ast.Return) and (s.value is None
                                           or isinstance(s.value, (ast.Call, ast.Attribute, ast.Name))):
            continue
        if isinstance(s, ast.Assign) and isinstance(s.value, (ast.Attribute, ast.Name, ast.Call)):
            continue
        return False
    return True


def logic_fanin(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not _is_reexport_body(node.body):
                count += 1
    return count


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: fanin.py <path/to/module.py>")
        return 2
    path = Path(sys.argv[1])
    n = logic_fanin(path)
    print(f"logic fan-in {path}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
