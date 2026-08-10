#!/usr/bin/env python3
"""upward_imports -- try-wrapped imports of a higher layer from a lower one.

Phase 4 exit gate target: 0. Uses the same rank table `.importlinter-goals`
documents (interface > bridge > federation/distill > lifecycle > retrieval >
integrity > storage), and looks specifically for `try: import ...` /
`try: from ... import ...` blocks -- the shape Phase 4 converts into event
subscribers (A1.4) -- naming a module at a STRICTLY higher rank than the file
it appears in. An unwrapped (non-try) upward import is a separate, harder
failure `.importlinter` already catches; this script is about the soft,
silently-degrading kind.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_RANK = {
    "interface": 0, "bridge": 1, "federation": 2, "distill": 2,
    "lifecycle": 3, "retrieval": 4, "integrity": 5, "storage": 6,
    "kernel": 7, "effects": 7,
}


def _layer_of(rel: str) -> str | None:
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 else None


def _imported_module_layer(node: ast.stmt) -> str | None:
    names: list[str] = []
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module]
    for n in names:
        if n.startswith("memsom."):
            parts = n.split(".")
            if len(parts) > 1:
                return parts[1]
    return None


def find_violations() -> list[str]:
    hits: list[str] = []
    for path in g.iter_py_files():
        rel = path.relative_to(g.SRC).as_posix()
        layer = _layer_of(rel)
        if layer is None or layer not in _RANK:
            continue
        tree = g.parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for stmt in node.body:
                imp_layer = _imported_module_layer(stmt)
                if imp_layer and imp_layer in _RANK and _RANK[imp_layer] < _RANK[layer]:
                    hits.append(f"{rel}:{stmt.lineno} try-imports memsom.{imp_layer} "
                                f"(rank {_RANK[imp_layer]} < {layer}'s rank {_RANK[layer]})")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    hits = find_violations()
    for h in hits:
        print(f"  {h}")
    count = len(hits)
    print(f"try-wrapped upward imports: {count}")

    if args.check:
        return g.check("upward_imports.count", count)
    g.record("upward_imports.count", count,
              source="python scripts/upward_imports.py",
              control="control-tested by construction: a planted try:-import of a strictly "
                      "higher-ranked package is counted (see test coverage added alongside).",
              target=0,
              note="Phase 4 converts these into event subscribers (A1.4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
