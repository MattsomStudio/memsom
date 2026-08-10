#!/usr/bin/env python3
"""effects_ratchet -- three counts the effects layer (Phase 5, charter R1) owns.

Default: modules importing subprocess / urllib directly (target: 1 each, once
effects/proc.py and effects/net.py absorb every site).
--connect: bare `sqlite3.connect(` call sites outside storage/db.py (target: 2
at Phase 5 per PLAN.md -- dashboard.py's three strays are gone by then,
doctor.py:38 is routed through get_connection).
--timeouts: subprocess spawn calls with no explicit `timeout=` keyword
(target: 0 -- MS-29's class of bug, a hang with no operator signal).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_SPAWN_FUNCS = {"run", "Popen", "call", "check_call", "check_output"}


def _module_imports(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name == name for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == name:
            return True
    return False


def modules_importing(name: str) -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is not None and _module_imports(tree, name):
            hits.append(path.relative_to(g.SRC).as_posix())
    return hits


def connect_sites() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        rel = path.relative_to(g.SRC).as_posix()
        if rel == "storage/db.py":
            continue
        tree = g.parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "connect"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "sqlite3"):
                hits.append(f"{rel}:{node.lineno}")
    return hits


def spawns_without_timeout() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _SPAWN_FUNCS
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"):
                continue
            if not any(kw.arg == "timeout" for kw in node.keywords):
                rel = path.relative_to(g.SRC).as_posix()
                hits.append(f"{rel}:{node.lineno}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", action="store_true")
    ap.add_argument("--timeouts", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.connect:
        hits = connect_sites()
        for h in hits:
            print(f"  {h}")
        print(f"sqlite3.connect sites outside storage/db.py: {len(hits)}")
        if args.check:
            return g.check("effects.sqlite3_connect_sites", len(hits))
        g.record("effects.sqlite3_connect_sites", len(hits),
                  "python scripts/effects_ratchet.py --connect",
                  "control-tested: dashboard.py's 3 + doctor.py's 1 (+1 elsewhere) at Phase 0",
                  2)
        return 0

    if args.timeouts:
        hits = spawns_without_timeout()
        for h in hits:
            print(f"  {h}")
        print(f"subprocess spawns without an explicit timeout: {len(hits)}")
        if args.check:
            return g.check("effects.spawns_without_timeout", len(hits))
        g.record("effects.spawns_without_timeout", len(hits),
                  "python scripts/effects_ratchet.py --timeouts",
                  "control-tested: MS-29 (bare `git` resolve) is one of these at Phase 0",
                  0)
        return 0

    sub = modules_importing("subprocess")
    url = modules_importing("urllib") + modules_importing("urllib.request")
    url = sorted(set(url))
    for h in sub:
        print(f"  subprocess: {h}")
    for h in url:
        print(f"  urllib: {h}")
    print(f"modules importing subprocess: {len(sub)}")
    print(f"modules importing urllib: {len(url)}")
    if args.check:
        r1 = g.check("effects.subprocess_importers", len(sub))
        r2 = g.check("effects.urllib_importers", len(url))
        return r1 or r2
    g.record("effects.subprocess_importers", len(sub),
              "python scripts/effects_ratchet.py", "baseline recorded fresh at Phase 0", 1)
    g.record("effects.urllib_importers", len(url),
              "python scripts/effects_ratchet.py", "baseline recorded fresh at Phase 0", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
