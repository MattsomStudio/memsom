#!/usr/bin/env python3
"""env_ratchet -- os.environ reads outside memsom/tuning.py.

Phase 8 (A2.2) centralizes every tunable knob's env lookup in `tuning.py` (a
new top-level module -- see its docstring for why it lives there); a site
anywhere else reading `os.environ`/`os.getenv` directly is a knob the
registry does not know about.

Four exceptions are named, not oversights -- the first three documented at
length in memsom/tuning.py's module docstring:

* MEMDAG_HOME / MEMDAG_DB, anywhere they appear (storage/db.py's own
  resolution, plus the handful of callers that pin or default them --
  interface/audit.py, federation/broker.py's --selfcheck) -- the store
  location has to resolve before there is a data dir to look for a
  canonical.json override in, so tuning cannot be the thing that resolves
  it without a circular import. This is the same variable pair Sec6.0.1
  calls "pin the store FIRST" and treats specially throughout the plan.
* kernel/paths.py (MEMDAG_BRIDGE_MEMORY_DIR) -- kernel is rank 0 and cannot
  import tuning upward. The knob is still registered in tuning.py (visible
  to `tuning list`); kernel/paths.py keeps its own independent read.
* kernel/syncguard.py (MEMSOM_EXTRA_SYNC_MARKERS, plus the OneDrive/
  OneDriveCommercial/OneDriveConsumer sync-root env vars, Phase 10) -- same
  reason as kernel/paths.py above: rank 0, cannot import tuning. The custom-
  marker knob is registered in tuning.py as `storage.sync_extra_markers` for
  `tuning list` visibility; the OneDrive vars are OS/vendor state, not a
  memsom knob, same category as $PATH below.
* Any file's read of $PATH (effects/proc.py today) -- not a memsom knob, the
  OS executable search path, the same category `shutil.which` reads.
* childenv.py copies the WHOLE `os.environ` mapping (`dict(os.environ)`) to
  build a filtered/minimal environment for a spawned child -- a security
  boundary (credential-name stripping), not a named knob lookup. There is no
  single key here for tuning.py to own.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_OWNER = "tuning.py"
_BOOTSTRAP_EXEMPT_FILES = {"kernel/paths.py", "childenv.py", "kernel/syncguard.py"}
_BOOTSTRAP_EXEMPT_VARS = {"PATH", "MEMDAG_HOME", "MEMDAG_DB"}


class _ParentVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parents: dict[int, ast.AST] = {}

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parents[id(child)] = node
        super().generic_visit(node)


def _literal_var_name(node, parents: dict[int, ast.AST]) -> str | None:
    """Given the os.environ/getenv node, find the literal env-var-name string
    argument in its immediate call/subscript, if any."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.Subscript):  # os.environ["PATH"]
        key = parent.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value
        return None
    if isinstance(parent, ast.Attribute):  # os.environ.get(...) / .setdefault(...)
        call = parents.get(id(parent))
        if isinstance(call, ast.Call) and call.args:
            arg0 = call.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                return arg0.value
        return None
    if isinstance(parent, ast.Call):  # os.getenv("PATH", ...) -- node IS the Call
        if parent.args:
            arg0 = parent.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                return arg0.value
    return None


def find_reads() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        rel = path.relative_to(g.SRC).as_posix()
        if rel == _OWNER or rel in _BOOTSTRAP_EXEMPT_FILES:
            continue
        tree = g.parse(path)
        if tree is None:
            continue
        pv = _ParentVisitor()
        pv.visit(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                if isinstance(node.value, (ast.Attribute, ast.Name)):
                    if _literal_var_name(node, pv.parents) in _BOOTSTRAP_EXEMPT_VARS:
                        continue
                    hits.append(f"{rel}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "getenv":
                if _literal_var_name(node, pv.parents) in _BOOTSTRAP_EXEMPT_VARS:
                    continue
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
              "MEASURED at Phase 8: every remaining bare os.environ/getenv site "
              "outside tuning.py is one of the four named exceptions.",
              0,
              note="Phase 8 built memsom/tuning.py and migrated every knob read "
                   "into it; the exceptions are named and justified in the "
                   "module's own docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
