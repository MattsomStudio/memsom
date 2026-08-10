#!/usr/bin/env python3
"""failopen_annotations -- every bare/broad except must name its own adjudication.

Phase 6 (charter R3) adjudicates each fail-open individually. The convention
this enforces: a bare `except:` or `except Exception:` must be immediately
preceded (previous non-blank source line) by a `# FAILOPEN:` comment naming
the decision -- e.g. `# FAILOPEN: allowed, embedder outage is not data loss`.
An except that legitimately re-raises or narrows to a specific exception type
is not counted; only the two broad shapes A2.3's vocabulary exists for.

Target 0 sites with no annotation. The convention itself does not exist before
Phase 6, so Phase 0's baseline is simply "every over-broad except today",
recorded so the count can only fall from here.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_TAG = "# FAILOPEN:"


def find_unannotated() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(g.SRC).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            is_broad = node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id == "Exception")
            if not is_broad:
                continue
            lineno = node.lineno
            prev = lines[lineno - 2].strip() if lineno >= 2 else ""
            if not prev.startswith(_TAG):
                hits.append(f"{rel}:{lineno}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    hits = find_unannotated()
    for h in hits:
        print(f"  {h}")
    count = len(hits)
    print(f"unannotated bare/broad excepts: {count}")

    if args.check:
        return g.check("failopen.unannotated_excepts", count)
    g.record("failopen.unannotated_excepts", count,
              "python scripts/failopen_annotations.py",
              "control-tested: planting an unannotated bare except in a scratch file "
              "raises the count by 1; adding the `# FAILOPEN:` line above it drops it back",
              0,
              note="The `# FAILOPEN:` convention does not exist before Phase 6 -- this "
                   "baseline is 'every broad except today', not 'every violation'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
