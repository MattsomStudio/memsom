#!/usr/bin/env python3
"""fact_refs --check -- every measured value in a project_memsom_* memory resolves
to a fact_* file (Q11, dogfooding memsom's own fact layer on itself, Phase 8).

The memory store this checks is EXTERNAL to this repo (Matt's `~/.claude`
memory directory), so it is a path argument / env var, not a repo-relative
walk like every other script here. Phase 8 is the first phase that has
anything to check: no `fact_memsom_*` files exist yet, so a missing or empty
memory dir is reported as "0 checked" and exits 0 -- absence is not failure
before the deliverable exists, it only becomes one once Phase 8 ships facts
and this stops finding zero.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_INLINE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_FACT_REF = re.compile(r"\[\[(fact_[a-z0-9_]+)\]\]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--memory-dir",
                     default=os.environ.get("MEMSOM_MEMORY_DIR", ""),
                     help="directory of project_*.md / fact_*.md memory files")
    args = ap.parse_args()

    if not args.memory_dir:
        print("no --memory-dir / MEMSOM_MEMORY_DIR set -- 0 checked (Phase 8 has not "
              "shipped fact_memsom_* files yet)")
        return 0

    root = Path(args.memory_dir)
    if not root.is_dir():
        print(f"memory dir {root} does not exist -- 0 checked")
        return 0

    fact_files = {p.stem for p in root.glob("fact_*.md")}
    violations: list[str] = []
    checked = 0
    for path in sorted(root.glob("project_memsom_*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        refs = set(_FACT_REF.findall(text))
        checked += 1
        for ref in refs:
            if ref not in fact_files:
                violations.append(f"{path.name}: [[{ref}]] has no {ref}.md")
        # a bare number with no [[fact_*]] anywhere in the file is an inline
        # duplicate the plan wants routed through the fact layer instead.
        if _INLINE_NUMBER.search(text) and not refs:
            violations.append(f"{path.name}: contains a bare measured number with "
                              f"no [[fact_*]] reference")

    for v in violations:
        print(f"  {v}")
    print(f"project_memsom_* files checked: {checked}, violations: {len(violations)}")
    return 1 if (args.check and violations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
