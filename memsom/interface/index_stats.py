"""memsom index-stats — per-section line/byte counts of MEMORY.md vs budgets.

Anti-creep, mechanism 4: the numbers that make drift visible.  Reads the
on-disk MEMORY.md (the always-loaded index), the live caps from the store's
canonical.json params (`memory_budget`, `memory_max_lines`, `section_budgets`)
and the last render's shed receipt (`.weights/shed.json`), and prints one line
per section.  Read-only; no store connection needed.

  memsom index-stats            # human table
  memsom index-stats --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memsom.bridge.bridge_import import default_memory_dir
from memsom.bridge.bridge_render import section_table
from memsom.distill.digest import (resolve_budget, resolve_max_lines,
                                   resolve_section_budgets, section_stats)


def index_stats(memory_dir) -> dict:
    memory_dir = Path(memory_dir)
    md = memory_dir / "MEMORY.md"
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    shed = {}
    try:
        data = json.loads((memory_dir / ".weights" / "shed.json").read_text(encoding="utf-8"))
        shed = data.get("by_reason") or {}
    except Exception:  # noqa: BLE001 — receipt is optional
        shed = {}
    return {
        "memory_dir": str(memory_dir),
        "bytes": len(text.encode("utf-8")),
        "budget": resolve_budget(memory_dir),
        "lines": text.count("\n"),
        "max_lines": resolve_max_lines(memory_dir),
        "sections": section_table(section_stats(text), resolve_section_budgets(memory_dir)),
        "shed_by_reason": shed,
    }


def format_stats(st) -> str:
    out = [f"MEMORY.md: {st['lines']}/{st['max_lines']} lines, "
           f"{st['bytes']}/{st['budget']} bytes"]
    for sec, s in st["sections"].items():
        cap = f" (budget {s['budget']})" if s["budget"] is not None else ""
        over = "  OVER" if s["budget"] is not None and s["bytes"] > s["budget"] else ""
        out.append(f"  {sec}: {s['lines']} lines / {s['bytes']} bytes{cap}{over}")
    if st["shed_by_reason"]:
        out.append("shed (last render): " + ", ".join(
            f"{k}={v}" for k, v in sorted(st["shed_by_reason"].items())))
    return "\n".join(out) + "\n"


def _cmd(args):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    mem = Path(args.memory_dir) if args.memory_dir else default_memory_dir()
    st = index_stats(mem)
    if args.json:
        print(json.dumps(st, indent=2))
    else:
        sys.stdout.write(format_stats(st))
    return 0


def register(sub) -> None:
    p = sub.add_parser("index-stats",
                       help="per-section line/byte counts of MEMORY.md against their budgets")
    p.add_argument("memory_dir", nargs="?", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="memsom_index_stats", description=__doc__)
    ap.add_argument("memory_dir", nargs="?", default=None)
    ap.add_argument("--json", action="store_true")
    sys.exit(_cmd(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
