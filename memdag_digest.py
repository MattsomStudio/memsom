"""memdag_digest — render the always-on MEMORY.md digest from memdag (Phase 3).

This is the piece that lets memdag be the store-of-record while the harness-native
always-on MEMORY.md survives: it queries the bridge-imported memory nodes and
renders the sectioned `- [Title](file.md) — hook` index the Claude Code harness
loads each session.

Selection (the forgetting layer decides what's "hot enough" to inject):
  - literal nodes (the file-less hand-authored index lines)  -> always rendered.
  - endorsed (pinned: user_/feedback_/personal_)             -> always rendered.
  - user-channel (project_/reference_) with forget_tier='hot' -> rendered.
  - user-channel 'cold' / un-sectioned                        -> dropped (still in
                                                                 the store, just
                                                                 out of context).

Budget: the rendered file must be <= 16,384 bytes (the harness loads it in full).
If over, the lowest-RS user lines are dropped first; pinned + literal lines are
never dropped.  If pinned+literal alone exceed the cap, DigestTooLarge is raised
(surfaced, never silently truncated).

SHADOW mode (Phase 3): write_shadow writes MEMORY.memdag.md NEXT TO the real
MEMORY.md (never overwrites it).  compare_index does the per-section file-set
equality check that is the cutover GO criterion.

Frozen core untouched; read-only over the DB (render/compare never write nodes).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import memdag
import memdag_schema
from memdag_bridge_import import (split_frontmatter, fm_top_level,
                                  parse_index_entries, parse_primary_index)

# section order must match the hand-curated MEMORY.md taxonomy
SECTIONS = [
    "About the User",
    "Personal context",
    "Hardware",
    "Current Setup & Learning",
    "Work / redacted Group",
    "Personal projects",
    "References",
    "Feedback",
]
BUDGET = 16384
DEFAULT_TITLE = "# Memory - Matthew Somanlall"


class DigestTooLarge(Exception):
    """Raised when pinned + literal content alone exceeds the byte budget."""


# --- read the bridge nodes ----------------------------------------------------

def _rows(conn):
    has_tier = memdag_schema.column_exists(conn, "nodes", "forget_tier")
    has_rs = memdag_schema.column_exists(conn, "nodes", "forget_rs")
    tcol = "forget_tier" if has_tier else "NULL AS forget_tier"
    rcol = "forget_rs" if has_rs else "NULL AS forget_rs"
    return conn.execute(
        f"SELECT content, channel, source_ref, {tcol}, {rcol} FROM nodes "
        "WHERE tombstoned = 0 AND source_ref LIKE 'memory:%'"
    ).fetchall()


def _entry(content, channel, sref, tier, rs):
    fm_lines, body, _ = split_frontmatter(content or "")
    fm = fm_top_level(fm_lines)
    is_literal = (sref.startswith("memory:literal:")
                  or str(fm.get("literal", "")).lower() in ("true", "1", "yes"))
    section = fm.get("section") or None
    if is_literal:
        return {"kind": "literal", "section": section, "line": body.strip(),
                "channel": channel}
    stem = sref.split(":", 1)[1] if sref.startswith("memory:") else sref
    # prefer the curated MEMORY.md title + hook (terser, byte-matches the
    # hand-maintained index); fall back to frontmatter name + a LENGTH-CAPPED
    # description so a node imported without curated text can't bloat the file.
    name = fm.get("index_title") or fm.get("name", stem)
    hook = fm.get("index_hook")
    if not hook:
        d = fm.get("description", "")
        hook = (d[:70].rstrip() + "…") if len(d) > 71 else d
    return {"kind": "file", "section": section, "stem": stem,
            "name": name, "desc": hook,
            "pinned": channel == "endorsed", "tier": tier or "hot",
            "rs": rs, "channel": channel}


def _select_hot(entries):
    """Entries that belong in the always-on digest."""
    out = []
    for e in entries:
        if e["kind"] == "literal":
            out.append(e)                      # literals always render
        elif e["section"] and (e["pinned"] or e["tier"] == "hot"):
            out.append(e)                      # sectioned + (pinned or hot)
    return out


def _assemble(title, entries):
    by_sec = {}
    for e in entries:
        by_sec.setdefault(e["section"], []).append(e)
    lines = [title, ""]
    order = SECTIONS + sorted(s for s in by_sec if s and s not in SECTIONS)
    for sec in order:
        if sec not in by_sec:
            continue
        lines.append(f"## {sec}")
        for e in [x for x in by_sec[sec] if x["kind"] == "literal"]:
            lines.append(e["line"])
        for e in sorted([x for x in by_sec[sec] if x["kind"] == "file"],
                        key=lambda x: x["stem"]):
            hook = f" — {e['desc']}" if e["desc"] else ""
            lines.append(f"- [{e['name']}]({e['stem']}.md){hook}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_digest(conn, *, title=None, budget=BUDGET):
    """Render the MEMORY.md digest string from the live bridge nodes."""
    title = title or os.environ.get("MEMDAG_DIGEST_TITLE", DEFAULT_TITLE)
    hot = _select_hot([_entry(*r) for r in _rows(conn)])
    # droppable = non-pinned user files, lowest RS first (dropped in THIS order)
    droppable = sorted([e for e in hot if e["kind"] == "file" and not e["pinned"]],
                       key=lambda e: (e["rs"] if e["rs"] is not None else 0.0))
    dropped = set()  # ids of dropped entries
    while True:
        text = _assemble(title, [e for e in hot if id(e) not in dropped])
        if len(text.encode("utf-8")) <= budget:
            return text
        nxt = next((e for e in droppable if id(e) not in dropped), None)
        if nxt is None:
            raise DigestTooLarge(
                f"pinned + literal content alone exceeds {budget} bytes")
        dropped.add(id(nxt))


def write_shadow(conn, memory_dir, *, name="MEMORY.memdag.md", title=None):
    """Render and write the SHADOW digest next to the real MEMORY.md.

    Never touches the real MEMORY.md.  Returns (path, text).
    """
    text = render_digest(conn, title=title)
    path = Path(memory_dir) / name
    path.write_text(text, encoding="utf-8")
    return path, text


# --- verification (the cutover GO check) -------------------------------------

def index_sets(text):
    """{section: {"files": set(filenames), "literals": set(lines)}} from an index.

    Files counted are PRIMARY entries only (line-leading), so secondary inline
    links — which the digest never renders as their own line — are excluded on
    both sides of the comparison.  Literals come from the full entry parse.
    """
    out = {}
    for fname, (title, hook, section) in parse_primary_index(text).items():
        out.setdefault(section, {"files": set(), "literals": set()})["files"].add(fname)
    for sec, kind, payload in parse_index_entries(text):
        if kind == "literal":
            out.setdefault(sec, {"files": set(), "literals": set()})["literals"].add(payload)
    return out


def compare_index(real_text, shadow_text):
    """Per-section diff of FILE sets between two indexes (the GO criterion:
    'same files present per section').  Returns {} when equivalent.

    Each non-empty section entry reports missing_files (in real, absent from
    shadow) and extra_files (in shadow, absent from real).
    """
    a, b = index_sets(real_text), index_sets(shadow_text)
    diffs = {}
    for sec in sorted(set(a) | set(b)):
        af = a.get(sec, {}).get("files", set())
        bf = b.get(sec, {}).get("files", set())
        missing, extra = af - bf, bf - af
        if missing or extra:
            diffs[sec] = {"missing_files": sorted(missing),
                          "extra_files": sorted(extra)}
    return diffs


# --- CLI ----------------------------------------------------------------------

def _default_memory_dir():
    env = os.environ.get("MEMDAG_BRIDGE_MEMORY_DIR")
    if env:
        return Path(env)
    return Path.home() / ".claude" / "projects" / "C--Users-you" / "memory"


def _cmd_shadow(args):
    conn = memdag.get_connection()
    try:
        mem = Path(args.memory_dir) if args.memory_dir else _default_memory_dir()
        path, text = write_shadow(conn, mem)
        size = len(text.encode("utf-8"))
        print(f"[digest] wrote shadow {path} ({size} / {BUDGET} bytes)")
        real = mem / "MEMORY.md"
        if real.exists():
            diffs = compare_index(real.read_text(encoding="utf-8"), text)
            if not diffs:
                print("[digest] per-section file sets: EQUIVALENT to MEMORY.md ✓")
            else:
                print(f"[digest] per-section file-set DIFFERENCES in {len(diffs)} section(s):")
                for sec, d in diffs.items():
                    if d["missing_files"]:
                        print(f"  [{sec}] missing from shadow: {d['missing_files']}")
                    if d["extra_files"]:
                        print(f"  [{sec}] extra in shadow: {d['extra_files']}")
    finally:
        conn.close()


def register(sub) -> None:
    p = sub.add_parser("digest-shadow",
                       help="render MEMORY.memdag.md shadow + diff vs MEMORY.md (Phase 3)")
    p.add_argument("memory_dir", nargs="?", default=None)
    p.set_defaults(func=_cmd_shadow)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="memdag_digest", description=__doc__)
    ap.add_argument("memory_dir", nargs="?", default=None)
    _cmd_shadow(ap.parse_args(argv))


if __name__ == "__main__":
    main()
