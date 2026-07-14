"""memsom.bridge.facts — read-time resolution of [[fact_*]] references (Phases 2-3).

docs/facts-design.md: a fact lives ONCE as a fact_*.md memory (value/unit/
last-verified frontmatter); every other memory stores only the reference
`[[fact_<stem>]]`. Stored memory content NEVER changes when a fact does —
this module substitutes the value wherever memory content becomes *visible*
(the MEMORY.md digest, the retrieve output), so staleness is structurally
impossible for anything that references instead of quotes.

The supersede chain (tombstoned predecessors of the same bridge_path) is the
fact's history. Resolution modes:
  - current (no as_of):        "61 tok/s"
  - as_of predates an update:  "61 tok/s (was 45 tok/s when written, 2026-07-17)"
  - retired (no live version): "45 tok/s (last known — fact retired 2026-07-30)"
  - no versions at all:        the raw [[link]] is left in place (a typo'd
                               reference should LOOK broken, not resolve).

Read-only over the DB; never writes nodes.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import memsom
from memsom.bridge.bridge_import import (
    split_frontmatter, fm_top_level, stamp_fm, default_memory_dir,
    migrate as bridge_migrate,
)
from memsom.storage import schema as memsom_schema

# Only fact_ links resolve; ordinary [[wikilinks]] stay untouched (they are
# navigation, not variables). Stems are FILENAME stems (underscored), per the
# Phase 0 resolver finding — [[fact_pc_gpu]] for fact_pc_gpu.md.
FACT_REF_RE = re.compile(r"\[\[(fact_[A-Za-z0-9_-]+)\]\]")

# The reconcile sweep's generic reason adds nothing a reader can use; a custom
# retire reason (a future `memsom fact retire`) would.
_GENERIC_RETIRE = "source file removed (bridge reconcile)"


def _fm_of(content: str) -> dict:
    return fm_top_level(split_frontmatter(content or "")[0])


def fact_versions(conn: sqlite3.Connection, stem: str) -> list[dict]:
    """Every version of fact *stem*, oldest first (the supersede chain).

    Keyed on bridge_path = "<stem>.md" — the same identity the importer uses,
    so a renamed `name:` slug can't detach a fact from its history. Redacted
    versions are excluded (their payload is destroyed; resolving through them
    would resurrect nothing and imply a value that no longer exists).
    """
    if not memsom_schema.column_exists(conn, "nodes", "bridge_path"):
        return []  # store predates the bridge — no facts can exist
    redacted_clause = ""
    if memsom_schema.column_exists(conn, "nodes", "redacted"):
        redacted_clause = " AND (redacted IS NULL OR redacted = 0)"
    rows = conn.execute(
        "SELECT id, content, created_at, tombstoned, tombstoned_at, revoke_reason "
        "FROM nodes WHERE bridge_path = ?" + redacted_clause + " ORDER BY id",
        (f"{stem}.md",),
    ).fetchall()
    out = []
    for nid, content, created_at, tombstoned, tombstoned_at, revoke_reason in rows:
        fm = _fm_of(content)
        if (fm.get("type") or "").strip() != "fact":
            continue  # same path, but not a fact version (defensive)
        out.append({
            "nid": nid,
            "value": fm.get("value"),
            "unit": fm.get("unit"),
            "last_verified": fm.get("last-verified"),
            "created_at": created_at,
            "tombstoned": bool(tombstoned),
            "tombstoned_at": tombstoned_at,
            "revoke_reason": revoke_reason,
        })
    return out


def _fmt(v: dict) -> str:
    if v["value"] is None:
        return ""
    return f"{v['value']} {v['unit']}" if v.get("unit") else str(v["value"])


def _date(iso: str | None) -> str:
    return (iso or "")[:10]


def resolve_ref(conn: sqlite3.Connection, stem: str, *, as_of: str | None = None) -> str | None:
    """The display text for one fact reference, or None (leave the raw link).

    ISO-8601 UTC strings throughout (memsom.now_iso), so plain string
    comparison is chronological. *as_of* is the referencing memory's
    created_at: the as-of version is the LAST version created at or before it
    (a memory written before the fact existed shows the fact's first known
    value — the earliest truth is closer than no truth).
    """
    versions = fact_versions(conn, stem)
    if not versions:
        return None
    live = next((v for v in versions if not v["tombstoned"]), None)

    if live is None:
        # Retired: last known value, flagged with when (and why, if the reason
        # says more than the sweep's boilerplate).
        last = versions[-1]
        txt = _fmt(last)
        if not txt:
            return None
        when = _date(last["tombstoned_at"])
        reason = (last["revoke_reason"] or "").strip()
        suffix = f" — fact retired {when}" if when else " — fact retired"
        if reason and reason != _GENERIC_RETIRE:
            suffix += f": {reason}"
        return f"{txt} (last known{suffix})"

    cur = _fmt(live)
    if not cur:
        return None
    if as_of:
        at = next((v for v in reversed(versions) if v["created_at"] <= as_of),
                  versions[0])
        old = _fmt(at)
        if at["nid"] != live["nid"] and old and old != cur:
            return f"{cur} (was {old} when written, {_date(as_of)})"
    return cur


def resolve_fact_refs(conn: sqlite3.Connection, text: str, *, as_of: str | None = None) -> str:
    """Substitute every [[fact_*]] in *text*; unresolvable refs stay verbatim."""
    if not text or "[[fact_" not in text:
        return text

    def _sub(m: re.Match) -> str:
        return resolve_ref(conn, m.group(1), as_of=as_of) or m.group(0)

    return FACT_REF_RE.sub(_sub, text)


# ---------------------------------------------------------------------------
# CLI (Phase 4): `fact-set` edits the fact FILE (the store-of-record); the DB
# only follows on the next bridge-import — this module never writes a node
# directly, so the "read-only over the DB" guarantee in the module docstring
# above still holds. `fact-log` is a thin front end on fact_versions().
# ---------------------------------------------------------------------------

def _fact_stem(raw: str) -> str:
    """Normalize a CLI stem arg: strip a trailing .md, require the fact_ prefix.

    Cheap name-only gate before the file even opens; the type: fact check in
    cmd_fact_set is the second (content) gate — together they keep `fact-set`
    from ever rewriting a non-fact memory's frontmatter.
    """
    stem = raw[:-3] if raw.endswith(".md") else raw
    if not stem.startswith("fact_"):
        raise ValueError(f"'{raw}': not a fact (stem must start with 'fact_')")
    return stem


def cmd_fact_set(args) -> None:
    try:
        stem = _fact_stem(args.stem)
    except ValueError as exc:
        print(f"[memsom] {exc}", file=sys.stderr)
        sys.exit(1)

    memory_dir = Path(args.memory_dir) if args.memory_dir else default_memory_dir()
    path = memory_dir / f"{stem}.md"
    if not path.exists():
        print(
            f"[memsom] no fact file at {path} — create it first with a sourced "
            "value and description (fact-set never scaffolds one from nothing)",
            file=sys.stderr,
        )
        sys.exit(1)

    content = path.read_text(encoding="utf-8")
    fm_lines, _, _ = split_frontmatter(content)
    fm = fm_top_level(fm_lines)
    fm_type = (fm.get("type") or "").strip()
    if fm_type != "fact":
        print(
            f"[memsom] {path.name} has type: {fm_type or '(none)'}, not type: fact "
            "— refusing to rewrite a non-fact memory's frontmatter",
            file=sys.stderr,
        )
        sys.exit(1)

    verified = args.verified or memsom.local_date(memsom.now_iso())
    kv = {"value": args.value, "last-verified": verified}
    if args.unit is not None:
        kv["unit"] = args.unit  # omitted --unit preserves whatever unit is on file
    path.write_text(stamp_fm(content, **kv), encoding="utf-8")

    unit_note = f" {args.unit}" if args.unit is not None else ""
    print(f"[memsom] {path.name}: value={args.value}{unit_note} last-verified={verified}")
    print("(file updated; DB unchanged until the next bridge-import)")


def cmd_fact_log(args) -> None:
    try:
        stem = _fact_stem(args.stem)
    except ValueError as exc:
        print(f"[memsom] {exc}", file=sys.stderr)
        sys.exit(1)

    conn = memsom.get_connection()
    try:
        # bridge_path is added by bridge_import.migrate(); on a DB that has
        # never seen a bridge-import there are no fact nodes at all, and
        # fact_versions already degrades to [] for that case.
        bridge_migrate(conn)
        versions = fact_versions(conn, stem)
    finally:
        conn.close()

    if not versions:
        print(f"[memsom] no fact versions found for {stem}", file=sys.stderr)
        sys.exit(1)

    print(f"[memsom] {stem} — {len(versions)} version(s), oldest first:")
    for v in versions:
        txt = _fmt(v) or "(no value)"
        frm = _date(v["created_at"]) or "?"
        until = (_date(v["tombstoned_at"]) or "now") if v["tombstoned"] else "now"
        line = f"  {txt}  from {frm}  until {until}"
        reason = (v.get("revoke_reason") or "").strip()
        if v["tombstoned"] and reason and reason != _GENERIC_RETIRE:
            line += f"  ({reason})"
        print(line)


def register(subparsers) -> None:
    """Mount fact-set / fact-log (flat, hyphenated — the convention every
    other multi-action module in this CLI uses, e.g. corroborate.py's
    register-root/assert-claim/claims-list)."""
    p_set = subparsers.add_parser(
        "fact-set", help="edit a fact file's value (the file is the store-of-record)"
    )
    p_set.add_argument("stem", help="fact_<name> stem, with or without .md")
    p_set.add_argument("value")
    p_set.add_argument("--unit", default=None)
    p_set.add_argument("--verified", default=None, help="YYYY-MM-DD (default: today)")
    p_set.add_argument("--memory-dir", default=None,
                        help="override the memory dir (default: live PC store / "
                             "$MEMDAG_BRIDGE_MEMORY_DIR)")
    p_set.set_defaults(func=cmd_fact_set)

    p_log = subparsers.add_parser(
        "fact-log", help="print a fact's value history from the supersede chain"
    )
    p_log.add_argument("stem", help="fact_<name> stem, with or without .md")
    p_log.add_argument("--memory-dir", default=None,
                        help="unused — history comes from the DB, not the file; "
                             "accepted for CLI symmetry with fact-set")
    p_log.set_defaults(func=cmd_fact_log)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_facts")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
