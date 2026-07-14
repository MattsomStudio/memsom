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

import re
import sqlite3

from memsom.bridge.bridge_import import split_frontmatter, fm_top_level
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
