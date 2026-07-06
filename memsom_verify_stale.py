"""memsom_verify_stale — verification-age staleness for bridge memory notes.

A NEW TRIGGER feeding the memsom_stale engine (which owns mark/clear/supersede).
Where memsom_stale's cascade fires on "the source CONTENT changed" (content_hash
on re-ingest), this pass fires on "reality moved and the note didn't": a
state-bearing memory whose last-verified age has gone stale, or whose DUE date has
passed.  The bridge re-import tombstone-replaces a CHANGED file, so the cascade
never fires for memory:% nodes — this trigger is the only thing that ever sets
stale=1 on a bridge memory.

Design:
  - STATE-BEARING gate first.  Stable facts ("the user prefers tabs") never go stale no
    matter how old; only notes that assert a transient state ("NOT deployed", "IN
    PROGRESS", "DUE <date>") are candidates.  The regex is the main false-positive
    tunable; it is held to the verified phrase list (precision over recall).
  - AS-OF time = `last-verified:` frontmatter if present, else the file's
    bridge_mtime (last-touched).  A note untouched past the threshold while still
    asserting a transient state is, by definition, unverified.
  - REASON-NAMESPACE OWNERSHIP (the critical safety boundary): this pass only ever
    CLEARS staleness whose reason it set ("unverified since …" / "overdue …").  A
    supersession-cascade flag from memsom_stale is left untouched.
  - Reversibility is structural: editing a note mints a NEW node (stale=0) and
    tombstones the old; the scan filters tombstoned=0, so the flag clears on its own.

Threshold via $MEMDAG_VERIFY_STALE_DAYS (default 21, matching the forget grace
window); <= 0 DISABLES the pass and clears every verify-owned flag (kill switch).

Library discipline: the pure helpers + recompute_verify_stale never print or
sys.exit; only main()/_cmd_* do I/O.  Frozen core untouched.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone

import memsom
import memsom_schema
import memsom_stale
from memsom_bridge_import import split_frontmatter, fm_top_level

# --- config -------------------------------------------------------------------

DEFAULT_THRESHOLD_DAYS = 21          # matches forget DEFAULTS["grace_days"]
ENV_DAYS = "MEMDAG_VERIFY_STALE_DAYS"
OWNED_PREFIXES = ("unverified since", "overdue")   # reasons THIS pass owns

# State-risk phrases — PRECISION FIRST.  Only STRONG completion-negations: an
# explicit "this work is not finished" about the note's own subject.  Deliberately
# excludes prose-common words ("to do", "in progress", "not yet", "kill date") that
# match incidental mentions or status-column NAMES rather than this note's state —
# those produced the false positives in the first dry-run.  The {0,1} gap admits
# "NOT fully migrated" / "NOT yet deployed" without spanning clause boundaries.
# This is the main tunable; widen deliberately, never casually.
STATE_RISK_RE = re.compile(
    r"\bnot\s+(?:\w+\s+){0,1}(?:"
    r"deployed|applied|migrated|pushed|merged|installed|built|rebuilt|configured"
    r"|fixed|wired|implemented|shipped|done|complete|completed|finished|live"
    r")\b"
    r"|\bun(?:deployed|pushed|merged|migrated|applied|finished|shipped)\b",
    re.IGNORECASE)

# A concrete deadline, used ONLY for the overdue (past-date) trigger.
DUE_RE = re.compile(
    r"\b(?:due|deadline)\b[:\s]+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


# --- pure helpers -------------------------------------------------------------

def _threshold_days() -> int:
    raw = (os.environ.get(ENV_DAYS) or "").strip()
    if not raw:
        return DEFAULT_THRESHOLD_DAYS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_THRESHOLD_DAYS


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(s):
    """'YYYY-MM-DD' or a full ISO timestamp -> utc-aware datetime, else None."""
    if not s:
        return None
    s = str(s).strip()
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _as_of(content, bridge_mtime):
    """When this note was last KNOWN-good: last-verified frontmatter wins, else the
    file mtime (bridge_mtime is "<ns>:<size>").  None if neither is parseable."""
    fm = fm_top_level(split_frontmatter(content or "")[0])
    lv = _parse_date(fm.get("last-verified") or fm.get("last_verified"))
    if lv is not None:
        return lv
    if not bridge_mtime:
        return None
    head = str(bridge_mtime).split(":", 1)[0]
    try:
        ns = int(head)
    except ValueError:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def is_state_bearing(content) -> bool:
    return STATE_RISK_RE.search(content or "") is not None


def past_due_date(content, now):
    """The earliest DUE/kill-date that is now in the PAST (string), else None."""
    passed = []
    for m in DUE_RE.finditer(content or ""):
        d = _parse_date(m.group(1))
        if d is not None and d <= now:
            passed.append((d, m.group(1)))
    if not passed:
        return None
    passed.sort(key=lambda t: t[0])
    return passed[0][1]


def _owned(reason) -> bool:
    return bool(reason) and str(reason).startswith(OWNED_PREFIXES)


def assess(content, bridge_mtime, now, threshold_days):
    """(stale, reason) for ONE memory node.  Deterministic per (node, clock):
    a node only ever transitions fresh->stale as the clock advances (age crossing
    or a DUE date passing); it never self-heals (that needs an edit -> new node)."""
    if threshold_days <= 0:                       # disabled
        return (False, None)
    due = past_due_date(content, now)             # overdue: stale on its own
    if due:
        return (True, f"overdue: {due} passed")
    if not is_state_bearing(content):             # stable fact -> never stale
        return (False, None)
    asof = _as_of(content, bridge_mtime)
    if asof is None:                              # can't prove age -> don't flag
        return (False, None)
    if (now - asof).days > threshold_days:
        return (True, f"unverified since {asof:%Y-%m}")
    return (False, None)


# --- the reconciler -----------------------------------------------------------

def recompute_verify_stale(conn, *, now=None, threshold_days=None, dry_run=False):
    """Reconcile verification-age staleness over live memory:% FILE nodes.

    Idempotent (re-running with no clock change marks nothing) and reversible
    (an edited note's old stale row tombstones out; the fresh node re-assesses).
    Returns {"marked": [ids], "cleared": [ids], "scanned": int,
             "stale_stems": set[str]}.  stale_stems are BARE stems (e.g.
    "project_x", not "memory:project_x") for an optional forget hand-off.
    """
    memsom_stale.migrate(conn)                    # defensive (also done in bi.migrate)
    memsom_schema.add_column(conn, "nodes", "bridge_mtime", "TEXT")  # defensive ditto
    now = now or _now()
    thr = threshold_days if threshold_days is not None else _threshold_days()
    marked, cleared, stale_stems = [], [], set()
    # Only user-channel (project_/reference_) notes are verification-staleness
    # candidates: endorsed (user_/personal_/feedback_) notes are pinned-durable
    # rules/facts that don't go reality-stale.  endorsed nodes still fall through
    # to the clear-if-owned branch (want stays False), so a kill-switch run clears
    # any verify flag left on one — but they are never freshly marked.
    rows = conn.execute(
        "SELECT id, content, source_ref, bridge_mtime, stale, stale_reason, channel "
        "FROM nodes WHERE tombstoned = 0 AND source_ref LIKE 'memory:%' "
        "AND source_ref NOT LIKE 'memory:literal:%'"
    ).fetchall()
    for nid, content, sref, mtime, cur_stale, cur_reason, channel in rows:
        stem = sref.split(":", 1)[1] if sref.startswith("memory:") else sref
        if channel == "user":
            want, reason = assess(content, mtime, now, thr)
        else:
            want, reason = False, None            # endorsed: never a candidate
        if want:
            stale_stems.add(stem)
            if not cur_stale:
                if not dry_run:
                    memsom_stale.mark_stale_cascade(conn, nid, reason)   # audited
                marked.append(nid)
            # already stale (ours OR a supersession): keep the original record
        elif cur_stale and _owned(cur_reason):
            if not dry_run:
                memsom_stale.unstale(conn, nid)                          # audited
            cleared.append(nid)
    return {"marked": marked, "cleared": cleared,
            "scanned": len(rows), "stale_stems": stale_stems}


# --- CLI ----------------------------------------------------------------------

def _cmd_verify_stale(args):
    conn = memsom.get_connection()
    try:
        res = recompute_verify_stale(conn, dry_run=not args.apply)
        mode = "APPLIED" if args.apply else "DRY-RUN (no writes)"
        print(f"[verify-stale] {mode}  threshold={_threshold_days()}d")
        print(f"  scanned : {res['scanned']} memory file nodes")
        print(f"  marked  : {len(res['marked'])} -> stale")
        print(f"  cleared : {len(res['cleared'])} (re-verified / no longer stale)")
        if res["stale_stems"]:
            print(f"  stale   : {', '.join(sorted(res['stale_stems']))}")
    finally:
        conn.close()


def register(sub) -> None:
    p = sub.add_parser("verify-stale",
                       help="flag state-bearing memory notes whose verification "
                            "age has gone stale")
    p.add_argument("--apply", action="store_true",
                   help="apply the marks/clears (default: dry-run)")
    p.set_defaults(func=_cmd_verify_stale)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass  # reconfigure unsupported on this stream — keep its default encoding
    ap = argparse.ArgumentParser(prog="memsom_verify_stale", description=__doc__)
    ap.add_argument("--apply", action="store_true", help="apply (default: dry-run)")
    _cmd_verify_stale(ap.parse_args(argv))


if __name__ == "__main__":
    main()
