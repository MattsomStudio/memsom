"""memsom_forget — the RS/SS "forgetting" layer, ported into memsom (Phase 2).

This is a faithful port of the user's flat-file forgetting layer
(~/.claude/episodic/mem_weights.py) onto memsom nodes.  It is the ranking signal
the digest exporter (Phase 3) needs: which memories are "hot" enough to render
into the always-on MEMORY.md vs which have gone cold (kept in the store, dropped
from the digest).  Nothing is ever deleted by disuse.

Two-number model (Bjork's New Theory of Disuse):
  RS = retrieval strength — accessibility; decays with disuse (slowed by SS);
       drives hot/cold.
  SS = storage strength — durability; does not decay; grows with SPACED
       retrieval; seeded at birth from a memory's salience.

The pure computation (`compute`, `_seed_rs_ss`, `_decay_rs`, `_salience_of`,
DEFAULTS, `_parse_iso`, `now_iso`, `parse_frontmatter`) is copied VERBATIM from
mem_weights.py so its behaviour is identical — proven by the golden-parity test
(test_memsom_forget.TestParity), which feeds the same inputs to both and asserts
equality.  Only the storage adapter differs: state lives in node columns
(forget_rs/forget_ss/forget_tier/...) instead of canonical.json, and the memory
inventory is read from bridge-imported nodes (source_ref LIKE 'memory:%') instead
of the filesystem.

Frozen core (memsom.py) is untouched; all new columns are additive + nullable.
Library discipline: compute/recompute never print or sys.exit.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import memsom
import memsom_schema


# ====================================================================== #
# VERBATIM PORT from mem_weights.py — do not edit without re-running      #
# test_memsom_forget.TestParity (golden parity against the original).     #
# ====================================================================== #

DEFAULTS = {
    "rs_cap": 1.0,
    "rs_seed": 1.0,
    "decay_base": 0.5,
    "rs_gain": 0.15,
    "demote_below": 0.2,
    "grace_days": 21,
    "promote_at": 0.5,
    "ss_floor": 0.0,
    "ss_cap": 3.0,
    "ss_gain": 0.1,
    "ss_decay_k": 1.0,
    "ss_mig_k": 0.5,
    "salience_default": 0.0,
}

FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
PIN_TYPES = {"user", "feedback"}
PIN_PREFIXES = ("user_", "feedback_")


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_frontmatter(text):
    """Return {key: value} from a markdown file's YAML-ish frontmatter (flat)."""
    m = FM_RE.match(text)
    out = {}
    if not m:
        return out
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _salience_of(m, p):
    raw = m.get("salience")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return float(p.get("salience_default", 0.0))


def _seed_rs_ss(st, m, p):
    cap = p["rs_cap"] or 1.0
    if "rs" in st:
        rs, ss = float(st["rs"]), float(st.get("ss", p["ss_floor"]))
    elif "weight" in st:
        rs = float(st["weight"])
        cnt = max(0, int(st.get("count", 0)))
        ss = p["ss_floor"] + p["ss_mig_k"] * math.log1p(cnt)
    else:
        rs = float(p["rs_seed"])
        ss = p["ss_floor"] + (p["ss_cap"] - p["ss_floor"]) * _salience_of(m, p)
    return min(rs, cap), max(0.0, min(ss, p["ss_cap"]))


def _decay_rs(rs, ss, t0, t1, p):
    if not (t0 and t1) or t1 <= t0:
        return rs
    dt_weeks = (t1 - t0).total_seconds() / (7 * 86400.0)
    return rs * (p["decay_base"] ** (dt_weeks / (1.0 + p["ss_decay_k"] * ss)))


def compute(canon, mems, events, now=None, stale=None):
    """Apply one decay+reinforce pass over the (RS, SS) model and decide
    promote/demote actions.  Pure — touches no disk.  See mem_weights.compute
    for the full contract; this is a verbatim copy."""
    now = now or datetime.now(timezone.utc)
    stale = stale or set()
    p = canon["params"]
    rs_cap = p["rs_cap"] or 1.0
    prev = canon["memories"]
    new = {}
    actions = []
    by_stem = {m["stem"]: m for m in mems}

    for stem, m in by_stem.items():
        st = dict(prev.get(stem, {}))
        rs, ss = _seed_rs_ss(st, m, p)
        count = int(st.get("count", 0))
        first_seen = st.get("first_seen") or now_iso()
        last_used = st.get("last_used")
        last_t = _parse_iso(last_used) or _parse_iso(first_seen) or now

        stem_events = [] if stem in stale else events.get(stem, [])
        for ts_iso, hits in sorted(stem_events, key=lambda e: e[0] or ""):
            t = _parse_iso(ts_iso) or now
            if t > now:
                t = now
            rs = _decay_rs(rs, ss, last_t, t, p)
            for _ in range(int(hits)):
                ss = max(0.0, min(p["ss_cap"], ss + p["ss_gain"] * (1.0 - rs / rs_cap)))
                rs = min(rs_cap, rs + p["rs_gain"] * (1.0 + math.log1p(ss)))
            count += int(hits)
            last_t = t
            last_used = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        rs = _decay_rs(rs, ss, last_t, now, p)
        rs = round(rs, 4)
        ss = round(ss, 4)

        current = m["tier"]
        had_stash = bool(st.get("index_line"))
        age_days = (now - (_parse_iso(first_seen) or now)).days
        if m["pinned"]:
            target = "hot"
        elif current == "hot" and rs < p["demote_below"] and age_days >= p["grace_days"]:
            target = "cold"
        elif current == "cold" and had_stash and rs >= p["promote_at"]:
            target = "hot"
        else:
            target = current

        if target != current:
            actions.append({
                "stem": stem, "file": m["file"], "from": current, "to": target,
                "rs": rs, "ss": ss, "pinned": m["pinned"],
                "reason": ("RS decayed below %.2f (age %dd)" % (p["demote_below"], age_days)
                           if target == "cold" else
                           "RS recalled back above %.2f" % p["promote_at"]),
            })

        new[stem] = {
            "rs": rs, "ss": ss, "count": count, "first_seen": first_seen,
            "last_used": last_used, "tier": target, "pinned": m["pinned"],
        }
        if target == "cold" and st.get("index_line"):
            new[stem]["index_line"] = st["index_line"]
            new[stem]["index_section"] = st.get("index_section")
    return new, actions


def read_usage_events(usage_dir, after_ts=None):
    """{stem: [(ts_iso, hits), …]} from append-only usage JSONLs in *usage_dir*.

    Verbatim logic from mem_weights.read_all_events, but the directory is a
    parameter (memsom is shippable — no hard-coded ~/.claude path).  Returns {}
    if the dir is absent, so a fresh/CI store just decays with no reinforcement.
    """
    out = {}
    usage_dir = Path(usage_dir) if usage_dir else None
    if not usage_dir or not usage_dir.is_dir():
        return out
    cutoff = _parse_iso(after_ts) if after_ts else None
    for f in sorted(usage_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("ts")
            if cutoff:
                t = _parse_iso(ts)
                if t and t <= cutoff:
                    continue
            n = int(rec.get("hits", 0))
            if n:
                out.setdefault(rec["stem"], []).append((ts, n))
    for stem in out:
        out[stem].sort(key=lambda e: e[0] or "")
    return out


# ====================================================================== #
# memsom storage adapter — state in node columns, inventory from nodes    #
# ====================================================================== #

# new columns are forget_-prefixed to guarantee no collision with any other
# module's column.  All additive + nullable (frozen core never reads them).
_COLS = (
    ("forget_rs", "REAL"),
    ("forget_ss", "REAL"),
    ("forget_count", "INTEGER DEFAULT 0"),
    ("forget_first_seen", "TEXT"),
    ("forget_last_used", "TEXT"),
    ("forget_tier", "TEXT DEFAULT 'hot'"),
)


def migrate(conn) -> None:
    """Idempotent: add the forget_* columns + a one-row meta table for the
    last-reconcile watermark."""
    for col, decl in _COLS:
        memsom_schema.add_column(conn, "nodes", col, decl)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS forget_meta "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), updated TEXT)"
    )


def _is_pinned(channel, fm):
    """Pinning maps to channel: endorsed (user_/feedback_/personal_) never
    demotes; an explicit frontmatter pin also pins."""
    if channel == "endorsed":
        return True
    if str(fm.get("pin", "")).lower() in ("true", "1", "yes"):
        return True
    return False


def _bridge_rows(conn):
    """Live bridge-imported memory nodes (source_ref LIKE 'memory:%').

    Scoped by source_ref, NOT obsidian_path, so Obsidian vault notes (which also
    carry obsidian_path) are never swept by the forgetting layer.  Deliberately
    does NOT select obsidian_path — forget owns only its forget_* columns and the
    base nodes table, so forget.migrate is self-sufficient (no dependency on the
    obsidian module's schema).
    """
    return conn.execute(
        "SELECT id, content, channel, source_ref, "
        "       forget_rs, forget_ss, forget_count, forget_first_seen, "
        "       forget_last_used, forget_tier "
        "FROM nodes WHERE tombstoned = 0 AND channel != 'agent-derived' "
        "AND source_ref LIKE 'memory:%' "
        "AND source_ref NOT LIKE 'memory:literal:%'"  # literals are fixed, not ranked
    ).fetchall()


def _stem_of(source_ref):
    if source_ref and source_ref.startswith("memory:"):
        return source_ref.split(":", 1)[1]
    return source_ref


def build_inventory(conn):
    """Return (mems, canon, node_by_stem) from the live bridge nodes.

    mems / canon are shaped exactly as compute() expects, so the ported pure
    function runs unchanged.  `index_line` is synthesized for cold nodes (in
    memsom every cold node was demoted BY this layer — there is no hand-curated
    exclusion as in the flat MEMORY.md — so a cold node is always re-promotable
    when its RS rebounds; setting index_line makes compute's had_stash gate fire).
    """
    mems, memories, node_by_stem = [], {}, {}
    for (nid, content, channel, sref, rs, ss, cnt, fseen, lused, tier) in _bridge_rows(conn):
        fm = parse_frontmatter(content or "")
        stem = _stem_of(sref)
        node_by_stem[stem] = nid
        tier = tier or "hot"
        mems.append({
            "stem": stem,
            "file": f"{stem}.md",
            "name": fm.get("name", stem),
            "type": fm.get("type", ""),
            "pinned": _is_pinned(channel, fm),
            "tier": tier,
            "salience": fm.get("salience"),
        })
        st = {"tier": tier}
        if rs is not None:
            st["rs"] = rs
            st["ss"] = ss if ss is not None else DEFAULTS["ss_floor"]
        if cnt is not None:
            st["count"] = cnt
        if fseen:
            st["first_seen"] = fseen
        if lused:
            st["last_used"] = lused
        if tier == "cold":
            st["index_line"] = stem  # synthetic had_stash (see docstring)
        memories[stem] = st
    canon = {"version": 1, "updated": _get_updated(conn),
             "params": dict(DEFAULTS), "memories": memories}
    return mems, canon, node_by_stem


def _get_updated(conn):
    row = conn.execute("SELECT updated FROM forget_meta WHERE id = 1").fetchone()
    return row[0] if row else None


def _set_updated(conn, ts):
    conn.execute(
        "INSERT INTO forget_meta(id, updated) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET updated = excluded.updated",
        (ts,),
    )


def check_pins(conn):
    """Invariant: a pinned (endorsed) bridge memory must never be cold — losing
    one from the digest would forget who the user is.  compute() enforces this, so
    a violation means external tampering or a bug.  Returns a list of violations."""
    if not memsom_schema.column_exists(conn, "nodes", "forget_tier"):
        return []
    rows = conn.execute(
        "SELECT id, source_ref FROM nodes WHERE channel = 'endorsed' "
        "AND forget_tier = 'cold' AND tombstoned = 0 "
        "AND source_ref LIKE 'memory:%'"
    ).fetchall()
    return [{"kind": "pin-violation", "node": nid,
             "detail": f"pinned (endorsed) memory {sref} is cold"}
            for nid, sref in rows]


def fix_pins(conn):
    """Force every pinned (endorsed) bridge memory back to hot.  Returns count."""
    if not memsom_schema.column_exists(conn, "nodes", "forget_tier"):
        return 0
    with conn:
        cur = conn.execute(
            "UPDATE nodes SET forget_tier = 'hot' WHERE channel = 'endorsed' "
            "AND forget_tier = 'cold' AND tombstoned = 0 "
            "AND source_ref LIKE 'memory:%'")
    return cur.rowcount


def recompute_forget(conn, *, usage_dir=None, events=None, now=None,
                     stale=None, dry_run=False):
    """One decay+reinforce pass over all bridge memories; writes RS/SS/tier back
    to node columns.  Returns the list of promote/demote actions.

    `events` (pre-built {stem: [(ts, hits)]}) wins; else read from `usage_dir`;
    else no reinforcement (pure decay).  Single-writer discipline: run this
    PC-only (mirrors mem_reconcile's Sunday cadence) so a synced DB never has two
    authorities racing the same rows.
    """
    mems, canon, node_by_stem = build_inventory(conn)
    if events is None:
        events = read_usage_events(usage_dir, after_ts=canon["updated"])
    new, actions = compute(canon, mems, events, now=now, stale=stale)
    if not dry_run:
        with conn:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            for stem, st in new.items():
                nid = node_by_stem.get(stem)
                if nid is None:
                    continue
                conn.execute(
                    "UPDATE nodes SET forget_rs = ?, forget_ss = ?, forget_count = ?, "
                    "forget_first_seen = ?, forget_last_used = ?, forget_tier = ? "
                    "WHERE id = ?",
                    (st["rs"], st["ss"], st["count"], st["first_seen"],
                     st["last_used"], st["tier"], nid),
                )
            _set_updated(conn, now_iso())
    return actions
