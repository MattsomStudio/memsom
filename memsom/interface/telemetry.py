"""memsom.interface.telemetry -- the panel's contract surface (PROMOTE-Q11-PANEL.md Part B1).

memsom_panel (memsom-agentic-os) reads exactly three things from this module:

    build_telemetry(memory_dir=None, *, conn=None) -> dict
    load_weights(conn=None) -> list[dict]
    default_memory_dir                              (re-export of kernel.paths)

Ported from the pre-refactor memsom/interface/dashboard.py (798 LOC, deleted by
A-9: it held 3 of memsom's 5 stray direct SQLite connections and 2 subprocess
sites -- the HTML-dashboard renderer's `open_file`). This module keeps every
payload SHAPE the panel depends on byte-for-byte identical and drops only what
the panel never called: `render`/`open_file`/`main`/`_cmd_dashboard`/`register`
(the `memsom dashboard` HTML command stays deleted; A-9 stands).

Rule 4 (one connection owner): every read goes through
`memsom.storage.db.get_connection` -- including the two sub-readers that are
NOT the memsom store itself (the episodic sessions archive, and the code-RAG
index). The code-RAG rows turned out to already live in the memsom store's own
`code_chunks` table (memsom.retrieval.code_index), so that sub-reader just
reuses the caller's connection -- no foreign path involved. Only the episodic
sessions archive (~/.claude/episodic/sessions.db, a separate SQLite file) is a
genuinely foreign path; `get_connection(path=<foreign>, read_only=True)` does
not schema-check its target, so it opens that file too. If a real read against
it fails (corrupt file, locked, unexpected schema), the `sessions` key is
still returned -- as `{"count": None, "reason": "..."}"` instead of a plain
`None` -- so the 'telemetry' feature probe can tell "no episodic archive on
this machine" (normal, matches live behaviour) from "the archive is there and
broken" (degraded).

Rule 5 (one taint primitive): `load_weights`'s pool filters `nodes` by
`tombstoned` (the live query's only taint predicate), so its WHERE is built
through `memsom.storage.schema.taint_filter_clauses` rather than the live
code's hand-rolled `tombstoned = 0`. Called with `clearance=3` (the same
"administrative, sees the whole store" clearance `integrity.ingest` and
`lifecycle.compact` use for their own whole-store maintenance passes) --
telemetry is an operator view over everything the forgetting layer tracks, not
a clearance-gated retrieval, and clearance=3 is the only choice that does not
narrow the live dashboard's (unfiltered) result set by default.

Rule 7 (no bare env reads): memory dir -> `kernel.paths.default_memory_dir()`;
budget + forget thresholds -> `lifecycle.forget.load_params(...)`; the one
remaining env read the live code had (`MEMSOM_CONSOLIDATION_DIR`, for
`last_consolidation`'s report directory) is now the `telemetry.consolidation_dir`
knob in `memsom/tuning.py`.

Rule 10 (effects boxed / no subprocess): the live `open_file()` (2 subprocess
sites, one per platform branch) is dropped along with `render`/`main` -- there
is no HTML dashboard here to open. `last_consolidation()` was already a pure
filesystem read (glob + mtime), not a subprocess call, so it is kept as-is
modulo the env -> knob swap above.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from memsom.bridge.bridge_import import iter_memory_files
from memsom.kernel.frontmatter import fm_top_level, split_frontmatter
from memsom.kernel.paths import default_memory_dir
from memsom.lifecycle import forget
from memsom.storage import db as memsom_db
from memsom.storage import schema as memsom_schema
from memsom import tuning as memsom_tuning

__all__ = ["build_telemetry", "load_weights", "default_memory_dir"]

_TYPE_PREFIXES = ("user", "feedback", "project", "personal", "reference")

# Trust-channel enum surfaced to the panel. The store's `channel` column is
# authoritative (endorsed > user > agent-derived > external, CHECK-constrained
# at insert); 'agent-derived' folds to the short 'agent' the panel enum uses.
# Fallback (channel absent -- e.g. a synthetic test row) derives from the
# memory TYPE, matching bridge_import.CHANNEL_BY_TYPE: user/feedback/personal
# = endorsed; project/reference/fact = user.
_CHANNEL_ENUM = {
    "endorsed": "endorsed",
    "user": "user",
    "agent-derived": "agent",
    "agent": "agent",
    "external": "external",
}
_TYPE_CHANNEL = {
    "user": "endorsed", "feedback": "endorsed", "personal": "endorsed",
    "project": "user", "reference": "user", "fact": "user",
}


def _parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stem_type(stem: str) -> str:
    head = stem.split("_", 1)[0]
    return head if head in _TYPE_PREFIXES else "other"


def _memory_channel(raw_channel, stem: str) -> str:
    """Authoritative trust channel for a memory node, as the panel enum
    ("endorsed"|"user"|"agent"|"external"). Prefers the store's channel value;
    derives from the type prefix only when the store value is missing."""
    if raw_channel:
        mapped = _CHANNEL_ENUM.get(str(raw_channel).strip().lower())
        if mapped:
            return mapped
    return _TYPE_CHANNEL.get(stem.split("_", 1)[0], "user")


def load_weights(conn=None) -> list[dict]:
    """Live forgetting telemetry from memsom's forget_* columns.

    Row shape (unchanged from the pre-refactor dashboard):
    stem/weight/count/last_used/first_seen/tier/channel/pinned. `weight` =
    forget_rs (RS/accessibility); `pinned` = endorsed channel
    (user_/feedback_/personal_) or an explicit frontmatter pin.

    *conn* is read-only; when omitted this opens (and closes) its own
    connection to the default store via `storage.db.get_connection`.
    """
    own_conn = conn is None
    if own_conn:
        conn = memsom_db.get_connection(read_only=True)
    try:
        # The forget_* columns are created by forget.migrate, which only runs
        # from bridge-render -- a fresh store (init/migrate_all only) doesn't
        # have them, and this read-only connection can't add them.
        if not memsom_schema.column_exists(conn, "nodes", "forget_rs"):
            raise RuntimeError(
                "no forgetting telemetry yet: run `memsom bridge-render` once "
                "to populate the forget_* columns")
        clauses, params = memsom_schema.taint_filter_clauses(conn, clearance=3)
        where = " AND ".join(clauses) + (
            " AND source_ref LIKE 'memory:%' "
            "AND source_ref NOT LIKE 'memory:literal:%'")
        rows = conn.execute(
            "SELECT source_ref, content, channel, forget_rs, forget_count, "
            "forget_last_used, forget_first_seen, forget_tier FROM nodes "
            f"WHERE {where}", params).fetchall()
    finally:
        if own_conn:
            conn.close()
    out = []
    for sref, content, channel, rs, cnt, lused, fseen, tier in rows:
        out.append({
            "stem": sref.split(":", 1)[1],
            "weight": float(rs) if rs is not None else 1.0,
            "count": int(cnt or 0),
            "last_used": lused,
            "first_seen": fseen,
            "tier": tier or "hot",
            "channel": channel,
            "pinned": 1 if (channel == "endorsed" or str(
                fm_top_level(split_frontmatter(content or "")[0]).get("pin", "")
            ).strip().lower() in ("1", "true", "yes")) else 0,
        })
    return out


def _build_graph(mem_dir, rows):
    """Relationship graph: MEMORY.md sections are parent hubs, memories are
    their siblings (tree edges), and [[wikilinks]] in memory bodies are
    cross-links. Pure filesystem read; no DB access."""
    if not mem_dir:
        return {"nodes": [], "links": [], "sections": []}
    mf = Path(mem_dir) / "MEMORY.md"
    if not mf.exists():
        return {"nodes": [], "links": [], "sections": []}

    link_re = re.compile(r"\[[^\]]*\]\(([a-z0-9_]+)\.md\)")
    section_of = {}
    order = []
    cur = None
    for line in mf.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            if cur not in order:
                order.append(cur)
        elif cur:
            m = link_re.search(line)
            if m:
                section_of.setdefault(m.group(1), cur)

    DEMO = "(demoted)"
    nodes, links = [], []
    have_demo = False
    for s in order:
        nodes.append({"id": "§" + s, "label": s, "kind": "section", "section": s})

    for r in rows:
        stem = r["stem"]
        sec = section_of.get(stem, DEMO)
        if sec == DEMO and not have_demo:
            have_demo = True
            order.append(DEMO)
            nodes.append({"id": "§" + DEMO, "label": DEMO,
                          "kind": "section", "section": DEMO})
        nodes.append({"id": stem, "label": stem, "kind": "memory",
                      "section": sec, "type": _stem_type(stem),
                      "count": int(r["count"]), "tier": r["tier"],
                      "pinned": int(r["pinned"]),
                      "channel": _memory_channel(r.get("channel"), stem)})
        links.append({"source": "§" + sec, "target": stem, "kind": "tree"})

    nodeset = {n["id"] for n in nodes}
    wl_re = re.compile(r"\[\[([a-z0-9_-]+)\]\]")
    seen = set()
    for p in iter_memory_files(mem_dir):
        if p.stem not in nodeset:
            continue
        body = p.read_text(encoding="utf-8", errors="ignore")
        for target in wl_re.findall(body):
            if target in nodeset and target != p.stem:
                key = tuple(sorted((p.stem, target)))
                if key not in seen:
                    seen.add(key)
                    links.append({"source": p.stem, "target": target, "kind": "link"})

    return {"nodes": nodes, "links": links, "sections": order}


def _build_code_graph(conn, mem_dir, memory_ids):
    """Parallel graph for the code-RAG index, merged into the same view as the
    memory graph. Repo hubs, their files (sized by chunk count), and a
    cross-edge from a memory to any indexed code file whose basename it names
    verbatim in its body ('this memory documents this code').

    `code_chunks` lives in the SAME store as everything else (memsom.retrieval
    .code_index), so this reuses *conn* -- it is not a foreign-path read.
    Empty when the code index is absent or unused, so a store without
    code-RAG renders exactly the memory graph it did before."""
    empty = {"nodes": [], "links": [], "repos": []}
    try:
        has = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='code_chunks'"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT repo, path, COUNT(*) FROM code_chunks GROUP BY repo, path"
        ).fetchall() if has else []
    except sqlite3.Error:
        # FAILOPEN: allowed, an unexpectedly-shaped code_chunks table must not
        # crash the memory-only telemetry payload -- the code graph is an
        # optional overlay, never load-bearing.
        return empty
    if not rows:
        return empty

    nodes, links, repos = [], [], []
    seen_repo = set()
    base_to_file = {}
    for repo, path, cnt in rows:
        rid = "repo:" + repo
        if repo not in seen_repo:
            seen_repo.add(repo)
            repos.append(repo)
            nodes.append({"id": rid, "label": repo, "kind": "repo"})
        fid = "file:" + repo + "::" + path
        nodes.append({"id": fid, "label": path, "kind": "file",
                      "repo": repo, "count": int(cnt)})
        links.append({"source": rid, "target": fid, "kind": "codetree"})
        base = path.rsplit("/", 1)[-1]
        if not base.startswith("__"):
            base_to_file.setdefault(base, fid)

    if mem_dir and base_to_file:
        bases = sorted(base_to_file, key=len, reverse=True)
        pat = re.compile(r"(?<![\w./])(" +
                         "|".join(re.escape(b) for b in bases) + r")(?![\w])")
        for p in iter_memory_files(mem_dir):
            if p.stem not in memory_ids:
                continue
            body = p.read_text(encoding="utf-8", errors="ignore")
            for fid in {base_to_file[m.group(1)] for m in pat.finditer(body)}:
                links.append({"source": p.stem, "target": fid, "kind": "codelink"})
    return {"nodes": nodes, "links": links, "repos": repos}


def _consolidation_dir() -> Path:
    """Where the weekly consolidation sweep writes its dated reports.
    Override via the `telemetry.consolidation_dir` knob (tuning.py); same
    idiom the live code used with a raw env read."""
    override = memsom_tuning.resolve("telemetry.consolidation_dir")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "consolidation"


def _last_consolidation():
    """ISO8601 (UTC) of when the memory consolidation sweep last completed on
    this machine, or None if it has never run here. Pure filesystem read
    (glob + mtime) -- never a subprocess call."""
    base = _consolidation_dir()
    candidates = []
    reports = base / "reports"
    if reports.is_dir():
        candidates.extend(reports.glob("report-*.md"))
    latest = base / "latest-report.md"
    if latest.exists():
        candidates.append(latest)
    if not candidates:
        return None
    try:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return datetime.fromtimestamp(newest.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def _session_count():
    """Optional episodic session archive (~/.claude/episodic/sessions.db, a
    separate project's SQLite file -- absent on a fresh machine). Routed
    through `storage.db.get_connection` (Rule 4) even though it is not the
    memsom store: a read-only open does no schema check, so any existing
    SQLite file opens. A missing file is normal (matches the live dashboard's
    behaviour: the session-count card is simply omitted, `None`); a real read
    failure against a file that IS there comes back as `{"count": None,
    "reason": ...}` so the 'telemetry' feature probe can report 'degraded'
    instead of silently losing the signal."""
    override = memsom_tuning.resolve("telemetry.episodic_db")
    sdb = Path(override) if override else Path.home() / ".claude" / "episodic" / "sessions.db"
    try:
        conn = memsom_db.get_connection(path=sdb, read_only=True)
    except FileNotFoundError:
        return None
    except sqlite3.Error as exc:
        return {"count": None, "reason": f"episodic db unreadable: {exc!r}"}
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for name in ("sessions", "session", "transcripts", "chunks"):
            if name in tables:
                n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                return {"table": name, "count": n}
        return None
    except sqlite3.Error as exc:
        return {"count": None, "reason": f"episodic db query failed: {exc!r}"}
    finally:
        conn.close()


def build_telemetry(memory_dir=None, *, conn=None) -> dict:
    """The panel's `/api/memory` payload. Same 14 keys as the pre-refactor
    dashboard.build_telemetry(): generated, last_consolidation, totals, tier,
    types, hist, top_access, scatter, growth, stale, budget, sessions,
    thresholds, graph.

    Both params are optional -- `build_telemetry()` (the live zero-arg call)
    still works, resolving the store via `storage.db.get_connection` and the
    memory dir via `kernel.paths.default_memory_dir()`.
    """
    own_conn = conn is None
    if own_conn:
        conn = memsom_db.get_connection(read_only=True)
    try:
        rows = load_weights(conn=conn)
        now = datetime.now(timezone.utc)

        total = len(rows)
        hot = sum(1 for r in rows if r["tier"] == "hot")
        cold = total - hot
        pinned = sum(1 for r in rows if r["pinned"])

        type_counts = Counter(_stem_type(r["stem"]) for r in rows)

        buckets = [0] * 10
        for r in rows:
            w = max(0.0, min(1.0, float(r["weight"])))
            idx = min(9, int(w * 10))
            buckets[idx] += 1
        hist_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]

        top = sorted(rows, key=lambda r: r["count"], reverse=True)[:15]
        top_access = [{"stem": r["stem"], "count": r["count"], "tier": r["tier"]}
                      for r in top]

        scatter = [{"x": float(r["weight"]), "y": int(r["count"]),
                    "stem": r["stem"], "tier": r["tier"], "pinned": int(r["pinned"])}
                   for r in rows]

        by_date = defaultdict(int)
        for r in rows:
            d = _parse_iso(r["first_seen"])
            if d:
                by_date[d.date().isoformat()] += 1
        growth = []
        run = 0
        for day in sorted(by_date):
            run += by_date[day]
            growth.append({"date": day, "cumulative": run})

        risk = sorted(
            [r for r in rows if not r["pinned"]],
            key=lambda r: (float(r["weight"]), r["last_used"] or "")
        )[:12]

        def age_days(r):
            d = _parse_iso(r["last_used"])
            return (now - d).days if d else None

        stale = [{"stem": r["stem"], "weight": round(float(r["weight"]), 3),
                  "count": r["count"], "tier": r["tier"],
                  "age_days": age_days(r)} for r in risk]

        mem_dir = Path(memory_dir) if memory_dir is not None else default_memory_dir()

        graph = _build_graph(mem_dir, rows)
        mem_ids = {n["id"] for n in graph["nodes"] if n["kind"] == "memory"}
        code = _build_code_graph(conn, mem_dir, mem_ids)
        graph["nodes"].extend(code["nodes"])
        graph["links"].extend(code["links"])
        graph["repos"] = code["repos"]

        params, _warnings = forget.load_params(
            (mem_dir / ".weights" / "canonical.json") if mem_dir else None)
        budget = None
        if mem_dir:
            mf = mem_dir / "MEMORY.md"
            if mf.exists():
                size = mf.stat().st_size
                cap = params["memory_budget"]
                budget = {"bytes": size, "cap": cap,
                          "pct": round(100 * size / cap, 1)}

        return {
            "generated": now.strftime("%Y-%m-%d %H:%M UTC"),
            "last_consolidation": _last_consolidation(),
            "totals": {"total": total, "hot": hot, "cold": cold, "pinned": pinned},
            "tier": {"hot": hot, "cold": cold},
            "types": dict(type_counts),
            "hist": {"labels": hist_labels, "data": buckets},
            "top_access": top_access,
            "scatter": scatter,
            "growth": growth,
            "stale": stale,
            "budget": budget,
            "sessions": _session_count(),
            "thresholds": {"demote_below": params["demote_below"],
                            "promote_at": params["promote_at"]},
            "graph": graph,
        }
    finally:
        if own_conn:
            conn.close()
