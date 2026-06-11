#!/usr/bin/env python3
"""memdag — derivation-DAG memory store, explain/revoke vertical slice.

Every memory is a node; edges mean CAME-FROM (provenance), not relates-to.
Invariants (locked — see Vault Security/Teachings/2026-06-10-derivation-dag-product-reframe.md):
  1. Integrity labels are assigned by CHANNEL, never by content:
     endorsed(3) > user(2) > agent-derived(1) > external(0).
  2. A derived node's label = min(parent labels) — Biba low-water-mark, one hop.
  3. History is immutable: changes mint NEW nodes; rows are never edited or deleted.
  4. revoke = tombstone + cascade to all transitive descendants. Rows, edges and
     payloads all survive (redaction is a separate, out-of-scope mode); liveness
     is filtered at READ time via WHERE tombstoned=0.
  5. ask refuses to compose from zero live sources — no unprovenanced answers.

CLI: seed [--offline] [--reset] · ask "question" · explain <id> ·
     revoke <id> [--reason ...] [--yes]  (dry-run by default) · dump
DB:  memdag.db beside this file (override: MEMDAG_DB env var). Keep it OUT of
     Syncthing-synced trees — same rule as sessions.db.
"""

import argparse, os, re, sqlite3, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HOME = Path(__file__).resolve().parent
VAULT_SRC = Path("C:/Users/you/Vault/Notes/Lab/mesh-notes.md")
EXT_URL = "https://raw.githubusercontent.com/slackhq/nebula/master/README.md"
FALLBACK = HOME / "external_fallback.txt"
RANK = {"endorsed": 3, "user": 2, "agent-derived": 1, "external": 0}
NAME = {3: "ENDORSED", 2: "USER", 1: "AGENT-DERIVED", 0: "EXTERNAL"}
STOP = {"how", "should", "i", "a", "an", "the", "do", "does", "what", "my",
        "to", "is", "it", "of", "for", "with", "in", "on", "and", "or"}
USER_FACT = ("Nebula cert policy: follow the documented cert policy with the 'trusted' "
             "group - give each device its own group and an explicit host-name rule "
             "in host.yml inbound (my rule since the Kali untrust).")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  content       TEXT    NOT NULL,
  channel       TEXT    NOT NULL
                CHECK (channel IN ('endorsed','user','agent-derived','external')),
  label         INTEGER NOT NULL CHECK (label BETWEEN 0 AND 3),
  source_ref    TEXT,
  created_at    TEXT    NOT NULL,
  tombstoned    INTEGER NOT NULL DEFAULT 0,
  tombstoned_at TEXT,
  revoke_reason TEXT
);
CREATE TABLE IF NOT EXISTS edges (
  child  INTEGER NOT NULL REFERENCES nodes(id),
  parent INTEGER NOT NULL REFERENCES nodes(id),
  PRIMARY KEY (child, parent)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent);
"""

CASCADE_CTE = """WITH RECURSIVE descendants(id) AS (
  SELECT ? UNION SELECT e.child FROM edges e JOIN descendants d ON e.parent = d.id
)"""  # UNION (not UNION ALL): dedupes -> terminates on cycles, visits diamonds once


def db_path():
    return Path(os.environ.get("MEMDAG_DB") or HOME / "memdag.db")


def now_iso():
    # ISO-8601 TEXT, never datetime objects (3.12 sqlite3 adapter is deprecated)
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection(path=None):
    path = Path(path or db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")  # per-connection, OFF by default
    conn.executescript(SCHEMA)
    return conn


# ---- store primitives (library discipline: no prints, no sys.exit) ----

def insert_node(conn, content, channel, label=None, source_ref=None):
    if label is None:
        label = RANK[channel]  # labels come from the channel; explicit label is the
    cur = conn.execute(        # derive/manual-elevation path only
        "INSERT INTO nodes(content, channel, label, source_ref, created_at) VALUES (?,?,?,?,?)",
        (content, channel, label, source_ref, now_iso()))
    return cur.lastrowid


def derive_node(conn, content, parent_ids):
    if not parent_ids:
        raise ValueError("derived node needs at least one parent")
    qmarks = ",".join("?" * len(parent_ids))
    with conn:  # liveness check + node + edges are ONE unit, write-locked up front:
        if not conn.in_transaction:  # a revoke can't land between check and insert (TOCTOU)
            conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(f"SELECT id, label, tombstoned FROM nodes WHERE id IN ({qmarks})",
                            tuple(parent_ids)).fetchall()
        if len(rows) != len(set(parent_ids)):
            raise ValueError("unknown parent id")
        if any(t for _, _, t in rows):
            raise ValueError("tombstoned parent")
        label = min(l for _, l, _ in rows)
        nid = insert_node(conn, content, "agent-derived", label)
        conn.executemany("INSERT INTO edges(child, parent) VALUES (?,?)",
                         [(nid, p) for p in set(parent_ids)])
    return nid, label


def get_node(conn, nid):
    row = conn.execute(
        "SELECT id, content, channel, label, source_ref, created_at,"
        " tombstoned, tombstoned_at, revoke_reason FROM nodes WHERE id = ?", (nid,)).fetchone()
    if not row:
        return None
    keys = ("id", "content", "channel", "label", "source_ref", "created_at",
            "tombstoned", "tombstoned_at", "revoke_reason")
    return dict(zip(keys, row))


def live_sources(conn):
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE tombstoned = 0 AND channel != 'agent-derived'"
        " ORDER BY label DESC, id ASC").fetchall()


def parents_of(conn, nid):
    return conn.execute(
        "SELECT n.id, n.content, n.channel, n.label, n.source_ref, n.created_at,"
        " n.tombstoned, n.tombstoned_at, n.revoke_reason"
        " FROM edges e JOIN nodes n ON n.id = e.parent"
        " WHERE e.child = ? ORDER BY n.label DESC, n.id ASC", (nid,)).fetchall()


def cascade_set(conn, seed):
    return conn.execute(
        CASCADE_CTE + " SELECT n.id, n.channel, n.tombstoned FROM nodes n"
        " WHERE n.id IN (SELECT id FROM descendants) ORDER BY n.id", (seed,)).fetchall()


def revoke_cascade(conn, seed, reason):
    with conn:
        conn.execute(
            CASCADE_CTE + """
            UPDATE nodes SET tombstoned = 1, tombstoned_at = ?,
                   revoke_reason = CASE WHEN id = ? THEN ? ELSE 'cascade from node ' || ? END
             WHERE id IN (SELECT id FROM descendants) AND tombstoned = 0""",
            (seed, now_iso(), seed, reason, seed))
        # cursor.rowcount is -1 for WITH-prefixed DML (sqlite3 misdetects it as a query)
        n = conn.execute("SELECT changes()").fetchone()[0]
    return n  # first death wins: already-dead rows keep their record


# ---- deterministic answer composition ----

def stems(text):
    return {w[:6] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def prose_lines(content):
    """Yield content lines that are prose: stateful skip of YAML frontmatter and
    code-fence INTERIORS (prefix checks alone let those leak), plus markdown noise."""
    lines = content.splitlines()
    in_front = bool(lines) and lines[0].strip() == "---"
    in_fence = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if in_front:
            in_front = not (i > 0 and line == "---")
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or raw.startswith("    "):
            continue
        if not line or line.startswith(("#", "|", "---", ">", "**", "![", "[!",
                                        "- [ ]", "- [x]", "- [X]")):
            continue
        yield line


def strip_furniture(line):
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # [text](url) -> text
    return line.replace("**", "").replace("`", "")


def snippet(content, width=70):
    line = strip_furniture(next(prose_lines(content), content))
    return " ".join(line.split())[:width]


def candidate_sentences(content):
    out = []
    for line in prose_lines(content):
        line = re.sub(r"^[-*]\s+", "", strip_furniture(line))
        for sent in line.split(". "):
            sent = sent.strip().rstrip(".:")
            if 30 <= len(sent) <= 200:  # camera-readable claims, not war stories
                out.append(sent + ".")
    return out


def compose(question, sources):
    """Pure + deterministic: same inputs -> byte-identical answer. No LLM, no clock."""
    keys = stems(question)
    bullets, used = [], []
    for sid, content, channel, _label, _ref in sources:  # label DESC, id ASC = trust order
        cands = candidate_sentences(content)
        if not cands:  # nothing survived the filter: first non-empty line, capped
            first = next((l.strip() for l in content.splitlines() if l.strip()), "")
            if first:
                cands = [first[:200].rstrip(".:") + "."]
        scored = [(sum(1 for k in keys if k in s.lower()), pos, s)
                  for pos, s in enumerate(cands)]
        top = sorted([t for t in scored if t[0] > 0], key=lambda t: (-t[0], t[1]))[:2]
        picked = [s for _, _, s in sorted(top, key=lambda t: t[1])]
        if not picked and cands:  # every live source contributes >=1 claim,
            picked = [cands[0]]   # so revoking any source is visible by construction
        if picked:
            bullets += [f"- {s} [mem:{sid}|{channel}]" for s in picked]
            used.append(sid)
    if not bullets:
        return None, []
    text = f"Q: {question}\nA (composed from {len(used)} live sources):\n" + "\n".join(bullets)
    return text, used


# ---- commands ----

def fetch_external(offline):
    if not offline:
        try:
            req = urllib.request.Request(EXT_URL, headers={"User-Agent": "memdag/0.1"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read().decode("utf-8", "replace"), f"{EXT_URL} (fetched, stored)"
        except Exception as err:
            print(f"[memdag] live fetch failed ({err}); using stored fallback", file=sys.stderr)
    return FALLBACK.read_text(encoding="utf-8", errors="replace"), f"{EXT_URL} (local snapshot)"


def cmd_seed(args):
    if args.reset:
        try:
            db_path().unlink(missing_ok=True)
        except PermissionError:
            print(f"[memdag] {db_path().name} is held open by another process - close it"
                  " (DB browser? stale shell?) and re-run", file=sys.stderr)
            sys.exit(1)
    conn = get_connection()
    try:
        n = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        if n:
            print(f"[memdag] already seeded ({n} nodes); use --reset to start over",
                  file=sys.stderr)
            sys.exit(1)
        try:
            vault_text = VAULT_SRC.read_text(encoding="utf-8", errors="replace")
            ext_text, ext_ref = fetch_external(args.offline)
        except OSError as err:
            print(f"[memdag] seed source unavailable: {err}", file=sys.stderr)
            sys.exit(1)
        with conn:  # fixed insert order -> stable demo ids 1/2/3
            u = insert_node(conn, USER_FACT, "user")
            v = insert_node(conn, vault_text, "endorsed", source_ref=str(VAULT_SRC))
            e = insert_node(conn, ext_text, "external", source_ref=ext_ref)
        for nid in (u, v, e):
            node = get_node(conn, nid)
            ref = node["source_ref"] or "(stated directly)"
            print(f"[{nid}] {node['channel']:<13} integrity={NAME[node['label']]:<13}"
                  f" {len(node['content']):>6} chars  {ref}")
    finally:
        conn.close()


def cmd_ask(args):
    conn = get_connection()
    try:
        considered = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel != 'agent-derived'").fetchone()[0]
        sources = live_sources(conn)
        if not sources:
            print("[memdag] no live sources - refusing to compose an unprovenanced answer",
                  file=sys.stderr)
            sys.exit(1)
        text, used = compose(args.question, sources)
        if not text:
            print("[memdag] no live source yielded any claim; nothing stored", file=sys.stderr)
            sys.exit(1)
        nid, label = derive_node(conn, text, used)
        excluded = considered - len(sources)
        print(text)
        print(f"\nstored as node [{nid}] | integrity: {NAME[label]} (floor of {len(used)} parents)"
              f" | sources considered: {considered}, used: {len(used)},"
              f" excluded: {excluded} (tombstoned)")
    finally:
        conn.close()


def local_date(iso):
    try:  # stored UTC (canonical); shown local so an evening take doesn't say "tomorrow"
        return datetime.fromisoformat(iso).astimezone().date().isoformat()
    except (TypeError, ValueError):
        return (iso or "")[:10]


def fmt_node(node, indent=""):
    line = (f"{indent}[{node['id']}] {node['channel']}"
            f"  integrity={NAME[node['label']]}  {local_date(node['created_at'])}")
    if node["tombstoned"]:
        line += f"  [REVOKED {local_date(node['tombstoned_at'])}: {node['revoke_reason']}]"
    out = [line]
    ref = node["source_ref"] or ("(stated directly)" if node["channel"] == "user" else None)
    if ref:
        out.append(f"{indent}      {ref}")
    out.append(f'{indent}      "{snippet(node["content"])}..."')
    return out


def cmd_explain(args):
    conn = get_connection()
    try:
        node = get_node(conn, args.id)
        if not node:
            print(f"[memdag] no node [{args.id}]", file=sys.stderr)
            sys.exit(1)
        print()
        for line in fmt_node(node):
            print(line)
        keys = ("id", "content", "channel", "label", "source_ref", "created_at",
                "tombstoned", "tombstoned_at", "revoke_reason")
        seen = {node["id"]}
        # iterator stack, not recursion: a long derivation chain must not hit the
        # interpreter recursion limit (revoke's CTE already handles any depth)
        stack = [(iter(parents_of(conn, node["id"])), "  ")]
        while stack:
            it, indent = stack[-1]
            row = next(it, None)
            if row is None:
                stack.pop()
                continue
            p = dict(zip(keys, row))
            lines = fmt_node(p, indent)
            lines[0] = lines[0].replace(f"{indent}[", f"{indent}<- [", 1)
            if p["id"] in seen:
                lines[0] += "  (ancestry shown above)"  # diamond: elided, not absent
            for line in lines:
                print(line)
            if p["id"] not in seen:
                seen.add(p["id"])
                stack.append((iter(parents_of(conn, p["id"])), indent + "   "))
        parents = parents_of(conn, node["id"])
        if parents and node["channel"] == "agent-derived":
            floor = min(parents, key=lambda r: (r[3], r[0]))
            print(f"\nlabel rule: integrity = min(parent labels) = {NAME[node['label']]}"
                  f"  (floor set by [{floor[0]}])")
    finally:
        conn.close()


def cmd_revoke(args):
    conn = get_connection()
    try:
        node = get_node(conn, args.id)
        if not node:
            print(f"[memdag] no node [{args.id}]", file=sys.stderr)
            sys.exit(1)
        if node["tombstoned"]:
            print(f"[memdag] node [{args.id}] already tombstoned {node['tombstoned_at']}"
                  " - nothing to do (first death wins)")
            return
        rows = cascade_set(conn, args.id)
        pending = [r for r in rows if not r[2]]
        print(f"will tombstone {len(pending)} node(s):")
        for nid, channel, dead in rows:
            role = "seed" if nid == args.id else "descendant"
            note = "  - already tombstoned, skipped" if dead else ""
            print(f"  [{nid}] {channel} ({role}){note}")
        if not args.yes:
            print("dry run - re-run with --yes to apply.")
            return
        n = revoke_cascade(conn, args.id, args.reason)
        print(f"done - {n} tombstoned, 0 rows deleted, all edges intact.")
    finally:
        conn.close()


def cmd_dump(args):
    conn = get_connection()
    try:
        for row in conn.execute("SELECT id, channel, label, tombstoned, created_at, content"
                                " FROM nodes ORDER BY id"):
            nid, channel, label, dead, created, content = row
            flag = "T" if dead else "."
            print(f"[{nid}] {flag} {channel:<13} integrity={NAME[label]:<13}"
                  f" {local_date(created)}  {snippet(content, 60)}")
        edges = conn.execute("SELECT child, parent FROM edges ORDER BY child, parent").fetchall()
        print(f"edges ({len(edges)}):" if edges else "edges (0): none yet - ask something.")
        for child, parent in edges:
            print(f"  [{child}] <- [{parent}]")
    finally:
        conn.close()


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:  # io.StringIO under tests has no reconfigure
            pass
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s_seed = sub.add_parser("seed")
    s_seed.add_argument("--offline", action="store_true")
    s_seed.add_argument("--reset", action="store_true")
    s_seed.set_defaults(func=cmd_seed)
    s_ask = sub.add_parser("ask")
    s_ask.add_argument("question")
    s_ask.set_defaults(func=cmd_ask)
    s_explain = sub.add_parser("explain")
    s_explain.add_argument("id", type=int)
    s_explain.set_defaults(func=cmd_explain)
    s_revoke = sub.add_parser("revoke")
    s_revoke.add_argument("id", type=int)
    s_revoke.add_argument("--reason", default="revoked by user")
    s_revoke.add_argument("--yes", action="store_true")
    s_revoke.set_defaults(func=cmd_revoke)
    sub.add_parser("dump").set_defaults(func=cmd_dump)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
