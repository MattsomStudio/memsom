"""memsom_relate — GraphRAG made safe: associative 'relates-to' edges.

A SECOND edge kind ('relates-to', associative) kept strictly separate from the
came-from provenance edges.  Provenance-safe traversal: the integrity floor
propagates along the PATH, so a poisoned neighbor cannot ride into the result
set, and nothing beyond it can either.

Important: clearance filters RESULTS only — an above-clearance node still
conducts integrity propagation.  Integrity, not confidentiality, governs the
BFS floor.

Public API
----------
migrate(conn)                                    idempotent schema migration
relate(conn, a, b, kind='relates-to')            insert undirected rel_edge
neighborhood(conn, nid, hops=2,
             min_integrity=0, clearance=3)
    -> list[dict]                                BFS with widest-path relaxation

CLI
---
relate <a> <b> [--kind k]
neighborhood <id> [--hops N] [--min-integrity NAME] [--clearance NAME]
register(subparsers)
main(argv=None)
"""

import collections
import sys

import memsom
import memsom_schema
import memsom_confid
import memsom_quarantine
import memsom_redact

# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_REL_EDGES_DDL = """\
CREATE TABLE IF NOT EXISTS rel_edges (
  a          INTEGER NOT NULL REFERENCES nodes(id),
  b          INTEGER NOT NULL REFERENCES nodes(id),
  kind       TEXT    NOT NULL DEFAULT 'relates-to',
  created_at TEXT    NOT NULL,
  PRIMARY KEY (a, b, kind)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_rel_edges_b ON rel_edges(b);"""


def migrate(conn):
    """Idempotent migration for memsom_relate.

    Chains confid / quarantine / redact migrations first, then creates rel_edges.
    Demo #1 is untouched (new table only; all columns have defaults).
    """
    memsom_confid.migrate(conn)
    memsom_quarantine.migrate(conn)
    memsom_redact.migrate(conn)
    memsom_schema.ensure_table(conn, _REL_EDGES_DDL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_integrity(val):
    """Accept int 0-3 or a RANK key (case-insensitive). Raise ValueError otherwise."""
    if isinstance(val, int):
        if 0 <= val <= 3:
            return val
        raise ValueError(f"integrity label must be 0-3, got {val!r}")
    if isinstance(val, str):
        key = val.lower()
        if key in memsom.RANK:
            return memsom.RANK[key]
        try:
            n = int(key)
        except ValueError:
            raise ValueError(f"unknown integrity label {val!r}; valid = {list(memsom.RANK)}") from None
        if 0 <= n <= 3:
            return n
        raise ValueError(f"integrity label must be 0-3, got {n!r}")
    raise ValueError(f"parse_integrity expects int or str, got {type(val).__name__}")


def _node_label(conn, nid):
    """Return the integrity label for *nid*, or -1 if not found."""
    row = conn.execute("SELECT label FROM nodes WHERE id=?", (nid,)).fetchone()
    return row[0] if row else -1


def _node_conf(conn, nid):
    """Return conf_label for *nid* (0 if column absent or node missing)."""
    try:
        row = conn.execute("SELECT conf_label FROM nodes WHERE id=?", (nid,)).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def _is_dead(conn, nid):
    """Return True if node is tombstoned or quarantined (excluded from BFS conduct)."""
    row = conn.execute(
        "SELECT tombstoned, status FROM nodes WHERE id=?", (nid,)
    ).fetchone()
    if row is None:
        return True
    tombstoned, status = row
    return bool(tombstoned) or (status == "quarantined")


def _is_redacted(conn, nid):
    """Return True if node has the redacted flag set."""
    try:
        row = conn.execute("SELECT redacted FROM nodes WHERE id=?", (nid,)).fetchone()
        return bool(row[0]) if row else False
    except Exception:
        return False


def _is_archived(conn, nid):
    """RELATE-A1: True if node is archived (consolidated-away). Such nodes must
    not re-enter the neighborhood result pool — the other read paths exclude them
    via taint_filter_clauses; this is the relate mirror."""
    if not memsom_schema.column_exists(conn, "nodes", "archived"):
        return False
    row = conn.execute("SELECT archived FROM nodes WHERE id=?", (nid,)).fetchone()
    return bool(row[0]) if row else False


def _rel_neighbors(conn, nid):
    """Return all undirected rel_edges neighbors of *nid* (any kind)."""
    rows = conn.execute(
        "SELECT b FROM rel_edges WHERE a=? "
        "UNION "
        "SELECT a FROM rel_edges WHERE b=?",
        (nid, nid)
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def relate(conn, a, b, kind="relates-to"):
    """Create an associative edge between nodes *a* and *b*.

    Idempotent (INSERT OR IGNORE).  Raises ValueError if either id is unknown
    or a == b.  Stored once; traversal treats the edge as undirected.
    """
    if a == b:
        raise ValueError(f"cannot relate a node to itself (id={a})")
    if memsom.get_node(conn, a) is None:
        raise ValueError(f"unknown node id: {a}")
    if memsom.get_node(conn, b) is None:
        raise ValueError(f"unknown node id: {b}")
    ts = memsom.now_iso()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO rel_edges(a, b, kind, created_at) VALUES (?,?,?,?)",
            (a, b, kind, ts)
        )


def neighborhood(conn, nid, hops=2, min_integrity=0, clearance=3):
    """BFS over rel_edges with widest-path (highest floor) relaxation.

    Parameters
    ----------
    nid           : int    — start node (ValueError if unknown)
    hops          : int    — maximum number of hops from nid
    min_integrity : int|str — minimum path_min for a node to appear in results
    clearance     : int|str — maximum conf_label allowed in results
                              (above-clearance nodes still CONDUCT integrity;
                              confidentiality filters results only)

    Algorithm
    ---------
    Widest-path BFS with relaxation:
    - best[n] = highest path_min seen so far to reach n
    - For each frontier node n, for each undirected rel_edges neighbor m:
        * Skip m if tombstoned OR quarantined (dead/quarantined nodes neither
          appear nor conduct further)
        * candidate = min(best[n], label(m))
        * If candidate > best.get(m, -1): update best[m], record hops (first
          time only), add m to next frontier (re-add on improvement so a
          better floor can propagate within the hop budget)

    Returns
    -------
    list[dict] with keys:
        id, channel, label, label_name, conf_label, conf_name,
        path_min, path_min_name, hops, line
    where line = '[REDACTED]' if the node is redacted else memsom.snippet(content).
    Ordered: hops ASC, label DESC, id ASC.

    Note on clearance: a node with conf_label > clearance is invisible in the
    result set but is NOT excluded from conducting integrity.  Integrity, not
    confidentiality, governs the BFS floor.
    """
    min_integrity = _parse_integrity(min_integrity)
    clearance = memsom_confid.parse_conf(clearance)

    start = memsom.get_node(conn, nid)
    if start is None:
        raise ValueError(f"unknown node id: {nid}")

    start_label = start["label"]

    # best[node_id] = highest path_min so far (integrity floor from nid to that node)
    best = {nid: start_label}
    # hop_dist[node_id] = number of hops from nid (first time reached)
    hop_dist = {}

    frontier = collections.deque([nid])

    for _hop in range(hops):
        if not frontier:
            break
        next_frontier = []
        while frontier:
            n = frontier.popleft()
            n_best = best[n]
            for m in _rel_neighbors(conn, n):
                if m == nid:
                    continue
                # Dead/quarantined nodes neither appear nor conduct
                if _is_dead(conn, m):
                    continue
                m_label = _node_label(conn, m)
                candidate = min(n_best, m_label)
                if candidate > best.get(m, -1):
                    improved = m not in best
                    best[m] = candidate
                    if m not in hop_dist:
                        hop_dist[m] = _hop + 1
                    next_frontier.append(m)
        frontier = collections.deque(next_frontier)

    # Build result list
    result = []
    for m, path_min in best.items():
        if m == nid:
            continue
        if path_min < min_integrity:
            continue
        # RELATE-A2: coerce conf to a valid int 0-3 at the single chokepoint, so
        # the comparison, the clearance filter, AND the CONF_NAME lookup are all
        # safe from a malformed conf_label — not dependent on FED-DOS-2 being the
        # sole gatekeeper.
        try:
            c_label = min(3, max(0, int(_node_conf(conn, m) or 0)))
        except (ValueError, TypeError):
            c_label = 0
        if c_label > clearance:
            continue
        if _is_archived(conn, m):   # RELATE-A1: consolidated-away nodes stay out
            continue

        row = conn.execute(
            "SELECT id, content, channel, label FROM nodes WHERE id=?", (m,)
        ).fetchone()
        if row is None:
            continue
        node_id, content, channel, label = row

        if _is_redacted(conn, m):
            line = "[REDACTED]"
        else:
            line = memsom.snippet(content)

        result.append({
            "id": node_id,
            "channel": channel,
            "label": label,
            # RELATE-A2: defensive .get — an out-of-range label/conf (e.g. from a
            # malformed federation import) must not crash the formatter. The
            # federation conf clamp fixes the root; this is defense-in-depth.
            "label_name": memsom.NAME.get(label, "?"),
            "conf_label": c_label,
            "conf_name": memsom_confid.CONF_NAME.get(c_label, "?"),
            "path_min": path_min,
            "path_min_name": memsom.NAME.get(path_min, "?"),
            "hops": hop_dist.get(m, 1),
            "line": line,
        })

    # Sort: hops ASC, label DESC, id ASC
    result.sort(key=lambda d: (d["hops"], -d["label"], d["id"]))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_relate(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        kind = args.kind or "relates-to"
        try:
            relate(conn, args.a, args.b, kind)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"related [{args.a}] <-> [{args.b}] ({kind})")
    finally:
        conn.close()


def cmd_neighborhood(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        hops = args.hops if args.hops is not None else 2
        min_int = args.min_integrity if args.min_integrity is not None else 0
        clr = args.clearance if args.clearance is not None else 3
        try:
            nodes = neighborhood(conn, args.id, hops=hops,
                                 min_integrity=min_int, clearance=clr)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        if not nodes:
            print("no neighbors pass the filters")
        else:
            for d in nodes:
                print(
                    f"[{d['id']}] {d['channel']}"
                    f"  integrity={d['label']}"
                    f"  path-min={d['path_min']}"
                    f"  conf={d['conf_label']}"
                    f"  hops={d['hops']}"
                    f'  "{d["line"]}..."'
                )
    finally:
        conn.close()


def register(subparsers):
    p_rel = subparsers.add_parser(
        "relate", help="create an associative rel_edge between two nodes"
    )
    p_rel.add_argument("a", type=int)
    p_rel.add_argument("b", type=int)
    p_rel.add_argument("--kind", default="relates-to")
    p_rel.set_defaults(func=cmd_relate)

    p_nb = subparsers.add_parser(
        "neighborhood",
        help="BFS over rel_edges with integrity-floor propagation"
    )
    p_nb.add_argument("id", type=int)
    p_nb.add_argument("--hops", type=int, default=None)
    p_nb.add_argument("--min-integrity", default=None, dest="min_integrity")
    p_nb.add_argument("--clearance", default=None)
    p_nb.set_defaults(func=cmd_neighborhood)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_relate")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
