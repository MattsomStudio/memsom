"""memsom_corroborate — Corroboration v1: registered-roots-only, capped at agent-derived(1), fail-closed.

Invariants (load-bearing):
  - Corroboration LIFTS external(0) -> agent-derived(1) ONLY. The cap is RANK["agent-derived"] = 1
    and is never influenced by min(parents). It NEVER mints user(2) or endorsed(3).
  - Independence is by ROOT, not by node count. Two nodes under the same root count once.
  - Fail-closed: assertions under unregistered roots are REJECTED (ValueError), nothing recorded.
  - Lift drop on revocation is NATIVE: the lift node is a CHILD of every asserting node,
    so revoke_cascade on any asserting node tombstones the lift automatically.
  - Lift nodes have a row in the elevations table so that recompute_all treats them as
    fixed points and never claws label 1 back down to min(parents)=0.
  - Idempotent: corroborate() returns the existing live lift if one exists; only mints
    a new node when there is no live lift and the threshold is met.
  - History is immutable: once a lift is tombstoned (by cascade) it stays tombstoned.
    A subsequent corroborate() mints a NEW node; elevation rows are never deleted.

Public API
----------
migrate(conn)
extract_claim(text) -> tuple[str,str,str] | None
register_root(conn, root, by, descr=None) -> bool
assert_claim(conn, node_id, triple, independence_root) -> int  (claim_id)
live_root_count(conn, claim_id) -> int
corroborate(conn, claim_id, k=2) -> int | None
list_claims(conn) -> list[dict]
list_roots(conn) -> list[dict]

CLI
---
register-root <root> [--descr D] --by <who>
assert-claim <node> --root <root> [--auto | --subject S --predicate P --value V]
corroborate <claim_id> [--k N]
claims-list
roots-list

register(subparsers) mounts into a unified CLI.
main(argv=None) is provided as a standalone entry point.
"""

import re
import sys

import memsom
import memsom_schema
import memsom_rederive
import memsom_trust
import memsom_recompute

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTRACTOR_VERSION = "v1"


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Create all corroboration tables idempotently."""
    # The elevations table must exist before any mint (recompute_all needs it)
    memsom_trust.migrate(conn)
    # The recipe table must exist before any mint (record_recipe writes to it)
    memsom_rederive.migrate(conn)

    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS independence_roots (
  root TEXT PRIMARY KEY,
  descr TEXT,
  added_by TEXT NOT NULL,
  added_at TEXT NOT NULL
);""")

    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  value TEXT NOT NULL,
  extractor_version TEXT NOT NULL,
  UNIQUE(subject, predicate, value)
);""")

    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS claim_assertions (
  claim_id INTEGER NOT NULL REFERENCES claims(id),
  node_id INTEGER NOT NULL REFERENCES nodes(id),
  independence_root TEXT NOT NULL REFERENCES independence_roots(root),
  PRIMARY KEY (claim_id, node_id)
) WITHOUT ROWID;""")

    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS corroborations (
  claim_id INTEGER NOT NULL REFERENCES claims(id),
  node_id INTEGER NOT NULL REFERENCES nodes(id),
  k_used INTEGER NOT NULL,
  roots_count INTEGER NOT NULL,
  ts TEXT NOT NULL,
  PRIMARY KEY (claim_id, node_id)
) WITHOUT ROWID;""")


# ---------------------------------------------------------------------------
# extract_claim — deterministic, narrow, conservative. No LLM, no semantics.
# ---------------------------------------------------------------------------

def extract_claim(text):
    """Extract a structured (subject, predicate, value) triple from *text*.

    Patterns tried in order (first match wins):
      1. sha256: 64 hex chars
      2. ipv4: dotted-quad
      3. port: "port NNNN" (1-65535)
      4. semver: X.Y.Z
      5. key=value: first matching line

    Returns a 3-tuple of non-empty strings, or None if nothing matches.
    The order is load-bearing (sha256 before ipv4 so hex runs aren't misread;
    ipv4 before semver so 192.168.1.107 isn't parsed as a semver fragment).
    """
    # 1. sha256: exactly 64 hex chars as a word
    m = re.search(r"\b([0-9a-fA-F]{64})\b", text)
    if m:
        return ("sha256", "is", m.group(1).lower())

    # 2. ipv4: strict dotted-quad
    m = re.search(
        r"\b((?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d))\b",
        text
    )
    if m:
        return ("ipv4", "is", m.group(1))

    # 3. port: "port NNNN" (case-insensitive), 1..65535
    m = re.search(r"\bport\s+(\d{1,5})\b", text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 65535:
            return ("port", "is", m.group(1))

    # 4. semver: X.Y.Z
    m = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
    if m:
        return ("version", "is", m.group(1))

    # 5. key=value: first matching line
    for line in text.splitlines():
        lm = re.match(r"^\s*([A-Za-z_][\w.-]*)\s*=\s*(\S+)\s*$", line)
        if lm:
            return (lm.group(1), "=", lm.group(2))

    return None


# ---------------------------------------------------------------------------
# register_root
# ---------------------------------------------------------------------------

def register_root(conn, root, by, descr=None):
    """Register an independence root.

    Returns True if the root was newly inserted, False if it already existed.
    Raises ValueError if root is empty or whitespace.

    This is the ONLY path that grants independence credit.
    """
    if not root or not root.strip():
        raise ValueError("root must be a non-empty, non-whitespace string")

    migrate(conn)

    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO independence_roots(root, descr, added_by, added_at)"
            " VALUES (?,?,?,?)",
            (root, descr, by, memsom.now_iso())
        )
        changed = conn.execute("SELECT changes()").fetchone()[0]

    return changed == 1


# ---------------------------------------------------------------------------
# assert_claim
# ---------------------------------------------------------------------------

def assert_claim(conn, node_id, triple, independence_root):
    """Record that *node_id* asserts *triple* under *independence_root*.

    FAIL CLOSED: raises ValueError immediately if independence_root is not
    registered — nothing is recorded, no credit is given.

    triple must be a 3-tuple of non-empty strings.
    node_id must exist (live or tombstoned) in the nodes table.

    Returns the claim_id.
    """
    migrate(conn)

    # Fail-closed: reject unregistered roots before touching anything
    if conn.execute(
        "SELECT 1 FROM independence_roots WHERE root=?", (independence_root,)
    ).fetchone() is None:
        raise ValueError(
            f"unregistered independence root: {independence_root!r} - "
            "no corroboration credit (fail-closed)"
        )

    # Validate node existence
    if memsom.get_node(conn, node_id) is None:
        raise ValueError(f"unknown node id: {node_id}")

    # Validate triple
    if (not isinstance(triple, (tuple, list)) or len(triple) != 3
            or not all(isinstance(s, str) and s for s in triple)):
        raise ValueError("triple must be a 3-tuple of non-empty strings")

    subject, predicate, value = triple

    with conn:
        # INSERT OR IGNORE: if claim already exists, the UNIQUE constraint no-ops
        conn.execute(
            "INSERT OR IGNORE INTO claims(subject, predicate, value, extractor_version)"
            " VALUES (?,?,?,?)",
            (subject, predicate, value, EXTRACTOR_VERSION)
        )
        claim_id = conn.execute(
            "SELECT id FROM claims WHERE subject=? AND predicate=? AND value=?",
            (subject, predicate, value)
        ).fetchone()[0]

        # INSERT OR IGNORE: PK(claim_id, node_id) — first root wins, immutable history
        conn.execute(
            "INSERT OR IGNORE INTO claim_assertions(claim_id, node_id, independence_root)"
            " VALUES (?,?,?)",
            (claim_id, node_id, independence_root)
        )

    return claim_id


# ---------------------------------------------------------------------------
# live_root_count
# ---------------------------------------------------------------------------

def live_root_count(conn, claim_id):
    """Count distinct LIVE registered independence roots for *claim_id*.

    Two nodes under the same root count as one (independence is by root).
    The JOIN to independence_roots is belt-and-braces with the FK constraint.
    """
    row = conn.execute(
        "SELECT COUNT(DISTINCT ca.independence_root)"
        " FROM claim_assertions ca"
        " JOIN nodes n ON n.id = ca.node_id"
        " JOIN independence_roots ir ON ir.root = ca.independence_root"
        " WHERE ca.claim_id = ? AND n.tombstoned = 0",
        (claim_id,)
    ).fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# corroborate
# ---------------------------------------------------------------------------

def corroborate(conn, claim_id, k=2):
    """Attempt to mint a corroboration lift node for *claim_id* at threshold *k*.

    Rules:
      - k < 1 -> ValueError
      - Claim must exist -> ValueError
      - Idempotent: if a live lift exists, return it (never double-mint)
      - If live distinct registered roots < k: return None
      - Otherwise: mint ONE agent-derived node (label=1, THE CAP — never higher),
        wire edges to all live asserting nodes, insert elevation row (fixed point),
        insert corroborations row, call recompute_all.

    Returns lift node_id (int) or None.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    migrate(conn)

    # Claim must exist
    claim_row = conn.execute(
        "SELECT subject, predicate, value FROM claims WHERE id=?", (claim_id,)
    ).fetchone()
    if claim_row is None:
        raise ValueError(f"unknown claim id: {claim_id}")

    subject, predicate, value = claim_row

    # CORROBORATE-1/2: every decision read AND the mint must happen inside ONE
    # write transaction. The previous code read idempotency / root-count /
    # asserting-set BEFORE acquiring the lock and used them verbatim inside it —
    # so a concurrent revoke landing in the window could strand a stale count
    # (lift minted "confirmed by N" with < k live roots, wired to an already
    # tombstoned parent, recorded as a fixed point recompute can never claw back),
    # and two concurrent writers could both pre-read "no live lift" and double-mint.
    # Re-reading under BEGIN IMMEDIATE is the actual TOCTOU guard derive_node uses.
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        # 1. Idempotency (re-checked under the lock): return existing live lift.
        row = conn.execute(
            "SELECT c.node_id FROM corroborations c"
            " JOIN nodes n ON n.id = c.node_id"
            " WHERE c.claim_id = ? AND n.tombstoned = 0",
            (claim_id,)
        ).fetchone()
        if row:
            return row[0]

        # 2. Live root count (re-checked under the lock).
        n_roots = live_root_count(conn, claim_id)
        if n_roots < k:
            return None

        # 3. Collect live asserting nodes (re-checked under the lock, ordered).
        asserting = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT ca.node_id"
                " FROM claim_assertions ca"
                " JOIN nodes n ON n.id = ca.node_id"
                " JOIN independence_roots ir ON ir.root = ca.independence_root"
                " WHERE ca.claim_id = ? AND n.tombstoned = 0"
                " ORDER BY ca.node_id",
                (claim_id,)
            ).fetchall()
        ]

        # 4. Mint.
        content = (
            f"CORROBORATED claim [{claim_id}] {subject} {predicate} {value}"
            f" - confirmed by {n_roots} independent registered roots (k={k})."
        )

        # THE CAP: label=RANK["agent-derived"]=1, never min(parents), never user/endorsed
        lift = memsom.insert_node(conn, content, "agent-derived",
                                  label=memsom.RANK["agent-derived"])
        assert lift is not None
        # Verify the cap is holding (belt-and-braces; catches refactoring accidents)
        assert memsom.RANK["agent-derived"] == 1, "cap constant drift — fix before proceeding"

        # Wire edges: lift -> each asserting node
        conn.executemany(
            "INSERT INTO edges(child, parent) VALUES (?,?)",
            [(lift, a) for a in asserting]
        )
        # Recipe: corroborate count template. Parked (not deterministically replayed)
        # in v1 — a corroborate lift is a descendant, so revoke tombstones it anyway.
        memsom_rederive.record_recipe(conn, lift, "corroborate")

        # CORR-CONF-1: stamp the lift's confidentiality high-water at the mint
        # (same transaction). corroborate only calls recompute_all (integrity);
        # the conf axis is never recomputed for the lift, so insert_node's DEFAULT
        # conf_label=0 would make a lift over SECRET asserting nodes readable at
        # PUBLIC. Inline (not memsom_confid.recompute_conf, whose own `with conn:`
        # would commit this transaction early). Guarded: conf_label may be absent.
        if memsom_schema.column_exists(conn, "nodes", "conf_label"):
            lift_conf = conn.execute(
                "SELECT COALESCE(MAX(n.conf_label), 0) FROM edges e"
                " JOIN nodes n ON n.id = e.parent"
                " WHERE e.child = ? AND n.tombstoned = 0",
                (lift,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE nodes SET conf_label = ? WHERE id = ?", (lift_conf, lift)
            )

        # Elevation row: marks this node as a fixed point for recompute_all
        conn.execute(
            "INSERT INTO elevations"
            "(node, from_label, to_label, reason, elevated_by, forced, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                lift,
                0,  # from_label: external is the floor of the parents
                1,  # to_label: agent-derived, the cap
                f"corroboration: claim {claim_id} confirmed by {n_roots}"
                " independent registered roots",
                "memsom_corroborate",
                0,  # not forced (this is the designed path, not an override)
                memsom.now_iso(),
            )
        )

        # Audit row for the corroboration
        conn.execute(
            "INSERT INTO corroborations(claim_id, node_id, k_used, roots_count, ts)"
            " VALUES (?,?,?,?,?)",
            (claim_id, lift, k, n_roots, memsom.now_iso())
        )

    # 5. Re-floor downstream nodes; the lift survives via its elevations row
    memsom_recompute.recompute_all(conn)

    return lift


# ---------------------------------------------------------------------------
# list_claims / list_roots
# ---------------------------------------------------------------------------

def list_claims(conn):
    """Return a list of dicts, one per claim.

    Each dict: id, subject, predicate, value, assertion_count,
               live_root_count (int), lift_node_id (int or None).
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id, subject, predicate, value FROM claims ORDER BY id"
    ).fetchall()
    result = []
    for cid, subject, predicate, value in rows:
        assertion_count = conn.execute(
            "SELECT COUNT(*) FROM claim_assertions WHERE claim_id=?", (cid,)
        ).fetchone()[0]
        lrc = live_root_count(conn, cid)
        lift_row = conn.execute(
            "SELECT c.node_id FROM corroborations c"
            " JOIN nodes n ON n.id = c.node_id"
            " WHERE c.claim_id = ? AND n.tombstoned = 0",
            (cid,)
        ).fetchone()
        result.append({
            "id": cid,
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "assertion_count": assertion_count,
            "live_root_count": lrc,
            "lift_node_id": lift_row[0] if lift_row else None,
        })
    return result


def list_roots(conn):
    """Return a list of dicts, one per independence root.

    Each dict: root, descr, added_by, added_at.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT root, descr, added_by, added_at FROM independence_roots ORDER BY added_at, root"
    ).fetchall()
    return [
        {"root": r[0], "descr": r[1], "added_by": r[2], "added_at": r[3]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def cmd_register_root(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        try:
            inserted = register_root(conn, args.root, args.by, descr=args.descr)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        if inserted:
            print(f"registered root '{args.root}'")
        else:
            print(f"root already registered")
    finally:
        conn.close()


def cmd_assert_claim(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        node_id = args.node

        # Resolve the triple
        if args.auto:
            node = memsom.get_node(conn, node_id)
            if node is None:
                print(f"[memsom] unknown node id: {node_id}", file=sys.stderr)
                sys.exit(1)
            triple = extract_claim(node["content"])
            if triple is None:
                print(
                    f"[memsom] no structured claim found in node [{node_id}]"
                    " (extractor is conservative)",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            if not (args.subject and args.predicate and args.value):
                print(
                    "[memsom] explicit triple requires --subject, --predicate, and --value",
                    file=sys.stderr,
                )
                sys.exit(2)
            triple = (args.subject, args.predicate, args.value)

        try:
            cid = assert_claim(conn, node_id, triple, args.root)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)

        s, p, v = triple
        print(f"claim [{cid}] {s} {p} {v}  asserted by node [{node_id}] under root '{args.root}'")
    finally:
        conn.close()


def cmd_corroborate(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        try:
            lift = corroborate(conn, args.claim_id, k=args.k)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)

        if lift is not None:
            node = memsom.get_node(conn, lift)
            n_roots = conn.execute(
                "SELECT roots_count FROM corroborations WHERE node_id=?", (lift,)
            ).fetchone()[0]
            print(
                f"LIFT: minted node [{lift}] integrity=AGENT-DERIVED (capped)"
                f" from {n_roots} roots"
            )
        else:
            # Count how many live roots we have vs how many are needed
            # (claim_id must be valid to get this far)
            actual = conn.execute(
                "SELECT COUNT(DISTINCT ca.independence_root)"
                " FROM claim_assertions ca"
                " JOIN nodes n ON n.id = ca.node_id"
                " JOIN independence_roots ir ON ir.root = ca.independence_root"
                " WHERE ca.claim_id = ? AND n.tombstoned = 0",
                (args.claim_id,)
            ).fetchone()[0]
            print(
                f"no lift: {actual} of {args.k} required independent registered roots"
            )
    finally:
        conn.close()


def cmd_claims_list(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        claims = list_claims(conn)
        if not claims:
            print("no claims")
        else:
            for c in claims:
                lift_str = f"lift=[{c['lift_node_id']}]" if c["lift_node_id"] else "no lift"
                print(
                    f"[{c['id']}] {c['subject']} {c['predicate']} {c['value']}"
                    f"  assertions={c['assertion_count']}"
                    f"  live_roots={c['live_root_count']}"
                    f"  {lift_str}"
                )
    finally:
        conn.close()


def cmd_roots_list(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        roots = list_roots(conn)
        if not roots:
            print("no roots registered")
        else:
            for r in roots:
                descr = f"  ({r['descr']})" if r["descr"] else ""
                print(f"'{r['root']}' by={r['added_by']}  added={r['added_at']}{descr}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# register / main
# ---------------------------------------------------------------------------

def register(subparsers):
    """Mount corroborate subcommands onto *subparsers*."""

    # register-root
    p_rr = subparsers.add_parser(
        "register-root", help="register an independence root for corroboration"
    )
    p_rr.add_argument("root", help="root identifier (e.g. 'nist-sp800-63', 'cve-2024-1234')")
    p_rr.add_argument("--descr", default=None, help="optional description of this root")
    p_rr.add_argument("--by", required=True, help="who is registering this root")
    p_rr.set_defaults(func=cmd_register_root)

    # assert-claim
    p_ac = subparsers.add_parser(
        "assert-claim", help="assert a structured claim under a registered root"
    )
    p_ac.add_argument("node", type=int, help="node id whose content contains the claim")
    p_ac.add_argument("--root", required=True, help="independence root (must be registered)")
    p_ac.add_argument("--auto", action="store_true",
                      help="auto-extract claim from node content (conservative extractor)")
    p_ac.add_argument("--subject", default=None, help="claim subject (explicit triple)")
    p_ac.add_argument("--predicate", default=None, help="claim predicate (explicit triple)")
    p_ac.add_argument("--value", default=None, help="claim value (explicit triple)")
    p_ac.set_defaults(func=cmd_assert_claim)

    # corroborate
    p_corr = subparsers.add_parser(
        "corroborate", help="mint a corroboration lift node when k roots agree"
    )
    p_corr.add_argument("claim_id", type=int, help="claim id to corroborate")
    p_corr.add_argument("--k", type=int, default=2,
                         help="required number of independent roots (default 2)")
    p_corr.set_defaults(func=cmd_corroborate)

    # claims-list
    subparsers.add_parser(
        "claims-list", help="list all claims"
    ).set_defaults(func=cmd_claims_list)

    # roots-list
    subparsers.add_parser(
        "roots-list", help="list all registered independence roots"
    ).set_defaults(func=cmd_roots_list)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_corroborate")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
