"""memsom_federation — multi-machine sync that FIXES the Syncthing additive-deletion bug.

The core invariant: death (tombstone) and redaction are MONOTONIC and PROPAGATE.
Once a node is dead/redacted on ANY machine it stays dead everywhere. Importing
an older still-live copy of a locally-tombstoned/redacted node must NOT resurrect it
(first-death-wins across machines).

Public API
----------
default_origin() -> str
backfill_uuids(conn, origin) -> int
export_changeset(conn, since=None, origin=None) -> dict
write_jsonl(path, changeset)
read_jsonl(path) -> dict
import_changeset(conn, changeset) -> dict  # stats

CLI
---
export <file> [--since <iso>] [--origin <name>]
import <file>
register(subparsers)
main(argv=None)
"""

import argparse
import json
import os
import platform
import sys

import memsom
import memsom_schema
import memsom_redact
import memsom_quarantine
import memsom_confid
import memsom_recompute


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Untrusted (unregistered) origins may never self-classify as PUBLIC(0).
CONF_FLOOR_UNTRUSTED = 1  # internal — untrusted nodes may never self-classify as PUBLIC(0).


def _coerce_conf(raw, floor=0):
    """FED-DOS-2: a changeset's conf_label is attacker/transport-controlled. Coerce
    it to an int clamped to [floor, 3] — a non-numeric value must NOT crash int()
    (untrusted branch), and an out-of-range value must NOT land verbatim in the
    INTEGER column where it later crashes CONF_NAME lookups (RELATE-A2) or
    mis-compares in recompute_conf_all. Bad input falls back to the floor.
    """
    try:
        return min(3, max(floor, int(raw)))
    except (ValueError, TypeError):
        return floor


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Idempotent federation migration.

    Runs dependency migrations first, then adds federation-specific columns.
    Also ensures secondary columns needed for full changeset fidelity are present.
    """
    # Pull in dependency schema
    memsom_redact.migrate(conn)
    memsom_quarantine.migrate(conn)
    memsom_confid.migrate(conn)

    # Federation identity columns
    memsom_schema.add_column(conn, "nodes", "uuid", "TEXT")
    memsom_schema.add_column(conn, "nodes", "origin", "TEXT")

    # Ensure redact/quarantine timestamp columns exist (dependency modules add them,
    # but guard here too for forward-compat / standalone testing)
    memsom_schema.add_column(conn, "nodes", "redacted_at", "TEXT")
    memsom_schema.add_column(conn, "nodes", "redact_reason", "TEXT")
    memsom_schema.add_column(conn, "nodes", "quarantined_at", "TEXT")
    memsom_schema.add_column(conn, "nodes", "quarantine_reason", "TEXT")

    # Unique index: multiple NULLs are fine under SQLite UNIQUE (pre-federation rows)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_uuid ON nodes(uuid)"
    )

    # Trust allowlist: default-deny federation boundary.
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS trusted_origins (
    origin   TEXT PRIMARY KEY,
    descr    TEXT,
    added_by TEXT,
    added_at TEXT
  );""")
    # Auto-trust this machine so local round-trips always work.
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO trusted_origins(origin,descr,added_by,added_at)"
            " VALUES (?,?,?,?)",
            (default_origin(), "self (auto-registered)", "system", memsom.now_iso())
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_origin():
    """Return the default origin for this machine."""
    return os.environ.get("MEMDAG_ORIGIN") or platform.node() or "unknown"


# ---------------------------------------------------------------------------
# Trust-origin management (default-deny allowlist)
# ---------------------------------------------------------------------------

def is_trusted(conn, origin):
    """Return True if *origin* is in the trusted_origins allowlist."""
    if not origin:
        return False
    return conn.execute(
        "SELECT 1 FROM trusted_origins WHERE origin=?", (origin,)
    ).fetchone() is not None


def register_origin(conn, origin, by="user", descr=None):
    """Add *origin* to the trusted_origins allowlist (idempotent).

    Calls migrate() to ensure the table exists.
    Returns True if the origin is now trusted (always True on success).
    No print/sys.exit — library discipline.
    """
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO trusted_origins(origin,descr,added_by,added_at)"
            " VALUES (?,?,?,?)",
            (origin, descr, by, memsom.now_iso())
        )
    return is_trusted(conn, origin)


def list_origins(conn):
    """Return list of (origin, descr, added_by, added_at) ordered by origin."""
    migrate(conn)
    return conn.execute(
        "SELECT origin, descr, added_by, added_at FROM trusted_origins ORDER BY origin"
    ).fetchall()


def backfill_uuids(conn, origin):
    """Assign uuid=f'{origin}:{id}' and origin=origin to all rows where uuid IS NULL.

    Idempotent: rows that already have a uuid are untouched.
    Returns the count of rows updated.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id FROM nodes WHERE uuid IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return 0
    with conn:
        for (nid,) in rows:
            conn.execute(
                "UPDATE nodes SET uuid=?, origin=? WHERE id=? AND uuid IS NULL",
                (f"{origin}:{nid}", origin, nid)
            )
    return len(rows)


def _row_to_node_dict(row):
    """Convert a DB row (fetched with all federation columns) to an export dict."""
    (nid, content, channel, label, conf_label, status, source_ref, created_at,
     tombstoned, tombstoned_at, revoke_reason,
     redacted, redacted_at, redact_reason,
     quarantined_at, quarantine_reason,
     uuid, origin) = row

    return {
        "uuid": uuid,
        "origin": origin,
        "content": "" if (redacted if redacted is not None else 0) else content,
        "channel": channel,
        "label": label,
        "conf_label": conf_label if conf_label is not None else 0,
        "status": status if status is not None else "live",
        "tombstoned": tombstoned,
        "tombstoned_at": tombstoned_at,
        "revoke_reason": revoke_reason,
        "redacted": redacted if redacted is not None else 0,
        "redacted_at": redacted_at,
        "redact_reason": redact_reason,
        "quarantine_reason": quarantine_reason,
        "quarantined_at": quarantined_at,
        "source_ref": source_ref,
        "created_at": created_at,
    }


_SELECT_COLS = (
    "id, content, channel, label, "
    "COALESCE(conf_label, 0) as conf_label, "
    "COALESCE(status, 'live') as status, "
    "source_ref, created_at, "
    "tombstoned, tombstoned_at, revoke_reason, "
    "COALESCE(redacted, 0) as redacted, redacted_at, redact_reason, "
    "quarantined_at, quarantine_reason, "
    "uuid, origin"
)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_changeset(conn, since=None, origin=None):
    """Build a changeset dict for all (or recently changed) nodes.

    since:  ISO-8601 TEXT cutoff. When given, include only nodes where any of
            created_at / tombstoned_at / redacted_at / quarantined_at >= since.
            When None, export all nodes.
    origin: override the default origin name (default_origin() if not given).

    Returns a dict with keys: format, origin, exported_at, nodes, edges.
    """
    migrate(conn)
    origin = origin or default_origin()
    backfill_uuids(conn, origin)
    now = memsom.now_iso()

    if since is None:
        rows = conn.execute(
            f"SELECT {_SELECT_COLS} FROM nodes ORDER BY id"
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_SELECT_COLS} FROM nodes WHERE "
            "created_at >= ? OR tombstoned_at >= ? OR redacted_at >= ? "
            "OR quarantined_at >= ? ORDER BY id",
            (since, since, since, since)
        ).fetchall()

    node_dicts = [_row_to_node_dict(r) for r in rows]

    # Export ALL redaction records regardless of `since` (priority propagation —
    # redaction events must never be filtered out).
    redaction_rows = conn.execute(
        'SELECT uuid, redacted_at FROM redaction_log ORDER BY uuid'
    ).fetchall()
    redactions = [{'uuid': u, 'redacted_at': at} for (u, at) in redaction_rows]

    # Build uuid -> local id map for edge resolution
    selected_ids = set()
    for r in rows:
        selected_ids.add(r[0])  # r[0] = id

    # Collect edges where child is in selected set
    # Also resolve parent uuid (parent node always exists locally — FK constraint)
    uuid_by_id = {}
    for r in rows:
        nid = r[0]
        uuid = r[16]  # uuid column index in the SELECT
        if uuid is not None:
            uuid_by_id[nid] = uuid

    # For since-filtered exports, we also need parent uuids even if parent not exported
    # Fetch uuids for all local nodes to resolve parent references
    all_uuids = conn.execute("SELECT id, uuid FROM nodes WHERE uuid IS NOT NULL").fetchall()
    for nid, uuid in all_uuids:
        if nid not in uuid_by_id:
            uuid_by_id[nid] = uuid

    edge_rows = []
    if selected_ids:
        qmarks = ",".join("?" * len(selected_ids))
        edge_db_rows = conn.execute(
            f"SELECT child, parent FROM edges WHERE child IN ({qmarks}) ORDER BY child, parent",
            tuple(sorted(selected_ids))
        ).fetchall()
        for child_id, parent_id in edge_db_rows:
            child_uuid = uuid_by_id.get(child_id)
            parent_uuid = uuid_by_id.get(parent_id)
            if child_uuid and parent_uuid:
                edge_rows.append([child_uuid, parent_uuid])

    return {
        "format": "memsom-changeset-v1",
        "origin": origin,
        "exported_at": now,
        "nodes": node_dicts,
        "edges": edge_rows,
        "redactions": redactions,
    }


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------

def write_jsonl(path, changeset):
    """Write changeset to a JSONL file.

    Line 1: header with format/origin/exported_at.
    Then one line per node, one line per edge.
    UTF-8, ensure_ascii=False.
    """
    with open(path, "w", encoding="utf-8") as f:
        header = {
            "type": "header",
            "format": changeset["format"],
            "origin": changeset["origin"],
            "exported_at": changeset["exported_at"],
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for node in changeset.get("nodes", []):
            record = {"type": "node"}
            record.update(node)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for child_uuid, parent_uuid in changeset.get("edges", []):
            f.write(json.dumps(
                {"type": "edge", "child": child_uuid, "parent": parent_uuid},
                ensure_ascii=False
            ) + "\n")
        for r in changeset.get("redactions", []):
            f.write(json.dumps(
                {"type": "redaction", "uuid": r["uuid"], "redacted_at": r.get("redacted_at")},
                ensure_ascii=False
            ) + "\n")


def read_jsonl(path):
    """Reassemble a changeset dict from a JSONL file.

    Raises ValueError on missing or malformed header.
    Returns dict matching export_changeset format.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        raise ValueError("empty JSONL file")

    header = json.loads(lines[0])
    if header.get("type") != "header" or header.get("format") != "memsom-changeset-v1":
        raise ValueError(f"bad or missing header in JSONL: {lines[0]!r}")

    nodes = []
    edges = []
    redactions = []
    for line in lines[1:]:
        # FED-DOS-1: a corrupted/truncated line must skip, not crash the import.
        # Mirror the FED-3 tolerance in import_changeset so both the transport
        # (read_jsonl) and in-memory paths are robust to disk/transport corruption.
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        rec_type = rec.get("type")
        if rec_type == "node":
            d = {k: v for k, v in rec.items() if k != "type"}
            nodes.append(d)
        elif rec_type == "edge":
            child, parent = rec.get("child"), rec.get("parent")
            if child is not None and parent is not None:
                edges.append([child, parent])
        elif rec_type == "redaction":
            if rec.get("uuid"):
                redactions.append({"uuid": rec["uuid"], "redacted_at": rec.get("redacted_at")})

    return {
        "format": header["format"],
        "origin": header["origin"],
        "exported_at": header["exported_at"],
        "nodes": nodes,
        "edges": edges,
        "redactions": redactions,
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_changeset(conn, changeset):
    """Import a changeset into this database.

    Returns stats dict: nodes_new, nodes_updated, edges_new, edges_skipped,
    resurrections_blocked.

    TRUST BOUNDARY (default-deny):
      - If the changeset origin is NOT in trusted_origins, all new nodes are
        clamped to channel='external', label=RANK['external'](0), and
        conf_label >= CONF_FLOOR_UNTRUSTED(1).
      - Tombstone/redact/quarantine state changes are honoured only if the
        changeset header origin matches the node's stored origin (you can only
        kill/redact/quarantine what you own).
      - Edges are only inserted if both endpoints arrived in this changeset OR
        the child's stored origin matches the changeset header origin.
      - After import, recompute_all() and recompute_conf_all() re-floor every
        accepted agent-derived label and conf_label to the correct values.

    Monotonic rules (applied in a single transaction):
      Tombstone: incoming dead + local live -> mark dead IFF owned.
                 incoming live + local dead -> BLOCKED (resurrection prevented).
      Redact:    incoming redacted + local unredacted -> redact locally IFF owned.
                 incoming unredacted + local redacted -> BLOCKED.
      Quarantine: incoming quarantined + local live -> quarantine IFF owned.
                  incoming live + local quarantined -> BLOCKED (promote is a local act).
      Content/label/channel/created_at: NEVER overwritten on existing rows.
    """
    if changeset.get("format") != "memsom-changeset-v1":
        raise ValueError(
            f"unsupported changeset format: {changeset.get('format')!r}"
        )

    migrate(conn)

    # --- Trust determination ---
    header_origin = changeset.get("origin")
    trusted = is_trusted(conn, header_origin)

    # FED-1: track UUIDs ACTUALLY INSERTED as new nodes this import (populated in
    # Pass 1), NOT the UUIDs merely *claimed* in the attacker-supplied nodes array.
    # An untrusted peer can echo a pre-existing local node's uuid into the nodes
    # array; that dict hits the existing-node branch (no insert) but must NOT count
    # as "arrived in this changeset" for edge authorization, or it could be wired as
    # a forged provenance edge endpoint (both_in_changeset bypass).
    inserted_uuids = set()

    # Cascade tombstones to apply AFTER the main transaction commits.
    cascade_tombstones = []  # list of (local_id, reason)
    # FED-2: redactions applied on import must cascade to descendants (mirror the
    # local redact_node default cascade=True and the tombstone-import cascade).
    # Collected here, applied post-transaction once Pass 2 has wired edges.
    cascade_redactions = []  # list of (local_id, reason)

    stats = {
        "nodes_new": 0,
        "nodes_updated": 0,
        "edges_new": 0,
        "edges_skipped": 0,
        "resurrections_blocked": 0,
    }

    with conn:
        # ----- Pass 0: merge redaction RECORDS (priority propagation) -----
        # A redaction event is a first-class record. Merge incoming records from
        # TRUSTED origins (consistent with the default-deny boundary — prevents an
        # untrusted attacker from injecting redaction-DoS), then scrub ANY node whose
        # uuid is known-redacted regardless of the (possibly stale) changeset fields.
        if trusted:
            for rec in changeset.get("redactions", []):
                ru = rec.get("uuid")
                if ru:
                    conn.execute(
                        'INSERT OR IGNORE INTO redaction_log(uuid, redacted_at) VALUES (?,?)',
                        (ru, rec.get("redacted_at"))
                    )
        redacted_at_by_uuid = {
            r[0]: r[1]
            for r in conn.execute("SELECT uuid, redacted_at FROM redaction_log")
        }

        # ----- Pass 1: nodes -----
        for node in changeset.get("nodes", []):
            uuid = node.get("uuid")
            if not uuid:
                continue

            local = conn.execute(
                "SELECT id, tombstoned, tombstoned_at, revoke_reason, "
                "redacted, redacted_at, redact_reason, "
                "COALESCE(status, 'live') as status, quarantined_at, quarantine_reason, "
                "origin "
                "FROM nodes WHERE uuid=?",
                (uuid,)
            ).fetchone()

            if local is None:
                # --- New node: INSERT with clamped values ---
                inc_channel = node.get("channel", "external")
                inc_redacted = 1 if node.get("redacted", 0) else 0
                # F-07 (closeable part): if we ALREADY hold the redaction record for this
                # uuid, force-scrub even though THIS changeset is stale (redacted=0, full
                # content). KNOWN RESIDUAL: a cold machine that never received the record
                # still gets the content — see test_known_limit_cold_machine_stale_changeset.
                record_redacted = 1 if uuid in redacted_at_by_uuid else 0
                forced_redacted = 1 if (inc_redacted or record_redacted) else 0

                if trusted:
                    # Trusted origin: keep channel if valid, but NEVER trust label
                    # for source channels; derived labels will be recomputed below.
                    channel = inc_channel if inc_channel in memsom.RANK else "external"
                    if channel == "agent-derived":
                        # Label from node dict; recompute_all re-floors it in step (e).
                        label = node.get("label", 0)
                    else:
                        # Source channel: derive label from channel, never from node.label
                        label = memsom.RANK[channel]
                    # Trusted: keep conf_label as-is (coerced/clamped 0-3);
                    # recompute_conf_all re-floors derived.
                    conf_label = _coerce_conf(node.get("conf_label", 0), floor=0)
                else:
                    # Untrusted origin: clamp channel and label to external floor.
                    channel = "external"
                    label = memsom.RANK["external"]  # 0
                    # Untrusted: never self-classify as PUBLIC(0) — floor to INTERNAL(1).
                    conf_label = _coerce_conf(node.get("conf_label", 0),
                                              floor=CONF_FLOOR_UNTRUSTED)

                # F-04 / F-07: arriving-redacted node carries NO payload.
                # Also scrub if we already hold the redaction record for this uuid.
                content = "" if forced_redacted else node.get("content", "")

                cur = conn.execute(
                    "INSERT INTO nodes("
                    "  content, channel, label, conf_label, status, source_ref, created_at,"
                    "  tombstoned, tombstoned_at, revoke_reason,"
                    "  redacted, redacted_at, redact_reason,"
                    "  quarantined_at, quarantine_reason,"
                    "  uuid, origin"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        content,
                        channel,
                        label,
                        conf_label,
                        node.get("status", "live"),
                        node.get("source_ref"),
                        node.get("created_at", memsom.now_iso()),
                        node.get("tombstoned", 0),
                        node.get("tombstoned_at"),
                        node.get("revoke_reason"),
                        forced_redacted,
                        node.get("redacted_at") or (redacted_at_by_uuid.get(uuid) if record_redacted else None),
                        node.get("redact_reason") or ("redacted (federated record)" if record_redacted else None),
                        node.get("quarantined_at"),
                        node.get("quarantine_reason"),
                        uuid,
                        node.get("origin"),   # provenance stored; NOT used for authz
                    )
                )
                stats["nodes_new"] += 1
                inserted_uuids.add(uuid)  # FED-1: only freshly-inserted uuids authorize edges
                if forced_redacted:
                    # FED-2: cascade to descendants once edges are wired (post-txn).
                    cascade_redactions.append(
                        (cur.lastrowid,
                         node.get("redact_reason") or "redacted (federated import)")
                    )

            else:
                # --- Existing node: monotonic merge ---
                (local_id, local_tomb, local_tomb_at, local_tomb_reason,
                 local_redacted, local_redacted_at, local_redact_reason,
                 local_status, local_q_at, local_q_reason,
                 local_origin) = local

                # Authorization rule for state changes: only the same origin may
                # tombstone/redact/quarantine what it owns.
                # header_origin is the CHANGESET header (not the node-dict field —
                # that is attacker-controlled and is NOT used for authz).
                owned = (
                    header_origin is not None
                    and header_origin == local_origin
                )

                updated = False

                # (a) Tombstone monotonic rule — F-03 / F-11
                inc_tomb = node.get("tombstoned", 0)
                if inc_tomb and not local_tomb:
                    if owned:
                        # Incoming dead, local live, same origin -> propagate death.
                        conn.execute(
                            "UPDATE nodes SET tombstoned=1, tombstoned_at=?, revoke_reason=?"
                            " WHERE id=?",
                            (node.get("tombstoned_at"), node.get("revoke_reason"), local_id)
                        )
                        updated = True
                        # Queue cascade: revoke_cascade runs AFTER txn closes (F-11).
                        cascade_tombstones.append(
                            (local_id, node.get("revoke_reason") or "cascade from federation import")
                        )
                    # else: not owned -> ignore (F-03 blocked)
                elif not inc_tomb and local_tomb:
                    # Incoming live, local dead -> BLOCK resurrection (always, regardless of origin).
                    stats["resurrections_blocked"] += 1

                # (b) Redact monotonic rule — F-04 / F-07
                inc_redacted = node.get("redacted", 0)
                record_redacted = uuid in redacted_at_by_uuid
                if (inc_redacted or record_redacted) and not local_redacted:
                    if record_redacted or owned:
                        # Incoming redacted (or redaction record held), local unredacted -> scrub.
                        conn.execute(
                            "UPDATE nodes SET content='', redacted=1, redacted_at=?, redact_reason=?"
                            " WHERE id=?",
                            (
                                node.get("redacted_at") or redacted_at_by_uuid.get(uuid),
                                node.get("redact_reason") or "redacted (federated record)",
                                local_id,
                            )
                        )
                        updated = True
                        # FED-2: scrub descendants too (post-txn), matching the
                        # local redact_node(cascade=True) contract.
                        cascade_redactions.append(
                            (local_id,
                             node.get("redact_reason") or "redacted (federated record)")
                        )
                    # else: incoming redacted but not owned and no record -> ignore (F-03 preserved)
                elif not inc_redacted and local_redacted:
                    # Incoming unredacted over local redacted -> BLOCK (F-07 / F-04 old-changeset restore).
                    stats["resurrections_blocked"] += 1

                # (c) Quarantine monotonic rule (status field) — F-12
                inc_status = node.get("status", "live")
                if inc_status == "quarantined" and local_status == "live":
                    if owned:
                        conn.execute(
                            "UPDATE nodes SET status='quarantined', quarantine_reason=?,"
                            " quarantined_at=? WHERE id=?",
                            (node.get("quarantine_reason"), node.get("quarantined_at"), local_id)
                        )
                        updated = True
                    # else: not owned -> ignore (F-12 blocked)
                elif inc_status == "live" and local_status == "quarantined":
                    # Incoming live over local quarantined -> BLOCK.
                    stats["resurrections_blocked"] += 1

                if updated:
                    stats["nodes_updated"] += 1

        # ----- Pass 2: edges (F-02) -----
        # Build uuid -> local id map (all local nodes, including just-inserted).
        uuid_to_id = {
            r[0]: r[1]
            for r in conn.execute("SELECT uuid, id FROM nodes WHERE uuid IS NOT NULL")
        }

        # Build uuid -> stored origin map for edge authorization.
        origin_by_uuid = {
            uuid: origin
            for uuid, origin in conn.execute(
                "SELECT uuid, origin FROM nodes WHERE uuid IS NOT NULL"
            )
        }

        for edge in changeset.get("edges", []):
            # FED-3: a malformed edge (not a [child, parent] pair) must skip, not
            # crash the whole import. The changeset is honor-system single-operator
            # transport, so the realistic source is disk/transport corruption.
            if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                stats["edges_skipped"] += 1
                continue
            child_uuid, parent_uuid = edge
            child_id = uuid_to_id.get(child_uuid)
            parent_id = uuid_to_id.get(parent_uuid)
            if child_id is None or parent_id is None:
                stats["edges_skipped"] += 1
                continue

            # Authorization (FED-1 / F-02): accept the edge only if
            #   (a) BOTH endpoints were freshly INSERTED this import — a trusted peer
            #       shipping a self-contained subgraph (untrusted peers' inserts are
            #       all external-clamped, so external<-external edges are harmless), OR
            #   (b) the child is a node owned by the TRUSTED changeset origin
            #       (re-sync: a peer declaring provenance for its own nodes).
            # Rejected: a pre-existing local node becoming an edge endpoint just
            # because its uuid was echoed in the nodes array (the both_in_changeset
            # bypass), and the None==None match against NULL-origin local nodes.
            both_inserted = (
                child_uuid in inserted_uuids and parent_uuid in inserted_uuids
            )
            child_owned = (
                trusted
                and header_origin is not None
                and origin_by_uuid.get(child_uuid) == header_origin
            )

            if not (both_inserted or child_owned):
                stats["edges_skipped"] += 1
                continue

            # INSERT OR IGNORE: idempotent
            conn.execute(
                "INSERT OR IGNORE INTO edges(child, parent) VALUES (?,?)",
                (child_id, parent_id)
            )
            n = conn.execute("SELECT changes()").fetchone()[0]
            if n:
                stats["edges_new"] += 1
            else:
                stats["edges_skipped"] += 1

    # ----- Post-transaction: cascade tombstones + redactions + recompute -----
    # These run AFTER the main txn commits so nested-context double-commit is avoided.
    for local_id, reason in cascade_tombstones:
        memsom.revoke_cascade(conn, local_id, reason)  # F-11: propagate to all descendants

    # FED-2: cascade each import-applied redaction to its live descendants. Edges
    # are now wired (Pass 2 committed), so cascade_set sees the full subtree.
    # redact_node is idempotent (already-redacted nodes are skipped) and also
    # de-indexes + records redaction events.
    for local_id, reason in cascade_redactions:
        memsom_redact.redact_node(conn, local_id, reason, cascade=True)

    # F-09 / F-02-B: re-floor every accepted agent-derived label to min(parents).
    memsom_recompute.recompute_all(conn)
    # F-10: re-floor every derived conf_label to max(parents).
    memsom_confid.recompute_conf_all(conn)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_register_origin(args):
    conn = memsom.get_connection()
    try:
        ok = register_origin(conn, args.origin, by=args.by, descr=args.descr)
        print(f"registered origin: {args.origin}" if ok else f"failed to register origin: {args.origin}")
    finally:
        conn.close()


def cmd_origins_list(args):
    conn = memsom.get_connection()
    try:
        rows = list_origins(conn)
        if not rows:
            print("(no trusted origins)")
        for origin, descr, added_by, added_at in rows:
            print(f"{origin!r}  descr={descr!r}  by={added_by!r}  at={added_at!r}")
    finally:
        conn.close()


def cmd_export(args):
    conn = memsom.get_connection()
    try:
        changeset = export_changeset(
            conn,
            since=getattr(args, "since", None),
            origin=getattr(args, "origin", None),
        )
        write_jsonl(args.file, changeset)
        print(f"exported {len(changeset['nodes'])} node(s),"
              f" {len(changeset['edges'])} edge(s) -> {args.file}")
    finally:
        conn.close()


def cmd_import(args):
    changeset = read_jsonl(args.file)
    conn = memsom.get_connection()
    try:
        stats = import_changeset(conn, changeset)
        for k, v in stats.items():
            print(f"{k}: {v}")
    finally:
        conn.close()


def register(subparsers):
    p_exp = subparsers.add_parser("export", help="export a changeset to JSONL")
    p_exp.add_argument("file", help="output .jsonl path")
    p_exp.add_argument("--since", default=None,
                       help="ISO-8601 cutoff; only nodes changed since this time")
    p_exp.add_argument("--origin", default=None,
                       help="override the machine origin name")
    p_exp.set_defaults(func=cmd_export)

    p_imp = subparsers.add_parser("import", help="import a changeset from JSONL")
    p_imp.add_argument("file", help="input .jsonl path")
    p_imp.set_defaults(func=cmd_import)

    p_reg = subparsers.add_parser("register-origin", help="trust a federation origin (default-deny)")
    p_reg.add_argument("origin")
    p_reg.add_argument("--by", default="user")
    p_reg.add_argument("--descr", default=None)
    p_reg.set_defaults(func=cmd_register_origin)

    p_ls = subparsers.add_parser("origins-list", help="list trusted origins")
    p_ls.set_defaults(func=cmd_origins_list)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_federation")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
