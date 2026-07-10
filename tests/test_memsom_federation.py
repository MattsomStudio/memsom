#!/usr/bin/env python3
"""Tests for memsom_federation — multi-machine sync with monotonic death/redact.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_federation.py \
    -t <repo> -v
"""

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.federation import federation as memsom_federation
from memsom.integrity import confid as memsom_confid
from memsom.integrity import quarantine as memsom_quarantine
from memsom.retrieval import recompute as memsom_recompute
from memsom.integrity import redact as memsom_redact


HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    """Provides two isolated temp DBs: conn_a and conn_b.

    MEMDAG_DB is set to conn_a's path to satisfy any code that calls
    memsom.db_path() — but tests use the explicit connection objects directly.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.db_a = tmp / "a" / "a.db"
        self.db_b = tmp / "b" / "b.db"
        # Set MEMDAG_DB to A so db_path()-based code has a valid target
        os.environ["MEMDAG_DB"] = str(self.db_a)
        self.conn_a = memsom.get_connection(self.db_a)
        self.conn_b = memsom.get_connection(self.db_b)
        # Apply federation migration to both
        memsom_federation.migrate(self.conn_a)
        memsom_federation.migrate(self.conn_b)
        # Register all legit test origins on both connections so the trust
        # boundary does not clamp channels for legitimate sync tests.
        for o in ("machineA", "machineB", "A", "B", "local", "test-machine",
                  "machine-α"):
            memsom_federation.register_origin(self.conn_a, o, by="test")
            memsom_federation.register_origin(self.conn_b, o, by="test")

    def tearDown(self):
        self.conn_a.close()
        self.conn_b.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, conn, content, channel):
        with conn:
            return memsom.insert_node(conn, content, channel, memsom.RANK[channel])

    def export_file(self, conn, suffix="cs.jsonl", **kw):
        path = str(Path(self.tmp.name) / suffix)
        cs = memsom_federation.export_changeset(conn, **kw)
        memsom_federation.write_jsonl(path, cs)
        return path, cs

    def import_file(self, conn, path):
        cs = memsom_federation.read_jsonl(path)
        return memsom_federation.import_changeset(conn, cs)


# ---------------------------------------------------------------------------
# Test 1 — round-trip preserves graph
# ---------------------------------------------------------------------------

class TestRoundTripPreservesGraph(Base):

    def test_round_trip_preserves_graph(self):
        """Build on A (3 sources + derived + classified + quarantined); round-trip to B."""
        # Build graph on A
        e = self.add(self.conn_a, "endorsed source content text", "endorsed")
        u = self.add(self.conn_a, "user source content text two", "user")
        x = self.add(self.conn_a, "external source content text three", "external")

        # Derive an answer from all three
        q = "How does the system work in general terms?"
        srcs = memsom.live_sources(self.conn_a)
        text, used = memsom.compose(q, srcs)
        if text is None:
            # fallback: derive manually
            text = "derived answer"
            used = [e, u, x]
        d, _ = memsom.derive_node(self.conn_a, text, used)

        # Classify one as secret; recompute so A is canonical before export.
        memsom_confid.classify(self.conn_a, x, "secret")
        memsom_recompute.recompute_all(self.conn_a)
        memsom_confid.recompute_conf_all(self.conn_a)

        # Quarantine a node (uses quarantine_node which already sets quarantined_at)
        memsom_quarantine.quarantine_node(self.conn_a, x, "suspicious")

        # Export A -> import into empty B
        path, cs = self.export_file(self.conn_a, "round_trip.jsonl", origin="machineA")
        stats = self.import_file(self.conn_b, path)

        # Node count must match
        count_a = self.conn_a.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        count_b = self.conn_b.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(count_a, count_b,
                         f"A has {count_a} nodes but B has {count_b}")
        self.assertEqual(stats["nodes_new"], count_a)

        # Per-uuid equality: content, channel, label, conf_label, status
        nodes_a = {
            r[0]: r for r in self.conn_a.execute(
                "SELECT uuid, content, channel, label,"
                " COALESCE(conf_label,0), COALESCE(status,'live') FROM nodes"
            ).fetchall()
        }
        nodes_b = {
            r[0]: r for r in self.conn_b.execute(
                "SELECT uuid, content, channel, label,"
                " COALESCE(conf_label,0), COALESCE(status,'live') FROM nodes"
            ).fetchall()
        }
        self.assertEqual(set(nodes_a.keys()), set(nodes_b.keys()), "uuid sets differ")
        for uuid in nodes_a:
            self.assertEqual(nodes_a[uuid], nodes_b[uuid],
                             f"mismatch for uuid={uuid}: A={nodes_a[uuid]} B={nodes_b[uuid]}")

        # Edge count must match
        edges_a = self.conn_a.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        edges_b = self.conn_b.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(edges_a, edges_b)

        # Edges on B connect the same uuids as on A
        def edge_uuids(conn):
            return set(
                conn.execute(
                    "SELECT c.uuid, p.uuid FROM edges e"
                    " JOIN nodes c ON c.id=e.child"
                    " JOIN nodes p ON p.id=e.parent"
                ).fetchall()
            )

        self.assertEqual(edge_uuids(self.conn_a), edge_uuids(self.conn_b))


# ---------------------------------------------------------------------------
# Test 2 — tombstone propagates, no resurrection
# ---------------------------------------------------------------------------

class TestTombstoneNoPropagation(Base):

    def test_tombstone_propagates_no_resurrection(self):
        """Revoke on A propagates to B; stale live copy does NOT resurrect."""
        # Build a node on A
        a = self.add(self.conn_a, "source fact content text line.", "user")
        b, _ = memsom.derive_node(self.conn_a, "derived answer from source.", [a])

        # Sync A -> B (both alive)
        path_pre, _ = self.export_file(self.conn_a, "pre_revoke.jsonl", origin="machineA")
        stats_pre = self.import_file(self.conn_b, path_pre)
        self.assertEqual(stats_pre["nodes_new"], 2)

        # Capture stale changeset of B while node b is still live
        path_stale, _ = self.export_file(self.conn_b, "stale_b.jsonl", origin="machineB")

        # Revoke node b on A
        memsom.revoke_cascade(self.conn_a, b, "bad answer")

        # Export A (post-revoke) -> import into B
        path_post, _ = self.export_file(self.conn_a, "post_revoke.jsonl", origin="machineA")
        stats_post = self.import_file(self.conn_b, path_post)

        # b must now be dead on B with A's reason
        uuid_b = self.conn_a.execute("SELECT uuid FROM nodes WHERE id=?", (b,)).fetchone()[0]
        b_on_b = self.conn_b.execute(
            "SELECT tombstoned, revoke_reason FROM nodes WHERE uuid=?", (uuid_b,)
        ).fetchone()
        self.assertEqual(b_on_b[0], 1, "node b should be tombstoned on B")
        self.assertEqual(b_on_b[1], "bad answer")

        # Now import the STALE pre-revoke changeset (both from B back into B and into A)
        # This must NOT resurrect the dead node
        stale_cs = memsom_federation.read_jsonl(path_stale)

        stats_stale_b = memsom_federation.import_changeset(self.conn_b, stale_cs)
        self.assertGreaterEqual(stats_stale_b["resurrections_blocked"], 1,
                                "stale live import into B should block resurrection")
        # Node must still be dead on B
        b_on_b_after = self.conn_b.execute(
            "SELECT tombstoned FROM nodes WHERE uuid=?", (uuid_b,)
        ).fetchone()
        self.assertEqual(b_on_b_after[0], 1, "resurrection must be blocked on B")

        stats_stale_a = memsom_federation.import_changeset(self.conn_a, stale_cs)
        self.assertGreaterEqual(stats_stale_a["resurrections_blocked"], 1,
                                "stale live import into A should block resurrection")
        # Node must still be dead on A
        b_on_a_after = self.conn_a.execute(
            "SELECT tombstoned FROM nodes WHERE id=?", (b,)
        ).fetchone()
        self.assertEqual(b_on_a_after[0], 1, "resurrection must be blocked on A")


# ---------------------------------------------------------------------------
# Test 3 — redaction propagates
# ---------------------------------------------------------------------------

class TestRedactionPropagates(Base):

    def test_redaction_propagates(self):
        """Redact on A, export/import, B has content='' redacted=1.
        Stale unredacted copy does not restore content.
        """
        n = self.add(self.conn_a, "sensitive information content text here.", "user")

        # Sync to B before redaction
        path_pre, _ = self.export_file(self.conn_a, "pre_redact.jsonl", origin="machineA")
        self.import_file(self.conn_b, path_pre)

        # Capture stale unredacted copy from B
        path_stale, _ = self.export_file(self.conn_b, "stale_unredacted.jsonl", origin="machineB")

        # Redact on A — use direct SQL to set all fields including our extended columns
        with self.conn_a:
            self.conn_a.execute(
                "UPDATE nodes SET content='', redacted=1, redacted_at=?, redact_reason=?"
                " WHERE id=?",
                (memsom.now_iso(), "gdpr removal", n)
            )

        # Export A (post-redact) -> import into B
        path_post, _ = self.export_file(self.conn_a, "post_redact.jsonl", origin="machineA")
        stats = self.import_file(self.conn_b, path_post)

        uuid_n = self.conn_a.execute("SELECT uuid FROM nodes WHERE id=?", (n,)).fetchone()[0]
        n_on_b = self.conn_b.execute(
            "SELECT content, redacted FROM nodes WHERE uuid=?", (uuid_n,)
        ).fetchone()
        self.assertEqual(n_on_b[0], "", "content must be empty after redact propagation")
        self.assertEqual(n_on_b[1], 1, "redacted flag must be 1 on B")

        # Now import the stale unredacted changeset back into B — must not restore content
        stale_cs = memsom_federation.read_jsonl(path_stale)
        stats_stale = memsom_federation.import_changeset(self.conn_b, stale_cs)
        self.assertGreaterEqual(stats_stale["resurrections_blocked"], 1,
                                "stale unredacted import should be blocked")

        n_on_b_after = self.conn_b.execute(
            "SELECT content, redacted FROM nodes WHERE uuid=?", (uuid_n,)
        ).fetchone()
        self.assertEqual(n_on_b_after[0], "", "content must still be empty after stale import")
        self.assertEqual(n_on_b_after[1], 1, "redacted must still be 1 after stale import")


# ---------------------------------------------------------------------------
# Test 4 — re-import is idempotent
# ---------------------------------------------------------------------------

class TestReimportIdempotent(Base):

    def test_reimport_idempotent(self):
        """Import the same file twice: second import nodes_new==0, edges_new==0."""
        e = self.add(self.conn_a, "source content for idempotent test.", "endorsed")
        u = self.add(self.conn_a, "user source content for idempotent test.", "user")
        srcs = memsom.live_sources(self.conn_a)
        text, used = memsom.compose("What is the system?", srcs)
        if text is None:
            text = "derived"
            used = [e, u]
        memsom.derive_node(self.conn_a, text, used)

        path, _ = self.export_file(self.conn_a, "idem.jsonl", origin="A")

        stats1 = self.import_file(self.conn_b, path)
        count_n = self.conn_b.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        count_e = self.conn_b.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        stats2 = self.import_file(self.conn_b, path)
        self.assertEqual(stats2["nodes_new"], 0,
                         f"second import should add 0 nodes, got {stats2['nodes_new']}")
        self.assertEqual(stats2["edges_new"], 0,
                         f"second import should add 0 edges, got {stats2['edges_new']}")

        count_n2 = self.conn_b.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        count_e2 = self.conn_b.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(count_n, count_n2, "node count changed on second import")
        self.assertEqual(count_e, count_e2, "edge count changed on second import")


# ---------------------------------------------------------------------------
# Test 5 — quarantine propagates but promote is local
# ---------------------------------------------------------------------------

class TestQuarantinePropagatesPromoteIsLocal(Base):

    def test_quarantine_propagates_but_promote_is_local(self):
        """Quarantine on A propagates to B; a later 'live' import does not un-quarantine B."""
        n = self.add(self.conn_a, "agent derived source text.", "user")
        d, _ = memsom.derive_node(self.conn_a, "derived from user source.", [n])

        # Sync A -> B (both live)
        path_pre, _ = self.export_file(self.conn_a, "pre_quarantine.jsonl", origin="A")
        self.import_file(self.conn_b, path_pre)

        # Quarantine d on A (quarantine_node sets quarantined_at automatically)
        memsom_quarantine.quarantine_node(self.conn_a, d, "suspicious content found")

        # Export A (post-quarantine) -> import into B
        path_quar, _ = self.export_file(self.conn_a, "quarantined.jsonl", origin="A")
        stats_quar = self.import_file(self.conn_b, path_quar)

        uuid_d = self.conn_a.execute("SELECT uuid FROM nodes WHERE id=?", (d,)).fetchone()[0]
        d_on_b = self.conn_b.execute(
            "SELECT COALESCE(status,'live') FROM nodes WHERE uuid=?", (uuid_d,)
        ).fetchone()
        self.assertEqual(d_on_b[0], "quarantined",
                         "quarantine must propagate to B")

        # Now export B with d still quarantined, but craft a 'live' version and re-import
        # Simulate: export B, manually patch d to 'live', import back -> must be blocked
        path_b, cs_b = self.export_file(self.conn_b, "b_snapshot.jsonl", origin="B")
        # Patch the changeset in memory to set d's status to 'live'
        for node in cs_b["nodes"]:
            if node.get("uuid") == uuid_d:
                node["status"] = "live"
        stats_live_over_quarantine = memsom_federation.import_changeset(self.conn_b, cs_b)
        self.assertGreaterEqual(stats_live_over_quarantine["resurrections_blocked"], 1,
                                "'live' import over quarantined should be blocked")

        d_on_b_after = self.conn_b.execute(
            "SELECT COALESCE(status,'live') FROM nodes WHERE uuid=?", (uuid_d,)
        ).fetchone()
        self.assertEqual(d_on_b_after[0], "quarantined",
                         "B's quarantine must survive the live-status import")


# ---------------------------------------------------------------------------
# Test 6 — backfill idempotent and uuid format
# ---------------------------------------------------------------------------

class TestBackfillIdempotent(Base):

    def test_backfill_idempotent_and_uuid_format(self):
        """backfill_uuids sets uuid=origin:id; second call returns 0."""
        n1 = self.add(self.conn_a, "first node", "user")
        n2 = self.add(self.conn_a, "second node", "endorsed")

        origin = "test-machine"
        count1 = memsom_federation.backfill_uuids(self.conn_a, origin)
        self.assertEqual(count1, 2, f"expected 2 backfilled, got {count1}")

        # Check uuid format
        for nid in (n1, n2):
            uuid = self.conn_a.execute(
                "SELECT uuid FROM nodes WHERE id=?", (nid,)
            ).fetchone()[0]
            self.assertEqual(uuid, f"{origin}:{nid}",
                             f"uuid format wrong for node {nid}: {uuid!r}")

        # Second call: idempotent
        count2 = memsom_federation.backfill_uuids(self.conn_a, origin)
        self.assertEqual(count2, 0, f"second backfill should return 0, got {count2}")

        # UUIDs unchanged
        for nid in (n1, n2):
            uuid = self.conn_a.execute(
                "SELECT uuid FROM nodes WHERE id=?", (nid,)
            ).fetchone()[0]
            self.assertEqual(uuid, f"{origin}:{nid}")


# ---------------------------------------------------------------------------
# Test 7 — since filter
# ---------------------------------------------------------------------------

class TestSinceFilter(Base):

    def test_since_filter(self):
        """export with since later than everything -> zero nodes;
        revoke one node, export since=just-before-revoke -> includes that node."""
        n = self.add(self.conn_a, "a node for since filter test text.", "user")
        d, _ = memsom.derive_node(self.conn_a, "derived for since filter.", [n])

        # Export with a far-future since -> 0 nodes
        future = "2099-01-01T00:00:00+00:00"
        cs_empty = memsom_federation.export_changeset(
            self.conn_a, since=future, origin="A"
        )
        self.assertEqual(len(cs_empty["nodes"]), 0,
                         f"since=future should yield 0 nodes, got {len(cs_empty['nodes'])}")

        # Record the time just before revoking
        just_before = memsom.now_iso()

        # Revoke d -> sets tombstoned_at
        memsom.revoke_cascade(self.conn_a, d, "test revoke for since filter")

        # Export since=just_before -> should include at minimum d (tombstoned_at >= just_before)
        cs_since = memsom_federation.export_changeset(
            self.conn_a, since=just_before, origin="A"
        )
        uuids_in_cs = {node["uuid"] for node in cs_since["nodes"]}

        uuid_d = self.conn_a.execute("SELECT uuid FROM nodes WHERE id=?", (d,)).fetchone()[0]
        self.assertIn(uuid_d, uuids_in_cs,
                      "recently-tombstoned node should appear in since-filtered export")

        # The tombstoned node should have tombstoned=1 in the changeset
        d_in_cs = next(nd for nd in cs_since["nodes"] if nd["uuid"] == uuid_d)
        self.assertEqual(d_in_cs["tombstoned"], 1)


# ---------------------------------------------------------------------------
# Test 8 — write_jsonl / read_jsonl round-trip
# ---------------------------------------------------------------------------

class TestWriteReadJsonlRoundtrip(Base):

    def test_write_read_jsonl_roundtrip(self):
        """dict -> file -> dict equality on nodes/edges/origin/format."""
        n = self.add(self.conn_a, "unicode test: em-dash — arrow → check ✓", "endorsed")
        d, _ = memsom.derive_node(self.conn_a, "derived from endorsed.", [n])

        path, cs_orig = self.export_file(self.conn_a, "jsonl_rt.jsonl", origin="machine-α")

        cs_loaded = memsom_federation.read_jsonl(path)

        self.assertEqual(cs_orig["format"], cs_loaded["format"])
        self.assertEqual(cs_orig["origin"], cs_loaded["origin"])
        self.assertEqual(cs_orig["exported_at"], cs_loaded["exported_at"])
        self.assertEqual(len(cs_orig["nodes"]), len(cs_loaded["nodes"]))
        self.assertEqual(len(cs_orig["edges"]), len(cs_loaded["edges"]))

        # Per-node equality
        orig_by_uuid = {nd["uuid"]: nd for nd in cs_orig["nodes"]}
        loaded_by_uuid = {nd["uuid"]: nd for nd in cs_loaded["nodes"]}
        self.assertEqual(set(orig_by_uuid.keys()), set(loaded_by_uuid.keys()))
        for uuid in orig_by_uuid:
            self.assertEqual(orig_by_uuid[uuid], loaded_by_uuid[uuid],
                             f"node mismatch for uuid={uuid}")

        # Edge sets equal
        self.assertEqual(
            sorted(map(tuple, cs_orig["edges"])),
            sorted(map(tuple, cs_loaded["edges"]))
        )

        # Bad header raises ValueError
        bad_path = str(Path(self.tmp.name) / "bad.jsonl")
        with open(bad_path, "w") as f:
            f.write(json.dumps({"type": "node", "uuid": "x"}) + "\n")
        with self.assertRaises(ValueError):
            memsom_federation.read_jsonl(bad_path)


# ---------------------------------------------------------------------------
# Bonus: validate format key check in import_changeset
# ---------------------------------------------------------------------------

class TestImportValidation(Base):

    def test_import_rejects_bad_format(self):
        bad_cs = {"format": "unknown-v99", "nodes": [], "edges": []}
        with self.assertRaises(ValueError):
            memsom_federation.import_changeset(self.conn_b, bad_cs)


# ---------------------------------------------------------------------------
# Test — FIX D: redaction-record propagation (F-07 closeable part + honest residual)
# ---------------------------------------------------------------------------

class TestRedactionRecordPropagation(Base):
    """Regression: mirrors poc5 Test 2 / bypass04 — redaction-event propagation."""

    def test_redaction_record_scrubs_stale_changeset_when_record_held(self):
        """Warm-machine case: B holds the redaction record BEFORE a stale changeset arrives.

        The redaction EVENT reaches B first (record-only changeset), then a stale
        pre-redaction changeset with full content follows. Because B already holds the
        record, the content MUST be scrubbed — FIX D closeable part.
        """
        SECRET = "FederationSecret warm-machine-key-9f3a never travel"
        sid = self.add(self.conn_a, SECRET, "user")
        memsom_federation.backfill_uuids(self.conn_a, "machineA")
        uuid = self.conn_a.execute(
            "SELECT uuid FROM nodes WHERE id=?", (sid,)
        ).fetchone()[0]

        # Export PRE-redaction (full content captured in stale changeset).
        stale = memsom_federation.export_changeset(self.conn_a, origin="machineA")

        # Redact on A — this also writes the redaction_log entry.
        memsom_redact.redact_node(self.conn_a, sid, "redact-warm", cascade=True)

        # The redaction EVENT reaches B first as a minimal record-only changeset.
        # (No node payload — B has never seen this UUID yet.)
        redaction_cs = {
            "format": "memsom-changeset-v1",
            "origin": "machineA",
            "exported_at": "2026-01-01T00:00:00+00:00",
            "nodes": [],
            "edges": [],
            "redactions": [{"uuid": uuid, "redacted_at": "2026-01-01T00:00:00+00:00"}],
        }
        memsom_federation.import_changeset(self.conn_b, redaction_cs)

        # THEN the stale pre-redaction changeset arrives with full content.
        memsom_federation.import_changeset(self.conn_b, stale)

        row = self.conn_b.execute(
            "SELECT content, redacted FROM nodes WHERE uuid=?", (uuid,)
        ).fetchone()
        self.assertIsNotNone(row, "node must be inserted on B")
        self.assertEqual(row[0], "", "FIXED: content must be scrubbed by held record")
        self.assertEqual(row[1], 1, "FIXED: redacted flag must be 1")

    def test_known_limit_cold_machine_stale_changeset(self):
        """HONEST RESIDUAL (Guarantee #6): a cold machine that NEVER received the
        redaction record gets the content from a stale changeset.

        This is the deletion-vs-immutability limit (Git's secret-in-history problem).
        Passing test encodes the boundary so the suite states it honestly.
        Mitigations: (1) deliver redaction events first; (2) never distribute stale
        changesets; (3) do not federate a source you may need to hard-delete.
        """
        SECRET = "FederationSecret cold-machine-key-77 never travel"
        sid = self.add(self.conn_a, SECRET, "user")
        memsom_federation.backfill_uuids(self.conn_a, "machineA")
        uuid = self.conn_a.execute(
            "SELECT uuid FROM nodes WHERE id=?", (sid,)
        ).fetchone()[0]

        # Export PRE-redaction — redactions=[] at this point (no events yet).
        stale = memsom_federation.export_changeset(self.conn_a, origin="machineA")

        # Redact on A AFTER the stale export.
        memsom_redact.redact_node(self.conn_a, sid, "redact-cold", cascade=True)

        # B never receives the redaction record; only the stale changeset arrives.
        memsom_federation.import_changeset(self.conn_b, stale)

        row = self.conn_b.execute(
            "SELECT content, redacted FROM nodes WHERE uuid=?", (uuid,)
        ).fetchone()
        self.assertIsNotNone(row, "node must be inserted on B")
        # KNOWN RESIDUAL: content arrives because B never held the redaction record.
        self.assertIn(
            "cold-machine-key",
            row[0] or "",
            "RESIDUAL: content arrives on cold machine — deletion-vs-immutability limit",
        )
        self.assertEqual(row[1], 0, "RESIDUAL: redacted=0 on cold machine without record")


if __name__ == "__main__":
    unittest.main()
