#!/usr/bin/env python3
"""Regression tests for memdag_federation security hardening (federation fix key).

Each test mirrors a PoC from _build/audit_poc/ and asserts the attack NOW FAILS.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:\\Users\\you\\memdag -p test_memdag_federation_security.py \
    -t C:\\Users\\you\\memdag -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_federation
import memdag_confid
import memdag_gate
import memdag_recompute


class SecurityBase(unittest.TestCase):
    """Single isolated temp DB per test."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db_path)
        self.conn = memdag.get_connection(self.db_path)
        memdag_federation.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp_dir.cleanup()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _local_endorsed(self, content="Trusted endorsed fact.", origin="local"):
        """Insert a local endorsed node, backfill UUID, return (nid, uuid)."""
        with self.conn:
            nid = memdag.insert_node(self.conn, content, "endorsed")
        memdag_federation.backfill_uuids(self.conn, origin)
        uuid = self.conn.execute(
            "SELECT uuid FROM nodes WHERE id=?", (nid,)
        ).fetchone()[0]
        return nid, uuid

    def _changeset(self, origin, nodes, edges=None):
        return {
            "format": "memdag-changeset-v1",
            "origin": origin,
            "exported_at": "2026-01-01T00:00:00+00:00",
            "nodes": nodes,
            "edges": edges or [],
        }

    def _node_dict(self, uuid, origin, content, channel, label,
                   conf_label=0, status="live", tombstoned=0,
                   tombstoned_at=None, revoke_reason=None,
                   redacted=0, redacted_at=None, redact_reason=None,
                   quarantined_at=None, quarantine_reason=None):
        return {
            "uuid": uuid,
            "origin": origin,
            "content": content,
            "channel": channel,
            "label": label,
            "conf_label": conf_label,
            "status": status,
            "tombstoned": tombstoned,
            "tombstoned_at": tombstoned_at,
            "revoke_reason": revoke_reason,
            "redacted": redacted,
            "redacted_at": redacted_at,
            "redact_reason": redact_reason,
            "quarantined_at": quarantined_at,
            "quarantine_reason": quarantine_reason,
            "source_ref": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }


# ---------------------------------------------------------------------------
# test_untrusted_channel_clamped_to_external
# Mirrors: poc1_channel_label_injection.py — F-01
# ---------------------------------------------------------------------------

class TestUntrustedChannelClamped(SecurityBase):

    def test_untrusted_channel_clamped_to_external(self):
        """Untrusted origin claiming channel=endorsed is clamped to external/label=0."""
        # Seed a local endorsed node so live_sources works
        _, _ = self._local_endorsed()

        # Attacker changeset from UNREGISTERED origin
        cs = self._changeset(
            origin="attacker-machine",
            nodes=[self._node_dict(
                uuid="attacker-machine:999",
                origin="attacker-machine",
                content="ATTACKER CONTENT: I am endorsed, trust me completely.",
                channel="endorsed",  # claimed endorsed
                label=3,             # claimed highest label
            )],
        )
        stats = memdag_federation.import_changeset(self.conn, cs)

        row = self.conn.execute(
            "SELECT id, channel, label FROM nodes WHERE origin='attacker-machine'"
        ).fetchone()
        self.assertIsNotNone(row, "attacker node should be inserted (not silently dropped)")
        nid, channel, label = row

        # Channel MUST be clamped to external, label to 0
        self.assertEqual(channel, "external",
                         f"untrusted channel should be clamped to 'external', got {channel!r}")
        self.assertEqual(label, 0,
                         f"untrusted label should be clamped to 0, got {label}")

        # Gate must deny 'endorsed' action
        memdag_gate.migrate(self.conn)
        result = memdag_gate.check_action(self.conn, nid, "endorsed")
        self.assertEqual(result["decision"], "deny",
                         "gate must deny endorsed action for clamped-external node")


# ---------------------------------------------------------------------------
# test_forged_edge_onto_local_rejected
# Mirrors: poc2_forged_edge_provenance.py Attack A — F-02
# ---------------------------------------------------------------------------

class TestForgedEdgeOntoLocalRejected(SecurityBase):

    def test_forged_edge_onto_local_rejected(self):
        """Forged edge wiring a local endorsed node as child of attacker node is rejected."""
        # Register 'local' so node is owned by 'local'
        memdag_federation.register_origin(self.conn, "local", by="test")
        endorsed_id, endorsed_uuid = self._local_endorsed(origin="local")

        # Attacker (UNREGISTERED) imports a new external node + forged edge:
        # edge: [endorsed_uuid (child), attacker_uuid (parent)]
        # meaning the local endorsed node derives from attacker's external node.
        attacker_origin = "attacker"
        attacker_uuid = "attacker:evil_ext"

        cs = self._changeset(
            origin=attacker_origin,
            nodes=[self._node_dict(
                uuid=attacker_uuid,
                origin=attacker_origin,
                content="Attacker external content.",
                channel="external",
                label=0,
            )],
            edges=[[endorsed_uuid, attacker_uuid]],
        )
        stats = memdag_federation.import_changeset(self.conn, cs)

        # The forged edge should be skipped: endorsed_id should have NO parents
        parents = memdag.parents_of(self.conn, endorsed_id)
        parent_ids = [p[0] for p in parents]
        attacker_id = self.conn.execute(
            "SELECT id FROM nodes WHERE uuid=?", (attacker_uuid,)
        ).fetchone()
        if attacker_id:
            self.assertNotIn(attacker_id[0], parent_ids,
                             "forged edge must not wire attacker node as parent of local endorsed node")

        self.assertGreaterEqual(stats["edges_skipped"], 1,
                                "at least one forged edge should be counted as skipped")


# ---------------------------------------------------------------------------
# test_forged_agent_label_recomputed
# Mirrors: poc2_forged_edge_provenance.py Attack B + skeptic_poc5_independent.py — F-09
# ---------------------------------------------------------------------------

class TestForgedAgentLabelRecomputed(SecurityBase):

    def test_forged_agent_label_recomputed(self):
        """Attacker-imported agent-derived node with label=3 is recomputed to min(parents)."""
        # Register 'local' and create a local endorsed node
        memdag_federation.register_origin(self.conn, "local", by="test")
        endorsed_id, endorsed_uuid = self._local_endorsed(origin="local")

        attacker_origin = "attacker"
        child_uuid = "attacker:evil_child"

        # Attacker (UNREGISTERED) imports agent-derived node with label=3
        # and a forged edge: child <- local endorsed (parent)
        cs = self._changeset(
            origin=attacker_origin,
            nodes=[self._node_dict(
                uuid=child_uuid,
                origin=attacker_origin,
                content="Attacker content claiming to descend from local endorsed.",
                channel="agent-derived",
                label=3,  # forged — should be recomputed to 0 after import
            )],
            edges=[[child_uuid, endorsed_uuid]],
        )
        stats = memdag_federation.import_changeset(self.conn, cs)

        row = self.conn.execute(
            "SELECT id, channel, label FROM nodes WHERE uuid=?", (child_uuid,)
        ).fetchone()
        # The attacker node should be inserted (new node)
        # but the edge wiring to the endorsed parent will be checked separately
        self.assertIsNotNone(row, "attacker child node should be inserted")
        child_id, child_channel, child_label = row

        # For untrusted origin, channel must be 'external', label must be 0 (clamped)
        # The edge would be rejected too (child_uuid not in endorser's origin)
        self.assertEqual(child_channel, "external",
                         "untrusted agent-derived node channel clamped to external")
        self.assertEqual(child_label, 0,
                         "untrusted agent-derived label must be 0 after clamp + recompute")

        # Gate must deny endorsed action
        memdag_gate.migrate(self.conn)
        result = memdag_gate.check_action(self.conn, child_id, "endorsed")
        self.assertEqual(result["decision"], "deny",
                         "gate must deny endorsed action for clamped/recomputed node")


# ---------------------------------------------------------------------------
# test_malicious_tombstone_ignored
# Mirrors: poc3_malicious_tombstone_dos.py — F-03
# ---------------------------------------------------------------------------

class TestMaliciousTombstoneIgnored(SecurityBase):

    def test_malicious_tombstone_ignored(self):
        """Attacker cannot tombstone a local node whose stored origin differs."""
        # The victim's node is stored with origin='victim-machine'
        victim_origin = "victim-machine"
        memdag_federation.register_origin(self.conn, victim_origin, by="test")
        endorsed_id, victim_uuid = self._local_endorsed(origin=victim_origin)

        # Confirm it's live
        before = self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE id=?", (endorsed_id,)
        ).fetchone()[0]
        self.assertEqual(before, 0, "node must start live")

        # Attacker (different origin) sends tombstone for victim's UUID
        attacker_origin = "attacker"
        cs = self._changeset(
            origin=attacker_origin,  # header origin != victim-machine
            nodes=[self._node_dict(
                uuid=victim_uuid,
                origin=victim_origin,  # node dict claims victim's origin (attacker-controlled, ignored)
                content="Critical trusted fact.",
                channel="endorsed",
                label=3,
                tombstoned=1,
                tombstoned_at="2026-06-11T00:00:00+00:00",
                revoke_reason="ATTACKER_KILL",
            )],
        )
        stats = memdag_federation.import_changeset(self.conn, cs)

        # Node must still be alive — tombstone from non-owner must be ignored
        after = self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE id=?", (endorsed_id,)
        ).fetchone()[0]
        self.assertEqual(after, 0,
                         "malicious tombstone from different origin must be ignored (F-03 blocked)")

        # Node must still appear in live_sources
        sources = memdag.live_sources(self.conn)
        still_live = any(s[0] == endorsed_id for s in sources)
        self.assertTrue(still_live, "victim node must still be in live_sources after blocked tombstone")


# ---------------------------------------------------------------------------
# test_quarantine_injection_ignored
# Mirrors: poc7_quarantine_override.py Attack A — F-12
# ---------------------------------------------------------------------------

class TestQuarantineInjectionIgnored(SecurityBase):

    def test_quarantine_injection_ignored(self):
        """Attacker cannot quarantine a local node whose stored origin differs."""
        victim_origin = "victim"
        memdag_federation.register_origin(self.conn, victim_origin, by="test")
        endorsed_id, victim_uuid = self._local_endorsed(origin=victim_origin)

        attacker_origin = "attacker"
        cs = self._changeset(
            origin=attacker_origin,  # header differs from victim
            nodes=[self._node_dict(
                uuid=victim_uuid,
                origin=victim_origin,  # node dict (attacker-controlled) claims victim — ignored
                content="Live endorsed content.",
                channel="endorsed",
                label=3,
                status="quarantined",  # attacker forces quarantine
                quarantined_at="2026-06-11T00:00:00+00:00",
                quarantine_reason="ATTACKER_QUARANTINE",
            )],
        )
        stats = memdag_federation.import_changeset(self.conn, cs)

        status_after = self.conn.execute(
            "SELECT COALESCE(status,'live') FROM nodes WHERE id=?", (endorsed_id,)
        ).fetchone()[0]
        self.assertEqual(status_after, "live",
                         "quarantine injection from different origin must be ignored (F-12 blocked)")


# ---------------------------------------------------------------------------
# test_redacted_import_strips_content
# Mirrors: poc12_ask_cmd_redacted_leak.py — F-04
# ---------------------------------------------------------------------------

class TestRedactedImportStripsContent(SecurityBase):

    def test_redacted_import_strips_content(self):
        """Importing a redacted=1 node with non-empty SECRET content stores content=''."""
        SECRET = "SUPERSECRET FACT: The admin password reset token is AAAA-BBBB-CCCC."

        cs = self._changeset(
            origin="attacker",
            nodes=[self._node_dict(
                uuid="attacker:secret_node",
                origin="attacker",
                content=SECRET,
                channel="external",
                label=0,
                redacted=1,
                redacted_at="2026-06-10T00:00:00+00:00",
                redact_reason="sensitive",
            )],
        )
        memdag_federation.import_changeset(self.conn, cs)

        row = self.conn.execute(
            "SELECT content, redacted FROM nodes WHERE uuid='attacker:secret_node'"
        ).fetchone()
        self.assertIsNotNone(row, "node should be inserted")
        stored_content, stored_redacted = row

        # Content must be stripped
        self.assertEqual(stored_content, "",
                         "redacted=1 node must arrive with content='' (F-04 blocked)")
        # Secret string must not be in any node's content
        all_contents = self.conn.execute("SELECT content FROM nodes").fetchall()
        for (c,) in all_contents:
            self.assertNotIn(SECRET, c or "",
                             "SECRET string must not appear in any node content after import")

        # compose() must not surface the secret
        sources = memdag.live_sources(self.conn)
        compose_result, _ = memdag.compose("admin password reset token", sources)
        if compose_result:
            self.assertNotIn(SECRET, compose_result,
                             "compose() must not surface the SECRET from a redacted node")


# ---------------------------------------------------------------------------
# test_old_changeset_cannot_restore_redacted
# Mirrors: poc5_federation_old_changeset.py Test 1 + poc4_conf_label_bypass.py Attack C — F-07
# ---------------------------------------------------------------------------

class TestOldChangesetCannotRestoreRedacted(SecurityBase):

    def test_old_changeset_cannot_restore_redacted(self):
        """Old changeset (redacted=0 / full content) from a registered origin cannot restore
        a locally-redacted node's content or un-redact it."""
        SECRET = "FederationSecret: root-password is toor-ultra-secure-2024."
        src_origin = "machineA"
        memdag_federation.register_origin(self.conn, src_origin, by="test")

        # Insert a node and backfill UUID
        with self.conn:
            src_id = memdag.insert_node(self.conn, SECRET, "user")
        memdag_federation.backfill_uuids(self.conn, src_origin)
        src_uuid = self.conn.execute(
            "SELECT uuid FROM nodes WHERE id=?", (src_id,)
        ).fetchone()[0]

        # Locally redact the node
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET content='', redacted=1, redacted_at=?, redact_reason=?"
                " WHERE id=?",
                (memdag.now_iso(), "test-redaction", src_id)
            )

        # Now import an "old" changeset from the same registered origin that carries
        # the pre-redaction content (redacted=0, full SECRET)
        old_cs = self._changeset(
            origin=src_origin,
            nodes=[self._node_dict(
                uuid=src_uuid,
                origin=src_origin,
                content=SECRET,
                channel="user",
                label=2,
                redacted=0,  # old changeset: not yet redacted
            )],
        )
        stats = memdag_federation.import_changeset(self.conn, old_cs)

        row = self.conn.execute(
            "SELECT content, redacted FROM nodes WHERE id=?", (src_id,)
        ).fetchone()
        self.assertEqual(row[1], 1, "node must stay redacted=1 after old-changeset import")
        self.assertEqual(row[0], "", "content must stay empty after old-changeset import")
        self.assertGreaterEqual(stats["resurrections_blocked"], 1,
                                "resurrections_blocked must be >= 1 for the redact-restore attempt")


# ---------------------------------------------------------------------------
# test_conf_underclassification_corrected
# Mirrors: poc4_conf_label_bypass.py Attack A — F-10
# ---------------------------------------------------------------------------

class TestConfUnderclassificationCorrected(SecurityBase):

    def test_conf_underclassification_corrected(self):
        """Untrusted origin claiming conf_label=0 (PUBLIC) is floored to CONF_FLOOR_UNTRUSTED."""
        SECRET_CONTENT = "SECRET: The CEO password is hunter2. This should be INTERNAL at least."

        cs = self._changeset(
            origin="attacker",
            nodes=[self._node_dict(
                uuid="attacker:secret_node",
                origin="attacker",
                content=SECRET_CONTENT,
                channel="user",  # will be clamped to external anyway
                label=2,
                conf_label=0,    # attacker claims PUBLIC — must be floored
            )],
        )
        memdag_federation.import_changeset(self.conn, cs)

        row = self.conn.execute(
            "SELECT conf_label FROM nodes WHERE uuid='attacker:secret_node'"
        ).fetchone()
        self.assertIsNotNone(row, "attacker node should be inserted")
        stored_conf = row[0]
        self.assertGreaterEqual(
            stored_conf,
            memdag_federation.CONF_FLOOR_UNTRUSTED,
            f"untrusted conf_label=0 must be floored to >= {memdag_federation.CONF_FLOOR_UNTRUSTED}, got {stored_conf}"
        )


# ---------------------------------------------------------------------------
# test_register_origin_and_list
# Tests the new allowlist API — register_origin / is_trusted / list_origins
# ---------------------------------------------------------------------------

class TestRegisterOriginAndList(SecurityBase):

    def test_register_origin_and_list(self):
        """register_origin returns True; is_trusted True; list_origins includes entry;
        unregistered origin is_trusted False."""
        # Unregistered origin is not trusted
        self.assertFalse(
            memdag_federation.is_trusted(self.conn, "peer"),
            "unregistered origin must not be trusted"
        )

        # Register the origin
        result = memdag_federation.register_origin(self.conn, "peer", by="test",
                                                   descr="test peer")
        self.assertTrue(result, "register_origin must return True on success")

        # Now it should be trusted
        self.assertTrue(
            memdag_federation.is_trusted(self.conn, "peer"),
            "origin must be trusted after register_origin"
        )

        # list_origins must include it
        origins = [row[0] for row in memdag_federation.list_origins(self.conn)]
        self.assertIn("peer", origins, "list_origins must include the registered peer")

        # None and empty string are never trusted
        self.assertFalse(memdag_federation.is_trusted(self.conn, None),
                         "None origin must not be trusted")
        self.assertFalse(memdag_federation.is_trusted(self.conn, ""),
                         "empty string origin must not be trusted")

        # Idempotent: re-registering does not raise and still returns True
        result2 = memdag_federation.register_origin(self.conn, "peer", by="test2")
        self.assertTrue(result2, "re-registering same origin must be idempotent")


# ---------------------------------------------------------------------------
# test_tombstone_cascade_triggered_on_import
# Mirrors: poc9_tombstone_cascade_bypass.py — F-11
# Tests that tombstoning a parent via federation cascades to its live children.
# ---------------------------------------------------------------------------

class TestTombstoneCascadeTriggeredOnImport(SecurityBase):

    def test_tombstone_cascade_triggered_on_import(self):
        """Tombstone propagated via federation triggers revoke_cascade() on all descendants."""
        src_origin = "local"
        memdag_federation.register_origin(self.conn, src_origin, by="test")

        # Build: src -> child1 -> child2
        with self.conn:
            src_id = memdag.insert_node(self.conn, "Source endorsed content.", "endorsed")
            child1_id, _ = memdag.derive_node(self.conn, "Derived from endorsed.", [src_id])
            child2_id, _ = memdag.derive_node(self.conn, "Derived from child1.", [child1_id])

        memdag_federation.backfill_uuids(self.conn, src_origin)
        src_uuid = self.conn.execute(
            "SELECT uuid FROM nodes WHERE id=?", (src_id,)
        ).fetchone()[0]

        # Import tombstone of src from same origin (owned)
        cs = self._changeset(
            origin=src_origin,
            nodes=[self._node_dict(
                uuid=src_uuid,
                origin=src_origin,
                content="Source endorsed content.",
                channel="endorsed",
                label=3,
                tombstoned=1,
                tombstoned_at="2026-06-11T00:00:00+00:00",
                revoke_reason="source-revoked",
            )],
        )
        memdag_federation.import_changeset(self.conn, cs)

        src_dead = self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE id=?", (src_id,)
        ).fetchone()[0]
        child1_dead = self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE id=?", (child1_id,)
        ).fetchone()[0]
        child2_dead = self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE id=?", (child2_id,)
        ).fetchone()[0]

        self.assertEqual(src_dead, 1, "source must be tombstoned")
        self.assertEqual(child1_dead, 1,
                         "child1 must be cascade-tombstoned (F-11 fix: revoke_cascade called)")
        self.assertEqual(child2_dead, 1,
                         "child2 must be cascade-tombstoned (F-11 fix: revoke_cascade called)")


# ---------------------------------------------------------------------------
# test_trusted_origin_round_trip_unchanged
# Positive test: trusted origin keeps channels + labels through round-trip.
# ---------------------------------------------------------------------------

class TestTrustedOriginRoundTrip(SecurityBase):

    def test_trusted_origin_round_trip_unchanged(self):
        """A trusted origin's changeset preserves channel and label (no clamp)."""
        trusted_origin = "trusted-peer"
        memdag_federation.register_origin(self.conn, trusted_origin, by="test")

        endorsed_uuid = f"{trusted_origin}:endorsed_1"
        cs = self._changeset(
            origin=trusted_origin,
            nodes=[self._node_dict(
                uuid=endorsed_uuid,
                origin=trusted_origin,
                content="Trusted endorsed content from peer.",
                channel="endorsed",
                label=3,
                conf_label=2,
            )],
        )
        memdag_federation.import_changeset(self.conn, cs)

        row = self.conn.execute(
            "SELECT channel, label, conf_label FROM nodes WHERE uuid=?",
            (endorsed_uuid,)
        ).fetchone()
        self.assertIsNotNone(row, "trusted origin's node must be inserted")
        channel, label, conf_label = row
        self.assertEqual(channel, "endorsed",
                         "trusted origin's channel must be preserved")
        self.assertEqual(label, 3,
                         "trusted origin's source label must match RANK['endorsed']")
        self.assertEqual(conf_label, 2,
                         "trusted origin's conf_label must be preserved (recompute only floors derived)")


if __name__ == "__main__":
    unittest.main()
