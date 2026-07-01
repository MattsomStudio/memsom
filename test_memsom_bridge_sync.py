"""Tests for memsom_bridge_sync — Mac<->PC changeset round-trip (Phase 4).

Run:  python -m unittest discover -s . -p test_memsom_bridge_sync.py
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_bridge_sync as sync


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.syncdir = self.root / "claude-sync" / "memsom"
        # Pin a sentinel self-origin so federation's migrate-time self-trust
        # auto-registers THIS sentinel, not the ambient machine's MEMDAG_ORIGIN.
        # Without this, on a host whose MEMDAG_ORIGIN equals a peer name used
        # below (e.g. 'pc'), the peer collides with the auto-trusted self origin
        # and the untrusted-import clamp test sees it as trusted (env leak).
        self._saved_origin = os.environ.get("MEMDAG_ORIGIN")
        os.environ["MEMDAG_ORIGIN"] = "bridge-test-self"
        self.pc = memsom.get_connection(self.root / "pc.db")
        self.mac = memsom.get_connection(self.root / "mac.db")
        sync.migrate(self.pc)
        sync.migrate(self.mac)

    def tearDown(self):
        self.pc.close()
        self.mac.close()
        os.environ.pop("MEMDAG_DB", None)
        if self._saved_origin is None:
            os.environ.pop("MEMDAG_ORIGIN", None)
        else:
            os.environ["MEMDAG_ORIGIN"] = self._saved_origin
        self.tmp.cleanup()

    def add(self, conn, stem, channel="endorsed"):
        nid = memsom.insert_node(conn, f"content {stem}", channel,
                                 source_ref=f"memory:{stem}")
        conn.commit()
        return nid

    def node(self, conn, stem):
        row = conn.execute(
            "SELECT channel, label, tombstoned FROM nodes WHERE source_ref = ?",
            (f"memory:{stem}",)).fetchone()
        return row


class TestRoundTrip(Base):
    def test_pc_node_reaches_mac_with_channel_preserved(self):
        self.add(self.pc, "user_x", "endorsed")
        sync.export_for_sync(self.pc, self.syncdir, origin="pc")
        res = sync.import_from_sync(self.mac, self.syncdir, self_origin="mac")
        self.assertIn("pc", res)
        ch, label, tomb = self.node(self.mac, "user_x")
        self.assertEqual(ch, "endorsed")   # trusted peer -> channel preserved
        self.assertEqual(label, 3)
        self.assertEqual(tomb, 0)

    def test_does_not_import_own_changeset(self):
        self.add(self.mac, "mac_local", "user")
        sync.export_for_sync(self.mac, self.syncdir, origin="mac")
        self.add(self.pc, "pc_one", "user")
        sync.export_for_sync(self.pc, self.syncdir, origin="pc")
        res = sync.import_from_sync(self.mac, self.syncdir, self_origin="mac")
        self.assertIn("pc", res)
        self.assertNotIn("mac", res)        # skipped own file

    def test_reimport_idempotent(self):
        self.add(self.pc, "user_y", "endorsed")
        sync.export_for_sync(self.pc, self.syncdir, origin="pc")
        sync.import_from_sync(self.mac, self.syncdir, self_origin="mac")
        before = self.mac.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        res = sync.import_from_sync(self.mac, self.syncdir, self_origin="mac")
        after = self.mac.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(before, after)               # no duplication
        self.assertEqual(res["pc"]["nodes_new"], 0)

    def test_untrusted_import_clamps_to_external(self):
        self.add(self.pc, "user_z", "endorsed")
        sync.export_for_sync(self.pc, self.syncdir, origin="pc")
        sync.import_from_sync(self.mac, self.syncdir, self_origin="mac", trust=False)
        ch, label, _ = self.node(self.mac, "user_z")
        self.assertEqual(ch, "external")   # default-deny clamp
        self.assertEqual(label, 0)

    def test_tombstone_propagates_from_owner(self):
        nid = self.add(self.pc, "user_w", "endorsed")
        sync.export_for_sync(self.pc, self.syncdir, origin="pc")
        sync.import_from_sync(self.mac, self.syncdir, self_origin="mac")
        self.assertEqual(self.node(self.mac, "user_w")[2], 0)  # live on mac
        # owner (pc) tombstones, re-exports
        self.pc.execute(
            "UPDATE nodes SET tombstoned = 1, tombstoned_at = ? WHERE id = ?",
            (memsom.now_iso(), nid))
        self.pc.commit()
        sync.export_for_sync(self.pc, self.syncdir, origin="pc")
        sync.import_from_sync(self.mac, self.syncdir, self_origin="mac")
        self.assertEqual(self.node(self.mac, "user_w")[2], 1)  # now dead on mac


if __name__ == "__main__":
    unittest.main()
