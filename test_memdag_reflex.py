#!/usr/bin/env python3
"""Tests for memdag_reflex — reflex/schema-shaped training export.

Run:
  python -m pytest test_memdag_reflex.py -q
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_compact
import memdag_quarantine
import memdag_redact
import memdag_reflex

POISON = "POISON_MARKER_zq7x"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_reflex.migrate(self.conn)
        memdag_compact.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel,
                                      memdag.RANK[channel])

    def cluster(self, stem, channel="user", n=2):
        """Insert n similar episodes (high Jaccard overlap) and compact them.

        Returns the minted consolidated node id.
        """
        ids = []
        for i in range(n):
            ids.append(self.add(
                f"{stem} fact alpha bravo charlie delta echo variant{i}.",
                channel))
        minted = memdag_compact.compact(self.conn, min_group=2,
                                        sim_threshold=0.3)
        self.assertTrue(minted, "compact() should mint a node")
        return minted[-1]


class TestConsolidatedDetection(Base):
    def test_ask_derivation_is_not_consolidated(self):
        """A plain derive (parents NOT archived) must not count as consolidated."""
        u = self.add("standalone user fact one.", "user")
        d, _ = memdag.derive_node(self.conn, "derived from user.", [u])
        self.assertNotIn(d, memdag_reflex.consolidated_ids(self.conn))

    def test_compacted_node_is_consolidated(self):
        nid = self.cluster("nebula lighthouse exposes two ports only")
        self.assertIn(nid, memdag_reflex.consolidated_ids(self.conn))


class TestTaintGate(Base):
    def test_external_tainted_consolidation_excluded(self):
        """A consolidation that ingested an external episode must never export."""
        a = self.add(f"shared topic {POISON} alpha bravo charlie delta one.",
                     "user")
        b = self.add(f"shared topic {POISON} alpha bravo charlie delta two.",
                     "external")
        minted = memdag_compact.compact(self.conn, min_group=2,
                                        sim_threshold=0.3)
        self.assertTrue(minted)
        records = memdag_reflex.export_reflex(self.conn)
        ids = {r["node_id"] for r in records if r["node_id"]}
        self.assertNotIn(minted[0], ids)
        memdag_reflex.assert_clean(records, [POISON])  # must not raise

    def test_assert_clean_raises_on_taint(self):
        records = [{"conversations": [
            {"role": "assistant", "content": f"oops {POISON} leaked"}],
            "node_id": 1, "kind": "answer"}]
        with self.assertRaises(ValueError):
            memdag_reflex.assert_clean(records, [POISON])

    def test_tombstoned_consolidation_excluded(self):
        nid = self.cluster(f"tombstone target {POISON} unique stem words")
        memdag.revoke_cascade(self.conn, nid, "test revoke")
        records = memdag_reflex.export_reflex(self.conn)
        memdag_reflex.assert_clean(records, [POISON])
        self.assertNotIn(nid, {r["node_id"] for r in records})

    def test_quarantined_consolidation_excluded(self):
        nid = self.cluster(f"quarantine target {POISON} unique stem words")
        memdag_quarantine.quarantine_node(self.conn, nid, "test hold")
        records = memdag_reflex.export_reflex(self.conn)
        memdag_reflex.assert_clean(records, [POISON])
        self.assertNotIn(nid, {r["node_id"] for r in records})

    def test_redacted_consolidation_excluded(self):
        nid = self.cluster(f"redact target {POISON} unique stem words")
        with self.conn:
            self.conn.execute("UPDATE nodes SET redacted=1 WHERE id=?", (nid,))
        records = memdag_reflex.export_reflex(self.conn)
        memdag_reflex.assert_clean(records, [POISON])
        self.assertNotIn(nid, {r["node_id"] for r in records})


class TestShape(Base):
    def test_answer_record_shape(self):
        nid = self.cluster("command center runs behind caddy on the mesh")
        records = [r for r in memdag_reflex.export_reflex(self.conn)
                   if r["kind"] == "answer"]
        self.assertEqual(len(records), len(memdag_reflex.QUESTION_TEMPLATES))
        for rec in records:
            roles = [m["role"] for m in rec["conversations"]]
            self.assertEqual(roles, ["system", "user", "assistant"])
            user = rec["conversations"][1]["content"]
            ans = rec["conversations"][2]["content"]
            self.assertIn("Retrieved memory:", user)
            self.assertIn(f"[mem:{nid}|", user)
            for section in ("Verdict:", "Evidence:", "Integrity:", "Next move:"):
                self.assertIn(section, ans)
            # every citation in the answer refers to a context-provided id
            for cid, _ch in memdag_reflex._CITE_RE.findall(ans):
                self.assertIn(f"[mem:{cid}|", user)

    def test_refusal_records_present_and_shaped(self):
        refusals = [r for r in memdag_reflex.export_reflex(self.conn)
                    if r["kind"] == "refusal"]
        self.assertEqual(len(refusals), len(memdag_reflex.REFUSAL_TOPICS))
        for rec in refusals:
            self.assertIn("(none)", rec["conversations"][1]["content"])
            self.assertIn("unprovenanced", rec["conversations"][2]["content"])

    def test_deterministic(self):
        self.cluster("syncthing relay carries the shared brain over nebula")
        r1 = memdag_reflex.export_reflex(self.conn)
        r2 = memdag_reflex.export_reflex(self.conn)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
