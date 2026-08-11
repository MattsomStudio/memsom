#!/usr/bin/env python3
"""Tests for memsom.interface.remote -- Sec3.5 point 3 / MS-02.

"Tokens carry a clearance ceiling, enforced server-side... The client never
states its own clearance." A public-ceiling token must never see a topsecret
node's content, regardless of what --clearance the tool call itself asks for.
"""

import os
import tempfile
import unittest
from pathlib import Path

import memsom
from memsom.integrity import confid as memsom_confid
from memsom.interface import cli as memsom_cli
from memsom.interface import remote as R


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_cli.migrate_all(self.conn)
        R.reset_device_sessions()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()


class TestEffectiveClearance(unittest.TestCase):
    """Pure function, no DB needed."""

    def test_no_request_defaults_to_device_ceiling(self):
        self.assertEqual(R.effective_clearance_name(0), "public")
        self.assertEqual(R.effective_clearance_name(3), "topsecret")

    def test_request_above_ceiling_is_clamped(self):
        self.assertEqual(R.effective_clearance_name(0, "topsecret"), "public")
        self.assertEqual(R.effective_clearance_name(1, "topsecret"), "internal")

    def test_request_below_ceiling_is_honoured(self):
        self.assertEqual(R.effective_clearance_name(3, "public"), "public")

    def test_request_equal_to_ceiling(self):
        self.assertEqual(R.effective_clearance_name(2, "secret"), "secret")


class TestServerSideClamp(Base):
    def _seed(self):
        pub = memsom.insert_node(self.conn, "public fact about weather", "user")
        top = memsom.insert_node(self.conn, "topsecret launch codes revealed", "user")
        memsom_confid.classify(self.conn, pub, "public")
        memsom_confid.classify(self.conn, top, "topsecret")
        return pub, top

    def test_public_ceiling_device_cannot_see_topsecret_via_dump(self):
        pub, top = self._seed()
        d = R.add_device(self.conn, "laptop", "public", [])
        # even asking explicitly for topsecret is clamped server-side
        r = R.handle_request(self.conn, d["token"], "explain",
                             {"id": top, "clearance": "topsecret"})
        self.assertEqual(r["decision"], "allow")
        self.assertNotIn("launch codes", r["text"])
        self.assertIn("ABOVE CLEARANCE", r["text"])

    def test_public_ceiling_device_can_see_public_node(self):
        pub, top = self._seed()
        d = R.add_device(self.conn, "laptop", "public", [])
        r = R.handle_request(self.conn, d["token"], "explain", {"id": pub})
        self.assertEqual(r["decision"], "allow")
        self.assertIn("public fact about weather", r["text"])

    def test_topsecret_ceiling_device_sees_everything(self):
        pub, top = self._seed()
        d = R.add_device(self.conn, "server-admin", "topsecret", [])
        r = R.handle_request(self.conn, d["token"], "explain", {"id": top})
        self.assertEqual(r["decision"], "allow")
        self.assertIn("launch codes", r["text"])

    def test_clearance_used_reported_in_response(self):
        pub, top = self._seed()
        d = R.add_device(self.conn, "laptop", "public", [])
        r = R.handle_request(self.conn, d["token"], "ask",
                             {"question": "weather", "clearance": "topsecret"})
        self.assertEqual(r["clearance_used"], "public")


if __name__ == "__main__":
    unittest.main()
