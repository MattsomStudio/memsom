#!/usr/bin/env python3
"""Tests for the seed-data scrub + the scrub-gate.

Run:  python -m unittest -v test_scrub
"""

import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
import scrub_gate  # noqa: E402

import memdag  # noqa: E402


class TestScrubGate(unittest.TestCase):
    def test_scrub_gate_passes_on_clean_tree(self):
        # The real repo tree must be clean — no author-identifying tokens.
        hits = scrub_gate.scan(HERE)
        self.assertEqual(hits, [], f"scrub gate found leaks: {hits}")

    def test_scrub_gate_catches_planted_leak(self):
        # Prove the gate actually bites: plant a token and confirm it's caught.
        # The username token is assembled at runtime so this test file itself
        # stays clean of the literal token (otherwise the gate flags its own test).
        username = "Fur" + "io"
        with tempfile.TemporaryDirectory() as d:
            leak = Path(d) / "note.md"
            leak.write_text(f"setup runs under C:/Users/{username}/memdag here\n", encoding="utf-8")
            hits = scrub_gate.scan(d)
            self.assertTrue(hits, "gate failed to catch a planted username token")
            self.assertTrue(any(tok == username.lower() for _, _, tok, _ in hits))


class TestSeedNoPersonalData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "memdag.db"
        os.environ["MEMDAG_DB"] = str(self.db)

    def tearDown(self):
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def test_cmd_seed_no_personal_data(self):
        # Run the (scrubbed) demo seed offline; no node content/source_ref may
        # contain any leak token.
        memdag.cmd_seed(Namespace(reset=True, offline=True))
        conn = memdag.get_connection()
        try:
            rows = conn.execute("SELECT content, COALESCE(source_ref,'') FROM nodes").fetchall()
        finally:
            conn.close()
        self.assertEqual(len(rows), 3, "seed should create exactly 3 demo nodes")
        blob = "\n".join(f"{c}\n{r}" for c, r in rows).lower()
        for tok in scrub_gate.LEAK_TOKENS:
            self.assertNotIn(tok, blob, f"seed content leaks token {tok!r}")


if __name__ == "__main__":
    unittest.main()
