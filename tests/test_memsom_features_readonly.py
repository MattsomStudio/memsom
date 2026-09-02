"""Review fix ARCH-06: `memsom features` / `memsom doctor` open the store
read-only (memsom.get_connection(read_only=True)) -- no registrant's status()
probe may attempt a write on that connection. Before this fix, two probes
called their module's migrate(conn) (a CREATE TABLE) on a fresh store opened
read-only, so a healthy store reported retrieval.dense/federation.sync as
`error` with 'attempt to write a readonly database' instead of a real state.

Run:  python -m unittest discover -s . -p test_memsom_features_readonly.py
"""
import os
import tempfile
import unittest
from pathlib import Path

import memsom
from memsom.interface import cli as memsom_cli
from memsom.interface import features as memsom_features


class ReadOnlyFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved = {
            k: os.environ.get(k) for k in ("MEMDAG_HOME", "MEMDAG_DB")
        }
        os.environ["MEMDAG_HOME"] = str(self.root)
        os.environ["MEMDAG_DB"] = str(self.root / "memdag.db")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_no_registrant_errors_on_a_readonly_fresh_store(self):
        db = str(self.root / "memdag.db")
        conn = memsom.get_connection(db)
        try:
            memsom_cli.migrate_all(conn)
        finally:
            conn.close()

        ro = memsom.get_connection(db, read_only=True)
        try:
            statuses = memsom_features.all_statuses(ro)
        finally:
            ro.close()

        errored = {name: st["detail"] for name, st in statuses.items() if st["state"] == "error"}
        self.assertEqual(errored, {}, f"registrant(s) errored on a read-only connection: {errored}")


if __name__ == "__main__":
    unittest.main()
