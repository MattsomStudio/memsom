#!/usr/bin/env python3
"""Phase 4 — `doctor`: paste-ready report (env + verbatim selfcheck), never throws.

Run:  python -m unittest -v test_doctor
"""

import contextlib
import io
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import memdag
import memdag_doctor


class TestDoctor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "memdag.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        memdag.get_connection().close()  # create the DB so node-count path runs

    def tearDown(self):
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_EMBED_URL", None)
        self.tmp.cleanup()

    def _run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            memdag_doctor.cmd_doctor(Namespace(json=False))
        return out.getvalue()

    def test_doctor_reports_env_and_selfcheck(self):
        # Point Ollama at a dead port so the probe is fast + deterministic.
        os.environ["MEMDAG_EMBED_URL"] = "http://127.0.0.1:1/api/embeddings"
        text = self._run()
        # environment facts doctor adds
        for marker in ("memdag version", "OS", "Python", "DB path", str(self.db)):
            self.assertIn(marker, text)
        # the wrapped selfcheck section is present
        self.assertIn("selfcheck", text)

    def test_doctor_ollama_unreachable_graceful(self):
        os.environ["MEMDAG_EMBED_URL"] = "http://127.0.0.1:1/api/embeddings"
        text = self._run()  # must not raise
        self.assertIn("unreachable", text)


if __name__ == "__main__":
    unittest.main()
