#!/usr/bin/env python3
"""Smoke test for bench/ — every script must at least compile.

bench/ is deliberately excluded from test discovery (research/eval harnesses,
not shipped), which means nothing ever catches a syntax error or a
Python-version-incompatible construct until someone runs the script by hand.
This test compiles every .py under bench/ WITHOUT importing it — no module
side effects, no DB access, no model downloads.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_bench_smoke.py \
    -t <repo> -v
"""

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench"


class TestBenchScriptsCompile(unittest.TestCase):

    def test_bench_dir_exists(self):
        self.assertTrue(BENCH.is_dir(), f"expected bench/ at {BENCH}")

    def test_every_bench_script_compiles(self):
        scripts = sorted(BENCH.rglob("*.py"))
        self.assertTrue(scripts, "bench/ should contain at least one .py script")
        for script in scripts:
            with self.subTest(script=str(script.relative_to(REPO))):
                source = script.read_text(encoding="utf-8", errors="replace")
                compile(source, str(script), "exec")


if __name__ == "__main__":
    unittest.main()
