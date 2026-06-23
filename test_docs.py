#!/usr/bin/env python3
"""Phase 7 — docs: README is leak-free and the issue template prompts for doctor.

Run:  python -m unittest -v test_docs
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
import scrub_gate  # noqa: E402


class TestDocs(unittest.TestCase):
    def test_readme_has_no_leak_tokens(self):
        text = (HERE / "README.md").read_text(encoding="utf-8")
        hits = scrub_gate.scan_text(text)
        self.assertEqual(hits, [], f"README leaks tokens: {hits}")

    def test_issue_template_exists_and_prompts_doctor(self):
        tpl = HERE / ".github" / "ISSUE_TEMPLATE" / "bug_report.md"
        self.assertTrue(tpl.exists(), "bug-report issue template missing")
        body = tpl.read_text(encoding="utf-8")
        self.assertIn("memdag doctor", body)

    def test_readme_documents_bootstrap_and_data_dir(self):
        text = (HERE / "README.md").read_text(encoding="utf-8")
        self.assertIn("bootstrap.py", text)
        self.assertIn("~/.memdag", text)


if __name__ == "__main__":
    unittest.main()
