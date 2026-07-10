#!/usr/bin/env python3
"""Tests for memsom_distill — provenance-filtered training-set export.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_distill.py \
    -t <repo> -v
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.distill import distill as memsom_distill
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import redact as memsom_redact

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_distill.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])


class TestExternalDerivedExcluded(Base):
    """test_external_derived_excluded:
    A derived node whose parent is external must be absent at min_integrity=1.
    Even if we elevate its label to 2 (elevated-but-tainted), the ancestor CTE
    still catches and excludes it.
    """

    def test_external_derived_excluded(self):
        ext = self.add("external source content", "external")
        d_ext, _ = memsom.derive_node(self.conn, "answer from external", [ext])

        # At min_integrity=1: excluded because label=0
        records = memsom_distill.export_training(self.conn, min_integrity=1)
        ids = [r["provenance"][0]["id"] if r["provenance"] else None for r in records]
        self.assertNotIn(d_ext, [r["output"] for r in records],
                         "node derived from external must be absent at min_integrity=1")
        # Confirm by id
        exported_ids = self._export_ids(min_integrity=1)
        self.assertNotIn(d_ext, exported_ids)

        # Elevate d_ext label to 2 (raw UPDATE — simulates a trust elevation bypass attempt)
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=2 WHERE id=?", (d_ext,))

        # Still excluded: ancestor CTE detects the live external ancestor
        exported_ids = self._export_ids(min_integrity=1)
        self.assertNotIn(d_ext, exported_ids,
                         "elevated-but-tainted node must still be excluded by ancestor CTE")

    def _export_ids(self, min_integrity=1):
        records = memsom_distill.export_training(self.conn, min_integrity=min_integrity)
        # Recover IDs by matching output content against nodes
        result = set()
        for r in records:
            row = self.conn.execute(
                "SELECT id FROM nodes WHERE content=? AND channel='agent-derived'",
                (r["output"],)
            ).fetchone()
            if row:
                result.add(row[0])
        return result


class TestRedactedExcluded(Base):
    """test_redacted_excluded: a redacted derived node must be absent."""

    def test_redacted_excluded(self):
        u = self.add("user source fact about something important", "user")
        d, _ = memsom.derive_node(self.conn, "Q: what is important?\nA (composed from 1 live sources):\n- user source fact. [mem:1|user]", [u])

        # Before redaction: present
        ids = self._export_ids()
        self.assertIn(d, ids, "live derived node must appear before redaction")

        # After redaction: absent
        memsom_redact.redact_node(self.conn, d, reason="test redaction")
        ids = self._export_ids()
        self.assertNotIn(d, ids, "redacted node must be absent from training export")

    def _export_ids(self):
        records = memsom_distill.export_training(self.conn, min_integrity=1)
        result = set()
        for r in records:
            row = self.conn.execute(
                "SELECT id FROM nodes WHERE content=? AND channel='agent-derived'",
                (r["output"],)
            ).fetchone()
            if row:
                result.add(row[0])
        return result


class TestQuarantinedExcluded(Base):
    """test_quarantined_excluded: a quarantined derived node must be absent."""

    def test_quarantined_excluded(self):
        u = self.add("user source fact for quarantine test", "user")
        d, _ = memsom.derive_node(self.conn, "answer derived from user source quarantine", [u])

        ids = self._export_ids()
        self.assertIn(d, ids, "live node must appear before quarantine")

        memsom_quarantine.quarantine_node(self.conn, d, reason="suspicious")
        ids = self._export_ids()
        self.assertNotIn(d, ids, "quarantined node must be absent from training export")

    def _export_ids(self):
        records = memsom_distill.export_training(self.conn, min_integrity=1)
        result = set()
        for r in records:
            row = self.conn.execute(
                "SELECT id FROM nodes WHERE content=? AND channel='agent-derived'",
                (r["output"],)
            ).fetchone()
            if row:
                result.add(row[0])
        return result


class TestMinIntegrityHonored(Base):
    """test_min_integrity_honored:
    - d_user (from user source, label=2) present at min=1 and 2, absent at 3
    - endorsed-derived (label=3) present at min=3
    """

    def test_min_integrity_honored(self):
        # d_user: derived from user source -> label = min(2) = 2
        u = self.add("user source content for integrity test long enough", "user")
        d_user, _ = memsom.derive_node(self.conn, "answer from user source for integrity check", [u])
        self.assertEqual(memsom.get_node(self.conn, d_user)["label"], 2)

        # d_endorsed: derived from endorsed source -> label = min(3) = 3
        e = self.add("endorsed source content for integrity test long enough", "endorsed")
        d_endorsed, _ = memsom.derive_node(self.conn, "answer from endorsed source for integrity check", [e])
        self.assertEqual(memsom.get_node(self.conn, d_endorsed)["label"], 3)

        def ids_at(floor):
            records = memsom_distill.export_training(self.conn, min_integrity=floor)
            result = set()
            for r in records:
                row = self.conn.execute(
                    "SELECT id FROM nodes WHERE content=? AND channel='agent-derived'",
                    (r["output"],)
                ).fetchone()
                if row:
                    result.add(row[0])
            return result

        # d_user (label=2) present at floor 1 and 2
        self.assertIn(d_user, ids_at(1))
        self.assertIn(d_user, ids_at(2))
        # d_user absent at floor 3
        self.assertNotIn(d_user, ids_at(3))

        # d_endorsed (label=3) present at floor 3
        self.assertIn(d_endorsed, ids_at(3))

        # String name for min_integrity
        self.assertIn(d_user, ids_at("agent-derived"))  # floor=1
        self.assertIn(d_user, ids_at("user"))            # floor=2
        self.assertNotIn(d_user, ids_at("endorsed"))     # floor=3


class TestValidJsonlAndInstructionRecovery(Base):
    """test_valid_jsonl_and_instruction_recovery:
    - compose+derive a real answer (content starts 'Q: ...')
    - export + write
    - re-read file: every line parses as JSON
    - record['instruction'] == the question
    - record['output'] == full node content
    - provenance ids match the parents
    """

    def test_valid_jsonl_and_instruction_recovery(self):
        # Create sources
        u = self.add("Nebula lighthouse requires a stable public IP for hole punching.", "user")
        e = self.add("Endorsed nebula config note: use static_host_map for reliability.", "endorsed")

        # Compose a real answer
        sources = memsom.live_sources(self.conn)
        question = "How should I configure Nebula?"
        text, used = memsom.compose(question, sources)
        self.assertIsNotNone(text, "compose must succeed with live sources")
        self.assertTrue(text.startswith("Q: "), f"composed text should start with 'Q: ', got: {text[:50]!r}")

        nid, _ = memsom.derive_node(self.conn, text, used)

        # Export
        records = memsom_distill.export_training(self.conn, min_integrity=1)
        self.assertTrue(len(records) >= 1, "at least one record must be exported")

        # Find the record for this node
        matching = [r for r in records if r["output"] == text]
        self.assertEqual(len(matching), 1, "exactly one record should match our node's content")
        rec = matching[0]

        # instruction == the question (first line minus 'Q: ')
        self.assertEqual(rec["instruction"], question,
                         f"instruction should be the question; got {rec['instruction']!r}")

        # input is empty string
        self.assertEqual(rec["input"], "")

        # output == full content
        self.assertEqual(rec["output"], text)

        # provenance ids match parents
        parent_ids = {r[0] for r in memsom.parents_of(self.conn, nid)}
        prov_ids = {p["id"] for p in rec["provenance"]}
        self.assertEqual(prov_ids, parent_ids,
                         f"provenance ids {prov_ids} should equal parent ids {parent_ids}")

        # Write JSONL to a temp file and re-parse every line
        out_path = Path(self.tmp.name) / "out.jsonl"
        memsom_distill.write_jsonl(out_path, records)
        self.assertTrue(out_path.exists())
        lines = out_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(records))
        for i, line in enumerate(lines):
            parsed = json.loads(line)  # must not raise
            self.assertIn("instruction", parsed)
            self.assertIn("output", parsed)
            self.assertIn("provenance", parsed)


class TestRevokedExcluded(Base):
    """test_revoked_excluded: a tombstoned (revoked) derived node must be absent."""

    def test_revoked_excluded(self):
        u = self.add("user source for revoke test", "user")
        d, _ = memsom.derive_node(self.conn, "answer to be revoked for distill test", [u])

        ids = self._export_ids()
        self.assertIn(d, ids, "live node must appear before revoke")

        memsom.revoke_cascade(self.conn, d, "bad answer")
        ids = self._export_ids()
        self.assertNotIn(d, ids, "tombstoned node must be absent from training export")

    def _export_ids(self):
        records = memsom_distill.export_training(self.conn, min_integrity=1)
        result = set()
        for r in records:
            row = self.conn.execute(
                "SELECT id FROM nodes WHERE content=? AND channel='agent-derived'",
                (r["output"],)
            ).fetchone()
            if row:
                result.add(row[0])
        return result


class TestDistillPlanWritesStub(Base):
    """test_distill_plan_writes_stub:
    - plan text mentions the model and the manual GPU boundary
    - distill.ps1 exists and contains 'TODO' and 'export-training'
    - distill_config.json parses and carries the model
    - NO subprocess is spawned (assert 'subprocess' not in source)
    - shutil.which is used for detection only (monkeypatched to return a fake path)
    """

    def test_distill_plan_writes_stub(self):
        # Confirm subprocess is not imported by the module at all
        src = Path(memsom_distill.__file__).read_text(encoding="utf-8")
        self.assertNotIn("subprocess", src,
                         "memsom_distill must not import subprocess")

        out_dir = Path(self.tmp.name) / "plan_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        test_model = "test-model:latest"

        # Monkeypatch shutil.which to return a fake path (simulating ollama present)
        with patch.object(shutil, "which", return_value="/usr/local/bin/ollama") as mock_which:
            plan = memsom_distill.distill_plan(model=test_model, out_dir=str(out_dir))
            # shutil.which was called (for detection)
            mock_which.assert_called()

        # Plan text must mention the model
        self.assertIn(test_model, plan,
                      "plan text must mention the model name")

        # Plan text must mention the GPU boundary
        self.assertIn("manual GPU step", plan,
                      "plan text must mention the manual GPU step boundary")

        # Plan text should note ollama detected
        self.assertIn("ollama detected", plan,
                      "plan text must note ollama detected when shutil.which returns a path")

        # distill.ps1 exists
        ps1 = out_dir / "distill.ps1"
        self.assertTrue(ps1.exists(), "distill.ps1 must be written")
        ps1_text = ps1.read_text(encoding="utf-8")
        self.assertIn("TODO", ps1_text, "distill.ps1 must contain TODO markers")
        self.assertIn("export-training", ps1_text,
                      "distill.ps1 must reference the export-training step")

        # distill_config.json parses and carries the model
        config_path = out_dir / "distill_config.json"
        self.assertTrue(config_path.exists(), "distill_config.json must be written")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["model"], test_model,
                         "distill_config.json must carry the model name")
        self.assertIn("dataset", config)
        self.assertIn("output", config)

        # Also test with ollama NOT found
        with patch.object(shutil, "which", return_value=None):
            plan_no_ollama = memsom_distill.distill_plan(model=test_model, out_dir=str(out_dir))
        self.assertIn("ollama not found", plan_no_ollama,
                      "plan text must note ollama not found when shutil.which returns None")


if __name__ == "__main__":
    unittest.main()
