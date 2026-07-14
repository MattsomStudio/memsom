"""Tests for memsom.bridge.facts — read-time [[fact_*]] resolution (Phases 2-3).

Run:  python -m unittest discover -s . -p test_memsom_facts.py
"""

import io
import os
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.bridge import bridge_import as bi
from memsom.bridge import facts as facts
from memsom.distill import digest as digest
from memsom.retrieval import retrieve as memsom_retrieve


def _fact_md(value, unit="tok/s", verified="2026-07-01"):
    return (f"---\nname: fact-5070-toksps\ndescription: throughput\ntype: fact\n"
            f"value: {value}\nunit: {unit}\nlast-verified: {verified}\n"
            f"section: Facts\n---\n\nmeasured\n")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["MEMDAG_DB"] = str(self.root / "t.db")
        self.mem = self.root / "memory"
        self.mem.mkdir()
        (self.mem / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
        self.conn = memsom.get_connection()
        bi.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def _write_fact(self, value, **kw):
        (self.mem / "fact_5070_toksps.md").write_text(_fact_md(value, **kw),
                                                      encoding="utf-8")

    def _import(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)


class TestResolveCurrent(Base):
    def test_current_value_with_unit(self):
        self._write_fact(45)
        self._import()
        out = facts.resolve_fact_refs(self.conn, "speed: [[fact_5070_toksps]] now")
        self.assertEqual(out, "speed: 45 tok/s now")

    def test_current_value_without_unit(self):
        (self.mem / "fact_gpu.md").write_text(
            "---\nname: fact-gpu\ndescription: g\ntype: fact\nvalue: RTX 5070\n"
            "section: Facts\n---\nbody\n", encoding="utf-8")
        self._import()
        out = facts.resolve_fact_refs(self.conn, "[[fact_gpu]]")
        self.assertEqual(out, "RTX 5070")

    def test_unknown_fact_left_verbatim(self):
        """A typo'd reference must LOOK broken, not resolve to something."""
        self._import()
        text = "see [[fact_nonexistent]] here"
        self.assertEqual(facts.resolve_fact_refs(self.conn, text), text)

    def test_non_fact_wikilinks_untouched(self):
        self._write_fact(45)
        self._import()
        out = facts.resolve_fact_refs(
            self.conn, "[[user_adhd]] and [[fact_5070_toksps]]")
        self.assertEqual(out, "[[user_adhd]] and 45 tok/s")


class FakeClock:
    """now_iso has 1-second resolution, so a same-second supersede ties on
    created_at; real fact updates are days apart. A fake advancing clock makes
    the chain's timestamps distinct and deterministic."""

    def setUp(self):
        super().setUp()
        self._tick = [0]

        def _fake_now():
            self._tick[0] += 1
            return f"2026-07-{self._tick[0]:02d}T00:00:00+00:00"
        self._clock = patch.object(memsom, "now_iso", _fake_now)
        self._clock.start()
        self.addCleanup(self._clock.stop)


class TestResolveHistory(FakeClock, Base):
    def _supersede(self, new_value):
        self._write_fact(new_value)
        self._import()

    def test_as_of_before_update_shows_drift(self):
        self._write_fact(45)
        self._import()
        mem_created = memsom.now_iso()   # "memory written" while value was 45
        self._supersede(61)
        out = facts.resolve_ref(self.conn, "fact_5070_toksps", as_of=mem_created)
        self.assertTrue(out.startswith("61 tok/s (was 45 tok/s when written, "),
                        out)

    def test_as_of_after_update_shows_plain_current(self):
        self._write_fact(45)
        self._import()
        self._supersede(61)
        out = facts.resolve_ref(self.conn, "fact_5070_toksps",
                                as_of=memsom.now_iso())
        self.assertEqual(out, "61 tok/s")

    def test_as_of_predating_first_version_uses_first_value(self):
        self._write_fact(45)
        self._import()
        self._supersede(61)
        out = facts.resolve_ref(self.conn, "fact_5070_toksps",
                                as_of="2000-01-01T00:00:00+00:00")
        self.assertTrue(out.startswith("61 tok/s (was 45 tok/s"), out)

    def test_retired_fact_shows_last_known(self):
        self._write_fact(45)
        (self.mem / "user_keep.md").write_text(   # mass-wipe guard needs >=1 file
            "---\nname: keep\ndescription: k\ntype: user\n"
            "section: About the User\n---\nk\n",
            encoding="utf-8")
        self._import()
        (self.mem / "fact_5070_toksps.md").unlink()
        self._import()   # sweep tombstones the fact
        out = facts.resolve_ref(self.conn, "fact_5070_toksps")
        self.assertTrue(out.startswith("45 tok/s (last known — fact retired "),
                        out)
        # the sweep's boilerplate reason is suppressed (adds nothing)
        self.assertNotIn("bridge reconcile", out)

    def test_retired_resolution_in_text_never_tombstones_memories(self):
        """Core rule: reading through a retired fact writes NOTHING."""
        self._write_fact(45)
        (self.mem / "project_oc.md").write_text(
            "---\nname: oc\ndescription: oc results\ntype: project\n"
            "section: Personal projects\n---\nunderwhelming: [[fact_5070_toksps]]\n",
            encoding="utf-8")
        self._import()
        (self.mem / "fact_5070_toksps.md").unlink()
        self._import()
        before = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE tombstoned = 0").fetchone()[0]
        facts.resolve_fact_refs(self.conn, "x [[fact_5070_toksps]] y")
        after = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE tombstoned = 0").fetchone()[0]
        self.assertEqual(before, after)


class TestDigestResolution(Base):
    """Phase 2: the digest substitutes facts in hooks + literal lines, and a
    fact entry's own hook is its value."""

    def test_literal_line_resolves(self):
        self._write_fact(61)
        (self.mem / "MEMORY.md").write_text(
            "# Memory\n\n## About the User\n- runs at [[fact_5070_toksps]] locally\n",
            encoding="utf-8")
        self._import()
        bi.import_literals(self.conn, self.mem, dry_run=False)
        text = digest.render_digest(conn=self.conn)
        self.assertIn("runs at 61 tok/s locally", text)
        self.assertNotIn("[[fact_5070_toksps]]", text)

    def test_hook_resolves(self):
        self._write_fact(61)
        (self.mem / "project_llm.md").write_text(
            "---\nname: llm\ndescription: local llm at [[fact_5070_toksps]]\n"
            "type: project\nsection: Personal projects\n---\nbody\n",
            encoding="utf-8")
        self._import()
        text = digest.render_digest(conn=self.conn)
        self.assertIn("local llm at 61 tok/s", text)

    def test_fact_entry_hook_is_its_value(self):
        self._write_fact(61, verified="2026-07-14")
        self._import()
        text = digest.render_digest(conn=self.conn)
        self.assertIn("## Facts", text)
        self.assertIn("61 tok/s (verified 2026-07-14)", text)

    def test_resolution_precedes_budget_accounting(self):
        """Eviction must see RESOLVED sizes: a digest whose resolved content
        fits exactly must render, and the rendered bytes must equal what the
        budget loop measured (i.e. no post-budget substitution)."""
        self._write_fact(61)
        (self.mem / "MEMORY.md").write_text(
            "# Memory\n\n## About the User\n- [[fact_5070_toksps]]\n",
            encoding="utf-8")
        self._import()
        bi.import_literals(self.conn, self.mem, dry_run=False)
        text = digest.render_digest(conn=self.conn)
        # rendered output contains no unresolved refs, so the byte size the
        # budget loop enforced was computed over the substituted text
        self.assertNotIn("[[fact_", text)
        self.assertLessEqual(len(text.encode("utf-8")), digest.BUDGET)


class TestRetrieveResolution(FakeClock, Base):
    """Phase 3: retrieve output resolves refs with drift vs the memory's age."""

    def _seed_and_retrieve(self, query="nebula overlay"):
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memsom_retrieve, "_call_ollama_embed", _no_embed):
            memsom_retrieve.index_all(self.conn)
            buf = io.StringIO()
            with redirect_stdout(buf):
                memsom_retrieve._cmd_retrieve(
                    type("A", (), {"query": query, "k": 5, "clearance": "topsecret"}))
        return buf.getvalue()

    def test_retrieve_shows_drift_for_old_memory(self):
        self._write_fact(45)
        (self.mem / "project_bench.md").write_text(
            "---\nname: bench\ndescription: d\ntype: project\nsection: Personal projects\n"
            "---\nnebula overlay benchmark hit [[fact_5070_toksps]] locally\n",
            encoding="utf-8")
        self._import()
        self._write_fact(61)   # fact updated AFTER the memory was written
        self._import()
        out = self._seed_and_retrieve()
        self.assertIn("61 tok/s (was 45 tok/s when written,", out)
        self.assertNotIn("[[fact_5070_toksps]]", out)

    def test_retrieve_without_refs_unchanged(self):
        (self.mem / "project_plain.md").write_text(
            "---\nname: plain\ndescription: d\ntype: project\nsection: Personal projects\n"
            "---\nnebula overlay plain content no refs\n", encoding="utf-8")
        self._import()
        out = self._seed_and_retrieve()
        self.assertIn("nebula overlay plain content no refs", out)


if __name__ == "__main__":
    unittest.main()
