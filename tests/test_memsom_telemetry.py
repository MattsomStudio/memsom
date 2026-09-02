"""Tests for memsom.interface.telemetry -- the panel's contract surface
(PROMOTE task W2 / PROMOTE-Q11-PANEL.md Part B1), the memsom.interface.dashboard
deprecation shim, and the 'telemetry' feature probe.

Run:  python -m unittest discover -s . -p test_memsom_telemetry.py
"""
import contextlib
import inspect
import io
import os
import tempfile
import unittest
from pathlib import Path

import memsom
from memsom.bridge import bridge_render as br
from memsom.interface import cli as memsom_cli
from memsom.interface import features as memsom_features

# Frozen contract (PROMOTE-Q11-PANEL.md Part B1/B2): these key sets must never
# gain or lose a member without a deliberate, reviewed panel-side migration.
FROZEN_TELEMETRY_KEYS = {
    "generated", "last_consolidation", "totals", "tier", "types", "hist",
    "top_access", "scatter", "growth", "stale", "budget", "sessions",
    "thresholds", "graph",
}
FROZEN_WEIGHT_ROW_KEYS = {
    "stem", "weight", "count", "last_used", "first_seen", "tier", "channel", "pinned",
}

# Generic fixtures -- no author-identifying content (the scrub gate scans this file).
MEMORY_FILES = {
    "user_editor.md": "---\nname: Editor\ndescription: prefers tabs\ntype: user\n---\nbody\n",
    "project_widget.md": "---\nname: Widget\ndescription: status\ntype: project\n---\ns\n",
}
INDEX = """# Memory

## About the User
- [Editor](user_editor.md) — prefers tabs

## Personal projects
- [Widget](project_widget.md) — status
"""

# G-3: the wikilink-graph-edge fixture (ported from the deleted
# test_memsom_dashboard.py::test_wikilink_becomes_graph_link).
LINKED_PROJECT = "---\nname: Widget\ndescription: status\ntype: project\n---\nsee [[user_editor]]\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["MEMDAG_HOME"] = str(self.root)
        os.environ["MEMDAG_DB"] = str(self.root / "memdag.db")
        # keep ingest off the network -- BM25-only for the suite (same idiom
        # test_memsom_bridge_render.py uses).
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
        # NEVER let claude-sync touch the real ~/.claude/CLAUDE.md during tests.
        os.environ["CLAUDE_MD_PATH"] = str(self.root / "CLAUDE.md")
        # default_memory_dir()'s bootstrap-exempt env var (kernel/paths.py) --
        # pins the zero-arg build_telemetry()/default_memory_dir() calls away
        # from the real ~/.claude/projects/*/memory on this machine.
        self.mem = self.root / "memory"
        self.mem.mkdir()
        os.environ["MEMDAG_BRIDGE_MEMORY_DIR"] = str(self.mem)
        # G-4: build_telemetry() otherwise reads the operator's LIVE
        # ~/.claude/episodic/sessions.db and globs ~/.claude/consolidation --
        # pin both away from this machine's real brain for the whole suite.
        os.environ["MEMSOM_CONSOLIDATION_DIR"] = str(self.root / "consolidation")
        os.environ["MEMSOM_EPISODIC_DB"] = str(self.root / "no_sessions.db")
        for name, text in MEMORY_FILES.items():
            (self.mem / name).write_text(text, encoding="utf-8")
        (self.mem / "MEMORY.md").write_text(INDEX, encoding="utf-8")

        self._run_cli("init")
        # Noise nodes via the real ingest-text CLI surface, across different
        # channels. Their --ref values are NOT 'memory:'-namespaced (that
        # namespace is reserved for the bridge importer --
        # integrity.ingest.enforce_source_ref_namespace refuses it from any
        # other caller), so they must never surface in telemetry's counts.
        self._run_cli("ingest-text", "noise one", "--channel", "user", "--ref", "note:1")
        self._run_cli("ingest-text", "noise two", "--channel", "agent-derived", "--ref", "note:2")
        self._run_cli("ingest-text", "noise three", "--channel", "external", "--ref", "note:3")

        self.conn = memsom.get_connection()
        result = br.bridge_render(self.conn, self.mem)
        self.assertTrue(result["ok"], result)

    def tearDown(self):
        self.conn.close()
        for key in ("MEMDAG_HOME", "MEMDAG_DB", "MEMDAG_EMBED_BACKEND",
                    "CLAUDE_MD_PATH", "MEMDAG_BRIDGE_MEMORY_DIR",
                    "MEMSOM_CONSOLIDATION_DIR", "MEMSOM_EPISODIC_DB"):
            os.environ.pop(key, None)
        self.tmp.cleanup()

    def _run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memsom_cli.main(list(argv))
        return buf.getvalue()


class TestBuildTelemetry(Base):
    def test_key_set_is_frozen(self):
        from memsom.interface import telemetry
        data = telemetry.build_telemetry(memory_dir=self.mem, conn=self.conn)
        self.assertEqual(set(data.keys()), FROZEN_TELEMETRY_KEYS)
        # G-4: pinned away from the operator's real ~/.claude by setUp -- this
        # is the fixture that goes RED on a dev box if that pin is removed.
        self.assertIsNone(data["sessions"])
        self.assertIsNone(data["last_consolidation"])

    def test_totals_count_matches_memory_dir_only(self):
        from memsom.interface import telemetry
        data = telemetry.build_telemetry(memory_dir=self.mem, conn=self.conn)
        # 2 memory-dir files imported by bridge-render; the 3 ingest-text
        # noise nodes (source_ref='note:N', not 'memory:%') must not count.
        self.assertEqual(data["totals"]["total"], 2)
        self.assertEqual(data["totals"]["total"], data["tier"]["hot"] + data["tier"]["cold"])

    def test_zero_arg_call_matches_explicit_conn(self):
        # Keyword-compatible with the live zero-arg call: memory_dir resolves
        # via default_memory_dir() (MEMDAG_BRIDGE_MEMORY_DIR, pinned above),
        # conn opens its own read-only connection to the same MEMDAG_DB.
        from memsom.interface import telemetry
        data = telemetry.build_telemetry()
        self.assertEqual(data["totals"]["total"], 2)


class TestGraph(Base):
    def test_wikilink_becomes_graph_link(self):
        from memsom.interface import telemetry
        (self.mem / "project_widget.md").write_text(LINKED_PROJECT, encoding="utf-8")
        self.assertTrue(br.bridge_render(self.conn, self.mem)["ok"])
        data = telemetry.build_telemetry(memory_dir=self.mem, conn=self.conn)
        self.assertIn("link", {l["kind"] for l in data["graph"]["links"]})


class TestLoadWeights(Base):
    def test_row_keys_are_frozen(self):
        from memsom.interface import telemetry
        rows = telemetry.load_weights(conn=self.conn)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(set(row.keys()), FROZEN_WEIGHT_ROW_KEYS)

    def test_stems_match_memory_dir_files_only(self):
        from memsom.interface import telemetry
        rows = telemetry.load_weights(conn=self.conn)
        stems = {r["stem"] for r in rows}
        self.assertEqual(stems, {"user_editor", "project_widget"})

    def test_own_connection_when_none_given(self):
        from memsom.interface import telemetry
        rows = telemetry.load_weights()
        self.assertEqual(len(rows), 2)

    def test_body_pin_line_does_not_mark_pinned(self):
        from memsom.interface import telemetry
        (self.mem / "project_pinbody.md").write_text(
            "---\nname: P\ndescription: d\ntype: project\n---\n"
            "notes:\npin: yes please\n", encoding="utf-8")
        self.assertTrue(br.bridge_render(self.conn, self.mem)["ok"])
        rows = {r["stem"]: r for r in telemetry.load_weights(conn=self.conn)}
        self.assertEqual(rows["project_pinbody"]["pinned"], 0)


class TestDefaultMemoryDir(Base):
    def test_reexport_is_kernel_paths_function(self):
        from memsom.interface import telemetry
        from memsom.kernel.paths import default_memory_dir as kernel_default
        self.assertIs(telemetry.default_memory_dir, kernel_default)

    def test_resolves_pinned_dir(self):
        from memsom.interface import telemetry
        self.assertEqual(telemetry.default_memory_dir().resolve(), self.mem.resolve())


class TestDashboardShim(Base):
    def test_import_warns_deprecation(self):
        import sys
        sys.modules.pop("memsom.interface.dashboard", None)
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import memsom.interface.dashboard as dashboard  # noqa: F401
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_exposes_same_function_objects(self):
        import warnings
        from memsom.interface import telemetry
        # dashboard is the deprecated shim under test elsewhere
        # (test_import_warns_deprecation); this test only checks identity of
        # the re-exported objects, so the deprecation warning its import
        # fires is expected and must not fail the run under -W error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from memsom.interface import dashboard
        self.assertIs(dashboard.build_telemetry, telemetry.build_telemetry)
        self.assertIs(dashboard.load_weights, telemetry.load_weights)
        self.assertIs(dashboard.default_memory_dir, telemetry.default_memory_dir)

    def test_no_bare_sqlite3_connect_in_source(self):
        # Rule 4's guard, restated at the source level: neither module may
        # hand-roll its own connection.
        import memsom.interface.telemetry as telemetry_mod
        import memsom.interface.dashboard as dashboard_mod
        for mod in (telemetry_mod, dashboard_mod):
            src = Path(inspect.getfile(mod)).read_text(encoding="utf-8")
            self.assertNotIn("sqlite3.connect", src, f"{mod.__name__} hand-rolls a connection")


class TestTelemetryFeature(Base):
    def test_active_with_full_key_set(self):
        status = memsom_features.all_statuses(self.conn)["telemetry"]
        self.assertIn(status["state"], ("active", "degraded"))

    def test_cli_features_json_reports_telemetry(self):
        out = self._run_cli("features", "--json")
        self.assertIn('"telemetry"', out)
        import json
        payload = json.loads(out)
        self.assertIn(payload["telemetry"]["state"], ("active", "degraded"))

    def test_conn_none_reports_active_module_present(self):
        status = memsom_features.all_statuses(None)["telemetry"]
        self.assertEqual(status["state"], "active")

    def test_probe_never_raises_through_safe(self):
        # Invert-the-gate control: a probe that DOES raise must come back as
        # 'error' via _safe, never propagate and never silently become 'active'.
        def _boom(conn):
            raise RuntimeError("synthetic failure")
        status = memsom_features._safe("telemetry", _boom)
        self.assertEqual(status["state"], "error")


class TestMissingDB(unittest.TestCase):
    def test_load_weights_raises_when_db_absent(self):
        from memsom.interface import telemetry
        with tempfile.TemporaryDirectory() as d:
            os.environ["MEMDAG_HOME"] = d
            os.environ["MEMDAG_DB"] = str(Path(d) / "nope.db")
            try:
                with self.assertRaises(FileNotFoundError):
                    telemetry.load_weights()
            finally:
                os.environ.pop("MEMDAG_HOME", None)
                os.environ.pop("MEMDAG_DB", None)


class TestPreBridgeRenderStore(unittest.TestCase):
    def test_load_weights_errors_cleanly_without_forget_columns(self):
        from memsom.interface import telemetry
        with tempfile.TemporaryDirectory() as d:
            os.environ["MEMDAG_HOME"] = d
            os.environ["MEMDAG_DB"] = str(Path(d) / "fresh.db")
            try:
                conn = memsom.get_connection()
                memsom_cli.migrate_all(conn)
                conn.commit()
                conn.close()
                with self.assertRaises(RuntimeError) as ctx:
                    telemetry.load_weights()
                self.assertIn("bridge-render", str(ctx.exception))
            finally:
                os.environ.pop("MEMDAG_HOME", None)
                os.environ.pop("MEMDAG_DB", None)


if __name__ == "__main__":
    unittest.main()
