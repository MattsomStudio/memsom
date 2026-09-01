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
                    "CLAUDE_MD_PATH", "MEMDAG_BRIDGE_MEMORY_DIR"):
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


if __name__ == "__main__":
    unittest.main()
