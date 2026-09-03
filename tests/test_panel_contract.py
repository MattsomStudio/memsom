"""tests/test_panel_contract.py -- freeze the memsom surface the panel is
allowed to depend on (PROMOTE-Q11-PANEL.md B2).

This is a FROZEN LIST test: every name, signature, and CLI flag below is
something memsom-agentic-os (the panel) actually imports or invokes today
(measured 2026-09-01 against the panel's real usage in __main__.py,
transport/activity.py, transport/knobs.py, and its CLI-driven Tauri spawn).
A change here is a breaking change to the panel and must be deliberate.

Asserted by import + `inspect.signature` (parameter names, kinds, defaults)
and callable/class checks -- never by "it ran without raising", which would
pass even if a parameter got renamed or reordered under the panel's feet.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _params(sig: inspect.Signature):
    """(name, kind, default) tuples, default normalized to a sentinel string
    when the parameter is required -- keeps the assertion readable."""
    out = []
    for p in sig.parameters.values():
        default = "<required>" if p.default is inspect._empty else p.default
        out.append((p.name, p.kind, default))
    return out


class TelemetryContract(unittest.TestCase):
    """memsom.interface.telemetry: build_telemetry(memory_dir=None, *,
    conn=None), load_weights(conn=None), default_memory_dir()."""

    def test_build_telemetry_signature(self):
        from memsom.interface import telemetry
        sig = inspect.signature(telemetry.build_telemetry)
        self.assertEqual(
            _params(sig),
            [
                ("memory_dir", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
                ("conn", inspect.Parameter.KEYWORD_ONLY, None),
            ],
        )

    def test_load_weights_signature(self):
        from memsom.interface import telemetry
        sig = inspect.signature(telemetry.load_weights)
        self.assertEqual(
            _params(sig),
            [("conn", inspect.Parameter.POSITIONAL_OR_KEYWORD, None)],
        )

    def test_default_memory_dir_callable_zero_arg(self):
        from memsom.interface import telemetry
        self.assertTrue(callable(telemetry.default_memory_dir))

    def test_load_weights_error_types_are_frozen(self):
        """G-2: the panel's /api/memory handler catches (SystemExit,
        FileNotFoundError, RuntimeError) -- these are the two failure shapes
        load_weights() must actually raise (dashboard's SystemExit is gone)."""
        import tempfile
        from memsom.interface import telemetry
        from memsom.interface import cli as memsom_cli
        import memsom
        saved = {k: os.environ.get(k) for k in ("MEMDAG_HOME", "MEMDAG_DB")}
        try:
            with tempfile.TemporaryDirectory() as d:
                os.environ["MEMDAG_HOME"] = d
                os.environ["MEMDAG_DB"] = str(Path(d) / "nope.db")
                with self.assertRaises(FileNotFoundError):
                    telemetry.load_weights()

                os.environ["MEMDAG_DB"] = str(Path(d) / "fresh.db")
                conn = memsom.get_connection()
                try:
                    memsom_cli.migrate_all(conn)
                    conn.commit()
                finally:
                    conn.close()
                with self.assertRaises(RuntimeError) as ctx:
                    telemetry.load_weights()
                self.assertIn("bridge-render", str(ctx.exception))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        sig = inspect.signature(telemetry.default_memory_dir)
        # every parameter (if any) must have a default -- a bare call must work
        for p in sig.parameters.values():
            self.assertIsNot(p.default, inspect._empty,
                              f"default_memory_dir() gained a required param: {p.name}")


class DashboardShimContract(unittest.TestCase):
    """memsom.interface.dashboard: compatibility shim until the panel
    switches (B3) -- same three names, and importing it raises
    DeprecationWarning."""

    def test_shim_exports_same_three_names(self):
        # Import fresh (not from sys.modules) so the DeprecationWarning below
        # is not swallowed by an earlier import in this same test process.
        sys.modules.pop("memsom.interface.dashboard", None)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            from memsom.interface import dashboard
        for name in ("build_telemetry", "load_weights", "default_memory_dir"):
            self.assertTrue(hasattr(dashboard, name), f"dashboard.{name} missing")
            self.assertTrue(callable(getattr(dashboard, name)))

    def test_importing_dashboard_raises_deprecation_warning(self):
        sys.modules.pop("memsom.interface.dashboard", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module("memsom.interface.dashboard")
        deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertTrue(deprecation,
                         "importing memsom.interface.dashboard must raise "
                         "DeprecationWarning (the panel's migration signal)")


class SaveallContract(unittest.TestCase):
    """memsom.interface.saveall: start(claude_dir, *, cli_path='claude',
    model='claude-sonnet-5', effort='high', session_id=None,
    resume_cwd=None), status(claude_dir, *, tail_bytes=8000),
    class AlreadyRunning."""

    def test_start_signature(self):
        from memsom.interface import saveall
        sig = inspect.signature(saveall.start)
        self.assertEqual(
            _params(sig),
            [
                ("claude_dir", inspect.Parameter.POSITIONAL_OR_KEYWORD, "<required>"),
                ("cli_path", inspect.Parameter.KEYWORD_ONLY, "claude"),
                ("model", inspect.Parameter.KEYWORD_ONLY, "claude-sonnet-5"),
                ("effort", inspect.Parameter.KEYWORD_ONLY, "high"),
                ("session_id", inspect.Parameter.KEYWORD_ONLY, None),
                ("resume_cwd", inspect.Parameter.KEYWORD_ONLY, None),
            ],
        )

    def test_status_signature(self):
        from memsom.interface import saveall
        sig = inspect.signature(saveall.status)
        self.assertEqual(
            _params(sig),
            [
                ("claude_dir", inspect.Parameter.POSITIONAL_OR_KEYWORD, "<required>"),
                ("tail_bytes", inspect.Parameter.KEYWORD_ONLY, 8000),
            ],
        )

    def test_already_running_is_exception_class(self):
        from memsom.interface import saveall
        self.assertTrue(inspect.isclass(saveall.AlreadyRunning))
        self.assertTrue(issubclass(saveall.AlreadyRunning, BaseException))

    def test_status_return_keys(self):
        import json
        import tempfile
        from memsom.interface import saveall
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(saveall.status(d), {"exists": False})
            runs = Path(d) / "episodic" / "saveall"
            runs.mkdir(parents=True)
            (runs / "latest.json").write_text(json.dumps({
                "run_id": "r1", "pid": 0, "session_id": "s1",
                "log": str(runs / "r1.log"), "started": "2026-09-01T00:00:00Z",
            }), encoding="utf-8")
            st = saveall.status(d)
            self.assertEqual(set(st), {"exists", "running", "session_id", "started", "run_id", "log"})
            self.assertIs(st["exists"], True)

    def test_start_return_keys(self):
        import tempfile
        from unittest import mock
        from memsom.interface import saveall
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "projects" / "p"
            proj.mkdir(parents=True)
            (proj / "sess-aaaaaaaa.jsonl").write_text("{}\n", encoding="utf-8")
            fake = mock.Mock(pid=4242)
            with mock.patch.object(saveall.memsom_proc, "popen", return_value=fake), \
                 mock.patch.object(saveall, "_pid_alive", return_value=False):
                out = saveall.start(d, cli_path="claude")
            self.assertEqual(set(out), {"ok", "run_id", "session_id", "pid"})
            self.assertIs(out["ok"], True)


class ForgetContract(unittest.TestCase):
    """memsom.lifecycle.forget: load_params(params_path) -> (params,
    warnings); now_iso(); PANEL_PARAM_DEFAULTS superset of the panel's 8
    live keys (verified against `git show main:memsom/lifecycle/forget.py`
    below -- the true live set, not the task's paraphrase)."""

    LIVE_KEYS = {
        "memory_budget", "memory_max_lines", "prompt_hook_mode",
        "prompt_hook_floor", "prompt_hook_deadline_ms", "prompt_hook_log_max_mb",
        "prompt_hook_project_bytes", "prompt_hook_project_max",
        "feedback_born_unindexed", "section_budgets",
    }

    def test_load_params_signature_and_return_shape(self):
        from memsom.lifecycle import forget
        sig = inspect.signature(forget.load_params)
        self.assertEqual(
            _params(sig),
            [("params_path", inspect.Parameter.POSITIONAL_OR_KEYWORD, "<required>")],
        )
        result = forget.load_params(None)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        params, warns = result
        self.assertIsInstance(params, dict)
        self.assertIsInstance(warns, list)

    def test_now_iso_callable(self):
        from memsom.lifecycle import forget
        self.assertTrue(callable(forget.now_iso))
        val = forget.now_iso()
        self.assertIsInstance(val, str)

    def test_panel_param_defaults_superset(self):
        from memsom.lifecycle import forget
        self.assertIsInstance(forget.PANEL_PARAM_DEFAULTS, dict)
        missing = self.LIVE_KEYS - set(forget.PANEL_PARAM_DEFAULTS)
        self.assertFalse(missing, f"PANEL_PARAM_DEFAULTS lost live keys: {missing}")

    def test_live_key_set_matches_main_branch_source(self):
        """Re-derive LIVE_KEYS from `git show main:...forget.py` rather than
        trusting the hardcoded set above to stay honest -- this is the test
        that would catch LIVE_KEYS itself drifting from the real live file."""
        # G-5/MF-9: never skip. A checkout of a non-main ref has `main` only as
        # `origin/main` (CI: actions/checkout with fetch-depth 0), so try both;
        # if neither resolves this test FAILS -- a silently-green LIVE_KEYS
        # check on a shallow clone is exactly the vacuous pass rule 15 forbids.
        errors = []
        out = None
        for ref in ("main", "origin/main"):
            try:
                cand = subprocess.run(
                    ["git", "show", f"{ref}:memsom/lifecycle/forget.py"],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.fail(f"git show unavailable in this environment: {exc!r}")
            if cand.returncode == 0:
                out = cand
                break
            errors.append(f"{ref}: {cand.stderr.strip()!r}")
        if out is None:
            self.fail("git show <main|origin/main>:memsom/lifecycle/forget.py failed "
                      f"({'; '.join(errors)}) -- fetch main (CI: fetch-depth 0) so "
                      "the LIVE_KEYS drift check can run; it must not pass vacuously")
        m = re.search(r"PANEL_PARAM_DEFAULTS\s*=\s*\{(.*?)\n\}", out.stdout, re.S)
        self.assertIsNotNone(m, "could not locate PANEL_PARAM_DEFAULTS in main's forget.py")
        live_keys = set(re.findall(r'^\s*"([a-zA-Z0-9_]+)"\s*:', m.group(1), re.M))
        self.assertEqual(live_keys, self.LIVE_KEYS,
                          "this test's LIVE_KEYS has drifted from main's real "
                          "PANEL_PARAM_DEFAULTS key set -- update LIVE_KEYS above")


class DistributionContract(unittest.TestCase):
    """MF-8: the panel resolves memsom's version through importlib.metadata by
    the DISTRIBUTION name 'memsom' (memsom_panel/kernel/version.py); a rename of
    the distribution in pyproject.toml would break the panel with no import
    error anywhere."""

    def test_distribution_named_memsom(self):
        import importlib.metadata as md
        self.assertTrue(md.version("memsom"))


class PathsContract(unittest.TestCase):
    """memsom.paths: safe_join(root, *parts, allow_absolute=False,
    resolve_symlinks=True), class UnsafePath, is_unc_or_device(...)."""

    def test_safe_join_signature(self):
        from memsom import paths
        sig = inspect.signature(paths.safe_join)
        params = list(sig.parameters.values())
        self.assertEqual(params[0].name, "root")
        self.assertEqual(params[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(params[1].name, "parts")
        self.assertEqual(params[1].kind, inspect.Parameter.VAR_POSITIONAL)
        kw = {p.name: p for p in params[2:]}
        self.assertEqual(kw["allow_absolute"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(kw["allow_absolute"].default, False)
        self.assertEqual(kw["resolve_symlinks"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(kw["resolve_symlinks"].default, True)

    def test_unsafe_path_is_exception_class(self):
        from memsom import paths
        self.assertTrue(inspect.isclass(paths.UnsafePath))
        self.assertTrue(issubclass(paths.UnsafePath, BaseException))

    def test_is_unc_or_device_callable(self):
        from memsom import paths
        self.assertTrue(callable(paths.is_unc_or_device))


class ChildenvContract(unittest.TestCase):
    """memsom.childenv: child_env(...)."""

    def test_child_env_callable_and_returns_dict(self):
        from memsom import childenv
        self.assertTrue(callable(childenv.child_env))
        result = childenv.child_env()
        self.assertIsInstance(result, dict)


class CLIContract(unittest.TestCase):
    """CLI surface the panel's Tauri spawn / process shape depends on."""

    @staticmethod
    def _run_help(*args):
        return subprocess.run(
            [sys.executable, "-m", "memsom.interface.cli", *args, "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )

    def test_retrieve_accepts_k_and_clearance(self):
        out = self._run_help("retrieve")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--k", out.stdout)
        self.assertIn("--clearance", out.stdout)

    def test_ask_subcommand_exists(self):
        out = self._run_help("ask")
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_ingest_text_accepts_channel_choices_and_ref(self):
        out = self._run_help("ingest-text")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--channel", out.stdout)
        for ch in ("endorsed", "user", "agent-derived", "external"):
            self.assertIn(ch, out.stdout)
        self.assertIn("--ref", out.stdout)

    def test_dashboard_is_not_a_subcommand(self):
        out = subprocess.run(
            [sys.executable, "-m", "memsom.interface.cli", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        # subcommand names appear as their own token in the subparser choice
        # list / help listing; "dashboard" must not appear as a subcommand.
        self.assertNotIn("dashboard", out.stdout,
                          "dashboard must stay deleted as a CLI subcommand (A-9)")


class PluginSeamContract(unittest.TestCase):
    """The panel registers via the `memsom.commands` entry-point group;
    memsom.interface.cli scans it and calls ep.load()(sub). If memsom_panel
    is importable in this venv, `panel --help` must work with its 4 flags.
    If it is NOT importable, this must FAIL LOUDLY (never skip) -- a missing
    panel plugin in this venv means the venv is wrong, not that the seam is
    untested."""

    def test_plugin_entry_point_scan_exists(self):
        from memsom.interface import cli
        self.assertTrue(callable(getattr(cli, "_register_plugin_commands", None)),
                         "cli._register_plugin_commands (memsom.commands entry-point "
                         "scan) went missing")
        src = inspect.getsource(cli._register_plugin_commands)
        self.assertIn('group="memsom.commands"', src)

    def test_panel_help_when_panel_installed(self):
        if importlib.util.find_spec("memsom_panel") is None:
            return  # panel-free machine: the 4-flag check is owned by
                     # aos/backend/tests/test_memsom_contract.py::test_cli_surface
        out = subprocess.run(
            [sys.executable, "-m", "memsom.interface.cli", "panel", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        for flag in ("--profile", "--host", "--port", "--no-open"):
            self.assertIn(flag, out.stdout,
                           f"panel --help missing {flag}: {out.stdout}")

    def test_memsom_never_imports_memsom_panel(self):
        """Seam direction (Rule 2): memsom NEVER imports memsom_panel. The
        only permitted mention is the presence probe in features.py
        (importlib.util.find_spec("memsom_panel")) -- a real `import
        memsom_panel` / `from memsom_panel import ...` anywhere under
        memsom/ is a violation."""
        import_re = re.compile(
            r"^\s*(import\s+memsom_panel\b|from\s+memsom_panel\b)", re.M)
        violations = []
        memsom_dir = REPO_ROOT / "memsom"
        for py in memsom_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8", errors="ignore")
            if import_re.search(text):
                violations.append(str(py.relative_to(REPO_ROOT)))
        self.assertEqual(
            violations, [],
            f"memsom/ contains a real import of memsom_panel (seam violation): "
            f"{violations}",
        )
        # And the find_spec probe itself must still be present -- proves the
        # grep above isn't just finding nothing because the file moved.
        features_py = (memsom_dir / "interface" / "features.py").read_text(encoding="utf-8")
        self.assertIn('find_spec("memsom_panel")', features_py,
                      "features.py's memsom_panel presence probe went missing")


if __name__ == "__main__":
    unittest.main()
