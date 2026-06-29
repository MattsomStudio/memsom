#!/usr/bin/env python3
"""Phase 6 — bootstrap decision seams (OS, install plan, exe path, opt-in, Ollama).

The IO flow (real installers) isn't unit-tested; the decision logic is.

Run:  python -m unittest -v test_bootstrap
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import bootstrap


class TestPythonGate(unittest.TestCase):
    def test_python_version_gate(self):
        self.assertFalse(bootstrap.python_ok((3, 11, 0)))
        self.assertTrue(bootstrap.python_ok((3, 12, 0)))
        self.assertTrue(bootstrap.python_ok((3, 14, 1)))


class TestOsDetection(unittest.TestCase):
    def test_install_plan_per_os(self):
        self.assertEqual(bootstrap.ollama_install_plan("Darwin")["primary"][:2], ["brew", "install"])
        self.assertIn("winget", bootstrap.ollama_install_plan("Windows")["primary"])
        self.assertIn("curl -fsSL https://ollama.com/install.sh | sh",
                      bootstrap.ollama_install_plan("Linux")["primary"])


class TestExePath(unittest.TestCase):
    # Compare via Path.name / .parts so the test is host-separator-agnostic
    # (pathlib renders with the HOST separator; only the bin-dir + suffix logic
    # is os_name-driven, which is exactly the cross-OS behaviour we care about).
    def test_records_absolute_exe_path(self):
        win = bootstrap.resolve_exe_path("venv", "~/.memdag", os_name="Windows")
        self.assertEqual(win.name, "memdag-mcp.exe")
        self.assertIn("Scripts", win.parts)
        mac = bootstrap.resolve_exe_path("venv", "~/.memdag", os_name="Darwin")
        self.assertEqual(mac.name, "memdag-mcp")
        self.assertIn("bin", mac.parts)
        self.assertTrue(mac.is_absolute())  # expanduser -> absolute on any host

    def test_pipx_path(self):
        p = bootstrap.resolve_exe_path("pipx", "~/.memdag", os_name="Linux", home="/home/u")
        self.assertEqual(p.name, "memdag-mcp")
        self.assertEqual(p.parts[-2], "bin")
        for part in ("pipx", "venvs", "memdag"):
            self.assertIn(part, p.parts)


class TestOptIn(unittest.TestCase):
    def test_opt_in_default_no(self):
        for ans in ("", "n", "no", " N "):
            self.assertFalse(bootstrap.should_ingest(ans))
        for ans in ("y", "yes", " Y "):
            self.assertTrue(bootstrap.should_ingest(ans))


class TestOllamaGraceful(unittest.TestCase):
    def test_ollama_failure_continues(self):
        # Installer raises -> install_ollama must return a status, not propagate.
        def boom(*a, **k):
            raise OSError("no installer here")

        status = bootstrap.install_ollama(
            bootstrap.ollama_install_plan("Linux"),
            runner=boom, which=lambda _n: None)
        self.assertFalse(status["ok"])
        self.assertIn("manual", status)

    def test_ollama_missing_prereq_reports_manual(self):
        # macOS plan needs brew; if brew absent, report manual without running.
        status = bootstrap.install_ollama(
            bootstrap.ollama_install_plan("Darwin"),
            runner=lambda *a, **k: self.fail("runner should not be called"),
            which=lambda _n: None)
        self.assertFalse(status["ok"])
        self.assertIn("brew", status["reason"])


class TestWireFailureAborts(unittest.TestCase):
    """BOOTSTRAP-1 / CFG-PRINT-SOFTFAIL-1 user-visible symptom: a wire-config that
    exits nonzero on a real run must abort main() (return 1) and must NOT print
    '=== done ===' — i.e. an unconfigured client is never reported as success."""

    def test_nonzero_wire_aborts_and_no_done(self):
        with mock.patch.object(bootstrap, "install_memdag",
                               return_value=("venv", Path("/abs/memdag-mcp"), Path("/abs/memdag"))), \
             mock.patch.object(bootstrap, "install_ollama", return_value={"ok": True}), \
             mock.patch.object(bootstrap, "run_init", return_value="/abs/data/memdag.db"), \
             mock.patch.object(bootstrap.subprocess, "run",
                               return_value=mock.Mock(returncode=1)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = bootstrap.main(["--no-ingest"])
        out = buf.getvalue()
        self.assertEqual(rc, 1)
        self.assertNotIn("=== done ===", out)

    def test_zero_wire_completes(self):
        # Control: a clean (rc 0) wire reaches '=== done ===' and returns 0.
        with mock.patch.object(bootstrap, "install_memdag",
                               return_value=("venv", Path("/abs/memdag-mcp"), Path("/abs/memdag"))), \
             mock.patch.object(bootstrap, "install_ollama", return_value={"ok": True}), \
             mock.patch.object(bootstrap, "run_init", return_value="/abs/data/memdag.db"), \
             mock.patch.object(bootstrap.subprocess, "run",
                               return_value=mock.Mock(returncode=0)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = bootstrap.main(["--no-ingest"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("=== done ===", out)


class TestMemoryLoopStepSoftFails(unittest.TestCase):
    """Step [6/6] (wire-claude) is a SOFT step: if it fails, the MCP server still
    works, so main() warns but completes (rc 0, '=== done ===' printed)."""

    def test_wire_claude_failure_warns_but_completes(self):
        # call order: subprocess.run #1 = wire-config (ok), #2 = wire-claude (fails)
        with mock.patch.object(bootstrap, "install_memdag",
                               return_value=("venv", Path("/abs/memdag-mcp"), Path("/abs/memdag"))), \
             mock.patch.object(bootstrap, "install_ollama", return_value={"ok": True}), \
             mock.patch.object(bootstrap, "run_init", return_value="/abs/data/memdag.db"), \
             mock.patch.object(bootstrap.subprocess, "run",
                               side_effect=[mock.Mock(returncode=0), mock.Mock(returncode=1)]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = bootstrap.main(["--no-ingest"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)                                  # soft fail, not fatal
        self.assertIn("[6/6]", out)
        self.assertIn("memory-loop wiring incomplete", out)
        self.assertIn("=== done ===", out)


if __name__ == "__main__":
    unittest.main()
