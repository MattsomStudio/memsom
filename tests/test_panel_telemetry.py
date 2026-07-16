"""Tests for memsom.interface.telemetry — the panel system-telemetry sampler.

Run:  python -m pytest tests/test_panel_telemetry.py -q
"""

import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from memsom.interface import telemetry as tel

try:
    import psutil
except ImportError:
    psutil = None


def _fake_run(returncode=0, stdout="", raise_exc=None):
    def run(cmd, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    return run


class _FakeResponse:
    """Minimal context-manager stand-in for urllib's addinfourl."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


SYNCTHING_XML = """<configuration version="37">
  <gui enabled="true">
    <apikey>{key}</apikey>
  </gui>
</configuration>
"""


class TestGpu(unittest.TestCase):
    def test_happy_path_parses_name_and_memory(self):
        calls = []

        def run(cmd, **kwargs):
            calls.append(cmd)
            if "--query-gpu=name,memory.used,memory.total" in cmd[1]:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="NVIDIA GeForce RTX 0000, 2048, 12288\n", stderr="")
            return subprocess.CompletedProcess(
                cmd, 0, stdout="1234, proc.exe, 512\n", stderr="")

        st = tel.SystemTelemetry({}, run=run)
        gpu = st.sample()["gpu"]
        self.assertTrue(gpu["available"])
        self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 0000")
        self.assertEqual(gpu["vram_used_mb"], 2048)
        self.assertEqual(gpu["vram_total_mb"], 12288)
        self.assertEqual(gpu["procs"], [{"pid": 1234, "name": "proc.exe", "used_mb": 512}])
        self.assertEqual(len(calls), 2)

    def test_missing_binary_reports_unavailable(self):
        st = tel.SystemTelemetry({}, run=_fake_run(raise_exc=FileNotFoundError()))
        gpu = st.sample()["gpu"]
        self.assertEqual(gpu, {"available": False})

    def test_timeout_reports_unavailable(self):
        exc = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
        st = tel.SystemTelemetry({}, run=_fake_run(raise_exc=exc))
        gpu = st.sample()["gpu"]
        self.assertEqual(gpu, {"available": False})

    def test_nonzero_exit_reports_unavailable(self):
        st = tel.SystemTelemetry({}, run=_fake_run(returncode=1, stdout=""))
        gpu = st.sample()["gpu"]
        self.assertEqual(gpu, {"available": False})


class TestProbes(unittest.TestCase):
    def test_open_and_closed_port(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        open_port = listener.getsockname()[1]

        closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        closed.bind(("127.0.0.1", 0))
        closed_port = closed.getsockname()[1]
        closed.close()  # nothing listens here now -> connection refused

        try:
            profile = {
                "probe_targets": [
                    {"name": "open-svc", "host": "127.0.0.1", "port": open_port},
                    {"name": "closed-svc", "host": "127.0.0.1", "port": closed_port},
                ],
                "probe_timeout_s": 1.0,
            }
            st = tel.SystemTelemetry(profile)
            probes = st.sample()["probes"]
        finally:
            listener.close()

        by_name = {p["name"]: p for p in probes}
        self.assertTrue(by_name["open-svc"]["ok"])
        self.assertFalse(by_name["closed-svc"]["ok"])
        for p in probes:
            self.assertIsInstance(p["ms"], float)

    def test_no_targets_returns_empty_list(self):
        st = tel.SystemTelemetry({})
        self.assertEqual(st.sample()["probes"], [])


class TestSyncthing(unittest.TestCase):
    def test_reachable_and_key_never_leaked(self):
        key = "s3cr3t-api-key-placeholder"
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "config.xml"
            cfg_path.write_text(SYNCTHING_XML.format(key=key), encoding="utf-8")

            seen_headers = {}

            def fake_urlopen(req, timeout=None):
                seen_headers.update(req.headers)
                return _FakeResponse(b'{"connections": {"peer-a": {"connected": true}}}')

            profile = {"syncthing": {"base_url": "http://192.0.2.1:8384",
                                      "config_xml": str(cfg_path)}}
            st = tel.SystemTelemetry(profile, urlopen=fake_urlopen)
            result = st.sample()["syncthing"]

        self.assertTrue(result["reachable"])
        self.assertIn("connections", result)
        blob = str(result)
        self.assertNotIn(key, blob, "syncthing api key leaked into the returned payload")
        # request itself must have carried it (case-folded header lookup)
        self.assertTrue(any(v == key for v in seen_headers.values()))

    def test_connection_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "config.xml"
            cfg_path.write_text(SYNCTHING_XML.format(key="whatever-key"), encoding="utf-8")

            def fake_urlopen(req, timeout=None):
                raise ConnectionRefusedError("refused")

            profile = {"syncthing": {"base_url": "http://192.0.2.1:8384",
                                      "config_xml": str(cfg_path)}}
            st = tel.SystemTelemetry(profile, urlopen=fake_urlopen)
            result = st.sample()["syncthing"]

        self.assertFalse(result["reachable"])
        self.assertIn("error", result)
        self.assertNotIn("whatever-key", str(result))

    def test_missing_config_file(self):
        profile = {"syncthing": {"base_url": "http://192.0.2.1:8384",
                                  "config_xml": str(Path(tempfile.gettempdir()) / "does-not-exist.xml")}}
        st = tel.SystemTelemetry(profile)
        result = st.sample()["syncthing"]
        self.assertFalse(result["reachable"])
        self.assertIn("error", result)

    def test_not_configured(self):
        st = tel.SystemTelemetry({})
        result = st.sample()["syncthing"]
        self.assertEqual(result, {"reachable": False, "error": "not configured"})


class TestDiskDelta(unittest.TestCase):
    def test_first_sample_null_delta_then_correct_delta(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"x" * 100)

            st = tel.SystemTelemetry({"disk_dirs": [d]})
            first = st.sample()["disks"][0]
            self.assertEqual(first["path"], d)
            self.assertEqual(first["bytes"], 100)
            self.assertIsNone(first["delta_bytes"])

            f.write_bytes(b"x" * 260)
            second = st.sample()["disks"][0]
            self.assertEqual(second["bytes"], 260)
            self.assertEqual(second["delta_bytes"], 160)

    def test_skips_unreadable_entries_without_raising(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "real.bin").write_bytes(b"y" * 10)
            st = tel.SystemTelemetry({"disk_dirs": [d, str(Path(d) / "nonexistent-subdir")]})
            disks = st.sample()["disks"]
            self.assertEqual(len(disks), 2)
            self.assertEqual(disks[0]["bytes"], 10)
            self.assertEqual(disks[1]["bytes"], 0)  # nonexistent dir -> zero, not a crash


class TestSectionIsolation(unittest.TestCase):
    def test_gpu_dependency_raising_does_not_kill_other_sections(self):
        def angry_run(cmd, **kwargs):
            raise RuntimeError("unexpected failure deep in the run() shim")

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "f.bin").write_bytes(b"z" * 5)
            profile = {
                "disk_dirs": [d],
                "probe_targets": [{"name": "n", "host": "192.0.2.5", "port": 1},
                                   ],
                "probe_timeout_s": 0.2,
            }
            st = tel.SystemTelemetry(profile, run=angry_run)
            result = st.sample()

        self.assertEqual(result["gpu"], {"available": False})
        self.assertIn("ts", result)
        self.assertEqual(len(result["disks"]), 1)
        self.assertEqual(result["disks"][0]["bytes"], 5)
        self.assertEqual(len(result["probes"]), 1)

    def test_syncthing_dependency_raising_does_not_kill_other_sections(self):
        def angry_urlopen(req, timeout=None):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "config.xml"
            cfg_path.write_text(SYNCTHING_XML.format(key="k"), encoding="utf-8")
            disk_dir = Path(d) / "watched"
            disk_dir.mkdir()
            (disk_dir / "f.bin").write_bytes(b"q" * 7)
            profile = {
                "disk_dirs": [str(disk_dir)],
                "syncthing": {"base_url": "http://192.0.2.1:8384", "config_xml": str(cfg_path)},
            }
            st = tel.SystemTelemetry(profile, urlopen=angry_urlopen)
            result = st.sample()

        self.assertEqual(result["syncthing"], {"reachable": False, "error": "request failed"})
        self.assertEqual(result["disks"][0]["bytes"], 7)
        self.assertIn("ts", result)


@unittest.skipUnless(psutil is not None, "psutil not installed")
class TestPsutilBacked(unittest.TestCase):
    def test_physical_ram_present_when_psutil_available(self):
        st = tel.SystemTelemetry({})
        ram = st.sample()["ram"]
        physical = ram["physical"]
        self.assertNotIn("error", physical)
        self.assertGreater(physical["total"], 0)
        self.assertGreaterEqual(physical["available"], 0)
        self.assertGreaterEqual(physical["used_pct"], 0.0)

    def test_top_procs_shape_and_top_n(self):
        st = tel.SystemTelemetry({"top_n": 3})
        top = st.sample()["top_procs"]
        self.assertNotIn("error", top)
        self.assertLessEqual(len(top["by_working_set"]), 3)
        self.assertLessEqual(len(top["by_commit"]), 3)
        for row in top["by_working_set"]:
            self.assertIn("pid", row)
            self.assertIn("name", row)
            self.assertIn("rss", row)


@unittest.skipUnless(psutil is None, "only meaningful when psutil is absent")
class TestPsutilAbsent(unittest.TestCase):
    def test_degrades_gracefully(self):
        st = tel.SystemTelemetry({})
        ram = st.sample()["ram"]
        self.assertEqual(ram["physical"], {"error": tel.PSUTIL_HINT})


class TestPsutilMissingSimulated(unittest.TestCase):
    """Exercise the degrade path deterministically regardless of whether psutil
    happens to be installed in this environment, by stubbing the import helper."""

    def test_ram_and_top_procs_report_hint_when_psutil_unimportable(self):
        orig = tel._import_psutil
        tel._import_psutil = lambda: None
        try:
            st = tel.SystemTelemetry({})
            result = st.sample()
        finally:
            tel._import_psutil = orig
        self.assertEqual(result["ram"]["physical"], {"error": tel.PSUTIL_HINT})
        self.assertEqual(result["top_procs"], {"error": tel.PSUTIL_HINT})


@unittest.skipUnless(sys.platform == "win32", "GetPerformanceInfo is Windows-only")
class TestCommitCharge(unittest.TestCase):
    def test_commit_total_within_limit(self):
        commit = tel._read_commit()
        self.assertNotIn("error", commit)
        self.assertGreater(commit["total"], 0)
        self.assertLess(commit["total"], commit["limit"])

    def test_ram_commit_section_populated_via_sample(self):
        st = tel.SystemTelemetry({})
        commit = st.sample()["ram"]["commit"]
        self.assertNotIn("error", commit)
        self.assertGreater(commit["total"], 0)


@unittest.skipUnless(sys.platform != "win32", "checks the non-windows fallback")
class TestCommitChargeNonWindows(unittest.TestCase):
    def test_windows_only_error(self):
        self.assertEqual(tel._read_commit(), {"error": "windows-only"})


class TestSampleShape(unittest.TestCase):
    def test_every_section_present_with_empty_profile(self):
        st = tel.SystemTelemetry({})
        result = st.sample()
        for key in ("ts", "ram", "top_procs", "gpu", "disks", "probes", "syncthing"):
            self.assertIn(key, result)
        self.assertIsInstance(result["disks"], list)
        self.assertIsInstance(result["probes"], list)


if __name__ == "__main__":
    unittest.main()
