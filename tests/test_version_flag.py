"""`memsom --version` contract (consumed by the desktop app's Update-memsom
button): stdout is exactly `memsom X.Y.Z`, exit 0, and the store is never
opened (so it works on a machine with no ~/.memdag at all)."""
import os
import re
import subprocess
import sys
import tempfile
import unittest


class VersionFlag(unittest.TestCase):
    def _run(self, args):
        env = dict(os.environ)
        tmp = tempfile.mkdtemp(prefix="memsom-version-")
        env["MEMDAG_HOME"] = os.path.join(tmp, "never-created")
        env["MEMDAG_DB"] = os.path.join(tmp, "never-created", "memdag.db")
        r = subprocess.run([sys.executable, "-m", "memsom.interface.cli", *args],
                           capture_output=True, text=True, env=env, timeout=60)
        return r, env["MEMDAG_HOME"]

    def test_prints_semver_and_exits_zero(self):
        r, _ = self._run(["--version"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stdout, r"^memsom \d+\.\d+\.\d+\S*\n$")

    def test_never_touches_the_store(self):
        r, home = self._run(["--version"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(os.path.exists(home), "--version must not create the data dir")

    def test_version_matches_pyproject(self):
        here = os.path.dirname(os.path.abspath(__file__))
        pyproject = os.path.join(here, "..", "pyproject.toml")
        with open(pyproject, "rb") as f:
            import tomllib
            ver = tomllib.load(f)["project"]["version"]
        r, _ = self._run(["--version"])
        # Editable installs report the metadata version, which is pyproject's.
        self.assertEqual(r.stdout.strip(), f"memsom {ver}")
