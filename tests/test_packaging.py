#!/usr/bin/env python3
"""Phase 1 — packaging: console entry points resolve, wheel ships every module.

Run:  python -m unittest -v test_packaging
"""

import importlib
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _pyproject():
    with open(HERE / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


class TestEntryPoints(unittest.TestCase):
    def test_entry_points_resolve(self):
        scripts = _pyproject()["project"]["scripts"]
        self.assertIn("memsom", scripts)
        self.assertIn("memsom-mcp", scripts)
        for name, target in scripts.items():
            mod_name, _, attr = target.partition(":")
            mod = importlib.import_module(mod_name)
            self.assertTrue(hasattr(mod, attr),
                            f"entry point {name!r} -> {target!r}: {attr} missing on {mod_name}")
            self.assertTrue(callable(getattr(mod, attr)))


class TestWheelManifest(unittest.TestCase):
    def test_wheel_ships_the_package(self):
        # The wheel ships the whole memsom/ package; every runtime module lives
        # under it, so there is no per-module allowlist to keep in sync anymore.
        wheel = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]
        self.assertEqual(wheel.get("packages"), ["memsom"])

    def test_mcp_server_module_shipped(self):
        # mcp server module is inside the shipped package.
        self.assertTrue((HERE / "memsom" / "interface" / "mcp.py").is_file())


if __name__ == "__main__":
    unittest.main()
