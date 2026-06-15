#!/usr/bin/env python3
"""Phase 1 — packaging: console entry points resolve, wheel ships every module.

Run:  python -m unittest -v test_packaging
"""

import importlib
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _pyproject():
    with open(HERE / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


class TestEntryPoints(unittest.TestCase):
    def test_entry_points_resolve(self):
        scripts = _pyproject()["project"]["scripts"]
        self.assertIn("memdag", scripts)
        self.assertIn("memdag-mcp", scripts)
        for name, target in scripts.items():
            mod_name, _, attr = target.partition(":")
            mod = importlib.import_module(mod_name)
            self.assertTrue(hasattr(mod, attr),
                            f"entry point {name!r} -> {target!r}: {attr} missing on {mod_name}")
            self.assertTrue(callable(getattr(mod, attr)))


class TestWheelManifest(unittest.TestCase):
    def test_wheel_includes_every_runtime_module(self):
        # Every memdag*.py runtime module that exists on disk must be in the wheel
        # allowlist (the classic "added a module, forgot the allowlist" footgun).
        only = set(_pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"])
        on_disk = {p.name for p in HERE.glob("memdag*.py")}
        missing = on_disk - only
        self.assertEqual(missing, set(),
                         f"runtime modules missing from wheel only-include: {sorted(missing)}")

    def test_mcp_server_module_shipped(self):
        only = set(_pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"])
        self.assertIn("memdag_mcp.py", only)


if __name__ == "__main__":
    unittest.main()
