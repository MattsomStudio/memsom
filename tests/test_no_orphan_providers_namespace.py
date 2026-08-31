"""The `providers` directory left the tree; prove it cannot import back in.

Conformance gate C-32, written 2026-08-03 during the post-refactor audit.

The 2026-07-30 split (`439abcf`) moved `memsom/providers/` out to the panel
repo and deleted its tracked files, but the on-disk directory survived, holding
43 orphan `.pyc` under three `__pycache__` trees. Because memsom runs as an
editable install, `import memsom.providers` then SUCCEEDED against an empty
PEP-420 namespace package -- so any existence probe by import got a false
positive instead of `ModuleNotFoundError`. That is the exact namespace-package
trap Phase 1 documented in the panel repo, where deleting a package's
`__init__.py` would have silently disabled five import contracts with the suite
staying green.

The directory is deleted. This gate stops it coming back: a resurrected
`memsom/providers/` (even empty) makes `import memsom.providers` resolve again,
and this test goes red on contact. Control: `mkdir memsom/providers` -> RED;
`rmdir` -> green.
"""

import importlib

try:
    import pytest
except ImportError:  # stdlib-only unittest discover (CI runs these under pytest)
    import unittest
    raise unittest.SkipTest("pytest-style module; run under the CI pytest step")


def test_memsom_providers_does_not_resolve():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("memsom.providers")
