"""Session-wide safety net: pin memsom's on-disk store to a throwaway location
before any test reads or writes it, for BOTH test runners the repo uses
(`pytest`, via tests/conftest.py, and `python -m unittest discover`, via
tests/__init__.py — unittest never reads conftest.py at all).

`memsom.DATA_DIR` (memsom/__init__.py) resolves `MEMDAG_HOME` at ACCESS time
(module `__getattr__`, Q6 — fixed alongside this safety net: the pin is
worthless if the constant it sets can freeze stale before it runs), so it is
safe regardless of import order — including `unittest discover -s .`, which
walks the repo root alphabetically and imports the `memsom` package (no test
files, but still executed as a directory it recurses into) before it ever
reaches `tests/`. Setting the env var as a side effect of importing THIS
module still happens as early as possible: both tests/__init__.py and
tests/conftest.py import `tests._isolation` as their first line, so whichever
loads first in a given run wins and the other is a no-op (guarded below).

`tests/gates/conftest.py` (ported from the pentest, A5.1) hard-pins its own
scratch DB independently and is deliberately left alone — this module does
not replace it, only closes the same hole for everything outside tests/gates/.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_SENTINEL = "MEMSOM_TEST_ISOLATION_PINNED"

REAL_HOME_DIR = (Path.home() / ".memdag").resolve()
REAL_DB_PATH = REAL_HOME_DIR / "memdag.db"


def pin() -> None:
    """Point MEMDAG_HOME/MEMDAG_DB at a fresh throwaway dir, once per process."""
    if os.environ.get(_SENTINEL) == "1":
        return
    root = Path(tempfile.mkdtemp(prefix="memsom_test_"))
    os.environ["MEMDAG_HOME"] = str(root)
    os.environ["MEMDAG_DB"] = str(root / "isolated.db")
    os.environ[_SENTINEL] = "1"


def assert_pinned() -> None:
    """Call once memsom IS imported: prove the pin actually took.

    Two independent checks, because either alone can pass by accident: a
    DATA_DIR that merely differs from the real path could still be some other
    unintended location, and a MEMDAG_HOME that matches os.environ could still
    not be what DATA_DIR actually resolved to if something overrode the env
    var again after the pin.
    """
    import memsom  # local import: this module must not require memsom itself

    resolved = memsom.DATA_DIR.resolve()
    if resolved == REAL_HOME_DIR:
        raise RuntimeError(
            f"memsom.DATA_DIR resolved to the REAL store ({REAL_HOME_DIR}). "
            "The isolation pin did not take — refusing to let tests run."
        )
    pinned = os.environ.get("MEMDAG_HOME")
    if pinned is None or resolved != Path(pinned).resolve():
        raise RuntimeError(
            f"memsom.DATA_DIR ({resolved}) does not match the currently "
            f"pinned MEMDAG_HOME ({pinned})."
        )


pin()
