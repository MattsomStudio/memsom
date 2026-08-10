"""Session-wide safety net: pin memsom's on-disk store to a throwaway location
before memsom is ever imported by this process, for BOTH test runners the repo
uses (`pytest`, via tests/conftest.py, and `python -m unittest discover`, via
tests/__init__.py — unittest never reads conftest.py at all).

WHY THIS HAS TO BE AN IMPORT-TIME SIDE EFFECT, NOT A FIXTURE
--------------------------------------------------------------
`memsom.DATA_DIR` (memsom/__init__.py) is a MODULE-LEVEL constant read from
`MEMDAG_HOME` once, the first time `memsom` is imported anywhere in the
process (Q6 — Phase 1 fixes the freeze; until then this is the only guard).
A `MEMDAG_HOME` set inside a pytest fixture body runs too late if any test
module, or a script this phase writes, imports memsom during COLLECTION.
Setting the env var as a side effect of importing THIS module is the only
ordering pytest and unittest both actually honor: both tests/__init__.py and
tests/conftest.py import `tests._isolation` as their first line, so whichever
loads first in a given run wins and the other is a no-op (guarded below).

`tests/gates/conftest.py` (ported from the pentest, A5.1) hard-pins its own
scratch DB independently and is deliberately left alone — this module does
not replace it, only closes the same hole for everything outside tests/gates/.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_SENTINEL = "MEMSOM_TEST_ISOLATION_PINNED"

REAL_HOME_DIR = (Path.home() / ".memdag").resolve()
REAL_DB_PATH = REAL_HOME_DIR / "memdag.db"


def pin() -> None:
    """Point MEMDAG_HOME/MEMDAG_DB at a fresh throwaway dir, once per process.

    Hard-fails if `memsom` is already in sys.modules: DATA_DIR would already
    be frozen at whatever MEMDAG_HOME held at that earlier import, and this
    pin would silently do nothing for it.
    """
    if os.environ.get(_SENTINEL) == "1":
        return
    if "memsom" in sys.modules:
        raise RuntimeError(
            "memsom was imported before tests._isolation could pin "
            "MEMDAG_HOME — memsom.DATA_DIR is frozen at import time (Q6) and "
            "may now point at the real ~/.memdag. Every test entry point "
            "must import tests, tests.conftest, or tests._isolation before "
            "it imports memsom."
        )
    root = Path(tempfile.mkdtemp(prefix="memsom_test_"))
    os.environ["MEMDAG_HOME"] = str(root)
    os.environ["MEMDAG_DB"] = str(root / "isolated.db")
    os.environ[_SENTINEL] = "1"


def assert_pinned() -> None:
    """Call once memsom IS imported: prove the pin actually took.

    Two independent checks, because either alone can pass by accident: a
    DATA_DIR that merely differs from the real path could still be some other
    unintended location, and a MEMDAG_HOME that matches os.environ could still
    not be what DATA_DIR actually resolved to if memsom was imported earlier
    in the process (see `pin()`'s own guard for that case).
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
            f"pinned MEMDAG_HOME ({pinned}) — it was frozen at an earlier "
            "import with a different value. Import tests._isolation first."
        )


pin()
