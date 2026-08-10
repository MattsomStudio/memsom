"""Session-scoped hard pin of MEMDAG_DB for every gate in this directory.

The repo's own `tests/` directory has NO conftest.py -- every test class there
hand-rolls its own MEMDAG_DB save/set/restore dance, and `memsom.db_path()`
falls back to `DATA_DIR / "memdag.db"`, which is the LIVE store. A test that
forgets the dance, or a teardown that raises before restoring, writes to the
real brain.

This fixture is autouse and session-scoped, and it ASSERTS the pin took effect
before any test body runs. Shipping the same idea into `tests/conftest.py`
closes that hole permanently for the repo as well.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ported from 01-pentest/gates (Phase 0, A5.1) unmodified except this path:
# derived from this file's own location instead of hardcoded, so the gates
# run correctly against whichever checkout they live in.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "memsom_pentest_gates"


@pytest.fixture(scope="session", autouse=True)
def _pin_db():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    os.environ["MEMDAG_HOME"] = str(SCRATCH)
    os.environ["MEMDAG_DB"] = str(SCRATCH / "gates.db")
    import memsom
    live = (Path.home() / ".memdag" / "memdag.db").resolve()
    assert memsom.db_path().resolve() != live, "REFUSING: db_path() is the live store"
    yield


@pytest.fixture()
def conn(request):
    """A fresh, fully-migrated scratch DB per test."""
    import memsom
    from memsom.interface import cli as memsom_cli
    path = SCRATCH / f"{request.node.name[:60]}.db"
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(path) + suffix)
        if f.exists():
            try:
                f.unlink()
            except PermissionError:
                pass
    os.environ["MEMDAG_DB"] = str(path)
    c = memsom.get_connection()
    memsom_cli.migrate_all(c)
    c.commit()
    yield c
    c.close()
    os.environ["MEMDAG_DB"] = str(SCRATCH / "gates.db")
