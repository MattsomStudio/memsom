"""memsom.storage.db -- the SQLite connection + schema every store opens through.

db_path()/get_connection()/DATA_DIR moved out of memsom/__init__.py (Phase 2,
the core split; Q6/MS-28 land alongside the move per SECURITY-REMEDIATION.md).

DATA_DIR resolves MEMDAG_HOME at ACCESS time via module __getattr__ (PEP 562,
Q6) -- freezing it at import time is exactly the bug that broke `unittest
discover`'s package-walk import order (see tests/_isolation.py). User DATA
dir, separate from the package dir, so the DB survives venv upgrade/reinstall
(site-packages does not).

NOTE: the store dir (~/.memdag), db file (memdag.db) and the MEMDAG_* env vars
intentionally keep the legacy name across the memsom rename -- they are private
on-disk plumbing, and renaming them would orphan existing installs' data for
zero user-visible benefit. Do NOT "fix" these to memsom.
"""

import os
import sqlite3
import importlib
from pathlib import Path


def _data_dir():
    return Path(os.environ.get("MEMDAG_HOME") or Path.home() / ".memdag")


# Phase 2 goals gate: memsom/__init__.py sits under every `import memsom` in the
# package, so a module-level `from memsom.integrity.dag import ...` or `from
# memsom.interface.cli import ...` there would put integrity/interface on the
# transitive closure of every layer that merely does `import memsom` -- storage
# importing interface via the facade, retrieval importing interface via the
# facade, and so on. importlib.import_module() is a function call, not a
# static `import` statement, so it never becomes a lint-imports graph edge;
# the two maps below list every name the facade still owes each caller for
# backward compatibility (import memsom; memsom.X).
_DAG_NAMES = frozenset((
    "CASCADE_CTE", "insert_node", "derive_node", "get_node", "live_sources",
    "parents_of", "cascade_set", "revoke_cascade",
))
_CLI_NAMES = {
    "USER_FACT": "USER_FACT", "ENDORSED_FACT": "ENDORSED_FACT",
    "cmd_seed": "frozen_cmd_seed", "cmd_ask": "frozen_cmd_ask",
    "cmd_explain": "frozen_cmd_explain", "cmd_revoke": "frozen_cmd_revoke",
    "cmd_dump": "frozen_cmd_dump", "main": "frozen_main",
}


def resolve_facade_attr(name):
    """PEP-562 glue for memsom/__init__.py's re-exports (Q6, Phase 2 goals gate).

    memsom/__init__.py imports this AS its own __getattr__ (import-rename, no
    def statement there) so the facade stays fan-in 0. DATA_DIR resolves fresh
    on every access, two layers down; the integrity.dag and interface.cli
    names resolve lazily via importlib so the facade never carries a static
    import of a higher layer (see the module-level comment above).
    """
    if name == "DATA_DIR":
        return _data_dir()
    if name in _DAG_NAMES:
        dag = importlib.import_module("memsom.integrity.dag")
        return getattr(dag, name)
    if name in _CLI_NAMES:
        cli = importlib.import_module("memsom.interface.cli")
        return getattr(cli, _CLI_NAMES[name])
    raise AttributeError(f"module 'memsom' has no attribute {name!r}")


SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  content       TEXT    NOT NULL,
  channel       TEXT    NOT NULL
                CHECK (channel IN ('endorsed','user','agent-derived','external')),
  label         INTEGER NOT NULL CHECK (label BETWEEN 0 AND 3),
  source_ref    TEXT,
  created_at    TEXT    NOT NULL,
  tombstoned    INTEGER NOT NULL DEFAULT 0,
  tombstoned_at TEXT,
  revoke_reason TEXT
);
CREATE TABLE IF NOT EXISTS edges (
  child  INTEGER NOT NULL REFERENCES nodes(id),
  parent INTEGER NOT NULL REFERENCES nodes(id),
  PRIMARY KEY (child, parent)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent);
"""


def db_path():
    return Path(os.environ.get("MEMDAG_DB") or _data_dir() / "memdag.db")


def get_connection(path=None, *, read_only=False):
    """*read_only* (Phase 5, effects): the one other connection shape the
    package needs -- `doctor.py`'s diagnostic read. Same URI/timeout/busy_timeout
    shape `doctor.py` used to build for itself; centralised here so it is the
    only caller of sqlite3.connect outside this module (effects_ratchet.py).
    Raises FileNotFoundError rather than creating an empty store, since a
    read-only opener has no business initializing one.
    """
    path = Path(path or db_path())
    if read_only:
        if not path.exists():
            raise FileNotFoundError(str(path))
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")  # per-connection, OFF by default
    # MS-28: without secure_delete, `UPDATE nodes SET content=''` (redact) leaves
    # the original overflow pages on the freelist with their bytes intact --
    # recoverable from the raw .db file. secure_delete makes SQLite overwrite
    # freed content with zeros instead. Per-connection PRAGMA, so every opener
    # of this store gets it, not just the redact path.
    conn.execute("PRAGMA secure_delete = ON")
    # busy_timeout: a BEGIN IMMEDIATE (derive_node / revoke_cascade) that meets a
    # concurrent writer should WAIT for the lock, not fail-fast with SQLITE_BUSY.
    # A long revoke_cascade holds the write lock for the duration of its recursive
    # CTE; without this, a parallel ask/derive would raise immediately. 5s is plenty
    # for a single-user/single-agent store and still bounds a true deadlock.
    # (The cascade CTE already uses a COVERING INDEX -- idx_edges_parent on the
    # WITHOUT-ROWID edges table appends the PK, so child is covered -- so the lock
    # window is short; the timeout is the belt to that suspenders.)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA)
    return conn
