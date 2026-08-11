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
from pathlib import Path


def _data_dir():
    return Path(os.environ.get("MEMDAG_HOME") or Path.home() / ".memdag")


def resolve_facade_attr(name):
    """PEP-562 glue for memsom/__init__.py's DATA_DIR forwarding (Q6).

    memsom/__init__.py imports this AS its own __getattr__ (import-rename, no
    def statement there) so the facade stays fan-in 0 while DATA_DIR still
    resolves fresh on every access, two layers down.
    """
    if name == "DATA_DIR":
        return _data_dir()
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


def get_connection(path=None):
    path = Path(path or db_path())
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
