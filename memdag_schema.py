"""memdag_schema — shared idempotent migration helpers.

Used by every memdag_* feature module.  NO CLI, no register(), no DB access at
import time.  Call these functions inside your own get_connection() context.

Public API
----------
table_exists(conn, name)          -> bool
column_exists(conn, table, col)   -> bool
add_column(conn, table, col, decl)-> bool   (True=added, False=already present)
ensure_table(conn, create_sql)    -> None   (idempotent; requires IF NOT EXISTS)
taint_filter_clauses(conn, clearance=None, include_quarantined=False)
                                  -> (clauses, params)
    THE shared untainted-pool WHERE fragments (tombstoned / quarantined /
    redacted / archived / conf_label).  Every full-pool read path
    (memdag_cli._build_pool, memdag_retrieve._build_retrieve_pool,
    memdag_anticipatory._untainted_clauses) builds on this so a future taint
    column is added in ONE place and all of them inherit it.

Re-exports for convenience
--------------------------
RANK, NAME  (from memdag)
"""

import re
import sqlite3

import memdag

# Re-exports
RANK = memdag.RANK
NAME = memdag.NAME

# Security gate: every table/column name that is f-string-interpolated into
# PRAGMA or ALTER statements must be a plain SQL identifier.
_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _check_ident(name: str) -> None:
    """Raise ValueError if *name* is not a safe SQL identifier."""
    if not _IDENT.match(name):
        raise ValueError(f"bad identifier: {name!r}")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Return True if *name* is an existing table or view in this database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,)
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    """Return True if column *col* exists in *table*.

    Returns False (not an error) for a non-existent table — PRAGMA table_info
    yields zero rows in that case, so the any() short-circuits to False.
    """
    _check_ident(table)
    _check_ident(col)
    return any(
        r[1] == col
        for r in conn.execute(f"PRAGMA table_info({table})")
    )


def add_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> bool:
    """Add *col* with declaration *decl* to *table* if it does not already exist.

    Returns True if the column was added, False if it was already present.
    Handles a rare concurrent-caller race where two threads both pass the
    column_exists check before either ALTER completes: the duplicate-column
    OperationalError is swallowed and False is returned (no-op).

    NOTE: SQLite requires a constant DEFAULT for NOT-NULL columns added via
    ALTER TABLE.  All callers in this project comply.
    """
    _check_ident(table)
    _check_ident(col)
    if column_exists(conn, table, col):
        return False
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" in str(exc).lower():
            return False  # concurrent caller already added it — no-op
        raise
    return True


def ensure_table(conn: sqlite3.Connection, create_sql: str) -> None:
    """Execute *create_sql* idempotently.

    Raises ValueError if 'IF NOT EXISTS' (case-insensitive) is absent — this
    guards against accidentally destructive DDL being passed in.  An index
    creation statement may ride along in the same create_sql string.
    """
    if "if not exists" not in create_sql.lower():
        raise ValueError(
            "ensure_table requires 'IF NOT EXISTS' in create_sql to be idempotent"
        )
    conn.executescript(create_sql)


def taint_filter_clauses(conn: sqlite3.Connection, clearance=None,
                         include_quarantined: bool = False):
    """Shared WHERE fragments for the untainted read pool — ONE primitive.

    Returns (clauses, params) suitable for ' AND '.join(clauses).

    Semantics (mirrors the spine's pool filters; security load-bearing):
      - tombstoned = 0          ALWAYS.
      - status != 'quarantined' when the column exists, unless the caller
                                explicitly opts in via include_quarantined=True
                                (only memdag_retrieve's exclude_quarantined=False
                                flag may widen this — and ONLY this — dimension).
      - redacted = 0            when the column exists.  Never widenable.
      - archived = 0            when the column exists.  Never widenable.
      - conf_label <= ?         when *clearance* (an ALREADY-PARSED int 0-3) is
                                given and the column exists (BLP no-read-up).

    A missing column means the owning feature module has never run, so no node
    can carry that taint marker — there is nothing to exclude.

    Callers add their own channel / label / ordering clauses on top; this
    function owns only the taint dimensions, so the next taint column is added
    here once and every pool inherits it (the historical _build_pool vs
    _build_retrieve_pool divergence bug class cannot recur).
    """
    clauses = ["tombstoned = 0"]
    params = []
    if not include_quarantined and column_exists(conn, "nodes", "status"):
        clauses.append("status != 'quarantined'")
    if column_exists(conn, "nodes", "redacted"):
        clauses.append("redacted = 0")
    if column_exists(conn, "nodes", "archived"):
        clauses.append("archived = 0")
    if clearance is not None and column_exists(conn, "nodes", "conf_label"):
        clauses.append("conf_label <= ?")
        params.append(clearance)
    return clauses, params
