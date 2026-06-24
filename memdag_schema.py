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


# ---------------------------------------------------------------------------
# Versioned migration registry
# ---------------------------------------------------------------------------
#
# This layer sits ABOVE the per-module migrate() functions (which stay as-is,
# called ~40 times as an additive, idempotent self-healing safety net).  The
# registry owns operations that must run EXACTLY ONCE in order and cannot live
# in a swallow-duplicate ALTER — chiefly the destructive nodes-table rebuild
# that adds the status CHECK.
#
# Driven by PRAGMA user_version (per-database, survives reopen): step N runs
# only when user_version < N, inside a transaction, then bumps user_version.
# A current DB skips every step on every command, eliminating the per-command
# re-scan cost while leaving the cheap additive add_column no-ops in place.

CURRENT_VERSION = 1

# The status values the CHECK permits.  Verified against the codebase: the only
# writes are status='live' and status='quarantined'.  'redacted'/'archived'/
# 'tombstoned' are SEPARATE columns, not status values.
_STATUS_VALUES = ("live", "quarantined")
_STATUS_CHECK_SQL = "CHECK (status IN ('live','quarantined'))"
# A normalized fingerprint used to detect the CHECK already being present in the
# saved CREATE TABLE nodes DDL (whitespace-insensitive substring match below).
_STATUS_CHECK_FINGERPRINT = "statusin('live','quarantined')"


def _current_max_id(conn: sqlite3.Connection) -> int:
    """Return the current MAX(id) of nodes, or 0 when empty."""
    row = conn.execute("SELECT MAX(id) FROM nodes").fetchone()
    return row[0] if row and row[0] is not None else 0


def _nodes_ddl(conn: sqlite3.Connection):
    """Return the CREATE TABLE DDL string for nodes, or None if absent."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='nodes'"
    ).fetchone()
    return row[0] if row and row[0] else None


def _status_check_present(conn: sqlite3.Connection) -> bool:
    """True if the nodes DDL already carries the status IN (...) CHECK."""
    ddl = _nodes_ddl(conn)
    if not ddl:
        return False
    normalized = re.sub(r"\s+", "", ddl).lower()
    return _STATUS_CHECK_FINGERPRINT in normalized


def _step_status_check(conn: sqlite3.Connection) -> None:
    """Version-1 step: ensure nodes.status carries CHECK (status IN (...)).

    No-op when the status column is absent (the owning module hasn't run, so
    there is nothing to constrain yet) or when the CHECK is already present
    (idempotent — makes the step safe even if a DB was stamped wrong).

    Otherwise performs an introspection-driven 12-step table rebuild that
    reconstructs nodes from PRAGMA table_info so it survives whatever columns
    other modules added (uuid, origin, redacted, archived, conf_label, ...).

    PRECONDITION: must be called OUTSIDE a transaction.  It toggles
    foreign_keys (a no-op inside a transaction) and then drives its own
    BEGIN/COMMIT, because a destructive rebuild cannot share the caller's
    transaction boundary.
    """
    if not column_exists(conn, "nodes", "status"):
        return  # nothing to constrain yet
    if _status_check_present(conn):
        return  # already applied — idempotent no-op

    # Guard against data that would violate the new CHECK, so we raise a clear
    # error instead of a cryptic constraint failure mid-rebuild.
    bad = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE status NOT IN ('live','quarantined')"
    ).fetchone()[0]
    if bad:
        raise ValueError(
            f"cannot add status CHECK: {bad} node(s) have a status outside "
            f"{_STATUS_VALUES!r}; resolve those rows first"
        )

    # (1) foreign_keys is per-connection and ON; toggling it inside a
    # transaction is a silent no-op, so it MUST happen before BEGIN.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        # (3) read every column name + exact declaration, in order.
        cols = conn.execute("PRAGMA table_info(nodes)").fetchall()
        # cid, name, type, notnull, dflt_value, pk
        col_names = [c[1] for c in cols]

        # (5) rebuild each column's decl, splicing the CHECK onto status.
        col_defs = []
        for _cid, name, ctype, notnull, dflt, pk in cols:
            _check_ident(name)
            parts = [name]
            if ctype:
                parts.append(ctype)
            if pk:
                # The original nodes PK is `id INTEGER PRIMARY KEY AUTOINCREMENT`.
                # PRAGMA can't tell us AUTOINCREMENT, so reproduce it verbatim for
                # the integer rowid PK to preserve id-allocation semantics.
                if (ctype or "").upper() == "INTEGER":
                    parts.append("PRIMARY KEY AUTOINCREMENT")
                else:
                    parts.append("PRIMARY KEY")
            if notnull:
                parts.append("NOT NULL")
            if dflt is not None:
                parts.append(f"DEFAULT {dflt}")
            if name == "status":
                parts.append(_STATUS_CHECK_SQL)
            col_defs.append(" ".join(parts))

        # (4) preserve the table-level channel/label CHECKs verbatim by carrying
        # them across as standalone table constraints harvested from the DDL.
        ddl = _nodes_ddl(conn) or ""
        table_constraints = _extract_table_check_constraints(ddl)

        create_body = ",\n  ".join(col_defs + table_constraints)
        create_new = f"CREATE TABLE nodes_new (\n  {create_body}\n)"

        # (9) collect dependent nodes indexes/triggers to recreate post-rename.
        recreate = [
            r[0] for r in conn.execute(
                "SELECT sql FROM sqlite_master"
                " WHERE tbl_name='nodes' AND type IN ('index','trigger')"
                "   AND sql IS NOT NULL"
            ).fetchall()
        ]

        # (2) BEGIN — explicit so the whole rebuild is one atomic unit.
        conn.execute("BEGIN IMMEDIATE")
        # (7) create the new table.
        conn.execute(create_new)
        # (8) copy rows with an EXPLICIT column list (never *), so any column
        # order mismatch cannot silently corrupt data.
        collist = ", ".join(col_names)
        conn.execute(
            f"INSERT INTO nodes_new ({collist}) SELECT {collist} FROM nodes"
        )
        # Preserve the AUTOINCREMENT high-water mark.  History is immutable in
        # memdag (ids are never reused), so the rebuilt table must continue the
        # id sequence rather than reset it.  sqlite_sequence tracks the max
        # rowid ever allocated; copy nodes' entry onto nodes_new before the drop.
        seq_row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name='nodes'"
        ).fetchone()
        # (10) drop the old table, (11) rename the new one into place.
        conn.execute("DROP TABLE nodes")
        conn.execute("ALTER TABLE nodes_new RENAME TO nodes")
        if seq_row is not None:
            # nodes_new (with AUTOINCREMENT) has its own sqlite_sequence entry
            # seeded from the copied rows; force it to the original high-water
            # mark so deleted-tail ids are never re-handed-out.
            conn.execute(
                "UPDATE sqlite_sequence SET seq=? WHERE name='nodes'",
                (max(seq_row[0], _current_max_id(conn)),),
            )
        # (9 cont.) recreate nodes-owned indexes/triggers.
        for sql in recreate:
            conn.execute(sql)
        # (12) validate FK integrity (edges -> nodes) BEFORE commit.
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(
                f"foreign_key_check failed after nodes rebuild: {violations!r}"
            )
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    finally:
        # restore the per-connection default (outside any transaction now).
        conn.execute("PRAGMA foreign_keys = ON")


def _extract_table_check_constraints(ddl: str):
    """Return table-level CHECK(...) constraints from a CREATE TABLE DDL.

    The nodes table declares its channel and label CHECKs inline on the column,
    not as table-level constraints, so PRAGMA-driven column decls already carry
    them?  No — PRAGMA table_info does NOT expose inline CHECK text.  To preserve
    them we parse the original DDL for any standalone (comma-separated) CHECK
    clause that is not attached to a column we rebuild.

    In the current schema the channel/label CHECKs are INLINE on their columns,
    so they are lost by a pure table_info rebuild.  We therefore harvest EVERY
    CHECK(...) expression from the DDL and re-attach the non-status ones as
    table-level constraints (semantically identical to inline column CHECKs).
    """
    constraints = []
    # Find each CHECK (...) with balanced parens.
    i = 0
    low = ddl
    while True:
        idx = low.lower().find("check", i)
        if idx == -1:
            break
        # advance to the opening paren
        j = idx + len("check")
        while j < len(low) and low[j] in " \t\r\n":
            j += 1
        if j >= len(low) or low[j] != "(":
            i = idx + len("check")
            continue
        # balanced-paren scan
        depth = 0
        k = j
        while k < len(low):
            if low[k] == "(":
                depth += 1
            elif low[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        expr = low[idx:k + 1]
        normalized = re.sub(r"\s+", "", expr).lower()
        # Skip the status CHECK (we splice a fresh one onto the status column).
        if "status" not in normalized:
            constraints.append(expr.strip())
        i = k + 1
    return constraints


def _ready_status_check(conn: sqlite3.Connection) -> bool:
    """Precondition for the v1 step: the status column must exist.

    If it does not, the owning per-module migrate() has not run yet, so the
    step can do nothing and we must NOT stamp version 1 — otherwise a later
    add_column would create status without the CHECK and the gated step would
    never re-fire.  Returning False keeps the DB at its current version so the
    step runs once the column actually appears.
    """
    return column_exists(conn, "nodes", "status")


# Ordered list of (version, ready-predicate, step-callable).  Step N applies
# only when user_version < N AND its predicate holds; n is always an int
# literal — never user input.
_MIGRATIONS = [
    (1, _ready_status_check, _step_status_check),
]


def _stamp_baseline_if_legacy(conn: sqlite3.Connection) -> None:
    """Reconcile a user_version==0 DB to a conservative starting version.

    Decides the true starting version WITHOUT running any step.  The ONLY
    forward stamp we make is to version 1, and ONLY when the status CHECK is
    already present in the nodes DDL (a legacy DB that was already rebuilt, or a
    DB created with the CHECK inline).  Otherwise we leave it at 0 so the v1 step
    actually runs.

    This is deliberately conservative: a fresh DB (no status column yet) stays
    at 0; the v1 step is a no-op there because there is no status column to
    constrain, and the per-module migrate()s create columns additively.  Even a
    mis-stamped DB self-corrects because the step itself re-checks for the CHECK.
    """
    if _status_check_present(conn):
        conn.execute(f"PRAGMA user_version = {CURRENT_VERSION}")


def run_versioned_migrations(conn: sqlite3.Connection) -> None:
    """Apply ordered, once-only migration steps gated by PRAGMA user_version.

    Called AFTER the per-module migrate()s in migrate_all().  Reads the current
    user_version; if 0, reconciles a legacy/fresh baseline first.  Then for each
    (n, step) where n > current_version, runs the step and bumps user_version to
    n.  A current DB (user_version == CURRENT_VERSION) does nothing.

    NOTE: each step is responsible for its own transaction boundary, because the
    v1 rebuild must toggle foreign_keys (a no-op inside a transaction) and drive
    its own BEGIN/COMMIT.  We therefore do NOT wrap the step in `with conn:`.
    The user_version bump is a single statement committed immediately after.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current == 0:
        _stamp_baseline_if_legacy(conn)
        current = conn.execute("PRAGMA user_version").fetchone()[0]

    for n, ready, step in _MIGRATIONS:
        if n <= current:
            continue
        if not ready(conn):
            # Precondition not met (owning module hasn't run); leave the version
            # untouched so this step re-fires on a later call once it is ready.
            continue
        step(conn)
        # n is an int literal from _MIGRATIONS, never user input.
        conn.execute(f"PRAGMA user_version = {n}")
        current = n


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

    DELIBERATELY EXCLUDED: `stale` (memdag_stale). Staleness is NOT a security
    dimension — a stale node still answers, the answer is just flagged, and
    exclusion is an explicit operator opt-in (`ask --fresh-only`). Folding `stale`
    in here would make "outdated" silently mean "dead". The opt-in fragment lives
    in memdag_stale.stale_exclude_clauses and is appended only on the fresh-only path.
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
