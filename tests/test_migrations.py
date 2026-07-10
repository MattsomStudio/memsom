#!/usr/bin/env python3
"""Tests for the versioned migration registry in memsom_schema.

Covers:
  (a) fresh DB -> migrate_all stamps CURRENT_VERSION; status CHECK enforced.
  (b) legacy DB (status column, no CHECK, user_version=0) migrates with no
      data loss; CHECK lands; user_version bumps.
  (c) idempotency: second run is a no-op; a row inserted between runs survives.
  (d) column-preservation: extra module columns survive the introspection-driven
      rebuild with their values intact.
  (e) edge integrity: parent/child edges + foreign_key_check survive the rebuild.
  (f) baseline reconciler: a DB that already has the CHECK + user_version=0 is
      stamped to CURRENT_VERSION without a rebuild.

Run:
  python -m unittest test_migrations -v
"""

import os
import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.interface import cli as memsom_cli
from memsom.storage import schema as memsom_schema


def _insert(conn, content, channel="user", label=1, status="live", **extra):
    cols = ["content", "channel", "label", "source_ref", "created_at", "status"]
    vals = [content, channel, label, None, memsom.now_iso(), status]
    for k, v in extra.items():
        cols.append(k)
        vals.append(v)
    qmarks = ", ".join("?" for _ in cols)
    collist = ", ".join(cols)
    with conn:
        cur = conn.execute(
            f"INSERT INTO nodes({collist}) VALUES ({qmarks})", vals
        )
    return cur.lastrowid


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def _add_legacy_status(self):
        """Mimic the per-module add_column path: status without the CHECK."""
        memsom_schema.add_column(
            self.conn, "nodes", "status", "TEXT NOT NULL DEFAULT 'live'"
        )


class TestFreshDb(Base):
    def test_fresh_migrate_all_stamps_version_and_enforces_check(self):
        memsom_cli.migrate_all(self.conn)
        v = self.conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(v, memsom_schema.CURRENT_VERSION)
        self.assertTrue(memsom_schema._status_check_present(self.conn))

        # valid statuses succeed
        _insert(self.conn, "a", status="live")
        _insert(self.conn, "b", status="quarantined")

        # invalid status rejected by the DB-level CHECK
        with self.assertRaises(sqlite3.IntegrityError):
            _insert(self.conn, "c", status="bogus")

    def test_fresh_preserves_channel_and_label_checks(self):
        memsom_cli.migrate_all(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            _insert(self.conn, "x", channel="not-a-channel")
        with self.assertRaises(sqlite3.IntegrityError):
            _insert(self.conn, "y", label=9)


class TestLegacyDb(Base):
    def test_legacy_migrates_without_data_loss(self):
        self._add_legacy_status()
        self.conn.execute("PRAGMA user_version = 0")
        pid = _insert(self.conn, "parent", channel="endorsed", label=3,
                      status="live")
        cid = _insert(self.conn, "child", channel="agent-derived", label=1,
                      status="quarantined")
        with self.conn:
            self.conn.execute(
                "INSERT INTO edges(child, parent) VALUES (?,?)", (cid, pid)
            )

        self.assertFalse(memsom_schema._status_check_present(self.conn))
        memsom_schema.run_versioned_migrations(self.conn)

        # version bumped, CHECK present
        self.assertEqual(
            self.conn.execute("PRAGMA user_version").fetchone()[0],
            memsom_schema.CURRENT_VERSION,
        )
        self.assertTrue(memsom_schema._status_check_present(self.conn))

        # rows intact: ids, content, status
        rows = self.conn.execute(
            "SELECT id, content, channel, label, status FROM nodes ORDER BY id"
        ).fetchall()
        self.assertEqual(
            rows,
            [(pid, "parent", "endorsed", 3, "live"),
             (cid, "child", "agent-derived", 1, "quarantined")],
        )
        # edge intact
        self.assertEqual(
            self.conn.execute(
                "SELECT child, parent FROM edges"
            ).fetchall(),
            [(cid, pid)],
        )
        # CHECK now enforced
        with self.assertRaises(sqlite3.IntegrityError):
            _insert(self.conn, "z", status="bogus")

    def test_legacy_with_bad_status_raises_clear_error(self):
        self._add_legacy_status()
        self.conn.execute("PRAGMA user_version = 0")
        _insert(self.conn, "ok", status="live")
        # sneak in a forbidden status value (no CHECK yet)
        with self.conn:
            self.conn.execute(
                "INSERT INTO nodes(content,channel,label,source_ref,"
                "created_at,status) VALUES('bad','user',1,NULL,?, 'archived')",
                (memsom.now_iso(),),
            )
        with self.assertRaises(ValueError) as ctx:
            memsom_schema.run_versioned_migrations(self.conn)
        self.assertIn("status CHECK", str(ctx.exception))
        # rebuild aborted; data untouched; version still 0
        self.assertEqual(
            self.conn.execute("PRAGMA user_version").fetchone()[0], 0
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0], 2
        )


class TestIdempotency(Base):
    def test_second_run_is_noop_and_preserves_inserted_row(self):
        self._add_legacy_status()
        self.conn.execute("PRAGMA user_version = 0")
        _insert(self.conn, "a", status="live")
        memsom_schema.run_versioned_migrations(self.conn)
        v1 = self.conn.execute("PRAGMA user_version").fetchone()[0]

        # a row inserted between the two runs must survive (no drop/recreate)
        rid = _insert(self.conn, "between", status="live")
        memsom_schema.run_versioned_migrations(self.conn)
        v2 = self.conn.execute("PRAGMA user_version").fetchone()[0]

        self.assertEqual(v1, v2, "version must not change on second run")
        self.assertIsNotNone(
            self.conn.execute(
                "SELECT 1 FROM nodes WHERE id=?", (rid,)
            ).fetchone()
        )

    def test_current_db_skips_all_steps(self):
        memsom_cli.migrate_all(self.conn)
        # full migrate again — should be a pure no-op at CURRENT_VERSION
        before = self.conn.execute("PRAGMA user_version").fetchone()[0]
        memsom_cli.migrate_all(self.conn)
        after = self.conn.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(before, after, memsom_schema.CURRENT_VERSION)


class TestColumnPreservation(Base):
    def test_extra_columns_survive_rebuild(self):
        self._add_legacy_status()
        memsom_schema.add_column(self.conn, "nodes", "uuid", "TEXT")
        memsom_schema.add_column(
            self.conn, "nodes", "archived", "INTEGER NOT NULL DEFAULT 0"
        )
        memsom_schema.add_column(
            self.conn, "nodes", "conf_label", "INTEGER NOT NULL DEFAULT 0"
        )
        self.conn.execute("PRAGMA user_version = 0")
        rid = _insert(
            self.conn, "row", status="live",
            uuid="u-123", archived=1, conf_label=2,
        )

        memsom_schema.run_versioned_migrations(self.conn)

        row = self.conn.execute(
            "SELECT uuid, archived, conf_label FROM nodes WHERE id=?",
            (rid,),
        ).fetchone()
        self.assertEqual(row, ("u-123", 1, 2))
        # the columns themselves still exist in the schema
        for c in ("uuid", "archived", "conf_label", "status"):
            self.assertTrue(memsom_schema.column_exists(self.conn, "nodes", c))


class TestEdgeIntegrity(Base):
    def test_edges_and_fk_check_survive_rebuild(self):
        self._add_legacy_status()
        self.conn.execute("PRAGMA user_version = 0")
        a = _insert(self.conn, "a", channel="endorsed", label=3)
        b = _insert(self.conn, "b", channel="agent-derived", label=1)
        c = _insert(self.conn, "c", channel="agent-derived", label=1)
        with self.conn:
            self.conn.execute("INSERT INTO edges(child,parent) VALUES (?,?)", (b, a))
            self.conn.execute("INSERT INTO edges(child,parent) VALUES (?,?)", (c, b))

        memsom_schema.run_versioned_migrations(self.conn)

        self.assertEqual(
            self.conn.execute(
                "SELECT child, parent FROM edges ORDER BY child"
            ).fetchall(),
            [(b, a), (c, b)],
        )
        self.assertEqual(
            self.conn.execute("PRAGMA foreign_key_check").fetchall(), []
        )

    def test_autoincrement_high_water_survives(self):
        """Deleted-tail ids must never be re-handed-out (history immutability)."""
        self._add_legacy_status()
        self.conn.execute("PRAGMA user_version = 0")
        _insert(self.conn, "a")
        b = _insert(self.conn, "b")
        with self.conn:
            self.conn.execute("DELETE FROM nodes WHERE id=?", (b,))
        memsom_schema.run_versioned_migrations(self.conn)
        new_id = _insert(self.conn, "c")
        self.assertGreater(new_id, b)


class TestBaselineReconciler(Base):
    def test_db_with_check_and_v0_is_stamped_without_rebuild(self):
        # Bring the DB fully current so the CHECK is present.
        memsom_cli.migrate_all(self.conn)
        self.assertTrue(memsom_schema._status_check_present(self.conn))
        # Force user_version back to 0 to simulate a current-schema DB that was
        # never stamped (e.g. created before the registry existed).
        self.conn.execute("PRAGMA user_version = 0")
        ddl_before = memsom_schema._nodes_ddl(self.conn)

        memsom_schema.run_versioned_migrations(self.conn)

        # stamped to current; DDL untouched (no rebuild happened)
        self.assertEqual(
            self.conn.execute("PRAGMA user_version").fetchone()[0],
            memsom_schema.CURRENT_VERSION,
        )
        self.assertEqual(memsom_schema._nodes_ddl(self.conn), ddl_before)


if __name__ == "__main__":
    unittest.main()
