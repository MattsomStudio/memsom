"""Tests for memdag_forget — the RS/SS forgetting layer ported into memdag.

Run:  python -m unittest discover -s . -p test_memdag_forget.py

TestParity proves the ported pure functions are identical to the originals in
~/.claude/episodic/mem_weights.py; it SKIPS where that file isn't present (CI).
"""

import importlib.util
import os
import sys
import tempfile
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_forget as forget


# --- locate the original mem_weights for the parity test (optional) ----------
_MW_PATH = Path.home() / ".claude" / "episodic" / "mem_weights.py"


def _load_mem_weights():
    if not _MW_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("mem_weights_orig", _MW_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _sample_inputs():
    """A canon + mems + events exercising new/legacy/two-number/cold paths."""
    params = dict(forget.DEFAULTS)
    canon = {
        "version": 1, "updated": None, "params": params,
        "memories": {
            "project_new": {"tier": "hot", "first_seen": "2026-01-01T00:00:00Z"},
            "project_legacy": {"tier": "hot", "weight": 0.4, "count": 12,
                               "first_seen": "2026-01-01T00:00:00Z",
                               "last_used": "2026-05-01T00:00:00Z"},
            "project_two": {"tier": "hot", "rs": 0.8, "ss": 1.2, "count": 5,
                            "first_seen": "2026-01-01T00:00:00Z",
                            "last_used": "2026-05-20T00:00:00Z"},
            "project_cold": {"tier": "cold", "rs": 0.6, "ss": 0.5, "count": 3,
                             "first_seen": "2026-01-01T00:00:00Z",
                             "last_used": "2026-06-20T00:00:00Z",
                             "index_line": "- [Cold](project_cold.md) — x",
                             "index_section": "Personal projects"},
        },
    }
    mems = [
        {"stem": "project_new", "file": "project_new.md", "pinned": False,
         "tier": "hot", "salience": "0.3"},
        {"stem": "project_legacy", "file": "project_legacy.md", "pinned": False,
         "tier": "hot", "salience": None},
        {"stem": "project_two", "file": "project_two.md", "pinned": False,
         "tier": "hot", "salience": None},
        {"stem": "project_cold", "file": "project_cold.md", "pinned": False,
         "tier": "cold", "salience": None},
        {"stem": "user_pinned", "file": "user_pinned.md", "pinned": True,
         "tier": "hot", "salience": None},
    ]
    events = {
        "project_two": [("2026-05-25T00:00:00Z", 1), ("2026-06-10T00:00:00Z", 2)],
        "project_cold": [("2026-06-22T00:00:00Z", 3)],
    }
    now = datetime(2026, 6, 24, tzinfo=timezone.utc)
    return canon, mems, events, now


class TestParity(unittest.TestCase):
    def setUp(self):
        self.mw = _load_mem_weights()
        if self.mw is None:
            self.skipTest("mem_weights.py not present (CI / non-Matthew machine)")

    def test_compute_identical_to_original(self):
        canon, mems, events, now = _sample_inputs()
        # deep-copy canon for each call (compute reads canon['memories'])
        import copy
        new_a, act_a = forget.compute(copy.deepcopy(canon), mems, events, now=now)
        new_b, act_b = self.mw.compute(copy.deepcopy(canon), mems, events, now=now)
        self.assertEqual(new_a, new_b, "ported compute() diverged from mem_weights")
        self.assertEqual(act_a, act_b, "ported actions diverged from mem_weights")

    def test_defaults_identical(self):
        self.assertEqual(forget.DEFAULTS, self.mw.DEFAULTS)


# --- storage-adapter tests (always run) --------------------------------------

class Adapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        forget.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add_mem(self, stem, channel, salience=None, body="content body", **cols):
        fm = [f"name: {stem}"]
        if salience is not None:
            fm.append(f"salience: {salience}")
        content = "---\n" + "\n".join(fm) + "\n---\n\n" + body + "\n"
        nid = memdag.insert_node(self.conn, content, channel, source_ref=f"memory:{stem}")
        if cols:
            sets = ", ".join(f"{k} = ?" for k in cols)
            self.conn.execute(f"UPDATE nodes SET {sets} WHERE id = ?",
                              (*cols.values(), nid))
        self.conn.commit()
        return nid

    def tier_of(self, stem):
        row = self.conn.execute(
            "SELECT forget_tier FROM nodes WHERE source_ref = ? AND tombstoned = 0",
            (f"memory:{stem}",)).fetchone()
        return row[0] if row else None

    def test_migrate_adds_columns(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(nodes)")}
        for c in ("forget_rs", "forget_ss", "forget_count",
                  "forget_first_seen", "forget_last_used", "forget_tier"):
            self.assertIn(c, cols)

    def test_inventory_scoped_to_memory_nodes_only(self):
        self.add_mem("project_a", "user")
        # a non-memory node (e.g. an Obsidian vault note) — different source_ref
        memdag.insert_node(self.conn, "vault note", "user", source_ref="vault:Security/note.md")
        self.conn.commit()
        mems, _, _ = forget.build_inventory(self.conn)
        stems = {m["stem"] for m in mems}
        self.assertIn("project_a", stems)
        self.assertEqual(len(mems), 1)  # only the memory: node is swept

    def test_endorsed_is_pinned_never_demotes(self):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.add_mem("user_x", "endorsed",
                     forget_rs=0.01, forget_ss=0.0,
                     forget_first_seen=old, forget_last_used=old, forget_tier="hot")
        forget.recompute_forget(self.conn)
        self.assertEqual(self.tier_of("user_x"), "hot")  # pinned, despite rs≈0 + old

    def test_unpinned_old_low_rs_demotes(self):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.add_mem("project_b", "user",
                     forget_rs=0.05, forget_ss=0.0,
                     forget_first_seen=old, forget_last_used=old, forget_tier="hot")
        actions = forget.recompute_forget(self.conn)
        self.assertEqual(self.tier_of("project_b"), "cold")
        self.assertTrue(any(a["stem"] == "project_b" and a["to"] == "cold" for a in actions))

    def test_cold_high_rs_promotes(self):
        now = datetime.now(timezone.utc)
        recent = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        old = (now - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.add_mem("project_c", "user",
                     forget_rs=0.6, forget_ss=0.5,
                     forget_first_seen=old, forget_last_used=recent, forget_tier="cold")
        forget.recompute_forget(self.conn, now=now)
        self.assertEqual(self.tier_of("project_c"), "hot")  # cold + rs>=promote_at

    def test_fresh_node_seeds_hot(self):
        self.add_mem("project_d", "user", salience="0.3")
        forget.recompute_forget(self.conn)
        self.assertEqual(self.tier_of("project_d"), "hot")
        row = self.conn.execute(
            "SELECT forget_rs FROM nodes WHERE source_ref = ?",
            ("memory:project_d",)).fetchone()
        self.assertAlmostEqual(row[0], 1.0, places=3)  # rs_seed, no decay (born now)

    def test_usage_events_reinforce(self):
        usage = Path(self.tmp.name) / "usage"
        usage.mkdir()
        (usage / "pc.jsonl").write_text(
            '{"ts": "2026-06-23T00:00:00Z", "stem": "project_e", "hits": 2}\n',
            encoding="utf-8")
        old = (datetime.now(timezone.utc) - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.add_mem("project_e", "user",
                     forget_rs=0.3, forget_ss=0.0,
                     forget_first_seen=old, forget_last_used=old, forget_tier="hot")
        before = self.conn.execute(
            "SELECT forget_rs FROM nodes WHERE source_ref = ?",
            ("memory:project_e",)).fetchone()[0]
        forget.recompute_forget(self.conn, usage_dir=usage)
        after = self.conn.execute(
            "SELECT forget_rs FROM nodes WHERE source_ref = ?",
            ("memory:project_e",)).fetchone()[0]
        self.assertGreater(after, before)  # 2 hits lifted RS

    def test_watermark_advances(self):
        self.add_mem("project_f", "user")
        self.assertIsNone(forget._get_updated(self.conn))
        forget.recompute_forget(self.conn)
        self.assertIsNotNone(forget._get_updated(self.conn))

    def test_dry_run_writes_nothing(self):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.add_mem("project_g", "user",
                     forget_rs=0.05, forget_first_seen=old, forget_last_used=old,
                     forget_tier="hot")
        forget.recompute_forget(self.conn, dry_run=True)
        self.assertEqual(self.tier_of("project_g"), "hot")  # unchanged
        self.assertIsNone(forget._get_updated(self.conn))


if __name__ == "__main__":
    unittest.main()
