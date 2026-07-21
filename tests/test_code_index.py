#!/usr/bin/env python3
"""Tests for memsom_code_index — the code-RAG index.

Hermetic: the Qwen server is forced unavailable so every test runs BM25-only (the
degrade path). That both keeps tests offline AND exercises the load-bearing
"no embedder -> still works" contract.

Run:
  python -m unittest discover -s <repo> -p test_code_index.py -t <repo> -v
"""
import os
import tempfile
import time
import unittest
from pathlib import Path

import memsom
from memsom.retrieval import code_index as ci
from memsom.retrieval import qwen_embed


SAMPLE_PY = '''\
"""A tiny module."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def reciprocal_rank_fusion(rank_lists, c=60):
    """Fuse ranked lists by reciprocal rank."""
    out = {}
    for ranks in rank_lists:
        for pos, item in enumerate(ranks, start=1):
            out[item] = out.get(item, 0.0) + 1.0 / (c + pos)
    return sorted(out.items(), key=lambda x: -x[1])


class Widget:
    """A widget."""

    def spin(self):
        return "spinning"
'''

SAMPLE_SH = "#!/bin/sh\n" + "\n".join(f"echo line {i}" for i in range(150)) + "\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "code_test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        # Force the embedder OFFLINE so tests are hermetic + exercise the BM25 degrade.
        qwen_embed._AVAILABLE = False
        qwen_embed._AVAILABLE_AT = time.time() + 10_000

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        qwen_embed._AVAILABLE = None
        qwen_embed._AVAILABLE_AT = 0.0
        self.tmp.cleanup()

    def _repo(self, files):
        root = Path(self.tmp.name) / "repo"
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return str(root)


class TestSchema(Base):
    def test_migrate_creates_tables(self):
        ci.migrate(self.conn)
        for t in ("code_chunks", "code_postings", "code_docstats", "code_embeddings"):
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)
            ).fetchone()
            self.assertIsNotNone(row, f"{t} table should exist after migrate")

    def test_migrate_idempotent(self):
        ci.migrate(self.conn)
        ci.migrate(self.conn)  # must not raise


class TestChunking(Base):
    def test_python_ast_chunks(self):
        p = Path(self.tmp.name) / "m.py"
        p.write_text(SAMPLE_PY, encoding="utf-8")
        chunks = ci.chunk_file(str(p))
        syms = {c["symbol"]: c["kind"] for c in chunks}
        self.assertEqual(syms.get("add"), "function")
        self.assertEqual(syms.get("reciprocal_rank_fusion"), "function")
        self.assertEqual(syms.get("Widget"), "class")
        self.assertEqual(syms.get("spin"), "function")  # nested method captured
        # raw source is kept (docstring + comments preserved)
        rrf = next(c for c in chunks if c["symbol"] == "reciprocal_rank_fusion")
        self.assertIn("Fuse ranked lists", rrf["content"])

    def test_non_python_windows(self):
        p = Path(self.tmp.name) / "s.sh"
        p.write_text(SAMPLE_SH, encoding="utf-8")
        chunks = ci.chunk_file(str(p))
        self.assertTrue(chunks)
        self.assertTrue(all(c["kind"] == "window" for c in chunks))
        # 150 lines / 60 per window -> 3 windows
        self.assertEqual(len(chunks), 3)


class TestIndexAndSearch(Base):
    def test_index_repo_counts(self):
        root = self._repo({"pkg/m.py": SAMPLE_PY, "run.sh": SAMPLE_SH})
        stats = ci.index_repo(self.conn, root, repo="t")
        self.assertEqual(stats["repo"], "t")
        self.assertEqual(stats["files"], 2)
        self.assertGreaterEqual(stats["chunks"], 4)
        self.assertEqual(stats["vectors"], 0)  # embedder offline -> BM25-only

    def test_bm25_finds_by_keyword(self):
        root = self._repo({"pkg/m.py": SAMPLE_PY})
        ci.index_repo(self.conn, root, repo="t")
        results = ci.retrieve(self.conn, "reciprocal rank fusion of ranked lists", k=3)
        self.assertTrue(results)
        symbols = [r[2] for r in results]  # (repo, path, symbol, start, end, content)
        self.assertIn("reciprocal_rank_fusion", symbols)

    def test_retrieve_degrades_without_embedder(self):
        # vector_search must return [] when the server is unavailable, and retrieve
        # must still return BM25 hits — the core degrade contract.
        root = self._repo({"pkg/m.py": SAMPLE_PY})
        ci.index_repo(self.conn, root, repo="t")
        self.assertEqual(ci.vector_search(self.conn, "sum of a and b"), [])
        results = ci.retrieve(self.conn, "return the sum of a and b", k=2)
        self.assertTrue(results)

    def test_repo_filter(self):
        r1 = self._repo({"a.py": SAMPLE_PY})
        ci.index_repo(self.conn, r1, repo="one")
        # second repo with a distinct function
        root2 = Path(self.tmp.name) / "repo2"
        root2.mkdir()
        (root2 / "b.py").write_text("def uniquefunc():\n    return 1\n", encoding="utf-8")
        ci.index_repo(self.conn, str(root2), repo="two")
        hits = ci.retrieve(self.conn, "uniquefunc", k=5, repo="one")
        self.assertFalse(any(r[2] == "uniquefunc" for r in hits))
        hits2 = ci.retrieve(self.conn, "uniquefunc", k=5, repo="two")
        self.assertTrue(any(r[2] == "uniquefunc" for r in hits2))

    def test_reindex_is_idempotent(self):
        root = self._repo({"pkg/m.py": SAMPLE_PY})
        ci.index_repo(self.conn, root, repo="t")
        n1 = self.conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
        ci.index_repo(self.conn, root, repo="t")
        n2 = self.conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
        self.assertEqual(n1, n2)  # full-file rebuild -> no duplicates

    def test_deindex_path_removes_chunks(self):
        root = self._repo({"pkg/m.py": SAMPLE_PY})
        ci.index_repo(self.conn, root, repo="t")
        ci.deindex_path(self.conn, "t", "pkg/m.py")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM code_postings").fetchone()[0], 0)


class TestFeatureGate(Base):
    def test_disabled_by_default(self):
        os.environ.pop("MEMSOM_CODE_RAG", None)
        self.assertFalse(ci._enabled())

    def test_enabled_env(self):
        os.environ["MEMSOM_CODE_RAG"] = "1"
        try:
            self.assertTrue(ci._enabled())
        finally:
            os.environ.pop("MEMSOM_CODE_RAG", None)

    def _write_flags(self, obj):
        import json
        ci._flags_path().parent.mkdir(parents=True, exist_ok=True)
        ci._flags_path().write_text(json.dumps(obj), encoding="utf-8")

    def test_file_flag_overrides_env(self):
        # env says ON, the panel-written file says OFF -> file wins (it's authoritative).
        os.environ["MEMSOM_CODE_RAG"] = "1"
        try:
            self._write_flags({"enabled": False})
            self.assertFalse(ci._enabled())
            self._write_flags({"enabled": True})
            self.assertTrue(ci._enabled())
        finally:
            os.environ.pop("MEMSOM_CODE_RAG", None)

    def test_file_flag_absent_falls_back_to_env(self):
        # no "enabled" key -> env var still decides (backward compatible).
        os.environ.pop("MEMSOM_CODE_RAG", None)
        self._write_flags({"auto_reindex": True})   # unrelated key present, no "enabled"
        self.assertFalse(ci._enabled())
        os.environ["MEMSOM_CODE_RAG"] = "1"
        try:
            self.assertTrue(ci._enabled())
        finally:
            os.environ.pop("MEMSOM_CODE_RAG", None)

    def test_auto_reindex_defaults_on(self):
        # absent flag file -> auto-reindex is ON so existing hooks keep working.
        self.assertTrue(ci._auto_reindex_enabled())

    def test_auto_reindex_toggle(self):
        self._write_flags({"enabled": True, "auto_reindex": False})
        self.assertFalse(ci._auto_reindex_enabled())
        self._write_flags({"enabled": True, "auto_reindex": True})
        self.assertTrue(ci._auto_reindex_enabled())


class TestRepoRegistry(Base):
    """The multi-repo registry: which projects the code-RAG takes in."""

    def test_empty_by_default(self):
        self.assertEqual(ci.registered_repos(), [])

    def test_add_is_idempotent_and_names_default_to_dirname(self):
        root = self._repo({"a.py": SAMPLE_PY})
        first = ci.add_repo(root)
        self.assertTrue(first["added"])
        self.assertEqual(first["repo"]["name"], "repo")
        again = ci.add_repo(root)
        self.assertFalse(again["added"])
        self.assertEqual(len(ci.registered_repos()), 1)

    def test_remove_by_path_or_name(self):
        root = self._repo({"a.py": SAMPLE_PY})
        ci.add_repo(root, name="proj")
        self.assertTrue(ci.remove_repo("proj"))
        self.assertEqual(ci.registered_repos(), [])
        ci.add_repo(root)
        self.assertTrue(ci.remove_repo(root))
        self.assertEqual(ci.registered_repos(), [])
        self.assertFalse(ci.remove_repo("nope"))

    def test_add_preserves_unrelated_flags(self):
        # the panel owns enabled/auto_reindex in the same file — a repo write
        # must not clobber them
        ci._write_flags({"enabled": True, "auto_reindex": False})
        ci.add_repo(self._repo({"a.py": SAMPLE_PY}))
        flags = ci._read_flags()
        self.assertTrue(flags["enabled"])
        self.assertFalse(flags["auto_reindex"])
        self.assertEqual(len(flags["repos"]), 1)

    def test_bare_string_entries_accepted(self):
        root = self._repo({"a.py": SAMPLE_PY})
        ci._write_flags({"repos": [root]})
        repos = ci.registered_repos()
        self.assertEqual(len(repos), 1)
        self.assertEqual(repos[0]["name"], "repo")

    def test_discover_finds_repos_and_stops_at_the_repo(self):
        base = Path(self.tmp.name) / "code"
        (base / "one" / ".git").mkdir(parents=True)
        (base / "one" / "nested" / ".git").mkdir(parents=True)   # submodule-ish
        (base / "two" / ".git").mkdir(parents=True)
        (base / "notrepo").mkdir(parents=True)
        found = {os.path.basename(p) for p in ci.discover_repos(str(base))}
        self.assertEqual(found, {"one", "two"})


class TestIncrementalSweep(Base):
    def test_unchanged_files_are_skipped_on_reindex(self):
        root = self._repo({"a.py": SAMPLE_PY, "b.sh": SAMPLE_SH})
        first = ci.index_repo(self.conn, root, repo="t")
        self.assertEqual(first["files"], 2)
        self.assertEqual(first["unchanged"], 0)

        second = ci.index_repo(self.conn, root, repo="t")
        self.assertEqual(second["files"], 0)
        self.assertEqual(second["unchanged"], 2)
        self.assertEqual(second["chunks"], 0)
        # the index still holds everything from the first pass
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM code_chunks WHERE repo='t'").fetchone()[0],
            first["chunks"])

    def test_edited_file_is_reindexed(self):
        root = self._repo({"a.py": SAMPLE_PY})
        ci.index_repo(self.conn, root, repo="t")
        Path(root, "a.py").write_text(SAMPLE_PY + "\n\ndef added():\n    return 1\n",
                                      encoding="utf-8")
        stats = ci.index_repo(self.conn, root, repo="t")
        self.assertEqual(stats["files"], 1)
        self.assertEqual(stats["unchanged"], 0)
        hits = ci.retrieve(self.conn, "added", k=5, repo="t")
        self.assertTrue(any("added" in r[2] for r in hits))

    def test_force_reindexes_everything(self):
        root = self._repo({"a.py": SAMPLE_PY})
        ci.index_repo(self.conn, root, repo="t")
        stats = ci.index_repo(self.conn, root, repo="t", force=True)
        self.assertEqual(stats["files"], 1)
        self.assertEqual(stats["unchanged"], 0)

    def test_vanished_file_is_pruned_on_full_walk(self):
        root = self._repo({"a.py": SAMPLE_PY, "gone.py": "def orphan():\n    return 0\n"})
        ci.index_repo(self.conn, root, repo="t")
        os.remove(os.path.join(root, "gone.py"))
        stats = ci.index_repo(self.conn, root, repo="t")
        self.assertEqual(stats["pruned"], 1)
        rows = self.conn.execute(
            "SELECT COUNT(*) FROM code_chunks WHERE repo='t' AND path='gone.py'").fetchone()[0]
        self.assertEqual(rows, 0)


class TestGitScoping(Base):
    """In a git repo the index takes what git shows: tracked + untracked-not-ignored,
    never gitignored trees (vendored source, build output, node_modules)."""

    def _git(self, root, *args):
        import subprocess
        return subprocess.run(["git", "-C", root, *args],
                              capture_output=True, text=True, timeout=60)

    def test_gitignored_files_are_not_indexed(self):
        root = self._repo({
            "src/mine.py": SAMPLE_PY,
            "vendor/theirs.py": "def vendored_thing():\n    return 'nope'\n",
            ".gitignore": "vendor/\n",
        })
        if self._git(root, "init", "-q").returncode != 0:
            self.skipTest("git unavailable")
        listed = ci._git_listed_files(root)
        self.assertIsNotNone(listed)
        rels = {os.path.relpath(p, root).replace(os.sep, "/") for p in listed}
        self.assertIn("src/mine.py", rels)
        self.assertNotIn("vendor/theirs.py", rels)

        ci.index_repo(self.conn, root, repo="t")
        paths = {r[0] for r in self.conn.execute(
            "SELECT DISTINCT path FROM code_chunks WHERE repo='t'")}
        self.assertIn("src/mine.py", paths)
        self.assertNotIn("vendor/theirs.py", paths)

    def test_exclude_keeps_a_subtree_out(self):
        # non-git tree that mixes his own code with downloaded content
        root = self._repo({"mine/a.py": SAMPLE_PY,
                           "plugins/theirs.py": "def downloaded():\n    return 1\n"})
        stats = ci.index_repo(self.conn, root, repo="t", exclude=["plugins"])
        self.assertEqual(stats["files"], 1)
        paths = {r[0] for r in self.conn.execute(
            "SELECT DISTINCT path FROM code_chunks WHERE repo='t'")}
        self.assertEqual(paths, {"mine/a.py"})

    def test_exclude_round_trips_through_the_registry(self):
        root = self._repo({"a.py": SAMPLE_PY})
        ci.add_repo(root, name="mixed", exclude=["plugins", "tmp/"])
        self.assertEqual(ci.registered_repos()[0]["exclude"], ["plugins", "tmp"])

    def test_non_git_dir_still_walks(self):
        root = self._repo({"a.py": SAMPLE_PY})
        self.assertIsNone(ci._git_listed_files(root))
        stats = ci.index_repo(self.conn, root, repo="t")
        self.assertEqual(stats["files"], 1)


if __name__ == "__main__":
    unittest.main()
