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


if __name__ == "__main__":
    unittest.main()
