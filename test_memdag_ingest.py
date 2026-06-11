#!/usr/bin/env python3
"""Tests for memdag_ingest.

Run:
  python -W error::DeprecationWarning -m unittest discover \
      -s C:\\Users\\you\\memdag -p test_memdag_ingest.py \
      -t C:\\Users\\you\\memdag -v
"""

import os
import tempfile
import unittest
import urllib.error
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_ingest
import memdag_schema

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_ingest.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


class TestMigration(Base):
    def test_content_hash_column_exists_after_migrate(self):
        self.assertTrue(
            memdag_schema.column_exists(self.conn, "nodes", "content_hash"),
            "content_hash column should exist after migrate()",
        )

    def test_migrate_idempotent(self):
        # Call migrate() a second time — should not raise
        memdag_ingest.migrate(self.conn)
        self.assertTrue(memdag_schema.column_exists(self.conn, "nodes", "content_hash"))

    def test_existing_nodes_have_null_content_hash(self):
        """Rows inserted BEFORE migrate() should have NULL content_hash."""
        # Close the current connection (migrate already called in setUp)
        self.conn.close()

        # Fresh DB without migrate
        fresh_db = Path(self.tmp.name) / "fresh.db"
        os.environ["MEMDAG_DB"] = str(fresh_db)
        conn2 = memdag.get_connection()
        try:
            nid = memdag.insert_node(conn2, "some content", "user")
            # NOW migrate
            memdag_ingest.migrate(conn2)
            row = conn2.execute(
                "SELECT content_hash FROM nodes WHERE id=?", (nid,)
            ).fetchone()
            self.assertIsNone(row[0], "pre-migration rows should have NULL content_hash")
        finally:
            conn2.close()
            # Restore
            os.environ["MEMDAG_DB"] = str(self.db)
            self.conn = memdag.get_connection()
            memdag_ingest.migrate(self.conn)


# ---------------------------------------------------------------------------
# ingest_text — basic
# ---------------------------------------------------------------------------


class TestIngestTextBasic(Base):
    def test_creates_node_at_declared_channel_endorsed(self):
        ids = memdag_ingest.ingest_text(self.conn, "Hello endorsed world.", "endorsed")
        self.assertEqual(len(ids), 1)
        node = memdag.get_node(self.conn, ids[0])
        self.assertEqual(node["channel"], "endorsed")
        self.assertEqual(node["label"], memdag.RANK["endorsed"])

    def test_creates_node_at_declared_channel_external(self):
        ids = memdag_ingest.ingest_text(self.conn, "External article content.", "external")
        self.assertEqual(len(ids), 1)
        node = memdag.get_node(self.conn, ids[0])
        self.assertEqual(node["channel"], "external")
        self.assertEqual(node["label"], memdag.RANK["external"])

    def test_creates_node_at_declared_channel_user(self):
        ids = memdag_ingest.ingest_text(self.conn, "User stated fact.", "user")
        self.assertEqual(len(ids), 1)
        node = memdag.get_node(self.conn, ids[0])
        self.assertEqual(node["channel"], "user")

    def test_content_hash_set_on_inserted_node(self):
        ids = memdag_ingest.ingest_text(self.conn, "Content with a hash.", "user")
        row = self.conn.execute(
            "SELECT content_hash FROM nodes WHERE id=?", (ids[0],)
        ).fetchone()
        self.assertIsNotNone(row[0], "content_hash must be set after ingest")
        self.assertEqual(len(row[0]), 64, "SHA-256 hex should be 64 chars")

    def test_source_ref_stored_single_chunk(self):
        ids = memdag_ingest.ingest_text(
            self.conn, "Single chunk text.", "user", source_ref="myref"
        )
        node = memdag.get_node(self.conn, ids[0])
        self.assertEqual(node["source_ref"], "myref")

    def test_empty_text_returns_empty_list(self):
        ids = memdag_ingest.ingest_text(self.conn, "", "user")
        self.assertEqual(ids, [])

    def test_whitespace_only_returns_empty_list(self):
        ids = memdag_ingest.ingest_text(self.conn, "   \n\n   ", "user")
        self.assertEqual(ids, [])


# ---------------------------------------------------------------------------
# ingest_text — deduplication
# ---------------------------------------------------------------------------


class TestIngestTextDedup(Base):
    def test_identical_content_dedups_to_same_id(self):
        text = "This is a unique piece of text for dedup testing."
        ids1 = memdag_ingest.ingest_text(self.conn, text, "user")
        ids2 = memdag_ingest.ingest_text(self.conn, text, "user")
        self.assertEqual(len(ids1), 1)
        self.assertEqual(len(ids2), 1)
        self.assertEqual(ids1[0], ids2[0], "Same content should reuse existing node")

    def test_whitespace_normalized_for_dedup(self):
        """Content that differs only in whitespace should hash to the same node."""
        text1 = "Whitespace normalization  test."
        text2 = "Whitespace  normalization test."  # different internal spacing
        # These normalize differently (different words), so use same internal structure
        text_a = "Exact  same   content."
        text_b = "Exact same content."   # normalized form matches
        ids_a = memdag_ingest.ingest_text(self.conn, text_a, "user")
        ids_b = memdag_ingest.ingest_text(self.conn, text_b, "user")
        self.assertEqual(ids_a[0], ids_b[0], "Whitespace normalization should dedup")

    def test_different_content_gets_different_ids(self):
        ids1 = memdag_ingest.ingest_text(self.conn, "Alpha content for test.", "user")
        ids2 = memdag_ingest.ingest_text(self.conn, "Beta content for test.", "user")
        self.assertNotEqual(ids1[0], ids2[0])

    def test_tombstoned_node_not_reused_for_dedup(self):
        """A tombstoned node with the same hash should NOT be reused."""
        text = "Content that will be tombstoned."
        ids1 = memdag_ingest.ingest_text(self.conn, text, "user")
        nid1 = ids1[0]
        # Tombstone the node
        memdag.revoke_cascade(self.conn, nid1, "test revoke")
        # Now ingest the same text again — should create a NEW node
        ids2 = memdag_ingest.ingest_text(self.conn, text, "user")
        self.assertEqual(len(ids2), 1)
        self.assertNotEqual(ids2[0], nid1, "Should create new node, not reuse tombstoned one")


# ---------------------------------------------------------------------------
# ingest_text — chunking
# ---------------------------------------------------------------------------


class TestIngestTextChunking(Base):
    def _make_long_doc(self, n_paragraphs=10, words_per_para=40):
        """Build a doc that is guaranteed to exceed chunk_chars=1200."""
        paras = []
        for i in range(n_paragraphs):
            para = " ".join(f"word{i}_{j}" for j in range(words_per_para))
            paras.append(para)
        return "\n\n".join(paras)

    def test_long_doc_creates_multiple_chunks(self):
        doc = self._make_long_doc(n_paragraphs=10, words_per_para=50)
        self.assertGreater(len(doc), 1200, "Precondition: doc must be longer than chunk_chars")
        ids = memdag_ingest.ingest_text(self.conn, doc, "user", chunk_chars=1200)
        self.assertGreater(len(ids), 1, "Long doc should produce multiple chunk nodes")

    def test_chunk_nodes_have_chunk_ref_in_source_ref(self):
        doc = self._make_long_doc()
        ids = memdag_ingest.ingest_text(
            self.conn, doc, "user", source_ref="test_doc.md", chunk_chars=1200
        )
        for idx, nid in enumerate(ids):
            node = memdag.get_node(self.conn, nid)
            self.assertIn(
                f"#chunk{idx}", node["source_ref"],
                f"chunk {idx} should have #chunk{idx} in source_ref",
            )

    def test_chunk_ref_includes_base_ref(self):
        doc = self._make_long_doc()
        ids = memdag_ingest.ingest_text(
            self.conn, doc, "endorsed", source_ref="docs/file.md", chunk_chars=1200
        )
        self.assertGreater(len(ids), 1)
        node = memdag.get_node(self.conn, ids[0])
        self.assertTrue(
            node["source_ref"].startswith("docs/file.md"),
            "source_ref should start with the base ref",
        )

    def test_chunking_disabled_stores_single_node(self):
        doc = self._make_long_doc()
        ids = memdag_ingest.ingest_text(self.conn, doc, "user", chunk=False)
        self.assertEqual(len(ids), 1, "chunk=False should produce exactly one node")

    def test_short_doc_not_chunked(self):
        text = "Short enough to fit in one chunk."
        ids = memdag_ingest.ingest_text(self.conn, text, "user", chunk_chars=1200)
        self.assertEqual(len(ids), 1)

    def test_chunk_dedup_within_same_doc(self):
        """Repeated identical paragraphs in a doc dedup to the same node id."""
        repeated_para = "This paragraph repeats itself verbatim across the document.\n\n"
        # Build a doc where the same paragraph appears multiple times
        doc = repeated_para * 6  # 6 identical paragraphs
        ids = memdag_ingest.ingest_text(self.conn, doc, "user", chunk_chars=200)
        # All should resolve to the same node (deduped)
        self.assertEqual(
            len(set(ids)), 1,
            "Repeated identical paragraphs should all dedup to one node",
        )

    def test_all_content_covered(self):
        """Verify no content is dropped: all chars in chunks appear in some node."""
        doc = self._make_long_doc(n_paragraphs=5, words_per_para=60)
        ids = memdag_ingest.ingest_text(self.conn, doc, "user", chunk_chars=500)
        seen_ids = set(ids)
        all_content = ""
        for nid in seen_ids:
            node = memdag.get_node(self.conn, nid)
            all_content += node["content"] + " "
        # Every word from the original doc should appear somewhere in the stored content
        import re
        original_words = set(re.findall(r"\w+", doc))
        stored_words = set(re.findall(r"\w+", all_content))
        missing = original_words - stored_words
        self.assertEqual(missing, set(), f"These words were dropped: {missing}")


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------


class TestIngestFile(Base):
    def test_ingest_file_creates_node(self):
        f = Path(self.tmp.name) / "note.md"
        f.write_text("# Title\n\nSome endorsed content about networking.", encoding="utf-8")
        ids = memdag_ingest.ingest_file(self.conn, f, "endorsed")
        self.assertGreater(len(ids), 0)
        node = memdag.get_node(self.conn, ids[0])
        self.assertEqual(node["channel"], "endorsed")

    def test_ingest_file_source_ref_is_path(self):
        f = Path(self.tmp.name) / "ref_test.md"
        f.write_text("Content for source ref test.", encoding="utf-8")
        ids = memdag_ingest.ingest_file(self.conn, f, "user")
        node = memdag.get_node(self.conn, ids[0])
        self.assertIn(str(f), node["source_ref"])

    def test_ingest_file_utf8_replace_on_bad_bytes(self):
        """Files with invalid UTF-8 sequences are read with errors=replace."""
        f = Path(self.tmp.name) / "bad_utf8.txt"
        # Write bytes that are invalid UTF-8
        f.write_bytes(b"Good text before \xff\xfe bad bytes after.")
        ids = memdag_ingest.ingest_file(self.conn, f, "user")
        self.assertGreater(len(ids), 0)
        # Should not raise, and content should be stored

    def test_ingest_file_missing_raises_oserror(self):
        with self.assertRaises(OSError):
            memdag_ingest.ingest_file(
                self.conn, Path(self.tmp.name) / "nonexistent.md", "user"
            )


# ---------------------------------------------------------------------------
# ingest_dir
# ---------------------------------------------------------------------------


class TestIngestDir(Base):
    def _make_dir_tree(self):
        """Create a small directory tree with .md and .txt files."""
        base = Path(self.tmp.name) / "vault"
        base.mkdir()
        (base / "a.md").write_text("First markdown file content.", encoding="utf-8")
        (base / "b.md").write_text("Second markdown file content.", encoding="utf-8")
        (base / "notes.txt").write_text("Text file, should be ignored.", encoding="utf-8")
        sub = base / "sub"
        sub.mkdir()
        (sub / "c.md").write_text("Nested markdown file content.", encoding="utf-8")
        return base

    def test_ingest_dir_finds_md_files(self):
        base = self._make_dir_tree()
        ids = memdag_ingest.ingest_dir(self.conn, base, "endorsed")
        # 3 .md files, each short enough for 1 node
        self.assertEqual(len(ids), 3)

    def test_ingest_dir_ignores_non_glob_files(self):
        base = self._make_dir_tree()
        ids = memdag_ingest.ingest_dir(self.conn, base, "endorsed", glob="*.md")
        # notes.txt should be excluded
        for nid in ids:
            node = memdag.get_node(self.conn, nid)
            self.assertFalse(
                node["source_ref"].endswith(".txt"),
                "txt file should not be ingested",
            )

    def test_ingest_dir_custom_glob(self):
        base = self._make_dir_tree()
        ids = memdag_ingest.ingest_dir(self.conn, base, "user", glob="*.txt")
        self.assertEqual(len(ids), 1, "Only the .txt file should match")
        node = memdag.get_node(self.conn, ids[0])
        self.assertIn("notes.txt", node["source_ref"])

    def test_ingest_dir_channel_stamped_correctly(self):
        base = self._make_dir_tree()
        ids = memdag_ingest.ingest_dir(self.conn, base, "external")
        for nid in ids:
            node = memdag.get_node(self.conn, nid)
            self.assertEqual(node["channel"], "external")

    def test_ingest_dir_empty_dir_returns_empty_list(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        ids = memdag_ingest.ingest_dir(self.conn, empty, "user")
        self.assertEqual(ids, [])


# ---------------------------------------------------------------------------
# ingest_url — monkeypatched
# ---------------------------------------------------------------------------


class TestIngestUrl(Base):
    def _make_mock_urlopen(self, body: bytes):
        """Return a context-manager mock that yields a response with .read() -> body."""
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen = MagicMock(return_value=resp)
        return mock_urlopen

    def test_ingest_url_channel_forced_external(self):
        body = b"Remote content from an external source."
        mock_open = self._make_mock_urlopen(body)
        with patch("urllib.request.urlopen", mock_open):
            ids = memdag_ingest.ingest_url(self.conn, "https://example.com/doc")
        self.assertGreater(len(ids), 0)
        node = memdag.get_node(self.conn, ids[0])
        self.assertEqual(node["channel"], "external",
                         "URL ingestion must always stamp channel=external")

    def test_ingest_url_source_ref_is_url(self):
        body = b"Content from a URL."
        url = "https://example.com/article"
        mock_open = self._make_mock_urlopen(body)
        with patch("urllib.request.urlopen", mock_open):
            ids = memdag_ingest.ingest_url(self.conn, url)
        node = memdag.get_node(self.conn, ids[0])
        self.assertEqual(node["source_ref"], url)

    def test_ingest_url_user_agent_set(self):
        """The User-Agent header must be set on the request (SPINE etiquette)."""
        body = b"Content."
        captured_req = []

        def mock_open(req, timeout=None):
            captured_req.append(req)
            resp = MagicMock()
            resp.read.return_value = body
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", mock_open):
            memdag_ingest.ingest_url(self.conn, "https://example.com/")

        self.assertTrue(len(captured_req) > 0)
        ua = captured_req[0].get_header("User-agent")
        self.assertIsNotNone(ua, "User-Agent header should be set")
        self.assertIn("memdag", ua.lower())

    def test_ingest_url_raises_on_network_failure(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(urllib.error.URLError):
                memdag_ingest.ingest_url(self.conn, "https://unreachable.invalid/")

    def test_ingest_url_dedup_same_url_same_content(self):
        body = b"Stable remote content that does not change."
        mock_open = self._make_mock_urlopen(body)
        with patch("urllib.request.urlopen", mock_open):
            ids1 = memdag_ingest.ingest_url(self.conn, "https://example.com/stable")
        with patch("urllib.request.urlopen", mock_open):
            ids2 = memdag_ingest.ingest_url(self.conn, "https://example.com/stable")
        self.assertEqual(ids1[0], ids2[0], "Same content from same URL should dedup")

    def test_ingest_url_latin1_fallback(self):
        """Bytes that are invalid UTF-8 should fall back to latin-1 decode."""
        body = b"Content with latin-1 char: \xe9l\xe8ve."
        mock_open = self._make_mock_urlopen(body)
        with patch("urllib.request.urlopen", mock_open):
            ids = memdag_ingest.ingest_url(self.conn, "https://example.com/latin")
        self.assertGreater(len(ids), 0)


# ---------------------------------------------------------------------------
# Chunking internals
# ---------------------------------------------------------------------------


class TestChunkSplitter(unittest.TestCase):
    """Unit tests for the internal _split_chunks helper."""

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = memdag_ingest._split_chunks(text, 1200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text.strip())

    def test_paragraph_boundary_preferred(self):
        """Split should prefer double-newline over single-newline."""
        para1 = "A " * 300  # 600 chars
        para2 = "B " * 300
        text = para1.rstrip() + "\n\n" + para2.rstrip()
        chunks = memdag_ingest._split_chunks(text, 700)
        self.assertGreater(len(chunks), 1)
        # First chunk should end with para1 content (not mid-para2)
        self.assertNotIn("B", chunks[0].replace(" ", ""))

    def test_no_empty_chunks(self):
        text = "\n\n".join("word " * 100 for _ in range(5))
        chunks = memdag_ingest._split_chunks(text, 400)
        for ch in chunks:
            self.assertTrue(len(ch.strip()) > 0, f"Empty chunk found: {ch!r}")

    def test_all_content_preserved(self):
        import re
        text = " ".join(f"token_{i}" for i in range(500))
        chunks = memdag_ingest._split_chunks(text, 500)
        combined = " ".join(chunks)
        original_tokens = set(re.findall(r"token_\d+", text))
        stored_tokens = set(re.findall(r"token_\d+", combined))
        self.assertEqual(original_tokens, stored_tokens)

    def test_exactly_chunk_chars_boundary(self):
        """Text exactly at chunk_chars should produce one chunk."""
        text = "x" * 1200
        chunks = memdag_ingest._split_chunks(text, 1200)
        self.assertEqual(len(chunks), 1)

    def test_one_over_chunk_chars_produces_two(self):
        """Text slightly over chunk_chars should split."""
        text = "word " * 300  # ~1500 chars
        chunks = memdag_ingest._split_chunks(text, 1200)
        self.assertGreater(len(chunks), 1)


# ---------------------------------------------------------------------------
# Retrieve integration — graceful degrade
# ---------------------------------------------------------------------------


class TestRetrieveIntegration(Base):
    def test_ingest_works_without_memdag_retrieve(self):
        """ingest_text must not crash even when memdag_retrieve is absent."""
        import sys
        # Temporarily make memdag_retrieve unimportable
        original = sys.modules.get("memdag_retrieve", None)
        sys.modules["memdag_retrieve"] = None  # causes ImportError on import
        try:
            ids = memdag_ingest.ingest_text(
                self.conn, "Content ingested without retrieve module.", "user"
            )
            self.assertEqual(len(ids), 1)
        finally:
            if original is None:
                sys.modules.pop("memdag_retrieve", None)
            else:
                sys.modules["memdag_retrieve"] = original

    def test_ingest_works_when_index_node_raises(self):
        """ingest_text must not crash when memdag_retrieve.index_node raises."""
        mock_retrieve = MagicMock()
        mock_retrieve.index_node.side_effect = RuntimeError("index failed")
        import sys
        original = sys.modules.get("memdag_retrieve", None)
        sys.modules["memdag_retrieve"] = mock_retrieve
        try:
            ids = memdag_ingest.ingest_text(
                self.conn, "Content ingested despite retrieve error.", "endorsed"
            )
            self.assertEqual(len(ids), 1)
        finally:
            if original is None:
                sys.modules.pop("memdag_retrieve", None)
            else:
                sys.modules["memdag_retrieve"] = original


# ---------------------------------------------------------------------------
# CLI register smoke test
# ---------------------------------------------------------------------------


class TestRegister(Base):
    def test_register_mounts_three_subcommands(self):
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        memdag_ingest.register(sub)
        # Parse each subcommand name to confirm they are registered
        args = p.parse_args(["ingest", "some/path.md", "--channel", "user"])
        self.assertEqual(args.command, "ingest")
        args = p.parse_args(["ingest-dir", "some/dir", "--channel", "endorsed"])
        self.assertEqual(args.command, "ingest-dir")
        args = p.parse_args(["ingest-url", "https://example.com/"])
        self.assertEqual(args.command, "ingest-url")

    def test_ingest_dir_default_glob_is_md(self):
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        memdag_ingest.register(sub)
        args = p.parse_args(["ingest-dir", "some/dir", "--channel", "endorsed"])
        self.assertEqual(args.glob, "*.md")

    def test_ingest_text_subcommand_parses(self):
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        memdag_ingest.register(sub)
        args = p.parse_args(["ingest-text", "some text", "--channel", "user"])
        self.assertEqual(args.command, "ingest-text")
        self.assertEqual(args.text, "some text")
        self.assertEqual(args.channel, "user")
        self.assertIsNone(args.ref)

    def test_ingest_text_subcommand_functional(self):
        """_cmd_ingest_text stores a node at the declared channel."""
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        memdag_ingest.register(sub)
        unique_text = "Hello from ingest-text subcommand functional test unique_xyz987."
        args = p.parse_args(
            ["ingest-text", unique_text, "--channel", "endorsed", "--ref", "myref"]
        )
        # call the handler directly (uses MEMDAG_DB env already set by setUp)
        memdag_ingest._cmd_ingest_text(args)
        # verify the node was actually created — search by source_ref
        conn = memdag.get_connection()
        try:
            rows = conn.execute(
                "SELECT id, channel, source_ref FROM nodes WHERE source_ref='myref'"
            ).fetchall()
            self.assertEqual(len(rows), 1, "ingest-text handler should create exactly one node with source_ref=myref")
            self.assertEqual(rows[0][1], "endorsed")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Frozen-core non-interference: existing nodes.status CHECK constraint intact
# ---------------------------------------------------------------------------


class TestFrozenCoreCompat(Base):
    def test_insert_node_still_works_after_migrate(self):
        """memdag.insert_node must work normally after ingest migration."""
        nid = memdag.insert_node(self.conn, "post-migrate insert", "user")
        node = memdag.get_node(self.conn, nid)
        self.assertEqual(node["channel"], "user")

    def test_derive_node_still_works_after_migrate(self):
        a = self.add("parent content", "endorsed")
        b, label = memdag.derive_node(self.conn, "derived content", [a])
        self.assertEqual(label, memdag.RANK["endorsed"])

    def test_live_sources_unaffected_by_migrate(self):
        self.add("source one", "endorsed")
        self.add("source two", "user")
        sources = memdag.live_sources(self.conn)
        self.assertEqual(len(sources), 2)

    def test_content_hash_null_on_direct_insert(self):
        """Nodes inserted via insert_node (not ingest_text) have NULL content_hash."""
        nid = memdag.insert_node(self.conn, "direct insert", "user")
        row = self.conn.execute(
            "SELECT content_hash FROM nodes WHERE id=?", (nid,)
        ).fetchone()
        self.assertIsNone(row[0])


# ---------------------------------------------------------------------------
# F-14 / F-13: caller-layer trust guards
# ---------------------------------------------------------------------------


class TestChannelLabelLock(Base):
    """F-14 (AUDIT 2026-06-11): a source node's label is pinned to its channel."""

    def test_authoritative_label_matches_channel(self):
        for ch, rank in memdag.RANK.items():
            self.assertEqual(memdag_ingest.authoritative_label(ch), rank)
        with self.assertRaises(ValueError):
            memdag_ingest.authoritative_label("superuser")

    def test_ingest_text_stamps_channel_label_not_caller_label(self):
        """No ingest path can produce a node whose label != RANK[channel]."""
        for ch in ("external", "user", "endorsed"):
            ids = memdag_ingest.ingest_text(
                self.conn, f"some {ch} content line about nebula", ch)
            for nid in ids:
                node = memdag.get_node(self.conn, nid)
                self.assertEqual(node["channel"], ch)
                self.assertEqual(node["label"], memdag.RANK[ch],
                                 f"{ch} node must carry label RANK[{ch}]")


class TestChannelCeiling(Base):
    """F-13 (AUDIT 2026-06-11): optional channel ceiling, permissive by default."""

    def tearDown(self):
        os.environ.pop(memdag_ingest.CHANNEL_CEILING_ENV, None)
        super().tearDown()

    def test_default_is_permissive(self):
        self.assertIsNone(memdag_ingest.channel_ceiling())
        # endorsed allowed when no ceiling set
        ids = memdag_ingest.ingest_text(self.conn, "endorsed line one two", "endorsed")
        self.assertEqual(len(ids), 1)

    def test_ceiling_blocks_over_rank_channel(self):
        os.environ[memdag_ingest.CHANNEL_CEILING_ENV] = "user"
        self.assertEqual(memdag_ingest.channel_ceiling(), memdag.RANK["user"])
        # user and below are fine
        self.assertEqual(memdag_ingest.enforce_channel_ceiling("user"), "user")
        self.assertEqual(memdag_ingest.enforce_channel_ceiling("external"), "external")
        # endorsed exceeds the ceiling -> refused
        with self.assertRaises(ValueError):
            memdag_ingest.enforce_channel_ceiling("endorsed")
        with self.assertRaises(ValueError):
            memdag_ingest.ingest_text(self.conn, "attacker endorsed text", "endorsed")

    def test_ceiling_accepts_numeric_value(self):
        os.environ[memdag_ingest.CHANNEL_CEILING_ENV] = "1"
        self.assertEqual(memdag_ingest.channel_ceiling(), 1)
        with self.assertRaises(ValueError):
            memdag_ingest.enforce_channel_ceiling("user")  # rank 2 > 1

    def test_ceiling_invalid_value_raises(self):
        os.environ[memdag_ingest.CHANNEL_CEILING_ENV] = "bogus"
        with self.assertRaises(ValueError):
            memdag_ingest.channel_ceiling()


if __name__ == "__main__":
    unittest.main()
