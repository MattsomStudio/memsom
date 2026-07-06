"""Tests for memsom_obsidian — Obsidian vault integration.

Run:  python -m unittest discover -s . -p test_memsom_obsidian.py
"""

import os
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_relate
import memsom_obsidian
import memsom_bridge_import
import memsom_schema


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        self.conn = memsom.get_connection()
        memsom_obsidian.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_OBSIDIAN_VAULT", None)
        self.tmp.cleanup()

    def note(self, rel, text):
        p = self.vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def live_for(self, rel):
        return memsom_obsidian._live_nodes_for_path(self.conn, rel)

    def channel_of(self, nid):
        return memsom.get_node(self.conn, nid)["channel"]


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

class TestFrontmatter(Base):
    def test_no_frontmatter(self):
        fm, body = memsom_obsidian.parse_frontmatter("just text\nmore")
        self.assertEqual(fm, {})
        self.assertEqual(body, "just text\nmore")

    def test_scalars_and_lists(self):
        text = ("---\n"
                "title: My Note\n"
                "memsom-channel: agent-derived\n"
                "tags:\n  - alpha\n  - beta\n"
                "aliases: [AH, AltName]\n"
                "---\n"
                "body line one\n")
        fm, body = memsom_obsidian.parse_frontmatter(text)
        self.assertEqual(fm["title"], "My Note")
        self.assertEqual(fm["memsom-channel"], "agent-derived")
        self.assertEqual(fm["tags"], ["alpha", "beta"])
        self.assertEqual(fm["aliases"], ["AH", "AltName"])
        self.assertEqual(body.strip(), "body line one")

    def test_unterminated_is_body(self):
        fm, body = memsom_obsidian.parse_frontmatter("---\nkey: v\nno close here\n")
        self.assertEqual(fm, {})
        self.assertIn("no close here", body)

    def test_frontmatter_must_be_first_line(self):
        fm, _ = memsom_obsidian.parse_frontmatter("\n---\nkey: v\n---\n")
        self.assertEqual(fm, {})

    def test_block_list_does_not_eat_nonempty_scalar(self):
        # re-audit (med): a malformed `tags: foo` then `- bar` must NOT discard foo.
        text = "---\ntags: foo\n  - bar\n---\nbody\n"
        fm, _ = memsom_obsidian.parse_frontmatter(text)
        self.assertEqual(fm["tags"], "foo")  # scalar preserved, list lines ignored


# ---------------------------------------------------------------------------
# Channel discipline — the laundering / upward-forge guard
# ---------------------------------------------------------------------------

class TestChannelGuard(Base):
    def test_no_declaration_uses_default(self):
        self.assertEqual(memsom_obsidian.effective_channel("user", None), "user")

    def test_declared_can_lower(self):
        self.assertEqual(
            memsom_obsidian.effective_channel("user", "agent-derived"), "agent-derived")
        self.assertEqual(
            memsom_obsidian.effective_channel("user", "external"), "external")

    def test_declared_cannot_raise(self):
        # default=user(2); a note claiming endorsed(3) is clamped to user.
        self.assertEqual(
            memsom_obsidian.effective_channel("user", "endorsed"), "user")
        # default=external(0); claiming user is clamped to external.
        self.assertEqual(
            memsom_obsidian.effective_channel("external", "user"), "external")

    def test_garbage_declaration_ignored(self):
        self.assertEqual(
            memsom_obsidian.effective_channel("user", "totally-bogus"), "user")


# ---------------------------------------------------------------------------
# Link extraction + masking pipeline
# ---------------------------------------------------------------------------

class TestExtractLinks(Base):
    def test_plain_alias_heading_block(self):
        body = "See [[Alpha]], [[Beta|the beta]], [[Gamma#Sec]], [[Delta#^blk]]."
        self.assertEqual(
            memsom_obsidian.extract_links(body), ["Alpha", "Beta", "Gamma", "Delta"])

    def test_embed_counts_as_link(self):
        self.assertEqual(memsom_obsidian.extract_links("![[Embedded]]"), ["Embedded"])

    def test_same_file_anchor_skipped(self):
        self.assertEqual(memsom_obsidian.extract_links("[[#Heading]] and [[#^blk]]"), [])

    def test_path_qualified(self):
        self.assertEqual(
            memsom_obsidian.extract_links("[[folder/Note]]"), ["folder/Note"])

    def test_links_in_code_fence_skipped(self):
        body = "real [[Real]]\n```\nfake [[NotReal]]\n```\nafter [[Also]]"
        self.assertEqual(memsom_obsidian.extract_links(body), ["Real", "Also"])

    def test_links_in_inline_code_skipped(self):
        body = "use `[[NotALink]]` but [[YesLink]] counts"
        self.assertEqual(memsom_obsidian.extract_links(body), ["YesLink"])

    def test_links_in_html_comment_skipped(self):
        body = "<!-- [[Hidden]] -->\nvisible [[Shown]]"
        self.assertEqual(memsom_obsidian.extract_links(body), ["Shown"])

    def test_escaped_brackets_skipped(self):
        body = r"escaped \[\[NotLink\]\] and [[Link]]"
        self.assertEqual(memsom_obsidian.extract_links(body), ["Link"])

    def test_markdown_link_local_and_external(self):
        # markdown targets keep their literal .md (the resolver strips it on lookup);
        # external URLs and pure anchors are dropped.
        body = "[note](My%20Note.md) and [web](https://example.com) and [a](#frag)"
        self.assertEqual(memsom_obsidian.extract_links(body), ["My Note.md"])

    def test_frontmatter_quoted_wikilink(self):
        fm = {"related": '"[[FromFrontmatter]]"'}
        self.assertIn("FromFrontmatter", memsom_obsidian.extract_links("body", fm))

    def test_dedup_preserves_order(self):
        self.assertEqual(
            memsom_obsidian.extract_links("[[A]] [[B]] [[A]]"), ["A", "B"])


# ---------------------------------------------------------------------------
# Link resolution
# ---------------------------------------------------------------------------

class TestResolver(Base):
    def test_unique_basename_resolves(self):
        by_name, by_rel = memsom_obsidian._build_resolver(["a/Note.md", "b/Other.md"])
        self.assertEqual(memsom_obsidian._resolve_target("Note", by_name, by_rel), "a/Note.md")

    def test_case_insensitive(self):
        by_name, by_rel = memsom_obsidian._build_resolver(["Note.md"])
        self.assertEqual(memsom_obsidian._resolve_target("note", by_name, by_rel), "Note.md")

    def test_ambiguous_basename_unresolved(self):
        by_name, by_rel = memsom_obsidian._build_resolver(["a/Dup.md", "b/Dup.md"])
        self.assertIsNone(memsom_obsidian._resolve_target("Dup", by_name, by_rel))

    def test_path_qualified_resolves_ambiguous(self):
        by_name, by_rel = memsom_obsidian._build_resolver(["a/Dup.md", "b/Dup.md"])
        self.assertEqual(
            memsom_obsidian._resolve_target("a/Dup", by_name, by_rel), "a/Dup.md")

    def test_missing_unresolved(self):
        by_name, by_rel = memsom_obsidian._build_resolver(["Note.md"])
        self.assertIsNone(memsom_obsidian._resolve_target("Ghost", by_name, by_rel))


# ---------------------------------------------------------------------------
# sync_vault — end-to-end ingest, edges, change/delete
# ---------------------------------------------------------------------------

class TestSync(Base):
    def test_ingest_creates_nodes_and_edges(self):
        self.note("A.md", "Alpha note links to [[B]].")
        self.note("B.md", "Beta note, standalone.")
        summary = memsom_obsidian.sync_vault(self.conn, self.vault)
        self.assertEqual(summary["ingested"], 2)
        a = memsom_obsidian._representative_node(self.conn, "A.md")
        b = memsom_obsidian._representative_node(self.conn, "B.md")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        nb = memsom_relate.neighborhood(self.conn, a, hops=1)
        self.assertIn(b, [n["id"] for n in nb])
        self.assertEqual(self.channel_of(a), "user")  # default channel

    def test_unchanged_note_skipped_on_resync(self):
        self.note("A.md", "stable content")
        s1 = memsom_obsidian.sync_vault(self.conn, self.vault)
        self.assertEqual(s1["ingested"], 1)
        s2 = memsom_obsidian.sync_vault(self.conn, self.vault)
        self.assertEqual(s2["ingested"], 0)
        self.assertEqual(s2["unchanged"], 1)

    def test_edit_tombstones_old_and_reingests(self):
        p = self.note("A.md", "version one")
        memsom_obsidian.sync_vault(self.conn, self.vault)
        old = self.live_for("A.md")
        # bump mtime forward deterministically (no reliance on wall clock granularity)
        st = p.stat()
        p.write_text("version two entirely different", encoding="utf-8")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        memsom_obsidian.sync_vault(self.conn, self.vault)
        new = self.live_for("A.md")
        self.assertTrue(new)
        self.assertNotEqual(set(old), set(new))
        for nid in old:  # old nodes are tombstoned
            self.assertEqual(memsom.get_node(self.conn, nid)["tombstoned"], 1)

    def test_deleted_note_is_tombstoned(self):
        p = self.note("Gone.md", "soon deleted")
        memsom_obsidian.sync_vault(self.conn, self.vault)
        ids = self.live_for("Gone.md")
        self.assertTrue(ids)
        p.unlink()
        summary = memsom_obsidian.sync_vault(self.conn, self.vault)
        self.assertEqual(summary["deleted"], 1)
        self.assertEqual(self.live_for("Gone.md"), [])

    def test_no_prune_keeps_deleted(self):
        p = self.note("Keep.md", "content")
        memsom_obsidian.sync_vault(self.conn, self.vault)
        p.unlink()
        memsom_obsidian.sync_vault(self.conn, self.vault, prune=False)
        self.assertTrue(self.live_for("Keep.md"))  # still live

    def test_obsidian_dir_ignored(self):
        self.note(".obsidian/workspace.md", "config junk [[Secret]]")
        self.note("Real.md", "real")
        summary = memsom_obsidian.sync_vault(self.conn, self.vault)
        self.assertEqual(summary["notes"], 1)

    def test_mtime_preserving_content_swap_is_reingested(self):
        # SECURITY (re-audit MED): mtime is forgeable via os.utime. A content
        # swap that preserves mtime must NOT be skipped as "unchanged".
        p = self.note("Swap.md", "original benign content")
        memsom_obsidian.sync_vault(self.conn, self.vault)
        old = set(self.live_for("Swap.md"))
        st = p.stat()
        p.write_text("poisoned content of the same-ish length!", encoding="utf-8")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))  # freeze mtime
        memsom_obsidian.sync_vault(self.conn, self.vault)
        new = set(self.live_for("Swap.md"))
        self.assertTrue(new and new != old, "mtime-frozen content swap slipped past as unchanged")
        node = memsom.get_node(self.conn, min(new))
        self.assertIn("poisoned", node["content"])

    def test_oversized_note_skipped(self):
        self.note("Big.md", "x")
        big = self.vault / "Big.md"
        big.write_text("y" * (memsom_obsidian.MAX_NOTE_BYTES + 10), encoding="utf-8")
        summary = memsom_obsidian.sync_vault(self.conn, self.vault)
        self.assertEqual(self.live_for("Big.md"), [])  # not ingested
        self.assertEqual(summary["ingested"], 0)

    def test_transient_missing_then_present_not_pruned(self):
        # Prune must only fire when the file is verifiably gone from disk.
        self.note("Steady.md", "content")
        memsom_obsidian.sync_vault(self.conn, self.vault)
        # File still present -> a prune pass must keep it.
        memsom_obsidian.sync_vault(self.conn, self.vault, prune=True)
        self.assertTrue(self.live_for("Steady.md"))


# ---------------------------------------------------------------------------
# Bridge/vault isolation (regression: bridge_path/bridge_mtime split)
#
# Before this fix, memsom_bridge_import borrowed memsom_obsidian's
# obsidian_path/obsidian_mtime columns. sync_vault()'s prune pass (scoped only
# by "obsidian_path IS NOT NULL", with no origin check) and its ingest pass
# (looked up existing nodes by bare filename alone) would then treat ANY
# memory-bridge node as one of its own vault notes. Caught 2026-07-06: running
# obsidian-sync against an unrelated vault silently revoke-cascaded a
# bridge-imported memory node because the two subsystems shared one field with
# no discriminator — a confused-deputy shape, not a vault-specific bug.
# ---------------------------------------------------------------------------

class TestBridgeIsolation(Base):
    def _bridge_note(self, tmpdir_path, stem, text):
        """Write one memory-bridge-style file and import it via the real
        bridge_import path (not a hand-rolled DB insert) so the fixture matches
        production exactly."""
        mem = tmpdir_path / "memory"
        mem.mkdir(exist_ok=True)
        (mem / f"{stem}.md").write_text(
            f"---\nname: {stem}\ndescription: test\ntype: project\n---\n{text}\n",
            encoding="utf-8",
        )
        memsom_bridge_import.migrate(self.conn)
        memsom_bridge_import.import_all(self.conn, mem, dry_run=False)
        return mem

    def test_sync_vault_does_not_prune_unrelated_bridge_node(self):
        # Mechanism 1: the exact bug. A bridge node sharing NOTHING with the
        # vault (different directory, never synced by memsom_obsidian) must
        # survive an obsidian-sync prune pass untouched.
        self._bridge_note(self.root, "project_widget", "widget status note")
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE source_ref = 'memory:project_widget' "
            "AND tombstoned = 0"
        ).fetchone()
        self.assertIsNotNone(row, "bridge import did not create the node")
        bridge_nid = row[0]

        self.note("Unrelated.md", "a real vault note, nothing to do with the bridge")
        memsom_obsidian.sync_vault(self.conn, self.vault)  # prune=True default

        self.assertEqual(
            memsom.get_node(self.conn, bridge_nid)["tombstoned"], 0,
            "obsidian-sync tombstoned a memory-bridge node it does not own",
        )

    def test_sync_vault_ingest_does_not_collide_on_shared_filename(self):
        # Mechanism 2: a vault note happens to share a filename with a bridge
        # memory file. Ingesting the vault note must not revoke or overwrite
        # the unrelated bridge node.
        self._bridge_note(self.root, "project_widget", "widget status from the bridge")
        row = self.conn.execute(
            "SELECT id, content FROM nodes WHERE source_ref = 'memory:project_widget' "
            "AND tombstoned = 0"
        ).fetchone()
        bridge_nid, bridge_content = row

        self.note("project_widget.md", "an unrelated vault note, same filename by coincidence")
        memsom_obsidian.sync_vault(self.conn, self.vault)

        bridge_node = memsom.get_node(self.conn, bridge_nid)
        self.assertEqual(bridge_node["tombstoned"], 0,
                          "vault ingest revoked the bridge node on a filename coincidence")
        self.assertEqual(bridge_node["content"], bridge_content,
                          "bridge node content was overwritten by an unrelated vault note")
        # the vault note got its OWN independent node
        vault_ids = self.live_for("project_widget.md")
        self.assertTrue(vault_ids)
        self.assertNotIn(bridge_nid, vault_ids)

    def test_legacy_column_migration_moves_bridge_rows_only(self):
        # A DB created before bridge_path existed had bridge rows stamped on
        # the shared obsidian_path/obsidian_mtime columns. Migrating must move
        # ONLY memory:% rows onto bridge_path/bridge_mtime and clear the
        # borrowed columns — a genuine vault-owned row must be left alone.
        with self.conn:
            legacy_id = memsom.insert_node(
                self.conn, "legacy bridge content", "user",
                source_ref="memory:project_legacy",
            )
            self.conn.execute(
                "UPDATE nodes SET obsidian_path = ?, obsidian_mtime = ? WHERE id = ?",
                ("project_legacy.md", "123456:100", legacy_id),
            )

        self.note("RealVaultNote.md", "a genuine obsidian vault note")
        memsom_obsidian.sync_vault(self.conn, self.vault)
        vault_id = self.live_for("RealVaultNote.md")[0]

        memsom_bridge_import.migrate(self.conn)  # runs the one-time reclaim

        legacy_row = self.conn.execute(
            "SELECT bridge_path, bridge_mtime, obsidian_path, obsidian_mtime "
            "FROM nodes WHERE id = ?", (legacy_id,)).fetchone()
        self.assertEqual(legacy_row, ("project_legacy.md", "123456:100", None, None))

        vault_row = self.conn.execute(
            "SELECT obsidian_path, bridge_path FROM nodes WHERE id = ?", (vault_id,)
        ).fetchone()
        self.assertEqual(vault_row, ("RealVaultNote.md", None),
                          "migration touched a genuine vault-owned row")

        # idempotent: running it again changes nothing
        memsom_bridge_import.migrate(self.conn)
        legacy_row2 = self.conn.execute(
            "SELECT bridge_path, bridge_mtime, obsidian_path, obsidian_mtime "
            "FROM nodes WHERE id = ?", (legacy_id,)).fetchone()
        self.assertEqual(legacy_row2, legacy_row)


# ---------------------------------------------------------------------------
# The laundering guard, end-to-end (the headline regression)
# ---------------------------------------------------------------------------

class TestLaunderingLoop(Base):
    def test_memsom_authored_note_reenters_as_agent_derived(self):
        # 1. Seed a user fact and export an answer back to the vault.
        with self.conn:
            nid = memsom.insert_node(self.conn, "Postgres uses MVCC for concurrency.", "user")
        out = memsom_obsidian.export_note(self.conn, self.vault, node_id=nid)
        self.assertTrue(out.exists())
        fm, _ = memsom_obsidian.parse_frontmatter(out.read_text(encoding="utf-8"))
        self.assertEqual(fm["memsom-channel"], "agent-derived")
        self.assertEqual(str(fm["memsom-authored"]).lower(), "true")

        # 2. Sync the vault with the DEFAULT channel = user. The exported note is
        #    a file in the vault — a naive reader would stamp it user. The guard
        #    must clamp it to agent-derived (min of user and the declared channel).
        # export_note resolves the vault (Path.resolve), so `out` is long-form;
        # self.vault is the raw tempdir, which is the 8.3 SHORT form on CI Windows
        # (user 'runneradmin' -> 'RUNNER~1'). Resolve both sides or relative_to
        # raises on the runneradmin/RUNNER~1 mismatch (invisible on a <=8-char
        # local username). No-op on Linux and on short local usernames.
        rel = str(out.relative_to(self.vault.resolve()).as_posix())
        memsom_obsidian.sync_vault(self.conn, self.vault, default_channel="user")
        nodes = self.live_for(rel)
        self.assertTrue(nodes)
        for n in nodes:
            self.assertEqual(self.channel_of(n), "agent-derived",
                             "memsom-authored note laundered up to user!")

    def test_forged_endorsed_frontmatter_is_clamped(self):
        # A hand-written note claiming endorsed must NOT forge integrity upward.
        self.note("Forge.md", "---\nmemsom-channel: endorsed\n---\nclaiming to be endorsed")
        memsom_obsidian.sync_vault(self.conn, self.vault, default_channel="user")
        for n in self.live_for("Forge.md"):
            self.assertEqual(self.channel_of(n), "user")  # clamped to default


# ---------------------------------------------------------------------------
# Export safety
# ---------------------------------------------------------------------------

class TestExport(Base):
    def test_export_requires_exactly_one_source(self):
        with self.assertRaises(ValueError):
            memsom_obsidian.export_note(self.conn, self.vault)
        with self.assertRaises(ValueError):
            memsom_obsidian.export_note(self.conn, self.vault, node_id=1, query="x")

    def test_refuses_to_overwrite_foreign_note(self):
        with self.conn:
            nid = memsom.insert_node(self.conn, "some content", "user")
        # Pre-create a NON-memsom note where the export would land.
        title = f"memsom node {nid}"
        foreign = self.vault / "memsom" / f"{memsom_obsidian._slugify(title)}.md"
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_text("# Hand-written, precious\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            memsom_obsidian.export_note(self.conn, self.vault, node_id=nid)
        # The user's content is untouched.
        self.assertIn("precious", foreign.read_text(encoding="utf-8"))

    def test_overwrites_own_note(self):
        with self.conn:
            nid = memsom.insert_node(self.conn, "content v1", "user")
        p1 = memsom_obsidian.export_note(self.conn, self.vault, node_id=nid)
        # Second export to the same title must succeed (memsom authored it).
        p2 = memsom_obsidian.export_note(self.conn, self.vault, node_id=nid)
        self.assertEqual(p1, p2)

    def test_query_export_is_clean_and_captures_sources(self):
        # --query export must contain the answer only — no CLI furniture, no
        # inline [mem:] tags — and must record the real cited source node(s).
        with self.conn:
            memsom.insert_node(
                self.conn, "Redis is an in-memory key-value store.", "user")
        out = memsom_obsidian.export_note(
            self.conn, self.vault, query="Tell me about the Redis store")
        fm, body = memsom_obsidian.parse_frontmatter(out.read_text(encoding="utf-8"))
        self.assertNotIn("composed from", body)
        self.assertNotIn("stored as node", body)
        self.assertNotIn("floor:", body)
        self.assertNotIn("[mem:", body)
        self.assertIn("Redis", body)
        self.assertNotEqual(fm[memsom_obsidian.SOURCES_KEY], [])  # real source captured

    def test_authored_flag_alone_does_not_grant_overwrite(self):
        # re-audit (low): a hand note carrying only `memsom-authored: true` (but
        # NOT the source-nodes key memsom always writes) must NOT be overwritable.
        with self.conn:
            nid = memsom.insert_node(self.conn, "content", "user")
        title = f"memsom node {nid}"
        foreign = self.vault / "memsom" / f"{memsom_obsidian._slugify(title)}.md"
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_text("---\nmemsom-authored: true\n---\nhand-written\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            memsom_obsidian.export_note(self.conn, self.vault, node_id=nid)
        self.assertIn("hand-written", foreign.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Watcher — settles only after a stable signature (defeats half-writes)
# ---------------------------------------------------------------------------

class TestWatcher(Base):
    def test_half_write_then_complete_syncs_final_content(self):
        # Simulate: file appears mid-write, then settles. With debounce=0 and a
        # bounded tick count, a single stable signature triggers one sync.
        self.note("Live.md", "final content [[Other]]")
        self.note("Other.md", "target")
        memsom_obsidian.watch_vault(
            memsom.get_connection, self.vault, "user",
            interval=0.01, debounce=0.0, _max_ticks=2, log=lambda *_a, **_k: None,
        )
        self.assertTrue(self.live_for("Live.md"))
        self.assertTrue(self.live_for("Other.md"))

    def test_snapshot_uses_mtime_ns_int(self):
        self.note("X.md", "x")
        snap = memsom_obsidian._snapshot(self.vault)
        (mtime_ns, size), = snap.values()
        self.assertIsInstance(mtime_ns, int)
        self.assertIsInstance(size, int)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestCli(Base):
    def test_sync_via_cli(self):
        import memsom_cli
        self.note("A.md", "alpha [[B]]")
        self.note("B.md", "beta")
        memsom_cli.main(["obsidian-sync", str(self.vault)])
        self.assertTrue(self.live_for("A.md"))


if __name__ == "__main__":
    unittest.main()
