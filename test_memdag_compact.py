#!/usr/bin/env python3
"""Tests for memdag_compact — the CONSOLIDATION ENGINE.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:/Users/you/memdag -p test_memdag_compact.py \
    -t C:/Users/you/memdag -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_schema
import memdag_compact
import memdag_quarantine
import memdag_redact
import memdag_federation
import memdag_confid


# ---------------------------------------------------------------------------
# Base (verbatim copy from test_memdag.py)
# ---------------------------------------------------------------------------

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])


# ---------------------------------------------------------------------------
# Fixture strings — Jaccard ~0.78 within cluster, ~0 across
# ---------------------------------------------------------------------------

EP1 = "Nebula lighthouse certificate rotation keeps mesh hosts trusted."
EP2 = "Nebula lighthouse certificate rotation keeps mesh hosts safe."
EP3 = "Nebula lighthouse certificate rotation keeps mesh devices trusted."
UNREL = "Quarterly budget spreadsheet totals for the financial tracker app."


# ---------------------------------------------------------------------------
# Mock urlopen helper (mirrors test_memdag_ingest.TestIngestUrl._make_mock_urlopen)
# ---------------------------------------------------------------------------

def _make_mock_urlopen(body: bytes):
    """Return a context-manager mock that yields a response with .read() -> body."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen = MagicMock(return_value=resp)
    return mock_urlopen


# ---------------------------------------------------------------------------
# 1. test_migrate_adds_archived_columns
# ---------------------------------------------------------------------------

class TestMigrateAddsArchivedColumns(Base):
    def test_migrate_adds_archived_columns(self):
        # Pre-existing row before migration
        nid = self.add("some content", "user")

        memdag_compact.migrate(self.conn)

        # Both columns must exist
        self.assertTrue(memdag_schema.column_exists(self.conn, "nodes", "archived"))
        self.assertTrue(memdag_schema.column_exists(self.conn, "nodes", "archived_at"))

        # archived default = 0 on the pre-existing row
        row = self.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (nid,)
        ).fetchone()
        self.assertEqual(row[0], 0)


# ---------------------------------------------------------------------------
# 2. test_migrate_idempotent
# ---------------------------------------------------------------------------

class TestMigrateIdempotent(Base):
    def test_migrate_idempotent(self):
        memdag_compact.migrate(self.conn)
        # Second call must not raise
        memdag_compact.migrate(self.conn)
        # Columns still exist
        self.assertTrue(memdag_schema.column_exists(self.conn, "nodes", "archived"))
        self.assertTrue(memdag_schema.column_exists(self.conn, "nodes", "archived_at"))


# ---------------------------------------------------------------------------
# 3. test_compact_mints_semantic_with_edges_to_every_episode
# ---------------------------------------------------------------------------

class TestCompactMintsSemanticWithEdgesToEveryEpisode(Base):
    def test_compact_mints_semantic_with_edges_to_every_episode(self):
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        minted = memdag_compact.compact(self.conn)

        self.assertEqual(len(minted), 1)
        semantic_id = minted[0]

        parent_ids = {p[0] for p in memdag.parents_of(self.conn, semantic_id)}
        self.assertEqual(parent_ids, {n1, n2, n3})


# ---------------------------------------------------------------------------
# 4. test_label_is_min_of_parents_no_laundering
# ---------------------------------------------------------------------------

class TestLabelIsMinOfParentsNoLaundering(Base):
    def test_label_is_min_of_parents_no_laundering(self):
        # user(2), user(2), external(0) -> label = min(2,2,0) = 0 (EXTERNAL)
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "external")

        minted = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted), 1)

        node = memdag.get_node(self.conn, minted[0])
        self.assertEqual(node["label"], 0)


# ---------------------------------------------------------------------------
# 5. test_episodes_archived_not_deleted
# ---------------------------------------------------------------------------

class TestEpisodesArchivedNotDeleted(Base):
    def test_episodes_archived_not_deleted(self):
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")
        ep_ids = {n1, n2, n3}
        ep_contents = {
            n: self.conn.execute(
                "SELECT content FROM nodes WHERE id=?", (n,)
            ).fetchone()[0]
            for n in ep_ids
        }

        pre_node_count = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        pre_edge_count = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        minted = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted), 1)
        semantic_id = minted[0]

        # Counts: +1 node (semantic), +3 edges (to each episode)
        post_node_count = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        post_edge_count = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(post_node_count, pre_node_count + 1)
        self.assertEqual(post_edge_count, pre_edge_count + len(ep_ids))

        # Every episode row intact: content equal, tombstoned=0, archived=1, archived_at not None
        for nid in ep_ids:
            row = self.conn.execute(
                "SELECT content, tombstoned, archived, archived_at FROM nodes WHERE id=?",
                (nid,),
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], ep_contents[nid])
            self.assertEqual(row[1], 0)   # tombstoned = 0
            self.assertEqual(row[2], 1)   # archived = 1
            self.assertIsNotNone(row[3])  # archived_at not None


# ---------------------------------------------------------------------------
# 6. test_external_tainted_group_quarantined
# ---------------------------------------------------------------------------

class TestExternalTaintedGroupQuarantined(Base):
    def test_external_tainted_group_quarantined(self):
        # Group with one external node -> semantic label=0 -> quarantined by gate
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "external")

        minted = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted), 1)

        status = self.conn.execute(
            "SELECT status FROM nodes WHERE id=?", (minted[0],)
        ).fetchone()[0]
        self.assertEqual(status, "quarantined")


# ---------------------------------------------------------------------------
# 7. test_clean_group_stays_live
# ---------------------------------------------------------------------------

class TestCleanGroupStaysLive(Base):
    def test_clean_group_stays_live(self):
        # All-user group -> semantic status = 'live'
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        minted = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted), 1)

        status = self.conn.execute(
            "SELECT status FROM nodes WHERE id=?", (minted[0],)
        ).fetchone()[0]
        self.assertEqual(status, "live")


# ---------------------------------------------------------------------------
# 8. test_extractive_summary_deterministic
# ---------------------------------------------------------------------------

class TestExtractiveSummaryDeterministic(Base):
    def test_extractive_summary_deterministic(self):
        rows = [(1, EP1), (2, EP2), (3, EP3)]

        result1 = memdag_compact.extractive_summary(rows)
        result2 = memdag_compact.extractive_summary(rows)
        self.assertEqual(result1, result2, "extractive_summary must be byte-identical on identical input")

    def test_compact_deterministic_across_fresh_dbs(self):
        """Two compact() runs on two fresh identical DBs produce identical semantic content."""
        # DB 1 (self.conn)
        n1a = self.add(EP1, "user")
        n2a = self.add(EP2, "user")
        n3a = self.add(EP3, "user")
        minted_a = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted_a), 1)
        content_a = memdag.get_node(self.conn, minted_a[0])["content"]

        # DB 2 (separate temp DB)
        tmp2 = tempfile.TemporaryDirectory()
        try:
            db2 = Path(tmp2.name) / "sub2" / "test2.db"
            os.environ["MEMDAG_DB"] = str(db2)
            conn2 = memdag.get_connection()
            try:
                with conn2:
                    memdag.insert_node(conn2, EP1, "user", memdag.RANK["user"])
                    memdag.insert_node(conn2, EP2, "user", memdag.RANK["user"])
                    memdag.insert_node(conn2, EP3, "user", memdag.RANK["user"])
                minted_b = memdag_compact.compact(conn2)
                self.assertEqual(len(minted_b), 1)
                content_b = memdag.get_node(conn2, minted_b[0])["content"]
            finally:
                conn2.close()
        finally:
            tmp2.cleanup()
            os.environ["MEMDAG_DB"] = str(self.db)

        self.assertEqual(content_a, content_b,
                         "compact() must produce byte-identical content on identical inputs")


# ---------------------------------------------------------------------------
# 9. test_compact_second_run_noop
# ---------------------------------------------------------------------------

class TestCompactSecondRunNoop(Base):
    def test_compact_second_run_noop(self):
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        minted_first = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted_first), 1)

        # Second run: archived episodes no longer candidates
        minted_second = memdag_compact.compact(self.conn)
        self.assertEqual(minted_second, [], "second compact() must return [] (idempotent)")


# ---------------------------------------------------------------------------
# 10. test_min_group_respected
# ---------------------------------------------------------------------------

class TestMinGroupRespected(Base):
    def test_min_group_respected(self):
        # Single unrelated node alone — with min_group=2, must not be compacted
        n = self.add(UNREL, "user")

        minted = memdag_compact.compact(self.conn, min_group=2)
        self.assertEqual(minted, [])

        # Episode must NOT be archived
        row = self.conn.execute(
            "SELECT archived FROM nodes WHERE id=?", (n,)
        ).fetchone()
        self.assertEqual(row[0], 0)


# ---------------------------------------------------------------------------
# 11. test_llm_falls_back_to_extractive_when_ollama_unreachable
# ---------------------------------------------------------------------------

class TestLlmFallsBackToExtractive(Base):
    def test_llm_falls_back_to_extractive_when_ollama_unreachable(self):
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        expected_rows = [(n1, EP1), (n2, EP2), (n3, EP3)]
        expected_summary = memdag_compact.extractive_summary(expected_rows)

        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("refused")):
            minted = memdag_compact.compact(self.conn, llm=True)

        self.assertEqual(len(minted), 1)
        content = memdag.get_node(self.conn, minted[0])["content"]
        self.assertEqual(content, expected_summary)


# ---------------------------------------------------------------------------
# 12. test_llm_used_when_reachable
# ---------------------------------------------------------------------------

class TestLlmUsedWhenReachable(Base):
    def test_llm_used_when_reachable(self):
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        llm_response = b'{"response": "- llm bullet one"}'
        mock_open = _make_mock_urlopen(llm_response)

        with patch("urllib.request.urlopen", mock_open):
            minted = memdag_compact.compact(self.conn, llm=True)

        self.assertEqual(len(minted), 1)
        content = memdag.get_node(self.conn, minted[0])["content"]
        self.assertIn("llm bullet one", content)


# ---------------------------------------------------------------------------
# 13. test_archived_excluded_from_default_retrieval
# ---------------------------------------------------------------------------

class TestArchivedExcludedFromDefaultRetrieval(Base):
    def test_archived_excluded_from_default_retrieval(self):
        import memdag_retrieve

        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        memdag_retrieve.index_all(self.conn)

        results_before = memdag_retrieve.retrieve(
            self.conn, "nebula lighthouse certificate"
        )
        result_ids_before = {r[0] for r in results_before}
        # At least one episode should be in results before compaction
        self.assertTrue(
            result_ids_before & {n1, n2, n3},
            "At least one episode should appear in retrieval before compact",
        )

        memdag_compact.compact(self.conn)

        results_after = memdag_retrieve.retrieve(
            self.conn, "nebula lighthouse certificate"
        )
        result_ids_after = {r[0] for r in results_after}

        # No archived episode id should appear in results
        for ep_id in (n1, n2, n3):
            self.assertNotIn(
                ep_id, result_ids_after,
                f"Archived episode [{ep_id}] should not appear in retrieval after compact",
            )


# ---------------------------------------------------------------------------
# 14. test_explain_and_blame_still_walk_archived
# ---------------------------------------------------------------------------

class TestExplainAndBlameStillWalkArchived(Base):
    def test_explain_and_blame_still_walk_archived(self):
        import memdag_cli

        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        minted = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted), 1)
        semantic = minted[0]

        # parents_of still returns the archived rows
        parent_ids = {p[0] for p in memdag.parents_of(self.conn, semantic)}
        self.assertEqual(parent_ids, {n1, n2, n3})

        # explain output contains each episode id
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memdag_cli.main(["explain", str(semantic)])
        explain_out = buf.getvalue()
        for ep_id in (n1, n2, n3):
            self.assertIn(f"[{ep_id}]", explain_out)

        # blame output contains each episode id
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            memdag_cli.main(["blame", str(semantic)])
        blame_out = buf2.getvalue()
        for ep_id in (n1, n2, n3):
            self.assertIn(f"[{ep_id}]", blame_out)


# ---------------------------------------------------------------------------
# 15. test_group_by_claim
# ---------------------------------------------------------------------------

class TestGroupByClaim(Base):
    def test_group_by_claim(self):
        import memdag_corroborate

        memdag_corroborate.migrate(self.conn)
        memdag_corroborate.register_root(self.conn, "rootA", by="test")
        memdag_corroborate.register_root(self.conn, "rootB", by="test")

        n1 = self.add("Service listens on port 4242 inside the mesh.", "user")
        n2 = self.add("Port 4242 is used by Nebula for tunnel traffic.", "user")

        claim_id = memdag_corroborate.assert_claim(
            self.conn, n1, ("port", "is", "4242"), "rootA"
        )
        memdag_corroborate.assert_claim(
            self.conn, n2, ("port", "is", "4242"), "rootB"
        )

        minted = memdag_compact.compact(self.conn, group_by="claim")
        self.assertEqual(len(minted), 1)

        parent_ids = {p[0] for p in memdag.parents_of(self.conn, minted[0])}
        self.assertEqual(parent_ids, {n1, n2})


# ---------------------------------------------------------------------------
# 16. test_group_by_claim_without_tables_returns_empty
# ---------------------------------------------------------------------------

class TestGroupByClaimWithoutTablesReturnsEmpty(Base):
    def test_group_by_claim_without_tables_returns_empty(self):
        # Fresh DB — no corroborate migrate called
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")

        # compact(group_by='claim') must return [] gracefully
        minted = memdag_compact.compact(self.conn, group_by="claim")
        self.assertEqual(minted, [])


# ---------------------------------------------------------------------------
# 17. test_invalid_group_by_raises
# ---------------------------------------------------------------------------

class TestInvalidGroupByRaises(Base):
    def test_invalid_group_by_raises(self):
        with self.assertRaises(ValueError):
            memdag_compact.compact(self.conn, group_by="bogus")


# ---------------------------------------------------------------------------
# 18. test_register_mounts_compact_and_archived_list
# ---------------------------------------------------------------------------

class TestRegisterMountsSubcommands(unittest.TestCase):
    def test_register_mounts_compact_and_archived_list(self):
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        memdag_compact.register(sub)

        # compact with all options
        args = p.parse_args(["compact", "--group-by", "claim", "--min-group", "3"])
        self.assertEqual(args.group_by, "claim")
        self.assertEqual(args.min_group, 3)
        self.assertTrue(hasattr(args, "func"))

        # archived-list
        args2 = p.parse_args(["archived-list"])
        self.assertTrue(hasattr(args2, "func"))


# ---------------------------------------------------------------------------
# 19. test_list_archived
# ---------------------------------------------------------------------------

class TestListArchived(Base):
    def test_list_archived(self):
        n1 = self.add(EP1, "user")
        n2 = self.add(EP2, "user")
        n3 = self.add(EP3, "user")

        minted = memdag_compact.compact(self.conn)
        self.assertEqual(len(minted), 1)

        archived = memdag_compact.list_archived(self.conn)
        archived_ids = {r["id"] for r in archived}
        self.assertEqual(archived_ids, {n1, n2, n3})

        for r in archived:
            self.assertIn("id", r)
            self.assertIn("channel", r)
            self.assertIn("label", r)
            self.assertIn("archived_at", r)
            self.assertIn("content", r)
            self.assertIsNotNone(r["archived_at"])


# ---------------------------------------------------------------------------
# 20. test_redact_episode_scrubs_semantic (regression for F-06 / poc2_compact_leak.py)
# ---------------------------------------------------------------------------

class TestRedactEpisodeScrubsSemantic(Base):
    """Regression: mirrors poc2_compact_leak.py — redacting an episode after compact()
    must cascade to the semantic node that baked its verbatim sentences.

    F-06: compact() copies verbatim sentences into the summary node; non-cascade
    source redaction left the secret in the summary. With cascade=True as the default,
    redacting an episode wipes the semantic node too.
    """

    def test_redact_episode_scrubs_semantic(self):
        memdag_redact.migrate(self.conn)
        memdag_federation.migrate(self.conn)

        SECRET_TOKEN = "alpha-key-99"
        # SECRET_TOKEN appears ONLY in source A; SECRET_B shares tokens for clustering
        # but does NOT contain the secret token itself.
        SECRET_A = f"TopSecret: {SECRET_TOKEN} credential must not be shared externally ever."
        SECRET_B = "TopSecret: beta-key-77 credential must not be shared externally either."

        id_a = self.add(SECRET_A, "user")
        id_b = self.add(SECRET_B, "user")

        # Compact with a low sim_threshold to guarantee clustering
        minted = memdag_compact.compact(
            self.conn, group_by="similarity", min_group=2, sim_threshold=0.1
        )
        self.assertTrue(minted, "compact() must mint at least one semantic node")
        semantic_id = minted[0]

        # Semantic node must contain the secret before redaction
        semantic_pre = self.conn.execute(
            "SELECT content FROM nodes WHERE id=?", (semantic_id,)
        ).fetchone()[0]
        self.assertIn(
            SECRET_TOKEN, semantic_pre,
            "semantic node must contain the secret before redaction"
        )

        # Redact source A with the default cascade (no cascade= kwarg)
        newly = memdag_redact.redact_node(self.conn, id_a, "audit-poc2")
        self.assertIn(id_a, newly)
        self.assertIn(
            semantic_id, newly,
            "semantic node must be cascade-redacted when its episode is redacted"
        )

        # Semantic node content must now be empty and marked redacted
        sem_row = self.conn.execute(
            "SELECT content, redacted FROM nodes WHERE id=?", (semantic_id,)
        ).fetchone()
        self.assertEqual(sem_row[0], "", "semantic node content must be wiped after cascade redaction")
        self.assertEqual(sem_row[1], 1, "semantic node must be marked redacted=1")

        # Secret must not appear in ANY node's content
        all_contents = self.conn.execute("SELECT content FROM nodes").fetchall()
        for (c,) in all_contents:
            self.assertNotIn(
                SECRET_TOKEN, c,
                f"secret '{SECRET_TOKEN}' leaked in node content: {c!r}"
            )

        # Edges from semantic node to its episodes must still exist
        parent_ids = {p[0] for p in memdag.parents_of(self.conn, semantic_id)}
        self.assertIn(id_a, parent_ids, "semantic->episode_a edge must survive redaction")
        self.assertIn(id_b, parent_ids, "semantic->episode_b edge must survive redaction")

        # Federation export must not carry the secret in any non-redacted node
        changeset = memdag_federation.export_changeset(self.conn, origin="test-machine")
        for node_rec in changeset.get("nodes", []):
            if node_rec.get("redacted", 0) == 0:
                self.assertNotIn(
                    SECRET_TOKEN, node_rec.get("content", ""),
                    f"secret '{SECRET_TOKEN}' leaked in federation export for "
                    f"non-redacted node uuid={node_rec.get('uuid')}"
                )


# ---------------------------------------------------------------------------
# 21. TestCompactConfHighWater — regression for PoC 03 (compact conf laundering)
# ---------------------------------------------------------------------------

class TestCompactConfHighWater(Base):
    def test_compact_propagates_secret_conf_not_public(self):
        s1 = self.add('Nebula mesh secret key rotation procedure revoke certs quarterly', 'user')
        s2 = self.add('Nebula mesh secret key rotation the CA private key path quarterly', 'user')
        memdag_confid.migrate(self.conn)
        memdag_confid.classify(self.conn, s1, 'secret')   # conf=2
        memdag_confid.classify(self.conn, s2, 'secret')   # conf=2
        minted = memdag_compact.compact(self.conn, group_by='similarity', min_group=2, sim_threshold=0.0)
        self.assertTrue(minted)
        nid = minted[0]
        conf = self.conn.execute('SELECT conf_label FROM nodes WHERE id=?', (nid,)).fetchone()[0]
        self.assertEqual(conf, 2)   # SECRET high-water, NOT 0/PUBLIC (laundering blocked)
        # Spine invariants still hold:
        node = memdag.get_node(self.conn, nid)
        parents = [p[0] for p in memdag.parents_of(self.conn, nid)]
        self.assertEqual(sorted(parents), sorted([s1, s2]))         # edge to every episode
        self.assertEqual(node['label'], min(memdag.RANK['user'], memdag.RANK['user']))  # min(parents)
        for sid in (s1, s2):
            arch = self.conn.execute('SELECT archived, tombstoned, content FROM nodes WHERE id=?', (sid,)).fetchone()
            self.assertEqual(arch[0], 1)          # archived
            self.assertEqual(arch[1], 0)          # NOT tombstoned
            self.assertNotEqual(arch[2], None)    # not deleted

    def test_public_clearance_does_not_surface_secret_summary(self):
        s1 = self.add('alpha mesh secret rotation key material quarterly cadence', 'user')
        s2 = self.add('alpha mesh secret rotation key material quarterly schedule', 'user')
        memdag_confid.migrate(self.conn)
        memdag_confid.classify(self.conn, s1, 'secret')
        memdag_confid.classify(self.conn, s2, 'secret')
        memdag_compact.compact(self.conn, group_by='similarity', min_group=2, sim_threshold=0.0)
        import memdag_retrieve
        hits = memdag_retrieve.retrieve(self.conn, 'mesh secret rotation key', clearance='public')
        # No public-clearance hit may carry the secret material (defense-in-depth:
        # derived nodes are excluded from the retrieve pool AND conf is now SECRET).
        for (_id, content, _ch, _lbl, _sr) in hits:
            self.assertNotIn('key material', content or '')


if __name__ == "__main__":
    unittest.main()
