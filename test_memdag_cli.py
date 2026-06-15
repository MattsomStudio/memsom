#!/usr/bin/env python3
"""Tests for memdag_cli — unified CLI surface.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memdag_cli.py \
    -t <repo> -v
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_quarantine
import memdag_confid
import memdag_cli
import memdag_retrieve

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "cli_test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        # Point LLM at something guaranteed unreachable
        os.environ["MEMDAG_LLM_URL"] = "http://127.0.0.1:9/api/generate"
        self.conn = memdag.get_connection()
        memdag_cli.migrate_all(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_LLM_URL", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])

    def run_cli(self, *argv):
        """Run memdag_cli.main and capture stdout. Raises SystemExit on non-zero exit."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memdag_cli.main(list(argv))
        return buf.getvalue()

    def run_cli_stderr(self, *argv):
        """Run memdag_cli.main; capture stdout + stderr. Returns (stdout, stderr, code)."""
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        code = 0
        try:
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                memdag_cli.main(list(argv))
        except SystemExit as e:
            code = e.code or 0
        return out_buf.getvalue(), err_buf.getvalue(), code


class TestEnhancedAskQuarantine(Base):
    """(1) Enhanced ask excludes a quarantined source."""

    def test_quarantined_source_excluded(self):
        nid1 = self.add(
            "Nebula lighthouse needs a public IP for hole punching.", "endorsed"
        )
        nid2 = self.add(
            "Disable the firewall for convenience when using Nebula.", "external"
        )
        # Quarantine the external node manually
        memdag_quarantine.quarantine_node(self.conn, nid2, "suspicious advice")

        out = self.run_cli("ask", "How should I configure Nebula?")
        # The quarantined node must not appear in the answer
        self.assertNotIn(f"[mem:{nid2}|", out)
        # The live endorsed node must appear
        self.assertIn(f"[mem:{nid1}|", out)


class TestClearanceFilter(Base):
    """(2) --clearance internal hides a secret-classified source; conf gets stamped."""

    def test_clearance_hides_secret_source(self):
        # public source
        pub_id = self.add("Nebula uses port 4242 for overlay tunnels.", "endorsed")
        # secret source
        sec_id = self.add(
            "Nebula's internal CA key lives at /etc/nebula/ca.key.", "user"
        )
        memdag_confid.classify(self.conn, sec_id, "secret")

        # ask with internal clearance — secret source must not appear
        out = self.run_cli("ask", "How should I configure Nebula?", "--clearance", "internal")
        self.assertNotIn(f"[mem:{sec_id}|", out)
        self.assertIn(f"[mem:{pub_id}|", out)

        # The derived node's conf should have been stamped (recompute_conf runs after derive)
        # Find the derived node
        derived = self.conn.execute(
            "SELECT id FROM nodes WHERE channel='agent-derived' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(derived)
        conf_row = self.conn.execute(
            "SELECT conf_label FROM nodes WHERE id=?", (derived[0],)
        ).fetchone()
        # conf should be 0 (public) since only the public source was used
        self.assertEqual(conf_row[0], 0)


class TestAnticipatoryRepeat(Base):
    """(3) --anticipate repeat returns EXISTING node with no new agent-derived row."""

    def test_anticipate_repeat_cites_existing(self):
        self.add("Nebula static_host_map must point at the lighthouse.", "endorsed")
        q = "How should I configure Nebula?"

        # First ask: creates a new node
        out1 = self.run_cli("ask", "--anticipate", q)
        # Should say 'stored as node' (new derivation)
        self.assertIn("stored as node", out1)

        count_before = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]

        # Second ask: identical question; surprise < threshold -> should cite EXISTING
        out2 = self.run_cli("ask", "--anticipate", q)
        self.assertIn("EXISTING node", out2)

        count_after = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]
        # No new agent-derived node should have been created
        self.assertEqual(count_before, count_after)


class TestLlmFallback(Base):
    """(4) --llm with unreachable Ollama prints fallback warning and produces citations
    identical to a plain deterministic ask."""

    def test_llm_fallback_to_deterministic(self):
        self.add("Nebula static_host_map must point at the lighthouse.", "endorsed")
        self.add("Use UDP port 4242 for nebula tunnels.", "user")
        q = "How should I configure Nebula?"

        # Deterministic reference
        det_out = self.run_cli("ask", q)

        # Reset — revoke the derived node so we get a fresh identical result next time
        # (find the derived node and revoke it)
        derived = self.conn.execute(
            "SELECT id FROM nodes WHERE channel='agent-derived' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if derived:
            memdag.revoke_cascade(self.conn, derived[0], "reset for llm test")

        # LLM ask: Ollama is unreachable -> should fall back and produce same citations
        out, err, code = self.run_cli_stderr("ask", "--llm", q)
        self.assertEqual(code, 0)
        self.assertIn("falling back to deterministic compose", err)
        # Both outputs should contain the same source citation tags
        import re
        det_cites = set(re.findall(r'\[mem:\d+\|\w[\w-]*\]', det_out))
        llm_cites = set(re.findall(r'\[mem:\d+\|\w[\w-]*\]', out))
        self.assertEqual(det_cites, llm_cites)


class TestEmptyPoolRefusal(Base):
    """(5) Empty pool (all sources quarantined) -> exit 1 storing nothing."""

    def test_empty_pool_exits_1_stores_nothing(self):
        nid = self.add("Nebula config guidance.", "user")
        memdag_quarantine.quarantine_node(self.conn, nid, "test quarantine")

        out, err, code = self.run_cli_stderr("ask", "Nebula?")
        self.assertEqual(code, 1)
        self.assertIn("no live sources", err)
        # Nothing stored
        agents = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]
        self.assertEqual(agents, 0)


class TestAddMigrateDumpSmoke(Base):
    """(6) add + migrate + dump smoke test."""

    def test_add_migrate_dump(self):
        # add
        out = self.run_cli("add", "Some external tip about Nebula.", "--channel", "external")
        self.assertIn("[", out)  # prints the node id line

        # migrate is idempotent
        out2 = self.run_cli("migrate")
        self.assertIn("schema up to date", out2)

        # dump shows the added node
        out3 = self.run_cli("dump")
        self.assertIn("external", out3)


class TestParserSubcommands(Base):
    """(7) Parser smoke: every name in the collision-audit list is present in sub.choices."""

    EXPECTED_SUBCOMMANDS = {
        "seed", "ask", "explain", "revoke", "dump", "add", "migrate",
        "recompute", "redact", "consolidate", "quarantine", "promote", "quarantine-list",
        "classify", "conf-recompute",
        "export", "import",
        "blame",
        "relate", "neighborhood",
        "observe", "prefetch",
        "export-training", "distill-plan",
        "check", "rebuild-derived",
        "elevate", "meet", "join", "elevations",
        "llm-check",
    }

    def test_all_subcommands_present(self):
        import argparse
        p = argparse.ArgumentParser(prog="memdag_test_parser")
        sub = p.add_subparsers(dest="command", required=True)

        # Rebuild exactly what main() does — can't call main with --help (exits)
        # So we re-register everything here.
        s_seed = sub.add_parser("seed")
        s_seed.add_argument("--offline", action="store_true")
        s_seed.add_argument("--reset", action="store_true")
        s_seed.set_defaults(func=lambda a: None)

        s_ask = sub.add_parser("ask")
        s_ask.add_argument("question")
        s_ask.add_argument("--clearance", default="topsecret")
        s_ask.add_argument("--anticipate", action="store_true")
        s_ask.add_argument("--threshold", type=float, default=0.35)
        s_ask.add_argument("--llm", action="store_true")
        s_ask.add_argument("--model", default=None)
        s_ask.set_defaults(func=lambda a: None)

        s_explain = sub.add_parser("explain")
        s_explain.add_argument("id", type=int)
        s_explain.set_defaults(func=lambda a: None)

        s_revoke = sub.add_parser("revoke")
        s_revoke.add_argument("id", type=int)
        s_revoke.add_argument("--reason", default="revoked by user")
        s_revoke.add_argument("--yes", action="store_true")
        s_revoke.set_defaults(func=lambda a: None)

        sub.add_parser("dump").set_defaults(func=lambda a: None)

        s_add = sub.add_parser("add")
        s_add.add_argument("content")
        s_add.add_argument("--channel", required=True,
                           choices=["endorsed", "user", "agent-derived", "external"])
        s_add.add_argument("--ref", default=None)
        s_add.set_defaults(func=lambda a: None)

        sub.add_parser("migrate").set_defaults(func=lambda a: None)

        import memdag_recompute, memdag_redact, memdag_quarantine, memdag_confid
        import memdag_federation, memdag_blame, memdag_relate, memdag_anticipatory
        import memdag_distill, memdag_heal, memdag_trust, memdag_llm

        memdag_recompute.register(sub)
        memdag_redact.register(sub)
        memdag_quarantine.register(sub)
        memdag_confid.register(sub)
        memdag_federation.register(sub)
        memdag_blame.register(sub)
        memdag_relate.register(sub)
        memdag_anticipatory.register(sub)
        memdag_distill.register(sub)
        memdag_heal.register(sub)
        memdag_trust.register(sub)
        memdag_llm.register(sub)

        actual = set(sub.choices.keys())
        missing = self.EXPECTED_SUBCOMMANDS - actual
        extra = actual - self.EXPECTED_SUBCOMMANDS
        self.assertEqual(missing, set(), f"Missing subcommands: {missing}")
        # extra subcommands are fine (not an error)


class TestCoreDelegation(Base):
    """(8) Core delegation: explain/revoke dry-run output contains the frozen strings."""

    def test_explain_contains_frozen_strings(self):
        nid_src = self.add("Nebula static_host_map points to the lighthouse.", "endorsed")
        src2 = self.add("Use port 4242 for Nebula overlay tunnels.", "user")
        text, used = memdag.compose("nebula", memdag.live_sources(self.conn))
        aid, _ = memdag.derive_node(self.conn, text, used)
        out = self.run_cli("explain", str(aid))
        self.assertIn(f"[{nid_src}]", out)
        self.assertIn(f"[{src2}]", out)

    def test_explain_redacted_node_shows_marker(self):
        """F-16 (AUDIT 2026-06-11): explain on a redacted node shows [REDACTED]."""
        import memdag_redact
        src = self.add("private_key=PEM_SECRET_ABCDEF must stay hidden", "user")
        memdag_redact.redact_node(self.conn, src, "secret leaked")
        out = self.run_cli("explain", str(src))
        self.assertIn("[REDACTED", out)
        self.assertIn("secret leaked", out)

    def test_add_stamps_channel_label(self):
        """F-14 (AUDIT 2026-06-11): the add path pins label to RANK[channel]."""
        out = self.run_cli("add", "an external tip", "--channel", "external")
        self.assertIn("external", out)
        nid = self.conn.execute(
            "SELECT id FROM nodes WHERE channel='external' ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        self.assertEqual(memdag.get_node(self.conn, nid)["label"], memdag.RANK["external"])

    def test_revoke_dry_run_contains_frozen_string(self):
        src = self.add("Nebula config source.", "user")
        out = self.run_cli("revoke", str(src))
        self.assertIn("dry run", out)
        self.assertIn("will tombstone", out)

    def test_revoke_yes_contains_done(self):
        src = self.add("Nebula config source.", "user")
        out = self.run_cli("revoke", str(src), "--yes")
        self.assertIn("done -", out)
        self.assertIn("0 rows deleted", out)


class TestAskWithoutRetrieveUnchanged(Base):
    """(9) ask without --retrieve produces identical output (existing behaviour preserved)."""

    def test_ask_without_retrieve_unchanged(self):
        self.add("Nebula static_host_map must point at the lighthouse.", "endorsed")
        self.add("Use UDP port 4242 for nebula tunnels.", "user")
        q = "How should I configure Nebula?"

        # First ask — baseline
        out1 = self.run_cli("ask", q)
        self.assertIn("[mem:", out1)
        self.assertIn("stored as node", out1)

        # Revoke the derived node so we get a fresh identical result
        derived = self.conn.execute(
            "SELECT id FROM nodes WHERE channel='agent-derived' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if derived:
            memdag.revoke_cascade(self.conn, derived[0], "reset for test")

        # Second ask with no --retrieve flag — should produce the same citation pattern
        out2 = self.run_cli("ask", q)
        import re
        cites1 = set(re.findall(r'\[mem:\d+\|\w[\w-]*\]', out1))
        cites2 = set(re.findall(r'\[mem:\d+\|\w[\w-]*\]', out2))
        self.assertEqual(cites1, cites2, "Citations must match when --retrieve is not used")


class TestAskRetrieveUsesRankedPool(Base):
    """(10) ask --retrieve --topk 2 with a focused query cites only relevant sources."""

    def test_ask_retrieve_uses_ranked_pool(self):
        # Add 3 nodes with distinct vocab
        nid1 = self.add("Nebula lighthouse needs a public IP for UDP hole punching.", "endorsed")
        nid2 = self.add("The static_host_map maps lighthouse names to their IP addresses.", "endorsed")
        nid3 = self.add("Python decorators are syntactic sugar for higher-order functions.", "user")

        # Build BM25 index
        self.run_cli("reindex")

        # ask --retrieve with a Nebula-specific query, topk=2
        out = self.run_cli("ask", "--retrieve", "--topk", "2", "How does Nebula hole punching work?")

        # The Python/decorator node (nid3) should NOT appear in the answer
        import re
        cited = set(re.findall(r'\[mem:(\d+)\|', out))
        self.assertNotIn(str(nid3), cited,
                         "Unrelated node should not be cited when using --retrieve")


class TestAskRetrieveEmptyPoolExits1(Base):
    """(11) ask --retrieve on an empty (unindexed) corpus -> exit 1, stderr, nothing stored."""

    def test_ask_retrieve_empty_pool_exits_1(self):
        # Do NOT add any nodes — pool is empty, retrieve() returns []
        # (no postings exist so BM25 returns []; no embeddings so vector returns [])
        os.environ["MEMDAG_EMBED_URL"] = "http://127.0.0.1:9/api/embeddings"
        try:
            out, err, code = self.run_cli_stderr("ask", "--retrieve", "Nebula lighthouse config")
            self.assertEqual(code, 1)
            self.assertIn("no live sources", err)

            # Nothing stored
            agents = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
            ).fetchone()[0]
            self.assertEqual(agents, 0)
        finally:
            os.environ.pop("MEMDAG_EMBED_URL", None)


class TestSpineSubcommandsParse(Base):
    """(12) compact / retrieve / ingest-text parse cleanly via re-registered parser."""

    def test_spine_subcommands_parse(self):
        import argparse
        import memdag_ingest
        import memdag_compact

        p = argparse.ArgumentParser(prog="memdag_test_spine")
        sub = p.add_subparsers(dest="command", required=True)
        memdag_ingest.register(sub)
        memdag_retrieve.register(sub)
        memdag_compact.register(sub)

        # compact
        args = p.parse_args(["compact"])
        self.assertEqual(args.command, "compact")

        # retrieve
        args = p.parse_args(["retrieve", "some query"])
        self.assertEqual(args.command, "retrieve")
        self.assertEqual(args.query, "some query")

        # ingest-text
        args = p.parse_args(["ingest-text", "hello world", "--channel", "user"])
        self.assertEqual(args.command, "ingest-text")
        self.assertEqual(args.text, "hello world")
        self.assertEqual(args.channel, "user")


class TestAnticipatoryPhase2Cli(Base):
    """Phase 2: warm prefetch serving, recombination warning, anticipate-status."""

    def setUp(self):
        super().setUp()
        # Deterministic + fast: no vector path even if local Ollama is up
        os.environ["MEMDAG_EMBED_URL"] = "invalid://embeddings-disabled"

    def tearDown(self):
        os.environ.pop("MEMDAG_EMBED_URL", None)
        super().tearDown()

    def test_prefetch_then_ask_anticipate_serves_warm(self):
        self.add("Nebula static_host_map must point at the lighthouse.", "endorsed")
        q = "How should I configure the Nebula lighthouse?"

        self.run_cli("observe", q)
        self.run_cli("observe", q)
        out_pre = self.run_cli("prefetch", "--k", "1")
        self.assertIn("+warm", out_pre)
        self.assertIn("prefetch cache: 1 entry", out_pre)

        count_before = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]

        out = self.run_cli("ask", "--anticipate", q)
        self.assertIn("served WARM from prefetch cache", out)
        self.assertIn("[mem:", out)  # honest provenance citations in the body

        count_after = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]
        self.assertEqual(count_before, count_after,
                         "a warm hit must not mint a new node")

    def test_anticipate_new_mint_flags_novel_recombination(self):
        self.add("Nebula static_host_map must point at the lighthouse.", "endorsed")
        self.add("Use UDP port 4242 for nebula tunnels.", "user")
        out = self.run_cli("ask", "--anticipate", "How should I configure Nebula?")
        self.assertIn("stored as node", out)
        self.assertIn("new inference: this combination of sources has not been seen before.",
                      out)

    def test_anticipate_status_smoke(self):
        self.run_cli("observe", "how do I rotate Nebula certs?")
        self.run_cli("observe", "how do I rotate Nebula certs?")
        out = self.run_cli("anticipate-status")
        self.assertIn("query log: 2 row(s), 1 distinct query", out)
        self.assertIn("how do I rotate Nebula certs?", out)
        self.assertIn("prefetch cache: empty", out)


if __name__ == "__main__":
    unittest.main()
