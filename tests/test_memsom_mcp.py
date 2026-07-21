#!/usr/bin/env python3
"""Tests for memsom_mcp — stdio MCP server.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_mcp.py \
    -t <repo> -v
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.interface import mcp as memsom_mcp

HERE = Path(__file__).resolve().parent.parent
MCP_MODULE = "memsom.interface.mcp"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "mcp_test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()


# ---------------------------------------------------------------------------
# In-process tests (drive handle() directly)
# ---------------------------------------------------------------------------

class TestHandleInProcess(Base):

    def _init_msg(self, msg_id=1):
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "initialize",
            "params": {"protocolVersion": memsom_mcp.PROTOCOL_VERSION},
        }

    def test_initialize_shape(self):
        resp = memsom_mcp.handle(self._init_msg())
        self.assertIsNotNone(resp)
        result = resp.get("result", {})
        self.assertIn("protocolVersion", result)
        self.assertIn("tools", result.get("capabilities", {}))
        self.assertIn("serverInfo", result)
        si = result["serverInfo"]
        self.assertIn("name", si)
        self.assertIn("version", si)

    def test_tools_list_returns_18_tools(self):
        resp = memsom_mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = resp["result"]["tools"]
        self.assertEqual(len(tools), 18)
        names = {t["name"] for t in tools}
        self.assertEqual(names, memsom_mcp.TOOL_NAMES)

    def test_tools_call_ask_returns_citation(self):
        # Seed the DB so ask has something to compose from
        with self.conn:
            memsom.insert_node(
                self.conn,
                "Nebula needs a lighthouse node with a public IP for hole punching.",
                "endorsed",
            )
            memsom.insert_node(
                self.conn,
                "Use UDP port 4242 for Nebula overlay tunnels.",
                "user",
            )
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "ask", "arguments": {"question": "How should I configure Nebula?"}},
        })
        self.assertFalse(resp["result"]["isError"])
        text = resp["result"]["content"][0]["text"]
        self.assertIn("[mem:", text)

    def test_unknown_tool_returns_32602(self):
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}},
        })
        self.assertEqual(resp["error"]["code"], -32602)

    def test_unknown_method_returns_32601(self):
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "completely/unknown",
        })
        self.assertEqual(resp["error"]["code"], -32601)

    def test_notification_returns_none(self):
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        self.assertIsNone(resp)

    def test_parse_error_path(self):
        # The parse error path is tested at the serve_stdio level;
        # handle() itself only receives a parsed dict.  We verify that a
        # completely unknown notification (no id) returns None.
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0",
            "method": "some/notification",
        })
        self.assertIsNone(resp)

    def test_tools_call_ingest_text_stores_node(self):
        """ingest_text tool stores a node at the declared channel; isError false."""
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "ingest_text",
                "arguments": {
                    "text": "Nebula requires a lighthouse node with a reachable public IP.",
                    "channel": "endorsed",
                    "source_ref": "mcp_test_ref",
                },
            },
        })
        self.assertFalse(resp["result"]["isError"],
                         f"ingest_text returned error: {resp['result']['content'][0]['text']}")
        # The node should now be in the DB
        row = self.conn.execute(
            "SELECT id, channel FROM nodes WHERE source_ref = 'mcp_test_ref'"
        ).fetchone()
        self.assertIsNotNone(row, "ingest_text should have stored a node")
        self.assertEqual(row[1], "endorsed")

    def test_tools_call_verify_stale_apply_marks_and_dry_run_does_not(self):
        """verify_stale threads apply through; dry-run (default) writes nothing."""
        from memsom.bridge import bridge_import as memsom_bridge_import
        memsom_bridge_import.migrate(self.conn)
        with self.conn:
            nid = memsom.insert_node(
                self.conn,
                "the homelab dashboard NOT deployed yet",
                "user",
                source_ref="memory:mcp_verify_test",
            )
            old = datetime.now(timezone.utc) - timedelta(days=60)
            self.conn.execute(
                "UPDATE nodes SET bridge_mtime = ? WHERE id = ?",
                (f"{int(old.timestamp() * 1e9)}:100", nid),
            )

        # Dry-run (no `apply` arg) must not mutate.
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0", "id": 20, "method": "tools/call",
            "params": {"name": "verify_stale", "arguments": {}},
        })
        self.assertFalse(resp["result"]["isError"])
        self.assertIn("DRY-RUN", resp["result"]["content"][0]["text"])
        row = self.conn.execute("SELECT stale FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 0)

        # apply=True commits the mark.
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0", "id": 21, "method": "tools/call",
            "params": {"name": "verify_stale", "arguments": {"apply": True}},
        })
        self.assertFalse(resp["result"]["isError"])
        self.assertIn("APPLIED", resp["result"]["content"][0]["text"])
        row = self.conn.execute("SELECT stale FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 1)

    def test_tools_call_retrieve_returns_hits(self):
        """retrieve tool returns text containing a [N] citation after seeding + reindex."""
        from memsom.interface import cli as memsom_cli
        # Seed two nodes via CLI
        memsom_cli.main(["add", "Nebula uses UDP port 4242 for overlay tunnels.", "--channel", "endorsed"])
        memsom_cli.main(["add", "Configure the Nebula lighthouse static_host_map entry.", "--channel", "user"])
        # Build BM25 index
        memsom_cli.main(["reindex"])
        # Call retrieve via MCP
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "retrieve",
                "arguments": {"query": "Nebula lighthouse"},
            },
        })
        self.assertFalse(resp["result"]["isError"],
                         f"retrieve returned error: {resp['result']['content'][0]['text']}")
        text = resp["result"]["content"][0]["text"]
        # Should contain at least one [N] citation
        import re
        self.assertTrue(
            bool(re.search(r'\[\d+\]', text)),
            f"retrieve output should contain a [N] citation; got: {text!r}",
        )


# ---------------------------------------------------------------------------
# Subprocess tests
# ---------------------------------------------------------------------------

class TestSubprocess(Base):

    def _env(self):
        return {**os.environ, "MEMDAG_DB": str(self.db)}

    def test_selfcheck_exits_0_and_3_json_lines(self):
        r = subprocess.run(
            [sys.executable, "-m", MCP_MODULE, "--selfcheck"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=self._env(),
            timeout=60,
        )
        self.assertEqual(r.returncode, 0, f"selfcheck failed:\nstdout:{r.stdout}\nstderr:{r.stderr}")
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        # Should have 3 JSON lines (initialize, tools/list, tools/call check)
        json_lines = []
        for l in lines:
            try:
                json_lines.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        self.assertEqual(len(json_lines), 3, f"Expected 3 JSON lines, got {len(json_lines)}: {lines}")

    def test_stdio_clean_eof_and_2_response_lines(self):
        """Feed initialize + tools/list + EOF; expect 2 JSON response lines, exit 0."""
        msg1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": {"protocolVersion": memsom_mcp.PROTOCOL_VERSION}})
        msg2 = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        stdin_data = msg1 + "\n" + msg2 + "\n"

        r = subprocess.run(
            [sys.executable, "-m", MCP_MODULE],
            input=stdin_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=self._env(),
            timeout=60,
        )
        self.assertEqual(r.returncode, 0, f"server crashed:\n{r.stderr}")

        lines = [l for l in r.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 2, f"Expected 2 response lines, got {len(lines)}: {lines}")

        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        self.assertEqual(r1["id"], 1)
        self.assertEqual(r2["id"], 2)


# ---------------------------------------------------------------------------
# redact argv mapping — cascade is OPT-IN per the tool schema
# ---------------------------------------------------------------------------
#
# The CLI cascades by default (--single opts out), but the MCP schema documents
# cascade as "Also redact all transitive descendants" i.e. opt-in. Regression:
# _tool_argv never emitted --single, so cascade:false (or omitted) still
# destroyed every descendant's payload — irreversible over-redaction.

class TestRedactArgv(unittest.TestCase):
    def test_cascade_false_maps_to_single(self):
        argv = memsom_mcp._tool_argv(
            "redact", {"id": 1, "reason": "r", "cascade": False, "apply": True})
        self.assertIn("--single", argv)

    def test_cascade_omitted_maps_to_single(self):
        argv = memsom_mcp._tool_argv("redact", {"id": 1, "reason": "r"})
        self.assertIn("--single", argv)

    def test_cascade_true_cascades(self):
        argv = memsom_mcp._tool_argv(
            "redact", {"id": 1, "reason": "r", "cascade": True})
        self.assertNotIn("--single", argv)


# ---------------------------------------------------------------------------
# _tool_argv — exact argv pin for every tool
# ---------------------------------------------------------------------------
#
# The argv translation is the one place a wrong flag silently changes what a
# tool does to the store (the 07-10 refactor class of bug). Pin the exact
# argv for all 17 tools, minimal and full-argument forms, so any drift
# between the MCP schema and the CLI mapping fails loudly.

class TestToolArgvMappings(unittest.TestCase):

    CASES = [
        # (tool, arguments, expected argv)
        ("ask", {"question": "q"}, ["ask", "q"]),
        ("ask",
         {"question": "q", "clearance": "secret", "anticipate": True,
          "llm": True, "graph": True, "hops": 2},
         ["ask", "q", "--clearance", "secret", "--anticipate", "--llm",
          "--graph", "--hops", "2"]),
        ("explain", {"id": 5}, ["explain", "5"]),
        ("blame", {"id": 5}, ["blame", "5"]),
        ("blame", {"id": 5, "clearance": "secret"},
         ["blame", "5", "--clearance", "secret"]),
        ("revoke", {"id": 5}, ["revoke", "5"]),
        ("revoke", {"id": 5, "reason": "r", "apply": True},
         ["revoke", "5", "--reason", "r", "--yes"]),
        ("redact", {"id": 5, "reason": "r"},
         ["redact", "5", "--reason", "r", "--single"]),
        ("redact", {"id": 5, "reason": "r", "cascade": True, "apply": True},
         ["redact", "5", "--reason", "r", "--yes"]),
        ("recompute", {}, ["recompute", "--all"]),
        ("recompute", {"all": True}, ["recompute", "--all"]),
        ("recompute", {"id": 5}, ["recompute", "5"]),
        ("consolidate", {}, ["consolidate"]),
        ("check", {}, ["check"]),
        ("export", {"path": "/tmp/out.jsonl"}, ["export", "/tmp/out.jsonl"]),
        ("export", {"path": "/tmp/out.jsonl", "since": "2026-01-01T00:00:00Z"},
         ["export", "/tmp/out.jsonl", "--since", "2026-01-01T00:00:00Z"]),
        ("neighborhood", {"id": 5}, ["neighborhood", "5"]),
        ("neighborhood",
         {"id": 5, "hops": 3, "min_integrity": "user", "clearance": "secret"},
         ["neighborhood", "5", "--hops", "3", "--min-integrity", "user",
          "--clearance", "secret"]),
        ("profile", {"id": 5}, ["profile", "5"]),
        ("check_action", {"id": 5, "required": "user"},
         ["check-action", "5", "--require", "user"]),
        ("retrieve", {"query": "q"}, ["retrieve", "q"]),
        ("retrieve", {"query": "q", "k": 3, "clearance": "secret"},
         ["retrieve", "q", "--k", "3", "--clearance", "secret"]),
        ("code_search", {"query": "q"}, ["code-search", "q"]),
        ("code_search", {"query": "q", "k": 3, "repo": "memsom"},
         ["code-search", "q", "--k", "3", "--repo", "memsom"]),
        ("ingest_text", {"text": "t", "channel": "user"},
         ["ingest-text", "t", "--channel", "user"]),
        ("ingest_text", {"text": "t", "channel": "user", "source_ref": "x"},
         ["ingest-text", "t", "--channel", "user", "--ref", "x"]),
        ("obsidian_sync", {}, ["obsidian-sync"]),
        ("obsidian_sync", {"vault": "/v", "channel": "user", "no_prune": True},
         ["obsidian-sync", "/v", "--channel", "user", "--no-prune"]),
        ("obsidian_export", {}, ["obsidian-export"]),
        ("obsidian_export",
         {"node": 5, "query": "q", "vault": "/v", "folder": "f", "title": "t"},
         ["obsidian-export", "5", "--query", "q", "--vault", "/v",
          "--folder", "f", "--title", "t"]),
        ("verify_stale", {}, ["verify-stale"]),
        ("verify_stale", {"apply": True}, ["verify-stale", "--apply"]),
    ]

    def test_every_tool_has_at_least_one_case(self):
        covered = {name for name, _, _ in self.CASES}
        self.assertEqual(covered, memsom_mcp.TOOL_NAMES)

    def test_argv_mappings(self):
        for name, arguments, expected in self.CASES:
            with self.subTest(tool=name, arguments=arguments):
                self.assertEqual(memsom_mcp._tool_argv(name, arguments), expected)

    def test_unknown_tool_raises_value_error(self):
        with self.assertRaises(ValueError):
            memsom_mcp._tool_argv("does_not_exist", {})

    def test_missing_required_argument_raises_key_error(self):
        with self.assertRaises(KeyError):
            memsom_mcp._tool_argv("redact", {"id": 1})  # no reason


# ---------------------------------------------------------------------------
# End-to-end dispatch — every tool not already exercised via tools/call
# ---------------------------------------------------------------------------
#
# The argv pins above guard the mapping; these guard the seam between the
# mapping and the real CLI parsers (a pinned-but-wrong flag fails here as
# argparse SystemExit(2) -> isError). Mutating tools additionally verify the
# DB effect, dry-run and applied.

class TestDispatchEndToEnd(Base):

    def setUp(self):
        super().setUp()
        from memsom.interface import cli as memsom_cli
        self.cli = memsom_cli
        self.cli.main(["migrate"])

    def _call(self, name, arguments=None, msg_id=100):
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        return resp["result"]["isError"], resp["result"]["content"][0]["text"]

    def _seed(self, content="Nebula lighthouse needs a public IP.", channel="endorsed"):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel)

    def test_explain(self):
        nid = self._seed()
        is_error, text = self._call("explain", {"id": nid})
        self.assertFalse(is_error, text)
        self.assertIn(str(nid), text)

    def test_blame_with_clearance(self):
        nid = self._seed()
        is_error, text = self._call("blame", {"id": nid, "clearance": "secret"})
        self.assertFalse(is_error, text)
        self.assertIn(str(nid), text)

    def test_profile(self):
        nid = self._seed()
        is_error, text = self._call("profile", {"id": nid})
        self.assertFalse(is_error, text)

    def test_neighborhood(self):
        nid = self._seed()
        is_error, text = self._call("neighborhood", {"id": nid, "hops": 2})
        self.assertFalse(is_error, text)

    def test_check(self):
        is_error, text = self._call("check")
        self.assertFalse(is_error, text)

    def test_check_action_allow_and_deny(self):
        endorsed = self._seed(channel="endorsed")
        is_error, text = self._call("check_action", {"id": endorsed, "required": "external"})
        self.assertFalse(is_error, text)
        self.assertIn("ALLOW", text)

        external = self._seed("untrusted web content", channel="external")
        is_error, text = self._call("check_action", {"id": external, "required": "endorsed"})
        self.assertTrue(is_error, "deny must surface as isError (exit 2)")
        self.assertIn("DENY", text)

    def test_consolidate(self):
        is_error, text = self._call("consolidate")
        self.assertFalse(is_error, text)

    def test_recompute_default_all(self):
        self._seed()
        is_error, text = self._call("recompute")
        self.assertFalse(is_error, text)

    def test_export_writes_changeset_file(self):
        self._seed()
        out = Path(self.tmp.name) / "changeset.jsonl"
        is_error, text = self._call("export", {"path": str(out)})
        self.assertFalse(is_error, text)
        self.assertTrue(out.exists(), "export should write the changeset file")
        self.assertTrue(out.read_text(encoding="utf-8").strip(),
                        "changeset should not be empty")

    def test_revoke_dry_run_then_apply(self):
        nid = self._seed()
        is_error, text = self._call("revoke", {"id": nid, "reason": "test"})
        self.assertFalse(is_error, text)
        row = self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 0, "dry-run revoke must not tombstone")

        is_error, text = self._call("revoke", {"id": nid, "reason": "test", "apply": True})
        self.assertFalse(is_error, text)
        row = self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 1, "apply=true revoke must tombstone")

    def test_redact_dry_run_then_apply_single(self):
        original = "secret payload to destroy"
        nid = self._seed(original)
        child, _ = memsom.derive_node(self.conn, "derived from the secret", [nid])

        is_error, text = self._call("redact", {"id": nid, "reason": "test"})
        self.assertFalse(is_error, text)
        row = self.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], original, "dry-run redact must not touch the payload")

        # cascade omitted -> --single: only the named node, never the child
        is_error, text = self._call("redact", {"id": nid, "reason": "test", "apply": True})
        self.assertFalse(is_error, text)
        row = self.conn.execute(
            "SELECT content, redacted FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], "", "applied redact must destroy the payload")
        self.assertEqual(row[1], 1)
        row = self.conn.execute(
            "SELECT redacted FROM nodes WHERE id = ?", (child,)).fetchone()
        self.assertEqual(row[0], 0, "cascade omitted must leave descendants intact")

    def test_obsidian_sync_ingests_vault_note(self):
        vault = Path(self.tmp.name) / "vault"
        vault.mkdir()
        (vault / "lighthouse.md").write_text(
            "Nebula lighthouse hosts run on UDP 4242.", encoding="utf-8")
        is_error, text = self._call(
            "obsidian_sync", {"vault": str(vault), "channel": "user"})
        self.assertFalse(is_error, text)
        row = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE content LIKE '%UDP 4242%'").fetchone()
        self.assertGreaterEqual(row[0], 1, "sync should ingest the vault note")

    def test_obsidian_export_writes_note(self):
        vault = Path(self.tmp.name) / "vault"
        (vault / "memsom").mkdir(parents=True)
        nid = self._seed()
        is_error, text = self._call(
            "obsidian_export", {"node": nid, "vault": str(vault), "title": "mcp-export-test"})
        self.assertFalse(is_error, text)
        notes = list((vault / "memsom").glob("*.md"))
        self.assertEqual(len(notes), 1, f"export should write one note; got {notes}")

    def test_missing_required_argument_is_client_error_not_crash(self):
        is_error, text = self._call("redact", {"id": 1})  # no reason
        self.assertTrue(is_error)
        self.assertIn("missing required argument", text)


if __name__ == "__main__":
    unittest.main()
