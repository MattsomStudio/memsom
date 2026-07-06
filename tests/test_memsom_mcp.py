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
import memsom_mcp

HERE = Path(__file__).resolve().parent.parent
MCP_SCRIPT = str(HERE / "memsom_mcp.py")


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

    def test_tools_list_returns_17_tools(self):
        resp = memsom_mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = resp["result"]["tools"]
        self.assertEqual(len(tools), 17)
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
        import memsom_obsidian
        memsom_obsidian.migrate(self.conn)
        with self.conn:
            nid = memsom.insert_node(
                self.conn,
                "the homelab dashboard NOT deployed yet",
                "user",
                source_ref="memory:mcp_verify_test",
            )
            old = datetime.now(timezone.utc) - timedelta(days=60)
            self.conn.execute(
                "UPDATE nodes SET obsidian_mtime = ? WHERE id = ?",
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
        import memsom_cli
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
            [sys.executable, MCP_SCRIPT, "--selfcheck"],
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
            [sys.executable, MCP_SCRIPT],
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


if __name__ == "__main__":
    unittest.main()
