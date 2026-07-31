#!/usr/bin/env python3
r"""The MCP `export` tool may not choose its own destination — F-10.

Seam S3 of the memsom-panel white-box engagement.

`export` serialises every node in the store — id, CONTENT, channel, edges — and
writes it with `open(path, "w")`. The `path` argument came off the wire as a
free string with no root, no extension check and no containment, and the author
of that string is the model: whatever text is in a session's context chooses
where the entire memory DAG gets written. There is no prompt in front of it,
because MCP tools run unattended.

That is one line reaching two primitives:

  EXFILTRATION   a destination inside a bidirectionally-replicated tree puts
                 the whole store on another machine, without the tool opening a
                 single socket. Nothing in the process looks like egress.
  TRUNCATION     `"w"` does not ask what was there. Any file this process can
                 write can be replaced with a changeset, including the scripts
                 and configs the surrounding system loads on startup.

THE GATE ASSERTS THE SIDE EFFECT, NOT THE RETURN VALUE. `_tool_argv` returning
a fenced string proves the string is fenced; it does not prove that no file
appeared where the caller asked. That distinction is exactly what went wrong in
the sibling finding on `file_read`, whose docstring promised something true of
the return value and false of the side effect. So the tests below run the tool
end to end and look at the disk, and
`test_the_probe_can_observe_the_write_it_denies` proves the probe would have
seen the file if the fence were not there.

WHAT THIS DOES NOT CLOSE. A model steered into calling this tool is a model
running in a session that already has a shell. Fencing `export` removes a
convenient primitive; it does not remove the capability, and a green run here
must not be read as "the store cannot leave the machine". The CLI
(`memsom export <path>`) is deliberately untouched: a human choosing a
destination with their own hands is not the threat.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import memsom
from memsom.interface import mcp as memsom_mcp

#: Destinations a model must not be able to name. Real shapes, not synthetic
#: strings: a traversal in both separator styles, a UNC share (the form that
#: also makes Windows open an outbound SMB session), and the Win32 device
#: namespace, which is the prefix that walks through a naive drive-letter check.
ESCAPES = [
    "../../../outside.jsonl",
    "..\\..\\outside.jsonl",
    "//attacker.example/share/dump.jsonl",
    "\\\\attacker.example\\share\\dump.jsonl",
    "\\\\?\\C:\\dump.jsonl",
]


class ExportFenceArgvTest(unittest.TestCase):
    """The translation layer, in isolation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.exports = Path(self.tmp.name) / "exports"
        self._prev = os.environ.get(memsom_mcp.MCP_EXPORT_DIR_ENV)
        os.environ[memsom_mcp.MCP_EXPORT_DIR_ENV] = str(self.exports)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(memsom_mcp.MCP_EXPORT_DIR_ENV, None)
        else:
            os.environ[memsom_mcp.MCP_EXPORT_DIR_ENV] = self._prev
        self.tmp.cleanup()

    def test_a_plain_filename_is_accepted_and_lands_in_the_export_dir(self):
        """The control for every refusal below: the tool must still work."""
        argv = memsom_mcp._tool_argv("export", {"path": "changeset.jsonl"})
        dest = Path(argv[1])
        self.assertEqual(dest.parent, self.exports.resolve())
        self.assertEqual(dest.name, "changeset.jsonl")

    def test_a_subdirectory_inside_the_export_dir_is_allowed(self):
        argv = memsom_mcp._tool_argv("export", {"path": "runs/monday.jsonl"})
        self.assertTrue(
            Path(argv[1]).is_relative_to(self.exports.resolve()), argv[1])

    def test_an_absolute_path_already_inside_the_export_dir_is_allowed(self):
        inside = str(self.exports / "already-inside.jsonl")
        argv = memsom_mcp._tool_argv("export", {"path": inside})
        self.assertEqual(Path(argv[1]).parent, self.exports.resolve())

    def test_escaping_paths_are_refused_not_rewritten(self):
        r"""Refused, never sanitized.

        Silently reducing `…/Vault/dump.jsonl` to its basename would make the
        tool report success for a call it did not perform. The caller has to be
        able to tell a fenced write from an honoured one, so this asserts a
        raise — and asserts the basename did NOT quietly become the answer.
        """
        for hostile in ESCAPES:
            with self.subTest(path=hostile):
                with self.assertRaises(ValueError) as ctx:
                    memsom_mcp._tool_argv("export", {"path": hostile})
                self.assertIn("refused", str(ctx.exception).lower())

    def test_an_absolute_path_outside_the_export_dir_is_refused(self):
        outside = str(Path(self.tmp.name) / "not-exports" / "dump.jsonl")
        with self.assertRaises(ValueError):
            memsom_mcp._tool_argv("export", {"path": outside})

    def test_a_non_jsonl_destination_is_refused(self):
        """The truncation half. Inside the export dir there is little to
        clobber, but the extension is what keeps `open(…, "w")` off anything
        that is not an export — including a future sibling that is."""
        for hostile in ("hook.py", "settings.json", "notes.md", "no-extension"):
            with self.subTest(path=hostile):
                with self.assertRaises(ValueError):
                    memsom_mcp._tool_argv("export", {"path": hostile})

    def test_the_since_argument_still_rides_through(self):
        argv = memsom_mcp._tool_argv(
            "export", {"path": "c.jsonl", "since": "2026-01-01T00:00:00Z"})
        self.assertEqual(argv[0], "export")
        self.assertEqual(argv[2:], ["--since", "2026-01-01T00:00:00Z"])

    def test_the_refusal_is_a_value_error_the_transport_already_handles(self):
        """`_call_tool` catches ValueError and returns an MCP tool error. A
        refusal that raised anything else would crash the stdio server, which
        is a worse outcome than the finding."""
        with self.assertRaises(ValueError):
            memsom_mcp._tool_argv("export", {"path": "../escape.jsonl"})


class ExportFenceSideEffectTest(unittest.TestCase):
    """End to end, looking at the disk — the assertion that actually matters."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = root / "mcp_export_test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        with self.conn:
            memsom.insert_node(self.conn, "a secret worth exfiltrating", "user")

        # The stand-in for the replicated tree: a directory the model names and
        # the fence must keep it out of.
        self.offbox = root / "replicated"
        self.offbox.mkdir()
        self.target = self.offbox / "sync-backup.jsonl"

        self.exports = root / "exports"
        self._prev = os.environ.get(memsom_mcp.MCP_EXPORT_DIR_ENV)
        os.environ[memsom_mcp.MCP_EXPORT_DIR_ENV] = str(self.exports)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        if self._prev is None:
            os.environ.pop(memsom_mcp.MCP_EXPORT_DIR_ENV, None)
        else:
            os.environ[memsom_mcp.MCP_EXPORT_DIR_ENV] = self._prev
        self.tmp.cleanup()

    def _call(self, arguments):
        """Drive the real tools/call path, not the argv helper."""
        resp = memsom_mcp.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "export", "arguments": arguments},
        })
        result = resp.get("result") or {}
        text = "".join(c.get("text", "") for c in result.get("content", []))
        return bool(result.get("isError")), text

    def test_the_probe_can_observe_the_write_it_denies(self):
        """THE CONTROL TEST.

        Every assertion below is "the file is not there", which is worthless
        unless this same probe demonstrably sees a file when one is written.
        Same tool, same call path, same `.exists()` check — only the
        destination is one the fence permits.
        """
        is_error, text = self._call({"path": "permitted.jsonl"})
        self.assertFalse(is_error, text)
        written = self.exports / "permitted.jsonl"
        self.assertTrue(written.exists(),
                        "the probe cannot see an export it allows, so it cannot "
                        "be trusted to see one it denies")
        body = written.read_text(encoding="utf-8")
        self.assertIn("a secret worth exfiltrating", body,
                      "the export really does carry node CONTENT — this is the "
                      "payload the finding is about")

    def test_no_file_appears_at_the_model_chosen_destination(self):
        """The finding, at the disk. `sync-backup.jsonl` must not exist."""
        is_error, text = self._call({"path": str(self.target)})
        self.assertTrue(is_error, "the refused call reported success")
        self.assertFalse(
            self.target.exists(),
            f"the store was written to a model-chosen path: {self.target}")
        self.assertEqual(list(self.offbox.iterdir()), [],
                         "something landed in the off-box tree")

    def test_an_existing_file_is_not_truncated(self):
        """The destructive variant, at the disk.

        `open(path, "w")` truncates before anything else happens, so a fence
        that ran late would leave the file empty even on a failed export. The
        assertion is on the BYTES, not on the status code.
        """
        victim = self.offbox / "important.jsonl"
        original = "#!/usr/bin/env python3\nload_bearing = True\n"
        victim.write_text(original, encoding="utf-8")

        is_error, _ = self._call({"path": str(victim)})
        self.assertTrue(is_error)
        self.assertEqual(victim.read_text(encoding="utf-8"), original,
                         "the victim file was truncated by the refused export")

    def test_a_unc_destination_writes_nothing_and_reports_an_error(self):
        is_error, text = self._call({"path": "//attacker.example/s/dump.jsonl"})
        self.assertTrue(is_error, text)
        self.assertEqual(list(self.offbox.iterdir()), [])

    def test_the_refusal_surfaces_as_a_tool_error_not_a_crash(self):
        """The stdio server must survive a refused call: a traceback out of
        `handle` would take the transport down with it."""
        is_error, text = self._call({"path": str(self.target)})
        self.assertTrue(is_error)
        self.assertTrue(text.strip(), "a refusal must say why")
        # ...and the server is still answering afterwards.
        resp = memsom_mcp.handle({"jsonrpc": "2.0", "id": 2,
                                  "method": "tools/list", "params": {}})
        self.assertIn("result", resp)
        self.assertIn("export",
                      {t["name"] for t in resp["result"]["tools"]})


class ExportCliIsUnaffectedTest(unittest.TestCase):
    """The fence is on the model, not on the human.

    `memsom export <path>` is a person choosing a destination with their own
    hands. If this test ever fails, the fix was applied one layer too deep and
    a legitimate workflow was broken to close a model-driven hole.
    """

    def test_the_cli_still_exports_to_an_arbitrary_path(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "cli.db"
            env = dict(os.environ, MEMDAG_DB=str(db))
            env.pop(memsom_mcp.MCP_EXPORT_DIR_ENV, None)
            out = Path(d) / "anywhere" / "human-chose-this.jsonl"
            out.parent.mkdir()

            import subprocess
            seed = subprocess.run(
                [sys.executable, "-m", "memsom.interface.cli", "ingest-text",
                 "cli seed", "--channel", "user"],
                capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(seed.returncode, 0, seed.stderr)

            proc = subprocess.run(
                [sys.executable, "-m", "memsom.interface.cli", "export", str(out)],
                capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out.exists(),
                            "the CLI export was fenced — the fix went too deep")


if __name__ == "__main__":
    unittest.main()
