"""Kernel layer tests: store CRUD/atomicity, argv construction, the scripted
CLI streaming path (pointer rolls on the result event), busy locking, error
and kill paths, refresh_ptr, and boot reconcile. The CLI seam is
``KernelRunner._argv`` — a scripted python stand-in emits canned stream-json,
so no real claude/codex CLI (or network) is involved.
"""

import json
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

from memsom.providers.base import ProviderError
from memsom.providers import kernels as K
from memsom.providers.kernels import KernelRunner, KernelStore

SID_A = "aaaaaaaa-1111-2222-3333-444444444444"
SID_B = "bbbbbbbb-5555-6666-7777-888888888888"

CLAUDE_SCRIPT = textwrap.dedent(f"""
    import json, sys
    prompt = sys.stdin.read()
    def emit(o): print(json.dumps(o), flush=True)
    emit({{"type": "system", "subtype": "hook_started", "session_id": "{SID_A}"}})
    emit({{"type": "system", "subtype": "init", "session_id": "{SID_A}"}})
    emit({{"type": "assistant", "session_id": "{SID_A}", "message": {{
        "content": [{{"type": "thinking", "thinking": "hm"}},
                     {{"type": "text", "text": "echo: " + prompt.strip()}},
                     {{"type": "tool_use", "name": "Bash"}}]}}}})
    emit({{"type": "rate_limit_event", "session_id": "{SID_A}"}})
    emit({{"type": "result", "subtype": "success", "is_error": False,
          "result": "ok", "session_id": "{SID_B}", "total_cost_usd": 0.01,
          "num_turns": 1, "usage": {{"output_tokens": 5, "service_tier": "x"}}}})
""")

ERROR_SCRIPT = textwrap.dedent("""
    import sys
    sys.stdin.read()
    sys.stderr.write("boom: bad flag")
    sys.exit(3)
""")

SLEEP_SCRIPT = textwrap.dedent("""
    import sys, time
    sys.stdin.read()
    time.sleep(30)
""")

CODEX_SCRIPT = textwrap.dedent(f"""
    import sys
    sys.stdin.read()
    print("session id: {SID_A}", flush=True)
    print("working on it", flush=True)
    print("all done", flush=True)
""")


class ScriptedRunner(KernelRunner):
    """KernelRunner whose CLI is a python script (the _argv seam)."""

    def __init__(self, store, sessions_dir, claude_dir, script_path):
        super().__init__(store, sessions_dir, claude_dir)
        self.script_path = str(script_path)

    def _argv(self, kernel):
        return [sys.executable, self.script_path]


def _wait_idle(store: KernelStore, kernel_id: str, timeout=10.0) -> None:
    """Wait for the runner thread's finally block to finish (status idle AND
    the per-kernel lock free) — Windows tempdir cleanup races it otherwise."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.get(kernel_id).get("status") == "idle":
            return
        time.sleep(0.05)
    raise AssertionError("kernel never returned to idle")


def _wait_terminal(path: Path, timeout=10.0) -> list:
    """Poll a run feed until a done/error line lands; return parsed events."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.is_file():
            lines = [json.loads(l) for l in
                     path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if any(e.get("t") in ("done", "error") for e in lines):
                return lines
        time.sleep(0.05)
    raise AssertionError(f"no terminal line within {timeout}s in {path}")


class KernelStoreTests(unittest.TestCase):
    def test_crud_and_archive_filter(self):
        with tempfile.TemporaryDirectory() as d:
            store = KernelStore(d)
            k = store.create("test", "claude", "opus", None, d, "claude")
            self.assertEqual(len(k["kernel_id"]), 12)
            self.assertIsNone(k["session_ptr"])
            got = store.get(k["kernel_id"])
            self.assertEqual(got["name"], "test")
            store.update(k["kernel_id"], status="archived")
            self.assertEqual(store.list(), [])
            self.assertEqual(len(store.list(include_archived=True)), 1)

    def test_id_fence(self):
        with tempfile.TemporaryDirectory() as d:
            store = KernelStore(d)
            for bad in ("../evil", "..", "ABCDEF123456", "short", ""):
                with self.assertRaises(ProviderError):
                    store.get(bad)

    def test_atomic_write_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as d:
            store = KernelStore(d)
            k = store.create("t", "claude", None, None, d, "claude")
            store.update(k["kernel_id"], name="renamed")
            leftovers = [p for p in Path(d).iterdir() if ".tmp-" in p.name]
            self.assertEqual(leftovers, [])


class ArgvTests(unittest.TestCase):
    def _runner(self, d):
        return KernelRunner(KernelStore(d), Path(d) / "s", Path(d) / "c")

    def test_claude_fresh_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._runner(d)
            k = {"engine": "claude", "cli_path": "claude", "model": "opus",
                 "effort": "high", "session_ptr": None}
            self.assertEqual(
                r._argv(k),
                ["claude", "-p", "--output-format", "stream-json", "--verbose",
                 "--model", "opus", "--effort", "high"])
            k["session_ptr"] = SID_A
            self.assertEqual(r._argv(k)[-2:], ["--resume", SID_A])

    def test_codex_fresh_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._runner(d)
            k = {"engine": "codex", "cli_path": "codex", "model": None,
                 "session_ptr": None}
            self.assertEqual(r._argv(k), ["codex", "exec"])
            k["session_ptr"] = SID_A
            self.assertEqual(r._argv(k), ["codex", "exec", "resume", SID_A])

    def test_corrupt_pointer_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._runner(d)
            k = {"engine": "claude", "cli_path": "claude", "model": None,
                 "effort": None, "session_ptr": "../../etc"}
            with self.assertRaises(ProviderError):
                r._argv(k)


class PromptFlowTests(unittest.TestCase):
    def _setup(self, script):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        script_path = d / "fake_cli.py"
        script_path.write_text(script, encoding="utf-8")
        store = KernelStore(d / "kernels")
        runner = ScriptedRunner(store, d / "sessions", d / "claude", script_path)
        kernel = store.create("t", "claude", None, None, str(d), "claude")
        self.addCleanup(self.tmp.cleanup)
        return store, runner, kernel, d

    def test_stream_maps_events_and_rolls_pointer(self):
        store, runner, kernel, d = self._setup(CLAUDE_SCRIPT)
        run_id = runner.prompt(kernel["kernel_id"], "hi there")
        self.assertTrue(run_id.startswith(f"krn-{kernel['kernel_id']}-"))
        events = _wait_terminal(d / "sessions" / f"{run_id}.jsonl")
        kinds = [e["t"] for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertIn("tok", kinds)
        self.assertEqual(kinds[-1], "done")
        toks = "".join(e.get("text", "") for e in events if e["t"] == "tok")
        self.assertIn("echo: hi there", toks)   # prompt rode stdin
        self.assertIn("[tool: Bash]", toks)     # tool_use surfaced readably
        done = events[-1]
        self.assertEqual(done["stats"]["cost_usd"], 0.01)
        # rolling pointer: the RESULT event's id wins, not the init id.
        deadline = time.time() + 5
        while time.time() < deadline:
            k = store.get(kernel["kernel_id"])
            if k["session_ptr"] == SID_B and k["status"] == "idle":
                break
            time.sleep(0.05)
        self.assertEqual(k["session_ptr"], SID_B)
        self.assertEqual(k["status"], "idle")
        self.assertEqual(k["prompt_count"], 1)

    def test_busy_lock(self):
        store, runner, kernel, d = self._setup(SLEEP_SCRIPT)
        runner.prompt(kernel["kernel_id"], "first")
        time.sleep(0.3)
        with self.assertRaises(ProviderError) as ctx:
            runner.prompt(kernel["kernel_id"], "second")
        self.assertIn("busy", str(ctx.exception))
        runner.kill(kernel["kernel_id"])
        _wait_idle(store, kernel["kernel_id"])

    def test_error_exit_stamps_error(self):
        store, runner, kernel, d = self._setup(ERROR_SCRIPT)
        run_id = runner.prompt(kernel["kernel_id"], "x")
        events = _wait_terminal(d / "sessions" / f"{run_id}.jsonl")
        self.assertEqual(events[-1]["t"], "error")
        self.assertIn("boom", events[-1]["error"])
        _wait_idle(store, kernel["kernel_id"])

    def test_kill_stamps_error_and_frees_lock(self):
        store, runner, kernel, d = self._setup(SLEEP_SCRIPT)
        run_id = runner.prompt(kernel["kernel_id"], "x")
        time.sleep(0.3)
        self.assertTrue(runner.kill(kernel["kernel_id"]))
        events = _wait_terminal(d / "sessions" / f"{run_id}.jsonl")
        self.assertEqual(events[-1]["t"], "error")
        self.assertIn("killed", events[-1]["error"])
        # lock released -> a new prompt is accepted (and killed right away).
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                runner.prompt(kernel["kernel_id"], "again")
                break
            except ProviderError:
                time.sleep(0.05)
        # kill() is a no-op until the thread registers its process — retry.
        deadline = time.time() + 5
        while not runner.kill(kernel["kernel_id"]) and time.time() < deadline:
            time.sleep(0.05)
        _wait_idle(store, kernel["kernel_id"])

    def test_prompt_cap_and_empty(self):
        store, runner, kernel, d = self._setup(CLAUDE_SCRIPT)
        with self.assertRaises(ProviderError):
            runner.prompt(kernel["kernel_id"], "")
        with self.assertRaises(ProviderError):
            runner.prompt(kernel["kernel_id"], "x" * 9000)

    def test_codex_stream_captures_pointer(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        script_path = d / "fake_codex.py"
        script_path.write_text(CODEX_SCRIPT, encoding="utf-8")
        store = KernelStore(d / "kernels")
        runner = ScriptedRunner(store, d / "sessions", d / "claude", script_path)
        kernel = store.create("t", "codex", None, None, str(d), "codex")
        self.addCleanup(self.tmp.cleanup)
        run_id = runner.prompt(kernel["kernel_id"], "go")
        events = _wait_terminal(d / "sessions" / f"{run_id}.jsonl")
        self.assertEqual(events[-1]["t"], "done")
        toks = "".join(e.get("text", "") for e in events if e["t"] == "tok")
        self.assertIn("all done", toks)
        deadline = time.time() + 5
        while time.time() < deadline:
            k = store.get(kernel["kernel_id"])
            if k["session_ptr"] == SID_A:
                break
            time.sleep(0.05)
        self.assertEqual(k["session_ptr"], SID_A)


class PointerOwnershipTests(unittest.TestCase):
    def test_pointer_only_moves_via_stream_events(self):
        """Regression for the removed refresh_ptr auto-scan (it adopted an
        unrelated concurrent session on the first live smoke — recency is not
        lineage): a pre-set pointer must survive a prompt whose stream never
        reports a session id."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            script = d / "silent_cli.py"
            script.write_text(
                "import sys\nsys.stdin.read()\nprint('plain text, no json')\n",
                encoding="utf-8")
            store = KernelStore(d / "kernels")
            runner = ScriptedRunner(store, d / "sessions", d / "claude", script)
            kernel = store.create("t", "claude", None, None, str(d), "claude")
            store.update(kernel["kernel_id"], session_ptr=SID_A)
            run_id = runner.prompt(kernel["kernel_id"], "x")
            _wait_terminal(d / "sessions" / f"{run_id}.jsonl")
            _wait_idle(store, kernel["kernel_id"])
            self.assertEqual(store.get(kernel["kernel_id"])["session_ptr"],
                             SID_A)


class ReconcileTests(unittest.TestCase):
    def test_busy_kernel_reconciled(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store = KernelStore(d / "kernels")
            runner = KernelRunner(store, d / "sessions", d / "claude")
            kernel = store.create("t", "claude", None, None, str(d), "claude")
            run_id = f"krn-{kernel['kernel_id']}-deadbeef"
            (d / "sessions").mkdir(parents=True, exist_ok=True)
            feed = d / "sessions" / f"{run_id}.jsonl"
            feed.write_text('{"t": "start"}\n{"t": "tok", "text": "hi"}\n',
                            encoding="utf-8")
            store.update(kernel["kernel_id"], status="busy", last_run_id=run_id)

            fixed = runner.reconcile_on_boot()
            self.assertEqual(fixed, 1)
            self.assertEqual(store.get(kernel["kernel_id"])["status"], "idle")
            lines = feed.read_text(encoding="utf-8").splitlines()
            last = json.loads(lines[-1])
            self.assertEqual(last["t"], "error")
            self.assertIn("interrupted", last["error"])
            # idempotent: a feed that already has a terminal line is untouched.
            store.update(kernel["kernel_id"], status="busy")
            runner.reconcile_on_boot()
            self.assertEqual(
                len(feed.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
