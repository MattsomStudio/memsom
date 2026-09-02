"""GATES for the two Phase 5 (effects layer) findings that had no test at all:
MS-26 (broker upstream stderr is never drained) and MS-29 (a bare `git`/`claude`
spawn lets Windows CreateProcess search the CWD ahead of PATH).

Both are control-tested here in the same commit as their fix (SECURITY-
REMEDIATION.md Sec1.2: "each of these needs a tests/gates/ test written before
its fix" -- written alongside it here, since the fix and its gate are the same
mechanical change).
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from memsom.effects import proc as memsom_proc
from memsom.federation import broker as memsom_broker


# ---------------------------------------------------------------------------
# MS-26 -- broker upstream stderr must be drained, and _rpc must not hang
# ---------------------------------------------------------------------------

# Writes well over any OS pipe buffer to stderr before answering the MCP
# handshake. An UNDRAINED stderr=PIPE blocks the CHILD's write(2) once the
# buffer fills, so it never reaches stdin -- Upstream.start() would stall.
_STUB_BIG_STDERR = r"""
import json, sys
sys.stderr.write("x" * (2 * 1024 * 1024))
sys.stderr.flush()
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    m = json.loads(raw)
    mid = m.get("id"); method = m.get("method", "")
    if mid is None and method != "initialize":
        continue
    if method == "initialize":
        r = {"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2024-11-05",
             "capabilities": {"tools": {}}, "serverInfo": {"name": "stub", "version": "0"}}}
    elif method == "tools/list":
        r = {"jsonrpc": "2.0", "id": mid, "result": {"tools": []}}
    else:
        r = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "nope"}}
    sys.stdout.write(json.dumps(r) + "\n")
    sys.stdout.flush()
"""

# Reads stdin and never answers anything -- proves _rpc gives up rather than
# blocking forever on an upstream that goes silent.
_STUB_SILENT = "import sys\nfor raw in sys.stdin:\n    pass\n"


def test_upstream_start_is_not_blocked_by_a_noisy_stderr(tmp_path):
    """MS-26: a 2MB stderr write from the upstream must not stall the MCP
    handshake -- proves the stdout/stderr pump threads added in Phase 5 keep
    the pipe drained instead of leaving stderr=PIPE unread."""
    stub = tmp_path / "noisy_stub.py"
    stub.write_text(_STUB_BIG_STDERR, encoding="utf-8")
    up = memsom_broker.Upstream("noisy", {"command": sys.executable, "args": [str(stub)]})
    started = time.monotonic()
    try:
        up.start()
        elapsed = time.monotonic() - started
        assert elapsed < 10, f"start() took {elapsed:.1f}s -- stderr is blocking the handshake"
        assert up.tools == []
    finally:
        up.stop()


def test_rpc_times_out_instead_of_hanging_forever(tmp_path, monkeypatch):
    """MS-26's other half: an upstream that never answers must not hang the
    broker forever -- _rpc's queue read now has a deadline."""
    stub = tmp_path / "silent_stub.py"
    stub.write_text(_STUB_SILENT, encoding="utf-8")
    monkeypatch.setattr(memsom_broker.Upstream, "RPC_TIMEOUT", 1)
    up = memsom_broker.Upstream("silent", {"command": sys.executable, "args": [str(stub)]})
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            up.start()
        elapsed = time.monotonic() - started
        assert elapsed < 5, f"took {elapsed:.1f}s -- RPC_TIMEOUT was not honoured"
    finally:
        up.stop()


# ---------------------------------------------------------------------------
# MS-29 -- a bare executable name must resolve via PATH, never the CWD
# ---------------------------------------------------------------------------

def test_proc_resolve_ignores_a_planted_executable_in_cwd(tmp_path, monkeypatch):
    """Windows CreateProcess (lpApplicationName=NULL) searches the calling
    process's CWD ahead of PATH for a bare name -- a `git.exe` planted in a
    hostile working directory would run instead of the real one.
    effects.proc.resolve() pins to shutil.which's PATH-based answer, computed
    BEFORE the spawn, so the CWD is never consulted."""
    real_git = shutil.which("git")
    if not real_git:
        pytest.fail("git is required for this gate (no git on PATH)")
    poison = tmp_path / "git.exe"
    poison.write_bytes(b"not a real executable")
    monkeypatch.chdir(tmp_path)
    resolved = memsom_proc.resolve("git")
    assert Path(resolved).resolve() != poison.resolve(), (
        "resolve() picked the CWD-planted executable, not the real PATH one")
    assert Path(resolved).resolve() == Path(real_git).resolve()


def test_proc_run_executes_the_real_git_not_a_planted_one(tmp_path, monkeypatch):
    """Behavioural half of the above: spawning "git" through proc.run() with a
    poisoned CWD must run the real git (or fail cleanly, if a caller passes a
    genuinely unresolvable name) -- never execute the planted file."""
    if not shutil.which("git"):
        pytest.fail("git is required for this gate (no git on PATH)")
    poison = tmp_path / "git.exe"
    poison.write_bytes(b"not a real executable")
    monkeypatch.chdir(tmp_path)
    result = memsom_proc.run(["git", "--version"], timeout=10,
                             capture_output=True, text=True)
    assert result.returncode == 0
    assert "git version" in result.stdout.lower()
