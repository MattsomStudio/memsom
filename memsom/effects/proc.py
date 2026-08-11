"""memsom.effects.proc -- outbound process spawns.

Every subprocess call in memsom absorbs two lessons here once: resolve the
executable to an absolute path via shutil.which (MS-29 -- Windows resolves a
bare name from the current directory ahead of PATH, so `git` or `claude`
spawned by name in a hostile CWD runs the planted one instead), and apply
child_env() rather than the full parent environment (MS-37 -- the credential
denylist, for exactly the least-trusted, longest-lived children memsom spawns:
broker upstreams, the detached /saveall child).

run() additionally requires an explicit timeout -- a `subprocess.run` with no
timeout is a hang with no operator signal (MS-29's other half). popen() has no
timeout parameter, deliberately: `subprocess.Popen.__init__` does not accept
one, and every Popen call this module wraps is a long-lived child (an MCP
upstream, a detached save) whose lifecycle is managed by its own read loop or
process tracking, not by blocking on spawn.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from memsom.childenv import child_env

#: Re-exported so a caller never needs its own `import subprocess` just for a
#: pipe/flag constant -- that alone would still count as a subprocess importer.
PIPE = subprocess.PIPE
DEVNULL = subprocess.DEVNULL
STDOUT = subprocess.STDOUT
TimeoutExpired = subprocess.TimeoutExpired
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def resolve(exe: str) -> str:
    """Absolute path for *exe* via PATH, deliberately never the CWD.

    `shutil.which(exe)` alone is NOT safe for this on win32: cpython's
    implementation inserts `os.curdir` at the front of the search path
    whenever *cmd* has no directory part, and that insertion is
    unconditional -- it happens even when a `path=` argument is supplied,
    it is not gated on `path is None`. So a bare `shutil.which("git")` still
    matches a `git.exe` planted in a hostile CWD.

    Passing each PATH entry as an explicit directory-qualified candidate
    (`shutil.which(os.path.join(d, exe))`) takes cpython's OTHER branch --
    "given a path with a directory part, look it up directly" -- which never
    consults the CWD. Falls back to the bare name when it isn't found on
    PATH -- the spawn then fails with the OS's own FileNotFoundError, which
    is the same failure mode as today.
    """
    if os.path.dirname(exe):
        return exe
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        found = shutil.which(os.path.join(d, exe))
        if found:
            return found
    return exe


def _build_env(env, keep):
    base = child_env(keep=keep)
    if env:
        base.update(env)
    return base


def run(argv, *, timeout, env=None, keep=(), **kwargs):
    """subprocess.run with the executable resolved and child_env() applied.

    *timeout* is required -- no caller may spawn an unbounded child. *env*
    overlays additional variables (e.g. a stub's own signalling var) on top of
    child_env()'s denylisted copy of the parent environment; *keep* re-admits
    specific credential names (see memsom.childenv.child_env).
    """
    argv = [resolve(argv[0]), *argv[1:]]
    return subprocess.run(argv, timeout=timeout, env=_build_env(env, keep), **kwargs)


def popen(argv, *, env=None, keep=(), **kwargs):
    """subprocess.Popen with the executable resolved and child_env() applied.

    No forced timeout -- see module docstring."""
    argv = [resolve(argv[0]), *argv[1:]]
    return subprocess.Popen(argv, env=_build_env(env, keep), **kwargs)
