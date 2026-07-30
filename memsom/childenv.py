"""Child-process environment scoping — one credential denylist, shared.

Why this is here and not in the panel
-------------------------------------
`memsom_panel` depends on `memsom`; nothing in `memsom` may import from the
panel.  Both spawn children (`memsom/interface/saveall.py` detaches a `claude`
process that outlives the session; the panel's providers spawn model servers,
CLI adapters and console utilities), and both need the same answer to the same
question, so the answer lives in the dependency.  Same placement argument as
`memsom.paths`.

Only `os` is imported, so the stdlib-only promise holds.

What this does and does NOT buy (S7 / F-19 + F-21, 2026-07-30)
--------------------------------------------------------------
`subprocess` inherits the parent environment by default and no site opted out,
so every child of the panel held `MEMSOM_ANTHROPIC_KEY` and `OPENAI_API_KEY`.

Be honest about the size of that: MEASURED, both keys live in `HKCU\\Environment`
in plaintext, so **any** process already running as this user reads them with one
`reg query`.  Child inheritance increases the same-user blast radius by zero,
and this module does not pretend otherwise.

What it does buy is that a DETACHED, long-lived child — procman's model servers
outlive the panel; saveall's `claude` outlives the session — no longer carries a
credential it never needed.  That turns a future stdout redirect, crash dump or
environment-logging provider from "writes the key down" into "writes nothing".
It is hygiene, deliberately rated as such, and it costs three lines at a choke
point.

A NAME DENYLIST, not an allowlist — deliberately
------------------------------------------------
A correct Windows allowlist has to carry `SystemRoot`, `PATH`, `PATHEXT`,
`COMSPEC`, `TEMP`, `TMP`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `windir`,
`PROCESSOR_ARCHITECTURE`, `NUMBER_OF_PROCESSORS`, `CUDA_PATH`, plus every
`MEMDAG_*` / `EPISODIC_*` tunable the recall stack reads — and the failure mode
of getting it wrong is `claude` or `llama-server` breaking at 2am for a reason
nobody connects to a security patch.  A denylist of known credential names has
no such failure mode and closes the actual finding.

`minimal_env()` gives a real allowlist for the few children whose needs are
small and fully known (ffmpeg, tasklist, taskkill, icacls, whoami).
"""

from __future__ import annotations

import os

#: Environment variables that are CREDENTIALS this system uses IN-PROCESS and
#: that no child it spawns has any use for.  The CLI adapters read these by NAME
#: (the profile carries `api_key_env`, never the key itself) and put the value
#: in an HTTP header; they never hand it to a subprocess.
CREDENTIAL_ENV_NAMES = (
    "MEMSOM_ANTHROPIC_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    # Not a model-provider key, but the same class and the same rule: it is the
    # panel's own bearer token's directory override, and a child that can read
    # it learns where the credential lives.
    "MEMSOM_PANEL_TOKEN_DIR",
)

#: The minimum a Windows console utility needs to start at all.  Used only where
#: the child's requirements are fully known and tiny — never for a CLI adapter
#: or a model server.
_MINIMAL_ENV_NAMES = (
    "SystemRoot", "windir", "PATH", "PATHEXT", "COMSPEC", "TEMP", "TMP",
    "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "SystemDrive",
)


def _fold(name: str) -> str:
    """Environment-variable names are case-insensitive on Windows.

    CPython's `os.environ` UPPERCASES every key on Windows, so a name written
    here in its conventional casing (`SystemRoot`, `windir`) matches nothing at
    all — and a `minimal_env()` built by exact comparison hands a console child
    an environment with no `SystemRoot`, which is a child that does not start.
    MEASURED 2026-07-30: the shipped spec for this had exactly that bug.
    """
    return name.upper() if os.name == "nt" else name


def child_env(*, keep=(), drop=(), base=None) -> dict:
    """The parent environment minus the credentials no child should inherit.

    *keep* re-admits specific names for a child that genuinely needs one (none
    do today — it exists so that the day one does, the exception is written down
    at the call site rather than by quietly widening the denylist).
    *drop* adds names for a single call site.
    *base* substitutes a different starting environment; defaults to
    ``os.environ``.

    Removes rather than blanks: a child that does ``os.environ.get(NAME)`` sees
    absence either way, and ``NAME in os.environ`` should also be False — a
    blanked variable still tells a child the name exists and is worth reading.
    """
    env = dict(os.environ if base is None else base)
    kept = {_fold(k) for k in keep}
    doomed = {_fold(n) for n in (*CREDENTIAL_ENV_NAMES, *drop)} - kept
    return {k: v for k, v in env.items() if _fold(k) not in doomed}


def minimal_env(*, extra=(), base=None) -> dict:
    """Only the variables a plain console utility needs.

    For children whose requirements are known and small; everything else uses
    `child_env`.  Names in *extra* are added when present in the environment.
    """
    src = dict(os.environ if base is None else base)
    allowed = {_fold(n) for n in _MINIMAL_ENV_NAMES}
    wanted = allowed | {_fold(n) for n in extra}
    return {k: v for k, v in src.items() if _fold(k) in wanted}
