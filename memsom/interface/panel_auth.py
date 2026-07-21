"""memsom panel_auth — the panel's bearer token: mint, load, present, compare.

Why this exists
---------------
The panel's original posture was "the box is the credential": loopback bind, a
Host-header allowlist, an Origin allowlist, and no auth at all.  That stops
exactly ONE attacker — a malicious web page doing DNS rebinding — because
browsers always attach an Origin on cross-origin fetch.  The check is
"verify Origin *if present*" (panel.py `_read_post_json`), and curl, requests,
a pip/npm postinstall hook, or a rogue editor extension send no Origin at all.
So the real boundary was "any process running as this Windows user", in front
of routes that grant a shell (`/api/agents/run`), an already-authenticated
Claude session with every live OAuth grant (`/api/kernels`), and writes into
future session context (`/api/inject`).

What a shared secret does and does NOT buy
------------------------------------------
It does NOT fix a same-user compromise.  Malware running as you reads this
token file exactly the way the legitimate client does, presents it, and
authenticates "correctly" — the server cannot tell the presenter is malware
wearing your identity, because on this OS anything running as you HAS your
identity.  That is the classic confused-deputy shape, and auth shrinks that
attacker population by zero.

What it DOES buy, and why it is still worth having:
  * a blind port-prober is refused;
  * anything that cannot read the token file is refused (another account, a
    lower-integrity sandboxed child, a remote host that got past the bind);
  * a client can distinguish the real server from something that raced to bind
    the port (impostor/port-squatting), since the impostor cannot produce it.

The layer that WOULD close a same-user compromise is a human-presence consent
gate on the shell-granting routes.  It is NOT BUILT.  Until it is, auth is the
only layer, and it is a floor, not a fix — do not read this module as though
something above it is catching what it misses.  Tracked as the known gap.

Storage
-------
`<episodic>/panel_token`, minted once with `secrets.token_urlsafe(32)` and
created O_EXCL so two racing panel boots cannot mint different tokens.  The
file is deliberately NOT placed anywhere Syncthing replicates: the only synced
tree here reaches a lighthouse device on a public IP.  The Mac's copy is
provisioned out-of-band over scp.

HANDLING THE VALUE (learned the hard way, 2026-07-21)
-----------------------------------------------------
Protecting the FILE is not protecting the SECRET.  The first build of this
layer ACL'd the file correctly and kept it off every synced tree, then printed
`http://…/?k=<token>` to stdout on every boot and echoed the same query string
into the request log.  Started from a Claude Code shell, that put the live
token in a session transcript, in the episodic DB, and in a Vault folder
Syncthing replicates to a public-IP device — the exact outcome the storage
rules above exist to prevent.

So: the token value must never reach stdout, a log line, a request line, or
any other channel that gets archived.  `redact_query` exists for that, the CLI
writes the bootstrap URL to a 0600 file instead of printing it, and the browser
cookie carries a SEPARATE per-process secret (`mint_cookie_secret`) so a
cookie jar on disk never holds the real credential.

Public API
----------
  token_path(episodic_dir)             -> Path
  load_or_create_token(episodic_dir)   -> str
  mint_cookie_secret()                 -> str    (per-process, never persisted)
  harden_permissions(path)             -> str | None   (reason on failure)
  token_ok(presented, expected)        -> bool
  bearer_from_header(value)            -> str | None
  token_from_cookie(value)             -> str | None
  redact_query(request_line)           -> str    (log-safe request line)
  COOKIE_NAME

stdlib only.  Never prints from a library path — `harden_permissions` returns a
reason string instead so the CLI entry point decides what to say.
"""

from __future__ import annotations

import hmac
import http.cookies
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

#: `METHOD path HTTP/x.y` — the shape `redact_query` is allowed to rewrite.
_REQUEST_LINE_RE = re.compile(r"^[A-Z]+ \S+ HTTP/\d\.\d$")

#: Name of the cookie the browser bootstrap sets.  Distinct from any header so
#: the two presentation paths can be told apart in the audit log.
COOKIE_NAME = "memsom_panel"

#: Windows CREATE_NO_WINDOW.  The panel is normally spawned detached (no
#: console), so any child that inherits a console flashes one — the documented
#: gotcha behind `providers.base.run_no_window`.  Reproduced here rather than
#: imported to keep this module stdlib-only, matching panel.py's discipline.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

#: 32 bytes of urlsafe base64 ≈ 43 chars.  Long enough that online guessing is
#: irrelevant next to the fact that the file is readable by the same user.
_TOKEN_BYTES = 32


def token_path(episodic_dir) -> Path:
    """Where the token lives: a sibling of the audit log, never under a synced tree."""
    return Path(episodic_dir) / "panel_token"


def load_or_create_token(episodic_dir) -> str:
    """Return the panel token, minting it on first call.

    Concurrency: the mint uses O_CREAT|O_EXCL, so if two panel processes boot
    together exactly one writes and the loser re-reads the winner's value.
    Never returns an empty string — a truncated/blank file is re-minted, since
    an empty expected-token would otherwise compare equal to an empty
    presented-token and silently disable auth.
    """
    path = token_path(episodic_dir)
    existing = _read_token(path)
    if existing:
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # Either a racing boot won, or the file existed but was blank/unreadable.
        winner = _read_token(path)
        if winner:
            return winner
        # Blank file left behind — replace it in place.
        path.write_text(token + "\n", encoding="utf-8")
        harden_permissions(path)
        return token

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        # Don't leave a half-written token behind for the next boot to trust.
        try:
            path.unlink()
        except OSError:
            pass
        raise

    harden_permissions(path)
    return token


def mint_cookie_secret() -> str:
    """A fresh browser-session secret, held in memory for this process only.

    The browser bootstrap used to hand the REAL token to the cookie jar, which
    left a working credential sitting in a browser profile on disk for a day.
    A separate value costs nothing, is never written anywhere, and dies with
    the panel — so closing the panel invalidates every browser session, and a
    stolen cookie is worthless against the Authorization-header path the app
    uses.
    """
    return secrets.token_urlsafe(_TOKEN_BYTES)


def redact_query(request_line) -> str:
    """Strip the query string off an HTTP request line so it is safe to log.

    `BaseHTTPRequestHandler.log_request` passes `self.requestline` verbatim,
    which for the bootstrap route reads `GET /?k=<TOKEN> HTTP/1.1`.  Anything
    after the first `?` in the path is replaced with `?<redacted>` — the
    method, path and version stay readable, which is all a log needs.

    Deliberately conservative: `log_message` is also used for error strings
    that may legitimately contain a `?`, so anything that does not look like
    `METHOD path HTTP/x.y` is returned untouched rather than truncated.
    """
    line = str(request_line or "")
    if not _REQUEST_LINE_RE.match(line):
        return line
    head, sep, rest = line.partition("?")
    if not sep:
        return line
    # Keep the trailing " HTTP/1.1" so a redacted line still parses as one.
    version = rest[rest.rfind(" "):] if " " in rest else ""
    return f"{head}?<redacted>{version}"


def _read_token(path: Path) -> str:
    """Read and strip the token file; '' if absent, unreadable, or blank."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def harden_permissions(path) -> str | None:
    """Restrict *path* to the current user.  Returns None on success, else a
    human-readable reason.

    Never raises and never prints: a failure here weakens defence-in-depth but
    must not stop the panel from booting, and this is a library path.  On a
    single-admin box this is close to cosmetic — it earns its keep the day a
    second account exists, or against a lower-integrity child.
    """
    path = Path(path)
    if sys.platform != "win32":
        try:
            os.chmod(path, 0o600)
            return None
        except OSError as exc:
            return f"chmod failed: {exc}"

    principal = _current_user_sid()
    if principal is None:
        # Name fallback. Bare USERNAME, never DOMAIN\\USERNAME: over SSH
        # USERDOMAIN reads as WORKGROUP while the account actually lives on the
        # machine account domain, and "WORKGROUP\\user" fails to map (icacls
        # 1332). A bare local name resolves in both contexts.
        user = os.environ.get("USERNAME") or ""
        if not user:
            return "cannot resolve current user; left inherited ACL in place"
        principal = user
    try:
        proc = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{principal}:F"],
            capture_output=True, text=True, timeout=15,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"icacls could not run: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return f"icacls exited {proc.returncode}: {detail[0] if detail else 'no output'}"
    return None


def _current_user_sid() -> str | None:
    """The current user's SID in icacls form (`*S-1-5-21-…`), or None.

    A SID always resolves in an ACL; a *name* may not. This matters in exactly
    the case that bit us: run over SSH, `USERDOMAIN` reads as `WORKGROUP` while
    the account really lives on the machine-account domain, so granting
    `WORKGROUP\\<user>` fails with "No mapping between account names and
    security IDs" (1332) and the file silently keeps its inherited ACL.
    """
    try:
        proc = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=15,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for field in (proc.stdout or "").replace('"', "").split(","):
        field = field.strip()
        if field.upper().startswith("S-1-"):
            return f"*{field}"
    return None


def token_ok(presented, expected) -> bool:
    """Constant-time token comparison.

    Empty/None on EITHER side is False — never let a missing expected-token
    turn into "everything authenticates".  `compare_digest` is used rather than
    `==` so a timing side channel can't be walked to recover the token.
    """
    if not presented or not expected:
        return False
    return hmac.compare_digest(str(presented), str(expected))


def bearer_from_header(value) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header value.

    The scheme is matched case-insensitively (RFC 7235 says it is
    case-insensitive); the token itself is returned verbatim.
    """
    if not value:
        return None
    parts = value.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def token_from_cookie(value) -> str | None:
    """Extract the token from a `Cookie:` header value, or None.

    Used by the browser bootstrap path (`GET /?k=…` sets an HttpOnly cookie),
    which exists because the served page's fetch helpers use relative URLs and
    therefore cannot carry an Authorization header without editing the inline
    script — and editing it would change the CSP script hash.
    """
    if not value:
        return None
    jar = http.cookies.SimpleCookie()
    try:
        jar.load(value)
    except http.cookies.CookieError:
        return None
    morsel = jar.get(COOKIE_NAME)
    if morsel is None:
        return None
    return morsel.value or None
