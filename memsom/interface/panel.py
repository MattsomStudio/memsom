"""memsom.interface.panel — the tuning/telemetry panel server (WP2 + WP3).

Why this exists: the forgetting layer, the digest budget, the recall/hook/sweep
tunables and a handful of Windows scheduled tasks are all "knobs" someone has to
turn by hand-editing JSON/`.cmd` files or fighting Task Scheduler's GUI. This
module is the ONE place that turns/read them from a browser instead: a small
stdlib-only HTTP server, a handful of file-format-aware providers, and a
hand-rolled dark-theme page with zero external requests.

Security posture (lead with the threats — this is a local dev tool, not a
public service, and the design leans entirely on that):
  - Loopback-only, no TLS story, no auth. `build_config`/`build_server` REFUSE
    to bind anything outside {127.0.0.1, localhost, ::1} — that's the whole
    trust boundary. There is deliberately no bearer/session layer: the box
    itself is the credential.
  - DNS-rebinding defence anyway: every request's Host header is checked
    against an exact allowlist (loopback names + the RUNTIME port) before any
    routing happens, so a malicious page in a browser tab that's *already* on
    localhost can't rebind a hostname to us and issue same-origin requests.
  - Every POST additionally requires `Content-Type: application/json` and, if
    an `Origin` header is present, that it names this exact origin — the CSRF
    belt to the Host-header suspenders.
  - The page ships a single inline `<script>`, CSP-pinned by its own sha256
    hash (`script-src 'sha256-...'`), `default-src 'none'`, no CDN, no
    innerHTML with server/knob data (textContent + DOM builders only) — a
    poisoned memory stem or task name can never become live markup or script.
  - Writes are bounds-checked and REJECTED on violation, never clamped, and
    every write attempt (including refusals) gets a two-phase JSONL audit
    line (intent, then result) so a crash mid-write is detectable, not silent.

Public API
----------
  build_config(profile_path, *, host, port, run=None, urlopen=None,
               memory_builder=None) -> PanelConfig
  build_server(config) -> ThreadingHTTPServer
  handle_knob_write(config, payload) -> (status, body)   the POST /api/knobs pipeline
  build_knobs_payload(config) -> dict                     the GET /api/knobs pipeline
  validate_bounds(knob, value) -> str | None               bounds/type check, never clamps
  scan_crash_markers(audit_log_path) -> list
  CanonicalParamsProvider / JsonFileProvider / SetLineProvider / SchtasksProvider
  register(sub)                                            CLI hook: `memsom panel`

stdlib only. Never prints from a library path (only the CLI entry does).
"""

from __future__ import annotations

import base64
import errno
import hashlib
import html
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from memsom.interface import dashboard
from memsom.interface import saveall as saveall_runner
from memsom.interface import telemetry
from memsom.lifecycle import forget
from memsom.providers import build_registry
from memsom.providers import handlers as provider_handlers
from memsom.providers import voice_handlers
from memsom.providers import agent_handlers
from memsom.providers import kernel_handlers
from memsom.providers.agent_store import GraphStore
from memsom.providers.kernels import KernelRunner, KernelStore
from memsom.providers.agents import AgentRunner
from memsom.providers.schedule import Scheduler
from memsom.providers.base import run_no_window
from memsom.providers.session import SessionRunner

try:
    from importlib.metadata import PackageNotFoundError as _PkgNotFound
    from importlib.metadata import version as _pkg_version
    try:
        _MEMSOM_VERSION = _pkg_version("memsom")
    except _PkgNotFound:
        _MEMSOM_VERSION = "dev"
except ImportError:  # pragma: no cover - importlib.metadata ships with 3.12
    _MEMSOM_VERSION = "dev"


# ---------------------------------------------------------------------------
# Bind policy: loopback only, no TLS story, no exceptions.
# ---------------------------------------------------------------------------

_ALLOWED_BIND_HOSTS = ("127.0.0.1", "localhost", "::1")
_ALLOWED_HOST_HEADER_NAMES = ("127.0.0.1", "localhost", "[::1]")


def enforce_bind_policy(host: str) -> None:
    """Raise ValueError unless *host* is exactly one of the loopback spellings.

    Called at config-build time, before any socket exists — "build time", not
    "bind time": there is no TLS story here on purpose, so anything else is a
    hard refusal, not a warning.
    """
    if host not in _ALLOWED_BIND_HOSTS:
        raise ValueError(
            f"refusing to build a panel config for host {host!r}: the panel is "
            f"loopback-only ({', '.join(_ALLOWED_BIND_HOSTS)}) — there is no "
            f"TLS/auth story here on purpose"
        )


def _host_header_allowed(host_header, port) -> bool:
    """True if *host_header* (raw `Host:` value) names loopback at the RUNTIME
    port. IPv6-bracket-correct. A missing Host, an off-list hostname, or a port
    that doesn't match the port we're actually bound to is rejected — this is
    the DNS-rebinding defence, so it must be exact, not permissive.
    """
    if not host_header:
        return False
    h = host_header.strip().lower()
    if h.startswith("["):
        end = h.find("]")
        if end == -1:
            return False
        hostname = h[:end + 1]
        rest = h[end + 1:]
        if rest:
            if not rest.startswith(":") or rest[1:] != str(port):
                return False
    else:
        parts = h.split(":", 1)
        hostname = parts[0]
        if len(parts) == 2 and parts[1] != str(port):
            return False
    return hostname in _ALLOWED_HOST_HEADER_NAMES


def _allowed_origins(config: "PanelConfig") -> set:
    return {
        f"http://127.0.0.1:{config.port}", f"http://localhost:{config.port}",
        # The Tauri desktop app's own webview origin. plugin-http forwards this
        # on POST (WebView2 on Windows = http://tauri.localhost; wkwebview on
        # macOS = tauri://localhost), which the loopback allowlist would other-
        # wise reject — blocking every /api/* mutation from the app. It's the
        # app's own loopback origin; the bind stays 127.0.0.1-only regardless.
        "http://tauri.localhost", "https://tauri.localhost", "tauri://localhost",
    }


def _q1(query: dict, key: str, default=None):
    """First value of a parsed querystring key (parse_qs yields lists)."""
    vals = query.get(key)
    return vals[0] if vals else default


def _qint(query: dict, key: str, default: int = 0) -> int:
    v = _q1(query, key)
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def load_profile(path) -> dict:
    """Load + minimally validate a panel_profile.json. Raises FileNotFoundError
    if missing, ValueError if malformed — never starts with a half-built one."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"panel profile not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed panel profile {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"panel profile {p}: top level must be a JSON object")
    data.setdefault("knobs", [])
    data.setdefault("tasks", [])
    data.setdefault("contracts", [])
    data.setdefault("telemetry", {})
    if not data.get("audit_log"):
        raise ValueError(f"panel profile {p}: 'audit_log' path is required")
    return data


def find_knob(profile: dict, knob_id: str):
    """Resolve a knob id against the profile: a real (provider-backed) knob
    from `knobs`, or a synthetic tier-3 (no-provider) entry from `contracts`.
    Returns None if unknown. NEVER reads target/key from anywhere but the
    profile — the caller must not be able to smuggle either through a request.
    """
    for k in profile.get("knobs", []):
        if k.get("id") == knob_id:
            entry = dict(k)
            entry.setdefault("tier", 1)
            return entry
    for c in profile.get("contracts", []):
        if c.get("id") == knob_id:
            return {"id": knob_id, "tier": 3, "provider": None,
                    "label": c.get("label"), "default": c.get("value"),
                    "note": c.get("note")}
    return None


def _task_entry(profile: dict, name):
    for t in profile.get("tasks", []):
        if t.get("name") == name:
            return t
    return None


# ---------------------------------------------------------------------------
# Bounds/type validation — shared by the route AND (redundantly, per-provider)
# by json-file. Rejects; never clamps.
# ---------------------------------------------------------------------------

class KnobValidationError(ValueError):
    """A candidate value failed bounds/type validation. str(exc) is the
    human-readable rejection reason. The write pipeline always refuses on
    this — it never clamps a rejected value into range."""


def _typename(value) -> str:
    return type(value).__name__


def validate_bounds(knob: dict, value) -> "str | None":
    """Return a rejection reason string, or None if *value* is acceptable for
    *knob*. Checked against `knob['type']` (int/float/bool/list-of-str/weights)
    and `knob['bounds']` (min/max, numeric types only). An unknown or absent
    type REFUSES the write — fail closed, the reject-not-clamp rule's sibling:
    a knob we can't validate is a knob we don't write. Never mutates or
    coerces `value`.
    """
    ktype = knob.get("type")
    if ktype == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected an int, got {_typename(value)}"
    elif ktype == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"expected a number, got {_typename(value)}"
    elif ktype == "bool":
        if not isinstance(value, bool):
            return f"expected a bool, got {_typename(value)}"
    elif ktype == "list-of-str":
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            return "expected a list of strings"
    elif ktype == "weights":
        # Structural check against the knob's DEFAULT: exactly the same keys,
        # each a list of the same length, every entry a number in [0, 1].
        # Shields consumers like recall.py's route_weights table, which index
        # the value by key with no guard of their own.
        default = knob.get("default")
        if not isinstance(value, dict) or not isinstance(default, dict):
            return "expected a weights table (object of numeric lists)"
        if set(value) != set(default):
            return f"weights keys must be exactly {sorted(default)}"
        for wk, wv in value.items():
            dv = default[wk]
            if (not isinstance(wv, list) or not isinstance(dv, list)
                    or len(wv) != len(dv)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or x < 0 or x > 1 for x in wv)):
                return f"weights[{wk!r}] must be {len(dv) if isinstance(dv, list) else '?'} numbers in [0, 1]"
    else:
        return f"unknown knob type {ktype!r}; write refused (fail closed)"

    if ktype in ("int", "float"):
        bounds = knob.get("bounds") or {}
        lo, hi = bounds.get("min"), bounds.get("max")
        if lo is not None and value < lo:
            return f"{value} is below the minimum ({lo})"
        if hi is not None and value > hi:
            return f"{value} is above the maximum ({hi})"
    return None


# ---------------------------------------------------------------------------
# Shared atomic writer — both file providers. Never reuses a fixed .tmp name
# (Windows readers / Syncthing can be mid-read of a stale one); retries a
# transient PermissionError a few times before giving up.
# ---------------------------------------------------------------------------

# Serializes every knob write in this process: ThreadingHTTPServer handles each
# request on its own thread, and the providers' read-modify-write of a shared
# file is a textbook lost-update race without it (two concurrent writes both
# reading the same snapshot; the second os.replace silently discards the
# first). One process-wide lock is plenty at single-user scale.
_KNOB_WRITE_LOCK = threading.Lock()

# Unique tmp name PER CALL, not just per process — two panel threads writing
# the same target would otherwise collide on one tmp path mid-replace.
_TMP_COUNTER = itertools.count()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(
        target.name + f".panel-{os.getpid()}-{next(_TMP_COUNTER)}.tmp")
    tmp.write_bytes(data)
    attempts = 5
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.1)


def _atomic_write_json(target: Path, obj) -> None:
    _atomic_write_bytes(target, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))


def _read_json_or_none(path) -> "dict | None":
    """A dict on success (possibly empty — that's a VALID read), None on
    missing/corrupt/non-object. Callers that must distinguish 'file is
    legitimately empty' from 'read failed' use this, not _read_json_or_empty."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_json_or_empty(path) -> dict:
    data = _read_json_or_none(path)
    return {} if data is None else data


# ---------------------------------------------------------------------------
# Providers — one per `knob["provider"]`. Each: read(knob) -> current,
# write(knob, value) -> the new current (or a result dict for schtasks).
# ---------------------------------------------------------------------------

_LEGACY_CRUFT_KEYS = ("cap", "decay", "gain", "seed")


class CanonicalParamsProvider:
    """provider = 'canonical-params': target is a store's `.weights/canonical.json`.

    `_pre_reread_hook`, if given, runs after the baseline read and before the
    fresh re-read that immediately precedes the atomic write — the seam a test
    uses to simulate a concurrent writer (the forgetting-layer reconcile)
    mutating `memories` mid-write, proving the merge survives it.
    """

    def __init__(self, *, _pre_reread_hook=None):
        self._pre_reread_hook = _pre_reread_hook

    def read(self, knob):
        params, _warnings = forget.load_params(knob["target"])
        return params[knob["key"]]

    def write(self, knob, value):
        target = Path(knob["target"])
        base = _read_json_or_empty(target)

        if self._pre_reread_hook is not None:
            self._pre_reread_hook(target)

        # Re-read immediately before assembling the write: defends the
        # reconcile read-modify-write window by taking `memories`/`version`/
        # every unknown key from the FRESHEST read, not our stale baseline.
        # `is not None`: a legitimately-empty {} re-read is a real read and
        # must win over the stale baseline, not be discarded as a failure.
        fresh = _read_json_or_none(target)
        source = fresh if fresh is not None else base

        params = dict(source.get("params", base.get("params", {})))
        params[knob["key"]] = value
        for cruft in _LEGACY_CRUFT_KEYS:
            params.pop(cruft, None)

        out = {k: v for k, v in source.items() if k not in ("params", "updated")}
        out.setdefault("version", 1)
        out.setdefault("memories", {})
        out["params"] = params
        out["updated"] = forget.now_iso()
        _atomic_write_json(target, out)
        return value


_MISSING = object()


def _dig(data, path_parts):
    cur = data
    for part in path_parts:
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _set_dig(data, path_parts, value):
    cur = data
    for part in path_parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[path_parts[-1]] = value


class JsonFileProvider:
    """provider = 'json-file': a dotted `key` path into any JSON file.
    Preserves every unrelated key; creates intermediate dicts on write."""

    def read(self, knob):
        data = _read_json_or_empty(knob["target"])
        value = _dig(data, str(knob["key"]).split("."))
        return knob.get("default") if value is _MISSING else value

    def write(self, knob, value):
        reason = validate_bounds(knob, value)
        if reason:
            raise KnobValidationError(reason)
        target = Path(knob["target"])
        data = _read_json_or_empty(target)
        _set_dig(data, str(knob["key"]).split("."), value)
        _atomic_write_json(target, data)
        return value


def _coerce_setline(ktype, raw: str):
    raw = raw.strip()
    if ktype == "int":
        return int(raw)
    if ktype == "float":
        return float(raw)
    if ktype == "bool":
        return raw.lower() in ("1", "true", "yes", "on")
    return raw


def _format_setline(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _setline_pattern(key: str) -> "re.Pattern":
    return re.compile(r"^set " + re.escape(key) + r"=[^\r\n]*\r?\n?", re.MULTILINE)


class SetLineProvider:
    """provider = 'set-line': target file holds `set KEY=VALUE` lines (batch
    env-var style) plus `@rem` comments. Exact-line rewrite (or append if the
    key is absent); every other byte in the file survives untouched. CRLF,
    ASCII-only (matches the real .cmd files this targets)."""

    def read(self, knob):
        target = Path(knob["target"])
        if not target.is_file():
            return knob.get("default")
        text = target.read_bytes().decode("ascii", errors="strict")
        m = _setline_pattern(knob["key"]).search(text)
        if not m:
            return knob.get("default")
        raw = m.group(0).split("=", 1)[1].rstrip("\r\n")
        return _coerce_setline(knob.get("type"), raw)

    def write(self, knob, value):
        reason = validate_bounds(knob, value)
        if reason:
            raise KnobValidationError(reason)
        target = Path(knob["target"])
        text = target.read_bytes().decode("ascii", errors="strict") if target.is_file() else ""
        new_line = f"set {knob['key']}={_format_setline(value)}\r\n"
        pattern = _setline_pattern(knob["key"])
        m = pattern.search(text)
        if m:
            new_text = text[:m.start()] + new_line + text[m.end():]
        else:
            if text and not text.endswith(("\r\n", "\n", "\r")):
                text += "\r\n"
            new_text = text + new_line
        _atomic_write_bytes(target, new_text.encode("ascii"))
        return value


def _ps_quote(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


_PS_TASK_READ = (
    "$t = Get-ScheduledTask -TaskName {name} -ErrorAction Stop; "
    "$i = Get-ScheduledTaskInfo -TaskName {name} -ErrorAction SilentlyContinue; "
    # Human trigger summary, not the raw CIM class name; datetimes as ISO strings
    # (ConvertTo-Json would otherwise emit /Date(ms)/ CIM artifacts), with the
    # Task Scheduler 1999 never-ran sentinel mapped to null.
    "$trig = ($t.Triggers | ForEach-Object {{ "
    "$c = $_.CimClass.CimClassName -replace '^MSFT_Task' -replace 'Trigger$'; "
    "$r = $_.Repetition.Interval; if ($r) {{ \"$c every $r\" }} else {{ $c }} }}) -join ', '; "
    "[PSCustomObject]@{{trigger=$trig; enabled=[bool]$t.Settings.Enabled; "
    "lastRunTime=$(if ($i.LastRunTime -and $i.LastRunTime.Year -gt 2000) "
    "{{ $i.LastRunTime.ToString('yyyy-MM-ddTHH:mm:ss') }} else {{ $null }}); "
    "nextRunTime=$(if ($i.NextRunTime) {{ $i.NextRunTime.ToString('yyyy-MM-ddTHH:mm:ss') }} "
    "else {{ $null }})}} "
    "| ConvertTo-Json -Compress"
)

_PS_TASK_WRITE = (
    "$tr = New-ScheduledTaskTrigger -Once -At (Get-Date) "
    "-RepetitionInterval (New-TimeSpan -Minutes {value}) "
    # NOT [TimeSpan]::MaxValue: it serializes to P99999999DT23H59M59S, which Task
    # Scheduler rejects as out of range (caught live). 3650 days matches the
    # repetition-duration convention of long-lived local tasks.
    "-RepetitionDuration (New-TimeSpan -Days 3650); "
    "Set-ScheduledTask -TaskName {name} -Trigger $tr | Out-Null"
)


def _ps_run(run, script, timeout=10):
    return run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
               capture_output=True, text=True, timeout=timeout)


def _elevated_command(name, value) -> str:
    inner = _PS_TASK_WRITE.format(name=_ps_quote(name), value=value)
    return 'powershell -NoProfile -Command "' + inner.replace('"', '\\"') + '"'


class SchtasksProvider:
    """provider = 'schtasks' (tier 2): `key` = a Windows Scheduled Task name.

    Read/write both go through the injectable `run` (a subprocess.run-alike),
    so unit tests never touch real PowerShell. Task names are only ever taken
    from the resolved knob dict — which itself only ever comes from the
    profile the panel loaded at startup — never from a request body, so a
    client cannot shell-inject an arbitrary task name.
    """

    def __init__(self, run=None):
        self._run = run or subprocess.run

    def read(self, knob):
        name = knob["key"]
        try:
            r = _ps_run(self._run, _PS_TASK_READ.format(name=_ps_quote(name)))
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return {"error": f"powershell unavailable: {exc}"}
        if r.returncode != 0:
            return {"error": (r.stderr or "Get-ScheduledTask failed").strip()}
        try:
            data = json.loads(r.stdout)
        except (json.JSONDecodeError, TypeError):
            return {"error": "could not parse scheduled-task output"}
        if not isinstance(data, dict):
            return {"error": "unexpected scheduled-task output shape"}
        return {"trigger": data.get("trigger"), "enabled": data.get("enabled"),
                "last_run": data.get("lastRunTime"), "next_run": data.get("nextRunTime")}

    def write(self, knob, value):
        """Attempt Set-ScheduledTask. On ANY failure (access denied is the
        expected case — task files need an elevated token) degrade to a
        copyable elevated command instead of raising."""
        name = knob["key"]
        elevated = _elevated_command(name, value)
        try:
            r = _ps_run(self._run, _PS_TASK_WRITE.format(name=_ps_quote(name), value=value),
                        timeout=20)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"ok": False, "degraded": True, "elevated_command": elevated}
        if r.returncode != 0:
            return {"ok": False, "degraded": True, "elevated_command": elevated}
        return {"ok": True, "current": value}


# ---------------------------------------------------------------------------
# Two-phase audit log + crash-marker detection
# ---------------------------------------------------------------------------

def _audit_append(path, obj) -> None:
    """Append one JSONL line, fsync'd. Raises OSError on failure — the caller
    decides what that means (an INTENT-line failure refuses the write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    fh = open(path, "a", encoding="utf-8")
    try:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    finally:
        fh.close()


def _audit_result(config: "PanelConfig", knob_id, result) -> None:
    """Best-effort RESULT line. A failure here must NOT be reported as a write
    failure — the write already happened (or was already refused); only the
    INTENT line gates the write itself."""
    try:
        _audit_append(config.audit_log_path, {"ts": forget.now_iso(), "knob": knob_id,
                                              "result": result})
    except OSError:
        pass


def scan_crash_markers(audit_log_path, *, tail_lines: int = 4000) -> list:
    """Scan the audit log tail for a 'pending' intent line with no following
    result line for the same knob — evidence of a crash mid-write."""
    path = Path(audit_log_path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) > tail_lines:
        lines = lines[-tail_lines:]
    pending: dict = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        knob = rec.get("knob")
        if knob is None:
            continue
        if rec.get("result") == "pending":
            pending[knob] = rec
        else:
            pending.pop(knob, None)
    return list(pending.values())


# ---------------------------------------------------------------------------
# The POST /api/knobs pipeline — resolve -> tier3/bounds -> two-phase audit ->
# provider write. Pure w.r.t. HTTP: the route handler is a thin JSON-in/
# JSON-out wrapper around this, so it's directly unit-testable.
# ---------------------------------------------------------------------------

def _perform_knob_write(config: "PanelConfig", knob: dict, value):
    knob_id = knob["id"]

    if knob.get("tier") == 3 or not knob.get("provider"):
        _audit_result(config, knob_id, "refused-tier3")
        return 403, {"ok": False, "error": "tier 3 is a fixed contract value; read-only"}

    reason = validate_bounds(knob, value)
    if reason:
        _audit_result(config, knob_id, f"refused-invalid: {reason}")
        return 400, {"ok": False, "error": reason}

    provider = config.providers.get(knob["provider"])
    if provider is None:
        _audit_result(config, knob_id, f"failed: unknown provider {knob['provider']!r}")
        return 500, {"ok": False, "error": f"unknown provider: {knob['provider']!r}"}

    if knob["provider"] == "schtasks":
        task = _task_entry(config.profile, knob["key"])
        writable = bool(task and task.get("writable"))
        if not writable:
            _audit_result(config, knob_id, "refused-readonly")
            return 403, {"ok": False, "error": "read-only by contract (password-logon task)"}
        # A schtasks value is a repetition interval in whole minutes: the knob
        # must be typed int (validated fail-closed above), so int() here is a
        # no-op guard, never a silent truncation. A float-typed schtasks knob
        # is a profile bug — refuse it rather than truncate 30.7 -> 30 while
        # the audit intent line says 30.7.
        if knob.get("type") != "int":
            _audit_result(config, knob_id,
                          "refused-invalid: schtasks knobs must be type int")
            return 400, {"ok": False,
                         "error": "schtasks knobs must be type int (whole minutes)"}
        try:
            with _KNOB_WRITE_LOCK:
                result = provider.write(knob, int(value))
        except Exception as exc:
            _audit_result(config, knob_id, f"failed: {exc}")
            return 500, {"ok": False, "error": str(exc)}
        if not result.get("ok", False):
            _audit_result(config, knob_id, "failed: degraded (elevated command required)")
            return 200, result
        _audit_result(config, knob_id, "ok")
        return 200, {"ok": True, "current": result.get("current", value)}

    try:
        with _KNOB_WRITE_LOCK:
            current = provider.write(knob, value)
    except KnobValidationError as exc:
        _audit_result(config, knob_id, f"refused-invalid: {exc}")
        return 400, {"ok": False, "error": str(exc)}
    except Exception as exc:
        _audit_result(config, knob_id, f"failed: {exc}")
        return 500, {"ok": False, "error": str(exc)}

    _audit_result(config, knob_id, "ok")
    return 200, {"ok": True, "current": current}


def handle_knob_write(config: "PanelConfig", payload: dict):
    """The full POST /api/knobs pipeline. Returns (http_status, body_dict)."""
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    knob_id = payload.get("id")
    if not knob_id or not isinstance(knob_id, str):
        return 400, {"ok": False, "error": "missing required field: id"}
    reset = bool(payload.get("reset"))
    if not reset and "value" not in payload:
        return 400, {"ok": False, "error": "missing required field: value (or reset: true)"}

    knob = find_knob(config.profile, knob_id)
    if knob is None:
        return 404, {"ok": False, "error": f"unknown knob id: {knob_id!r}"}

    value = knob.get("default") if reset else payload.get("value")
    provider = config.providers.get(knob.get("provider")) if knob.get("provider") else None
    old_value = knob.get("default")
    if provider is not None:
        try:
            old_value = provider.read(knob)
        except Exception:
            pass  # best-effort: the audit line still records SOMETHING for `old`

    try:
        _audit_append(config.audit_log_path, {
            "ts": forget.now_iso(), "knob": knob_id, "old": old_value, "new": value,
            "result": "pending",
        })
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit log unavailable; write refused: {exc}"}

    return _perform_knob_write(config, knob, value)


# ---------------------------------------------------------------------------
# GET /api/knobs payload assembly
# ---------------------------------------------------------------------------

def _evidence_for(config: "PanelConfig", knob: dict, current):
    """Evidence for exactly two canonical-params knobs (hardcoded pairing —
    everything else has no evidence)."""
    if knob.get("provider") != "canonical-params":
        return None
    key = knob.get("key")
    if key == "demote_below":
        try:
            rows = dashboard.load_weights()
        except SystemExit:
            return None
        threshold = current if isinstance(current, (int, float)) else knob.get("default")
        count = sum(1 for r in rows
                    if not r["pinned"] and threshold <= r["weight"] <= threshold + 0.05)
        return {"unpinned_within_0.05_above": count}
    if key == "memory_budget":
        try:
            mem = config.memory_cache.get()
        except SystemExit:
            return None
        budget = mem.get("budget")
        if not budget:
            return None
        return {"bytes": budget["bytes"], "cap": budget["cap"], "pct": budget["pct"]}
    return None


def _task_status(config: "PanelConfig", name):
    provider = config.providers.get("schtasks")
    if provider is None or not name:
        return {"error": "schtasks provider unavailable"}
    try:
        return provider.read({"key": name})
    except Exception as exc:
        return {"error": str(exc)}


def build_knobs_payload(config: "PanelConfig") -> dict:
    out_knobs = []
    for raw in config.profile.get("knobs", []):
        knob = dict(raw)
        knob.setdefault("tier", 1)
        provider = config.providers.get(knob.get("provider")) if knob.get("provider") else None
        current, error = None, None
        if provider is not None:
            try:
                current = provider.read(knob)
            except Exception as exc:
                error = str(exc)
        default = knob.get("default")
        entry = {
            "id": knob["id"], "tier": knob.get("tier"), "label": knob.get("label"),
            "provider": knob.get("provider"), "type": knob.get("type"),
            "bounds": knob.get("bounds") or {}, "default": default,
            "current": current, "dirty": None if error else (current != default),
            "requires": knob.get("requires"), "note": knob.get("note"),
        }
        if error:
            entry["error"] = error
        else:
            evidence = _evidence_for(config, knob, current)
            if evidence is not None:
                entry["evidence"] = evidence
        if knob.get("provider") == "schtasks" and knob.get("key"):
            entry["task"] = _task_status(config, knob["key"])
        out_knobs.append(entry)

    contracts = [dict(c, tier=3) for c in config.profile.get("contracts", [])]

    tasks = []
    for t in config.profile.get("tasks", []):
        entry = dict(t)
        entry["status"] = _task_status(config, t.get("name"))
        tasks.append(entry)

    return {
        "knobs": out_knobs,
        "tasks": tasks,
        "contracts": contracts,
        "crash_markers": scan_crash_markers(config.audit_log_path),
    }


# ---------------------------------------------------------------------------
# GET /api/memory — build_telemetry() cached with a 30s TTL
# ---------------------------------------------------------------------------

class MemoryCache:
    """In-process cache for dashboard.build_telemetry(): it re-reads every
    memory file, and the underlying data only changes once per Stop-hook
    render, so a fixed TTL is plenty. `builder`/`clock` are injectable so
    tests never touch real files or real time."""

    def __init__(self, *, builder=None, ttl: float = 30.0, clock=None, now=None):
        self._builder = builder or dashboard.build_telemetry
        self._ttl = ttl
        self._clock = clock or time.monotonic
        # `now` stamps `cached_at` (wall-clock, for the UI) separately from
        # `clock` (monotonic, for TTL comparison) - injectable so a test can
        # tell two builds apart even if they land in the same clock tick.
        self._now = now or (lambda: datetime.now(timezone.utc).isoformat())
        self._lock = threading.Lock()
        self._data = None
        self._at = None
        self._at_iso = None

    def get(self, *, refresh: bool = False) -> dict:
        with self._lock:
            now = self._clock()
            stale = self._data is None or self._at is None or (now - self._at) >= self._ttl
            if refresh or stale:
                self._data = self._builder()
                self._at = now
                self._at_iso = self._now()
            payload = dict(self._data)
            payload["cached_at"] = self._at_iso
            return payload


# ---------------------------------------------------------------------------
# Activity feeds (DECK): workflow runs read straight off Claude Code's on-disk
# session records. Read-only, cache-fronted, and defensive per-file — a
# half-written record from a LIVE workflow run must never 500 the endpoint.
# ---------------------------------------------------------------------------

def default_claude_dir() -> Path:
    """`~/.claude` from the live environment (same identity-fence idiom as the
    desktop app's settings: nothing user-specific is baked in)."""
    base = os.environ.get("USERPROFILE") or os.environ.get("HOME") or ""
    return Path(base) / ".claude"


def _scan_workflow_runs(projects_dir: Path, limit: int = 20) -> dict:
    """Newest workflow-run records across all projects/sessions.

    Source shape (one JSON file per run, written by the Claude Code harness):
    projects/<project>/<session>/workflows/wf_*.json with status, phases[],
    and workflowProgress[] carrying per-agent telemetry. Every field access
    is .get()-with-default: the schema is the harness's, not ours, and it may
    drift between versions.
    """
    try:
        candidates = sorted(
            projects_dir.glob("*/*/workflows/wf_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
    except OSError:
        candidates = []

    runs = []
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue  # live/half-written record: skip, never 500
        if not isinstance(record, dict):
            continue
        progress = record.get("workflowProgress") or []
        agents = [
            {
                "label": entry.get("label"),
                "agentId": entry.get("agentId"),
                "model": entry.get("model"),
                "state": entry.get("state"),
                "tokens": entry.get("tokens"),
                "toolCalls": entry.get("toolCalls"),
                "durationMs": entry.get("durationMs"),
                "lastToolName": entry.get("lastToolName"),
            }
            for entry in progress
            if isinstance(entry, dict) and entry.get("type") == "workflow_agent"
        ]
        phases = [
            {"title": ph.get("title"), "detail": ph.get("detail")}
            for ph in (record.get("phases") or [])
            if isinstance(ph, dict)
        ]
        runs.append(
            {
                "runId": record.get("runId"),
                "workflowName": record.get("workflowName"),
                "status": record.get("status"),
                "startTime": record.get("startTime"),
                "durationMs": record.get("durationMs"),
                "totalTokens": record.get("totalTokens"),
                "totalToolCalls": record.get("totalToolCalls"),
                "summary": record.get("summary"),
                "session_id": path.parts[-3] if len(path.parts) >= 3 else None,
                "project": path.parts[-4] if len(path.parts) >= 4 else None,
                "phases": phases,
                "agents": agents,
            }
        )
    return {"runs": runs}


def _tail_jsonl(path: Path, window_bytes: int = 256 * 1024) -> list:
    """Parse the last `window_bytes` of an append-only JSONL file, newest
    LAST. Bad/partial lines (the window may start mid-line, concurrent
    appends may interleave) are skipped, never raised."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - window_bytes))
            raw = f.read()
    except OSError:
        return []
    lines = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            lines.append(obj)
    return lines


# session_id becomes a FILENAME under episodic/inbox — this regex is the
# path-traversal fence. Claude Code session ids are UUIDs; anything that
# isn't a plain slug is refused, never sanitized.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$")


def _read_memory_activity(claude_dir: Path, limit: int = 100) -> dict:
    """Memory-injection events (askq_recall hits + panel injects) from
    memory_activity.jsonl, newest first, plus the live-session picker list
    (any transcript touched in the last 24h)."""
    events = list(reversed(_tail_jsonl(claude_dir / "episodic" / "memory_activity.jsonl")))[:limit]

    sessions = []
    cutoff = time.time() - 24 * 3600
    projects = claude_dir / "projects"
    try:
        for transcript in projects.glob("*/*.jsonl"):
            try:
                mtime = transcript.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                continue
            sid = transcript.stem
            if not _SESSION_ID_RE.match(sid):
                continue
            sessions.append(
                {
                    "session_id": sid,
                    "project": transcript.parent.name,
                    "last_active": datetime.fromtimestamp(
                        mtime, tz=timezone.utc).isoformat(),
                }
            )
    except OSError:
        pass
    sessions.sort(key=lambda s: s["last_active"], reverse=True)
    return {"events": events, "sessions": sessions[:15]}


def _audit_inject_result(config: "PanelConfig", session_id: str, result: str) -> None:
    """Best-effort RESULT line for an inject (same contract as _audit_result:
    only the INTENT line gates the write)."""
    try:
        _audit_append(config.audit_log_path, {"ts": forget.now_iso(),
                                              "inject": session_id, "result": result})
    except OSError:
        pass


def handle_inject(config: "PanelConfig", payload: dict):
    """POST /api/inject pipeline: validate -> two-phase audit -> append to the
    session's inbox file (drained into context by the inbox_drain.py hook) ->
    journal -> optional best-effort persist to the memory store.

    Pure w.r.t. HTTP, same as handle_knob_write. Returns (status, body)."""
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    session_id = payload.get("session_id")
    text = payload.get("text")
    persist = bool(payload.get("persist"))
    if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
        return 400, {"ok": False, "error": "invalid session_id"}
    if not isinstance(text, str) or not text.strip():
        return 400, {"ok": False, "error": "missing required field: text"}
    if len(text.encode("utf-8")) > 8192:
        # headroom under the harness's 10KB additionalContext cap
        return 400, {"ok": False, "error": "text too long (max 8KB)"}

    try:
        _audit_append(config.audit_log_path, {
            "ts": forget.now_iso(), "inject": session_id,
            "chars": len(text), "persist": persist, "result": "pending",
        })
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit log unavailable; inject refused: {exc}"}

    inbox = config.claude_dir / "episodic" / "inbox"
    try:
        inbox.mkdir(parents=True, exist_ok=True)
        # APPEND, never overwrite: two injects racing one drain must both
        # survive (the drain reads the whole file then removes it).
        fh = open(inbox / f"{session_id}.md", "a", encoding="utf-8")
        try:
            fh.write(text.strip() + "\n\n---\n\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fh.close()
    except OSError as exc:
        _audit_inject_result(config, session_id, f"failed: {exc}")
        return 500, {"ok": False, "error": f"inbox write failed: {exc}"}
    _audit_inject_result(config, session_id, "ok")

    # The panel's own injects show in the memories feed too.
    try:
        with (config.claude_dir / "episodic" / "memory_activity.jsonl").open(
                "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": forget.now_iso(), "session_id": session_id,
                "source": "inject", "memories": [],
                "query_preview": text[:160],
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass

    body = {"ok": True}
    if persist:
        # Best-effort: the inject already succeeded; a persist failure is a
        # warning, not an error. Same CLI path the MCP ingest_text tool rides.
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "memsom.interface.cli",
                 "ingest-text", text, "--channel", "user",
                 "--ref", f"panel-inject/{session_id}"],
                capture_output=True, text=True, timeout=30,
            )
            body["persisted"] = proc.returncode == 0
            if proc.returncode != 0:
                body["warning"] = (proc.stderr or proc.stdout or "ingest failed").strip()[:300]
        except Exception as exc:  # noqa: BLE001 — best-effort by contract
            body["persisted"] = False
            body["warning"] = str(exc)[:300]
    return 200, body


def _read_agent_activity(claude_dir: Path, limit: int = 100) -> dict:
    """Agent lifecycle events from the SubagentStart/SubagentStop hook journal
    (agent_journal.py), paired into running/done states, newest first.

    Pairing prefers agent_id (both events carry it — verified live); the
    (session_id, agent_type) oldest-open-Start heuristic is the fallback for
    journal lines from harness versions that omit it."""
    events = _tail_jsonl(claude_dir / "episodic" / "agent_activity.jsonl")

    open_starts: list = []
    finished: list = []
    for ev in events:
        kind = ev.get("event")
        if kind == "SubagentStart":
            open_starts.append(
                {
                    "ts": ev.get("ts"),
                    "session_id": ev.get("session_id"),
                    "agent_type": ev.get("agent_type"),
                    "agent_id": ev.get("agent_id"),
                    "prompt_preview": ev.get("prompt_preview"),
                    "state": "running",
                }
            )
        elif kind == "SubagentStop":
            match_idx = None
            stop_id = ev.get("agent_id")
            if stop_id:
                for idx, item in enumerate(open_starts):
                    if item.get("agent_id") == stop_id:
                        match_idx = idx
                        break
            if match_idx is None:
                for idx, item in enumerate(open_starts):
                    if (
                        item["session_id"] == ev.get("session_id")
                        and item["agent_type"] == ev.get("agent_type")
                    ):
                        match_idx = idx
                        break
            if match_idx is None:
                # Last resort for degraded payloads (missing/blank agent_id or
                # agent_type) — but ONLY when unambiguous: with several agents
                # open in the session, guessing would steal a still-running
                # agent's Start and corrupt two records instead of zero.
                session_open = [idx for idx, item in enumerate(open_starts)
                                if item["session_id"] == ev.get("session_id")]
                if len(session_open) == 1:
                    match_idx = session_open[0]
            if match_idx is not None:
                item = dict(open_starts[match_idx], state="done",
                            ended_ts=ev.get("ts"),
                            agent_id=stop_id or open_starts[match_idx].get("agent_id"))
                item["duration_s"] = _agent_duration(item.get("ts"), ev.get("ts"))
                if ev.get("tokens") is not None:
                    item["tokens"] = ev.get("tokens")
                del open_starts[match_idx]
                finished.append(item)
            else:
                finished.append(
                    {
                        "ts": ev.get("ts"),
                        "ended_ts": ev.get("ts"),
                        "session_id": ev.get("session_id"),
                        "agent_type": ev.get("agent_type"),
                        "agent_id": ev.get("agent_id"),
                        "prompt_preview": None,
                        "state": "done",
                    }
                )
    # Unmatched Starts are presumed still running; newest activity first.
    items = list(reversed(open_starts)) + list(reversed(finished))
    return {"events": items[:limit], "active": len(open_starts)}


def _agent_duration(start_iso, end_iso):
    """Seconds between two ISO-Z timestamps, or None if either is unparseable."""
    try:
        from datetime import datetime
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        return int((datetime.strptime(end_iso, fmt)
                    - datetime.strptime(start_iso, fmt)).total_seconds())
    except (TypeError, ValueError):
        return None


def _read_session_status(claude_dir: Path) -> dict:
    """Live session status for the DECK panel, projected from the JSON payload
    Claude Code pipes to the statusline (tee'd to episodic/statusline_state.json
    by statusline.sh). Reflects the most-recently-rendered active session — i.e.
    the chat you're currently in. Tolerant of missing fields / a partial write."""
    p = Path(claude_dir) / "episodic" / "statusline_state.json"
    if not p.is_file():
        return {"available": False}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return {"available": False}
    cw = raw.get("context_window") or {}
    rl = raw.get("rate_limits") or {}
    fh = rl.get("five_hour") or {}
    sd = rl.get("seven_day") or {}
    model = raw.get("model") or {}
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = None
    return {
        "available": True,
        "session_id": raw.get("session_id"),
        "session_name": raw.get("session_name"),
        "model": model.get("display_name") or model.get("id"),
        "context_used_pct": cw.get("used_percentage"),
        "context_window_size": cw.get("context_window_size"),
        "context_input_tokens": cw.get("total_input_tokens"),
        "exceeds_200k": raw.get("exceeds_200k_tokens"),
        "five_hour": {"used_pct": fh.get("used_percentage"),
                      "resets_at": fh.get("resets_at")},
        "seven_day": {"used_pct": sd.get("used_percentage"),
                      "resets_at": sd.get("resets_at")},
        "effort": (raw.get("effort") or {}).get("level"),
        "thinking": (raw.get("thinking") or {}).get("enabled"),
        "fast_mode": raw.get("fast_mode"),
        "output_style": (raw.get("output_style") or {}).get("name"),
        "cost_usd": (raw.get("cost") or {}).get("total_cost_usd"),
        "mtime": mtime,
    }


# ---------------------------------------------------------------------------
# Config + server construction
# ---------------------------------------------------------------------------

@dataclass
class PanelConfig:
    host: str
    port: int
    profile_path: Path
    profile: dict
    audit_log_path: Path
    providers: dict
    memory_cache: MemoryCache
    telemetry: "telemetry.SystemTelemetry"
    claude_dir: Path
    workflows_cache: MemoryCache
    agents_cache: MemoryCache
    memories_cache: MemoryCache
    # local-AI control plane (defaulted so existing constructors don't break)
    registry: dict = None
    session_runner: "SessionRunner" = None
    # voice tab: its own durable-session dir (beside inference/) for streamed chat
    voice_runner: "SessionRunner" = None
    # agent orchestration layer (same defaulting rationale)
    graph_store: "GraphStore" = None
    agent_runner: "AgentRunner" = None
    kernel_store: "KernelStore" = None
    kernel_runner: "KernelRunner" = None
    # scheduler is constructed here but only started by the serve loop (_cmd_panel)
    # so tests that build_config/build_server never spawn the background thread.
    scheduler: "Scheduler" = None


def build_config(profile_path, *, host: str = "127.0.0.1", port: int = 7788,
                  run=None, urlopen=None, memory_builder=None, memory_now=None) -> PanelConfig:
    enforce_bind_policy(host)
    # Default to the no-console-window runner so nvidia-smi / schtasks children
    # don't flash a console when the panel runs detached (app-spawned). Tests
    # that inject their own `run` keep it.
    run = run or run_no_window
    profile = load_profile(profile_path)
    providers = {
        "canonical-params": CanonicalParamsProvider(),
        "json-file": JsonFileProvider(),
        "set-line": SetLineProvider(),
        "schtasks": SchtasksProvider(run=run),
    }
    memory_cache = MemoryCache(builder=memory_builder or dashboard.build_telemetry, now=memory_now)
    sys_telemetry = telemetry.SystemTelemetry(profile.get("telemetry") or {},
                                              run=run, urlopen=urlopen)
    claude_dir = Path(profile.get("claude_dir") or default_claude_dir())
    workflows_cache = MemoryCache(
        builder=lambda: _scan_workflow_runs(claude_dir / "projects"), ttl=3.0)
    agents_cache = MemoryCache(
        builder=lambda: _read_agent_activity(claude_dir), ttl=3.0)
    memories_cache = MemoryCache(
        builder=lambda: _read_memory_activity(claude_dir), ttl=3.0)
    # local-AI control plane: adapters + durable inference sessions live next to
    # the audit log (episodic/inference/). servers.json tracks detached model
    # servers' PIDs so start/stop survives a panel restart.
    inference_dir = Path(profile["audit_log"]).parent / "inference"
    registry = build_registry(profile, servers_path=inference_dir / "servers.json")
    session_runner = SessionRunner(inference_dir)
    # voice chat streams into its own sessions dir (sibling of inference/) so the
    # voice tab's transcripts stay separate from the INFERENCE tab's history.
    voice_runner = SessionRunner(Path(profile["audit_log"]).parent / "voice")
    # agent orchestration: saved canvases + durable tool-loop runs, sibling of
    # inference/. Boot reconcile stamps runs orphaned by a server restart.
    agents_dir = Path(profile["audit_log"]).parent / "agents"
    graph_store = GraphStore(agents_dir / "graphs")
    agent_runner = AgentRunner(agents_dir / "runs", registry,
                               agents_dir / "audit.jsonl")
    agent_runner.reconcile_on_boot()
    # in-process schedule tick — fires enabled TRIGGER schedules through the same
    # runner. Constructed always, started only by the serve loop.
    scheduler = Scheduler(graph_store, agent_runner, registry,
                          agents_dir / "audit.jsonl",
                          agents_dir / "scheduler_state.json")
    # persistent CLI kernels: durable session pointers + per-prompt CLI runs
    # streamed into the SAME sessions dir (krn- prefixed) so the inference
    # read path serves them unchanged. Boot reconcile stamps busy strays.
    kernel_store = KernelStore(Path(profile["audit_log"]).parent / "kernels")
    kernel_runner = KernelRunner(kernel_store, inference_dir, claude_dir)
    kernel_runner.reconcile_on_boot()
    return PanelConfig(
        host=host, port=port, profile_path=Path(profile_path), profile=profile,
        audit_log_path=Path(profile["audit_log"]), providers=providers,
        memory_cache=memory_cache, telemetry=sys_telemetry,
        claude_dir=claude_dir, workflows_cache=workflows_cache,
        agents_cache=agents_cache, memories_cache=memories_cache,
        registry=registry, session_runner=session_runner,
        voice_runner=voice_runner,
        graph_store=graph_store, agent_runner=agent_runner,
        scheduler=scheduler,
        kernel_store=kernel_store, kernel_runner=kernel_runner,
    )


# ---------------------------------------------------------------------------
# The page: CSS + JS as string constants, assembled server-side, hashed once.
# ---------------------------------------------------------------------------

PAGE_CSS = r"""
:root{--bg:#0c0e14;--panel:#141823;--panel2:#1b2030;--line:#262d40;--ink:#e7ecf5;
  --dim:#8a93a8;--good:#3ddc97;--bad:#ff5c72;--warn:#f0883e;--accent:#6aa9ff;--pin:#c792ea;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  padding:24px clamp(16px,4vw,48px) 64px;}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
  border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:22px;}
h1{margin:0;font-size:20px;letter-spacing:.4px;}
h1 .tag{color:var(--accent);}
nav{display:flex;gap:6px;margin-left:auto;}
nav button{background:var(--panel2);border:1px solid var(--line);color:var(--dim);
  border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer;}
nav button[aria-current="true"]{color:var(--ink);border-color:var(--accent);}
section{display:none;}
section[data-active="true"]{display:block;}
h2{font-size:14px;margin:0 0 12px;color:var(--ink);}
.grid{display:grid;gap:14px;grid-template-columns:repeat(12,1fr);}
.box{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.span4{grid-column:span 4;} .span6{grid-column:span 6;}
.span8{grid-column:span 8;} .span12{grid-column:span 12;}
@media(max-width:900px){.span4,.span6,.span8{grid-column:span 12;}}
.bar{height:10px;border-radius:6px;background:var(--panel2);overflow:hidden;border:1px solid var(--line);}
.bar>i{display:block;height:100%;background:var(--accent);}
.bar.warn>i{background:var(--warn);} .bar.bad>i{background:var(--bad);}
table{width:100%;border-collapse:collapse;font-size:12px;}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--panel2);}
th{color:var(--dim);font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.5px;}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}
.dot.ok{background:var(--good);} .dot.fail{background:var(--bad);}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:10px;font-weight:600;}
.pill.tier1{background:rgba(106,169,255,.14);color:var(--accent);}
.pill.tier2{background:rgba(240,136,62,.16);color:var(--warn);}
.pill.tier3{background:rgba(138,147,168,.16);color:var(--dim);}
.pill.dirty{background:rgba(255,92,114,.16);color:var(--bad);}
.knob-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:8px 0;border-bottom:1px solid var(--panel2);}
.knob-row .label{flex:1 1 220px;min-width:180px;}
.knob-row input{background:var(--panel2);border:1px solid var(--line);color:var(--ink);
  border-radius:6px;padding:4px 8px;font:inherit;width:130px;}
.knob-row button{background:var(--panel2);border:1px solid var(--line);color:var(--dim);
  border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer;}
.knob-row .err{color:var(--bad);font-size:11px;flex-basis:100%;}
.knob-row .evidence{color:var(--dim);font-size:11px;flex-basis:100%;}
.knob-row.contract{opacity:.55;}
code.cmd{display:block;background:var(--panel2);border:1px solid var(--line);border-radius:6px;
  padding:6px 8px;font-size:11px;margin-top:4px;user-select:all;white-space:pre-wrap;word-break:break-all;}
.banner{background:rgba(255,92,114,.12);border:1px solid var(--bad);color:var(--bad);
  border-radius:10px;padding:10px 14px;margin-bottom:16px;}
.stale{color:var(--dim);font-size:11px;}
"""

PAGE_SCRIPT = r"""
(function () {
  'use strict';

  function el(tag, attrs) {
    var node = document.createElement(tag);
    attrs = attrs || {};
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (k === 'class') node.className = v;
      else node.setAttribute(k, v);
    });
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c === null || c === undefined) continue;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return node;
  }

  function fmtBytes(n) {
    if (n === null || n === undefined) return '?';
    var units = ['B', 'KB', 'MB', 'GB', 'TB'];
    var i = 0, v = n;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return v.toFixed(v >= 10 || i === 0 ? 0 : 1) + ' ' + units[i];
  }

  function fmtPct(n) {
    if (n === null || n === undefined) return '?';
    return Number(n).toFixed(1) + '%';
  }

  function getJSON(url) {
    return fetch(url, { headers: { Accept: 'application/json' } }).then(function (r) {
      return r.json().then(function (body) { return { ok: r.ok, status: r.status, body: body }; });
    });
  }

  function postJSON(url, payload) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(function (r) {
      return r.json().then(function (body) { return { ok: r.ok, status: r.status, body: body }; });
    });
  }

  function bar(pct, cls) {
    var i = el('i', { style: 'width:' + Math.max(0, Math.min(100, pct || 0)) + '%' });
    return el('div', { class: 'bar' + (cls ? ' ' + cls : '') }, i);
  }

  function initTabs() {
    var buttons = document.querySelectorAll('nav button[data-tab]');
    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = btn.getAttribute('data-tab');
        document.querySelectorAll('section[data-tab]').forEach(function (sec) {
          sec.setAttribute('data-active', sec.getAttribute('data-tab') === target ? 'true' : 'false');
        });
        buttons.forEach(function (b) { b.setAttribute('aria-current', b === btn ? 'true' : 'false'); });
      });
    });
  }

  function renderSystem(s) {
    var root = document.getElementById('sys-body');
    root.textContent = '';

    var ram = s.ram || {};
    var phys = ram.physical || {};
    var commit = ram.commit || {};
    var ramBox = el('div', { class: 'box span6' },
      el('h2', {}, 'RAM'),
      el('div', {}, 'physical: ' + (phys.error || fmtPct(phys.used_pct))),
      bar(phys.used_pct, phys.used_pct > 90 ? 'bad' : phys.used_pct > 75 ? 'warn' : ''),
      el('div', { style: 'margin-top:10px' }, 'commit (headline): ' + (commit.error || fmtPct(commit.used_pct))),
      bar(commit.used_pct, commit.used_pct > 90 ? 'bad' : commit.used_pct > 75 ? 'warn' : '')
    );

    var gpu = s.gpu || {};
    var gpuBox = el('div', { class: 'box span6' }, el('h2', {}, 'GPU'));
    if (gpu.available) {
      gpuBox.appendChild(el('div', {}, gpu.name || 'GPU'));
      var pct = gpu.vram_total_mb ? (100 * gpu.vram_used_mb / gpu.vram_total_mb) : 0;
      gpuBox.appendChild(bar(pct, pct > 90 ? 'bad' : pct > 75 ? 'warn' : ''));
      gpuBox.appendChild(el('div', { class: 'stale' }, (gpu.vram_used_mb || 0) + ' / ' + (gpu.vram_total_mb || 0) + ' MB'));
      (gpu.procs || []).forEach(function (p) {
        gpuBox.appendChild(el('div', { class: 'stale' }, p.name + ' — ' + p.used_mb + ' MB'));
      });
    } else {
      gpuBox.appendChild(el('div', { class: 'stale' }, 'no GPU visible'));
    }

    var probesBox = el('div', { class: 'box span6' }, el('h2', {}, 'Service probes'));
    (s.probes || []).forEach(function (p) {
      probesBox.appendChild(el('div', {},
        el('span', { class: 'dot ' + (p.ok ? 'ok' : 'fail') }),
        (p.name || (p.host + ':' + p.port)) + ' — ' + (p.ok ? p.ms + ' ms' : 'unreachable')
      ));
    });

    var sync = s.syncthing || {};
    var syncBox = el('div', { class: 'box span6' },
      el('h2', {}, 'Syncthing'),
      el('div', {}, el('span', { class: 'dot ' + (sync.reachable ? 'ok' : 'fail') }),
        sync.reachable ? 'reachable' : ('unreachable' + (sync.error ? ' (' + sync.error + ')' : '')))
    );

    var procBox = el('div', { class: 'box span12' }, el('h2', {}, 'Top processes'));
    var tabs = el('div', {});
    var wsBtn = el('button', { type: 'button' }, 'working set');
    var commitBtn = el('button', { type: 'button' }, 'commit');
    tabs.appendChild(wsBtn); tabs.appendChild(commitBtn);
    procBox.appendChild(tabs);
    var tbody = el('tbody', {});
    var tbl = el('table', {},
      el('thead', {}, el('tr', {}, el('th', {}, 'pid'), el('th', {}, 'name'), el('th', {}, 'mem'))),
      tbody);
    procBox.appendChild(tbl);
    var top = s.top_procs || {};
    function fillProcs(rows, key) {
      tbody.textContent = '';
      (rows || []).forEach(function (r) {
        tbody.appendChild(el('tr', {},
          el('td', {}, String(r.pid)), el('td', {}, r.name || ''), el('td', {}, fmtBytes(r[key]))));
      });
    }
    wsBtn.addEventListener('click', function () { fillProcs(top.by_working_set, 'rss'); });
    commitBtn.addEventListener('click', function () { fillProcs(top.by_commit, 'vms'); });
    fillProcs(top.by_working_set, 'rss');

    var diskBox = el('div', { class: 'box span12' }, el('h2', {}, 'Disk'));
    (s.disks || []).forEach(function (d) {
      var delta = d.delta_bytes;
      diskBox.appendChild(el('div', {}, d.path + ' — ' + fmtBytes(d.bytes) +
        (delta === null || delta === undefined ? '' : ' (' + (delta >= 0 ? '+' : '') + fmtBytes(delta) + ')')));
    });

    [ramBox, gpuBox, probesBox, syncBox, procBox, diskBox].forEach(function (b) { root.appendChild(b); });
    var ts = document.getElementById('sys-ts');
    if (ts) ts.textContent = s.ts || '';
  }

  function refreshSystem() {
    getJSON('/api/system').then(function (res) { if (res.ok) renderSystem(res.body); })['catch'](function () {});
  }

  function renderMemory(m) {
    var root = document.getElementById('mem-body');
    root.textContent = '';

    var totals = m.totals || {};
    var totalsBox = el('div', { class: 'box span12' }, el('h2', {}, 'Totals'),
      el('div', {}, 'total ' + totals.total + '  ·  hot ' + totals.hot +
        '  ·  cold ' + totals.cold + '  ·  pinned ' + totals.pinned));

    var budget = m.budget;
    var budgetBox = el('div', { class: 'box span12' }, el('h2', {}, 'MEMORY.md budget'));
    if (budget) {
      budgetBox.appendChild(bar(budget.pct, budget.pct > 90 ? 'bad' : budget.pct > 75 ? 'warn' : ''));
      budgetBox.appendChild(el('div', { class: 'stale' },
        fmtBytes(budget.bytes) + ' / ' + fmtBytes(budget.cap) + ' (' + budget.pct + '%)'));
    } else {
      budgetBox.appendChild(el('div', { class: 'stale' }, 'no budget data'));
    }

    var hist = m.hist || { labels: [], data: [] };
    var histBox = el('div', { class: 'box span12' }, el('h2', {}, 'RS histogram'));
    var maxCount = Math.max(1, (hist.data || []).reduce(function (a, b) { return Math.max(a, b); }, 0));
    (hist.labels || []).forEach(function (label, i) {
      histBox.appendChild(el('div', { style: 'display:flex;align-items:center;gap:8px;margin:2px 0' },
        el('span', { class: 'stale', style: 'width:70px;display:inline-block' }, label),
        bar(100 * hist.data[i] / maxCount),
        el('span', { class: 'stale' }, String(hist.data[i]))));
    });

    var riskTbody = el('tbody', {});
    var riskBox = el('div', { class: 'box span12' }, el('h2', {}, 'Demote-risk watchlist'),
      el('table', {}, el('thead', {}, el('tr', {},
        el('th', {}, 'memory'), el('th', {}, 'RS'), el('th', {}, 'uses'), el('th', {}, 'idle'))), riskTbody));
    (m.stale || []).forEach(function (r) {
      riskTbody.appendChild(el('tr', {}, el('td', {}, r.stem), el('td', {}, String(r.weight)),
        el('td', {}, String(r.count)), el('td', {}, r.age_days === null || r.age_days === undefined ? '—' : r.age_days + 'd')));
    });

    var th = m.thresholds || {};
    var thBox = el('div', { class: 'box span12' }, el('h2', {}, 'Live thresholds'),
      el('div', {}, 'demote_below ' + th.demote_below + '  ·  promote_at ' + th.promote_at));

    [totalsBox, budgetBox, histBox, riskBox, thBox].forEach(function (b) { root.appendChild(b); });
    var ts = document.getElementById('mem-ts');
    if (ts) ts.textContent = (m.generated || '') + '  (cached ' + (m.cached_at || '') + ')';
  }

  function refreshMemory(force) {
    getJSON('/api/memory' + (force ? '?refresh=1' : '')).then(function (res) {
      if (res.ok) renderMemory(res.body);
    })['catch'](function () {});
  }

  function coerceInput(k, raw) {
    if (k.type === 'int') return parseInt(raw, 10);
    if (k.type === 'float') return parseFloat(raw);
    if (k.type === 'bool') return raw === 'true' || raw === '1';
    if (k.type === 'list-of-str') return raw.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
    if (k.type === 'weights') {
      // structured value edited as JSON; a parse failure is sent as-is and the
      // server's fail-closed validator returns the reason inline.
      try { return JSON.parse(raw); } catch (e) { return raw; }
    }
    return raw;
  }

  function displayValue(k, v) {
    if (v === null || v === undefined) return '';
    return typeof v === 'object' ? JSON.stringify(v) : String(v);
  }

  function knobRow(k) {
    var row = el('div', { class: 'knob-row' + (k.tier === 3 ? ' contract' : '') });
    row.appendChild(el('span', { class: 'pill tier' + k.tier }, 'T' + k.tier));
    row.appendChild(el('span', { class: 'label' }, k.label || k.id));

    if (k.tier === 3 || !k.provider) {
      row.appendChild(el('span', { class: 'stale' }, displayValue(k, k.current !== null && k.current !== undefined ? k.current : k.default)));
      if (k.note) row.appendChild(el('span', { class: 'evidence' }, k.note));
      return row;
    }

    var input = el('input', { type: 'text', value: displayValue(k, k.current) });
    row.appendChild(input);
    if (k.dirty) row.appendChild(el('span', { class: 'pill dirty' }, 'dirty'));
    row.appendChild(el('span', { class: 'stale' }, 'default: ' + displayValue(k, k.default)));

    var errSpan = el('span', { class: 'err' });
    var saveBtn = el('button', { type: 'button' }, 'set');
    var resetBtn = el('button', { type: 'button' }, 'reset');

    saveBtn.addEventListener('click', function () {
      errSpan.textContent = '';
      postJSON('/api/knobs', { id: k.id, value: coerceInput(k, input.value) }).then(function (res) {
        if (!res.body.ok) errSpan.textContent = res.body.error || ('HTTP ' + res.status);
        loadKnobs();
      });
    });
    resetBtn.addEventListener('click', function () {
      errSpan.textContent = '';
      postJSON('/api/knobs', { id: k.id, reset: true }).then(function (res) {
        if (!res.body.ok) errSpan.textContent = res.body.error || ('HTTP ' + res.status);
        loadKnobs();
      });
    });
    row.appendChild(saveBtn);
    row.appendChild(resetBtn);

    if (k.requires) row.appendChild(el('span', { class: 'pill tier2' }, 'requires: ' + k.requires));
    if (k.evidence) row.appendChild(el('span', { class: 'evidence' }, 'evidence: ' + JSON.stringify(k.evidence)));
    if (k.note) {
      row.appendChild(el('span', { class: 'evidence' }, k.note));
      if (String(k.note).toLowerCase().indexOf('restart') !== -1) row.appendChild(el('code', { class: 'cmd' }, k.note));
    }
    if (k.task) row.appendChild(el('span', { class: 'evidence' },
      'trigger: ' + (k.task.trigger || k.task.error || '?') + '  enabled: ' + k.task.enabled));
    row.appendChild(errSpan);
    return row;
  }

  function renderKnobs(data) {
    var root = document.getElementById('knobs-body');
    root.textContent = '';

    if ((data.crash_markers || []).length) {
      root.appendChild(el('div', { class: 'banner' },
        'crash marker(s) — a write started and never confirmed, verify by hand: ' +
        data.crash_markers.map(function (c) { return c.knob; }).join(', ')));
    }

    [1, 2, 3].forEach(function (tier) {
      var rows = (data.knobs || []).filter(function (k) { return k.tier === tier; });
      if (tier === 3) rows = rows.concat((data.contracts || []).map(function (c) {
        return { id: c.id, tier: 3, label: c.label, current: c.value, default: c.value, note: c.note };
      }));
      if (!rows.length) return;
      var box = el('div', { class: 'box span12' }, el('h2', {}, 'Tier ' + tier));
      rows.forEach(function (k) { box.appendChild(knobRow(k)); });
      root.appendChild(box);
    });

    if ((data.tasks || []).length) {
      var taskBox = el('div', { class: 'box span12' }, el('h2', {}, 'Scheduled tasks'));
      data.tasks.forEach(function (t) {
        var status = t.status || {};
        taskBox.appendChild(el('div', {}, t.name + ' — ' +
          (status.error || JSON.stringify(status)) + (t.writable === false ? '  [read-only]' : '')));
      });
      root.appendChild(taskBox);
    }
  }

  function loadKnobs() {
    getJSON('/api/knobs').then(function (res) { if (res.ok) renderKnobs(res.body); })['catch'](function () {});
  }

  document.addEventListener('DOMContentLoaded', function () {
    initTabs();
    refreshSystem();
    setInterval(refreshSystem, 5000);
    refreshMemory(false);
    setInterval(function () { refreshMemory(false); }, 45000);
    var memBtn = document.getElementById('mem-refresh-btn');
    if (memBtn) memBtn.addEventListener('click', function () { refreshMemory(true); });
    loadKnobs();
  });
})();
"""

PAGE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>memsom panel</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>memsom<span class="tag">&middot;</span>panel</h1>
  <div class="stale">v{version}</div>
  <nav>
    <button type="button" data-tab="system" aria-current="true">system</button>
    <button type="button" data-tab="memory">memory</button>
    <button type="button" data-tab="knobs">knobs</button>
  </nav>
</header>
<section data-tab="system" data-active="true">
  <div class="stale" id="sys-ts"></div>
  <div class="grid" id="sys-body"></div>
</section>
<section data-tab="memory">
  <div style="display:flex;align-items:center;gap:10px">
    <div class="stale" id="mem-ts"></div>
    <button type="button" id="mem-refresh-btn">refresh</button>
  </div>
  <div class="grid" id="mem-body"></div>
</section>
<section data-tab="knobs">
  <div class="grid" id="knobs-body"></div>
</section>
<script>{script}</script>
</body>
</html>
"""

PAGE_HTML = PAGE_HTML_TEMPLATE.format(
    css=PAGE_CSS, version=html.escape(_MEMSOM_VERSION), script=PAGE_SCRIPT,
)
PAGE_HTML_BYTES = PAGE_HTML.encode("utf-8")

_SCRIPT_HASH_B64 = base64.b64encode(hashlib.sha256(PAGE_SCRIPT.encode("utf-8")).digest()).decode("ascii")
_CSP = (
    "default-src 'none'; script-src 'sha256-" + _SCRIPT_HASH_B64 + "'; "
    "style-src 'unsafe-inline'; connect-src 'self'; img-src data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
    "Content-Security-Policy": _CSP,
}

_MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MiB — knob JSON payloads are tiny
# Voice STT ships a base64'd compressed-audio blob, which blows past 1 MiB for
# anything longer than a short utterance. base64 inflates ~33%, so 8 MiB of body
# ≈ 6 MiB of opus ≈ minutes of speech — a comfortable ceiling for an utterance.
_MAX_VOICE_BODY_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def make_handler(config: PanelConfig):
    """Build a BaseHTTPRequestHandler bound to *config* (closure, no globals)."""

    class PanelHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"memsom-panel/{_MEMSOM_VERSION}"

        def log_message(self, fmt, *args):  # noqa: N802 (stdlib signature)
            print(f"[memsom-panel] {self.address_string()} - {fmt % args}", file=sys.stderr)

        # ---- response helpers ----

        def _send(self, status, body, ctype="application/json", extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in _SECURITY_HEADERS.items():
                self.send_header(k, v)
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _send_json(self, status, obj, extra_headers=None):
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       extra_headers=extra_headers)

        def _reject(self, status, obj):
            # Pre-body-consumption rejection: close the connection rather than
            # risk a desynced keep-alive stream.
            self.close_connection = True
            self._send_json(status, obj)

        def _split_path(self):
            if "?" in self.path:
                p, qs = self.path.split("?", 1)
            else:
                p, qs = self.path, ""
            return p, urllib.parse.parse_qs(qs)

        # ---- routing ----

        def do_GET(self):  # noqa: N802
            if not _host_header_allowed(self.headers.get("Host"), config.port):
                self._reject(403, {"error": "host not allowed"})
                return
            try:
                self._route_get()
            except Exception:
                print(f"[memsom-panel] GET error:\n{traceback.format_exc()}", file=sys.stderr)
                try:
                    self._send_json(500, {"error": "internal server error"})
                except Exception:
                    pass

        def _route_get(self):
            path, query = self._split_path()
            if path == "/":
                self._serve_page()
            elif path == "/health":
                self._serve_health()
            elif path == "/api/memory":
                self._serve_memory(query)
            elif path == "/api/system":
                self._serve_system()
            elif path == "/api/knobs":
                self._send_json(200, build_knobs_payload(config))
            elif path == "/api/activity/workflows":
                self._send_json(200, config.workflows_cache.get())
            elif path == "/api/activity/agents":
                self._send_json(200, config.agents_cache.get())
            elif path == "/api/activity/memories":
                self._send_json(200, config.memories_cache.get())
            elif path == "/api/providers":
                self._send_json(200,
                    provider_handlers.build_providers_payload(config.registry))
            elif path == "/api/providers/vram-estimate":
                st, body = provider_handlers.handle_vram_estimate(
                    config.registry, _q1(query, "provider"), _q1(query, "model"),
                    _qint(query, "ctx", 0), _q1(query, "kv") or "fp16")
                self._send_json(st, body)
            elif path == "/api/inference/sessions":
                st, body = provider_handlers.handle_inference_sessions(
                    config.session_runner)
                self._send_json(st, body)
            elif path == "/api/inference":
                st, body = provider_handlers.handle_inference_read(
                    config.session_runner, _q1(query, "session_id"),
                    _qint(query, "cursor", 0))
                self._send_json(st, body)
            elif path == "/api/voice/chat":
                st, body = voice_handlers.handle_voice_chat_read(
                    config.voice_runner, _q1(query, "session_id"),
                    _qint(query, "cursor", 0))
                self._send_json(st, body)
            elif path == "/api/saveall/status":
                self._send_json(200, saveall_runner.status(config.claude_dir))
            elif path == "/api/session-status":
                self._send_json(200, _read_session_status(config.claude_dir))
            elif path == "/api/agents/graphs":
                st, body = agent_handlers.handle_graphs_list(config.graph_store)
                self._send_json(st, body)
            elif path == "/api/agents/graph":
                st, body = agent_handlers.handle_graph_get(
                    config.graph_store, _q1(query, "id"))
                self._send_json(st, body)
            elif path == "/api/agents/tools":
                st, body = agent_handlers.handle_tool_catalog()
                self._send_json(st, body)
            elif path == "/api/agents/run":
                st, body = agent_handlers.handle_run_read(
                    config.agent_runner, _q1(query, "run_id"),
                    _qint(query, "cursor", 0))
                self._send_json(st, body)
            elif path == "/api/agents/runs":
                st, body = agent_handlers.handle_runs_list(config.agent_runner)
                self._send_json(st, body)
            elif path == "/api/kernels":
                include_archived = _q1(query, "all") in ("1", "true")
                st, body = kernel_handlers.handle_kernels_list(
                    config.kernel_store, config.kernel_runner,
                    include_archived=include_archived)
                self._send_json(st, body)
            elif path == "/api/agents/scheduler":
                st, body = agent_handlers.handle_scheduler_status(config.scheduler)
                self._send_json(st, body)
            else:
                self._send_json(404, {"error": "not found"})

        def _serve_page(self):
            self._send(200, PAGE_HTML_BYTES, ctype="text/html; charset=utf-8")

        def _serve_health(self):
            self._send_json(200, {"app": "memsom-panel", "version": _MEMSOM_VERSION, "ok": True})

        def _serve_memory(self, query):
            refresh = any(v in ("1", "true") for v in query.get("refresh", []))
            try:
                data = config.memory_cache.get(refresh=refresh)
            except SystemExit as exc:
                self._send_json(503, {"error": f"memory telemetry unavailable: {exc}"})
                return
            self._send_json(200, data)

        def _serve_system(self):
            self._send_json(200, config.telemetry.sample())

        def do_POST(self):  # noqa: N802
            if not _host_header_allowed(self.headers.get("Host"), config.port):
                self._reject(403, {"error": "host not allowed"})
                return
            try:
                self._route_post()
            except Exception:
                print(f"[memsom-panel] POST error:\n{traceback.format_exc()}", file=sys.stderr)
                try:
                    self._send_json(500, {"error": "internal server error"})
                except Exception:
                    pass

        def _read_post_json(self, max_bytes=_MAX_BODY_BYTES):
            """Shared POST gate: Content-Type check, Origin allowlist, bounded
            body read, JSON parse. Sends the error response itself and returns
            None on any failure; the payload dict otherwise. *max_bytes* raises
            the body ceiling for one route (voice STT ships base64 audio)."""
            ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                self._reject(403, {"ok": False, "error": "Content-Type must be application/json"})
                return None

            origin = self.headers.get("Origin")
            if origin is not None and origin not in _allowed_origins(config):
                self._reject(403, {"ok": False, "error": "origin not allowed"})
                return None

            raw_len = self.headers.get("Content-Length")
            try:
                length = int(raw_len) if raw_len is not None else 0
            except ValueError:
                self._reject(400, {"ok": False, "error": "bad Content-Length"})
                return None
            if length < 0 or length > max_bytes:
                self._reject(413, {"ok": False, "error": "request body too large"})
                return None
            body = self.rfile.read(length) if length else b""

            try:
                return json.loads(body.decode("utf-8")) if body else {}
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": f"invalid JSON body: {exc}"})
                return None

        def _route_post(self):
            path, _query = self._split_path()
            if path == "/api/knobs":
                payload = self._read_post_json()
                if payload is None:
                    return
                status, result = handle_knob_write(config, payload)
                self._send_json(status, result)
            elif path == "/api/inject":
                payload = self._read_post_json()
                if payload is None:
                    return
                status, result = handle_inject(config, payload)
                self._send_json(status, result)
            elif path == "/api/providers/action":
                payload = self._read_post_json()
                if payload is None:
                    return
                status, result = provider_handlers.handle_provider_action(
                    config.registry, config.audit_log_path,
                    payload.get("action"), payload)
                self._send_json(status, result)
            elif path == "/api/inference/start":
                payload = self._read_post_json()
                if payload is None:
                    return
                status, result = provider_handlers.handle_inference_start(
                    config.registry, config.session_runner,
                    config.audit_log_path, payload)
                self._send_json(status, result)
            elif path == "/api/voice/stt":
                # STT ships base64 audio — raise the body cap for this route only.
                payload = self._read_post_json(_MAX_VOICE_BODY_BYTES)
                if payload is None:
                    return
                status, result = voice_handlers.handle_voice_stt(
                    config.registry, config.audit_log_path, payload)
                self._send_json(status, result)
            elif path == "/api/voice/chat":
                payload = self._read_post_json()
                if payload is None:
                    return
                status, result = voice_handlers.handle_voice_chat_start(
                    config.registry, config.voice_runner,
                    config.audit_log_path, payload)
                self._send_json(status, result)
            elif path == "/api/voice/tts":
                payload = self._read_post_json()
                if payload is None:
                    return
                status, result = voice_handlers.handle_voice_tts(
                    config.registry, config.audit_log_path, payload)
                self._send_json(status, result)
            elif path == "/api/agents/graph":
                payload = self._read_post_json()
                if payload is None:
                    return
                st, body = agent_handlers.handle_graph_save(
                    config.graph_store, config.audit_log_path, payload)
                self._send_json(st, body)
            elif path == "/api/agents/graph/delete":
                payload = self._read_post_json()
                if payload is None:
                    return
                st, body = agent_handlers.handle_graph_delete(
                    config.graph_store, config.audit_log_path, payload)
                self._send_json(st, body)
            elif path == "/api/agents/run":
                payload = self._read_post_json()
                if payload is None:
                    return
                st, body = agent_handlers.handle_run_start(
                    config.graph_store, config.agent_runner, config.registry,
                    config.audit_log_path, payload)
                self._send_json(st, body)
            elif path == "/api/kernels":
                payload = self._read_post_json()
                if payload is None:
                    return
                st, body = kernel_handlers.handle_kernel_create(
                    config.kernel_store, config.profile,
                    config.audit_log_path, payload)
                self._send_json(st, body)
            elif path.startswith("/api/kernels/"):
                payload = self._read_post_json()
                if payload is None:
                    return
                parts = path.split("/")  # ['', 'api', 'kernels', <id>, <verb>]
                if len(parts) != 5:
                    self._send_json(404, {"error": "not found"})
                    return
                kid, verb = parts[3], parts[4]
                if verb == "prompt":
                    st, body = kernel_handlers.handle_kernel_prompt(
                        config.kernel_store, config.kernel_runner,
                        config.audit_log_path, kid, payload)
                elif verb == "kill":
                    st, body = kernel_handlers.handle_kernel_kill(
                        config.kernel_store, config.kernel_runner,
                        config.audit_log_path, kid)
                elif verb == "archive":
                    st, body = kernel_handlers.handle_kernel_archive(
                        config.kernel_store, config.audit_log_path, kid)
                else:
                    st, body = 404, {"error": "not found"}
                self._send_json(st, body)
            elif path == "/api/saveall/start":
                payload = self._read_post_json()
                if payload is None:
                    return
                claude_adapter = config.registry.get("claude")
                cli = getattr(claude_adapter, "cli_path", "claude") if claude_adapter else "claude"
                try:
                    _audit_append(config.audit_log_path, {
                        "ts": forget.now_iso(), "action": "saveall",
                        "result": "pending"})
                    result = saveall_runner.start(config.claude_dir, cli_path=cli)
                    _audit_append(config.audit_log_path, {
                        "ts": forget.now_iso(), "action": "saveall",
                        "session_id": result.get("session_id"), "result": "started"})
                    self._send_json(200, result)
                except Exception as exc:
                    _audit_append(config.audit_log_path, {
                        "ts": forget.now_iso(), "action": "saveall",
                        "result": f"failed: {exc}"})
                    self._send_json(500, {"ok": False, "error": str(exc)})
            else:
                self._send_json(404, {"error": "not found"})

    return PanelHandler


def build_server(config: PanelConfig) -> ThreadingHTTPServer:
    """Create the ThreadingHTTPServer. Enforces the bind policy again (defense
    in depth against a hand-built PanelConfig), resolves `config.port` to the
    OS-assigned runtime port (matters for `port=0` ephemeral binds — the Host/
    Origin allowlists both key off this resolved value), and marks the server
    daemon-threaded so a lingering keep-alive never blocks process exit."""
    enforce_bind_policy(config.host)
    handler = make_handler(config)
    httpd = ThreadingHTTPServer((config.host, config.port), handler)
    httpd.daemon_threads = True
    config.port = httpd.server_address[1]
    httpd.panel_config = config
    return httpd


# ---------------------------------------------------------------------------
# CLI entry point: `memsom panel`
# ---------------------------------------------------------------------------

def _cmd_panel(args):
    try:
        config = build_config(args.profile, host=args.host, port=args.port)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[memsom-panel] {exc}", file=sys.stderr)
        return 2

    try:
        httpd = build_server(config)
    except OSError as bind_exc:
        health_url = f"http://127.0.0.1:{args.port}/health"
        data = {}
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            data = {}
        if data.get("app") == "memsom-panel":
            print(f"panel: already running at http://127.0.0.1:{args.port}/")
            return 0
        # WinError 10013 (WSAEACCES) is NOT a squatter: on Windows it usually
        # means the port falls in a Hyper-V/WSL excluded port range, where any
        # bind fails with access-denied even though nothing is listening.
        if getattr(bind_exc, "winerror", None) == 10013 or bind_exc.errno == errno.EACCES:
            print(f"[memsom-panel] bind to port {args.port} denied by the OS — "
                  f"on Windows this usually means an excluded port range "
                  f"(`netsh interface ipv4 show excludedportrange protocol=tcp`); "
                  f"pick another with --port", file=sys.stderr)
            return 1
        print(f"[memsom-panel] port {args.port} in use by another process; "
              f"retry with --port", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"panel: {url}")
    if not args.no_open:
        webbrowser.open(url)
    if config.scheduler is not None:
        config.scheduler.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if config.scheduler is not None:
            config.scheduler.stop()
        httpd.shutdown()
        httpd.server_close()
    return 0


def register(sub) -> None:
    p = sub.add_parser("panel", help="launch the memsom tuning/telemetry panel server")
    p.add_argument("--profile", required=True, help="path to panel_profile.json")
    p.add_argument("--host", default="127.0.0.1", help="bind host (loopback only)")
    p.add_argument("--port", type=int, default=7788, help="bind port (default 7788)")
    p.add_argument("--no-open", action="store_true", help="don't launch a browser")
    p.set_defaults(func=_cmd_panel)
