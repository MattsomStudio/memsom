"""memsom.interface.remote -- Phase 10 remote auth: devices, capability
classification, and the shared dispatch `serve.py`'s HTTP handler calls.

Matt's seven-point design of record (00-matt-decisions.md, Q1 follow-up,
PLAN.md Sec3.5), implemented:

  1. bind guard                    -- owned by serve.py (socket-level)
  2. per-device credentials        -- remote_devices, one row per device
  3. clearance ceiling, server-side-- effective_clearance() below; MS-02
  4. read authority != write authority -- TOOL_CLASS + capgate, both here
  5. no custom crypto               -- serve.py: bearer token over the mesh
  6. audit every remote call        -- remote_audit, one row per dispatch
  7. fail closed                    -- authenticate() returns None on ANY
                                        auth failure; handle_request() never
                                        falls back to serving local data

TWO GATES IN SERIES (Sec3.5 point 4), not one:
  (a) the capability table -- ENFORCING. A device without `tool` in its
      capabilities set is refused before anything else runs. Static, cannot
      see provenance taint.
  (b) the action gate (capgate.check_capability, per-device session floor)
      -- SHADOW by default (remote.action_gate_mode), same ordering
      PLAN.md Sec3.5 point 4 mandates and the same pattern bridge.hook_mode
      already shipped in Phase 9: the verdict is always computed and always
      logged (capgate's own capability_log), but a deny from THIS gate only
      blocks the call once the knob is flipped to 'enforcing'.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid

import memsom
from memsom.storage import schema as memsom_schema
from memsom.storage import session as memsom_session
from memsom.integrity import capgate as memsom_capgate
from memsom.kernel.lattice import CONF_NAME, parse_conf
from memsom import tuning as memsom_tuning
from memsom.interface import mcp as memsom_mcp

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """CREATE TABLE IF NOT EXISTS remote_devices (
  device_id         TEXT PRIMARY KEY,
  name              TEXT NOT NULL,
  token_hash        TEXT NOT NULL,
  token_salt        TEXT NOT NULL,
  clearance_ceiling INTEGER NOT NULL CHECK (clearance_ceiling BETWEEN 0 AND 3),
  capabilities      TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  revoked_at        TEXT,
  last_seen         TEXT
);
CREATE TABLE IF NOT EXISTS remote_audit (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id      TEXT,
  tool           TEXT NOT NULL,
  arg_digest     TEXT NOT NULL,
  decision       TEXT NOT NULL,
  clearance_used TEXT,
  ts             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_remote_audit_device ON remote_audit(device_id);"""


def migrate(conn) -> None:
    memsom_schema.ensure_table(conn, _SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Tool classification (Sec3.5 point 4a). Mirrors the tool set the local MCP
# transport exposes (memsom.interface.mcp.TOOL_NAMES) -- remote mode is that
# SAME transport shape carried over HTTPS/mesh (Sec3.6), not a new surface.
# `ingest`, `revoke`, `redact`, `federate`, `obsidian_export` are mutate by
# definition (Sec3.5); everything else that writes joins them explicitly.
# ---------------------------------------------------------------------------

MUTATE_TOOLS = frozenset((
    "revoke", "redact", "recompute", "consolidate", "export",
    "ingest_text", "obsidian_sync", "obsidian_export", "verify_stale",
))
READ_TOOLS = frozenset(memsom_mcp.TOOL_NAMES) - MUTATE_TOOLS


def tool_class(tool: str) -> str:
    """'mutate' | 'read' | 'unknown'."""
    if tool in MUTATE_TOOLS:
        return "mutate"
    if tool in READ_TOOLS:
        return "read"
    return "unknown"


# ---------------------------------------------------------------------------
# Device CRUD
# ---------------------------------------------------------------------------

def _hash_token(token: str, salt: str) -> str:
    return hashlib.sha256((salt + token).encode("utf-8")).hexdigest()


def add_device(conn, name, clearance_ceiling, capabilities) -> dict:
    """Create a device row and return {device_id, token, ...}. The token is
    returned ONLY here -- it is never retrievable again (only its hash is
    stored, per-row salted, Sec3.5 point 2)."""
    migrate(conn)
    ceiling = parse_conf(clearance_ceiling)
    caps = sorted(set(capabilities))
    device_id = uuid.uuid4().hex
    token = secrets.token_urlsafe(32)
    salt = secrets.token_hex(16)
    now = memsom.now_iso()
    with conn:
        conn.execute(
            "INSERT INTO remote_devices"
            "(device_id, name, token_hash, token_salt, clearance_ceiling,"
            " capabilities, created_at, revoked_at, last_seen)"
            " VALUES (?,?,?,?,?,?,?,NULL,NULL)",
            (device_id, name, _hash_token(token, salt), salt, ceiling,
             json.dumps(caps), now),
        )
    return {"device_id": device_id, "token": token, "name": name,
            "clearance_ceiling": ceiling, "capabilities": caps}


def list_devices(conn) -> list:
    migrate(conn)
    rows = conn.execute(
        "SELECT device_id, name, clearance_ceiling, capabilities, created_at,"
        " revoked_at, last_seen FROM remote_devices ORDER BY created_at"
    ).fetchall()
    keys = ("device_id", "name", "clearance_ceiling", "capabilities",
            "created_at", "revoked_at", "last_seen")
    out = []
    for r in rows:
        d = dict(zip(keys, r))
        d["capabilities"] = json.loads(d["capabilities"])
        out.append(d)
    return out


def revoke_device(conn, device_id) -> bool:
    """Revocation is ONE row, not a global rotation (Sec3.5 point 2).
    Returns False if the device id is unknown or already revoked."""
    migrate(conn)
    row = conn.execute(
        "SELECT revoked_at FROM remote_devices WHERE device_id=?", (device_id,)
    ).fetchone()
    if row is None or row[0] is not None:
        return False
    with conn:
        conn.execute(
            "UPDATE remote_devices SET revoked_at=? WHERE device_id=?",
            (memsom.now_iso(), device_id),
        )
    return True


def authenticate(conn, token) -> dict | None:
    """Fail closed (Sec3.5 point 7): None for no/malformed/unknown/revoked
    token. A linear scan over devices -- correct at any device count this
    tool is sized for; the token itself carries no device_id so a lookup
    cannot short-circuit on an indexed column without weakening the hash to
    something guessable per-row.
    """
    if not token:
        return None
    migrate(conn)
    rows = conn.execute(
        "SELECT device_id, name, token_hash, token_salt, clearance_ceiling,"
        " capabilities, revoked_at FROM remote_devices"
    ).fetchall()
    for device_id, name, token_hash, salt, ceiling, caps_json, revoked_at in rows:
        if not hmac.compare_digest(_hash_token(token, salt), token_hash):
            continue
        if revoked_at is not None:
            return None  # known but revoked -> still refuse
        with conn:
            conn.execute(
                "UPDATE remote_devices SET last_seen=? WHERE device_id=?",
                (memsom.now_iso(), device_id),
            )
        return {"device_id": device_id, "name": name, "clearance_ceiling": ceiling,
                "capabilities": json.loads(caps_json)}
    return None  # unknown token


# ---------------------------------------------------------------------------
# Clearance clamp (Sec3.5 point 3 / MS-02)
# ---------------------------------------------------------------------------

def effective_clearance_name(device_ceiling: int, requested=None) -> str:
    """min(requested, device.clearance_ceiling) -- the server decides, the
    client's requested value can only NARROW it, never widen it."""
    requested_int = parse_conf(requested) if requested is not None else device_ceiling
    return CONF_NAME[min(requested_int, device_ceiling)].lower()


# ---------------------------------------------------------------------------
# Per-device session cache (Sec3.5 point 4b: "storage/session.py:begin_session
# per device connection"). One session per device per server process; kept in
# a plain dict on the module because a session_id has no meaning across
# process restarts anyway (storage.session's own docstring: process-lifetime
# == session).
# ---------------------------------------------------------------------------

_device_sessions: dict[str, str] = {}


def _device_session(conn, device_id) -> str:
    sid = _device_sessions.get(device_id)
    if sid is None:
        sid = memsom_session.begin_session(conn)
        _device_sessions[device_id] = sid
    return sid


def reset_device_sessions() -> None:
    """Test seam: clear the per-process device->session cache."""
    _device_sessions.clear()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _arg_digest(arguments: dict) -> str:
    blob = json.dumps(arguments or {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _record_audit(conn, device_id, tool, arguments, decision, clearance_used) -> None:
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT INTO remote_audit"
            "(device_id, tool, arg_digest, decision, clearance_used, ts)"
            " VALUES (?,?,?,?,?,?)",
            (device_id, tool, _arg_digest(arguments), decision, clearance_used,
             memsom.now_iso()),
        )


def recent_audit(conn, limit=20) -> list:
    migrate(conn)
    rows = conn.execute(
        "SELECT id, device_id, tool, arg_digest, decision, clearance_used, ts"
        " FROM remote_audit ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    keys = ("id", "device_id", "tool", "arg_digest", "decision", "clearance_used", "ts")
    return [dict(zip(keys, r)) for r in rows]


# ---------------------------------------------------------------------------
# Dispatch -- the ONE place a remote tool call is decided + executed.
# ---------------------------------------------------------------------------

def handle_request(conn, token, tool, arguments) -> dict:
    """Authenticate, gate (both series gates), dispatch, audit. Returns an
    envelope dict: {decision, tool, text, is_error, reason?}.

    Fail-closed by construction: every early-return path is a 'deny' that
    never reaches memsom.interface.mcp._call_tool, so a refused caller gets
    zero rows of store content -- not a degraded/local fallback (Sec3.5
    point 7 explicitly rules that out).
    """
    arguments = dict(arguments or {})
    device = authenticate(conn, token)
    if device is None:
        _record_audit(conn, None, tool, arguments, "deny", None)
        return {"decision": "deny", "tool": tool, "text": "", "is_error": True,
                "reason": "no/unknown/revoked token"}

    device_id = device["device_id"]

    cls = tool_class(tool)
    if cls == "unknown":
        _record_audit(conn, device_id, tool, arguments, "deny", None)
        return {"decision": "deny", "tool": tool, "text": "", "is_error": True,
                "reason": f"unknown tool: {tool!r}"}

    if cls == "mutate" and tool not in device["capabilities"]:
        _record_audit(conn, device_id, tool, arguments, "deny", None)
        return {"decision": "deny", "tool": tool, "text": "", "is_error": True,
                "reason": f"device {device_id} lacks capability {tool!r}"}

    action_gate_verdict = None
    if cls == "mutate":
        # Gate (b): shadow by default -- always computed + logged via
        # capgate's own capability_log, only BLOCKS in 'enforcing' mode.
        sid = _device_session(conn, device_id)
        floor = memsom_session.current_floor(conn, sid)
        required = memsom.RANK["user"]
        action_gate_verdict = memsom_capgate.check_capability(
            conn, sid, floor, tool, required)
        mode = str(memsom_tuning.resolve("remote.action_gate_mode") or "shadow").strip().lower()
        if mode == "enforcing" and action_gate_verdict["decision"] == "deny":
            _record_audit(conn, device_id, tool, arguments, "deny", None)
            return {"decision": "deny", "tool": tool, "text": "", "is_error": True,
                    "reason": f"action gate: {action_gate_verdict['reason']}"}

    clearance = effective_clearance_name(device["clearance_ceiling"],
                                         arguments.get("clearance"))
    if tool in ("ask", "explain", "blame", "neighborhood", "retrieve",
                "obsidian_export"):
        arguments["clearance"] = clearance

    text, is_error = memsom_mcp._call_tool(tool, arguments)
    _record_audit(conn, device_id, tool, arguments, "allow", clearance)
    return {"decision": "allow", "tool": tool, "text": text, "is_error": is_error,
            "clearance_used": clearance,
            "action_gate": action_gate_verdict}


# ---------------------------------------------------------------------------
# CLI: `memsom device add|list|revoke`, `memsom remote-audit-log`
# ---------------------------------------------------------------------------

def _cmd_device_add(args):
    conn = memsom.get_connection()
    try:
        caps = [c.strip() for c in (args.capabilities or "").split(",") if c.strip()]
        d = add_device(conn, args.name, args.clearance, caps)
        print(f"device [{d['device_id']}] {d['name']} clearance={d['clearance_ceiling']}"
              f" capabilities={','.join(d['capabilities']) or '(none)'}")
        print(f"TOKEN (shown once, not recoverable): {d['token']}")
    finally:
        conn.close()


def _cmd_device_list(args):
    conn = memsom.get_connection()
    try:
        devices = list_devices(conn)
        if not devices:
            print("no devices enrolled")
        for d in devices:
            status = "REVOKED" if d["revoked_at"] else "active"
            print(f"[{d['device_id']}] {d['name']:<20} {status:<8}"
                  f" clearance={d['clearance_ceiling']}"
                  f" caps={','.join(d['capabilities']) or '(none)'}"
                  f" last_seen={d['last_seen'] or 'never'}")
    finally:
        conn.close()


def _cmd_device_revoke(args):
    conn = memsom.get_connection()
    try:
        ok = revoke_device(conn, args.device_id)
        print(f"revoked [{args.device_id}]" if ok else
              f"[{args.device_id}] unknown or already revoked")
        if not ok:
            raise SystemExit(1)
    finally:
        conn.close()


def _cmd_remote_audit_log(args):
    conn = memsom.get_connection()
    try:
        rows = recent_audit(conn, args.limit)
        if not rows:
            print("remote audit log empty")
        for r in rows:
            dev = (r["device_id"] or "?")[:8]
            print(f"[{r['id']}] {r['ts']}  device={dev}  tool={r['tool']}"
                  f"  {r['decision'].upper()}  clearance={r['clearance_used']}")
    finally:
        conn.close()


def register(sub) -> None:
    p = sub.add_parser("device", help="remote device enrolment/revocation (Sec3.5)")
    dsub = p.add_subparsers(dest="device_command", required=True)

    a = dsub.add_parser("add", help="enrol a new device; prints its token once")
    a.add_argument("name")
    a.add_argument("--clearance", default="public",
                   help="clearance ceiling: public|internal|secret|topsecret (default public)")
    a.add_argument("--capabilities", default="",
                   help="comma-separated mutate-tool names this device may call")
    a.set_defaults(func=_cmd_device_add)

    b = dsub.add_parser("list", help="list enrolled devices (never prints tokens)")
    b.set_defaults(func=_cmd_device_list)

    c = dsub.add_parser("revoke", help="revoke a device (one row, not a global rotation)")
    c.add_argument("device_id")
    c.set_defaults(func=_cmd_device_revoke)

    q = sub.add_parser("remote-audit-log", help="recent remote-call audit rows (Sec3.5 point 6)")
    q.add_argument("--limit", type=int, default=20)
    q.set_defaults(func=_cmd_remote_audit_log)
