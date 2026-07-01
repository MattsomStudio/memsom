"""memsom_broker — Gate #3: a provenance-gating MCP broker (the action gate).

memsom's third gate.  The write gate (consolidation/quarantine) governs what an
external node may BECOME; the read gate (clearance floor) governs what may be
RECALLED; this gate governs what an external-tainted session may DO.

The broker is an MCP gateway: it fronts the user's other MCP servers, re-exposes
their tools dot-namespaced as `<upstream>.<tool>`, and funnels every tools/call
through a provenance check BEFORE forwarding.  A session starts at a high
integrity floor (default `user`); when untrusted content flows through the
broker (a web fetch, or any policy-tainting tool) the session floor drops via
min() (memsom_session).  A consequential tool (policy required > external) is
then DENIED — the call never reaches the upstream server.

Run as the `memsom-broker` console entry point (separate process from
`memsom-mcp`, so the frozen MCP server + its tests are untouched).

HONEST BOUNDARY (lead with this): the broker only sees MCP traffic.  Claude
Code's NATIVE tools (WebSearch/WebFetch/Bash/Edit) are not MCP and bypass the
broker entirely — poison-in via native WebFetch + act via native Bash is OUT OF
SCOPE.  The mitigation is operational: route untrusted ingress through the
broker (e.g. a fetch MCP server in `web_fetch_tools`).  See README security note.

Config (~/.memdag/broker.json, or $MEMDAG_BROKER_CONFIG):
  {
    "upstreams": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}},
    "web_fetch_tools": ["fetch.fetch"],
    "policy": "~/.memdag/capability_policy.json",
    "session_start": "user",
    "ingest_external": false
  }
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import memsom
import memsom_capgate
import memsom_policy
import memsom_schema
import memsom_session

SERVER_NAME = "memsom-broker"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def default_config_path() -> Path:
    env = os.environ.get("MEMDAG_BROKER_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".memdag" / "broker.json"


def default_policy_path() -> Path:
    return Path.home() / ".memdag" / "capability_policy.json"


def load_config(path=None) -> dict:
    """Load + validate broker config.  Raises on missing/malformed (never starts
    a broker with a silently-empty/permissive config)."""
    p = Path(path).expanduser() if path else default_config_path()
    if not p.exists():
        raise FileNotFoundError(f"broker config not found: {p} (run `memsom broker-init`)")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed broker config {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: broker config must be a JSON object")

    upstreams = raw.get("upstreams", {})
    if not isinstance(upstreams, dict):
        raise ValueError(f"{p}: 'upstreams' must be an object")
    for name, spec in upstreams.items():
        if "." in name:
            raise ValueError(f"{p}: upstream name {name!r} must not contain '.'")
        if not isinstance(spec, dict) or not spec.get("command"):
            raise ValueError(f"{p}: upstream {name!r} needs a 'command'")

    cfg = {
        "upstreams": upstreams,
        "web_fetch_tools": list(raw.get("web_fetch_tools", [])),
        "policy": raw.get("policy") or str(default_policy_path()),
        "session_start": raw.get("session_start", "user"),
        "ingest_external": bool(raw.get("ingest_external", False)),
    }
    return cfg


# ---------------------------------------------------------------------------
# Rug-pull pins — a tool's schema is hashed on first sight; a later change is
# refused until re-approved (defends dynamic tool redefinition).
# ---------------------------------------------------------------------------

_PIN_SQL = """CREATE TABLE IF NOT EXISTS upstream_tool_pin (
  upstream  TEXT NOT NULL,
  tool      TEXT NOT NULL,
  schema_hash TEXT NOT NULL,
  pinned_at TEXT NOT NULL,
  PRIMARY KEY (upstream, tool)
);"""


def migrate(conn):
    """Ensure the rug-pull pin table exists. Idempotent."""
    memsom_schema.ensure_table(conn, _PIN_SQL)


def _tool_hash(tool: dict) -> str:
    canon = json.dumps(
        {"name": tool.get("name"), "description": tool.get("description"),
         "inputSchema": tool.get("inputSchema")},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def pin_check(conn, upstream: str, tools: list) -> set:
    """Pin each tool's schema hash on first sight; return the set of tool names
    whose hash CHANGED since pinning (rug-pulled — caller must refuse them)."""
    migrate(conn)
    blocked = set()
    now = memsom.now_iso()
    with conn:
        for t in tools:
            name = t.get("name")
            if not name:
                continue
            h = _tool_hash(t)
            row = conn.execute(
                "SELECT schema_hash FROM upstream_tool_pin WHERE upstream=? AND tool=?",
                (upstream, name),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO upstream_tool_pin(upstream, tool, schema_hash, pinned_at)"
                    " VALUES (?,?,?,?)",
                    (upstream, name, h, now),
                )
            elif row[0] != h:
                blocked.add(name)
    return blocked


# ---------------------------------------------------------------------------
# Upstream stdio MCP client
# ---------------------------------------------------------------------------

class Upstream:
    """A spawned upstream MCP server reached over stdio JSON-RPC."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.command = spec["command"]
        self.args = list(spec.get("args", []))
        self.env = {**os.environ, **(spec.get("env") or {})}
        self.proc = None
        self._id = 0
        self.tools = []

    def start(self):
        self.proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, text=True, encoding="utf-8", bufsize=1,
        )
        # initialize handshake
        self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
        self._notify("notifications/initialized", {})
        res = self._rpc("tools/list", {})
        self.tools = (res or {}).get("tools", [])
        return self

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(self, method, params):
        self._id += 1
        my_id = self._id
        self._send({"jsonrpc": "2.0", "id": my_id, "method": method, "params": params})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"upstream {self.name!r} closed during {method}")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != my_id:
                continue  # skip notifications / mismatched ids
            if "error" in msg:
                raise RuntimeError(f"upstream {self.name!r} error on {method}: {msg['error']}")
            return msg.get("result")

    def call(self, tool: str, arguments: dict) -> dict:
        """Forward a tools/call to the upstream; return its result payload."""
        return self._rpc("tools/call", {"name": tool, "arguments": arguments}) or {}

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                # terminate/wait failed for any reason (timeout, already dead,
                # OS error) — force kill as the last-resort cleanup.
                self.proc.kill()


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _result_text(result: dict) -> str:
    parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The gate — decide, then forward.  `caller(up_name, tool, arguments) -> result`
# is injected so this is testable without real subprocesses.
# ---------------------------------------------------------------------------

def decide_and_forward(conn, cfg, policy, sid, fq, arguments, caller) -> dict:
    """Gate a namespaced tools/call `fq` and, if allowed, forward via *caller*.

    Returns an MCP result payload (content + isError).  On deny the call is
    NEVER forwarded.  On an allowed untrusted-content tool, the session floor is
    lowered AFTER a successful call so the NEXT consequential call sees it.
    """
    required = memsom_policy.required_floor(policy, fq)
    floor = memsom_session.current_floor(conn, sid)
    verdict = memsom_capgate.check_capability(conn, sid, floor, fq, required)

    if verdict["decision"] == "deny":
        return _text_result(
            f"memsom-broker DENIED {fq}: session @{verdict['floor_name']} "
            f"< required {verdict['required_name']}. "
            f"An untrusted source tainted this session; restart for a clean floor.",
            is_error=True,
        )

    up_name, _, tool = fq.partition(".")
    result = caller(up_name, tool, arguments)

    taint = memsom_policy.taints(policy, fq)
    if taint is not None and not result.get("isError"):
        reason = "web_fetch" if fq in cfg.get("web_fetch_tools", []) else "policy_taint"
        memsom_session.lower_floor(conn, sid, taint, fq, reason)
        if cfg.get("ingest_external"):
            import memsom_ingest
            text = _result_text(result)
            if text.strip():
                with conn:
                    memsom_ingest.ingest_text(conn, text, "external", source_ref=fq)
    return result


# ---------------------------------------------------------------------------
# Aggregated tool listing (namespaced)
# ---------------------------------------------------------------------------

def aggregated_tools(conn, upstreams: dict) -> list:
    """Return the namespaced union of memsom.* tools and every upstream.* tool,
    minus any rug-pulled (schema-changed) upstream tool."""
    import memsom_mcp  # lazy — avoids a cli<->mcp<->broker import cycle
    out = []
    for t in memsom_mcp.TOOLS:
        nt = dict(t)
        nt["name"] = f"memsom.{t['name']}"
        out.append(nt)
    for name, up in upstreams.items():
        blocked = pin_check(conn, name, up.tools)
        for t in up.tools:
            if t.get("name") in blocked:
                continue
            nt = dict(t)
            nt["name"] = f"{name}.{t['name']}"
            out.append(nt)
    return out


# ---------------------------------------------------------------------------
# Server loop
# ---------------------------------------------------------------------------

def _handle(conn, cfg, policy, upstreams, sid, msg):
    """Handle one parsed JSON-RPC message; return a response dict or None."""
    import memsom_mcp  # lazy
    if not isinstance(msg, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "Invalid Request"}}
    msg_id = msg.get("id")
    method = msg.get("method", "")

    if msg_id is None and method not in ("initialize", "ping"):
        return None

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}}}

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": aggregated_tools(conn, upstreams)}}

    if method == "tools/call":
        params = msg.get("params") or {}
        fq = params.get("name", "")
        arguments = params.get("arguments") or {}

        # memsom's own tools: dispatch in-process, ungated (read/store only).
        if fq.startswith("memsom."):
            inner = memsom_mcp.handle({
                "jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
                "params": {"name": fq[len("memsom."):], "arguments": arguments}})
            return inner

        up_name, _, tool = fq.partition(".")
        if up_name not in upstreams:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32602, "message": f"unknown tool: {fq!r}"}}

        # Rug-pull: refuse a tool whose schema changed since pinning.
        blocked = pin_check(conn, up_name, upstreams[up_name].tools)
        if tool in blocked:
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": _text_result(
                        f"memsom-broker REFUSED {fq}: tool schema changed since "
                        f"first sight (possible rug pull).", is_error=True)}

        def caller(u, t, a):
            return upstreams[u].call(t, a)

        try:
            result = decide_and_forward(conn, cfg, policy, sid, fq, arguments, caller)
        except Exception as exc:
            print(f"[memsom-broker] error forwarding {fq}: {exc}", file=sys.stderr)
            return {"jsonrpc": "2.0", "id": msg_id,
                    "result": _text_result(f"internal broker error for {fq!r}", is_error=True)}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"method not found: {method!r}"}}
    return None


def serve_broker_stdio(conn, cfg, policy, upstreams, sid):
    """Run the broker's stdio JSON-RPC loop until EOF."""
    for s in (sys.stdin, sys.stdout):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(conn, cfg, policy, upstreams, sid, msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# broker-init / policy-check CLI (mounted into the main `memsom` CLI)
# ---------------------------------------------------------------------------

_DEFAULT_BROKER_JSON = {
    "upstreams": {
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "env": {}}
    },
    "web_fetch_tools": ["fetch.fetch"],
    "policy": str(default_policy_path()),
    "session_start": "user",
    "ingest_external": False,
}

_DEFAULT_POLICY_JSON = {
    "default": "deny",
    "rules": [
        {"tool": "fetch.*", "required": "external", "taints": "external"},
        {"tool": "*.read_*", "required": "external"},
        {"tool": "*.get_*", "required": "external"},
        {"tool": "*.search*", "required": "external"},
        {"tool": "*.list_*", "required": "external"},
        {"tool": "*.send_*", "required": "user"},
        {"tool": "*.create_*", "required": "user"},
        {"tool": "*.update_*", "required": "user"},
        {"tool": "*.write_*", "required": "user"},
        {"tool": "*.delete_*", "required": "endorsed"}
    ],
}


def _write_json_if_absent(path: Path, data) -> str:
    if path.exists():
        return "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return "created"


def cmd_broker_init(args):
    cfg_path = Path(args.config).expanduser() if args.config else default_config_path()
    pol_path = default_policy_path()
    c = _write_json_if_absent(cfg_path, _DEFAULT_BROKER_JSON)
    p = _write_json_if_absent(pol_path, _DEFAULT_POLICY_JSON)
    print(f"broker config: {cfg_path}  [{c}]")
    print(f"capability policy: {pol_path}  [{p}]")
    if "exists" in (c, p):
        print("(existing files left untouched — edit them by hand or delete to regenerate)")


def cmd_policy_check(args):
    pol_path = Path(args.policy).expanduser() if args.policy else default_policy_path()
    policy = memsom_policy.load_policy(pol_path)
    req = memsom_policy.required_floor(policy, args.tool)
    conseq = memsom_policy.is_consequential(policy, args.tool)
    taint = memsom_policy.taints(policy, args.tool)
    req_name = memsom.NAME.get(req, "DENY")
    print(f"tool={args.tool}  required={req_name}  consequential={'yes' if conseq else 'no'}"
          + (f"  taints={memsom.NAME[taint]}" if taint is not None else ""))


def register(subparsers):
    p = subparsers.add_parser("broker-init", help="write default Gate #3 broker config + policy")
    p.add_argument("--config", help="broker.json path (default ~/.memdag/broker.json)")
    p.set_defaults(func=cmd_broker_init)

    q = subparsers.add_parser("policy-check", help="show a tool's required floor under the policy")
    q.add_argument("tool", help="namespaced tool name, e.g. gmail.send_message")
    q.add_argument("--policy", help="policy.json path (default ~/.memdag/capability_policy.json)")
    q.set_defaults(func=cmd_policy_check)


# ---------------------------------------------------------------------------
# memsom-broker entry point
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    """Boot the gate in-process with a fake upstream; prove one allow + one deny.
    Exit 0 on success, 1 on failure."""
    import tempfile
    tmp = tempfile.mkdtemp()
    os.environ["MEMDAG_DB"] = str(Path(tmp) / "selfcheck.db")
    conn = memsom.get_connection()
    try:
        policy = memsom_policy._normalize({
            "default": "deny",
            "rules": [
                {"tool": "fetch.fetch", "required": "external", "taints": "external"},
                {"tool": "mail.send", "required": "user"},
            ],
        })
        cfg = {"web_fetch_tools": ["fetch.fetch"], "ingest_external": False}
        sid = memsom_session.begin_session(conn, "user")

        calls = []

        def caller(u, t, a):
            calls.append(f"{u}.{t}")
            return _text_result("ok")

        # 1) consequential action allowed on a clean session
        r1 = decide_and_forward(conn, cfg, policy, sid, "mail.send", {}, caller)
        if r1.get("isError"):
            print("[selfcheck] FAIL: clean-session send was denied", file=sys.stderr)
            return 1
        # 2) untrusted fetch taints the session
        decide_and_forward(conn, cfg, policy, sid, "fetch.fetch", {}, caller)
        if memsom_session.current_floor(conn, sid) != memsom.RANK["external"]:
            print("[selfcheck] FAIL: fetch did not taint session", file=sys.stderr)
            return 1
        # 3) same action now denied, and NOT forwarded
        before = len(calls)
        r3 = decide_and_forward(conn, cfg, policy, sid, "mail.send", {}, caller)
        if not r3.get("isError"):
            print("[selfcheck] FAIL: tainted-session send was allowed", file=sys.stderr)
            return 1
        if len(calls) != before:
            print("[selfcheck] FAIL: denied call was still forwarded", file=sys.stderr)
            return 1
        print("[selfcheck] OK: allow -> taint -> deny (not forwarded)")
        return 0
    finally:
        conn.close()


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(prog="memsom-broker",
                                 description="Gate #3 provenance-gating MCP broker")
    ap.add_argument("--config", help="broker.json path")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run an in-process allow/taint/deny self-test and exit")
    args = ap.parse_args(argv)

    if args.selfcheck:
        sys.exit(_selfcheck())

    cfg = load_config(args.config)
    policy = memsom_policy.load_policy(cfg["policy"])
    conn = memsom.get_connection()
    upstreams = {}
    try:
        for name, spec in cfg["upstreams"].items():
            upstreams[name] = Upstream(name, spec).start()
        sid = memsom_session.begin_session(conn, cfg["session_start"])
        print(f"[memsom-broker] session {sid[:8]} @ {cfg['session_start']}; "
              f"{len(upstreams)} upstream(s); gating consequential tools.", file=sys.stderr)
        serve_broker_stdio(conn, cfg, policy, upstreams, sid)
    finally:
        for up in upstreams.values():
            up.stop()
        conn.close()


if __name__ == "__main__":
    main()
