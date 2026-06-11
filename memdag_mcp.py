#!/usr/bin/env python3
"""memdag_mcp — stdio MCP server (JSON-RPC 2.0, no Content-Length framing).

Transport: newline-delimited JSON on stdin/stdout.
  - One JSON object per line in, one JSON object per line out.
  - ALL diagnostics go to stderr only — stray stdout corrupts the protocol.

Entry points:
  python memdag_mcp.py               — run the stdio server (connect via MCP client)
  python memdag_mcp.py --selfcheck   — boot in-process, run 3 probes, exit 0/1

stdlib only.  No third-party deps.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import traceback


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "memdag"
SERVER_VERSION = "0.3.0"

TOOLS = [
    {
        "name": "ask",
        "description": "Compose an answer from live provenance-verified sources.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer"},
                "clearance": {"type": "string", "description": "Confidentiality clearance level (default: topsecret = no filter)"},
                "anticipate": {"type": "boolean", "description": "Use surprise-gating (cite existing if low-novelty)"},
                "llm": {"type": "boolean", "description": "Use local Ollama LLM (opt-in; falls back to deterministic)"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "explain",
        "description": "Show the full provenance tree for a node.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Node id"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "blame",
        "description": "Trace a node back to its root source(s).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Node id to blame"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "revoke",
        "description": "Tombstone a node and cascade to all descendants (dry-run by default).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Node id to revoke"},
                "reason": {"type": "string", "description": "Revocation reason"},
                "apply": {"type": "boolean", "description": "If true, apply (default false = dry-run)"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "redact",
        "description": "Destroy a node's payload while preserving the DAG shape (dry-run by default).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Node id to redact"},
                "reason": {"type": "string", "description": "Why the payload is being destroyed"},
                "cascade": {"type": "boolean", "description": "Also redact all transitive descendants"},
                "apply": {"type": "boolean", "description": "If true, apply (default false = dry-run)"},
            },
            "required": ["id", "reason"],
        },
    },
    {
        "name": "recompute",
        "description": "Recompute multi-hop integrity labels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Node id to recompute (mutually exclusive with all)"},
                "all": {"type": "boolean", "description": "Recompute all live derived nodes"},
            },
            "required": [],
        },
    },
    {
        "name": "consolidate",
        "description": "Run the consolidation gate: quarantine agent-derived nodes tainted by external sources.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "check",
        "description": "Check for invariant violations in the DAG.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "export",
        "description": "Export a changeset (for federation/sync).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Output file path"},
                "since": {"type": "string", "description": "ISO-8601 timestamp; export only nodes created after this"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "neighborhood",
        "description": "BFS over associative rel_edges with integrity-floor propagation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Start node id"},
                "hops": {"type": "integer", "description": "Maximum hops (default 2)"},
                "min_integrity": {"type": "string", "description": "Minimum integrity level (default 0/external)"},
                "clearance": {"type": "string", "description": "Max confidentiality clearance (default topsecret)"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "profile",
        "description": "Leaf-origin provenance histogram + floor (display-only; never gates).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Node id"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "check_action",
        "description": "Action-time integrity gate: allow/deny by floor, with weakest-leaf culprit. The ONLY gate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Node id"},
                "required": {"type": "string", "description": "Minimum integrity floor (external|agent-derived|user|endorsed or 0-3)"},
            },
            "required": ["id", "required"],
        },
    },
    {
        "name": "retrieve",
        "description": "Hybrid BM25 + optional-vector ranked retrieval over live sources.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "max results (default 8)"},
                "clearance": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ingest_text",
        "description": "Stamp and store raw text at a declared channel (channel set by transport, never inferred).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "channel": {"type": "string", "description": "endorsed|user|agent-derived|external"},
                "source_ref": {"type": "string"},
            },
            "required": ["text", "channel"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in TOOLS}


# ---------------------------------------------------------------------------
# Dispatch: map MCP tool call -> memdag_cli argv
# ---------------------------------------------------------------------------

def _tool_argv(name, arguments):
    """Convert tool name + arguments dict into a memdag_cli argv list.

    Returns a list of strings that can be passed to memdag_cli.main().
    Raises ValueError for unsupported tool names (caller converts to -32602).
    """
    if name == "ask":
        argv = ["ask", arguments["question"]]
        if arguments.get("clearance"):
            argv += ["--clearance", str(arguments["clearance"])]
        if arguments.get("anticipate"):
            argv.append("--anticipate")
        if arguments.get("llm"):
            argv.append("--llm")
        return argv

    if name == "explain":
        return ["explain", str(arguments["id"])]

    if name == "blame":
        return ["blame", str(arguments["id"])]

    if name == "revoke":
        argv = ["revoke", str(arguments["id"])]
        if arguments.get("reason"):
            argv += ["--reason", str(arguments["reason"])]
        if arguments.get("apply"):
            argv.append("--yes")
        return argv

    if name == "redact":
        argv = ["redact", str(arguments["id"]), "--reason", str(arguments["reason"])]
        if arguments.get("cascade"):
            argv.append("--cascade")
        if arguments.get("apply"):
            argv.append("--yes")
        return argv

    if name == "recompute":
        if arguments.get("all"):
            return ["recompute", "--all"]
        if arguments.get("id") is not None:
            return ["recompute", str(arguments["id"])]
        # Default: run --all if nothing specified
        return ["recompute", "--all"]

    if name == "consolidate":
        return ["consolidate"]

    if name == "check":
        return ["check"]

    if name == "export":
        argv = ["export", str(arguments["path"])]
        if arguments.get("since"):
            argv += ["--since", str(arguments["since"])]
        return argv

    if name == "neighborhood":
        argv = ["neighborhood", str(arguments["id"])]
        if arguments.get("hops") is not None:
            argv += ["--hops", str(arguments["hops"])]
        if arguments.get("min_integrity") is not None:
            argv += ["--min-integrity", str(arguments["min_integrity"])]
        if arguments.get("clearance") is not None:
            argv += ["--clearance", str(arguments["clearance"])]
        return argv

    if name == "profile":
        return ["profile", str(arguments["id"])]

    if name == "check_action":
        return ["check-action", str(arguments["id"]), "--require", str(arguments["required"])]

    if name == "retrieve":
        argv = ["retrieve", arguments["query"]]
        if arguments.get("k") is not None:
            argv += ["--k", str(arguments["k"])]
        if arguments.get("clearance"):
            argv += ["--clearance", str(arguments["clearance"])]
        return argv

    if name == "ingest_text":
        argv = ["ingest-text", arguments["text"], "--channel", str(arguments["channel"])]
        if arguments.get("source_ref"):
            argv += ["--ref", str(arguments["source_ref"])]
        return argv

    raise ValueError(f"unknown tool: {name!r}")


def _call_tool(name, arguments):
    """Execute a tool via memdag_cli.main; return (text, is_error)."""
    # Import here so no DB is opened at module import time
    import memdag_cli

    try:
        argv = _tool_argv(name, arguments)
    except ValueError as exc:
        return (str(exc), True)

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    is_error = False

    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            memdag_cli.main(argv)
    except SystemExit as exc:
        code = exc.code
        if code not in (0, None):
            is_error = True
    except Exception:
        is_error = True
        out_buf.write(traceback.format_exc())

    stdout_text = out_buf.getvalue()
    stderr_text = err_buf.getvalue()

    if is_error and stderr_text:
        text = stdout_text + ("\n" if stdout_text else "") + stderr_text
    else:
        text = stdout_text

    return (text, is_error)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 handler
# ---------------------------------------------------------------------------

def handle(msg):
    """Dispatch a parsed JSON-RPC message dict.  Returns a response dict or None.

    None is returned for id-less notifications (no response expected).
    """
    msg_id = msg.get("id")  # may be None for notifications
    method = msg.get("method", "")

    # ---- Notifications (no response) ----
    if msg_id is None and method not in ("initialize", "ping"):
        return None

    # ---- initialize ----
    if method == "initialize":
        # Echo back the protocol version the client requested if it looks like a
        # date string (YYYY-MM-DD...); otherwise use our default.
        requested = (msg.get("params") or {}).get("protocolVersion", "")
        version = requested if (isinstance(requested, str) and len(requested) >= 8 and requested[:4].isdigit()) else PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    # ---- ping ----
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    # ---- tools/list ----
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    # ---- tools/call ----
    if method == "tools/call":
        params = msg.get("params") or {}
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}

        if tool_name not in TOOL_NAMES:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": f"unknown tool: {tool_name!r}",
                },
            }

        try:
            text, is_error = _call_tool(tool_name, arguments)
        except Exception:
            tb = traceback.format_exc()
            print(f"[memdag-mcp] unhandled error in _call_tool: {tb}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": tb}],
                    "isError": True,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }

    # ---- unknown method ----
    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"method not found: {method!r}",
            },
        }
    return None


# ---------------------------------------------------------------------------
# stdio server loop
# ---------------------------------------------------------------------------

def serve_stdio():
    """Run the stdio server loop. Reads until EOF; writes one JSON line per response."""
    # Reconfigure streams to UTF-8 line-buffered
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n\r")
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        try:
            response = handle(msg)
        except Exception:
            tb = traceback.format_exc()
            print(f"[memdag-mcp] handle() crash: {tb}", file=sys.stderr)
            response = {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "error": {"code": -32603, "message": "internal error"},
            }

        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# --selfcheck mode
# ---------------------------------------------------------------------------

def selfcheck():
    """Boot the server in-process, run 3 probes, print each response, exit 0/1."""
    ok = True

    probe1 = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
               "params": {"protocolVersion": PROTOCOL_VERSION}}
    r1 = handle(probe1)
    print(json.dumps(r1, ensure_ascii=False))
    if not (r1 and r1.get("result", {}).get("serverInfo")):
        print("[selfcheck] FAIL: initialize did not return serverInfo", file=sys.stderr)
        ok = False

    probe2 = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    r2 = handle(probe2)
    print(json.dumps(r2, ensure_ascii=False))
    returned_names = {t["name"] for t in r2.get("result", {}).get("tools", [])}
    if returned_names != TOOL_NAMES:
        print(f"[selfcheck] FAIL: tools/list returned {returned_names} != {TOOL_NAMES}",
              file=sys.stderr)
        ok = False

    probe3 = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
               "params": {"name": "check", "arguments": {}}}
    r3 = handle(probe3)
    print(json.dumps(r3, ensure_ascii=False))
    if r3.get("result", {}).get("isError"):
        print("[selfcheck] FAIL: tools/call check returned isError=true", file=sys.stderr)
        ok = False

    if ok:
        print("[selfcheck] OK", file=sys.stderr)
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="memdag_mcp",
        description="memdag MCP stdio server (JSON-RPC 2.0)",
    )
    ap.add_argument("--selfcheck", action="store_true",
                    help="run in-process self-check and exit (safe on any DB)")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
    else:
        serve_stdio()


if __name__ == "__main__":
    main()
