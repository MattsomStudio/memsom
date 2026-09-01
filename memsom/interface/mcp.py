#!/usr/bin/env python3
"""memsom_mcp — stdio MCP server (JSON-RPC 2.0, no Content-Length framing).

Transport: newline-delimited JSON on stdin/stdout.
  - One JSON object per line in, one JSON object per line out.
  - ALL diagnostics go to stderr only — stray stdout corrupts the protocol.

Entry points:
  python memsom_mcp.py               — run the stdio server (connect via MCP client)
  python memsom_mcp.py --selfcheck   — boot in-process, run 3 probes, exit 0/1

stdlib only.  No third-party deps.
"""

import argparse
import contextlib
import io
import json
import sys
import traceback

from memsom import tuning as memsom_tuning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "memsom"
SERVER_VERSION = "0.4.0"

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
                "graph": {"type": "boolean", "description": "Re-rank retrieval using the rel_edges (wikilink) graph (implies retrieval)"},
                "hops": {"type": "integer", "description": "Graph expansion hops for graph mode (default 1)"},
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
                "clearance": {"type": "string",
                              "description": "confidentiality ceiling (public|internal|secret|topsecret); "
                                             "suppresses content of nodes above it"},
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
                "clearance": {"type": "string",
                              "description": "confidentiality ceiling (public|internal|secret|topsecret); "
                                             "suppresses content of roots above it"},
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
                # ASCII only, deliberately: selfcheck() prints tool descriptions
                # with ensure_ascii=False and, unlike serve_stdio(), does not
                # reconfigure stdout to UTF-8 -- so on Windows a non-ASCII
                # character here is written as cp1252 and the reader cannot
                # decode it. See the note in the S3 report.
                "path": {"type": "string", "description": "Output .jsonl FILENAME, written under the server's export directory. A path outside it is refused, not redirected: an export contains every node's content, so a tool call does not get to choose where the whole store lands."},
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
        "description": "Advisory node-integrity oracle: allow/deny by floor, with "
                        "weakest-leaf culprit. CLI/MCP only, zero internal callers -- "
                        "Gate #3's runtime enforcement is check_capability, not this.",
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
        "name": "code_search",
        "description": "Semantic + BM25 search over the code-RAG index (separate from the fact store; opt-in via MEMSOM_CODE_RAG).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "max results (default 8)"},
                "repo": {"type": "string", "description": "restrict to one indexed repo"},
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
                "channel": {"type": "string",
                            "description": "user|agent-derived|external "
                                           "(endorsed is above this transport's ceiling)"},
                "source_ref": {"type": "string",
                               "description": "free-form reference; the 'memory:' "
                                              "prefix is reserved for the bridge importer"},
            },
            "required": ["text", "channel"],
        },
    },
    {
        "name": "obsidian_sync",
        "description": "Sync an Obsidian vault into the DAG: ingest notes and map [[wikilinks]] to relate-edges. A note's frontmatter memsom-channel can only LOWER its integrity, never raise it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vault": {"type": "string",
                          "description": "Vault path (default: $MEMDAG_OBSIDIAN_VAULT). "
                                         "Must be inside the configured vault."},
                "channel": {"type": "string",
                            "description": "Default channel for un-stamped notes "
                                           "(default: user; endorsed is refused)"},
                "no_prune": {"type": "boolean", "description": "Do not tombstone notes deleted from the vault"},
            },
            "required": [],
        },
    },
    {
        "name": "obsidian_export",
        "description": "Write an answer back to the vault as an agent-derived, memsom-stamped note. Refuses to overwrite a note memsom did not author.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "integer", "description": "Node id to export (omit to use query)"},
                "query": {"type": "string", "description": "Compose an answer to export"},
                "clearance": {"type": "string",
                              "description": "confidentiality ceiling (public|internal|secret|topsecret) for the node_id branch"},
                "vault": {"type": "string", "description": "Vault path (default: $MEMDAG_OBSIDIAN_VAULT)"},
                "folder": {"type": "string", "description": "Subfolder within the vault (default: memsom)"},
                "title": {"type": "string", "description": "Note title (default: derived)"},
            },
            "required": [],
        },
    },
    {
        "name": "verify_stale",
        "description": "Flag state-bearing memory notes whose verification age has gone stale (dry-run by default).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "apply": {"type": "boolean", "description": "Apply marks/clears (default: dry-run, no writes)"},
            },
            "required": [],
        },
    },
]

TOOLS.append({
    "name": "features",
    "description": "Report every optional capability's live state (disabled|absent|degraded|active|error), never silently.",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
})

TOOL_NAMES = {t["name"] for t in TOOLS}


# ---------------------------------------------------------------------------
# Transport trust policy
# ---------------------------------------------------------------------------
#
# `ingest_text`'s own description says "channel set by transport, never
# inferred" — and this transport was setting it to whatever the caller asked
# for, on BOTH stamping tools. That is not the transport declaring a trust
# level; it is the transport forwarding the caller's claim about itself.
#
# THIS transport is a model holding a tool list. Its callers include a model
# that has just read a web page, a vault note, or a recalled transcript. The
# channel that means "the operator personally vouched for this" is not a claim
# such a caller can make, so the transport caps itself below it. Everything
# else is unchanged: `user`, `agent-derived` and `external` all still stamp.
#
# `endorsed` specifically is the rank that is PINNED in the always-loaded index
# and never shed by the byte budget, which is what made it the payload of
# choice. Overridable by the OPERATOR, who starts this process — not by the
# arguments of a call it receives.
MCP_CHANNEL_CEILING_ENV = "MEMSOM_MCP_CHANNEL_CEILING"
_DEFAULT_MCP_CHANNEL_CEILING = "user"

# Where a MODEL-driven export may land. `export` dumps every node's content —
# the whole store, not a summary — through `open(path, "w")`, and the path was
# a free string authored by the LLM. That made one tool call two primitives at
# once: write the entire memory DAG anywhere on disk (including into a
# bidirectionally-replicated tree, which puts it on another machine without the
# tool making a single outbound connection), and truncate any file the process
# can write, because "w" does not care what was there.
#
# A human typing `memsom export <path>` at a shell is choosing a destination
# with their own hands and stays unconstrained; the CLI is untouched. A model
# choosing one is repeating whatever text is in its context. Fence the model,
# not the human.
#
# The operator names the directory, in the environment, once — same shape as
# the vault fence below.
MCP_EXPORT_DIR_ENV = "MEMSOM_MCP_EXPORT_DIR"


def _mcp_channel_ceiling():
    """The highest channel rank this transport may stamp. Never None."""
    import memsom
    from memsom.kernel.lattice import parse_rank

    raw = (memsom_tuning.resolve("mcp.channel_ceiling") or "").strip()
    n = parse_rank(raw) if raw else None
    return n if n is not None else memsom.RANK[_DEFAULT_MCP_CHANNEL_CEILING]


def _checked_channel(raw):
    """Validate a caller-declared channel against this transport's ceiling."""
    import memsom

    key = str(raw).strip().lower()
    if key not in memsom.RANK:
        raise ValueError(
            f"unknown channel: {raw!r} (expected one of "
            f"{'|'.join(memsom.RANK)})")
    ceil = _mcp_channel_ceiling()
    if memsom.RANK[key] > ceil:
        raise ValueError(
            f"refused: this transport may not stamp channel {key!r}. A tool "
            f"call is not the operator vouching for a fact, and {key!r} is "
            f"trusted above what a tool call can establish. Ceiling is "
            f"{memsom.NAME[ceil].lower()!r}; the operator raises it with "
            f"{MCP_CHANNEL_CEILING_ENV}.")
    return key


def _checked_vault(raw):
    """Validate a caller-supplied vault path against the configured vault.

    The vault argument used to be any directory on disk. `obsidian_sync` walks
    what it is given, ingests every markdown file under it, and stamps the lot
    at a channel the same caller chose — so naming a directory was enough to
    turn arbitrary on-disk text into retrievable, trust-stamped memory, and
    naming a directory of attacker-written files was enough to do it with
    attacker-written text.

    The operator names the vault, in the environment, once. A caller may still
    pass a path — a subfolder is a legitimate narrowing — but only one that is
    provably inside it. With no vault configured there is no root to prove
    containment against, so a caller-supplied path is refused outright rather
    than silently trusted.
    """
    from memsom.bridge.obsidian import VAULT_ENV
    from memsom.paths import UnsafePath, safe_join

    root = (memsom_tuning.resolve("obsidian.vault") or "").strip()
    if not root:
        raise ValueError(
            f"refused: a vault path was supplied but no vault is configured. "
            f"Set {VAULT_ENV} to the vault this server may read; a path from a "
            f"tool call is not allowed to choose which directory on disk "
            f"becomes memory.")
    try:
        return str(safe_join(root, str(raw), allow_absolute=True))
    except UnsafePath as exc:
        raise ValueError(
            f"refused: vault path is outside {VAULT_ENV} ({exc})") from exc


def _mcp_export_dir():
    """The one directory a model-driven export may write into.

    Defaults beside the store rather than under the vault: an export is a full
    dump of every node's content, and the vault is a replicated tree, so the
    default destination must not be one that leaves the machine on its own.
    """
    from pathlib import Path

    import memsom

    raw = (memsom_tuning.resolve("mcp.export_dir") or "").strip()
    return Path(raw) if raw else memsom.DATA_DIR / "exports"


def _checked_export_path(raw):
    """Validate a caller-supplied export path against the export directory.

    Refused, never sanitized — the same rule the identifier fences follow.
    Silently rewriting `…/Vault/dump.jsonl` to a basename would make a tool
    call that did something other than what it said, and the caller would have
    no way to tell a fenced write from an honoured one. An error the model can
    read is worth more than a redirect it cannot see.

    A relative name, a subpath, or an absolute path already inside the
    directory are all accepted; a drive letter, a UNC share, `..`, a device
    name or anything else that escapes is not. The `.jsonl` requirement is not
    cosmetic: it keeps the truncating `open(path, "w")` off any other file that
    happens to live in the same directory.
    """
    from memsom.paths import UnsafePath, safe_join

    root = _mcp_export_dir()
    text = str(raw)
    if not text.lower().endswith(".jsonl"):
        raise ValueError(
            f"refused: export path must name a .jsonl file, got {text!r}")
    # The fence proves containment on the STRING before touching the disk, so
    # the directory is created only once the name is known to be acceptable.
    root.mkdir(parents=True, exist_ok=True)
    try:
        return str(safe_join(root, text, allow_absolute=True))
    except UnsafePath as exc:
        raise ValueError(
            f"refused: an export writes the whole store, so a tool call may "
            f"only place it under {root} (set {MCP_EXPORT_DIR_ENV} to move "
            f"that). {exc}") from exc


# ---------------------------------------------------------------------------
# Dispatch: map MCP tool call -> memsom_cli argv
# ---------------------------------------------------------------------------

def _tool_argv(name, arguments):
    """Convert tool name + arguments dict into a memsom_cli argv list.

    Returns a list of strings that can be passed to memsom_cli.main().
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
        if arguments.get("graph"):
            argv.append("--graph")
        if arguments.get("hops") is not None:
            argv += ["--hops", str(arguments["hops"])]
        return argv

    if name == "explain":
        argv = ["explain", str(arguments["id"])]
        if arguments.get("clearance"):
            argv += ["--clearance", str(arguments["clearance"])]
        return argv

    if name == "blame":
        argv = ["blame", str(arguments["id"])]
        if arguments.get("clearance"):
            argv += ["--clearance", str(arguments["clearance"])]
        return argv

    if name == "revoke":
        argv = ["revoke", str(arguments["id"])]
        if arguments.get("reason"):
            argv += ["--reason", str(arguments["reason"])]
        if arguments.get("apply"):
            argv.append("--yes")
        return argv

    if name == "redact":
        argv = ["redact", str(arguments["id"]), "--reason", str(arguments["reason"])]
        # The CLI cascades BY DEFAULT (--single opts out), but this tool's schema
        # documents cascade as opt-in. Honor the schema: anything short of an
        # explicit cascade=true must redact only the named node — over-redaction
        # is irreversible payload destruction.
        if not arguments.get("cascade"):
            argv.append("--single")
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
        argv = ["export", _checked_export_path(arguments["path"])]
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

    if name == "code_search":
        argv = ["code-search", arguments["query"]]
        if arguments.get("k") is not None:
            argv += ["--k", str(arguments["k"])]
        if arguments.get("repo"):
            argv += ["--repo", str(arguments["repo"])]
        return argv

    if name == "ingest_text":
        argv = ["ingest-text", arguments["text"],
                "--channel", _checked_channel(arguments["channel"])]
        if arguments.get("source_ref"):
            argv += ["--ref", str(arguments["source_ref"])]
        return argv

    if name == "obsidian_sync":
        argv = ["obsidian-sync"]
        if arguments.get("vault"):
            argv.append(_checked_vault(arguments["vault"]))
        if arguments.get("channel"):
            argv += ["--channel", _checked_channel(arguments["channel"])]
        if arguments.get("no_prune"):
            argv.append("--no-prune")
        return argv

    if name == "obsidian_export":
        argv = ["obsidian-export"]
        if arguments.get("node") is not None:
            # MS-17: `node` is declared nargs='?' on the CLI, so a bare string
            # value parses as whatever it looks like -- a value shaped like
            # `--vault=...` is read as the OPTION, not the positional, and
            # never reaches `_checked_vault`. Coerce to int at this boundary;
            # a non-numeric node id is refused here, never forwarded.
            argv.append(str(int(arguments["node"])))
        if arguments.get("clearance"):
            argv += ["--clearance", str(arguments["clearance"])]
        if arguments.get("query"):
            argv += ["--query", str(arguments["query"])]
        if arguments.get("vault"):
            argv += ["--vault", _checked_vault(arguments["vault"])]
        if arguments.get("folder"):
            argv += ["--folder", str(arguments["folder"])]
        if arguments.get("title"):
            argv += ["--title", str(arguments["title"])]
        return argv

    if name == "verify_stale":
        argv = ["verify-stale"]
        if arguments.get("apply"):
            argv.append("--apply")
        return argv

    if name == "features":
        return ["features", "--json"]

    raise ValueError(f"unknown tool: {name!r}")


def _call_tool(name, arguments):
    """Execute a tool via memsom_cli.main; return (text, is_error)."""
    # Import here so no DB is opened at module import time
    from memsom.interface import cli as memsom_cli

    try:
        argv = _tool_argv(name, arguments)
    except ValueError as exc:
        return (str(exc), True)
    except KeyError as exc:
        # MCP-2: a missing required argument is a client error, not an internal
        # crash — report the missing field, don't leak a traceback.
        return (f"missing required argument: {exc}", True)

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    is_error = False

    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            memsom_cli.main(argv)
    except SystemExit as exc:
        code = exc.code
        if code not in (0, None):
            is_error = True
    except Exception:
        # MCP-2 (inner path): _call_tool CATCHES the exception and returns its
        # text, so writing the traceback into out_buf leaked absolute paths + the
        # internal call graph to the client (the outer handle() guard never sees
        # it). Log the traceback to stderr for the operator; return only a generic
        # message — matching the MCP-2 contract on every route.
        is_error = True
        print(traceback.format_exc(), file=sys.stderr)
        out_buf.write(f"internal error in tool {name!r}")

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
    # MCP-1: a valid JSON line that decodes to a non-object (number/string/array/
    # null) must not crash the server. Reject it with a JSON-RPC -32600 instead of
    # raising AttributeError on .get().
    if not isinstance(msg, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request: expected a JSON object"},
        }
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
            # MCP-2: log the full traceback to stderr (operator), but NEVER return
            # it to the client — it leaks absolute paths + the internal call graph.
            tb = traceback.format_exc()
            print(f"[memsom-mcp] unhandled error in _call_tool: {tb}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"internal error in tool {tool_name!r}"}],
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

def _start_warm_endpoint():
    """Start the loopback retrieval endpoint the prompt hook queries (see
    retrieval/warm.py). Best-effort: a failure to bind is logged to stderr
    and the MCP server runs without it (the hook falls back to BM25).
    MEMDAG_WARM_ENDPOINT=0 disables it."""
    from memsom.retrieval import warm as memsom_warm
    if memsom_warm.disabled_by_env():
        return None
    try:
        srv = memsom_warm.WarmServer().start()
        print(f"[memsom-mcp] warm endpoint on 127.0.0.1:{srv.port} ({srv.file})",
              file=sys.stderr)
    # FAILOPEN: allowed, a bind failure logs and the server runs without it (falls back to BM25).
    except Exception as exc:
        print(f"[memsom-mcp] warm endpoint not started: {exc!r}", file=sys.stderr)
        return None
    # Watchdog: self-ping every ~60 s through the real socket; a listener that
    # accepts but never answers (the 2026-08-20 CLOSE_WAIT wedge) is restarted
    # on a fresh port + token, and the endpoint file is rewritten.
    try:
        srv.watchdog = memsom_warm.WarmWatchdog(srv).start()
    # FAILOPEN: allowed, a watchdog that fails to start just means no self-heal.
    except Exception as exc:
        print(f"[memsom-mcp] warm watchdog not started: {exc!r}", file=sys.stderr)
    return srv


def _stop_warm_endpoint(srv):
    """Stop the watchdog first (so it cannot restart a listener we are tearing
    down), then the listener; the endpoint file goes with it."""
    if srv is None:
        return
    wd = getattr(srv, "watchdog", None)
    if wd is not None:
        wd.stop()
    srv.stop()


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

    warm = _start_warm_endpoint()
    try:
        _serve_lines(sys.stdin)
    finally:
        _stop_warm_endpoint(warm)   # removes the endpoint file on every exit path


def _serve_lines(stream):
    for raw_line in stream:
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
            print(f"[memsom-mcp] handle() crash: {tb}", file=sys.stderr)
            response = {
                "jsonrpc": "2.0",
                "id": msg.get("id") if isinstance(msg, dict) else None,
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
        prog="memsom_mcp",
        description="memsom MCP stdio server (JSON-RPC 2.0)",
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
