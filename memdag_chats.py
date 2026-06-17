#!/usr/bin/env python3
"""memdag_chats — opt-in ingestion of a user's OWN chat history.

Parses local AI-client transcripts into clean user/assistant message text and
ingests each message stamped channel="user" (it is the operator's own data),
deduped by content hash. NONE of the author's chats ship in the repo; this only
ever reads the running user's local files, and only when they opt in.

Supported clients:
  - claude-code : ~/.claude/projects/**/*.jsonl
  - codex       : ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
  - claude-desktop : chats live server-side (no local store) -> manual export only.

Design: ONE jsonl iterator + a thin per-client extractor. The extractors filter
to real user/assistant message text and drop tool calls, mirrors, and metadata.
"""

import json
import sys
from pathlib import Path

import memdag
import memdag_ingest


# ---------------------------------------------------------------------------
# Shared jsonl core
# ---------------------------------------------------------------------------

def _iter_jsonl_records(path, extract):
    """Yield extract(obj) dicts (skipping None / blank / malformed lines).

    Each emitted record gets source_ref = '<filename>#L<lineno>' so re-ingesting
    the same transcript dedups cleanly and provenance points back at the line.
    """
    p = Path(path)
    with p.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            rec = extract(obj)
            if rec is None or not rec.get("text", "").strip():
                continue
            rec.setdefault("source_ref", f"{p.name}#L{lineno}")
            yield rec


def _content_to_text(content):
    """Flatten a message 'content' field that is either a plain string or a list
    of content-block objects, keeping only the text. Non-text blocks (tool_use,
    images, tool_result) carry no string 'text' and are naturally dropped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Per-client extractors
# ---------------------------------------------------------------------------

def _claude_code_extract(obj):
    # Claude Code: top-level type is the role; real text in message.content.
    if obj.get("type") not in ("user", "assistant"):
        return None  # drops queue-operation / attachment / ai-title / etc.
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    role = msg.get("role") or obj.get("type")
    return {"role": role, "text": _content_to_text(msg.get("content"))}


def _codex_extract(obj):
    # Codex: keep response_item/message only. Every message is ALSO written as an
    # event_msg mirror (text in payload.message); we drop those at the type gate
    # so the corpus is not doubled — do NOT rely on dedup to clean the mirror.
    if obj.get("type") != "response_item":
        return None  # drops event_msg, session_meta, turn_context
    payload = obj.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "message":
        return None  # drops function_call / function_call_output
    role = payload.get("role")
    if role not in ("user", "assistant"):
        return None  # drops developer/system
    return {"role": role, "text": _content_to_text(payload.get("content"))}


EXTRACTORS = {"claude-code": _claude_code_extract, "codex": _codex_extract}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def claude_code_files(home=None):
    base = Path(home or Path.home()) / ".claude" / "projects"
    return sorted(base.glob("**/*.jsonl")) if base.exists() else []


def codex_files(home=None):
    base = Path(home or Path.home()) / ".codex" / "sessions"
    return sorted(base.glob("**/rollout-*.jsonl")) if base.exists() else []


DISCOVERY = {"claude-code": claude_code_files, "codex": codex_files}


def parse_file(client, path):
    """Return the list of {role, text, source_ref} records from one transcript."""
    return list(_iter_jsonl_records(path, EXTRACTORS[client]))


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_chats(conn, client, files=None, dry_run=False):
    """Parse + ingest every transcript for *client*.

    CHATS-1: integrity channel is stamped from each message's ROLE — a user turn
    is 'user' (label 2), an assistant turn is 'agent-derived' (label 1). Stamping
    everything 'user' laundered assistant text (which can echo web/tool output or
    reflected injection) into the high-trust tier.
    CHATS-2: an unreadable transcript skips (counted), it never aborts the batch.
    """
    if files is None:
        files = DISCOVERY[client]()
    files = list(files)
    msgs = 0
    files_skipped = 0
    before = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    for path in files:
        try:
            records = list(_iter_jsonl_records(path, EXTRACTORS[client]))
        except OSError:
            files_skipped += 1
            continue
        for rec in records:
            msgs += 1
            if not dry_run:
                channel = "user" if rec.get("role") == "user" else "agent-derived"
                memdag_ingest.ingest_text(conn, rec["text"], channel,
                                          source_ref=rec["source_ref"])
    after = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    return {"client": client, "files": len(files), "messages": msgs,
            "files_skipped": files_skipped,
            "new_nodes": after - before, "dry_run": dry_run}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def register(subparsers):
    p = subparsers.add_parser(
        "ingest-chats",
        help="opt-in: seed memdag from your OWN local chat history (channel=user)")
    p.add_argument("--client", choices=["claude-code", "codex", "all"], default="all",
                   help="which client's local transcripts to ingest (default: all found)")
    p.add_argument("--file", default=None,
                   help="ingest one explicit transcript file (requires a single --client)")
    p.add_argument("--yes", action="store_true",
                   help="skip the opt-in prompt (the bootstrap passes this after asking)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + count only; insert nothing")
    p.set_defaults(func=cmd_ingest_chats)


def cmd_ingest_chats(args):
    if args.file and args.client == "all":
        print("[memdag] --file requires an explicit --client (claude-code or codex)",
              file=sys.stderr)
        sys.exit(1)

    clients = ["claude-code", "codex"] if args.client == "all" else [args.client]
    if args.file:
        files_map = {args.client: [args.file]}
    else:
        files_map = {c: DISCOVERY[c]() for c in clients}

    total = sum(len(v) for v in files_map.values())
    if total == 0:
        print("[memdag] no local transcripts found for "
              f"{', '.join(clients)}. (Claude Desktop chats live server-side — "
              "export them and use --file.)", file=sys.stderr)
        sys.exit(0)

    # Opt-in: never vacuum a user's history without an explicit yes.
    if not args.yes and not args.dry_run:
        print("Found local transcripts:", file=sys.stderr)
        for c, fs in files_map.items():
            print(f"  {c}: {len(fs)} file(s)", file=sys.stderr)
        resp = input("Ingest these into memdag (channel=user)? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("[memdag] aborted; nothing ingested.", file=sys.stderr)
            sys.exit(0)

    conn = memdag.get_connection()
    try:
        for c, fs in files_map.items():
            s = ingest_chats(conn, c, files=fs, dry_run=args.dry_run)
            tail = " (dry run)" if args.dry_run else ""
            print(f"[{c}] files={s['files']} messages={s['messages']} "
                  f"new_nodes={s['new_nodes']}{tail}")
    finally:
        conn.close()
