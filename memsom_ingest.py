"""memsom_ingest — real write path for the derivation DAG.

SPINE invariant: channel is stamped by the ADAPTER (transport), NEVER inferred
from content.  This module is that adapter — the caller declares the channel,
and this code enforces it without peeking at content.

Schema migration: adds content_hash TEXT (nullable, no default) to nodes, plus
a covering index idx_nodes_content_hash.  Existing rows get NULL — safe for all
frozen-core behaviour (content_hash is never read by memsom.py).

Public API
----------
migrate(conn)
    Idempotent: add content_hash column + index.

ingest_text(conn, text, channel, source_ref=None, chunk=True, chunk_chars=1200)
    -> list[int]
    Split long text into ~chunk_chars chunks on paragraph/sentence boundaries.
    For each chunk, sha256(normalized) -> if a LIVE node with that content_hash
    exists, reuse it (dedup); else insert_node(channel, source_ref=f"{ref}#chunk{i}")
    and store content_hash.  Best-effort: after insert, try to call
    memsom_retrieve.index_node(conn, nid) inside try/except.
    Returns list of node ids (reused or newly created).

ingest_file(conn, path, channel) -> list[int]
    Read UTF-8 (errors=replace), call ingest_text with source_ref=str(path).

ingest_dir(conn, dirpath, channel, glob="*.md") -> list[int]
    Walk + ingest each matching file.  Returns flat list of all node ids.

ingest_url(conn, url) -> list[int]
    urllib GET (User-Agent, timeout=15), channel forced "external", source_ref=url.
    Raises OSError / urllib.error.URLError on failure — callers decide to retry/log.

CLI sub-commands (via register(subparsers))
-------------------------------------------
ingest <path> --channel <c>
ingest-dir <dir> --channel <c> [--glob G]
ingest-url <url>
"""

import argparse
import fnmatch
import hashlib
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

import memsom
import memsom_schema

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_CHARS = 1200

# F-13: optional channel ceiling. The operator is the trust authority for this
# single-user tool, so the default (env unset) is permissive. When set, it caps
# the highest channel any stamping entry point (CLI add/ingest, MCP ingest_text)
# may declare — e.g. MEMDAG_CHANNEL_CEILING=user disallows stamping `endorsed`
# from these untrusted-by-policy entry points. ingest-url is already hard-locked
# to external and is therefore always under any ceiling.
CHANNEL_CEILING_ENV = "MEMDAG_CHANNEL_CEILING"


# ---------------------------------------------------------------------------
# Caller-layer trust guards (F-13 channel ceiling, F-14 channel/label lock)
# ---------------------------------------------------------------------------


def authoritative_label(channel: str) -> int:
    """F-14: a SOURCE node's integrity label is dictated SOLELY by its channel.

    The frozen insert_node() accepts an explicit label that may disagree with
    the channel; this is the caller-layer enforcement point. Entry points stamp
    RANK[channel] and never a caller-supplied label, so a channel/label mismatch
    cannot be injected through add/ingest. Raises ValueError on unknown channel.
    """
    if channel not in memsom.RANK:
        raise ValueError(f"unknown channel: {channel!r}")
    return memsom.RANK[channel]


def channel_ceiling():
    """Return the configured max channel RANK (int 0-3) or None if unset/permissive."""
    raw = os.environ.get(CHANNEL_CEILING_ENV)
    if raw is None or not raw.strip():
        return None
    key = raw.strip().lower()
    if key in memsom.RANK:
        return memsom.RANK[key]
    try:
        v = int(key)
    except ValueError:
        raise ValueError(
            f"invalid {CHANNEL_CEILING_ENV}={raw!r}: expected a channel name or 0-3"
        ) from None
    if v not in memsom.NAME:
        raise ValueError(f"{CHANNEL_CEILING_ENV} out of range 0-3: {v}")
    return v


def enforce_channel_ceiling(channel: str) -> str:
    """F-13: reject *channel* if a ceiling is configured and the channel exceeds it.

    Default (no ceiling configured) is permissive — returns the channel unchanged.
    Raises ValueError on an unknown channel or a ceiling violation.
    """
    if channel not in memsom.RANK:
        raise ValueError(f"unknown channel: {channel!r}")
    ceil = channel_ceiling()
    if ceil is not None and memsom.RANK[channel] > ceil:
        raise ValueError(
            f"channel {channel!r} (rank {memsom.RANK[channel]}) exceeds "
            f"{CHANNEL_CEILING_ENV}={ceil} ({memsom.NAME[ceil]}); refused by entry-point policy"
        )
    return channel

# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent: add content_hash column to nodes and create covering index."""
    memsom_schema.add_column(conn, "nodes", "content_hash", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_content_hash ON nodes(content_hash)"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normalize text for hashing: collapse all whitespace runs to single space,
    strip leading/trailing whitespace.  Ensures semantically identical chunks
    produce the same hash regardless of whitespace noise.
    """
    import re
    return re.sub(r"\s+", " ", text).strip()


def _content_hash(text: str) -> str:
    """Return hex-encoded SHA-256 of the normalized text."""
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _split_chunks(text: str, chunk_chars: int) -> list:
    """Split *text* into chunks of at most *chunk_chars* characters each.

    Splitting strategy (in order of preference):
      1. Double-newline (paragraph boundary) — keeps semantic units together.
      2. Single newline.
      3. '. ' (sentence boundary).
      4. Whitespace boundary nearest to chunk_chars (avoids mid-word cuts).
      5. Hard split at chunk_chars (absolute last resort when no whitespace found).

    Guarantees: every yielded chunk is non-empty; no content is dropped
    (all characters from the input appear in some chunk, in order).
    """
    # INGEST-4: chunk_chars <= 0 makes the slice window never shrink `remaining`,
    # spinning forever while holding the connection. Reject it at the boundary.
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be >= 1")
    if len(text) <= chunk_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks = []
    remaining = text

    while len(remaining) > chunk_chars:
        window = remaining[:chunk_chars]

        # 1. Try paragraph break
        pos = window.rfind("\n\n")
        if pos > 0:
            cut = pos + 2
        else:
            # 2. Try single newline
            pos = window.rfind("\n")
            if pos > 0:
                cut = pos + 1
            else:
                # 3. Try sentence boundary
                pos = window.rfind(". ")
                if pos > 0:
                    cut = pos + 2
                else:
                    # 4. Whitespace nearest to chunk_chars (avoid mid-word cut)
                    pos = window.rfind(" ")
                    if pos > 0:
                        cut = pos + 1  # include the space in the consumed slice
                    else:
                        # 5. True hard cut (no whitespace at all in window)
                        cut = chunk_chars

        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:]

    tail = remaining.strip()
    if tail:
        chunks.append(tail)

    return chunks


def _find_live_by_hash(conn: sqlite3.Connection, h: str, channel: str):
    """Return the id of an UNTAINTED, SAME-CHANNEL node with content_hash == h.

    INGEST-1/2: dedup must not reuse a redacted, quarantined, or archived node.
    A redacted node keeps its content_hash (content was zeroed, not the hash), and
    a quarantined node is excluded from the untainted pool — deduping onto either
    silently dropped the freshly-supplied content and/or handed back an excluded
    node. Source the WHERE fragment from the shared taint filter so this dedup
    path inherits every taint dimension the read pools already enforce.

    INGEST-DEDUP-CHANNEL / CHATS-1-DEDUP-LAUNDER: dedup must also match on CHANNEL.
    Otherwise identical bytes ingested under a DIFFERENT channel silently reuse the
    existing node — an endorsed ingest returning a lower-integrity external node,
    or an assistant turn (agent-derived) laundered onto an identical user node,
    defeating the channel->label invariant. A cross-channel match mints a fresh
    node with the caller's declared channel.
    """
    clauses, params = memsom_schema.taint_filter_clauses(conn)
    row = conn.execute(
        "SELECT id FROM nodes WHERE content_hash = ? AND channel = ? AND "
        + " AND ".join(clauses) + " LIMIT 1",
        [h, channel] + params,
    ).fetchone()
    return row[0] if row else None


def _try_index(conn: sqlite3.Connection, nid: int) -> None:
    """Best-effort call to memsom_retrieve.index_node(conn, nid).

    Silently ignores ImportError (memsom_retrieve absent) and any other
    exception — ingest must work without the retrieve module.
    """
    try:
        import memsom_retrieve  # noqa: PLC0415
        memsom_retrieve.index_node(conn, nid)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _find_live_predecessor(conn: sqlite3.Connection, source_ref: str, channel: str,
                           new_hash: str):
    """Return the LIVE, same-channel node with the SAME source_ref but a DIFFERENT
    content_hash — the prior version this re-ingest supersedes (newest if several).

    Sources its liveness WHERE-fragment from the shared taint filter (same
    discipline as _find_live_by_hash), so it never selects a tombstoned / redacted
    / quarantined / archived node — a re-ingest cannot "supersede" an already-dead
    version. Returns None when source_ref is empty or no prior version exists.
    """
    if not source_ref:
        return None
    clauses, params = memsom_schema.taint_filter_clauses(conn)
    row = conn.execute(
        "SELECT id FROM nodes WHERE source_ref = ? AND channel = ?"
        " AND content_hash IS NOT NULL AND content_hash != ? AND "
        + " AND ".join(clauses) + " ORDER BY id DESC LIMIT 1",
        [source_ref, channel, new_hash] + params,
    ).fetchone()
    return row[0] if row else None


def ingest_text(
    conn: sqlite3.Connection,
    text: str,
    channel: str,
    source_ref: str = None,
    chunk: bool = True,
    chunk_chars: int = _DEFAULT_CHUNK_CHARS,
) -> list:
    """Ingest *text* into the DAG under *channel*.

    Parameters
    ----------
    conn        : open sqlite3.Connection (get_connection() already called)
    text        : raw UTF-8 text
    channel     : one of endorsed / user / agent-derived / external
    source_ref  : optional string reference (file path, URL, etc.)
    chunk       : if True, split long text into ~chunk_chars chunks
    chunk_chars : target chunk size in characters

    Returns
    -------
    list[int] — node ids for each chunk, in order (may include reused ids)
    """
    migrate(conn)

    # Caller-layer trust guards: refuse an over-ceiling channel (F-13) and pin the
    # integrity label to the channel (F-14) — never trust a caller-supplied label.
    enforce_channel_ceiling(channel)
    label = authoritative_label(channel)

    if not chunk or len(text) <= chunk_chars:
        chunks = [text.strip()] if text.strip() else []
    else:
        chunks = _split_chunks(text, chunk_chars)

    if not chunks:
        # Nothing to store
        return []

    ids = []
    base_ref = source_ref or ""

    for i, chunk_text in enumerate(chunks):
        h = _content_hash(chunk_text)

        # Dedup: reuse LIVE node with same hash AND same channel
        existing = _find_live_by_hash(conn, h, channel)
        if existing is not None:
            ids.append(existing)
            continue

        # Build source_ref: if multi-chunk, append #chunk{i}; else use as-is
        if len(chunks) > 1:
            ref = f"{base_ref}#chunk{i}" if base_ref else f"#chunk{i}"
        else:
            ref = source_ref  # keep None as None for single-chunk text

        with conn:
            nid = memsom.insert_node(conn, chunk_text, channel,
                                     label=label, source_ref=ref)
            conn.execute(
                "UPDATE nodes SET content_hash = ? WHERE id = ?", (h, nid)
            )

        _try_index(conn, nid)
        ids.append(nid)

        # Staleness auto-detect (single-chunk only — multi-chunk has no clean
        # chunk-to-chunk supersession alignment; those use the manual stale-cascade
        # verb). If this re-ingest replaces a prior LIVE version of the SAME source,
        # record old->new and fire the staleness cascade. Best-effort: a failure
        # here must never break ingest (the new node is already committed).
        if len(chunks) == 1 and source_ref:
            pred = _find_live_predecessor(conn, source_ref, channel, h)
            if pred is not None and pred != nid:
                try:
                    import memsom_stale  # noqa: PLC0415 — lazy: avoid import cycle
                    memsom_stale.on_reingest_supersede(conn, pred, nid, source_ref)
                except Exception:  # noqa: BLE001 — staleness is best-effort
                    pass

    return ids


def ingest_file(conn: sqlite3.Connection, path, channel: str) -> list:
    """Ingest the file at *path* (UTF-8, errors replaced) under *channel*.

    Returns list[int] of node ids.
    Raises OSError if the file cannot be read.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return ingest_text(conn, text, channel, source_ref=str(path))


def ingest_dir(
    conn: sqlite3.Connection,
    dirpath,
    channel: str,
    glob: str = "*.md",
) -> list:
    """Ingest all files matching *glob* under *dirpath* (recursive walk).

    Returns flat list[int] of all node ids across all files.
    Files that cannot be read are silently skipped (OSError caught per file).
    """
    dirpath = Path(dirpath)
    ids = []
    for root, _dirs, files in os.walk(dirpath):
        for fname in sorted(files):  # sorted = deterministic order
            if fnmatch.fnmatch(fname, glob):
                fpath = Path(root) / fname
                try:
                    ids.extend(ingest_file(conn, fpath, channel))
                except OSError:
                    pass  # unreadable file: skip, don't abort the batch
    return ids


def ingest_url(conn: sqlite3.Connection, url: str) -> list:
    """Fetch *url* (GET) and ingest the response body.

    Channel is ALWAYS forced to "external" — the transport dictates the channel,
    never the content (SPINE invariant).
    source_ref is set to the URL.

    Returns list[int] of node ids.
    Raises urllib.error.URLError / OSError on network/HTTP failure.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "memsom-ingest/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()

    # Decode: try UTF-8, fall back to latin-1 (always succeeds)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    return ingest_text(conn, text, "external", source_ref=url)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_ingest(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        path = Path(args.path)
        try:
            ids = ingest_file(conn, path, args.channel)
        except ValueError as exc:
            print(f"[memsom-ingest] {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"ingested {len(ids)} node(s) from {path} [channel={args.channel}]")
        for nid in ids:
            node = memsom.get_node(conn, nid)
            print(
                f"  [{nid}] {node['channel']:<13}"
                f" integrity={memsom.NAME[node['label']]:<13}"
                f" {len(node['content']):>6} chars"
                f"  {node['source_ref'] or ''}"
            )
    except OSError as exc:
        print(f"[memsom-ingest] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def _cmd_ingest_dir(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        ids = ingest_dir(conn, args.dir, args.channel, glob=args.glob)
        print(
            f"ingested {len(ids)} node(s) from {args.dir}"
            f" [channel={args.channel}, glob={args.glob}]"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[memsom-ingest] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def _cmd_ingest_url(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        ids = ingest_url(conn, args.url)
        print(f"ingested {len(ids)} node(s) from {args.url} [channel=external]")
    except (urllib.error.URLError, OSError) as exc:
        print(f"[memsom-ingest] fetch failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def _cmd_ingest_text(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        try:
            ids = ingest_text(conn, args.text, args.channel, source_ref=args.ref)
        except ValueError as exc:
            print(f"[memsom-ingest] {exc}", file=sys.stderr)
            sys.exit(1)
        if not ids:
            print("[memsom-ingest] empty text - nothing stored", file=sys.stderr)
            sys.exit(1)
        print(f"ingested {len(ids)} node(s) [channel={args.channel}]")
        for nid in ids:
            node = memsom.get_node(conn, nid)
            print(
                f"  [{nid}] {node['channel']:<13}"
                f" integrity={memsom.NAME[node['label']]:<13}"
                f" {len(node['content']):>6} chars"
            )
    finally:
        conn.close()


def register(subparsers) -> None:
    """Mount ingest sub-commands onto an existing argparse subparsers object."""
    # ingest <path> --channel <c>
    p_ingest = subparsers.add_parser(
        "ingest", help="ingest a single file into the DAG"
    )
    p_ingest.add_argument("path", help="path to file")
    p_ingest.add_argument(
        "--channel",
        required=True,
        choices=list(memsom.RANK.keys()),
        help="channel to stamp the ingested node(s) with",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    # ingest-dir <dir> --channel <c> [--glob G]
    p_dir = subparsers.add_parser(
        "ingest-dir", help="ingest all matching files in a directory tree"
    )
    p_dir.add_argument("dir", help="root directory")
    p_dir.add_argument(
        "--channel",
        required=True,
        choices=list(memsom.RANK.keys()),
        help="channel to stamp the ingested node(s) with",
    )
    p_dir.add_argument(
        "--glob",
        default="*.md",
        help="file glob pattern (default: *.md)",
    )
    p_dir.set_defaults(func=_cmd_ingest_dir)

    # ingest-url <url>
    p_url = subparsers.add_parser(
        "ingest-url", help="fetch a URL and ingest the body (always external channel)"
    )
    p_url.add_argument("url", help="URL to fetch")
    p_url.set_defaults(func=_cmd_ingest_url)

    # ingest-text <text> --channel <c> [--ref R]
    p_txt = subparsers.add_parser(
        "ingest-text", help="ingest raw text directly (channel stamped by caller)"
    )
    p_txt.add_argument("text")
    p_txt.add_argument(
        "--channel",
        required=True,
        choices=list(memsom.RANK.keys()),
    )
    p_txt.add_argument("--ref", default=None, help="optional source reference")
    p_txt.set_defaults(func=_cmd_ingest_text)


def main(argv=None) -> None:
    """Thin CLI wrapper — for direct invocation."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom-ingest")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
