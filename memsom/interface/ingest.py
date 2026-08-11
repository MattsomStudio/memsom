"""memsom_ingest -- CLI surface + facade over the ingest write path.

PHASE 4 (PLAN.md Sec1.4): the write path itself (ingest_text/ingest_file/
ingest_dir, the F-13/F-14 guards, the dedup lookups) moved to
memsom.integrity.ingest -- rank 2, where the security invariant belongs, so
every OTHER stamping caller in the package (bridge, chats, federation broker)
imports it from there instead of reaching up into this rank-8 module. The pure
chunk-splitting helpers moved to memsom.kernel.chunking -- rank 0.

This module is now: the CLI subcommands (ingest / ingest-dir / ingest-url /
ingest-text), `ingest_url` (a network fetch that then calls the write path --
not itself security-sensitive the way F-13/F-14 are), and a facade re-export
of every symbol this module used to own directly, so `memsom.interface.ingest.X`
keeps working for every existing caller and test unchanged (the same discipline
memsom/__init__.py's Phase 2 facade uses).
"""

import argparse
import sys

import memsom
from memsom.effects import net as memsom_net
from memsom.integrity.ingest import (  # noqa: F401 -- facade re-export
    CHANNEL_CEILING_ENV,
    BRIDGE_NAMESPACE,
    authoritative_label,
    channel_ceiling,
    enforce_channel_ceiling,
    enforce_source_ref_namespace,
    enforce_no_path_steal,
    migrate,
    ingest_text,
    ingest_file,
    ingest_dir,
    _find_live_by_hash,
    _find_live_predecessor,
    _try_index,
    _DEFAULT_CHUNK_CHARS,
)
from memsom.kernel.chunking import (  # noqa: F401 -- facade re-export
    normalize as _normalize,
    content_hash as _content_hash,
    split_chunks as _split_chunks,
)


# ---------------------------------------------------------------------------
# ingest_url -- network fetch + call into the write path
# ---------------------------------------------------------------------------


def ingest_url(conn, url: str) -> list:
    """Fetch *url* (GET) and ingest the response body.

    Channel is ALWAYS forced to "external" -- the transport dictates the channel,
    never the content (SPINE invariant).
    source_ref is set to the URL.

    Returns list[int] of node ids.
    Raises memsom.effects.net.NetworkError on network/HTTP failure.
    """
    raw = memsom_net.fetch(url, headers={"User-Agent": "memsom-ingest/0.1"}, timeout=15)

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
    from pathlib import Path
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
    except memsom_net.NetworkError as exc:
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
    """Thin CLI wrapper -- for direct invocation."""
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
