"""memsom.integrity.ingest -- the ONE stamping write path for the derivation DAG.

Moved out of interface/ingest.py (Phase 4, PLAN.md Sec1.4): this is the write
path proper -- ingest_text/ingest_file/ingest_dir, the F-13/F-14 caller-layer
trust guards, and the dedup/supersession lookups. It sits at rank 2
(integrity), which is where the security invariant belongs: every stamping
entry point in the package (CLI add/ingest, MCP ingest_text, the bridge
importer, chat ingestion, the federation broker's inline-ingest) must import
DOWN into this module rather than calling insert_node directly, so channel
enforcement (F-13/F-14) cannot be bypassed by a caller that forgot to route
through it (MS-20's actual mechanism: bridge_import.py used to call
insert_node straight, which is exactly the inverse of invariant 1 -- "labels
are assigned by channel, never by content").

SPINE invariant: channel is stamped by the ADAPTER (transport), NEVER inferred
from content. This module is that adapter -- the caller declares the channel,
and this code enforces it without peeking at content.

The retrieval index update used to be a try-wrapped upward import
(`from memsom.retrieval import retrieve ...` inside integrity, swallowing every
exception). It is now `kernel.events.emit("node_ingested", ...)` -- a rank-0
import, no swallow. retrieval/retrieve.py subscribes; a failure is visible to
whoever asks (lifecycle.heal's unindexed-source check), not hidden forever
(MS-31).

Schema migration: adds content_hash TEXT (nullable, no default) to nodes, plus
a covering index idx_nodes_content_hash. Existing rows get NULL -- safe for all
frozen-core behaviour (content_hash is never read by memsom.py).

Public API
----------
migrate(conn)
    Idempotent: add content_hash column + index.

ingest_text(conn, text, channel, source_ref=None, chunk=True, chunk_chars=1200)
    -> list[int]
    Split long text into ~chunk_chars chunks, dedup by content_hash+channel,
    insert_node the rest, emit "node_ingested" per new node.

ingest_file(conn, path, channel) -> list[int]
ingest_dir(conn, dirpath, channel, glob="*.md") -> list[int]
"""

import fnmatch
import os
import sqlite3
from pathlib import Path

import memsom
from memsom.kernel import chunking as memsom_chunking
from memsom.kernel import events as memsom_events
from memsom.storage import schema as memsom_schema

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_CHARS = 1200

# F-13: optional channel ceiling. The operator is the trust authority for this
# single-user tool, so the default (env unset) is permissive. When set, it caps
# the highest channel any stamping entry point (CLI add/ingest, MCP ingest_text)
# may declare -- e.g. MEMDAG_CHANNEL_CEILING=user disallows stamping `endorsed`
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
    from memsom.kernel.lattice import parse_rank
    n = parse_rank(raw.strip())
    if n is None:
        raise ValueError(
            f"invalid {CHANNEL_CEILING_ENV}={raw!r}: expected a channel name or 0-3"
        )
    return n


def enforce_channel_ceiling(channel: str) -> str:
    """F-13: reject *channel* if a ceiling is configured and the channel exceeds it.

    Default (no ceiling configured) is permissive -- returns the channel unchanged.
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


#: `source_ref` prefix the bridge importer owns exclusively.
BRIDGE_NAMESPACE = "memory:"


def enforce_source_ref_namespace(source_ref) -> None:
    """Refuse a caller-declared ``source_ref`` in the bridge's own namespace.

    ``memory:<stem>`` is the identity the bridge importer mints for a file in
    the memory directory, and ``memory:literal:<hash>`` for a line in the index.
    Those two prefixes are what the digest reads to build the ALWAYS-LOADED
    MEMORY.md, and what the two reconcile sweeps use to take an entry back out
    again when its file or its index line disappears.

    A caller who declares that prefix here mints an entry with the bridge's
    authority and none of its lifecycle: ``insert_node`` leaves ``bridge_path``
    NULL, so ``import_memory_dir``'s sweep (which requires ``bridge_path IS NOT
    NULL``) cannot see it, and without the ``literal:`` infix neither can
    ``import_literals``. Stamped ``endorsed`` it is also pinned, so the byte
    budget never sheds it. The result is a permanent line in the brain, on every
    machine the store replicates to, with no file on disk for a human to notice.

    That is reachable from a tool call: MCP ``ingest_text`` takes both
    ``channel`` and ``source_ref`` straight from its arguments, so one call from
    a model reading an attacker's page is enough. This function is the door;
    ``digest._rows`` carries the structural half (it renders only nodes a sweep
    can also reach), because a door only helps at the doors somebody remembered.
    """
    if source_ref and str(source_ref).strip().lower().startswith(BRIDGE_NAMESPACE):
        raise ValueError(
            f"source_ref {str(source_ref)!r} claims the reserved "
            f"{BRIDGE_NAMESPACE!r} namespace, which the memory-directory bridge "
            f"importer owns. Entries under that prefix render into the "
            f"always-loaded index and are reconciled against files on disk; one "
            f"minted here would have no file, so no sweep could ever remove it. "
            f"Write the memory to a file in the memory directory and let the "
            f"bridge import it, or choose a different source_ref."
        )

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


def _find_live_by_hash(conn: sqlite3.Connection, h: str, channel: str):
    """Return the id of an UNTAINTED, SAME-CHANNEL node with content_hash == h.

    INGEST-1/2: dedup must not reuse a redacted, quarantined, or archived node.
    A redacted node keeps its content_hash (content was zeroed, not the hash), and
    a quarantined node is excluded from the untainted pool -- deduping onto either
    silently dropped the freshly-supplied content and/or handed back an excluded
    node. Source the WHERE fragment from the shared taint filter so this dedup
    path inherits every taint dimension the read pools already enforce.

    INGEST-DEDUP-CHANNEL / CHATS-1-DEDUP-LAUNDER: dedup must also match on CHANNEL.
    Otherwise identical bytes ingested under a DIFFERENT channel silently reuse the
    existing node -- an endorsed ingest returning a lower-integrity external node,
    or an assistant turn (agent-derived) laundered onto an identical user node,
    defeating the channel->label invariant. A cross-channel match mints a fresh
    node with the caller's declared channel.

    MS-19: this is also the path a duplicate note steals another note's identity
    through -- see enforce_no_path_steal below, which the bridge/vault stampers
    call after this returns a reused id.
    """
    # clearance=3 (topsecret) explicit: this is an existence/dedup check, not a
    # reader-facing exposure -- a topsecret-classified node's hash must still be
    # found so re-ingesting identical content doesn't mint a duplicate.
    clauses, params = memsom_schema.taint_filter_clauses(conn, clearance=3)
    row = conn.execute(
        "SELECT id FROM nodes WHERE content_hash = ? AND channel = ? AND "
        + " AND ".join(clauses) + " LIMIT 1",
        [h, channel] + params,
    ).fetchone()
    return row[0] if row else None


def enforce_no_path_steal(conn: sqlite3.Connection, node_id: int, path_col: str,
                          new_path: str) -> None:
    """MS-19: refuse to stamp *new_path* onto *node_id* if it already carries a
    DIFFERENT path in *path_col* (``bridge_path`` or ``obsidian_path``).

    Dedup returns an EXISTING node id on a content_hash+channel match. If the
    caller then unconditionally stamps its own path onto that id, a duplicate
    note whose body happens to match a trusted note's hash steals the trusted
    note's identity: supersession, the vault prune pass, and the reconcile sweep
    all key on that path column, so a later edit/retraction of the ORIGINAL note
    resolves against the WRONG node and revokes nothing. First path wins; a
    reused node with a foreign path already stamped is left alone by the caller
    (it must mint a fresh node instead). No-op when the column is absent, empty,
    or already equal to *new_path*.
    """
    if not memsom_schema.column_exists(conn, "nodes", path_col):
        return
    row = conn.execute(
        f"SELECT {path_col} FROM nodes WHERE id = ?", (node_id,)).fetchone()
    existing = row[0] if row else None
    if existing and existing != new_path:
        raise ValueError(
            f"node {node_id} already carries {path_col}={existing!r}; refusing "
            f"to steal its identity by re-stamping {new_path!r} (MS-19: a "
            f"content-hash dedup match is not the same note)"
        )


def _try_index(conn: sqlite3.Connection, nid: int) -> list:
    """Emit "node_ingested" for *nid*. Returns the list of subscriber failures
    (empty on success or when nothing subscribes -- retrieval indexing is
    optional and may never have been imported in this process).

    kernel.events never swallows, so a failure here is visible to whoever calls
    this -- ingest_text does not itself decide it is fatal (indexing is best-
    effort by design), but it no longer disappears: lifecycle.heal's
    unindexed-source check (MS-31) finds any live source with no postings row,
    independent of whether this call ever ran.
    """
    return memsom_events.emit("node_ingested", conn=conn, node_id=nid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _find_live_predecessor(conn: sqlite3.Connection, source_ref: str, channel: str,
                           new_hash: str):
    """Return the LIVE, same-channel node with the SAME source_ref but a DIFFERENT
    content_hash -- the prior version this re-ingest supersedes (newest if several).

    Sources its liveness WHERE-fragment from the shared taint filter (same
    discipline as _find_live_by_hash), so it never selects a tombstoned / redacted
    / quarantined / archived node -- a re-ingest cannot "supersede" an already-dead
    version. Returns None when source_ref is empty or no prior version exists.
    """
    if not source_ref:
        return None
    # clearance=3 (topsecret) explicit: same reasoning as _find_live_by_hash --
    # a topsecret-classified predecessor must still be found so the supersede
    # link and staleness cascade fire regardless of its confidentiality tier.
    clauses, params = memsom_schema.taint_filter_clauses(conn, clearance=3)
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
    list[int] -- node ids for each chunk, in order (may include reused ids)
    """
    migrate(conn)

    # Caller-layer trust guards: refuse an over-ceiling channel (F-13) and pin the
    # integrity label to the channel (F-14) -- never trust a caller-supplied label.
    # Then refuse a source_ref claiming the bridge importer's namespace: this is
    # the only entry point where the REF is caller-declared as well as the
    # channel, and the pair is what mints an un-sweepable line in the brain.
    enforce_channel_ceiling(channel)
    enforce_source_ref_namespace(source_ref)
    label = authoritative_label(channel)

    if not chunk or len(text) <= chunk_chars:
        chunks = [text.strip()] if text.strip() else []
    else:
        chunks = memsom_chunking.split_chunks(text, chunk_chars)

    if not chunks:
        # Nothing to store
        return []

    ids = []
    base_ref = source_ref or ""

    for i, chunk_text in enumerate(chunks):
        h = memsom_chunking.content_hash(chunk_text)

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

        # Contradiction auto-detect used to live here as a try-wrapped upward
        # import (integrity -> lifecycle.contradict), gated on $MEMDAG_CONTRADICT
        # BEFORE even attempting it. It is folded into the SAME "node_ingested"
        # emit above now: lifecycle.contradict subscribes to that event and
        # checks its own enabled()/_enforce_default() internally, so the
        # env-gate moved with it instead of living at the call site. kernel.
        # events.emit already isolates one subscriber's failure from the
        # others (and from ingest_text itself), so this stays exactly as
        # best-effort as it was.

        # Staleness auto-detect (single-chunk only -- multi-chunk has no clean
        # chunk-to-chunk supersession alignment; those use the manual stale-cascade
        # verb). If this re-ingest replaces a prior LIVE version of the SAME source,
        # record old->new and fire the staleness cascade. Best-effort: a failure
        # here must never break ingest (the new node is already committed).
        if len(chunks) == 1 and source_ref:
            pred = _find_live_predecessor(conn, source_ref, channel, h)
            if pred is not None and pred != nid:
                try:
                    from memsom.integrity import stale as memsom_stale
                    memsom_stale.on_reingest_supersede(conn, pred, nid, source_ref)
                except Exception:  # noqa: BLE001 -- staleness is best-effort
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
