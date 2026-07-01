"""memsom_obsidian — optional Obsidian-vault integration for the derivation DAG.

An Obsidian vault is already a graph: every ``[[wikilink]]`` is an edge.  This
module maps that graph onto memsom's edge layer instead of slurping notes as
flat text (which is all ``ingest-dir`` does).  It mirrors a vault both ways —
read notes in (nodes + ``relates-to`` edges from wikilinks), write answers back
out as notes — and an optional polling watcher keeps the import live.

Everything here is OFF by default: nothing runs unless a subcommand is invoked.
Pure stdlib (no PyYAML, no watchdog) to match the rest of the repo.

SECURITY — channel is stamped by PROVENANCE, never by "it is a file in the vault".
    The write-back path is a self-inflicted laundering vector: memsom writes an
    answer to the vault, the watcher sees a .md file, and a naive reader would
    re-ingest it as `user` — promoting agent-derived content one integrity tier
    through the filesystem (the exact attack Biba-min() exists to kill).  Guard:
    a note may carry `memsom-channel:` in frontmatter, but it can only ever
    *lower* integrity, never raise it:  effective = min(default, declared).
    So a memsom-authored note (stamped agent-derived) re-enters as agent-derived,
    and a hand-edited / synced note claiming `endorsed` is clamped to the
    configured default — no upward forge.  All actual writes go through
    memsom_ingest.ingest_text, so F-13 (channel ceiling) and F-14 (channel->label
    lock) still hold.

Public API
----------
migrate(conn)
    Idempotent: add obsidian_path / obsidian_mtime columns (+ chain ingest/relate).

parse_frontmatter(text) -> (dict, body)
    Stdlib YAML-subset parser for the leading ``---`` block.  body excludes it.

extract_links(body, frontmatter=None) -> list[str]
    Wikilink + embed + markdown-link targets, code/comments masked out.

sync_vault(conn, vault, default_channel="user", prune=True) -> dict
    The one-shot engine: walk -> ingest changed notes -> map wikilinks to
    relate() edges -> tombstone deleted notes.  Returns a counts summary.

export_note(conn, vault, *, node_id=None, query=None, folder="memsom",
            title=None, clearance="topsecret") -> Path
    Write an answer back to the vault as a memsom-stamped note (refuses to
    overwrite a note memsom did not author).

watch_vault(conn_factory, vault, default_channel="user", interval=1.0,
            debounce=0.4, prune=True) -> None
    Thin polling trigger over sync_vault.  Owns NO sync logic.

CLI sub-commands (via register(subparsers))
-------------------------------------------
obsidian-sync   <vault> [--channel C] [--no-prune]
obsidian-export [node] [--query Q] [--vault V] [--folder F] [--title T]
obsidian-watch  <vault> [--channel C] [--interval S] [--debounce S] [--no-prune]
"""

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path, PurePosixPath

import memsom
import memsom_schema
import memsom_ingest
import memsom_relate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default vault location (env-configurable; CLI arg overrides). No hardcoded
# personal path ships in the repo — keep the pre-public genericization intact.
VAULT_ENV = "MEMDAG_OBSIDIAN_VAULT"

# Frontmatter key a note carries to declare its provenance channel. Honored
# ONLY downward (see module docstring).
CHANNEL_KEY = "memsom-channel"
AUTHORED_KEY = "memsom-authored"
SOURCES_KEY = "memsom-source-nodes"

# Directories never walked, and transient editor files never read.
_IGNORE_DIRS = {".obsidian", ".trash", ".git", ".sync", ".stfolder"}
_TEMP_PATTERNS = (re.compile(r"^\..*\.tmp$"), re.compile(r".*~$"),
                  re.compile(r"^\.#"), re.compile(r".*\.tmp$"))

# Link kind stamped on rel_edges minted from wikilinks (distinct from the
# generic 'relates-to' so a vault edge is identifiable later).
LINK_KIND = "wikilink"

# Skip notes larger than this (a hostile/mis-saved multi-GB ".md" would
# otherwise be read fully into memory). Generous — real notes are tiny.
MAX_NOTE_BYTES = 8 * 1024 * 1024


def _within(base: Path, target: Path) -> bool:
    """True if *target* is inside *base*. Cross-drive (Windows) -> False, never raises."""
    try:
        return os.path.commonpath([str(base), str(target)]) == str(base)
    except ValueError:
        return False  # different drives / un-comparable -> treat as outside


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate(conn) -> None:
    """Idempotent: add obsidian columns; chain ingest + relate migrations.

    obsidian_path  : vault-relative POSIX path of the source note (note->node
                     identity for incremental sync; a note may chunk to several
                     nodes that all share this path).
    obsidian_mtime : last-seen change-detection signature as TEXT, of the form
                     "<mtime_ns>:<size>:<sha256(text)>" — CONTENT-derived, so an
                     mtime-preserving content swap cannot masquerade as unchanged.
    Both default NULL; the frozen core never reads them.
    """
    memsom_ingest.migrate(conn)
    memsom_relate.migrate(conn)
    memsom_schema.add_column(conn, "nodes", "obsidian_path", "TEXT")
    memsom_schema.add_column(conn, "nodes", "obsidian_mtime", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_obsidian_path ON nodes(obsidian_path)"
    )


# ---------------------------------------------------------------------------
# Channel discipline (the laundering / upward-forge guard)
# ---------------------------------------------------------------------------


def effective_channel(default_channel: str, declared) -> str:
    """Return the channel a note's content is stamped with.

    A frontmatter-declared channel may only LOWER integrity, never raise it:
    effective rank = min(RANK[default], RANK[declared]).  An unknown/garbage
    declared value is ignored (falls back to default).  This single rule closes
    both the write->re-ingest laundering loop and the upward-forge a malicious
    vault note would otherwise attempt.
    """
    if default_channel not in memsom.RANK:
        raise ValueError(f"unknown default channel: {default_channel!r}")
    if not declared:
        return default_channel
    dkey = str(declared).strip().lower()
    if dkey not in memsom.RANK:
        return default_channel  # garbage declaration -> ignore, use default
    rank = min(memsom.RANK[default_channel], memsom.RANK[dkey])
    return memsom.NAME[rank].lower()


# ---------------------------------------------------------------------------
# Frontmatter (YAML subset, stdlib only)
# ---------------------------------------------------------------------------

_FM_LIST_ITEM = re.compile(r"^\s*-\s+(.*)$")
_FM_KV = re.compile(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$")


def _strip_scalar(v: str):
    """Strip surrounding quotes from a YAML scalar; return the bare string."""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def parse_frontmatter(text: str):
    """Parse a leading ``---`` YAML block. Return (frontmatter_dict, body).

    Recognizes ``key: scalar``, ``key: [a, b]`` inline lists, and block lists
    (``key:`` then ``  - item`` lines). Values keep raw strings; lists -> list.
    If there is no valid frontmatter (no ``---`` on line 1), returns ({}, text).
    Only a tiny, predictable subset — enough for tags/aliases/memsom-* keys.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    # Find the closing fence.
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end is None:
        return {}, text  # unterminated -> treat whole file as body

    fm = {}
    cur_key = None
    for raw in lines[1:end]:
        if not raw.strip():
            continue
        m_item = _FM_LIST_ITEM.match(raw)
        if m_item and cur_key is not None:
            # Only an EMPTY scalar ("") preceding '- ' lines becomes a block list.
            # Promoting on any non-list value would silently discard a real scalar
            # (e.g. malformed `tags: foo` then `- bar` must not drop "foo").
            if fm.get(cur_key) == "":
                fm[cur_key] = []
            if isinstance(fm.get(cur_key), list):
                fm[cur_key].append(_strip_scalar(m_item.group(1)))
            continue
        m_kv = _FM_KV.match(raw)
        if m_kv:
            key = m_kv.group(1)
            val = m_kv.group(2).strip()
            cur_key = key
            if val == "":
                fm[key] = ""  # may become a block list on following '- ' lines
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                fm[key] = [_strip_scalar(x) for x in inner.split(",") if x.strip()] if inner else []
            else:
                fm[key] = _strip_scalar(val)

    body = "\n".join(lines[end + 1:])
    return fm, body


# ---------------------------------------------------------------------------
# Link extraction — mask code/comments FIRST, then match
# ---------------------------------------------------------------------------

_WIKILINK = re.compile(r"(!?)\[\[([^\[\]]+?)\]\]")
_MDLINK = re.compile(r"(!?)\[[^\]]*\]\(([^)]+)\)")


def _mask_inline_code(line: str) -> str:
    """Blank inline code spans (matched backtick runs) on a single line."""
    out = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "`":
            j = i
            while j < n and line[j] == "`":
                j += 1
            run = j - i  # opening run length
            # find a closing run of the SAME length
            k = j
            while k < n:
                if line[k] == "`":
                    s = k
                    while k < n and line[k] == "`":
                        k += 1
                    if k - s == run:
                        out.append(" " * (k - i))  # blank open..close inclusive
                        i = k
                        break
                else:
                    k += 1
            else:
                out.append(line[i:])  # no close -> keep remainder verbatim
                i = n
        else:
            out.append(line[i])
            i += 1
    return "".join(out)


def _mask(text: str) -> str:
    """Blank fenced code, inline code, HTML comments, and escaped brackets.

    Replaces masked regions so the link regexes never see example/code links.
    Length is not preserved exactly (not needed — we only run regex afterward).
    """
    # HTML comments span lines -> mask on whole text first.
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Escaped brackets never start a link.
    text = text.replace("\\[", "  ").replace("\\]", "  ")

    out_lines = []
    fence = None  # (char, length) of the currently-open fence, else None
    for raw in text.splitlines():
        stripped = raw.lstrip()
        m = re.match(r"(`{3,}|~{3,})", stripped)
        if fence is None and m:
            fence = (m.group(1)[0], len(m.group(1)))
            out_lines.append("")  # blank the opening fence line
            continue
        if fence is not None:
            # Inside a fence: blank everything; close on a matching-or-longer fence.
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1]:
                fence = None
            out_lines.append("")
            continue
        # Indented code block (4+ spaces) -> blank.
        if raw.startswith("    ") or raw.startswith("\t"):
            out_lines.append("")
            continue
        out_lines.append(_mask_inline_code(raw))
    return "\n".join(out_lines)


def _wikilink_target(inner: str):
    """Resolve a wikilink's inner text to a note target, or None to skip.

    Order inside ``[[ ... ]]`` is ``target #heading ^block | alias``.  Strip the
    alias (after first '|') then the heading/block (after first '#').  A leading
    '#' (same-file anchor) yields no note target -> None.
    """
    inner = inner.split("|", 1)[0]           # drop alias
    inner = inner.split("#", 1)[0]           # drop heading / block anchor
    inner = inner.strip()
    return inner or None                     # '' => same-file anchor => skip


def extract_links(body: str, frontmatter=None):
    """Return note-link targets found in *body* (+ quoted links in frontmatter).

    Targets are raw strings (note name or vault path, no .md normalization yet).
    Embeds ``![[x]]`` count as links. Markdown links to external URLs / pure
    anchors are dropped. Duplicates removed, original order preserved.
    """
    targets = []
    seen = set()

    def _add(t):
        if t and t not in seen:
            seen.add(t)
            targets.append(t)

    masked = _mask(body)

    for _bang, inner in _WIKILINK.findall(masked):
        t = _wikilink_target(inner)
        if t:
            _add(t)

    for _bang, raw_target in _MDLINK.findall(masked):
        # A markdown target is `URL` or `URL "title"`; the real URL holds no
        # literal spaces (they are %20), so split the title off BEFORE unquoting.
        url = raw_target.strip().split(None, 1)[0] if raw_target.strip() else ""
        tgt = urllib.parse.unquote(url)
        if not tgt or tgt.startswith("#") or "://" in tgt or tgt.startswith("mailto:"):
            continue
        tgt = tgt.split("#", 1)[0]  # drop in-file anchor
        if tgt:
            _add(tgt)

    # Native Obsidian supports quoted wikilinks inside frontmatter values.
    if frontmatter:
        for v in frontmatter.values():
            vals = v if isinstance(v, list) else [v]
            for item in vals:
                if isinstance(item, str):
                    for _bang, inner in _WIKILINK.findall(item):
                        t = _wikilink_target(inner)
                        if t:
                            _add(t)

    return targets


# ---------------------------------------------------------------------------
# Vault walking + link resolution
# ---------------------------------------------------------------------------


def _is_temp(name: str) -> bool:
    return any(p.match(name) for p in _TEMP_PATTERNS)


def _walk_markdown(vault: Path):
    """Yield (relpath_posix, abs_path, st_mtime_ns, st_size) for live .md files.

    os.walk does not follow symlinked directories (followlinks=False), but a
    symlinked .md FILE inside the vault could still point OUTSIDE it — reading
    that would ingest external content as a vault note. Each file's real path is
    asserted to stay under the vault before it is yielded (matters for a
    Syncthing-shared vault where a peer can plant links).
    """
    vault = Path(vault)
    vroot = vault.resolve()
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
        for fname in sorted(files):
            if not fname.lower().endswith(".md") or _is_temp(fname):
                continue
            ap = Path(root) / fname
            try:
                if not _within(vroot, ap.resolve()):
                    continue  # symlink (or junction) escaping the vault -> skip
                st = ap.stat()
            except OSError:
                continue
            rel = PurePosixPath(ap.relative_to(vault).as_posix())
            yield str(rel), ap, st.st_mtime_ns, st.st_size


def _build_resolver(rel_paths):
    """Index vault-relative paths for link resolution.

    Returns (by_name, by_relpath) where keys are lowercased and '.md' stripped.
    by_name maps a bare basename -> list of relpaths (>1 == ambiguous).
    """
    by_name = {}
    by_relpath = {}
    for rel in rel_paths:
        norm = rel[:-3].lower() if rel.lower().endswith(".md") else rel.lower()
        by_relpath[norm] = rel
        base = norm.rsplit("/", 1)[-1]
        by_name.setdefault(base, []).append(rel)
    return by_name, by_relpath


def _resolve_target(target: str, by_name, by_relpath):
    """Resolve a link target string to a vault relpath, or None if unresolved.

    Shortest-path-when-unique: a bare name resolves only if its basename is
    unique; a path-qualified target resolves against the full relpath. Matching
    is case-insensitive, with or without the .md extension.
    """
    norm = target.replace("\\", "/").strip().lower()
    if norm.endswith(".md"):
        norm = norm[:-3]
    if not norm:
        return None
    if "/" in norm:
        return by_relpath.get(norm)
    hits = by_name.get(norm)
    if hits and len(hits) == 1:
        return hits[0]
    return None  # missing or ambiguous -> no edge (logged by caller)


# ---------------------------------------------------------------------------
# DB helpers for note <-> node identity
# ---------------------------------------------------------------------------


def _live_nodes_for_path(conn, rel: str):
    rows = conn.execute(
        "SELECT id FROM nodes WHERE obsidian_path = ? AND tombstoned = 0 ORDER BY id",
        (rel,),
    ).fetchall()
    return [r[0] for r in rows]


def _stored_mtime(conn, rel: str):
    row = conn.execute(
        "SELECT obsidian_mtime FROM nodes WHERE obsidian_path = ? AND tombstoned = 0 LIMIT 1",
        (rel,),
    ).fetchone()
    return row[0] if row else None


def _all_known_paths(conn):
    rows = conn.execute(
        "SELECT DISTINCT obsidian_path FROM nodes "
        "WHERE obsidian_path IS NOT NULL AND tombstoned = 0"
    ).fetchall()
    return [r[0] for r in rows]


def _representative_node(conn, rel: str):
    nodes = _live_nodes_for_path(conn, rel)
    return nodes[0] if nodes else None


# ---------------------------------------------------------------------------
# The one-shot sync engine
# ---------------------------------------------------------------------------


def sync_vault(conn, vault, default_channel="user", prune=True, log=None):
    """Mirror *vault* into the DAG (the engine the watcher also calls).

    Returns a summary dict: notes, ingested, unchanged, deleted, edges, unresolved.
    """
    migrate(conn)
    if default_channel not in memsom.RANK:
        raise ValueError(f"unknown channel: {default_channel!r}")
    vault = Path(vault)
    if not vault.is_dir():
        raise ValueError(f"vault is not a directory: {vault}")
    say = log or (lambda *_a, **_k: None)

    files = list(_walk_markdown(vault))
    present = {rel for rel, *_ in files}
    summary = {"notes": len(files), "ingested": 0, "unchanged": 0,
               "deleted": 0, "edges": 0, "unresolved": 0}

    # --- Pass 1: ingest changed/new notes (skip unchanged via mtime signature) ---
    note_links = {}  # rel -> list[str] targets (for pass 2)
    for rel, ap, mtime_ns, size in files:
        if size > MAX_NOTE_BYTES:
            say(f"  ! skip oversized {rel} ({size} bytes > {MAX_NOTE_BYTES})")
            continue
        try:
            text = ap.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            say(f"  ! skip unreadable {rel}: {exc}")
            continue

        fm, body = parse_frontmatter(text)
        note_links[rel] = extract_links(body, fm)

        # Change-detection signature is CONTENT-derived, not mtime-only: mtime is
        # attacker-controllable (os.utime), so an mtime-preserving content swap
        # would otherwise slip past as "unchanged" and leave stale/poisoned data
        # live — a real risk in a Syncthing-shared vault. The note is read every
        # pass regardless, so hashing it is effectively free.
        sig = f"{mtime_ns}:{size}:{hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest()}"
        if _stored_mtime(conn, rel) == sig and _live_nodes_for_path(conn, rel):
            summary["unchanged"] += 1
            continue

        # Changed (or new): tombstone any prior nodes for this path, then re-ingest.
        for nid in _live_nodes_for_path(conn, rel):
            memsom.revoke_cascade(conn, nid, f"obsidian: note changed ({rel})")

        channel = effective_channel(default_channel, fm.get(CHANNEL_KEY))
        ids = memsom_ingest.ingest_text(conn, body, channel, source_ref=rel)
        if ids:
            qmarks = ",".join("?" * len(ids))
            with conn:
                conn.execute(
                    f"UPDATE nodes SET obsidian_path = ?, obsidian_mtime = ? "
                    f"WHERE id IN ({qmarks})",
                    [rel, sig] + ids,
                )
            summary["ingested"] += 1
            say(f"  + {rel} [{channel}] -> {len(ids)} node(s)")

    # --- Pass 2: map wikilinks to relate() edges (reps resolved after ingest) ---
    by_name, by_relpath = _build_resolver(present)
    for rel, targets in note_links.items():
        src = _representative_node(conn, rel)
        if src is None:
            continue
        for tgt in targets:
            tgt_rel = _resolve_target(tgt, by_name, by_relpath)
            if tgt_rel is None:
                summary["unresolved"] += 1
                continue
            dst = _representative_node(conn, tgt_rel)
            if dst is None or dst == src:
                continue
            try:
                memsom_relate.relate(conn, src, dst, kind=LINK_KIND)
                summary["edges"] += 1
            except ValueError:
                pass  # node vanished mid-sync; harmless

    # --- Pass 3: prune notes deleted from the vault (tombstone, never hard-delete) ---
    if prune:
        for rel in _all_known_paths(conn):
            if rel in present:
                continue
            # Distinguish a genuine deletion from a transient walk/stat error:
            # a tombstone cascades to derived children, so only prune when the
            # file is verifiably gone from disk (a recovered read error must not
            # tombstone a live note + its answers).
            if (vault / rel).exists():
                continue
            for nid in _live_nodes_for_path(conn, rel):
                memsom.revoke_cascade(conn, nid, f"obsidian: note deleted ({rel})")
            summary["deleted"] += 1
            say(f"  - {rel} (deleted from vault)")

    return summary


# ---------------------------------------------------------------------------
# Write-back: export an answer to the vault as a memsom-stamped note
# ---------------------------------------------------------------------------

_SLUG = re.compile(r"[^A-Za-z0-9]+")


def _slugify(s: str) -> str:
    s = _SLUG.sub("-", s).strip("-")
    return (s[:80] or "memsom-note")


_CITATION = re.compile(r"[ \t]*\[mem:\d+\|[^\]]*\]")


def _compose_answer(conn, query: str, clearance: str):
    """Compose a clean answer to *query*. Return (body, source_ids).

    Uses the same deterministic pool + compose path as the enhanced `ask`
    (so the answer matches what `ask` would give) but WITHOUT the CLI furniture
    ("Q:", "stored as node", "floor:", the profile block).  Inline ``[mem:N|ch]``
    citation tags are stripped from the prose — provenance is rendered instead
    as wikilink backlinks in a Sources section — and the REAL cited source node
    ids are returned (so memsom-source-nodes and the backlinks are populated).
    """
    import memsom_cli  # lazy: memsom_cli imports this module at its top level
    pool = memsom_cli._build_pool(conn, clearance)
    if not pool:
        return "", []
    text, used = memsom.compose(query, pool)
    if not text:
        return "", []
    # compose() prepends two header lines ("Q: ..." / "A (composed from N ...):")
    # before the answer bullets; drop them by prefix, then strip [mem:] tags.
    kept = [ln for ln in text.splitlines()
            if not ln.startswith("Q: ") and not ln.startswith("A (composed from ")]
    clean = _CITATION.sub("", "\n".join(kept)).strip()
    return clean, list(used)


def _note_is_memsom_authored(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    fm, _ = parse_frontmatter(text)
    val = str(fm.get(AUTHORED_KEY, "")).strip().lower()
    # Require BOTH the authored flag AND the source-nodes key memsom always
    # writes, so a hand-written note that merely happens to carry
    # `memsom-authored: true` is not silently overwritable.
    return val in ("true", "yes", "1") and SOURCES_KEY in fm


def export_note(conn, vault, *, node_id=None, query=None, folder="memsom",
                title=None, clearance="topsecret"):
    """Write an answer back into the vault as an agent-derived, memsom-stamped note.

    Exactly one of node_id / query must be given.  Returns the written Path.
    Refuses to write outside the vault or to overwrite a note memsom did not
    author.  The `memsom-channel: agent-derived` stamp is what lets the reader
    re-ingest this note safely (effective_channel clamps it, never `user`).
    """
    migrate(conn)
    if (node_id is None) == (query is None):
        raise ValueError("provide exactly one of node_id or query")
    vault = Path(vault).resolve()
    if not vault.is_dir():
        raise ValueError(f"vault is not a directory: {vault}")

    sources = []
    if node_id is not None:
        node = memsom.get_node(conn, node_id)
        if node is None or node["tombstoned"]:
            raise ValueError(f"node {node_id} is unknown or tombstoned")
        content = node["content"]
        sources = [node_id]
        title = title or f"memsom node {node_id}"
    else:
        content, sources = _compose_answer(conn, query, clearance)
        if not content:
            content = "(no answer composed — no live sources matched)"
        title = title or query

    # Containment: the resolved target must stay inside the vault. _within never
    # raises (cross-drive paths on Windows return False rather than ValueError).
    out_dir = (vault / folder).resolve()
    if not _within(vault, out_dir):
        raise ValueError("folder escapes the vault")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{_slugify(title)}.md").resolve()
    if not _within(vault, out_path):
        raise ValueError("output path escapes the vault")

    if out_path.exists() and not _note_is_memsom_authored(out_path):
        raise ValueError(
            f"refusing to overwrite a note memsom did not author: {out_path}"
        )

    # Provenance back-links: any source note with an obsidian_path links home.
    links = []
    for sid in sources:
        row = conn.execute(
            "SELECT obsidian_path FROM nodes WHERE id = ?", (sid,)
        ).fetchone()
        if row and row[0]:
            stem = PurePosixPath(row[0]).stem
            links.append(f"[[{stem}]]")

    src_list = ", ".join(str(s) for s in sources) if sources else ""
    fm_lines = [
        "---",
        f"{CHANNEL_KEY}: agent-derived",
        f"{AUTHORED_KEY}: true",
        f"{SOURCES_KEY}: [{src_list}]",
        f"created: {memsom.now_iso()}",
        "---",
        "",
    ]
    body = [f"# {title}", "", content, ""]
    if links:
        body += ["", "## Sources", "", *(f"- {l}" for l in links), ""]
    out_path.write_text("\n".join(fm_lines + body), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Watcher: thin polling trigger over sync_vault (owns NO sync logic)
# ---------------------------------------------------------------------------


def _snapshot(vault: Path):
    """Return {relpath: (st_mtime_ns, st_size)} for live .md files."""
    snap = {}
    for rel, _ap, mtime_ns, size in _walk_markdown(vault):
        snap[rel] = (mtime_ns, size)
    return snap


def watch_vault(conn_factory, vault, default_channel="user", interval=1.0,
                debounce=0.4, prune=True, log=None, _max_ticks=None):
    """Poll *vault*; on a settled change, run one full sync_vault().

    A change is acted on only once its (mtime_ns, size) signature has held
    steady for >= debounce seconds — which defeats half-written / non-atomic
    editor saves.  Renames surface as delete+create and are reconciled by the
    global sync.  Ctrl-C exits cleanly.  st_mtime_ns (int) is used throughout —
    never the float st_mtime, which loses sub-microsecond precision.

    _max_ticks bounds the loop for tests (None = run forever).
    """
    say = log or (lambda *_a, **_k: None)
    vault = Path(vault)
    # Initial sync: reconcile the vault's current state before watching for deltas
    # (a watcher that ignored pre-existing notes would feel broken on startup).
    conn = conn_factory()
    try:
        say(f"  initial: {sync_vault(conn, vault, default_channel, prune=prune, log=say)}")
    finally:
        conn.close()
    prev = _snapshot(vault)
    pending = {}  # rel -> (signature, last_change_monotonic). sig None == absent.
    say(f"watching {vault} (interval={interval}s, debounce={debounce}s) — Ctrl-C to stop")
    ticks = 0
    try:
        while _max_ticks is None or ticks < _max_ticks:
            ticks += 1
            time.sleep(interval)
            curr = _snapshot(vault)
            now = time.monotonic()

            # Any path whose signature differs from last poll (re)starts its timer.
            for rel in set(prev) | set(curr):
                sig = curr.get(rel)
                if sig != prev.get(rel):
                    pending[rel] = (sig, now)

            # A pending path settles when its signature matches the current poll
            # AND has been quiet for >= debounce.
            settled = [
                rel for rel, (sig, t) in pending.items()
                if curr.get(rel) == sig and (now - t) >= debounce
            ]
            if settled:
                for rel in settled:
                    pending.pop(rel, None)
                conn = conn_factory()
                try:
                    summary = sync_vault(conn, vault, default_channel,
                                         prune=prune, log=say)
                finally:
                    conn.close()
                say(f"  synced: {summary}")

            prev = curr
    except KeyboardInterrupt:
        say("\nstopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_vault(arg):
    v = arg or os.environ.get(VAULT_ENV)
    if not v:
        raise SystemExit(
            f"no vault given (pass a path or set {VAULT_ENV})"
        )
    return v


def _cmd_sync(args) -> None:
    conn = memsom.get_connection()
    try:
        vault = _default_vault(args.vault)
        summary = sync_vault(conn, vault, args.channel,
                             prune=not args.no_prune, log=print)
        print(
            f"sync: {summary['ingested']} ingested, {summary['unchanged']} unchanged, "
            f"{summary['deleted']} deleted, {summary['edges']} wikilink edge(s), "
            f"{summary['unresolved']} unresolved link(s) over {summary['notes']} note(s)"
        )
    except (ValueError, OSError) as exc:
        print(f"[obsidian-sync] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def _cmd_export(args) -> None:
    conn = memsom.get_connection()
    try:
        vault = _default_vault(args.vault)
        path = export_note(
            conn, vault,
            node_id=args.node, query=args.query,
            folder=args.folder, title=args.title, clearance=args.clearance,
        )
        print(f"wrote {path}")
    except (ValueError, OSError) as exc:
        print(f"[obsidian-export] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def _cmd_watch(args) -> None:
    vault = _default_vault(args.vault)
    watch_vault(
        memsom.get_connection, vault, args.channel,
        interval=args.interval, debounce=args.debounce,
        prune=not args.no_prune, log=print,
    )


def register(subparsers) -> None:
    """Mount obsidian sub-commands onto an existing argparse subparsers object."""
    p_sync = subparsers.add_parser(
        "obsidian-sync", help="sync an Obsidian vault into the DAG (wikilinks -> edges)"
    )
    p_sync.add_argument("vault", nargs="?", default=None,
                        help=f"vault path (default: ${VAULT_ENV})")
    p_sync.add_argument("--channel", default="user", choices=list(memsom.RANK.keys()),
                        help="default channel for un-stamped notes (default: user)")
    p_sync.add_argument("--no-prune", action="store_true",
                        help="do NOT tombstone notes deleted from the vault")
    p_sync.set_defaults(func=_cmd_sync)

    p_exp = subparsers.add_parser(
        "obsidian-export", help="write an answer back to the vault as a memsom note"
    )
    p_exp.add_argument("node", nargs="?", type=int, default=None,
                       help="node id to export (omit to use --query)")
    p_exp.add_argument("--query", default=None, help="compose an answer to export")
    p_exp.add_argument("--vault", default=None, help=f"vault path (default: ${VAULT_ENV})")
    p_exp.add_argument("--folder", default="memsom", help="subfolder within the vault")
    p_exp.add_argument("--title", default=None, help="note title (default: derived)")
    p_exp.add_argument("--clearance", default="topsecret", help="ask clearance for --query")
    p_exp.set_defaults(func=_cmd_export)

    p_watch = subparsers.add_parser(
        "obsidian-watch", help="watch a vault and live-sync on change (Ctrl-C to stop)"
    )
    p_watch.add_argument("vault", nargs="?", default=None,
                         help=f"vault path (default: ${VAULT_ENV})")
    p_watch.add_argument("--channel", default="user", choices=list(memsom.RANK.keys()))
    p_watch.add_argument("--interval", type=float, default=1.0, help="poll seconds")
    p_watch.add_argument("--debounce", type=float, default=0.4, help="quiet seconds before sync")
    p_watch.add_argument("--no-prune", action="store_true")
    p_watch.set_defaults(func=_cmd_watch)


def main(argv=None) -> None:
    """Thin CLI wrapper — for direct invocation."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom-obsidian")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
