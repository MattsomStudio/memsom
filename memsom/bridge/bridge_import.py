"""memsom_bridge_import — import the user's flat-file Claude memory into memsom.

Phase 1 of the "bridge" (see plan dreamy-giggling-piglet): make memsom the
store-of-record for the personal memory currently living as markdown files at
~/.claude/projects/.../memory/.  This is the READ-ONLY-toward-the-flat-files
importer: it reads every memory/*.md and mirrors each into ONE memsom node.

Design (locked in plan Phase 0):
  - One node per file (NOT chunked — these are already atomic memories).
  - The full markdown (frontmatter + body) is the node CONTENT, so the node is a
    loss-free, reversible copy of the file.  No sidecar table.
  - The MEMORY.md section a file lives under is NOT derivable from its type
    prefix (project_ spans several sections), so we parse the current MEMORY.md
    once and stamp a `section:` line into the stored frontmatter.  This bakes the
    hand-curated grouping as the canonical baseline the digest renders against.
  - Channel = trust grade by type:  user_/personal_/feedback_ -> endorsed (pinned,
    never demote);  project_/reference_ -> user (demotable, what RS ranking sorts).
  - Idempotent: keyed on nodes.bridge_path (the filename) + content_hash.  A
    re-run with no file change creates nothing.  A changed file tombstones the old
    node and inserts a new one (append-only ethos).
  - bridge_path/bridge_mtime are OWNED by this importer, not shared with
    memsom_obsidian's obsidian_path/obsidian_mtime (real Obsidian vault notes).
    They used to be the same columns — memsom_obsidian's vault-prune pass would
    then revoke-cascade bridge-imported memory nodes it never wrote, because
    nothing distinguished "my path" from "some other importer's path" sharing the
    same field. See memsom_forget's docstring for the same lesson learned earlier.

Library discipline: the import_* functions never print or sys.exit; only main()
and the _cmd_* wrapper do I/O.  Frozen core (memsom.py) is untouched — this is a
pure consumer of insert_node + the bridge_path/content_hash columns.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import memsom
from memsom.interface import ingest as memsom_ingest
from memsom.storage import schema as memsom_schema


# --- channel mapping (plan Phase 0b) -----------------------------------------
# pinned (never-demote) types land on the highest-integrity human channel.
CHANNEL_BY_TYPE = {
    "user": "endorsed",
    "personal": "endorsed",
    "feedback": "endorsed",
    "project": "user",
    "reference": "user",
}
DEFAULT_CHANNEL = "user"  # unknown/untyped memory -> user-grade, demotable


def migrate(conn) -> None:
    """Idempotent: ensure the columns this importer writes exist.

    bridge_path / bridge_mtime are this importer's own (not shared with
    memsom_obsidian); content_hash from memsom_ingest; the stale triple +
    source_supersedes/stale_log from memsom_stale (so the verification-staleness
    pass + the render's stale-awareness have their columns on BOTH machines via
    the existing bridge migrate chain).  All additive nullable columns (frozen
    core never reads them), so this is safe to call standalone or via
    migrate_all.
    """
    memsom_schema.add_column(conn, "nodes", "bridge_path", "TEXT")
    memsom_schema.add_column(conn, "nodes", "bridge_mtime", "TEXT")
    memsom_ingest.migrate(conn)
    from memsom.integrity import stale as memsom_stale
    memsom_stale.migrate(conn)   # nodes.{stale,stale_at,stale_reason} + supersedes/log
    _migrate_legacy_obsidian_columns(conn)


def _migrate_legacy_obsidian_columns(conn) -> None:
    """One-time, idempotent data move for DBs created before bridge_path existed.

    Before this fix, bridge-imported memory nodes were stamped on the SHARED
    obsidian_path/obsidian_mtime columns (borrowed from memsom_obsidian). Any
    such row is moved onto bridge_path/bridge_mtime and the borrowed columns are
    cleared, so memsom_obsidian's vault sync (scoped only by
    "obsidian_path IS NOT NULL") never mistakes a memory-bridge node for one of
    its own vault notes again. No-op on a DB that has never loaded memsom_obsidian
    (no obsidian_path column yet) and no-op on rows already migrated.
    """
    if not memsom_schema.column_exists(conn, "nodes", "obsidian_path"):
        return  # memsom_obsidian has never run here — nothing to reclaim
    with conn:
        conn.execute(
            "UPDATE nodes SET bridge_path = obsidian_path, bridge_mtime = obsidian_mtime, "
            "obsidian_path = NULL, obsidian_mtime = NULL "
            "WHERE source_ref LIKE 'memory:%' AND obsidian_path IS NOT NULL "
            "AND bridge_path IS NULL"
        )


# --- frontmatter parsing (light, stdlib) -------------------------------------

def split_frontmatter(text: str):
    """Return (fm_lines, body, had_fm).

    fm_lines is the raw list of lines between the opening and closing '---'
    fences (exclusive).  If there is no frontmatter, returns ([], text, False).
    """
    if not text.startswith("---"):
        return [], text, False
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], "\n".join(lines[i + 1:]), True
    return [], text, False  # unterminated fence -> treat as no frontmatter


def fm_top_level(fm_lines) -> dict:
    """Parse top-level (non-indented) `key: value` pairs from frontmatter lines.

    Nested blocks (e.g. an indented `metadata:` child) are ignored — we only
    need the flat keys (type, salience, pin, name, description).
    """
    out = {}
    for ln in fm_lines:
        if not ln or ln[0] in " \t#":  # skip indented children + comments
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):\s?(.*)$", ln)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def memory_type(stem: str, fm: dict) -> str:
    """Type from frontmatter `type:` if present, else the filename prefix."""
    t = (fm.get("type") or "").strip()
    if t:
        return t
    return stem.split("_", 1)[0] if "_" in stem else stem


def stamp_fm(text: str, **kv):
    """Return *text* with the given top-level frontmatter keys set (idempotent).

    Existing top-level lines for any key in *kv* are replaced; a key whose value
    is None is dropped.  If there was no frontmatter and nothing is added, the
    text is returned unchanged.
    """
    fm_lines, body, had = split_frontmatter(text)
    keys = set(kv)
    fm_lines = [ln for ln in fm_lines if ln.split(":", 1)[0].strip() not in keys]
    added = False
    for k, v in kv.items():
        if v is not None:
            fm_lines.append(f"{k}: {v}")
            added = True
    if not had and not added:
        return text
    return "---\n" + "\n".join(fm_lines) + "\n---\n" + body


def stamp_section(text: str, section):
    """Thin wrapper: stamp only the `section:` key (kept for the unit tests)."""
    return stamp_fm(text, section=section)


_HOOK_RE = re.compile(r"\]\(([^)]+\.md)\)\s*[—–-]\s*(.+\S)")
# a PRIMARY index entry: a bullet line whose FIRST link is the file, optionally
# followed by an em-dash hook.  Secondary inline links (not line-leading) and the
# file-less literal lines do not match — so each file gets its own line iff it has
# a curated primary line, exactly as in the hand-maintained MEMORY.md.
_PRIMARY_RE = re.compile(
    r"^\s*[-*]\s*\[([^\]]+)\]\(([^)]+\.md)\)(?:\s*[—–-]\s*(.+\S))?\s*$")


def index_hooks(memory_md_text: str) -> dict:
    """Map each linked filename -> its hand-curated hook (text after the em dash)."""
    out = {}
    for line in memory_md_text.split("\n"):
        m = _HOOK_RE.search(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _strip_render_marker(hook):
    """Drop a render-time staleness marker (' ⚠ ...') from a captured hook.

    The digest appends a ' ⚠' flag to stale lines AT RENDER.  Because the next
    import re-derives each hook by parsing the previous MEMORY.md, an un-stripped
    marker round-trips into the stored hook and compounds one glyph per cycle
    (the 16x-⚠ bug).  Stripping at capture makes the hook idempotent and
    self-healing: a polluted MEMORY.md collapses back to a single flag next render.
    """
    if not hook:
        return hook
    i = hook.find("⚠")
    if i != -1:
        hook = hook[:i].rstrip()
    return hook or None


def parse_primary_index(memory_md_text: str) -> dict:
    """{filename: (title, hook_or_None, section)} for line-leading primary entries.

    The curated title + hook are captured so the digest renders byte-for-byte like
    the hand-maintained index (frontmatter name/description are longer and bloat
    the file past its budget).  Files that appear only as secondary inline links
    (or not at all) are absent -> they get no digest line, matching MEMORY.md.
    """
    out = {}
    section = None
    for line in memory_md_text.split("\n"):
        h = re.match(r"^##\s+(.*\S)\s*$", line)
        if h:
            section = h.group(1).strip()
            continue
        if section is None:
            continue
        m = _PRIMARY_RE.match(line)
        if m:
            out[m.group(2)] = (m.group(1).strip(),
                               _strip_render_marker((m.group(3) or "").strip()),
                               section)
    return out


# --- MEMORY.md section map ----------------------------------------------------

_LINK_IN_LINE = re.compile(r"\]\(([^)]+\.md)\)")


def section_map(memory_md_text: str) -> dict:
    """Map each linked filename -> its `## Section` header in MEMORY.md."""
    out = {}
    current = None
    for line in memory_md_text.split("\n"):
        h = re.match(r"^##\s+(.*\S)\s*$", line)
        if h:
            current = h.group(1).strip()
            continue
        for m in _LINK_IN_LINE.finditer(line):
            out[m.group(1)] = current
    return out


def parse_index_entries(memory_md_text: str):
    """Yield (section, kind, payload) for every index line, in document order.

    kind 'file'    -> payload = linked filename (one yield per link on the line).
    kind 'literal' -> payload = the raw bullet line text (a hand-authored index
                      entry with no file behind it, e.g. the identity lead line or
                      the dated progress-check reminder).
    Section headers, the H1, and blank lines are skipped.
    """
    section = None
    for line in memory_md_text.split("\n"):
        h = re.match(r"^##\s+(.*\S)\s*$", line)
        if h:
            section = h.group(1).strip()
            continue
        if section is None:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        files = _LINK_IN_LINE.findall(line)
        if files:
            for f in files:
                yield (section, "file", f)
        elif stripped[0] in "-*⏰•":  # bullet / ⏰ / •  -> a literal entry
            yield (section, "literal", line.rstrip())


def _literal_content(section, text: str) -> str:
    return f"---\nliteral: true\nsection: {section}\n---\n{text}\n"


def import_literals(conn, memory_dir, *, dry_run: bool = True) -> dict:
    """Mirror the file-less MEMORY.md index lines into memsom as endorsed nodes.

    Keyed by a hash of the line text (source_ref = memory:literal:<hash>), so a
    re-run is idempotent and a line removed from the index gets tombstoned.
    """
    memory_dir = Path(memory_dir)
    index_path = memory_dir / "MEMORY.md"
    stats = {"created": 0, "updated": 0, "tombstoned": 0, "skipped": 0, "total": 0}
    if not index_path.exists():
        return stats
    desired = {}
    for section, kind, payload in parse_index_entries(index_path.read_text(encoding="utf-8")):
        if kind != "literal":
            continue
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        desired[f"memory:literal:{h}"] = (section, payload)
    stats["total"] = len(desired)
    existing = {r[0]: r[1] for r in conn.execute(
        "SELECT source_ref, id FROM nodes "
        "WHERE source_ref LIKE 'memory:literal:%' AND tombstoned = 0")}

    def _do():
        for sref, (section, payload) in desired.items():
            content = _literal_content(section, payload)
            if sref in existing:
                # The sref is a hash of the LINE TEXT only, so an existing sref
                # can still differ in section — the user moved the line under a
                # different ## heading. That placement is curated data; skipping
                # here would silently revert the move on the next render.
                # Supersede exactly like a changed file (tombstone + reinsert).
                nid = existing[sref]
                row = conn.execute(
                    "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
                if row and row[0] == content:
                    stats["skipped"] += 1
                    continue
                stats["updated"] += 1
                if dry_run:
                    continue
                conn.execute(
                    "UPDATE nodes SET tombstoned = 1, tombstoned_at = ?, revoke_reason = ? "
                    "WHERE id = ?",
                    (memsom.now_iso(), "superseded by bridge reimport", nid))
            else:
                stats["created"] += 1
                if dry_run:
                    continue
            nid = memsom.insert_node(conn, content, "endorsed", source_ref=sref)
            conn.execute("UPDATE nodes SET content_hash = ? WHERE id = ?",
                         (memsom_ingest._content_hash(content), nid))
        for sref, nid in existing.items():
            if sref not in desired:
                stats["tombstoned"] += 1
                if not dry_run:
                    conn.execute(
                        "UPDATE nodes SET tombstoned = 1, tombstoned_at = ?, revoke_reason = ? "
                        "WHERE id = ?",
                        (memsom.now_iso(), "literal removed from index", nid))

    if dry_run:
        _do()
    else:
        with conn:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            _do()
    return stats


def relate_wikilinks(conn, memory_dir, *, dry_run: bool = True) -> dict:
    """Pass 2: materialise the ``[[wikilinks]]`` in memory-file bodies as edges.

    import_memory_dir stores each file as ONE node but leaves its ``[[links]]``
    inert in the body text.  This pass parses them and creates associative
    ('wikilink') rel_edges between the corresponding nodes — the SAME edge kind
    memsom_obsidian builds for vault notes — so neighborhood()/graph-rerank can
    traverse the personal memory, not only the Obsidian vault.  Without it the
    associative half of the graph is dark for everything imported via the bridge.

    Resolution mirrors the vault path exactly: a bare ``[[stem]]`` resolves to
    ``<stem>.md`` iff that basename is unique among the memory files, and the two
    endpoints are the live nodes for those bridge_paths.  Links whose target file
    does not exist yet (the "write it later" convention) resolve to nothing and
    are counted as unresolved — a useful signal, not an error.

    Idempotent: relate() is INSERT OR IGNORE, and every run re-derives all edges
    from the current bodies against the current live nodes, so a changed file
    whose node id rolled over gets its edges rebuilt (stale edges to the old,
    now-tombstoned node stay inert — neighborhood's BFS skips dead nodes).

    Only wikilinks are parsed (not markdown links): memory files cross-reference
    each other by ``[[name]]`` convention, whereas a stray ``](path)`` usually
    points into the vault, a different corpus.  The parse is code-fence-masked
    (via memsom_obsidian._mask) so a ``[[x]]`` inside a fenced block is ignored.
    """
    from memsom.bridge import obsidian as memsom_obsidian
    from memsom.retrieval import relate as memsom_relate

    memory_dir = Path(memory_dir)
    stats = {"edges": 0, "resolved": 0, "unresolved": 0, "skipped_self": 0}

    files = sorted(p for p in memory_dir.glob("*.md") if p.name != "MEMORY.md")
    by_name, by_relpath = memsom_obsidian._build_resolver([p.name for p in files])

    # Parse each body's wikilinks with the same masking the vault sync uses, so a
    # [[x]] inside a code fence never becomes an edge. Dedup, preserve order.
    note_links = {}
    for path in files:
        body = split_frontmatter(path.read_text(encoding="utf-8"))[1]
        masked = memsom_obsidian._mask(body)
        seen, targets = set(), []
        for _bang, inner in memsom_obsidian._WIKILINK.findall(masked):
            t = memsom_obsidian._wikilink_target(inner)
            if t and t not in seen:
                seen.add(t)
                targets.append(t)
        note_links[path.name] = targets

    def _do():
        if not dry_run:
            memsom_relate.migrate(conn)   # ensure the rel_edges table exists
        for src_rel, targets in note_links.items():
            src_row = _live_node_for_path(conn, src_rel)
            if src_row is None:
                continue                  # file not imported (dry-run on empty DB)
            src = src_row[0]
            for tgt in targets:
                tgt_rel = memsom_obsidian._resolve_target(tgt, by_name, by_relpath)
                if tgt_rel is None:
                    stats["unresolved"] += 1
                    continue
                dst_row = _live_node_for_path(conn, tgt_rel)
                if dst_row is None:
                    stats["unresolved"] += 1
                    continue
                dst = dst_row[0]
                if dst == src:
                    stats["skipped_self"] += 1
                    continue
                stats["resolved"] += 1
                if dry_run:
                    continue
                try:
                    memsom_relate.relate(conn, src, dst,
                                         kind=memsom_obsidian.LINK_KIND)
                    stats["edges"] += 1
                except ValueError:
                    pass  # a node vanished mid-run — harmless, skip the edge

    _do()
    return stats


def import_all(conn, memory_dir, *, dry_run: bool = True) -> dict:
    """Import the per-file memories, the literal index lines, then wire edges.

    Order matters: relate_wikilinks runs LAST so every file already has a live
    node to point an edge at (Pass 2, exactly as memsom_obsidian sequences it).
    """
    files = import_memory_dir(conn, memory_dir, dry_run=dry_run)
    lits = import_literals(conn, memory_dir, dry_run=dry_run)
    edges = relate_wikilinks(conn, memory_dir, dry_run=dry_run)
    return {"files": files, "literals": lits, "edges": edges}


# --- DB helpers ---------------------------------------------------------------

def _live_node_for_path(conn, rel: str):
    """Return (id, content_hash, redacted) of the live node for *rel*, or None.

    *redacted* is 0 on a store predating the redact migration (column absent) — the
    resurrection guard just doesn't fire there, which is correct (no redactions yet).
    """
    rcol = ("redacted" if memsom_schema.column_exists(conn, "nodes", "redacted")
            else "0 AS redacted")
    return conn.execute(
        f"SELECT id, content_hash, {rcol} FROM nodes "
        "WHERE bridge_path = ? AND tombstoned = 0 ORDER BY id DESC LIMIT 1",
        (rel,),
    ).fetchone()


def _stored_index_meta(conn, rel: str) -> dict:
    """Index metadata (section/index_title/index_hook) already stamped on the live
    node for *rel*, or {}.  The last-resort fallback below: it is what keeps a
    budget-evicted memory filed (see the fallback chain in import_memory_dir)."""
    row = conn.execute(
        "SELECT content FROM nodes "
        "WHERE bridge_path = ? AND tombstoned = 0 ORDER BY id DESC LIMIT 1",
        (rel,),
    ).fetchone()
    if not row or not row[0]:
        return {}
    fm = fm_top_level(split_frontmatter(row[0])[0])
    return {k: fm[k] for k in ("section", "index_title", "index_hook") if fm.get(k)}


def _mtime_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


# --- the importer -------------------------------------------------------------

def import_memory_dir(conn, memory_dir, *, dry_run: bool = True) -> dict:
    """Import every memory/*.md (excluding MEMORY.md) into memsom.

    Returns stats: {total_files, created, updated, skipped, tombstoned}.
    Atomic: all writes happen in one transaction (or none, if dry_run).
    """
    memory_dir = Path(memory_dir)
    if not memory_dir.is_dir():
        raise ValueError(f"memory dir is not a directory: {memory_dir}")

    index_path = memory_dir / "MEMORY.md"
    primary = (parse_primary_index(index_path.read_text(encoding="utf-8"))
               if index_path.exists() else {})

    files = sorted(p for p in memory_dir.glob("*.md") if p.name != "MEMORY.md")
    stats = {"total_files": len(files), "created": 0, "updated": 0,
             "skipped": 0, "tombstoned": 0, "swept": 0, "refused_resurrect": 0}

    # Mass-wipe guard: importing a directory with ZERO memory files while the
    # store holds live memory nodes would make the reconcile sweep below (and
    # import_literals' index reconcile) tombstone EVERY one of them — the classic
    # mispointed-dir accident (e.g. a fallback path with no *.md files). That is
    # never a legitimate import: refuse loudly instead of silently blanking the
    # brain. A genuinely fresh setup (empty dir AND empty store) still imports.
    if not files:
        live = conn.execute(
            "SELECT COUNT(*) FROM nodes "
            "WHERE source_ref LIKE 'memory:%' AND tombstoned = 0"
        ).fetchone()[0]
        if live:
            raise ValueError(
                f"refusing bridge import: {memory_dir} contains no memory .md "
                f"files but the store has {live} live memory node(s) — the "
                f"reconcile sweep would tombstone all of them. Wrong directory? "
                f"Set MEMDAG_BRIDGE_MEMORY_DIR to the real memory dir.")

    def _do():
        for path in files:
            rel = path.name                      # bridge_path key, e.g. user_adhd.md
            stem = path.stem
            raw = path.read_text(encoding="utf-8")
            fm = fm_top_level(split_frontmatter(raw)[0])
            channel = CHANNEL_BY_TYPE.get(memory_type(stem, fm), DEFAULT_CHANNEL)
            title, hook, section = primary.get(rel, (None, None, None))
            # MEMORY.md is the CURATED source for index metadata — but it must not
            # be the ONLY one. digest._select_hot requires a truthy section, so any
            # file absent from the index resolved section=None and became
            # permanently unselectable, even while hot:
            #   - a BRAND-NEW memory (never in MEMORY.md) could never enter it —
            #     rendering alone could not file it, so /saveall silently dropped it;
            #   - a memory the digest's byte-budget EVICTED lost its section on the
            #     next import, so a purely-transient, RS-ordered eviction became a
            #     permanent unfiling it could not recover from.
            # Fall back: curated index > the file's own frontmatter > whatever is
            # already stamped on the live node. Eviction stays reversible (the digest
            # re-drops by RS each render); nothing silently loses its filing.
            if section is None or title is None or hook is None:
                prev = _stored_index_meta(conn, rel)
                section = section or fm.get("section") or prev.get("section")
                title = title or fm.get("index_title") or prev.get("index_title")
                hook = hook or fm.get("index_hook") or prev.get("index_hook")
            stamped = stamp_fm(raw, section=section, index_title=title, index_hook=hook)
            new_hash = memsom_ingest._content_hash(stamped)

            existing = _live_node_for_path(conn, rel)

            # Resurrection guard (checked BEFORE the hash-skip so an identical
            # resurfaced copy is caught too): the live predecessor for this path
            # was REDACTED. Its payload was deliberately destroyed, so the file
            # must NOT rehydrate it as a fresh, non-redacted node (the silent
            # un-redaction that redaction-reaches-disk exists to prevent) — nor
            # linger on disk. Refuse: leave the redacted node in place and unlink
            # the resurfaced file, whatever its hash. path is inside memory_dir by
            # construction (glob), so no traversal check is needed here.
            if existing and existing[2]:
                stats["refused_resurrect"] += 1
                if not dry_run:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                continue

            if existing and existing[1] == new_hash:
                stats["skipped"] += 1
                continue

            if dry_run:
                stats["updated" if existing else "created"] += 1
                if existing:
                    stats["tombstoned"] += 1
                continue

            if existing:
                conn.execute(
                    "UPDATE nodes SET tombstoned = 1, tombstoned_at = ?, revoke_reason = ? "
                    "WHERE id = ?",
                    (memsom.now_iso(), "superseded by bridge reimport", existing[0]),
                )
                stats["tombstoned"] += 1
                stats["updated"] += 1
            else:
                stats["created"] += 1

            nid = memsom.insert_node(conn, stamped, channel, source_ref=f"memory:{stem}")
            conn.execute(
                "UPDATE nodes SET bridge_path = ?, bridge_mtime = ?, content_hash = ? "
                "WHERE id = ?",
                (rel, _mtime_sig(path), new_hash, nid),
            )
            # Carry the forgetting-layer state (Bjork RS/SS model) from the
            # superseded predecessor: a reimport is an EDIT of the same memory,
            # not a new memory. Without this, every /saveall edit reseeded
            # rs=1.0/ss=0.0/count=0/first_seen=now/tier=hot — wiping the note's
            # accumulated storage strength and age each time it was touched.
            if existing and memsom_schema.column_exists(conn, "nodes", "forget_rs"):
                conn.execute(
                    "UPDATE nodes SET "
                    "(forget_rs, forget_ss, forget_count, forget_first_seen, "
                    " forget_last_used, forget_tier) = "
                    "(SELECT forget_rs, forget_ss, forget_count, forget_first_seen, "
                    "        forget_last_used, forget_tier FROM nodes WHERE id = ?) "
                    "WHERE id = ?",
                    (existing[0], nid),
                )

        # reconcile deletions: tombstone live file-backed nodes whose source
        # file has vanished. The loop above only ever touches files that EXIST,
        # so without this a deleted memory's node lingers live forever (the gap
        # that once orphaned a deleted note's node). Literals (bridge_path
        # IS NULL) are excluded here — import_literals reconciles those against
        # the index.
        present = {p.name for p in files}
        gone = conn.execute(
            "SELECT id, bridge_path FROM nodes "
            "WHERE source_ref LIKE 'memory:%' AND source_ref NOT LIKE 'memory:literal:%' "
            "AND tombstoned = 0 AND bridge_path IS NOT NULL"
        ).fetchall()
        for nid, opath in gone:
            if opath in present:
                continue
            stats["swept"] += 1
            if not dry_run:
                conn.execute(
                    "UPDATE nodes SET tombstoned = 1, tombstoned_at = ?, revoke_reason = ? "
                    "WHERE id = ?",
                    (memsom.now_iso(), "source file removed (bridge reconcile)", nid),
                )

    if dry_run:
        _do()
    else:
        with conn:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            _do()
    return stats


# --- CLI ----------------------------------------------------------------------

def default_memory_dir():
    """Locate the live Claude memory dir without hard-coding a username.

    Override with $MEMDAG_BRIDGE_MEMORY_DIR; otherwise discover the first
    `~/.claude/projects/*/memory/MEMORY.md` (the project-dir name differs per
    machine, so it is globbed, never hard-coded)."""
    env = os.environ.get("MEMDAG_BRIDGE_MEMORY_DIR")
    if env:
        return Path(env)
    candidates = [m.parent for m in
                  (Path.home() / ".claude" / "projects").glob("*/memory/MEMORY.md")]
    if not candidates:
        # Fail LOUDLY. The old fallback returned ~/.claude/projects itself — a
        # directory with zero .md files — which sent the importer's reconcile
        # sweep off to tombstone every live memory node (the mass-wipe path the
        # import guard also blocks). No plausible dir means no import.
        raise FileNotFoundError(
            "no Claude memory dir found (no ~/.claude/projects/*/memory/MEMORY.md); "
            "set MEMDAG_BRIDGE_MEMORY_DIR explicitly")
    # the real brain is the memory dir with the MOST .md files; project-scoped
    # memory dirs (created by running Claude in another cwd) hold ~1 file and must
    # not be mistaken for it just because they sort first.
    return max(candidates, key=lambda d: len(list(d.glob("*.md"))))


def _print_stats(stats, dry_run):
    mode = "DRY-RUN (no writes)" if dry_run else "APPLIED"
    f, l = stats["files"], stats["literals"]
    print(f"[bridge-import] {mode}")
    print(f"  files seen     : {f['total_files']}")
    print(f"  files created  : {f['created']}")
    print(f"  files updated  : {f['updated']} (old tombstoned: {f['tombstoned']})")
    print(f"  files skipped  : {f['skipped']} (unchanged)")
    print(f"  deleted swept  : {f['swept']} (source file gone -> node tombstoned)")
    if f.get("refused_resurrect"):
        print(f"  RESURRECT BLK  : {f['refused_resurrect']} (redacted node's file "
              f"resurfaced -> refused + file unlinked)")
    print(f"  literals       : {l['total']} total | {l['created']} created | "
          f"{l['skipped']} skipped | {l['tombstoned']} tombstoned")
    e = stats.get("edges")
    if e is not None:
        verb = "would relate" if dry_run else "created"
        print(f"  wikilink edges : {e['edges']} {verb} | {e['resolved']} resolved | "
              f"{e['unresolved']} unresolved | {e['skipped_self']} self-links skipped")


def _cmd_import(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        memory_dir = args.memory_dir or default_memory_dir()
        stats = import_all(conn, memory_dir, dry_run=not args.apply)
        _print_stats(stats, dry_run=not args.apply)
    finally:
        conn.close()


def register(sub) -> None:
    """Mount the bridge-import subcommand onto an argparse subparsers object."""
    p = sub.add_parser("bridge-import",
                       help="import flat-file memories into memsom (Phase 1)")
    p.add_argument("memory_dir", nargs="?", default=None,
                   help="memory dir (default: live PC store / $MEMDAG_BRIDGE_MEMORY_DIR)")
    p.add_argument("--apply", action="store_true",
                   help="apply the import (default: dry-run)")
    p.set_defaults(func=_cmd_import)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass  # reconfigure unsupported on this stream — keep its default encoding
    ap = argparse.ArgumentParser(prog="memsom_bridge_import", description=__doc__)
    ap.add_argument("memory_dir", nargs="?", default=None)
    ap.add_argument("--apply", action="store_true", help="apply (default: dry-run)")
    main_args = ap.parse_args(argv)
    _cmd_import(main_args)


if __name__ == "__main__":
    main()
