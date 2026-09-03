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
import re
import sys
from pathlib import Path

import memsom
from memsom.kernel.paths import default_memory_dir
from memsom.kernel import chunking as memsom_chunking
from memsom.kernel.frontmatter import (
    split_frontmatter, fm_top_level, fm_flat, memory_type, stamp_fm, stamp_section,
    parse_primary_index, parse_index_entries, section_map, index_hooks,
)
from memsom.integrity import ingest as memsom_ingest
from memsom.storage import schema as memsom_schema
from memsom import tuning as memsom_tuning


# --- channel mapping (plan Phase 0b) -----------------------------------------
# pinned (never-demote) types land on the highest-integrity human channel.
CHANNEL_BY_TYPE = {
    "user": "endorsed",
    "personal": "endorsed",
    "feedback": "endorsed",
    "project": "user",
    "reference": "user",
    "fact": "user",  # fact layer (docs/facts-design.md): demotable, earns keep by being referenced
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
    from memsom.lifecycle import stale as memsom_stale
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


# --- memory-dir layout --------------------------------------------------------

PROJECTS_SUBDIR = "projects"
# The two GENERATED index files. Never imported as memories: MEMORY.md is the
# main digest, projects/INDEX.md the project sub-index (both rendered by
# memsom.distill.digest from the store — importing them would loop).
INDEX_NAMES = frozenset({"MEMORY.md", "INDEX.md"})


class DuplicateMemoryStem(ValueError):
    """Two memory files share a filename anywhere in the memory tree AND the
    importer could not heal it (the quarantine write failed).

    The bridge keys nodes on the BASENAME (bridge_path) and wikilinks resolve
    by bare stem, so a stem must be globally unique across every level.  An
    additive sync layer (robocopy /E, rsync --update, Syncthing without
    deletes) turns every MOVE into a duplicate on the other machine — the old
    flat copy stays behind — so duplicates are a normal, recurring condition,
    not a configuration error.  iter_memory_files therefore never raises on
    one: it picks a canonical copy (see resolve_duplicates) and the importer
    heals the loser on disk.  This error is reserved for the heal itself
    failing, which would leave the two copies shadowing each other forever.
    """


DUP_QUARANTINE_SUBDIR = Path(".weights") / "dup_quarantine"


def _md_files(d):
    # A Syncthing conflict copy (`foo.sync-conflict-<ts>-<id>.md`) is a transient
    # duplicate, not a memory — importing it mints a bogus node under a mangled
    # stem. Excluded at the one walk both the importer and resolve_duplicates use.
    return [p for p in d.glob("*.md")
            if p.is_file() and p.name not in INDEX_NAMES
            and ".sync-conflict-" not in p.name]


def _all_memory_files(memory_dir):
    memory_dir = Path(memory_dir)
    files = _md_files(memory_dir)
    proj = memory_dir / PROJECTS_SUBDIR
    if proj.is_dir():
        files += _md_files(proj)
        for d in sorted(x for x in proj.iterdir() if x.is_dir()):
            files += _md_files(d)
    return files


def _depth(p):
    return len(Path(p).parts)


def resolve_duplicates(memory_dir):
    """Group same-named memory files and pick the canonical copy of each.

    Returns ``(canonical, dups)``: *canonical* is ``{name: Path}`` for every
    stem; *dups* is a list of ``{"name", "keep", "drop", "action"}`` for every
    stem that appeared more than once, where action is ``"delete"`` when the
    copies are byte-identical (the deepest path wins — a MOVE into projects/
    is the intended final state, the shallow copy is the sync leftover) or
    ``"quarantine"`` when they differ (the newest mtime wins; ties go to the
    deepest path).  Pure: touches nothing on disk.
    """
    by_name = {}
    for p in _all_memory_files(memory_dir):
        by_name.setdefault(p.name, []).append(p)
    canonical, dups = {}, []
    for name, paths in sorted(by_name.items()):
        if len(paths) == 1:
            canonical[name] = paths[0]
            continue
        blobs = {p: p.read_bytes() for p in paths}
        identical = len(set(blobs.values())) == 1
        if identical:
            keep = max(paths, key=lambda p: (_depth(p), str(p)))
            action = "delete"
        else:
            keep = max(paths, key=lambda p: (p.stat().st_mtime_ns, _depth(p), str(p)))
            action = "quarantine"
        canonical[name] = keep
        dups.append({"name": name, "keep": keep,
                     "drop": [p for p in paths if p != keep], "action": action})
    return canonical, dups


def heal_duplicates(memory_dir, *, dry_run=True) -> dict:
    """Self-heal duplicate stems on disk (the additive-sync leftover).

    Byte-identical copies: the shallower file(s) are deleted.  Differing
    copies: the older file is MOVED to ``.weights/dup_quarantine/<stem>.<mtime_ns>.md``
    (never deleted — a differing copy may hold an edit made on the other
    machine).  Returns ``{"dedup": n_deleted, "quarantined": n_moved,
    "duplicates": [...]}``; with *dry_run* nothing is touched and the counts
    describe what WOULD happen.  Raises DuplicateMemoryStem only when a
    quarantine write fails.
    """
    memory_dir = Path(memory_dir)
    _canon, dups = resolve_duplicates(memory_dir)
    stats = {"dedup": 0, "quarantined": 0, "duplicates": []}
    for d in dups:
        for loser in d["drop"]:
            rec = {"name": d["name"], "kept": str(d["keep"]), "dropped": str(loser),
                   "action": d["action"]}
            if d["action"] == "delete":
                stats["dedup"] += 1
                if not dry_run:
                    loser.unlink()
            else:
                stats["quarantined"] += 1
                qdir = memory_dir / DUP_QUARANTINE_SUBDIR
                dest = qdir / f"{loser.stem}.{loser.stat().st_mtime_ns}.md"
                rec["quarantined_to"] = str(dest)
                if not dry_run:
                    try:
                        qdir.mkdir(parents=True, exist_ok=True)
                        loser.replace(dest)
                    except OSError as exc:
                        raise DuplicateMemoryStem(
                            f"duplicate memory filename {d['name']!r}: could not "
                            f"quarantine {loser} -> {dest}: {exc}") from exc
            stats["duplicates"].append(rec)
    return stats


def iter_memory_files(memory_dir):
    """Yield every per-fact memory file under *memory_dir*, sorted.

    Layout: the flat ``*.md`` files, plus the project tree under ``projects/``:
    ``projects/project_<x>.md`` (standalone projects) and one level deeper
    ``projects/<slug>/*.md`` (a project directory: its ``project_<slug>.md``
    parent overview plus ``project_<slug>_<sub>.md`` subprojects).  Depth 2
    under the memory dir is the floor — nothing deeper is walked.  Generated
    index files (``MEMORY.md``, any ``INDEX.md``) are excluded at every level.

    Filenames are the node key and the wikilink target, so they must stay
    globally unique: when the same filename appears twice, ONE canonical copy
    is returned (resolve_duplicates picks it) and the other is left for the
    importer's heal_duplicates pass.  Never raises for a duplicate.
    """
    canonical, _dups = resolve_duplicates(memory_dir)
    return sorted(canonical.values(), key=lambda p: (p.name, str(p)))


PROJECTS_INDEX_SEED = "# Projects\n"


def scaffold_memory_dir(memory_dir, *, params=None) -> dict:
    """Create-if-absent scaffold for a memory dir so a fresh install is complete
    from day one.  Returns {name: "created" | "present"} for each piece.

      memory/projects/                 the project tree (hierarchical layout)
      memory/projects/INDEX.md         seeded empty so MEMORY.md's pointer line
                                       never dangles before the first render
      memory/.weights/canonical.json   the runtime-params file with the panel
                                       defaults (memory_budget, memory_max_lines,
                                       + the forgetting-layer defaults)

    Never overwrites: canonical.json is owned by whatever weights layer the
    user runs (memsom only READS it after this), and INDEX.md is rewritten by
    bridge-render.  Creating an absent file is not contention with either.
    """
    import json
    from memsom.lifecycle import forget as _forget
    memory_dir = Path(memory_dir)
    out = {}
    proj = memory_dir / PROJECTS_SUBDIR
    out["projects_dir"] = "present" if proj.is_dir() else "created"
    proj.mkdir(parents=True, exist_ok=True)
    idx = proj / "INDEX.md"
    if idx.exists():
        out["projects_index"] = "present"
    else:
        idx.write_bytes(PROJECTS_INDEX_SEED.encode("utf-8"))
        out["projects_index"] = "created"
    canon = memory_dir / ".weights" / "canonical.json"
    if canon.exists():
        out["canonical"] = "present"
    else:
        canon.parent.mkdir(parents=True, exist_ok=True)
        body = {"version": 1, "memories": {},
                "params": dict(params or {**_forget.DEFAULTS,
                                          **_forget.PANEL_PARAM_DEFAULTS})}
        tmp = canon.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
        tmp.replace(canon)
        out["canonical"] = "created"
    return out


def memory_subdir(memory_dir, path) -> str | None:
    """The memory-relative directory of *path* with forward slashes
    (``"projects"`` or ``"projects/<slug>"``), or None for a flat file.

    Paths come from iter_memory_files, so this is pure path arithmetic — no
    filesystem resolution."""
    try:
        rel = Path(path).parent.relative_to(Path(memory_dir))
    except ValueError:
        return None
    return rel.as_posix() if rel.parts else None


def unsectioned_by_frontmatter(fm: dict) -> bool:
    """True when the file's own top-level frontmatter withdraws it from the
    index: `section: none` (case-insensitive) or `index: false`."""
    if (fm.get("section") or "").strip().lower() == "none":
        return True
    return (fm.get("index") or "").strip().lower() in ("false", "no", "0")



def _literal_content(section, text: str) -> str:
    return f"---\nliteral: true\nsection: {section}\n---\n{text}\n"


def import_literals(conn, memory_dir, *, dry_run: bool = True) -> dict:
    """Mirror the file-less MEMORY.md index lines into memsom as endorsed nodes.

    Keyed by a hash of the line text (source_ref = memory:literal:<hash>), so a
    re-run is idempotent and a line removed from the index gets tombstoned.
    """
    memory_dir = Path(memory_dir)
    index_path = memory_dir / "MEMORY.md"
    stats = {"created": 0, "updated": 0, "tombstoned": 0, "skipped": 0, "total": 0,
             "indexed": 0, "deindexed": 0}
    if not index_path.exists():
        return stats
    new_ids, dead_ids = [], []
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

    # RMW-OK: runs inside import_literals' BEGIN IMMEDIATE (opened at the call site below)
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
                dead_ids.append(nid)
            else:
                stats["created"] += 1
                if dry_run:
                    continue
            # MS-20: route the channel through the F-13 ceiling guard even
            # though it is hardcoded here -- consistent with import_memory_dir
            # below, and defends this mint point if that ever changes.
            nid = memsom_ingest.mint_node(conn, content, "endorsed", source_ref=sref)
            conn.execute("UPDATE nodes SET content_hash = ? WHERE id = ?",
                         (memsom_chunking.content_hash(content), nid))
            new_ids.append(nid)
        for sref, nid in existing.items():
            if sref not in desired:
                stats["tombstoned"] += 1
                if not dry_run:
                    conn.execute(
                        "UPDATE nodes SET tombstoned = 1, tombstoned_at = ?, revoke_reason = ? "
                        "WHERE id = ?",
                        (memsom.now_iso(), "literal removed from index", nid))
                    dead_ids.append(nid)

    if dry_run:
        _do()
    else:
        with conn:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            _do()
        sync_index(conn, new_ids, dead_ids, stats)   # after the commit (see helper)
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

    files = iter_memory_files(memory_dir)
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


def relate_fact_deps(conn, memory_dir, *, dry_run: bool = True) -> dict:
    """Pass 2b: materialise fact ``depends_on:`` into the ``edges`` DAG (Phase 1).

    docs/facts-design.md ("Dependencies and cascade"): depends_on expresses
    real derivation ("this measurement is only true because of that
    hardware"), NOT association — so unlike relate_wikilinks (which fills
    rel_edges), this fills the SAME ``edges`` table CASCADE_CTE walks: the
    revoke/stale derivation DAG memsom.derive_node also writes. parent = the
    depended-on fact's live node, child = the dependent fact's live node.

    ``depends_on:`` values are filename stems, resolved via the identical
    _build_resolver/_resolve_target the wikilink pass uses (per the Phase 0
    "Verified" note: the stem, NOT the ``name:`` kebab slug) — comma/space
    separated for more than one dependency.

    Every call re-derives ALL depends_on edges from the CURRENT frontmatter
    against the CURRENT live nodes — not just files that changed this run —
    exactly like relate_wikilinks. That is what makes a supersede (spec
    Behavior 2) "rewire" for free: whichever endpoint (parent OR child) just
    rolled over to a new node id, the next call resolves that stem to its
    CURRENT live node and inserts the correct fresh edge; the edge row still
    pointing at the now-tombstoned predecessor is simply never re-derived and
    stays dormant (never traversed — every CASCADE_CTE walk in this codebase
    is seeded from a live id), the same fate relate_wikilinks leaves orphaned
    rel_edges rows to.

    Idempotent: edges' PK is (child, parent) and every insert is OR IGNORE.
    A ``depends_on:`` target with no live node yet (not written, or not
    imported this run) is counted "unresolved", never an error — Behavior 3,
    picked up automatically the next time this runs after the target exists.
    Self-deps are counted "skipped_self", not inserted.
    """
    from memsom.bridge import obsidian as memsom_obsidian

    memory_dir = Path(memory_dir)
    stats = {"edges": 0, "resolved": 0, "unresolved": 0, "skipped_self": 0}

    files = iter_memory_files(memory_dir)
    by_name, by_relpath = memsom_obsidian._build_resolver([p.name for p in files])

    # Pass A: parse every file's depends_on (comma/space-separated stems).
    fact_deps = {}
    for path in files:
        # fm_flat: a depends_on the Claude Code stamper nested under metadata: still counts
        fm = fm_flat(split_frontmatter(path.read_text(encoding="utf-8"))[0])
        raw = (fm.get("depends_on") or "").strip()
        if not raw:
            continue
        seen, targets = set(), []
        for tok in re.split(r"[,\s]+", raw):
            if tok and tok not in seen:
                seen.add(tok)
                targets.append(tok)
        if targets:
            fact_deps[path.name] = targets

    # Pass B: resolve each target to a LIVE node and materialise the edge.
    for child_rel, targets in fact_deps.items():
        child_row = _live_node_for_path(conn, child_rel)
        if child_row is None:
            continue  # this file has no live node yet (e.g. dry-run on an empty DB)
        child = child_row[0]
        for tgt in targets:
            parent_rel = memsom_obsidian._resolve_target(tgt, by_name, by_relpath)
            if parent_rel is None:
                stats["unresolved"] += 1
                continue
            parent_row = _live_node_for_path(conn, parent_rel)
            if parent_row is None:
                stats["unresolved"] += 1
                continue
            parent = parent_row[0]
            if parent == child:
                stats["skipped_self"] += 1
                continue
            stats["resolved"] += 1
            if dry_run:
                continue
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO edges(child, parent) VALUES (?, ?)",
                    (child, parent))
            # MS-21: a depends_on edge is a Biba derivation edge in the SAME
            # `edges` table CASCADE_CTE and derive_node's own min(parents)
            # clamp both treat as provenance -- it must not sit un-reflected
            # in the child's stored label. Re-derive (not re-mint: this is
            # an edit of an EXISTING fact, not a fresh compose) via the same
            # multi-hop walk heal.check()/recompute_all use, so revoking the
            # depended-on fact later floors this child exactly as any other
            # derived node would.
            from memsom.integrity import recompute as memsom_recompute
            memsom_recompute.recompute_node(conn, child)
            stats["edges"] += 1

    return stats


def import_all(conn, memory_dir, *, dry_run: bool = True, params=None) -> dict:
    """Import the per-file memories, the literal index lines, then wire edges.

    Order matters: relate_wikilinks / relate_fact_deps run LAST so every file
    already has a live node to point an edge at (Pass 2, exactly as
    memsom_obsidian sequences it).
    """
    files = import_memory_dir(conn, memory_dir, dry_run=dry_run, params=params)
    lits = import_literals(conn, memory_dir, dry_run=dry_run)
    edges = relate_wikilinks(conn, memory_dir, dry_run=dry_run)
    fact_edges = relate_fact_deps(conn, memory_dir, dry_run=dry_run)
    return {"files": files, "literals": lits, "edges": edges, "fact_edges": fact_edges}


# --- retrieval index upkeep ---------------------------------------------------

def index_enabled() -> bool:
    """The bridge keeps the retrieval index (postings/docstats/embeddings)
    current by default; MEMDAG_BRIDGE_INDEX=0 restores the old write-only
    behaviour (rebuild later with `memsom reindex`)."""
    return memsom_tuning.resolve("bridge.index_enabled") != "0"


def sync_index(conn, created, tombstoned, stats) -> None:
    """index_node every *created* id, deindex_node every *tombstoned* id.

    MUST be called after the importer's own BEGIN IMMEDIATE block has
    committed: both retrieve functions open their own `with conn:` and nesting
    that inside the open transaction would end it early (same rule as the
    stale cascade below).  Root cause this fixes: insert_node never indexes,
    so bridge-imported nodes were invisible to `memsom retrieve` until a
    manual `memsom reindex` (116/3065 nodes indexed on the live store).
    index_node degrades to BM25-only when no embedding backend is reachable,
    so this stays unconditional apart from the env kill-switch.
    """
    stats.setdefault("indexed", 0)
    stats.setdefault("deindexed", 0)
    if not index_enabled() or not (created or tombstoned):
        return
    from memsom.retrieval import retrieve as memsom_retrieve
    for nid in tombstoned:
        memsom_retrieve.deindex_node(conn, nid)
        stats["deindexed"] += 1
    for nid in created:
        if memsom_retrieve.index_node(conn, nid):
            stats["indexed"] += 1


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
    return {k: fm[k] for k in ("section", "index_title", "index_hook", "index_pending")
            if fm.get(k)}


# --- anti-creep: born-unindexed feedback --------------------------------------
# Every new feedback_* file used to be born with its own pinned index line and
# nothing ever merged, so the Feedback section grew one line per lesson (141
# lines before the 2026-08-20 collapse into feedback_cluster_* files).  A NEW
# feedback file is now imported with its section cleared unless it says WHY it
# deserves a standalone line; the lesson belongs in the matching cluster's body.
FEEDBACK_SECTION = "Feedback"
FEEDBACK_PREFIX = "feedback_"
CLUSTER_PREFIX = "feedback_cluster_"
OWN_LINE_KEY = "why_own_line"
NEEDS_CLUSTER = "needs_cluster"


def has_own_line_reason(fm: dict) -> bool:
    return bool((fm.get(OWN_LINE_KEY) or "").strip())


def born_unindexed(stem: str, fm: dict, section, *, curated: bool,
                   existing, prev_meta: dict, enabled: bool = True) -> bool:
    """True when this feedback file must be imported with its section cleared.

    Applies to a file that resolves to the Feedback section, is NOT a cluster,
    has no top-level `why_own_line:`, has no curated MEMORY.md line of its own,
    and is either new to the store (*existing* is None) or was already held
    back by this rule (``index_pending: needs_cluster`` on the stored node — so
    the next unchanged re-import cannot quietly index it).  Already-stored
    nodes that were indexed before the rule shipped are never unfiled.
    """
    if not enabled or not stem.startswith(FEEDBACK_PREFIX):
        return False
    if stem.startswith(CLUSTER_PREFIX) or curated or has_own_line_reason(fm):
        return False
    if (section or "").strip().lower() != FEEDBACK_SECTION.lower():
        return False
    return existing is None or prev_meta.get("index_pending") == NEEDS_CLUSTER


def _load_import_params(memory_dir):
    from memsom.lifecycle import forget as _forget
    params, _w = _forget.load_params(Path(memory_dir) / ".weights" / "canonical.json")
    return params


def _mtime_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


# --- the importer -------------------------------------------------------------

def import_memory_dir(conn, memory_dir, *, dry_run: bool = True, params=None) -> dict:
    """Import every memory/*.md and memory/projects/*.md (excluding the
    generated MEMORY.md / INDEX.md) into memsom.

    *params* is the store's runtime-param dict (forget.load_params); None
    loads it from ``<memory_dir>/.weights/canonical.json``.  Only
    ``feedback_born_unindexed`` is read here.

    Returns stats: {total_files, created, updated, skipped, tombstoned, ...,
    dedup, quarantined, born_unindexed}.
    Atomic: all writes happen in one transaction (or none, if dry_run) — with
    ONE exception: if the reconcile-deletion sweep tombstones a fact node this
    run, the Behavior-4 stale cascade for its dependents (docs/facts-design.md)
    runs in its own follow-up transaction(s) immediately after, once the sweep
    above has already committed (see the comment at the call site).
    """
    memory_dir = Path(memory_dir)
    if not memory_dir.is_dir():
        raise ValueError(f"memory dir is not a directory: {memory_dir}")

    index_path = memory_dir / "MEMORY.md"
    primary = (parse_primary_index(index_path.read_text(encoding="utf-8"))
               if index_path.exists() else {})
    if params is None:
        params = _load_import_params(memory_dir)
    born_rule = bool(params.get("feedback_born_unindexed", True))

    # Self-heal duplicate stems FIRST (the additive-sync leftover — see
    # DuplicateMemoryStem): identical shallow copies are deleted, differing
    # older copies quarantined.  Dry-run only counts; iter_memory_files picks
    # the same canonical copy either way, so the import never freezes on one.
    heal = heal_duplicates(memory_dir, dry_run=dry_run)

    files = iter_memory_files(memory_dir)
    stats = {"total_files": len(files), "created": 0, "updated": 0,
             "skipped": 0, "tombstoned": 0, "swept": 0, "refused_resurrect": 0,
             "stale_cascaded": 0, "indexed": 0, "deindexed": 0,
             "dedup": heal["dedup"], "quarantined": heal["quarantined"],
             "duplicates": heal["duplicates"], "born_unindexed": 0}

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

    # (nid, opath) tombstoned by the reconcile-deletion sweep below, THIS run.
    # Behavior 4 (docs/facts-design.md) cascades stale from these AFTER the
    # sweep's own transaction commits — see the comment at the bottom of this
    # function for why it must not run nested inside it.
    swept_ids = []
    new_ids, dead_ids = [], []   # retrieval-index upkeep, applied after commit

    # RMW-OK: runs inside import_memory_dir's BEGIN IMMEDIATE (opened at the call site below)
    def _do():
        for path in files:
            rel = path.name                      # bridge_path key, e.g. user_adhd.md
            stem = path.stem
            raw = path.read_text(encoding="utf-8")
            # fm_flat: type/section/index_*/salience/pin nested under metadata: by
            # the Claude Code stamper are read as if flat (the file is NOT rewritten)
            fm = fm_flat(split_frontmatter(raw)[0])
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
            # Explicit withdrawal: a file that says `section: none` (or
            # `index: false`) in its OWN frontmatter is deliberately out of the
            # index. It beats the curated line AND the stamped fallback — the
            # fallback exists to stop accidental unfiling, and this is the one
            # way to unfile on purpose without deleting the file.
            if unsectioned_by_frontmatter(fm):
                section = None
            existing = _live_node_for_path(conn, rel)
            # Anti-creep (born_unindexed docstring): a NEW feedback file with
            # no `why_own_line:` and no curated line is held out of the index
            # and marked `index_pending: needs_cluster` so the render can
            # record WHY it is absent (shed.json) and the next unchanged
            # re-import keeps holding it.
            pending = None
            if stem.startswith(FEEDBACK_PREFIX):
                prev_meta = _stored_index_meta(conn, rel)
                if born_unindexed(stem, fm, section, curated=rel in primary,
                                  existing=existing, prev_meta=prev_meta,
                                  enabled=born_rule):
                    section = None
                    pending = NEEDS_CLUSTER
                    stats["born_unindexed"] += 1
            # memory_subdir: "projects" for files under projects/, absent for
            # flat files — the digest links each entry relative to the index
            # it renders into (projects/INDEX.md vs MEMORY.md) from this key.
            # Node identity (bridge_path) stays the BASENAME, so moving a file
            # between the two levels is an edit of the same memory, not a new one.
            stamped = stamp_fm(raw, section=section, index_title=title, index_hook=hook,
                               memory_subdir=memory_subdir(memory_dir, path),
                               index_pending=pending)
            new_hash = memsom_chunking.content_hash(stamped)

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
                dead_ids.append(existing[0])
                stats["tombstoned"] += 1
                stats["updated"] += 1
            else:
                stats["created"] += 1

            # MS-20: `channel` above is read from the file's OWN frontmatter
            # `type:` (via CHANNEL_BY_TYPE) -- the file body dictating its own
            # trust channel is the inverse of invariant 1. Route it through
            # the F-13/F-14 guards the way every other stamping entry point
            # does, so MEMDAG_CHANNEL_CEILING actually bounds what a memory
            # file can claim, and the label always matches RANK[channel].
            nid = memsom_ingest.mint_node(conn, stamped, channel, source_ref=f"memory:{stem}")
            conn.execute(
                "UPDATE nodes SET bridge_path = ?, bridge_mtime = ?, content_hash = ? "
                "WHERE id = ?",
                (rel, _mtime_sig(path), new_hash, nid),
            )
            new_ids.append(nid)
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
                swept_ids.append((nid, opath))
                dead_ids.append(nid)

    if dry_run:
        _do()
    else:
        with conn:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            _do()
        # Retrieval index upkeep — after the commit, for the same reason the
        # stale cascade below waits (index/deindex open their own `with conn:`).
        sync_index(conn, new_ids, dead_ids, stats)

        # Behavior 4 (docs/facts-design.md): a swept node may be a fact other
        # facts depends_on. Its dependents must go STALE, never revoked or
        # tombstoned again — "selling the GPU retires the GPU fact and
        # *questions* the TPS fact; it does not destroy it." mark_stale_cascade
        # walks the SAME edges table relate_fact_deps materialised (parent =
        # this swept node), so sweeping a node nothing depends_on is a no-op
        # cascade beyond marking itself stale too (harmless — stale and
        # tombstoned are orthogonal flags; see memsom.lifecycle.stale's
        # module docstring).
        #
        # This MUST run after the sweep's own transaction above has already
        # committed: mark_stale_cascade opens its own `with conn:` block, and
        # nesting that inside the still-open BEGIN IMMEDIATE would let
        # sqlite3's commit-on-`with`-exit end the outer transaction early,
        # silently breaking this function's documented atomicity (the same
        # reason memsom_ingest.ingest_text fires on_reingest_supersede only
        # AFTER its own node-insert `with conn:` block has exited).
        if swept_ids:
            # MS-35: the UPDATE above only tombstones the SWEPT node itself --
            # any live DERIVED descendant (a compose()/derive_node() answer
            # that quoted this memory) was left live, and a live derived node
            # is exactly what distill/reflex export into training weights.
            #
            # Scoped to channel='agent-derived' rather than a bare
            # revoke_cascade: the SAME edges table also carries depends_on
            # fact-dependency edges (relate_fact_deps), and Behavior 4
            # (docs/facts-design.md, test_deleted_fact_file_marks_dependent_
            # stale_not_tombstoned) requires a dependent FACT to go stale,
            # never tombstoned, when the fact it depends on is retired. A
            # derive_node()/compose() descendant is unconditionally stamped
            # channel='agent-derived' by construction, so filtering on it is
            # exact -- it reaches training-export leakage without touching a
            # depends_on dependent, which the stale cascade below (unchanged)
            # already handles correctly.
            for nid, opath in swept_ids:
                ts = memsom.now_iso()
                for did, dchannel, dtombstoned in memsom.cascade_set(conn, nid):
                    if dchannel == "agent-derived" and not dtombstoned:
                        conn.execute(
                            "UPDATE nodes SET tombstoned = 1, tombstoned_at = ?,"
                            " revoke_reason = ? WHERE id = ? AND tombstoned = 0",
                            (ts, f"cascade from node {nid} (bridge reconcile,"
                                 f" source file removed: {opath})", did))
            from memsom.lifecycle import stale as memsom_stale
            for nid, opath in swept_ids:
                stats["stale_cascaded"] += memsom_stale.mark_stale_cascade(
                    conn, nid, f"dependency retired: source file removed ({opath})")

    return stats


# --- CLI ----------------------------------------------------------------------



def _print_stats(stats, dry_run):
    mode = "DRY-RUN (no writes)" if dry_run else "APPLIED"
    f, l = stats["files"], stats["literals"]
    print(f"[bridge-import] {mode}")
    print(f"  files seen     : {f['total_files']}")
    print(f"  files created  : {f['created']}")
    print(f"  files updated  : {f['updated']} (old tombstoned: {f['tombstoned']})")
    print(f"  files skipped  : {f['skipped']} (unchanged)")
    print(f"  deleted swept  : {f['swept']} (source file gone -> node tombstoned)")
    if f.get("dedup") or f.get("quarantined"):
        verb = "would heal" if dry_run else "healed"
        print(f"  dup stems      : {verb} {f['dedup']} identical copy(ies) deleted, "
              f"{f['quarantined']} differing copy(ies) -> .weights/dup_quarantine/")
    if f.get("born_unindexed"):
        print(f"  born unindexed : {f['born_unindexed']} new feedback file(s) held out "
              f"of MEMORY.md (no why_own_line:; merge into a feedback_cluster_*)")
    if f.get("refused_resurrect"):
        print(f"  RESURRECT BLK  : {f['refused_resurrect']} (redacted node's file "
              f"resurfaced -> refused + file unlinked)")
    if f.get("stale_cascaded"):
        print(f"  stale cascaded : {f['stale_cascaded']} node(s) (dependents of a "
              f"deleted fact — docs/facts-design.md Behavior 4)")
    print(f"  literals       : {l['total']} total | {l['created']} created | "
          f"{l['skipped']} skipped | {l['tombstoned']} tombstoned")
    e = stats.get("edges")
    if e is not None:
        verb = "would relate" if dry_run else "created"
        print(f"  wikilink edges : {e['edges']} {verb} | {e['resolved']} resolved | "
              f"{e['unresolved']} unresolved | {e['skipped_self']} self-links skipped")
    fe = stats.get("fact_edges")
    if fe is not None:
        verb = "would create" if dry_run else "created"
        print(f"  fact deps      : {fe['edges']} {verb} | {fe['resolved']} resolved | "
              f"{fe['unresolved']} unresolved | {fe['skipped_self']} self-deps skipped")


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
        # FAILOPEN: allowed, an unreconfigurable stream keeps its default encoding.
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="memsom_bridge_import", description=__doc__)
    ap.add_argument("memory_dir", nargs="?", default=None)
    ap.add_argument("--apply", action="store_true", help="apply (default: dry-run)")
    main_args = ap.parse_args(argv)
    _cmd_import(main_args)


if __name__ == "__main__":
    main()
