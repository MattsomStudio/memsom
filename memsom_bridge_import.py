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
  - Idempotent: keyed on nodes.obsidian_path (the filename) + content_hash.  A
    re-run with no file change creates nothing.  A changed file tombstones the old
    node and inserts a new one (append-only ethos).

Library discipline: the import_* functions never print or sys.exit; only main()
and the _cmd_* wrapper do I/O.  Frozen core (memsom.py) is untouched — this is a
pure consumer of insert_node + the obsidian_path/content_hash columns.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import memsom
import memsom_ingest
import memsom_obsidian


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

    obsidian_path / obsidian_mtime come from memsom_obsidian; content_hash from
    memsom_ingest; the stale triple + source_supersedes/stale_log from memsom_stale
    (so the verification-staleness pass + the render's stale-awareness have their
    columns on BOTH machines via the existing bridge migrate chain).  All additive
    nullable columns (frozen core never reads them), so this is safe to call
    standalone or via migrate_all.
    """
    memsom_obsidian.migrate(conn)
    memsom_ingest.migrate(conn)
    import memsom_stale          # lazy: mirror memsom_ingest's defensive import style
    memsom_stale.migrate(conn)   # nodes.{stale,stale_at,stale_reason} + supersedes/log


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
    stats = {"created": 0, "tombstoned": 0, "skipped": 0, "total": 0}
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
            if sref in existing:
                stats["skipped"] += 1
                continue
            stats["created"] += 1
            if dry_run:
                continue
            content = _literal_content(section, payload)
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


def import_all(conn, memory_dir, *, dry_run: bool = True) -> dict:
    """Import both the per-file memories and the file-less literal index lines."""
    files = import_memory_dir(conn, memory_dir, dry_run=dry_run)
    lits = import_literals(conn, memory_dir, dry_run=dry_run)
    return {"files": files, "literals": lits}


# --- DB helpers ---------------------------------------------------------------

def _live_node_for_path(conn, rel: str):
    """Return (id, content_hash) of the live node for *rel*, or None."""
    return conn.execute(
        "SELECT id, content_hash FROM nodes "
        "WHERE obsidian_path = ? AND tombstoned = 0 ORDER BY id DESC LIMIT 1",
        (rel,),
    ).fetchone()


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
             "skipped": 0, "tombstoned": 0, "swept": 0}

    def _do():
        for path in files:
            rel = path.name                      # obsidian_path key, e.g. user_adhd.md
            stem = path.stem
            raw = path.read_text(encoding="utf-8")
            fm = fm_top_level(split_frontmatter(raw)[0])
            channel = CHANNEL_BY_TYPE.get(memory_type(stem, fm), DEFAULT_CHANNEL)
            title, hook, section = primary.get(rel, (None, None, None))
            stamped = stamp_fm(raw, section=section, index_title=title, index_hook=hook)
            new_hash = memsom_ingest._content_hash(stamped)

            existing = _live_node_for_path(conn, rel)
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
                "UPDATE nodes SET obsidian_path = ?, obsidian_mtime = ?, content_hash = ? "
                "WHERE id = ?",
                (rel, _mtime_sig(path), new_hash, nid),
            )

        # reconcile deletions: tombstone live file-backed nodes whose source
        # file has vanished. The loop above only ever touches files that EXIST,
        # so without this a deleted memory's node lingers live forever (the gap
        # that once orphaned a deleted note's node). Literals (obsidian_path
        # IS NULL) are excluded here — import_literals reconciles those against
        # the index.
        present = {p.name for p in files}
        gone = conn.execute(
            "SELECT id, obsidian_path FROM nodes "
            "WHERE source_ref LIKE 'memory:%' AND source_ref NOT LIKE 'memory:literal:%' "
            "AND tombstoned = 0 AND obsidian_path IS NOT NULL"
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
        return Path.home() / ".claude" / "projects"
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
    print(f"  literals       : {l['total']} total | {l['created']} created | "
          f"{l['skipped']} skipped | {l['tombstoned']} tombstoned")


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
