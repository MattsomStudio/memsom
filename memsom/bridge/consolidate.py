"""memsom consolidate-feedback / consolidate-projects — the weekly merge proposer.

Anti-creep, mechanism 3 (see ARCHITECTURE.md "Anti-creep").  The importer now
births new feedback files UNINDEXED and the digest caps each section's bytes,
so MEMORY.md cannot grow one line per lesson any more — but the lessons still
have to land somewhere a session will read them.  That somewhere is the body
of the nearest ``feedback_cluster_*`` file.  This module proposes that merge
(and the analogous project one) and, on ``--apply``, performs it.  It never
deletes a file: a merged feedback file gets ``section: none`` and stays on
disk, searchable; a closed subproject gets ``index: false`` and a summary row
in its parent overview.

Both commands default to DRY-RUN: a markdown report on stdout plus a JSON
proposal file at ``<memory_dir>/.weights/consolidate_proposals.json``.

Similarity is BM25 only (``retrieval.retrieve.bm25`` over the store's own
postings — the same index ``memsom retrieve`` uses), restricted to the live
cluster nodes.  No model is loaded: this runs from a scheduled task next to
``bridge-render`` and must stay sub-second.

Scheduling: run it from whatever already drives your weekly sweep —

    # Windows (Task Scheduler, weekly, Sunday 18:00)
    schtasks /Create /SC WEEKLY /D SUN /ST 18:00 /TN memsom-consolidate ^
        /TR "\"<abs path>\\memsom.exe\" consolidate-feedback"
    # macOS / Linux (cron, Sunday 18:00)
    0 18 * * 0  /abs/path/to/venv/bin/memsom consolidate-feedback

Dry-run output is the review queue; add ``--apply`` once the proposals look
right (or leave it dry and apply by hand — the JSON carries every target).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import memsom
from memsom.bridge import bridge_import as bi
from memsom.storage import schema as memsom_schema

PROPOSALS_NAME = "consolidate_proposals.json"
DEFAULT_MIN_AGE_DAYS = 14
ABSORBED_HEADING = "## Absorbed"
THREADS_HEADING = "## Threads"
CLOSED_HEADING = "## Closed threads"


# --- shared helpers -----------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _parse_iso(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _live_memory_nodes(conn):
    """{stem: {"id", "content", "fm", "born", "tier"}} for every live
    file-backed memory node."""
    has_seen = memsom_schema.column_exists(conn, "nodes", "forget_first_seen")
    has_tier = memsom_schema.column_exists(conn, "nodes", "forget_tier")
    bcol = "COALESCE(forget_first_seen, created_at)" if has_seen else "created_at"
    tcol = "forget_tier" if has_tier else "NULL"
    rows = conn.execute(
        f"SELECT id, source_ref, content, {bcol}, {tcol} FROM nodes "
        "WHERE tombstoned = 0 AND source_ref LIKE 'memory:%' "
        "AND source_ref NOT LIKE 'memory:literal:%' AND bridge_path IS NOT NULL"
    ).fetchall()
    out = {}
    for nid, sref, content, born, tier in rows:
        stem = sref.split(":", 1)[1]
        fm = bi.fm_top_level(bi.split_frontmatter(content or "")[0])
        out[stem] = {"id": nid, "content": content or "", "fm": fm,
                     "born": _parse_iso(born), "tier": tier}
    return out


def _age_days(path, node):
    born = node["born"] if node else None
    if born is None:
        born = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (_now() - born).total_seconds() / 86400.0


def _files_by_stem(memory_dir):
    return {p.stem: p for p in bi.iter_memory_files(memory_dir)}


def _write_proposals(memory_dir, kind, proposals):
    weights = Path(memory_dir) / ".weights"
    weights.mkdir(parents=True, exist_ok=True)
    path = weights / PROPOSALS_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[kind] = {"generated_at": _now().isoformat(timespec="seconds"),
                  "proposals": proposals}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def _append_under_heading(text, heading, bullet):
    """Append *bullet* under the LAST occurrence of *heading* in *text*, creating
    the heading at the end when absent.  Returns the new text."""
    lines = text.rstrip("\n").split("\n")
    idx = max((i for i, ln in enumerate(lines) if ln.strip() == heading), default=-1)
    if idx == -1:
        lines += ["", heading, bullet]
        return "\n".join(lines) + "\n"
    # insert after the last non-blank line of that section
    end = idx + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    insert_at = end
    while insert_at > idx + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, bullet)
    return "\n".join(lines) + "\n"


# --- consolidate-feedback -----------------------------------------------------

def propose_feedback(conn, memory_dir, *, min_age_days=DEFAULT_MIN_AGE_DAYS):
    """For each INDEXED non-cluster feedback file older than *min_age_days*
    with no `why_own_line:`, the nearest feedback_cluster_* by BM25.

    Returns a list of {"stem", "path", "age_days", "cluster", "score",
    "description"}; cluster is None when no cluster shares a term with it.
    """
    from memsom.retrieval import retrieve as memsom_retrieve
    nodes = _live_memory_nodes(conn)
    files = _files_by_stem(memory_dir)
    clusters = {stem: n for stem, n in nodes.items()
                if stem.startswith(bi.CLUSTER_PREFIX) and stem in files}
    cluster_by_id = {n["id"]: stem for stem, n in clusters.items()}
    proposals = []
    for stem in sorted(files):
        if not stem.startswith(bi.FEEDBACK_PREFIX) or stem.startswith(bi.CLUSTER_PREFIX):
            continue
        node = nodes.get(stem)
        if node is None:
            continue                         # not imported yet — nothing to judge
        fm = node["fm"]
        if not fm.get("section") or bi.unsectioned_by_frontmatter(fm):
            continue                         # already out of the index
        if bi.has_own_line_reason(fm):
            continue                         # earned its line
        age = _age_days(files[stem], node)
        if age < min_age_days:
            continue
        best, score = None, 0.0
        if clusters:
            query = " ".join(x for x in (fm.get("name"), fm.get("description"),
                                         bi.split_frontmatter(node["content"])[1]) if x)
            for nid, sc in memsom_retrieve.bm25(conn, query, k=max(50, len(nodes))):
                if nid in cluster_by_id:
                    best, score = cluster_by_id[nid], sc
                    break
        proposals.append({"stem": stem, "path": str(files[stem]),
                          "age_days": round(age, 1), "cluster": best,
                          "score": round(score, 3),
                          "description": fm.get("description", "")})
    return proposals


def apply_feedback(memory_dir, proposals) -> dict:
    """Absorb each proposal with a cluster: append `- [[stem]] — <description>`
    under `## Absorbed <date>` in the cluster body and set `section: none` on
    the file.  Never deletes.  Returns {"absorbed": n, "skipped": n}."""
    files = _files_by_stem(memory_dir)
    today = _now().date().isoformat()
    out = {"absorbed": 0, "skipped": 0, "applied": []}
    for p in proposals:
        cluster = p.get("cluster")
        stem = p["stem"]
        if not cluster or cluster not in files or stem not in files:
            out["skipped"] += 1
            continue
        cpath, fpath = files[cluster], files[stem]
        desc = (p.get("description") or "").strip()
        bullet = f"- [[{stem}]] — {desc}" if desc else f"- [[{stem}]]"
        ctext = cpath.read_text(encoding="utf-8")
        if f"[[{stem}]]" not in ctext:
            cpath.write_text(_append_under_heading(ctext, f"{ABSORBED_HEADING} {today}", bullet),
                             encoding="utf-8", newline="\n")
        ftext = fpath.read_text(encoding="utf-8")
        fpath.write_text(bi.stamp_fm(ftext, section="none"), encoding="utf-8", newline="\n")
        out["absorbed"] += 1
        out["applied"].append({"stem": stem, "cluster": cluster})
    return out


def _feedback_report(proposals, *, applied=None):
    lines = ["# consolidate-feedback", ""]
    if not proposals:
        lines.append("Nothing to merge: no indexed, un-justified feedback file past min-age.")
        return "\n".join(lines) + "\n"
    lines += ["| file | proposed cluster | bm25 | age (d) |", "|---|---|---|---|"]
    for p in proposals:
        lines.append(f"| {p['stem']} | {p['cluster'] or '(no match)'} | "
                     f"{p['score']} | {p['age_days']} |")
    lines.append("")
    if applied is None:
        lines.append("DRY-RUN — re-run with --apply to absorb (appends to the cluster body, "
                     "sets `section: none` on the file; nothing is deleted).")
    else:
        lines.append(f"APPLIED: {applied['absorbed']} absorbed, {applied['skipped']} skipped "
                     f"(no cluster match).")
    return "\n".join(lines) + "\n"


# --- consolidate-projects -----------------------------------------------------

def propose_projects(conn, memory_dir, *, min_age_days=DEFAULT_MIN_AGE_DAYS):
    """Subprojects (`projects/<slug>/project_<slug>_<sub>.md`) that are
    `status: closed` (or forget tier cold) and older than *min_age_days*,
    still indexed (no `index: false`), with a parent overview present."""
    nodes = _live_memory_nodes(conn)
    files = _files_by_stem(memory_dir)
    proposals = []
    for stem in sorted(files):
        path = files[stem]
        sub = bi.memory_subdir(memory_dir, path) or ""
        if not sub.startswith(bi.PROJECTS_SUBDIR + "/"):
            continue
        slug = sub.split("/", 1)[1]
        parent_stem = f"project_{slug}"
        if stem == parent_stem or not stem.startswith(parent_stem + "_"):
            continue
        if parent_stem not in files:
            continue
        node = nodes.get(stem)
        fm = node["fm"] if node else bi.fm_top_level(
            bi.split_frontmatter(path.read_text(encoding="utf-8"))[0])
        if (fm.get("index") or "").strip().lower() in ("false", "no", "0"):
            continue
        closed = (fm.get("status") or "").strip().lower() == "closed"
        cold = bool(node) and node.get("tier") == "cold"
        if not (closed or cold):
            continue
        age = _age_days(path, node)
        if age < min_age_days:
            continue
        proposals.append({"stem": stem, "path": str(path), "parent": parent_stem,
                          "parent_path": str(files[parent_stem]),
                          "why": "closed" if closed else "cold",
                          "age_days": round(age, 1),
                          "name": fm.get("name", stem),
                          "description": fm.get("description", "")})
    return proposals


def _append_thread_row(text, p, today):
    """Append a summary row to the parent's `## Threads` table, or a bullet
    under `## Closed threads` when there is no table."""
    lines = text.rstrip("\n").split("\n")
    idx = next((i for i, ln in enumerate(lines) if ln.strip() == THREADS_HEADING), -1)
    if idx != -1:
        # find the table: header row + separator, then its last row
        i = idx + 1
        while i < len(lines) and not lines[i].lstrip().startswith("|"):
            if lines[i].startswith("## "):
                break
            i += 1
        if i < len(lines) and lines[i].lstrip().startswith("|"):
            ncols = len([c for c in lines[i].strip().strip("|").split("|")])
            j = i
            while j + 1 < len(lines) and lines[j + 1].lstrip().startswith("|"):
                j += 1
            cells = [f"[[{p['stem']}]]", f"{p['why']} {today}", p.get("description", "")]
            cells = (cells + [""] * ncols)[:ncols] if ncols >= 3 else \
                [cells[0], f"{p['why']} {today} — {p.get('description', '')}"][:ncols]
            lines.insert(j + 1, "| " + " | ".join(cells) + " |")
            return "\n".join(lines) + "\n"
    bullet = f"- [[{p['stem']}]] — {p['why']} {today}"
    if p.get("description"):
        bullet += f" — {p['description']}"
    return _append_under_heading("\n".join(lines) + "\n", CLOSED_HEADING, bullet)


def apply_projects(memory_dir, proposals) -> dict:
    today = _now().date().isoformat()
    out = {"withdrawn": 0, "skipped": 0, "applied": []}
    for p in proposals:
        ppath, spath = Path(p["parent_path"]), Path(p["path"])
        if not ppath.exists() or not spath.exists():
            out["skipped"] += 1
            continue
        ptext = ppath.read_text(encoding="utf-8")
        if f"[[{p['stem']}]]" not in ptext:
            ppath.write_text(_append_thread_row(ptext, p, today), encoding="utf-8",
                             newline="\n")
        stext = spath.read_text(encoding="utf-8")
        spath.write_text(bi.stamp_fm(stext, index="false"), encoding="utf-8", newline="\n")
        out["withdrawn"] += 1
        out["applied"].append({"stem": p["stem"], "parent": p["parent"]})
    return out


def _projects_report(proposals, *, applied=None):
    lines = ["# consolidate-projects", ""]
    if not proposals:
        lines.append("Nothing to fold: no closed/cold subproject past min-age.")
        return "\n".join(lines) + "\n"
    lines += ["| subproject | parent | why | age (d) |", "|---|---|---|---|"]
    for p in proposals:
        lines.append(f"| {p['stem']} | {p['parent']} | {p['why']} | {p['age_days']} |")
    lines.append("")
    if applied is None:
        lines.append("DRY-RUN — re-run with --apply to fold each into its parent's "
                     "`## Threads` table (or `## Closed threads`) and set `index: false`.")
    else:
        lines.append(f"APPLIED: {applied['withdrawn']} withdrawn, {applied['skipped']} skipped.")
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------

def _reconfigure():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        # FAILOPEN: allowed, an unreconfigurable stream keeps its default encoding.
        except Exception:
            pass


def _cmd_feedback(args):
    _reconfigure()
    mem = Path(args.memory_dir) if args.memory_dir else bi.default_memory_dir()
    conn = memsom.get_connection()
    try:
        bi.migrate(conn)
        proposals = propose_feedback(conn, mem, min_age_days=args.min_age_days)
    finally:
        conn.close()
    applied = apply_feedback(mem, proposals) if args.apply else None
    path = _write_proposals(mem, "feedback", proposals)
    sys.stdout.write(_feedback_report(proposals, applied=applied))
    print(f"proposals: {path}")
    return 0


def _cmd_projects(args):
    _reconfigure()
    mem = Path(args.memory_dir) if args.memory_dir else bi.default_memory_dir()
    conn = memsom.get_connection()
    try:
        bi.migrate(conn)
        proposals = propose_projects(conn, mem, min_age_days=args.min_age_days)
    finally:
        conn.close()
    applied = apply_projects(mem, proposals) if args.apply else None
    path = _write_proposals(mem, "projects", proposals)
    sys.stdout.write(_projects_report(proposals, applied=applied))
    print(f"proposals: {path}")
    return 0


def register(sub) -> None:
    for name, func, help_ in (
        ("consolidate-feedback", _cmd_feedback,
         "propose (or --apply) merging aged feedback files into feedback_cluster_* bodies"),
        ("consolidate-projects", _cmd_projects,
         "propose (or --apply) folding closed/cold subprojects into the parent overview"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("memory_dir", nargs="?", default=None,
                       help="memory dir (default: auto-detected ~/.claude/projects/*/memory)")
        p.add_argument("--apply", action="store_true",
                       help="perform the merge (default: dry-run report + proposals JSON)")
        p.add_argument("--min-age-days", type=float, default=DEFAULT_MIN_AGE_DAYS,
                       help=f"only files older than this (default {DEFAULT_MIN_AGE_DAYS})")
        p.set_defaults(func=func)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="memsom_consolidate", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    register(sub)
    args = ap.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
