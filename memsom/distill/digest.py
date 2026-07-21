"""memsom_digest — render the always-on MEMORY.md digest from memsom (Phase 3).

This is the piece that lets memsom be the store-of-record while the harness-native
always-on MEMORY.md survives: it queries the bridge-imported memory nodes and
renders the sectioned `- [Title](file.md) — hook` index the Claude Code harness
loads each session.

Selection (the forgetting layer decides what's "hot enough" to inject):
  - literal nodes (the file-less hand-authored index lines)  -> always rendered.
  - endorsed (pinned: user_/feedback_/personal_)             -> always rendered.
  - user-channel (project_/reference_) with forget_tier='hot' -> rendered.
  - user-channel 'cold' / un-sectioned                        -> dropped (still in
                                                                 the store, just
                                                                 out of context).

Budget: the rendered file must be <= 16,384 bytes (the harness loads it in full).
If over, the lowest-RS user lines are dropped first; pinned + literal lines are
never dropped.  If pinned+literal alone exceed the cap, DigestTooLarge is raised
(surfaced, never silently truncated).

SHADOW mode (Phase 3): write_shadow writes MEMORY.memsom.md NEXT TO the real
MEMORY.md (never overwrites it).  compare_index does the per-section file-set
equality check that is the cutover GO criterion.

Frozen core untouched; read-only over the DB (render/compare never write nodes).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import memsom
from memsom.storage import schema as memsom_schema
from memsom.bridge.bridge_import import split_frontmatter, fm_top_level, parse_index_entries, parse_primary_index, default_memory_dir
from memsom.lifecycle import forget as _forget

# Default section display order. Carries no user-specific taxonomy so the shipped
# module is identity-free; override with a comma-separated $MEMDAG_DIGEST_SECTIONS.
# Any section present in a memory file but absent here still renders (sorted, after
# the known ones), so a custom section never gets dropped — only reordered.
SECTIONS = [
    "About the User",
    "Personal context",
    "Hardware",
    "Current Setup & Learning",
    "Work",
    "Personal projects",
    "References",
    "Feedback",
]
BUDGET = 16384  # hard-fallback cap; live value = `memory_budget` in the store's
#                 canonical.json params (resolve_budget below / forget.load_params)


def resolve_budget(memory_dir):
    """The live byte cap for MEMORY.md: `memory_budget` from the store's
    canonical.json params, falling back to BUDGET when absent/invalid.  Resolved
    at call time — never bound into a signature default."""
    try:
        params, _ = _forget.load_params(
            Path(memory_dir) / ".weights" / "canonical.json")
        return int(params["memory_budget"])
    except Exception:
        return BUDGET
# Content floor (fail-safe): a render that keeps fewer than this fraction of the
# PRIOR MEMORY.md's index entries is rejected — a collapse that large means the
# store is wrong (empty/mispointed), not that the brain legitimately halved
# between two Stop hooks. Override with $MEMDAG_DIGEST_SHRINK_FLOOR (0..1);
# out-of-range values fall back to this default.
SHRINK_FLOOR = 0.5


def _shrink_floor() -> float:
    raw = os.environ.get("MEMDAG_DIGEST_SHRINK_FLOOR")
    if raw:
        try:
            v = float(raw)
            if 0.0 < v <= 1.0:
                return v
        except ValueError:
            pass
    return SHRINK_FLOOR


def _section_order():
    """Section display order: $MEMDAG_DIGEST_SECTIONS (comma-separated) if set,
    else the generic SECTIONS default."""
    env = os.environ.get("MEMDAG_DIGEST_SECTIONS")
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    return SECTIONS
# generic default; the real H1 is set per-user via $MEMDAG_DIGEST_TITLE so this
# shippable module carries no author identity.
DEFAULT_TITLE = "# Memory"


class DigestTooLarge(Exception):
    """Raised when pinned + literal content alone exceeds the byte budget."""


# --- read the bridge nodes ----------------------------------------------------

def _rows(conn):
    has_tier = memsom_schema.column_exists(conn, "nodes", "forget_tier")
    has_rs = memsom_schema.column_exists(conn, "nodes", "forget_rs")
    has_stale = memsom_schema.column_exists(conn, "nodes", "stale")
    tcol = "forget_tier" if has_tier else "NULL AS forget_tier"
    rcol = "forget_rs" if has_rs else "NULL AS forget_rs"
    scol = "stale" if has_stale else "0 AS stale"
    zcol = "stale_reason" if has_stale else "NULL AS stale_reason"
    # Taint gate from the ONE shared primitive (tombstoned/quarantined/redacted/
    # archived — each only when its column exists). The digest renders into the
    # ALWAYS-LOADED MEMORY.md, so it must exclude every taint dimension: a
    # redacted node's content is '' (its stem would leak as the fallback title),
    # and quarantined/archived nodes are out of every other read pool — they must
    # not resurface in the brain either.
    clauses, params = memsom_schema.taint_filter_clauses(conn)
    where = " AND ".join(clauses + ["source_ref LIKE 'memory:%'"])
    return conn.execute(
        f"SELECT content, channel, source_ref, {tcol}, {rcol}, {scol}, {zcol} "
        f"FROM nodes WHERE {where}",
        params,
    ).fetchall()


def _entry(content, channel, sref, tier, rs, stale=0, stale_reason=None):
    fm_lines, body, _ = split_frontmatter(content or "")
    fm = fm_top_level(fm_lines)
    is_literal = (sref.startswith("memory:literal:")
                  or str(fm.get("literal", "")).lower() in ("true", "1", "yes"))
    section = fm.get("section") or None
    if is_literal:
        return {"kind": "literal", "section": section, "line": body.strip(),
                "channel": channel,
                "stale": bool(stale), "stale_reason": stale_reason}
    stem = sref.split(":", 1)[1] if sref.startswith("memory:") else sref
    # prefer the curated MEMORY.md title + hook (terser, byte-matches the
    # hand-maintained index); fall back to frontmatter name + a LENGTH-CAPPED
    # description so a node imported without curated text can't bloat the file.
    name = fm.get("index_title") or fm.get("name", stem)
    hook = fm.get("index_hook")
    if hook and "⚠" in hook:                  # defensive: never re-emit a baked-in
        hook = hook.split("⚠", 1)[0].rstrip() or None   # render marker (see bridge bug)
    if not hook:
        d = fm.get("description", "")
        hook = (d[:70].rstrip() + "…") if len(d) > 71 else d
    # A fact's hook IS its current value (docs/facts-design.md): the digest is
    # the always-loaded surface, so the value must be readable without opening
    # the file. Verified date included — a fact whose freshness you can't see
    # is a number you can't trust.
    if (fm.get("type") or "").strip() == "fact" and fm.get("value") is not None:
        val = f"{fm['value']} {fm['unit']}" if fm.get("unit") else str(fm["value"])
        lv = fm.get("last-verified")
        hook = f"{val} (verified {lv})" if lv else val
    return {"kind": "file", "section": section, "stem": stem,
            "name": name, "desc": hook,
            "pinned": channel == "endorsed", "tier": tier or "hot",
            "rs": rs, "channel": channel,
            "stale": bool(stale), "stale_reason": stale_reason}


def _select_hot(entries):
    """Entries that belong in the always-on digest."""
    out = []
    for e in entries:
        if e["kind"] == "literal":
            out.append(e)                      # literals always render
        elif e["section"] and (e["pinned"] or e["tier"] == "hot"):
            out.append(e)                      # sectioned + (pinned or hot)
    return out


def _marker():
    """Inline staleness flag: a BARE glyph (cheap — ~4 bytes).  The reason lives in
    the droppable Needs Reverification section + `memsom verify-stale`, so a flag on
    a near-budget brain never evicts a real memory to make room for prose."""
    return " ⚠"


def _assemble(title, entries, *, include_reverify=True):
    by_sec = {}
    for e in entries:
        by_sec.setdefault(e["section"], []).append(e)
    lines = [title, ""]

    # Synthetic worklist: every stale note, as the FIRST block under the H1 (a
    # glanceable "go re-check these" list).  Built from the stale flag — not any
    # node's section: — so it carries no real files and compare_index ignores it.
    stale = [e for e in entries if e.get("stale")]
    if include_reverify and stale:
        lines.append("## Needs Reverification")
        for e in sorted([x for x in stale if x["kind"] == "file"],
                        key=lambda x: x["stem"]):
            lines.append(f"- [{e['name']}]({e['stem']}.md) — "
                         f"{e['stale_reason'] or 'unverified'}")
        for e in [x for x in stale if x["kind"] == "literal"]:
            lines.append(f"- {e['line']} — {e['stale_reason'] or 'unverified'}")
        lines.append("")

    secs = _section_order()
    order = secs + sorted(s for s in by_sec if s and s not in secs)
    for sec in order:
        if sec not in by_sec:
            continue
        lines.append(f"## {sec}")
        for e in [x for x in by_sec[sec] if x["kind"] == "literal"]:
            mk = _marker() if e.get("stale") else ""
            lines.append(e["line"] + mk)
        for e in sorted([x for x in by_sec[sec] if x["kind"] == "file"],
                        key=lambda x: x["stem"]):
            hook = f" — {e['desc']}" if e["desc"] else ""
            mk = _marker() if e.get("stale") else ""
            lines.append(f"- [{e['name']}]({e['stem']}.md){hook}{mk}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_digest(conn, *, title=None, budget=None, excluded_out=None):
    """Render the MEMORY.md digest string from the live bridge nodes.

    `excluded_out`: pass a list to learn WHICH memories this render left out of
    MEMORY.md and why.  Extended with {"stem", "reason", "rs"} dicts, where
    reason is one of:
      "cold"        — the forgetting layer demoted it (_select_hot skipped it)
      "unsectioned" — no section:, so there is nowhere in the index to put it
      "budget"      — it fit the rules but not the byte cap (lowest RS evicted
                      first; these are listed in the order they were dropped)
    Default None keeps the old behaviour byte-for-byte.

    Why this exists: memsom RENDERS MEMORY.md but never writes canonical.json
    (`~/.claude/episodic/mem_weights.py` is its sole author), so every exclusion
    decided here was invisible to the weights layer the audit reads.  The result
    was 86 memories on disk, absent from the index, with no record anywhere of
    why — indistinguishable from a corrupted index.  Reporting exclusions lets
    the caller persist a receipt, so "not in MEMORY.md" is always explainable.
    """
    if budget is None:
        budget = BUDGET
    title = title or os.environ.get("MEMDAG_DIGEST_TITLE", DEFAULT_TITLE)
    all_entries = [_entry(*r) for r in _rows(conn)]
    hot = _select_hot(all_entries)
    if excluded_out is not None:
        # Everything _select_hot filtered out, with the rule that filtered it.
        # Literals always render, so only files can appear here.
        live_ids = {id(e) for e in hot}
        for e in all_entries:
            if e["kind"] != "file" or id(e) in live_ids:
                continue
            reason = "cold" if e["section"] else "unsectioned"
            excluded_out.append({"stem": e["stem"], "reason": reason,
                                 "rs": e["rs"]})
    # Read-time fact resolution (docs/facts-design.md Phase 2): substitute
    # [[fact_*]] in hooks and literal lines with the CURRENT value. Must happen
    # BEFORE the budget loop below — resolved values change line length, and
    # eviction has to see the real rendered sizes, not the placeholder's.
    from memsom.bridge import facts as memsom_facts
    for e in hot:
        if e["kind"] == "literal":
            e["line"] = memsom_facts.resolve_fact_refs(conn, e["line"])
        elif e.get("desc"):
            e["desc"] = memsom_facts.resolve_fact_refs(conn, e["desc"])
    # droppable = non-pinned user files, lowest RS first (dropped in THIS order)
    droppable = sorted([e for e in hot if e["kind"] == "file" and not e["pinned"]],
                       key=lambda e: (e["rs"] if e["rs"] is not None else 0.0))
    dropped = set()  # ids of dropped entries
    include_reverify = True  # the worklist section is the FIRST thing shed if tight
    while True:
        live = [e for e in hot if id(e) not in dropped]
        text = _assemble(title, live, include_reverify=include_reverify)
        if len(text.encode("utf-8")) <= budget:
            if excluded_out is not None:
                # droppable order == drop order, so these read lowest-RS-first:
                # exactly the eviction sequence that ran.
                excluded_out.extend(
                    {"stem": e["stem"], "reason": "budget", "rs": e["rs"]}
                    for e in droppable if id(e) in dropped)
            return text
        if include_reverify:
            # the worklist is redundant with the inline ⚠ markers, so it sheds
            # first under budget pressure (the markers are the load-bearing signal).
            include_reverify = False
            continue
        nxt = next((e for e in droppable if id(e) not in dropped), None)
        if nxt is None:
            raise DigestTooLarge(
                f"pinned + literal content alone exceeds {budget} bytes")
        dropped.add(id(nxt))


def _entry_counts(text):
    """(file_entry_count, literal_entry_count) parsed from an index/digest text.

    Files are deduped (a filename linked twice is one memory); literals are
    counted per line.  Used by the content floor below — the H1 alone parses to
    (0, 0), which is exactly the empty-render signature the floor exists to catch.
    """
    files, literals = set(), 0
    for _sec, kind, payload in parse_index_entries(text or ""):
        if kind == "file":
            files.add(payload)
        else:
            literals += 1
    return len(files), literals


def validate(conn, *, budget=None, title=None, prior_text=None):
    """Export-boundary check: the digest must render, be non-empty, and fit budget.

    This is the Phase-6 cutover PRE-FLIGHT: the hook renders + validates, and only
    overwrites the real MEMORY.md when this returns [] — otherwise it leaves the
    existing good file in place (fail-safe, never fail-open).  Returns a list of
    problem dicts ([] = safe to write).

    Content floor: a render of an EMPTY (or mispointed) store produces just the H1
    — non-empty TEXT, so the byte checks alone would pass and write_live would
    overwrite the real brain with a one-line stub.  So a render with ZERO
    file/literal entries is always rejected, and when *prior_text* (the current
    on-disk MEMORY.md) is supplied, a render that keeps less than the shrink
    floor's fraction of the prior entries is rejected too — a stale-but-intact
    brain beats a freshly-blanked one every time."""
    if budget is None:
        budget = BUDGET
    try:
        text = render_digest(conn, title=title, budget=budget)
    except DigestTooLarge as exc:
        return [{"kind": "export-boundary", "detail": str(exc)}]
    except Exception as exc:  # any render failure must block the write, not crash it
        return [{"kind": "export-boundary", "detail": f"render failed: {exc!r}"}]
    problems = []
    if not text.strip():
        problems.append({"kind": "export-boundary", "detail": "rendered digest is empty"})
    size = len(text.encode("utf-8"))
    if size > budget:
        problems.append({"kind": "export-boundary",
                         "detail": f"digest {size} > {budget} byte budget"})
    new_files, new_lits = _entry_counts(text)
    new_total = new_files + new_lits
    if new_total == 0:
        problems.append({"kind": "export-boundary",
                         "detail": "rendered digest has zero file/literal entries "
                                   "(empty or mispointed store?)"})
    if prior_text is not None:
        prior_files, prior_lits = _entry_counts(prior_text)
        prior_total = prior_files + prior_lits
        floor = _shrink_floor()
        if prior_total > 0 and new_total < prior_total * floor:
            problems.append({
                "kind": "export-boundary",
                "detail": f"rendered digest keeps only {new_total}/{prior_total} "
                          f"entries of the existing MEMORY.md (< {floor:.0%} floor) "
                          f"— refusing to shrink the brain"})
    return problems


def write_shadow(conn, memory_dir, *, name="MEMORY.memsom.md", title=None):
    """Render and write the SHADOW digest next to the real MEMORY.md.

    Never touches the real MEMORY.md.  Returns (path, text).
    """
    # Same live budget the real render uses — otherwise the shadow trims (or
    # doesn't) against a different cap than the status line reports.
    text = render_digest(conn, title=title, budget=resolve_budget(memory_dir))
    path = Path(memory_dir) / name
    # write_bytes (not write_text): keep LF on Windows so the on-disk size matches
    # the budget accounting and the file's line endings match the original.
    path.write_bytes(text.encode("utf-8"))
    return path, text


def write_live(conn, memory_dir, *, name="MEMORY.md", title=None, budget=None):
    """CUTOVER write: validate, then overwrite the REAL MEMORY.md ONLY if valid.

    Fail-safe, never fail-open: on ANY validation problem the existing file is left
    exactly as-is and (False, problems) is returned — so a broken render can never
    blank or truncate the always-on brain.  On success writes atomically (tmp +
    replace) and returns (True, {"bytes", "path"}).  This is what the Phase-6 Stop
    hook calls; until that hook is wired, nothing invokes it.

    budget=None resolves the live `memory_budget` from this store's canonical.json
    (falling back to BUDGET) — callers that already loaded params pass it in.
    """
    if budget is None:
        budget = resolve_budget(memory_dir)
    path = Path(memory_dir) / name
    prior_text = None
    if path.exists():
        try:
            prior_text = path.read_text(encoding="utf-8")
        except OSError:
            prior_text = None  # unreadable prior: still enforce the zero-entry floor
    problems = validate(conn, budget=budget, title=title, prior_text=prior_text)
    if problems:
        return False, problems
    excluded = []
    text = render_digest(conn, title=title, budget=budget,
                         excluded_out=excluded)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # write_bytes (not write_text): keep LF on Windows so on-disk size == the
    # validated budget and the file's line endings match the original MEMORY.md.
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(path)
    # `excluded` is every memory THIS render left out of MEMORY.md, with the
    # reason — the caller persists it so an absent memory is always explainable.
    return True, {"bytes": len(text.encode("utf-8")), "path": str(path),
                  "excluded": excluded}


# --- verification (the cutover GO check) -------------------------------------

def index_sets(text):
    """{section: {"files": set(filenames), "literals": set(lines)}} from an index.

    Files counted are PRIMARY entries only (line-leading), so secondary inline
    links — which the digest never renders as their own line — are excluded on
    both sides of the comparison.  Literals come from the full entry parse.
    """
    out = {}
    for fname, (title, hook, section) in parse_primary_index(text).items():
        out.setdefault(section, {"files": set(), "literals": set()})["files"].add(fname)
    for sec, kind, payload in parse_index_entries(text):
        if kind == "literal":
            out.setdefault(sec, {"files": set(), "literals": set()})["literals"].add(payload)
    return out


def compare_index(real_text, shadow_text):
    """Per-section diff of FILE sets between two indexes (the GO criterion:
    'same files present per section').  Returns {} when equivalent.

    Each non-empty section entry reports missing_files (in real, absent from
    shadow) and extra_files (in shadow, absent from real).
    """
    a, b = index_sets(real_text), index_sets(shadow_text)
    diffs = {}
    for sec in sorted(set(a) | set(b)):
        af = a.get(sec, {}).get("files", set())
        bf = b.get(sec, {}).get("files", set())
        missing, extra = af - bf, bf - af
        if missing or extra:
            diffs[sec] = {"missing_files": sorted(missing),
                          "extra_files": sorted(extra)}
    return diffs


# --- CLI ----------------------------------------------------------------------

def _cmd_shadow(args):
    conn = memsom.get_connection()
    try:
        mem = Path(args.memory_dir) if args.memory_dir else default_memory_dir()
        path, text = write_shadow(conn, mem)
        size = len(text.encode("utf-8"))
        print(f"[digest] wrote shadow {path} ({size} / {resolve_budget(mem)} bytes)")
        real = mem / "MEMORY.md"
        if real.exists():
            diffs = compare_index(real.read_text(encoding="utf-8"), text)
            if not diffs:
                print("[digest] per-section file sets: EQUIVALENT to MEMORY.md ✓")
            else:
                print(f"[digest] per-section file-set DIFFERENCES in {len(diffs)} section(s):")
                for sec, d in diffs.items():
                    if d["missing_files"]:
                        print(f"  [{sec}] missing from shadow: {d['missing_files']}")
                    if d["extra_files"]:
                        print(f"  [{sec}] extra in shadow: {d['extra_files']}")
    finally:
        conn.close()


def register(sub) -> None:
    p = sub.add_parser("digest-shadow",
                       help="render MEMORY.memsom.md shadow + diff vs MEMORY.md (Phase 3)")
    p.add_argument("memory_dir", nargs="?", default=None)
    p.set_defaults(func=_cmd_shadow)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass  # reconfigure unsupported on this stream — keep its default encoding
    ap = argparse.ArgumentParser(prog="memsom_digest", description=__doc__)
    ap.add_argument("memory_dir", nargs="?", default=None)
    _cmd_shadow(ap.parse_args(argv))


if __name__ == "__main__":
    main()
