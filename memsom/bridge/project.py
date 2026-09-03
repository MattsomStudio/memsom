"""memsom.bridge.project — structured project memory: node + fixed sub-notes.

The problem this solves (docs: the structured-project-memory plan): today every
big addition to a project spawns a new ``project_*`` file, so an AI reading a
project cold cannot tell what is done, what is next, what was already decided
against, or where the code lives without opening twenty files.  This module owns
ONE node per project (``projects/<slug>/project_<slug>.md``, ``kind:
project-node``) carrying What / Status / Features / Rules / Creds / Where, plus a
fixed set of sub-notes (spec index, gotchas, decisions, interface_io,
architecture, tests) and one spec note PER FEATURE
(``project_<slug>_spec_<feature-id>.md``).  The node's ``## Features`` list is the
single source of truth for "what is truly left": every feature is marked
implemented / planned / active-decision / archived and links to its spec note.

Design constraints it obeys (RULES.md):
  - Bridge rank (7): imports DOWN only — distill.digest (PROJECT_NOTES /
    _is_project_note), bridge_import parsers, bridge.facts, kernel, paths.
  - NO DB writes.  The file is the store-of-record; the node/notes land in the DB
    on the next ``bridge-render`` (exactly like ``fact-set``).  So this module
    never touches insert_node / get_connection(write) — writer_census,
    gate_writeowner and gate_readpool stay untouched.
  - History is not rewritten in place: logs append dated entries; corrections are
    new entries with ``supersedes:``.

# P2: write_cache / project_aliases.json / the `project cache` verb (the prompt-
#     hook alias cache) land here, written by bridge_render on the Stop hook.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import memsom
from memsom.bridge.bridge_import import (
    split_frontmatter, fm_top_level, default_memory_dir, memory_subdir,
    iter_memory_files, PROJECTS_SUBDIR,
)
from memsom.distill.digest import PROJECT_NOTES, _is_project_note, PROJECT_PREFIX
from memsom.paths import safe_join

# --- the fixed schema -------------------------------------------------------

# Node body: these H2 sections, in this order (missing / reordered = ERROR).
NODE_SECTIONS = ("What", "Status", "Features", "Rules & gates", "Creds",
                 "Where", "Sub-notes", "Pointers")
# The Status block's H3 sub-headings, in order.
STATUS_H3 = ("Done", "Next", "Left", "Needs Matt")
# A feature note's fixed sections (missing = ERROR; the contract shape).
FEATURE_SECTIONS = ("Purpose", "Behaviour", "Interfaces", "Acceptance",
                    "Status", "Changes")
FEATURE_STATUSES = ("implemented", "planned", "active-decision", "archived")
# Which sub-notes are append-only logs vs rewritable refs (frontmatter `kind:`).
NOTE_KIND = {
    "spec": "project-ref", "interface_io": "project-ref", "architecture": "project-ref",
    "gotchas": "project-log", "decisions": "project-log", "tests": "project-log",
}
# Caps.  The ## Features list may grow unbounded — it is the source of truth for
# "what's left" and is never injected in full (only the Status block + a one-line
# tally reach the prompt).  Everything ELSE (What/Status/Rules/Creds/Where/
# Sub-notes/Pointers) keeps a tight budget; a generous whole-body guard catches
# runaway growth.
NODE_NONFEAT_LINE_CAP = 60
NODE_NONFEAT_BYTE_CAP = 4000
NODE_LINE_CAP = 220        # whole-body runaway guard (Features may grow)
NODE_BYTE_CAP = 16000      # whole-body runaway guard (Features may grow)
FEATURE_FENCE = 300           # a feature note over this is a contract-turned-essay
LOG_CAPS = {"gotchas": 150, "decisions": 200, "tests": 200,
            "interface_io": 120, "architecture": 150}
STALE_ACTIVE_DAYS = 14
STALE_PARKED_DAYS = 90
FEATURE_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")   # used with fullmatch (anchors both ends; no `$` — F-16)

# Secret shapes refused in a value (Creds is pointers-only) — shared by the
# writer and check.  A pointer NAMES where a secret lives; it never carries one.
_SECRET_RES = [
    re.compile(r"sk-[A-Za-z0-9]{6,}"),
    re.compile(r"ghp_[A-Za-z0-9]{6,}"),
    re.compile(r"AKIA[0-9A-Z]{8,}"),
    re.compile(r"xox[abp]-[A-Za-z0-9-]{6,}"),
    re.compile(r"-----BEGIN"),
    re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\b\s*[:=]\s*\S"),
]
_ENTROPY_RE = re.compile(r"[A-Za-z0-9+/=_-]{20,}")


class ProjectError(Exception):
    """A refusal the CLI turns into a non-zero exit (never a silent truncation)."""
    def __init__(self, msg, code=1):
        super().__init__(msg)
        self.code = code


def _today() -> str:
    return memsom.local_date(memsom.now_iso())


def _looks_secret(value: str) -> bool:
    """True when *value* carries a secret shape rather than a pointer to one.

    High-entropy heuristic only fires on a bare long token with no separators a
    path/env-name would have — ``C:\\svc\\creds.txt`` and ``MY_SERVICE_KEY``
    (env var name) are pointers, not secrets."""
    v = value.strip()
    for rx in _SECRET_RES:
        if rx.search(v):
            return True
    for m in _ENTROPY_RE.findall(v):
        tok = m
        if len(tok) >= 24 and any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok) \
                and "/" not in v and "\\" not in v and not tok.isupper() and "_" not in tok:
            return True
    return False


# --- paths ------------------------------------------------------------------

def _proj_dir(memory_dir, slug) -> Path:
    return safe_join(Path(memory_dir) / PROJECTS_SUBDIR, slug)


def _node_path(memory_dir, slug) -> Path:
    return _proj_dir(memory_dir, slug) / f"{PROJECT_PREFIX}{slug}.md"


def _note_path(memory_dir, slug, suffix) -> Path:
    return _proj_dir(memory_dir, slug) / f"{PROJECT_PREFIX}{slug}_{suffix}.md"


def _feature_path(memory_dir, slug, feat) -> Path:
    return _proj_dir(memory_dir, slug) / f"{PROJECT_PREFIX}{slug}_spec_{feat}.md"


# --- section parsing (pure text over a body) --------------------------------

def _sections(body: str) -> "OrderedDict[str, list]":
    """{h2_title: [lines]} in document order, splitting on '## ' headers."""
    out = OrderedDict()
    cur = None
    for ln in body.split("\n"):
        m = re.match(r"^##\s+(.*\S)\s*$", ln)
        if m:
            cur = m.group(1).strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return out


def _nonfeature_body(body: str) -> str:
    """The node body with the ``## Features`` section removed — that section is
    exempt from the tight non-feature cap (it grows with the project)."""
    out, skip = [], False
    for ln in body.split("\n"):
        if re.match(r"^##\s+", ln):
            skip = bool(re.match(r"^##\s+Features\s*$", ln))
        if not skip:
            out.append(ln)
    return "\n".join(out)


def _h3(lines) -> "OrderedDict[str, list]":
    out = OrderedDict()
    cur = None
    for ln in lines:
        m = re.match(r"^###\s+(.*\S)\s*$", ln)
        if m:
            cur = m.group(1).strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return out


_FEATURE_LINE_RE = re.compile(
    r"^\s*-\s*(?P<id>[a-z0-9][a-z0-9-]*)\s+[—-]\s+(?P<name>.+?)\s+[—-]\s+"
    r"(?P<status>implemented|planned|active-decision|archived)\s+[—-]\s+"
    r"\[\[(?P<link>project_[a-z0-9_-]+)\]\]\s*\Z")   # \Z not $ (F-16: $ matches before a trailing \n)


def _parse_features(section_lines) -> list:
    out = []
    for ln in section_lines:
        m = _FEATURE_LINE_RE.match(ln)
        if m:
            out.append({"id": m.group("id"), "name": m.group("name").strip(),
                        "status": m.group("status"), "link": m.group("link")})
    return out


def _feature_line(slug, feat, name, status) -> str:
    return f"- {feat} — {name} — {status} — [[{PROJECT_PREFIX}{slug}_spec_{feat}]]"


# --- scaffolding ------------------------------------------------------------

def _node_scaffold(slug, *, aliases=None, repo=None,
                   dir_pc=None, dir_mac=None, dir_droplet=None) -> str:
    fm = [f"name: {PROJECT_PREFIX}{slug}",
          f"description: {slug} — (set the Status headline)",
          "type: project", "kind: project-node", "status: active"]
    if aliases:
        fm.append(f"aliases: {aliases}")
    if repo:
        fm.append(f"repo: {repo}")
    if dir_pc:
        fm.append(f"dir_pc: {dir_pc}")
    if dir_mac:
        fm.append(f"dir_mac: {dir_mac}")
    if dir_droplet:
        fm.append(f"dir_droplet: {dir_droplet}")
    fm += [f"last-verified: {_today()}",
           f"index_title: {slug}",
           "index_hook: (project node — set the Status headline)"]
    subnotes = "\n".join(f"- [[{PROJECT_PREFIX}{slug}_{s}]]" for s in PROJECT_NOTES)
    body = (
        "## What\n(what this project is — ≤4 lines)\n\n"
        "## Status\n### Done\n### Next\n### Left\n### Needs Matt\n\n"
        "## Features\n\n"
        "## Rules & gates\n\n"
        "## Creds\n\n"
        "## Where\n\n"
        f"## Sub-notes\n{subnotes}\n\n"
        "## Pointers\n"
    )
    return "---\n" + "\n".join(fm) + "\n---\n\n" + body


def _note_scaffold(slug, suffix) -> str:
    kind = NOTE_KIND[suffix]
    fm = [f"name: {PROJECT_PREFIX}{slug}_{suffix}",
          f"description: {slug} — {suffix}",
          "type: project", f"kind: {kind}",
          f"depends_on: {PROJECT_PREFIX}{slug}", "status: active"]
    if kind == "project-ref":
        fm.append(f"last-verified: {_today()}")
    if suffix == "spec":
        body = "## Scope\n\n## Non-goals\n\n## Features\n"
    elif kind == "project-log":
        body = "## Entries\n"
    elif suffix == "interface_io":
        body = "### HTTP\n\n### CLI\n\n### IPC/MCP\n"
    else:  # architecture
        body = "## Layers\n\n## Invariants\n\n## Gates\n\n## Data flow\n\n## Changes\n"
    return "---\n" + "\n".join(fm) + "\n---\n\n" + body


def _feature_scaffold(slug, feat, name, status) -> str:
    fm = [f"name: {PROJECT_PREFIX}{slug}_spec_{feat}",
          f"description: {slug} spec — {name}",
          "type: project", "kind: project-ref",
          f"depends_on: {PROJECT_PREFIX}{slug}_spec", "status: active",
          f"last-verified: {_today()}"]
    body = (
        "## Purpose\n\n"
        "## Behaviour\n(inputs, outputs, invariants, limits, defaults)\n\n"
        "## Interfaces\n\n"
        "## Acceptance\n\n"
        f"## Status\n{status}\n\n"
        f"## Changes\n- {_today()} created ({status})\n"
    )
    return "---\n" + "\n".join(fm) + "\n---\n\n" + body


# --- writers ----------------------------------------------------------------

def init_project(memory_dir, slug, *, aliases=None, repo=None,
                 dir_pc=None, dir_mac=None, dir_droplet=None) -> dict:
    """Create-if-absent scaffold: node + six sub-notes.  Never overwrites."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
        raise ProjectError(f"bad slug {slug!r} (use [a-z0-9_-])")
    d = _proj_dir(memory_dir, slug)
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    node = _node_path(memory_dir, slug)
    if node.exists():
        out[node.name] = "present"
    else:
        node.write_text(_node_scaffold(slug, aliases=aliases, repo=repo,
                                       dir_pc=dir_pc, dir_mac=dir_mac,
                                       dir_droplet=dir_droplet), encoding="utf-8")
        out[node.name] = "created"
    for suffix in PROJECT_NOTES:
        p = _note_path(memory_dir, slug, suffix)
        if p.exists():
            out[p.name] = "present"
        else:
            p.write_text(_note_scaffold(slug, suffix), encoding="utf-8")
            out[p.name] = "created"
    return out


def _require_node(memory_dir, slug) -> tuple:
    node = _node_path(memory_dir, slug)
    if not node.exists():
        raise ProjectError(f"no project node for {slug!r} — run `memsom project init {slug}`")
    text = node.read_text(encoding="utf-8")
    fm = fm_top_level(split_frontmatter(text)[0])
    return node, text, fm


def set_status(memory_dir, slug, *, done=None, next_=None, left=None, ask=None,
               cred=None, verified=None) -> dict:
    """Append a bullet under the right ``### `` H3, or a ``## Creds`` pointer, or
    bump ``last-verified``.  Refuses a secret-shaped cred value."""
    node, text, _fm = _require_node(memory_dir, slug)
    fm_lines, body, _ = split_frontmatter(text)
    if cred is not None:
        if _looks_secret(cred):
            raise ProjectError("that Creds value looks like a secret — Creds is "
                               "pointers only (env var name, file path, 1Password item)")
        body = _append_under_h2(body, "Creds", f"- {cred}")
    for h3, val in (("Done", done), ("Next", next_), ("Left", left), ("Needs Matt", ask)):
        if val is not None:
            body = _append_under_h3(body, "Status", h3, f"- {val}")
    if verified is not None:
        fm_lines = _stamp_line(fm_lines, "last-verified",
                               verified if verified is not True else _today())
    node.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + body, encoding="utf-8")
    return {"node": node.name}


def set_feature(memory_dir, slug, feat, *, name=None, status=None,
                evidence=None, decision=None) -> dict:
    """Add/update the node ``## Features`` line, the spec-index line and the
    feature's own spec note together.  ``implemented`` refuses without evidence;
    a decision-driven status change requires a ``D-…`` reference."""
    if not FEATURE_ID_RE.fullmatch(feat):
        raise ProjectError(f"bad feature id {feat!r} (permanent slug [a-z0-9-])")
    if status is not None and status not in FEATURE_STATUSES:
        raise ProjectError(f"bad status {status!r} (one of {', '.join(FEATURE_STATUSES)})")
    if status == "implemented" and not evidence:
        raise ProjectError("--status implemented needs --evidence \"(MEASURED) …\"")
    node, text, _fm = _require_node(memory_dir, slug)
    fm_lines, body, _ = split_frontmatter(text)
    secs = _sections(body)
    feats = {f["id"]: f for f in _parse_features(secs.get("Features", []))}
    prev = feats.get(feat)
    fname = name or (prev["name"] if prev else feat)
    fstatus = status or (prev["status"] if prev else "planned")
    if status and prev and status != prev["status"] \
            and _is_decision_status(status, prev["status"]) and not decision:
        raise ProjectError(f"status change {prev['status']}→{status} reflects a decision — "
                           "pass --decision D-… so the notes cross-link")
    feats[feat] = {"id": feat, "name": fname, "status": fstatus,
                   "link": f"{PROJECT_PREFIX}{slug}_spec_{feat}"}
    body = _rewrite_features(body, slug, feats)
    node.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + body, encoding="utf-8")
    # spec index line
    _upsert_spec_index(memory_dir, slug, feats)
    # the feature note itself
    fp = _feature_path(memory_dir, slug, feat)
    if not fp.exists():
        fp.write_text(_feature_scaffold(slug, feat, fname, fstatus), encoding="utf-8")
    change = f"- {_today()} status → {fstatus}"
    if evidence:
        change += f" — {evidence}"
    if decision:
        change += f" ({decision})"
    _apply_feature_change(fp, status_line=fstatus, change=change)
    return {"feature": feat, "status": fstatus}


def set_spec(memory_dir, slug, feat, *, section, value, why) -> dict:
    """Edit one section of one feature note in place + append a dated Changes
    line — the only sanctioned way to change a spec (so it always carries a why)."""
    canon = {"purpose": "Purpose", "behaviour": "Behaviour",
             "interfaces": "Interfaces", "acceptance": "Acceptance"}
    if section not in canon:
        raise ProjectError(f"--set {section!r}: one of {', '.join(canon)}")
    fp = _feature_path(memory_dir, slug, feat)
    if not fp.exists():
        raise ProjectError(f"no spec note for feature {feat!r} — add it with "
                           f"`memsom project feature {slug} {feat} …`")
    text = fp.read_text(encoding="utf-8")
    fm_lines, body, _ = split_frontmatter(text)
    body = _set_h2(body, canon[section], value)
    body = _append_under_h2(body, "Changes", f"- {_today()} {section}: {why}")
    fp.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + body, encoding="utf-8")
    return {"feature": feat, "section": canon[section]}


# --- log --------------------------------------------------------------------

def log_entry(memory_dir, slug, note, entry, *, why=None, rejected=None,
              cause=None, fix=None, where=None, covers=None, run=None,
              supersedes=None, source=None) -> dict:
    """Append a fixed-template dated entry to a log note (gotchas/decisions/tests),
    newest first under ``## Entries``.  Dedupe first; refuse (code 2) at the cap."""
    if note not in ("gotchas", "decisions", "tests"):
        raise ProjectError(f"log target must be gotchas|decisions|tests, not {note!r}")
    p = _note_path(memory_dir, slug, note)
    if not p.exists():
        raise ProjectError(f"no {note} note for {slug!r} — run `memsom project init {slug}`")
    text = p.read_text(encoding="utf-8")
    fm_lines, body, _ = split_frontmatter(text)
    prefix = {"gotchas": "G", "decisions": "D", "tests": "T"}[note]
    today = _today()
    # dedupe by the bolded payload / test path
    existing = body
    dup = _find_dup(existing, note, entry)
    if dup is not None:
        body2 = _annotate_dup(existing, note, dup, today)
        p.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + body2, encoding="utf-8")
        return {"id": dup, "status": "reaffirmed"}
    seq = _next_seq(existing, prefix, today)
    eid = f"{prefix}-{today.replace('-', '')}-{seq:02d}"
    line = _render_log_line(note, eid, today, entry, why=why, rejected=rejected,
                            cause=cause, fix=fix, where=where, covers=covers,
                            run=run, supersedes=supersedes, source=source)
    body2 = _prepend_entry(body, line)
    cap = LOG_CAPS[note]
    if _body_lines(body2) > cap:
        raise ProjectError(f"{note} note would exceed its {cap}-line cap — run "
                           "`/reorgmem` to fold superseded entries into ## History",
                           code=2)
    p.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + body2, encoding="utf-8")
    return {"id": eid, "status": "added"}


# --- small text helpers -----------------------------------------------------

def _body_lines(body: str) -> int:
    return len([ln for ln in body.split("\n") if ln.strip()])


def _stamp_line(fm_lines, key, value) -> list:
    out = [ln for ln in fm_lines if ln.split(":", 1)[0].strip() != key]
    out.append(f"{key}: {value}")
    return out


def _append_under_h2(body, h2, bullet) -> str:
    lines = body.split("\n")
    idx = next((i for i, ln in enumerate(lines) if ln.strip() == f"## {h2}"), -1)
    if idx == -1:
        lines += ["", f"## {h2}", bullet]
        return "\n".join(lines)
    end = idx + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    ins = end
    while ins > idx + 1 and not lines[ins - 1].strip():
        ins -= 1
    lines.insert(ins, bullet)
    return "\n".join(lines)


def _append_under_h3(body, h2, h3, bullet) -> str:
    lines = body.split("\n")
    h2i = next((i for i, ln in enumerate(lines) if ln.strip() == f"## {h2}"), -1)
    if h2i == -1:
        return _append_under_h2(body, h2, bullet)
    end = h2i + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    h3i = next((i for i in range(h2i + 1, end) if lines[i].strip() == f"### {h3}"), -1)
    if h3i == -1:
        lines.insert(end, f"### {h3}")
        lines.insert(end + 1, bullet)
        return "\n".join(lines)
    sub_end = h3i + 1
    while sub_end < end and not lines[sub_end].startswith("### "):
        sub_end += 1
    ins = sub_end
    while ins > h3i + 1 and not lines[ins - 1].strip():
        ins -= 1
    lines.insert(ins, bullet)
    return "\n".join(lines)


def _set_h2(body, h2, value) -> str:
    """Replace the content of a ## section with *value* (keeps the header)."""
    lines = body.split("\n")
    idx = next((i for i, ln in enumerate(lines) if ln.strip() == f"## {h2}"), -1)
    if idx == -1:
        return body.rstrip("\n") + f"\n\n## {h2}\n{value}\n"
    end = idx + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return "\n".join(lines[:idx + 1] + [value, ""] + lines[end:])


def _rewrite_features(body, slug, feats) -> str:
    lines = [_feature_line(slug, f["id"], f["name"], f["status"])
             for f in sorted(feats.values(), key=lambda x: x["id"])]
    return _set_h2(body, "Features", "\n".join(lines))


def _upsert_spec_index(memory_dir, slug, feats) -> None:
    p = _note_path(memory_dir, slug, "spec")
    if not p.exists():
        p.write_text(_note_scaffold(slug, "spec"), encoding="utf-8")
    text = p.read_text(encoding="utf-8")
    fm_lines, body, _ = split_frontmatter(text)
    lines = [_feature_line(slug, f["id"], f["name"], f["status"])
             for f in sorted(feats.values(), key=lambda x: x["id"])]
    body = _set_h2(body, "Features", "\n".join(lines))
    p.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + body, encoding="utf-8")


def _apply_feature_change(fp: Path, *, status_line, change) -> None:
    text = fp.read_text(encoding="utf-8")
    fm_lines, body, _ = split_frontmatter(text)
    body = _set_h2(body, "Status", status_line)
    body = _append_under_h2(body, "Changes", change)
    fp.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + body, encoding="utf-8")


def _is_decision_status(new, old) -> bool:
    """A move into/out of active-decision, or a planned→implemented/archived flip,
    is a decision the notes must cross-link."""
    return "active-decision" in (new, old) or (old == "planned" and new in ("archived",))


# --- log rendering / dedupe -------------------------------------------------

def _prepend_entry(body, line) -> str:
    lines = body.split("\n")
    idx = next((i for i, ln in enumerate(lines) if ln.strip() == "## Entries"), -1)
    if idx == -1:
        return "## Entries\n" + line + "\n" + body
    lines.insert(idx + 1, line)
    return "\n".join(lines)


def _next_seq(body, prefix, today) -> int:
    tag = f"{prefix}-{today.replace('-', '')}-"
    n = 0
    for m in re.finditer(re.escape(tag) + r"(\d{2})", body):
        n = max(n, int(m.group(1)))
    return n + 1


def _bold(entry) -> str:
    m = re.search(r"\*\*(.+?)\*\*", entry)
    return (m.group(1) if m else entry).strip().lower()


def _find_dup(body, note, entry):
    """Return the id of an existing entry the new one duplicates, else None."""
    if note == "tests":
        m = re.search(r"`([^`]+)`", entry)
        needle = (m.group(1) if m else entry).strip()
        pat = re.compile(r"^-\s*(T-\d{8}-\d{2})\s+`" + re.escape(needle) + "`", re.M)
    else:
        needle = _bold(entry)
        idre = "G" if note == "gotchas" else "D"
        pat = re.compile(r"^-\s*(" + idre + r"-\d{8}-\d{2}).*?\*\*(.+?)\*\*", re.M)
    for m in pat.finditer(body):
        if note == "tests":
            return m.group(1)
        if m.group(2).strip().lower() == needle:
            return m.group(1)
    return None


def _annotate_dup(body, note, eid, today) -> str:
    tag = {"gotchas": "seen again", "decisions": "reaffirmed", "tests": "edited"}[note]
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        if re.match(r"^-\s*" + re.escape(eid) + r"\b", ln):
            lines[i] = ln.rstrip() + f" · {tag}: {today}"
            break
    return "\n".join(lines)


def _render_log_line(note, eid, today, entry, **f) -> str:
    if note == "gotchas":
        parts = [f"- {eid} ({today}) {entry}"]
        if f.get("cause"):
            parts.append(f"cause: {f['cause']}")
        if f.get("fix"):
            parts.append(f"fix: {f['fix']}")
        line = " / ".join(parts)
        if f.get("where"):
            line += f" · where: {f['where']}"
        if f.get("source"):
            line += f" / source: {f['source']}"
        return line
    if note == "decisions":
        parts = [f"- {eid} ({today}) {entry}"]
        if f.get("rejected"):
            parts.append(f"rejected: {f['rejected']}")
        if f.get("why"):
            parts.append(f"why: {f['why']}")
        line = " / ".join(parts)
        if f.get("supersedes"):
            line += f" · supersedes: {f['supersedes']}"
        if f.get("source"):
            line += f" / source: {f['source']}"
        return line
    # tests
    line = f"- {eid} {entry} created {today}"
    if f.get("covers"):
        line += f" / covers: {f['covers']}"
    if f.get("run"):
        line += f" · run: {f['run']}"
    line += f" · status: green {today}"
    return line


# --- check ------------------------------------------------------------------

def _F(name, sev, target, msg) -> dict:
    return {"name": name, "sev": sev, "target": target, "msg": msg}


def _iter_project_dirs(memory_dir):
    root = Path(memory_dir) / PROJECTS_SUBDIR
    if not root.is_dir():
        return
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        slug = d.name
        node = d / f"{PROJECT_PREFIX}{slug}.md"
        if node.exists():
            text = node.read_text(encoding="utf-8", errors="replace")
            fm = fm_top_level(split_frontmatter(text)[0])
            if (fm.get("kind") or "").strip() == "project-node":
                yield slug, d, node, text, fm


def _is_absorbed(path) -> bool:
    """True when a file has been deliberately folded into a node — ``index: false``
    AND an ``absorbed_into:`` pointer.  That is the plan's RESOLVED state for a
    loose file (kept on disk to preserve detail, withdrawn from the render), so the
    checker must not keep flagging it."""
    try:
        fm = fm_top_level(split_frontmatter(path.read_text(encoding="utf-8", errors="replace"))[0])
    # FAILOPEN: an unreadable/garbled loose file is treated as not-absorbed (it still WARNs)
    except Exception:
        return False
    idx = str(fm.get("index", "")).strip().lower()
    return idx == "false" and bool((fm.get("absorbed_into") or "").strip())


def check(memory_dir, slug=None) -> list:
    """Audit-shaped findings ({name,sev,target,msg}) for every project node (or
    just *slug*).  Fails CLOSED: a parse problem is a finding, not a crash."""
    memory_dir = Path(memory_dir)
    findings = []
    alias_owner = {}
    dirs = [t for t in _iter_project_dirs(memory_dir) if slug is None or t[0] == slug]
    # detect a node dir whose parent frontmatter is nested (flat keys invisible)
    if slug is not None and not dirs:
        d = memory_dir / PROJECTS_SUBDIR / slug
        np = d / f"{PROJECT_PREFIX}{slug}.md"
        if np.exists():
            raw = np.read_text(encoding="utf-8", errors="replace")
            if "metadata:" in raw and not fm_top_level(split_frontmatter(raw)[0]).get("kind"):
                findings.append(_F("project-nested-frontmatter", "WARN", np.name,
                                   "frontmatter is nested under metadata: — the importer "
                                   "sees no flat keys; flatten it"))
    for sl, d, node, text, fm in dirs:
        findings += _check_one(memory_dir, sl, d, node, text, fm, alias_owner)
    return findings


def _check_one(memory_dir, slug, d, node, text, fm, alias_owner) -> list:
    out = []
    fm_lines, body, _had = split_frontmatter(text)
    # nested frontmatter (flat keys missing but a metadata: block present)
    if "metadata:" in text and not fm.get("name"):
        out.append(_F("project-nested-frontmatter", "WARN", node.name,
                      "frontmatter is nested under metadata: — flatten it"))
    # H2 order
    secs = _sections(body)
    got = [s for s in secs if s in NODE_SECTIONS]
    if got != list(NODE_SECTIONS):
        missing = [s for s in NODE_SECTIONS if s not in secs]
        if missing:
            out.append(_F("project-schema", "ERROR", node.name,
                          f"missing H2 section(s): {', '.join(missing)}"))
        else:
            out.append(_F("project-schema", "ERROR", node.name,
                          f"H2 sections out of order: {got} vs {list(NODE_SECTIONS)}"))
    # Status H3s
    if "Status" in secs:
        h3 = _h3(secs["Status"])
        miss3 = [h for h in STATUS_H3 if h not in h3]
        if miss3:
            out.append(_F("project-schema", "ERROR", node.name,
                          f"Status missing H3: {', '.join(miss3)}"))
    # caps — the ## Features list is exempt (it grows); the rest keeps the tight
    # budget; a generous whole-body guard catches runaway growth.
    nonfeat = _nonfeature_body(body)
    nf_lines = _body_lines(nonfeat)
    if nf_lines > NODE_NONFEAT_LINE_CAP:
        out.append(_F("project-schema", "ERROR", node.name,
                      f"node body (excl. Features) {nf_lines} > {NODE_NONFEAT_LINE_CAP}-line cap"))
    if len(nonfeat.encode("utf-8")) > NODE_NONFEAT_BYTE_CAP:
        out.append(_F("project-schema", "ERROR", node.name,
                      f"node body (excl. Features) > {NODE_NONFEAT_BYTE_CAP}-byte cap"))
    if _body_lines(body) > NODE_LINE_CAP:
        out.append(_F("project-schema", "ERROR", node.name,
                      f"node body {_body_lines(body)} > {NODE_LINE_CAP}-line cap"))
    if len(body.encode("utf-8")) > NODE_BYTE_CAP:
        out.append(_F("project-schema", "ERROR", node.name,
                      f"node body > {NODE_BYTE_CAP}-byte cap"))
    # aliases
    for a in [x.strip().lower() for x in (fm.get("aliases") or "").split(",") if x.strip()]:
        if len(a) < 3:
            out.append(_F("project-schema", "ERROR", node.name,
                          f"alias {a!r} shorter than 3 chars"))
            continue
        if a in alias_owner and alias_owner[a] != slug:
            out.append(_F("project-alias-clash", "ERROR", node.name,
                          f"alias {a!r} also claimed by {alias_owner[a]!r}"))
        alias_owner[a] = slug
    # creds pointers only
    for ln in secs.get("Creds", []):
        val = ln.strip().lstrip("-").strip()
        if val and _looks_secret(val):
            out.append(_F("project-creds-value", "ERROR", node.name,
                          "a Creds line carries a secret value — pointers only"))
            break
    # stale status
    lv = (fm.get("last-verified") or "").strip()
    st = (fm.get("status") or "active").strip().lower()
    limit = STALE_PARKED_DAYS if st == "parked" else STALE_ACTIVE_DAYS
    if lv and _age_days(lv) > limit:
        out.append(_F("project-stale-status", "INFO", node.name,
                      f"status not verified in {int(_age_days(lv))}d (> {limit}); re-verify"))
    # features / spec cross-checks
    out += _check_features(memory_dir, slug, d, secs, body)
    # loose files
    allowed = {f"{PROJECT_PREFIX}{slug}.md"} | {
        f"{PROJECT_PREFIX}{slug}_{s}.md" for s in PROJECT_NOTES}
    for p in sorted(d.glob("*.md")):
        stem = p.stem
        if p.name in allowed:
            continue
        if _SYNC_CONFLICT_RE.search(p.name):   # Syncthing conflict copy — reorg's domain
            continue
        if _is_project_note(stem, slug):   # a per-feature spec note is fine
            continue
        if _is_absorbed(p):   # already folded (index:false + absorbed_into) — the resolved state, not a WARN
            continue
        if stem.startswith(f"{PROJECT_PREFIX}{slug}_") or stem == f"{PROJECT_PREFIX}{slug}":
            out.append(_F("project-loose-file", "WARN", p.name,
                          "loose project file outside the fixed node/sub-note set — "
                          "fold it into a sub-note or make it a feature spec"))
    return out


def _check_features(memory_dir, slug, d, secs, body) -> list:
    out = []
    node_feats = _parse_features(secs.get("Features", []))
    by_id = {f["id"]: f for f in node_feats}
    # spec index features
    spec_idx = _note_path(memory_dir, slug, "spec")
    idx_feats = {}
    if spec_idx.exists():
        isecs = _sections(split_frontmatter(spec_idx.read_text(encoding="utf-8"))[1])
        idx_feats = {f["id"]: f for f in _parse_features(isecs.get("Features", []))}
    # every node feature -> spec note exists, index line exists, statuses agree
    for fid, f in by_id.items():
        fp = _feature_path(memory_dir, slug, fid)
        if not fp.exists():
            out.append(_F("project-schema", "ERROR", f"{PROJECT_PREFIX}{slug}.md",
                          f"feature {fid!r} has no spec note {fp.name}"))
            continue
        ftext = fp.read_text(encoding="utf-8", errors="replace")
        fsecs = _sections(split_frontmatter(ftext)[1])
        # sections
        miss = [s for s in FEATURE_SECTIONS if s not in fsecs]
        if miss:
            out.append(_F("project-schema", "ERROR", fp.name,
                          f"feature note missing section(s): {', '.join(miss)}"))
        # fence
        flines = len(ftext.split("\n"))
        if flines > FEATURE_FENCE:
            out.append(_F("project-schema", "ERROR", fp.name,
                          f"feature note {flines} lines > {FEATURE_FENCE}-line fence"))
        # status agreement (node vs note)
        note_status = " ".join(fsecs.get("Status", [])).strip().lower()
        if note_status and f["status"] not in note_status:
            out.append(_F("project-schema", "ERROR", fp.name,
                          f"feature status {note_status!r} disagrees with node "
                          f"line {f['status']!r}"))
        # implemented needs measured evidence somewhere in the note
        if f["status"] == "implemented" and "(measured)" not in ftext.lower():
            out.append(_F("project-schema", "ERROR", fp.name,
                          "feature is implemented but the note carries no (MEASURED) "
                          "evidence line"))
        # index line present + agrees
        if idx_feats and fid not in idx_feats:
            out.append(_F("project-schema", "ERROR", spec_idx.name,
                          f"feature {fid!r} on the node has no spec-index line"))
        # spec.stale: implemented feature whose newest Changes date is older than a
        # decisions/gotchas/tests entry naming the feature id
        if f["status"] == "implemented":
            if _spec_is_stale(memory_dir, slug, fid, fsecs):
                out.append(_F("project-schema", "ERROR", fp.name,
                              f"spec for {fid!r} is stale — a later log entry names it "
                              "but its ## Changes was not updated"))
    # every spec note -> a node line
    for fp in sorted(d.glob(f"{PROJECT_PREFIX}{slug}_spec_*.md")):
        fid = fp.stem[len(f"{PROJECT_PREFIX}{slug}_spec_"):]
        if fid and fid not in by_id:
            out.append(_F("project-schema", "ERROR", fp.name,
                          f"spec note for {fid!r} has no ## Features line on the node"))
    # left/next consistency + needs-matt coverage
    left_next = _h3(secs.get("Status", [])).get("Left", []) + \
        _h3(secs.get("Status", [])).get("Next", [])
    needs = " ".join(_h3(secs.get("Status", [])).get("Needs Matt", [])).lower()
    for fid, f in by_id.items():
        named_in_left = any(re.search(r"\b" + re.escape(fid) + r"\b", ln) for ln in left_next)
        if named_in_left and f["status"] not in ("planned", "active-decision"):
            out.append(_F("project-schema", "ERROR", f"{PROJECT_PREFIX}{slug}.md",
                          f"Left/Next names {fid!r} but it is {f['status']} "
                          "(only planned/active-decision belong there)"))
        if f["status"] == "active-decision" and fid not in needs:
            out.append(_F("project-schema", "ERROR", f"{PROJECT_PREFIX}{slug}.md",
                          f"active-decision feature {fid!r} is not listed under Needs Matt"))
    return out


def _spec_is_stale(memory_dir, slug, fid, fsecs) -> bool:
    changes = fsecs.get("Changes", [])
    dates = [m.group(0) for ln in changes
             for m in [re.search(r"\d{4}-\d{2}-\d{2}", ln)] if m]
    newest_change = max(dates) if dates else "0000-00-00"
    for note in ("decisions", "gotchas", "tests"):
        p = _note_path(memory_dir, slug, note)
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8", errors="replace").split("\n"):
            if re.search(r"\b" + re.escape(fid) + r"\b", ln):
                m = re.search(r"\d{4}-\d{2}-\d{2}", ln)
                if m and m.group(0) > newest_change:
                    return True
    return False


def _age_days(iso_date: str) -> float:
    try:
        import datetime as _dt
        d = _dt.date.fromisoformat(iso_date[:10])
        return (_dt.date.today() - d).days
    # FAILOPEN: an unparseable date is not "stale" — return 0 so a malformed last-verified never manufactures a stale finding (the schema check owns shape).
    except Exception:
        return 0.0


# --- reorg (maintenance sweep) ----------------------------------------------
#
# `project reorg` is the deterministic half of the /reorgmem skill: it runs
# check() (every schema finding), adds the maintenance checks that check() does
# not own (sub-note presence/kind, caps, dangling wikilinks, missing fact refs,
# rules⊆architecture, stale sub-note counts, sync-conflict copies), applies ONLY
# the mechanical fixes that need no judgment, and routes everything else to Matt
# (interactive) or to .weights/reorgmem_pending.json (--sweep, no model).  It is
# the ONLY project writer allowed to touch a file it did not just scaffold, so it
# is conservative: content-bearing edits are proposed, never applied unattended.

# The two fixes that carry ZERO judgment — safe to apply on the headless sweep.
REORG_MECHANICAL = {"reorg-sync-conflict", "reorg-subnote-count"}
_SYNC_CONFLICT_RE = re.compile(r"\.sync-conflict-\d{8}-\d{6}-[A-Z0-9]+", re.I)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:[#|][^\]]*)?\]\]")
_ENTRY_ID_RE = re.compile(r"\b([GDT]-\d{8}-\d{2})\b")


def _all_stems(memory_dir) -> set:
    """Every memory stem in the store (flat + projects/**), for link resolution."""
    return {p.stem for p in Path(memory_dir).rglob("*.md")
            if p.name not in ("MEMORY.md", "INDEX.md")}


def _bullet_count(body: str) -> int:
    """Number of top-level ``- `` bullets in a note body — the sub-note count."""
    return len([ln for ln in body.split("\n") if re.match(r"^-\s", ln)])


def _subnote_line_count(memory_dir, slug, suffix) -> int:
    p = _note_path(memory_dir, slug, suffix)
    if not p.exists():
        return 0
    return _bullet_count(split_frontmatter(p.read_text(encoding="utf-8"))[1])


def _reorg_checks(memory_dir, slug, d, node, text, fm) -> list:
    """Maintenance findings check() does not own.  Each is tagged fix=mechanical
    (safe to auto-apply) or fix=content (needs Matt/the model)."""
    out = []
    stems = _all_stems(memory_dir)
    fm_lines, body, _ = split_frontmatter(text)
    secs = _sections(body)

    def F(name, sev, target, msg, fix="content"):
        f = _F(name, sev, target, msg)
        f["fix"] = fix
        return f

    # 1. every fixed sub-note exists with the right kind + parent
    for suffix in PROJECT_NOTES:
        p = _note_path(memory_dir, slug, suffix)
        if not p.exists():
            out.append(F("reorg-subnote-missing", "ERROR", p.name,
                         f"fixed sub-note {suffix!r} is missing — run "
                         f"`memsom project init {slug}`"))
            continue
        nfm = fm_top_level(split_frontmatter(p.read_text(encoding="utf-8"))[0])
        if (nfm.get("kind") or "").strip() != NOTE_KIND[suffix]:
            out.append(F("reorg-subnote-kind", "WARN", p.name,
                         f"kind {nfm.get('kind')!r} should be {NOTE_KIND[suffix]!r}"))
        if (nfm.get("depends_on") or "").strip() != f"{PROJECT_PREFIX}{slug}":
            out.append(F("reorg-subnote-kind", "WARN", p.name,
                         f"depends_on should be {PROJECT_PREFIX}{slug}"))

    # 2. log sub-notes over their cap → fold superseded entries into ## History
    for suffix, cap in LOG_CAPS.items():
        p = _note_path(memory_dir, slug, suffix)
        if p.exists():
            b = split_frontmatter(p.read_text(encoding="utf-8"))[1]
            n = _body_lines(b)
            if n > cap:
                out.append(F("reorg-subnote-cap", "WARN", p.name,
                             f"{n} lines > {cap} cap — fold superseded entries into ## History"))

    # 3. dangling wikilinks in the node + the six sub-notes (fact_* → fact check).
    #    A missing fixed sub-note is reported as reorg-subnote-missing above, not
    #    as a broken link, so those stems are skipped here.
    fixed_stems = {f"{PROJECT_PREFIX}{slug}_{s}" for s in PROJECT_NOTES} | {f"{PROJECT_PREFIX}{slug}"}
    scope = [node] + [_note_path(memory_dir, slug, s) for s in PROJECT_NOTES]
    seen_broken = set()
    for p in scope:
        if not p.exists():
            continue
        for target in _WIKILINK_RE.findall(p.read_text(encoding="utf-8", errors="replace")):
            target = target.strip()
            if not target or target in seen_broken or target in stems or target in fixed_stems:
                continue
            seen_broken.add(target)
            if target.startswith("fact_"):
                out.append(F("reorg-fact-missing", "WARN", p.name,
                             f"[[{target}]] resolves to no fact file"))
            else:
                out.append(F("reorg-link-broken", "WARN", p.name,
                             f"[[{target}]] resolves to no memory in the store"))

    # 4. index_hook set (empty / placeholder is a legibility miss)
    hook = (fm.get("index_hook") or "").strip()
    if not hook or hook.lower().startswith("(project node") or "set the status" in hook.lower():
        out.append(F("reorg-index-hook", "INFO", node.name,
                     "index_hook is empty/placeholder — set it to the Status headline"))

    # 5. node ## Rules & gates ⊆ the architecture note (Invariants + Gates)
    rules = [ln.strip().lstrip("-").strip() for ln in secs.get("Rules & gates", [])
             if ln.strip().startswith("-")]
    if rules:
        arch = _note_path(memory_dir, slug, "architecture")
        arch_text = arch.read_text(encoding="utf-8", errors="replace") if arch.exists() else ""
        for r in rules:
            key = re.sub(r"\W+", " ", r.split(":")[0]).strip().lower()
            if key and key not in re.sub(r"\W+", " ", arch_text).lower():
                out.append(F("reorg-rules-subset", "WARN", node.name,
                             f"Rule {r[:48]!r} has no matching Invariant/Gate in the "
                             "architecture note"))

    # 6. sync-conflict copies in the project dir (mechanical: union-merge + delete)
    for p in sorted(d.glob("*.sync-conflict-*.md")):
        out.append(F("reorg-sync-conflict", "WARN", p.name,
                     "Syncthing conflict copy — reorg union-merges it and deletes the copy",
                     fix="mechanical"))
    return out


def _canonical_of_conflict(p: Path) -> Path:
    """The real file a *.sync-conflict-* copy shadows."""
    return p.with_name(_SYNC_CONFLICT_RE.sub("", p.name))


def _merge_log_entries(canon_body: str, conflict_body: str) -> str:
    """Union the ## Entries bullets of two copies of a log note by entry ID,
    keeping the longer text on a clash, newest-first."""
    def entries(b):
        out = {}
        order = []
        for ln in b.split("\n"):
            if not re.match(r"^-\s", ln):
                continue
            mid = _ENTRY_ID_RE.search(ln)
            key = mid.group(1) if mid else ln.strip().lower()
            if key not in out or len(ln) > len(out[key]):
                out[key] = ln
            if key not in order:
                order.append(key)
        return out, order
    ca, oa = entries(canon_body)
    cb, _ob = entries(conflict_body)
    merged = dict(ca)
    for k, v in cb.items():
        if k not in merged or len(v) > len(merged[k]):
            merged[k] = v

    def datekey(ln):
        m = re.search(r"\d{4}-\d{2}-\d{2}", ln)
        return (m.group(0) if m else "0000-00-00", ln)
    lines = sorted(merged.values(), key=datekey, reverse=True)
    return "## Entries\n" + "\n".join(lines) + "\n"


def _apply_sync_conflict(canon: Path, conflict: Path) -> str:
    """Mechanically resolve one conflict copy.  Returns 'merged' | 'deleted' |
    'kept' (kept = a non-log divergence that needs a human)."""
    if not canon.exists():
        return "kept"
    ctext = conflict.read_text(encoding="utf-8", errors="replace")
    ntext = canon.read_text(encoding="utf-8", errors="replace")
    if ctext == ntext:
        conflict.unlink()
        return "deleted"
    # log notes (## Entries) union-merge; anything else is a real divergence
    fm_lines, nbody, _ = split_frontmatter(ntext)
    _cf, cbody, _ = split_frontmatter(ctext)
    if "## Entries" in nbody and "## Entries" in cbody:
        head = nbody.split("## Entries")[0]
        merged = _merge_log_entries(nbody, cbody)
        canon.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + head + merged,
                         encoding="utf-8")
        conflict.unlink()
        return "merged"
    return "kept"


def _fix_subnote_counts(memory_dir, slug) -> bool:
    """Rewrite the node ## Sub-notes wikilinks with fresh ``— N`` counts."""
    node = _node_path(memory_dir, slug)
    text = node.read_text(encoding="utf-8")
    fm_lines, body, _ = split_frontmatter(text)
    lines = body.split("\n")
    changed = False
    for i, ln in enumerate(lines):
        m = re.search(r"\[\[" + re.escape(PROJECT_PREFIX + slug) + r"_(\w+)\]\]", ln)
        if not m or m.group(1) not in PROJECT_NOTES or not ln.strip().startswith("-"):
            continue
        want = _subnote_line_count(memory_dir, slug, m.group(1))
        new = re.sub(r"\s*—\s*\d+\s*$", "", ln.rstrip()) + f" — {want}"
        if new != ln:
            lines[i] = new
            changed = True
    if changed:
        node.write_text("---\n" + "\n".join(fm_lines) + "\n---\n" + "\n".join(lines),
                        encoding="utf-8")
    return changed


def reorg(memory_dir, *, slug=None, sweep=False, apply=False) -> dict:
    """Run the maintenance pass.  Report-only by default; ``apply`` runs the
    mechanical fixes; ``sweep`` runs them headless and writes the pending file.
    Returns {projects, findings, mechanical, content, pending_path}."""
    memory_dir = Path(memory_dir)
    do_fix = sweep or apply
    findings_by_slug = {}
    mechanical, content = [], []

    dirs = [t for t in _iter_project_dirs(memory_dir) if slug is None or t[0] == slug]
    for sl, d, node, text, fm in dirs:
        # schema findings from check() are content (need judgment) …
        base = [dict(f, fix="content") for f in check(memory_dir, sl)]
        extra = _reorg_checks(memory_dir, sl, d, node, text, fm)
        fs = base + extra
        findings_by_slug[sl] = fs
        for f in fs:
            (mechanical if f.get("fix") == "mechanical" else content).append(dict(f, slug=sl))

    applied = []
    if do_fix:
        for sl, d, node, text, fm in dirs:
            for p in sorted(d.glob("*.sync-conflict-*.md")):
                res = _apply_sync_conflict(_canonical_of_conflict(p), p)
                applied.append({"slug": sl, "fix": "sync-conflict",
                                "target": p.name, "result": res})
            if _fix_subnote_counts(memory_dir, sl):
                applied.append({"slug": sl, "fix": "subnote-count", "target": node.name})

    pending_path = None
    if sweep:
        pending_path = _write_pending(memory_dir, findings_by_slug, applied)

    return {"projects": len(dirs), "findings": findings_by_slug,
            "mechanical": mechanical, "content": content,
            "applied": applied, "pending_path": str(pending_path) if pending_path else None}


def _write_pending(memory_dir, findings_by_slug, applied) -> Path:
    """Persist the content findings for the next interactive /reorgmem + log it."""
    import json
    wdir = Path(memory_dir) / ".weights"
    wdir.mkdir(parents=True, exist_ok=True)
    ts = memsom.now_iso()
    projects = {}
    for sl, fs in findings_by_slug.items():
        content = [f for f in fs if f.get("fix") != "mechanical"]
        if content:
            projects[sl] = {"findings": content}
    payload = {"version": 1, "built_at": ts, "projects": projects,
               "mechanical_applied": applied}
    pending = wdir / "reorgmem_pending.json"
    tmp = pending.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(pending)
    logline = {"ts": ts, "mode": "sweep",
               "projects_with_findings": len(projects),
               "content_findings": sum(len(v["findings"]) for v in projects.values()),
               "mechanical_applied": len(applied)}
    with (wdir / "reorgmem_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(logline) + "\n")
    return pending


# --- show / list ------------------------------------------------------------

def show(memory_dir, slug, *, note=None, status_only=False, inject=False) -> str:
    node = _node_path(memory_dir, slug)
    if note:
        if note not in PROJECT_NOTES:
            raise ProjectError(f"--note {note!r}: one of {', '.join(PROJECT_NOTES)}")
        p = _note_path(memory_dir, slug, note)
        if not p.exists():
            raise ProjectError(f"no {note} note for {slug!r}")
        return p.read_text(encoding="utf-8")
    if not node.exists():
        raise ProjectError(f"no project node for {slug!r}")
    text = node.read_text(encoding="utf-8")
    if status_only or inject:
        body = split_frontmatter(text)[1]
        return _sections(body).get("Status") and \
            "## Status\n" + "\n".join(_sections(body)["Status"]).strip() + "\n" or text
    return text


def list_projects(memory_dir) -> list:
    return [{"slug": sl, "status": (fm.get("status") or "active").strip(),
             "path": str(node)}
            for sl, d, node, text, fm in _iter_project_dirs(memory_dir)]


# --- P2: auto-load cache + prompt matcher -----------------------------------
#
# The Stop-hook render writes <memory>/.weights/project_aliases.json; the
# UserPromptSubmit hook reads it (file-only, ~1 ms, fails open) and injects a
# project's Status block when a prompt names it by slug or alias.  The cache is
# the seam: the hook never parses a node, and this module never touches the hook.

PROJECT_CACHE_NAME = "project_aliases.json"
PROJECT_CACHE_VERSION = 1
_RULES_KEEP = 2   # first N `## Rules & gates` lines carried into the injected block


def _features_tally(feats) -> str:
    from collections import Counter
    c = Counter(f["status"] for f in feats)
    order = ("implemented", "planned", "active-decision", "archived")
    parts = [f"{c[s]} {s}" for s in order if c.get(s)]
    return " · ".join(parts)


def build_inject_block(body: str, *, resolve=None) -> str:
    """The text injected for a matched project: its full ``## Status`` and
    ``## Creds`` sections plus the first two ``## Rules & gates`` lines, with
    ``[[fact_*]]`` refs resolved (when a resolver is given)."""
    secs = _sections(body)
    out = []
    status = secs.get("Status")
    if status is not None:
        out.append("## Status")
        out += [ln for ln in status]
    creds = [ln for ln in secs.get("Creds", []) if ln.strip()]
    if creds:
        out.append("## Creds")
        out += creds
    rules = [ln for ln in secs.get("Rules & gates", []) if ln.strip().startswith("-")]
    if rules:
        out.append("## Rules & gates")
        out += rules[:_RULES_KEEP]
    text = "\n".join(out).strip("\n")
    if resolve is not None:
        try:
            text = resolve(text)
        # FAILOPEN: a resolver failure keeps the literal [[fact_*]] refs, never crashes the cache
        except Exception:
            pass
    return text


def _truncate_lines(text: str, max_bytes: int) -> str:
    """Whole-line truncation to a byte cap (a half line is worse than a short one)."""
    kept = []
    for ln in text.split("\n"):
        cand = "\n".join(kept + [ln])
        if len(cand.encode("utf-8")) > max_bytes and kept:
            break
        kept.append(ln)
    return "\n".join(kept)


def write_cache(memory_dir, *, conn=None, project_bytes=1024, max_n=2) -> dict:
    """Build ``.weights/project_aliases.json`` from the project nodes.  Atomic
    tmp+replace (like ``_write_shed_manifest``).  An alias two nodes claim is
    dropped from BOTH (it is also a check ERROR)."""
    import json
    memory_dir = Path(memory_dir)
    resolve = None
    if conn is not None:
        from memsom.bridge.facts import resolve_fact_refs
        resolve = lambda t: resolve_fact_refs(conn, t)
    # first pass: collect aliases to find clashes
    entries = {}
    alias_owner = {}
    clashed = set()
    for slug, d, node, text, fm in _iter_project_dirs(memory_dir):
        fm_lines, body, _ = split_frontmatter(text)
        aliases = [a.strip().lower() for a in (fm.get("aliases") or "").split(",")
                   if len(a.strip()) >= 3]
        for a in aliases:
            if a in alias_owner and alias_owner[a] != slug:
                clashed.add(a)
            alias_owner[a] = slug
        feats = _parse_features(_sections(body).get("Features", []))
        block = _truncate_lines(build_inject_block(body, resolve=resolve), project_bytes)
        entries[slug] = {
            "aliases": aliases,
            "status": (fm.get("status") or "active").strip(),
            "headline": (fm.get("description") or slug).strip(),
            "last_verified": (fm.get("last-verified") or "").strip(),
            "features": _features_tally(feats),
            "path": str(node.relative_to(memory_dir)).replace("\\", "/"),
            "block": block,
        }
    # drop clashed aliases from every entry
    for e in entries.values():
        e["aliases"] = [a for a in e["aliases"] if a not in clashed]
    payload = {"version": PROJECT_CACHE_VERSION, "built_at": memsom.now_iso(),
               "max_default": max_n, "projects": entries}
    wdir = memory_dir / ".weights"
    wdir.mkdir(parents=True, exist_ok=True)
    dest = wdir / PROJECT_CACHE_NAME
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)
    return {"projects": len(entries), "dropped_aliases": sorted(clashed),
            "path": str(dest)}


def load_cache(memory_dir):
    """Read the alias cache; None on absent/corrupt (the hook then behaves as
    though no project matched — FAILOPEN)."""
    import json
    try:
        p = Path(memory_dir) / ".weights" / PROJECT_CACHE_NAME
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("projects"), dict):
            return data
    # FAILOPEN: an absent/corrupt cache means the hook injects no project, never crashes the turn
    except Exception:
        pass
    return None


def match_projects(prompt, cache, max_n=2):
    """Pure: which projects a prompt names, by slug or alias, word-boundary,
    case-insensitive, ordered by first occurrence.  Returns (primary, also) —
    up to ``max_n`` primary slugs, the rest as an ``also:`` trailer list."""
    if not cache:
        return [], []
    pl = (prompt or "").lower()
    hits = []
    for slug, meta in (cache.get("projects") or {}).items():
        terms = [slug] + list(meta.get("aliases") or [])
        best = None
        for t in terms:
            t = (t or "").lower().strip()
            if len(t) < 3:
                continue
            m = re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", pl)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            hits.append((best, slug))
    hits.sort()
    slugs = [s for _, s in hits]
    return slugs[:max_n], slugs[max_n:]


# --- CLI --------------------------------------------------------------------

def _memdir(args) -> Path:
    return Path(args.memory_dir) if getattr(args, "memory_dir", None) else default_memory_dir()


def _run(fn, *a, **kw) -> int:
    try:
        out = fn(*a, **kw)
    except ProjectError as exc:
        print(f"[memsom] {exc}", file=sys.stderr)
        return exc.code
    if isinstance(out, dict):
        print("[memsom] " + ", ".join(f"{k}={v}" for k, v in out.items()))
    elif isinstance(out, str):
        print(out)
    return 0


def cmd_project_init(args):
    return _run(init_project, _memdir(args), args.slug, aliases=args.alias,
                repo=args.repo, dir_pc=args.dir_pc, dir_mac=args.dir_mac,
                dir_droplet=args.dir_droplet)


def cmd_project_log(args):
    rc = _run(log_entry, _memdir(args), args.slug, args.note, args.entry,
              why=args.why, rejected=args.rejected, cause=args.cause, fix=args.fix,
              where=args.where, covers=args.covers, run=args.run,
              supersedes=args.supersedes, source=args.source)
    return rc


def cmd_project_status(args):
    return _run(set_status, _memdir(args), args.slug, done=args.done, next_=args.next,
                left=args.left, ask=args.ask, cred=args.cred,
                verified=(args.verified if args.verified is not None else None))


def cmd_project_feature(args):
    return _run(set_feature, _memdir(args), args.slug, args.feature, name=args.name,
                status=args.status, evidence=args.evidence, decision=args.decision)


def cmd_project_spec(args):
    return _run(set_spec, _memdir(args), args.slug, args.feature,
                section=args.set, value=args.value, why=args.why)


def cmd_project_show(args):
    return _run(show, _memdir(args), args.slug, note=args.note,
                status_only=args.status, inject=args.inject)


def cmd_project_list(args):
    rows = list_projects(_memdir(args))
    if args.json:
        import json
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"  {r['slug']}  [{r['status']}]  {r['path']}")
    return 0


def cmd_project_check(args):
    findings = check(_memdir(args), slug=args.slug)
    if args.json:
        import json
        print(json.dumps({"findings": findings,
                          "errors": sum(1 for f in findings if f["sev"] == "ERROR")},
                         indent=2))
    else:
        if not findings:
            print("[memsom] project check: clean")
        for f in findings:
            print(f"  {f['sev']:5}  [{f['name']}] {f['target']} — {f['msg']}")
    return 1 if any(f["sev"] == "ERROR" for f in findings) else 0


def cmd_project_cache(args):
    mem = _memdir(args)
    from memsom.lifecycle import forget
    params, _ = forget.load_params(mem / ".weights" / "canonical.json")
    conn = None
    if not args.no_facts:
        try:
            conn = memsom.get_connection()
        # FAILOPEN: no DB just means the cache carries literal fact refs, still useful
        except Exception:
            conn = None
    try:
        out = write_cache(mem, conn=conn,
                          project_bytes=int(params["prompt_hook_project_bytes"]),
                          max_n=int(params["prompt_hook_project_max"]))
    finally:
        if conn is not None:
            conn.close()
    print("[memsom] " + ", ".join(f"{k}={v}" for k, v in out.items()))
    return 0


def cmd_project_reorg(args):
    res = reorg(_memdir(args), slug=args.project, sweep=args.sweep, apply=args.apply)
    if args.json:
        import json
        print(json.dumps(res, indent=2))
        return 0
    print(f"[memsom] reorg: {res['projects']} project(s), "
          f"{len(res['mechanical'])} mechanical, {len(res['content'])} content")
    for f in res["applied"]:
        print(f"  FIXED  [{f['fix']}] {f.get('target','')} "
              f"{f.get('result','') and '('+f['result']+')'}".rstrip())
    for f in res["content"]:
        print(f"  {f['sev']:5}  [{f['name']}] {f['target']} — {f['msg']}")
    if res["pending_path"]:
        print(f"[memsom] pending written: {res['pending_path']}")
    return 1 if any(f["sev"] == "ERROR" for f in res["content"]) else 0


def register(subparsers) -> None:
    """Mount `project` with nested verbs (init/log/status/feature/spec/show/
    list/check/reorg), the same `register(sub)` contract every in-tree module uses."""
    p = subparsers.add_parser("project", help="structured project memory (node + sub-notes)")
    ps = p.add_subparsers(dest="project_command", required=True)

    pi = ps.add_parser("init", help="scaffold a project node + six sub-notes (create-if-absent)")
    pi.add_argument("slug")
    pi.add_argument("--alias", default=None, help="comma list of aliases")
    pi.add_argument("--repo", default=None)
    pi.add_argument("--dir-pc", dest="dir_pc", default=None)
    pi.add_argument("--dir-mac", dest="dir_mac", default=None)
    pi.add_argument("--dir-droplet", dest="dir_droplet", default=None)
    _md(pi)
    pi.set_defaults(func=cmd_project_init)

    pl = ps.add_parser("log", help="append a dated gotcha/decision/test entry")
    pl.add_argument("slug")
    pl.add_argument("note", choices=("gotchas", "decisions", "tests"))
    pl.add_argument("entry")
    for opt in ("why", "rejected", "cause", "fix", "where", "covers", "run",
                "supersedes", "source"):
        pl.add_argument(f"--{opt}", default=None)
    _md(pl)
    pl.set_defaults(func=cmd_project_log)

    pst = ps.add_parser("status", help="append to Done/Next/Left/Needs Matt or a cred pointer")
    pst.add_argument("slug")
    pst.add_argument("--done", default=None)
    pst.add_argument("--next", default=None)
    pst.add_argument("--left", default=None)
    pst.add_argument("--ask", default=None)
    pst.add_argument("--cred", default=None, help="pointer only (env var / path / item)")
    pst.add_argument("--verified", nargs="?", const=True, default=None,
                     help="bump last-verified (optional YYYY-MM-DD; default today)")
    _md(pst)
    pst.set_defaults(func=cmd_project_status)

    pf = ps.add_parser("feature", help="add/update a feature line + its spec note")
    pf.add_argument("slug")
    pf.add_argument("feature")
    pf.add_argument("--name", default=None)
    pf.add_argument("--status", choices=FEATURE_STATUSES, default=None)
    pf.add_argument("--evidence", default=None, help="(MEASURED) … — required for implemented")
    pf.add_argument("--decision", default=None, help="D-… id when the change is a decision")
    _md(pf)
    pf.set_defaults(func=cmd_project_feature)

    psp = ps.add_parser("spec", help="edit one section of one feature note (with a why)")
    psp.add_argument("slug")
    psp.add_argument("feature")
    psp.add_argument("--set", required=True,
                     choices=("purpose", "behaviour", "interfaces", "acceptance"))
    psp.add_argument("value")
    psp.add_argument("--why", required=True)
    _md(psp)
    psp.set_defaults(func=cmd_project_spec)

    psh = ps.add_parser("show", help="print a node or one sub-note")
    psh.add_argument("slug")
    psh.add_argument("--note", default=None, choices=PROJECT_NOTES)
    psh.add_argument("--status", action="store_true")
    psh.add_argument("--inject", action="store_true")
    _md(psh)
    psh.set_defaults(func=cmd_project_show)

    pls = ps.add_parser("list", help="list project nodes")
    pls.add_argument("--json", action="store_true")
    _md(pls)
    pls.set_defaults(func=cmd_project_list)

    pc = ps.add_parser("check", help="audit-shaped structural findings for project nodes")
    pc.add_argument("slug", nargs="?", default=None)
    pc.add_argument("--json", action="store_true")
    _md(pc)
    pc.set_defaults(func=cmd_project_check)

    pca = ps.add_parser("cache", help="rebuild the prompt-hook alias cache (project_aliases.json)")
    pca.add_argument("--no-facts", action="store_true",
                     help="do not open a DB to resolve [[fact_*]] refs (leave them literal)")
    _md(pca)
    pca.set_defaults(func=cmd_project_cache)

    pr = ps.add_parser("reorg", help="maintenance sweep: report + mechanical fixes")
    pr.add_argument("--project", default=None, help="limit to one slug")
    pr.add_argument("--sweep", action="store_true",
                    help="headless: apply mechanical fixes + write reorgmem_pending.json")
    pr.add_argument("--apply", action="store_true",
                    help="apply the mechanical fixes now (interactive)")
    pr.add_argument("--json", action="store_true")
    _md(pr)
    pr.set_defaults(func=cmd_project_reorg)


def _md(parser) -> None:
    parser.add_argument("--memory-dir", dest="memory_dir", default=None,
                        help="override the memory dir (default: live PC store)")


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_project")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
