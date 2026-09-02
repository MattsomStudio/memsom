#!/usr/bin/env python3
"""fact_refs --check -- every measured value in a project_memsom_* memory resolves
to a fact_* file (Q11, dogfooding memsom's own fact layer on itself).

The memory store this checks is EXTERNAL to this repo (Matt's `~/.claude`
memory directory), so it is a path argument / env var, not a repo-relative
walk like every other script here.

FAIL-CLOSED CHANGE (this rewrite): the previous version treated "0 project
files found" and "0 fact_memsom_* files" as a pass ("0 checked, exit 0" --
"absence is not failure before the deliverable exists"). AMENDMENTS.md A-17
is explicit that this grace expires at promote-time: once the fact layer has
shipped, an empty result from a gate that is supposed to be finding things
is a gate that has gone blind, not a gate reporting a clean corpus. Both
zero-cases are now violations (see `--allow-no-memsom-facts` for the one
still-legitimate exception -- a store this checker runs against before the
Q11 migration itself has landed).

Discovery
---------
Project files = every `project_memsom*.md` anywhere under the memory dir,
UNION every `*.md` under `<memory-dir>/projects/` at any depth (the store's
project files moved under `projects/<group>/` in 2026-08; this also covers
any future project group without a new pattern per group).

Fact files = `<memory-dir>/fact_*.md`, flat (the store keeps facts at the
memory-dir root, never nested). stem = filename stem; value = the
frontmatter `value:` key.

Checks (--check)
-----------------
1. Dangling refs: every `[[fact_x]]` in a project body whose `fact_x.md`
   does not exist among the discovered fact files.
2. Bare values: for facts whose stem starts with `fact_memsom_` ONLY (the
   refactor's own numbers -- restricting to this prefix is what keeps an
   unrelated fact's value, e.g. a dose of "50", from colliding with a
   hardware string like "...5070"), every project body is scanned for that
   fact's current value AND every prior value from the DB's supersede chain
   (`--db`), written bare instead of cited as `[[fact_memsom_x]]`.
3. Fail closed: 0 project files -> violation ("0 checked"), 0
   `fact_memsom_*` files -> violation, unless `--allow-no-memsom-facts`.

`--candidates` is advisory only (always exit 0): every number+unit hit in
project bodies, deduped, for choosing what becomes a fact next.

Stdlib + memsom imports only; no new dependency.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

# scripts/ is not inside the memsom/ package (rule 4's DB-connection-owner
# rule does not apply here), but it still needs memsom on sys.path -- the
# worker harness sets PYTHONPATH to the tree under test; this insert is a
# fallback for the case (the eventual real exit-gate line, run from an
# ordinary repo root against the venv's own editable install) where it is
# not set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsom.kernel.frontmatter import frontmatter_dict, split_frontmatter  # noqa: E402

_FACT_REF = re.compile(r"\[\[(fact_[A-Za-z0-9_-]+)\]\]")
_BRACKET_SPAN = re.compile(r"\[\[.*?\]\]")
_INLINE_CODE = re.compile(r"`[^`]*`")
_FENCE = re.compile(r"^\s*```")
_COMMA_UNDERSCORE_BETWEEN_DIGITS = re.compile(r"(?<=\d)[,_](?=\d)")

_UNITS = (r"tests|LOC|lines|MB|GB|KB|ms|s|%|knobs|tokens|commits|files|"
          r"nodes|rows")
_CANDIDATE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:" + _UNITS + r")(?![A-Za-z0-9_])")


# --- discovery ---------------------------------------------------------------

def _discover_project_files(root: Path) -> list[Path]:
    found = set(root.rglob("project_memsom*.md"))
    projects_dir = root / "projects"
    if projects_dir.is_dir():
        found |= set(projects_dir.rglob("*.md"))
    return sorted(found)


def _discover_fact_files(root: Path) -> dict[str, str | None]:
    """{stem: value_or_None} for every fact_*.md at the store root."""
    out: dict[str, str | None] = {}
    for path in sorted(root.glob("fact_*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        fm, _ = frontmatter_dict(text)
        out[path.stem] = fm.get("value")
    return out


# --- value -> match patterns --------------------------------------------------

def _patterns_for_value(value: str) -> list[tuple[str, str]]:
    """[(kind, pattern), ...] for one fact value. kind is 'numeric' or 'string'.

    Full string (case-insensitive, whitespace-collapsed) always; the leading
    numeric token (',' and '_' stripped) additionally when the value starts
    with a number. Patterns shorter than 2 chars are dropped by the caller.
    """
    value = (value or "").strip()
    if not value:
        return []
    out = [("string", re.sub(r"\s+", " ", value).lower())]
    m = re.match(r"^-?\d[\d,_]*(?:\.\d+)?", value)
    if m:
        numeric = m.group(0).replace(",", "").replace("_", "")
        out.append(("numeric", numeric))
    return out


def _owned_patterns(memsom_fact_values: dict[str, set[str]]) -> dict[str, list[tuple[str, str]]]:
    """stem -> deduped [(kind, pattern), ...], length-2+ only."""
    out: dict[str, list[tuple[str, str]]] = {}
    for stem, values in memsom_fact_values.items():
        seen = set()
        pats = []
        for v in values:
            for kind, pat in _patterns_for_value(v):
                if len(pat) < 2:
                    continue
                key = (kind, pat)
                if key in seen:
                    continue
                seen.add(key)
                pats.append(key)
        if pats:
            out[stem] = pats
    return out


# --- masking (code spans / wikilinks never count as bare prose) -------------

def _mask(line: str) -> str:
    masked = _INLINE_CODE.sub(lambda m: " " * (m.end() - m.start()), line)
    masked = _BRACKET_SPAN.sub(lambda m: " " * (m.end() - m.start()), masked)
    return masked


def _body_line_offset(text: str, body: str) -> int:
    """File line number of body's line 1 minus 1 (i.e. lines consumed by fm)."""
    return len(text.split("\n")) - len(body.split("\n"))


# --- per-file scans -----------------------------------------------------------

def _dangling_refs(rel: str, body: str, fact_stems: set[str]) -> list[str]:
    violations = []
    for ref in sorted(set(_FACT_REF.findall(body))):
        if ref not in fact_stems:
            violations.append(f"{rel}: [[{ref}]] has no {ref}.md")
    return violations


def _bare_values(rel: str, text: str, body: str,
                  owned: dict[str, list[tuple[str, str]]]) -> list[str]:
    if not owned:
        return []
    violations = []
    offset = _body_line_offset(text, body)
    in_fence = False
    for i, line in enumerate(body.split("\n"), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        masked = _mask(line)
        norm_ws = re.sub(r"\s+", " ", masked).lower()
        digits_joined = _COMMA_UNDERSCORE_BETWEEN_DIGITS.sub("", masked)
        file_line = offset + i
        for stem, pats in owned.items():
            hit = False
            for kind, pat in pats:
                if kind == "numeric":
                    if re.search(r"(?<!\d)" + re.escape(pat) + r"(?!\d)", digits_joined):
                        hit = True
                        break
                else:
                    if pat in norm_ws:
                        hit = True
                        break
            if hit:
                violations.append(
                    f"{rel}:{file_line}: bare value '{pat}' owned by {stem} "
                    f"- cite [[{stem}]]"
                )
    return violations


def _candidates(rel: str, text: str, body: str) -> list[str]:
    out = []
    offset = _body_line_offset(text, body)
    for i, line in enumerate(body.split("\n"), start=1):
        for m in _CANDIDATE.finditer(line):
            out.append(f"{rel}:{offset + i}: {m.group(0).strip()}")
    return out


# --- --db: prior values from the supersede chain -----------------------------

def _open_db_readonly(db_path: Path):
    """Returns (conn, note) or (None, note). Never raises."""
    try:
        from memsom.storage.db import get_connection
        conn = get_connection(str(db_path), read_only=True)
        return conn, "memsom.storage.db.get_connection(read_only=True)"
    # get_connection's read-only branch is documented to open any existing
    # path via `file:...?mode=ro` already, so this fallback is defensive
    # (e.g. a future version narrowing what paths it accepts), not the
    # expected path -- scripts/ is outside memsom/, so a bare sqlite3.connect
    # here does not trip rule 4.
    # FAILOPEN: allowed, fall back to a raw read-only sqlite3 open and say so in the note
    except Exception as exc:  # noqa: BLE001
        try:
            uri = f"file:{db_path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            return conn, f"sqlite3.connect(uri=True, mode=ro) fallback ({exc})"
        # FAILOPEN: allowed, an unopenable --db degrades to 'no prior values' and is reported in the note
        except Exception as exc2:  # noqa: BLE001
            return None, f"could not open --db {db_path}: {exc2}"


def _prior_values(conn, stem: str) -> list[str]:
    from memsom.bridge.facts import fact_versions
    out = []
    for v in fact_versions(conn, stem):
        if v.get("value") is not None:
            out.append(str(v["value"]))
    return out


# --- main ----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--memory-dir",
                     default=os.environ.get("MEMSOM_MEMORY_DIR", ""),
                     help="store root: project_*.md / projects/ / fact_*.md")
    ap.add_argument("--db", default=None,
                     help="memdag.db to read the fact supersede chain from "
                          "(read-only; never writes)")
    ap.add_argument("--allow-no-memsom-facts", action="store_true",
                     help="do not fail when zero fact_memsom_* files exist "
                          "(only legitimate before the Q11 migration lands)")
    args = ap.parse_args()

    if not args.memory_dir:
        print("no --memory-dir / MEMSOM_MEMORY_DIR set")
        if args.check:
            print("0 checked -- no memory dir given, refusing to pass with "
                  "nothing verified")
            print("checked: 0 files, facts: 0 (memsom: 0), violations: 1")
            return 1
        return 0

    root = Path(args.memory_dir)
    if not root.is_dir():
        print(f"memory dir {root} does not exist")
        if args.check:
            print("0 checked -- memory dir missing, refusing to pass with "
                  "nothing verified")
            print("checked: 0 files, facts: 0 (memsom: 0), violations: 1")
            return 1
        return 0

    project_files = _discover_project_files(root)
    fact_values = _discover_fact_files(root)
    fact_stems = set(fact_values)
    memsom_stems = {s for s in fact_stems if s.startswith("fact_memsom_")}

    violations: list[str] = []

    if args.check and not project_files:
        violations.append("0 checked -- no project files found under "
                           f"{root}, refusing to pass with nothing verified")

    if args.check and not memsom_stems and not args.allow_no_memsom_facts:
        violations.append("no fact_memsom_* fact exists (Q11 not migrated)")

    if args.check and project_files:
        # collect owned values (current + prior via --db) for memsom facts only
        memsom_fact_values: dict[str, set[str]] = {
            s: {fact_values[s]} if fact_values[s] else set() for s in memsom_stems
        }
        db_conn = None
        if args.db and memsom_stems:
            db_path = Path(args.db)
            db_conn, note = _open_db_readonly(db_path)
            print(f"[fact_refs] --db: {note}")
            if db_conn is not None:
                for stem in memsom_stems:
                    memsom_fact_values[stem].update(_prior_values(db_conn, stem))
        owned = _owned_patterns(memsom_fact_values)
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:  # noqa: BLE001
                # FAILOPEN: closing a read-only connection cannot corrupt
                # anything; a failure here is not worth failing the gate over.
                pass

        for path in project_files:
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            _, body = frontmatter_dict(text)
            violations.extend(_dangling_refs(rel, body, fact_stems))
            violations.extend(_bare_values(rel, text, body, owned))

    if args.candidates:
        cand_lines: list[str] = []
        seen = set()
        for path in project_files:
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            _, body = frontmatter_dict(text)
            for line in _candidates(rel, text, body):
                if line not in seen:
                    seen.add(line)
                    cand_lines.append(line)
        for line in cand_lines:
            print(line)
        print(f"candidates: {len(cand_lines)}")

    for v in violations:
        print(f"  {v}")
    print(f"checked: {len(project_files)} files, facts: {len(fact_stems)} "
          f"(memsom: {len(memsom_stems)}), violations: {len(violations)}")

    return 1 if (args.check and violations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
