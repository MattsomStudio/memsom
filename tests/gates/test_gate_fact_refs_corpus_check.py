"""GATE for Q11 -- scripts/fact_refs.py's corpus-check logic.

This is a REWRITE (the checker it tests was rewritten from a flat, always-
0-checked, digit-heuristic stub into a real fail-closed gate -- see
scripts/fact_refs.py's module docstring). AMENDMENTS.md A-17 promised this
file would prove the checker; it never existed against the old checker (no
test referenced fact_refs.py at all). It does now, against synthetic
fixtures built in tmp_path (never the real store) that mirror the production
layout: `fact_*.md` flat at the store root, `project_memsom_*.md` under
`projects/<group>/`.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fact_refs", Path(__file__).resolve().parents[2] / "scripts" / "fact_refs.py")
fact_refs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fact_refs)


def _run(*argv):
    buf = io.StringIO()
    old_argv = sys.argv
    sys.argv = ["fact_refs.py", *argv]
    try:
        with redirect_stdout(buf):
            rc = fact_refs.main()
    finally:
        sys.argv = old_argv
    return rc, buf.getvalue()


def _fact(tmp_path: Path, stem: str, value: str, *, name: str | None = None) -> None:
    (tmp_path / f"{stem}.md").write_text(
        "---\n"
        f"name: {name or stem.replace('_', '-')}\n"
        "description: d\n"
        "type: fact\n"
        f"value: {value}\n"
        "last-verified: 2026-09-01\n"
        "section: Live state\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


def _project(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nname: x\ndescription: d\ntype: project\nsection: Personal projects\n"
        "---\n" + body,
        encoding="utf-8",
    )


# --- basic discovery / fail-closed cases -------------------------------------

def test_good_cited_ref_is_not_a_violation(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "LOC is [[fact_memsom_loc]], cited properly.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0
    assert "violations: 0" in out


def test_decoy_unrelated_numbers_are_not_violations(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _fact(tmp_path, "fact_income_biweekly", "1200")
    _project(
        tmp_path / "projects" / "memsom" / "project_memsom_x.md",
        "Dated 2026-08-11, commit 60dce1c, ticket IT-5700, version 0.2.0, "
        "12 phases done, paid 1200 $ this period. LOC cited: [[fact_memsom_loc]].\n",
    )

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0, out
    assert "violations: 0" in out


def test_bare_inline_value_is_a_violation_with_file_and_line(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "line one\n12,500 LOC bare, no citation.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 1
    assert "bare value" in out
    assert "fact_memsom_loc" in out
    # frontmatter is 6 lines (open fence + 4 kv + close fence), body line 2
    # ("12,500 LOC bare...") is file line 8 -- assert the file:line locator.
    assert ":8:" in out


def test_bare_value_inside_code_span_is_not_a_violation(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "the value is `12,500 LOC` inside a code span.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0, out
    assert "violations: 0" in out


def test_bare_value_inside_fenced_code_block_is_not_a_violation(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(
        tmp_path / "projects" / "memsom" / "project_memsom_x.md",
        "prose\n```\n12500 LOC in a fence\n```\nmore prose\n",
    )

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0, out
    assert "violations: 0" in out


def test_dangling_ref_is_a_violation(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "runs on [[fact_memsom_nope]]\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 1
    assert "has no fact_memsom_nope.md" in out


def test_empty_store_is_fail_closed_zero_checked(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    # no project files at all

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 1
    assert "0 checked" in out


def test_no_memsom_fact_fails_closed_unless_allowed(tmp_path):
    _fact(tmp_path, "fact_gpu", "RTX 5070")  # not fact_memsom_*
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "clean prose, nothing to cite here.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 1
    assert "no fact_memsom_* fact exists" in out

    rc2, out2 = _run("--check", "--memory-dir", str(tmp_path),
                      "--allow-no-memsom-facts")
    assert rc2 == 0, out2


def test_nested_project_file_under_projects_is_found(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(tmp_path / "projects" / "memsom" / "sub" / "project_memsom_nested.md",
              "clean prose, [[fact_memsom_loc]] cited.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0, out
    assert "checked: 1 files" in out


def test_generated_index_under_projects_is_not_scanned(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "LOC is [[fact_memsom_loc]].\n")
    # bridge-render writes projects/INDEX.md from the same files, with the
    # fact RESOLVED to its value -- a scan of it would flag the digest itself.
    (tmp_path / "projects" / "INDEX.md").write_text(
        "# Projects\n- [x](memsom/project_memsom_x.md) — LOC is 12500\n",
        encoding="utf-8")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0, out
    assert "checked: 1 files" in out


def test_short_value_is_not_attributable_so_never_flagged(tmp_path):
    _fact(tmp_path, "fact_memsom_suite_time", "252")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "port 252 on the router, 252 tickets closed; nothing about the suite.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0, out
    assert "violations: 0" in out


def test_dotted_version_is_matched_whole_not_as_its_first_float(tmp_path):
    _fact(tmp_path, "fact_memsom_version", "0.2.0")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "demote_below 0.2 and rs 0.25 are tuning knobs, not versions.\n"
              "shipped memsom 0.2.0 to PyPI.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 1
    assert out.count("bare value") == 1, out
    assert "'0.2.0'" in out
    assert ":8:" in out   # only the second body line (file line 8: 6 fm lines + 2) is a hit


def test_distinct_false_opts_a_fact_out_of_the_bare_scan(tmp_path):
    (tmp_path / "fact_memsom_version.md").write_text(
        "---\nname: fact-memsom-version\ndescription: d\ntype: fact\n"
        "value: 0.2.0\nlast-verified: 2026-09-02\nsection: none\ndistinct: false\n"
        "---\nbody\n", encoding="utf-8")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "memsom-panel is also at 0.2.0, a different package.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0, out
    assert "facts: 1 (memsom: 1)" in out   # still counts as the memsom fact
    assert "violations: 0" in out


def test_candidates_never_gates_and_lists_the_phrase(tmp_path):
    _fact(tmp_path, "fact_memsom_loc", "12500")
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "12,500 LOC bare, no citation.\n")

    rc, out = _run("--candidates", "--memory-dir", str(tmp_path))
    assert rc == 0
    assert "12,500 LOC" in out
    assert "candidates: 1" in out


# --- prior-value (--db) -------------------------------------------------------

def _build_prior_value_db(tmp_path):
    """fact_memsom_loc: 12500 (v1) superseded by 13000 (v2), via memsom's own
    bridge-render entry point (the same one the Stop hook + `memsom
    bridge-render` CLI both call) -- not a hand-built fixture DB.
    """
    from memsom.storage.db import get_connection
    from memsom.bridge.bridge_render import bridge_render
    from memsom.bridge.facts import cmd_fact_set

    db_path = tmp_path / "chain.db"
    _fact(tmp_path, "fact_memsom_loc", "12500")

    conn = get_connection(str(db_path))
    try:
        bridge_render(conn, str(tmp_path))  # imports v1 (value=12500)
        cmd_fact_set(types.SimpleNamespace(
            stem="fact_memsom_loc", value="13000", unit=None,
            verified="2026-09-01", memory_dir=str(tmp_path)))
        bridge_render(conn, str(tmp_path))  # detects the file change, tombstones
                                             # v1, imports v2 (value=13000)
    finally:
        conn.close()
    return db_path


def test_prior_value_from_db_chain_is_caught(tmp_path):
    db_path = _build_prior_value_db(tmp_path)
    # fact file now reads 13000; project body cites neither, writes the OLD
    # (superseded) value bare -- only the --db supersede chain can catch this.
    _project(tmp_path / "projects" / "memsom" / "project_memsom_x.md",
              "measured 12500 at the time, bare.\n")

    rc, out = _run("--check", "--memory-dir", str(tmp_path), "--db", str(db_path))
    assert rc == 1, out
    assert "fact_memsom_loc" in out
    assert "bare value" in out
