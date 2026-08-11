"""GATE for Q11 -- scripts/fact_refs.py's corpus-check logic (Phase 8).

The plan's own exit gate invocation is bare (`python scripts/fact_refs.py
--check`, no --memory-dir): with no memory dir configured this is "0
checked, exit 0" by design (see the script's docstring, and AMENDMENTS.md
A-16/A-17 -- a copy-confined refactor agent must never point --memory-dir
at Matt's live store). What a copy-confined agent CAN and must prove is
that the checking logic itself is correct against synthetic fixtures.
"""

import importlib.util
import io
import sys
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


def test_bare_invocation_is_zero_checked_exit_zero(monkeypatch):
    monkeypatch.delenv("MEMSOM_MEMORY_DIR", raising=False)
    rc, out = _run("--check")
    assert rc == 0
    assert "0 checked" in out


def test_missing_memory_dir_is_zero_checked_exit_zero(tmp_path):
    rc, out = _run("--check", "--memory-dir", str(tmp_path / "does_not_exist"))
    assert rc == 0
    assert "0 checked" in out


def test_resolved_ref_is_not_a_violation(tmp_path):
    (tmp_path / "fact_gpu.md").write_text(
        "---\nname: fact-gpu\ndescription: g\ntype: fact\nvalue: RTX 5070\n"
        "section: Facts\n---\nbody\n", encoding="utf-8")
    (tmp_path / "project_memsom_x.md").write_text(
        "---\nname: x\ndescription: d\ntype: project\nsection: Personal projects\n"
        "---\nruns on [[fact_gpu]]\n", encoding="utf-8")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0
    assert "checked: 1, violations: 0" in out


def test_dangling_ref_is_a_violation(tmp_path):
    (tmp_path / "project_memsom_x.md").write_text(
        "---\nname: x\ndescription: d\ntype: project\nsection: Personal projects\n"
        "---\nruns on [[fact_nonexistent]]\n", encoding="utf-8")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 1
    assert "has no fact_nonexistent.md" in out


def test_bare_measured_number_with_no_fact_ref_is_a_violation(tmp_path):
    (tmp_path / "project_memsom_x.md").write_text(
        "---\nname: x\ndescription: d\ntype: project\nsection: Personal projects\n"
        "---\nmeasured 22265 LOC, no fact reference at all\n", encoding="utf-8")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 1
    assert "bare measured number" in out


def test_prose_with_no_numbers_is_clean(tmp_path):
    (tmp_path / "project_memsom_x.md").write_text(
        "---\nname: x\ndescription: d\ntype: project\nsection: Personal projects\n"
        "---\nno measured values in this one at all\n", encoding="utf-8")

    rc, out = _run("--check", "--memory-dir", str(tmp_path))
    assert rc == 0
    assert "checked: 1, violations: 0" in out
