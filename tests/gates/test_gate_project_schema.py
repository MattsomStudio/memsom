"""GATE: the structured-project-memory schema checker actually goes RED.

A check that cannot fail is not a check (RULES.md §1.15). project.check is what
the audit and /reorgmem lean on to keep a project node well-formed; this gate
pins that three specific corruptions each flip a specific finding, and that a
clean scaffold is clean. Each inversion mutates exactly ONE thing.

Function-style, no DB (project memory is pure file I/O); the session conftest
pins MEMDAG_DB regardless.
"""

from pathlib import Path

from memsom.bridge import project as P


def _scaffold(tmp_path) -> Path:
    mem = tmp_path / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    P.init_project(mem, "demo")
    return mem


def _errors(mem, slug=None):
    return {f["name"] for f in P.check(mem, slug) if f["sev"] == "ERROR"}


def test_control_clean_scaffold_has_no_findings(tmp_path):
    mem = _scaffold(tmp_path)
    assert P.check(mem, "demo") == []


def test_missing_rules_section_flips_project_schema(tmp_path):
    mem = _scaffold(tmp_path)
    n = P._node_path(mem, "demo")
    n.write_text(n.read_text(encoding="utf-8").replace("## Rules & gates\n\n", ""),
                 encoding="utf-8")
    assert _errors(mem, "demo") == {"project-schema"}


def test_creds_value_flips_project_creds_value(tmp_path):
    mem = _scaffold(tmp_path)
    n = P._node_path(mem, "demo")
    n.write_text(n.read_text(encoding="utf-8").replace(
        "## Creds\n", "## Creds\n- token=abcd1234efgh5678 (inline)\n"), encoding="utf-8")
    assert _errors(mem, "demo") == {"project-creds-value"}


def test_alias_clash_flips_project_alias_clash(tmp_path):
    mem = _scaffold(tmp_path)
    P.init_project(mem, "beta")
    for slug in ("demo", "beta"):
        n = P._node_path(mem, slug)
        n.write_text(n.read_text(encoding="utf-8").replace(
            "status: active", "status: active\naliases: shared, %s-only" % slug),
            encoding="utf-8")
    assert _errors(mem) == {"project-alias-clash"}
