"""GATE: the /reorgmem maintenance checks each actually go RED.

`project reorg` is the deterministic half of /reorgmem — the Sunday sweep leans
on it to keep every project node well-formed with no model in the loop.  A check
that cannot fail is not a check (RULES.md §1.15), so this gate pins that each
maintenance check flips its own finding on exactly one corruption, and that a
clean scaffold (with a real index_hook) reorg-checks clean.

Function-style, no DB (project memory is pure file I/O); the session conftest
pins MEMDAG_DB regardless.
"""

from pathlib import Path

from memsom.bridge import project as P


def _clean(tmp_path) -> Path:
    """A scaffold whose index_hook is set — reorg finds nothing on it."""
    mem = tmp_path / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    P.init_project(mem, "demo")
    n = P._node_path(mem, "demo")
    n.write_text(n.read_text(encoding="utf-8").replace(
        "index_hook: (project node — set the Status headline)",
        "index_hook: a real one-line hook"), encoding="utf-8")
    return mem


def _names(mem, slug="demo") -> set:
    r = P.reorg(mem, slug=slug)                 # report-only, no writes
    return {f["name"] for f in r["content"] + r["mechanical"]}


def test_control_clean_scaffold_reorgs_clean(tmp_path):
    mem = _clean(tmp_path)
    assert _names(mem) == set()


def test_missing_subnote_flips_reorg_subnote_missing(tmp_path):
    mem = _clean(tmp_path)
    P._note_path(mem, "demo", "gotchas").unlink()
    assert _names(mem) == {"reorg-subnote-missing"}


def test_wrong_subnote_kind_flips_reorg_subnote_kind(tmp_path):
    mem = _clean(tmp_path)
    p = P._note_path(mem, "demo", "gotchas")
    p.write_text(p.read_text(encoding="utf-8").replace(
        "kind: project-log", "kind: project-ref"), encoding="utf-8")
    assert _names(mem) == {"reorg-subnote-kind"}


def test_oversized_log_flips_reorg_subnote_cap(tmp_path):
    mem = _clean(tmp_path)
    p = P._note_path(mem, "demo", "gotchas")
    fm, body, _ = P.split_frontmatter(p.read_text(encoding="utf-8"))
    bloat = "\n".join(f"- G-20200101-{i:02d} ({'2020-01-01'}) **x{i}**" for i in range(160))
    p.write_text("---\n" + "\n".join(fm) + "\n---\n## Entries\n" + bloat + "\n",
                 encoding="utf-8")
    assert _names(mem) == {"reorg-subnote-cap"}


def test_dangling_wikilink_flips_reorg_link_broken(tmp_path):
    mem = _clean(tmp_path)
    n = P._node_path(mem, "demo")
    n.write_text(n.read_text(encoding="utf-8").replace(
        "## Pointers", "## Pointers\n- see [[no_such_memory]]"), encoding="utf-8")
    assert _names(mem) == {"reorg-link-broken"}


def test_missing_fact_ref_flips_reorg_fact_missing(tmp_path):
    mem = _clean(tmp_path)
    n = P._node_path(mem, "demo")
    n.write_text(n.read_text(encoding="utf-8").replace(
        "## Where\n", "## Where\n- lives at [[fact_no_such_fact]]\n"), encoding="utf-8")
    assert _names(mem) == {"reorg-fact-missing"}


def test_placeholder_index_hook_flips_reorg_index_hook(tmp_path):
    mem = _clean(tmp_path)
    n = P._node_path(mem, "demo")
    n.write_text(n.read_text(encoding="utf-8").replace(
        "index_hook: a real one-line hook", "index_hook: "), encoding="utf-8")
    assert _names(mem) == {"reorg-index-hook"}


def test_rule_not_in_architecture_flips_reorg_rules_subset(tmp_path):
    mem = _clean(tmp_path)
    n = P._node_path(mem, "demo")
    n.write_text(n.read_text(encoding="utf-8").replace(
        "## Rules & gates\n", "## Rules & gates\n- never zorp the frobnitz before a push\n"),
        encoding="utf-8")
    assert _names(mem) == {"reorg-rules-subset"}


def test_sync_conflict_copy_flips_reorg_sync_conflict(tmp_path):
    mem = _clean(tmp_path)
    g = P._note_path(mem, "demo", "gotchas")
    (g.parent / "project_demo_gotchas.sync-conflict-20260101-120000-ABCDEF.md").write_text(
        g.read_text(encoding="utf-8"), encoding="utf-8")
    assert _names(mem) == {"reorg-sync-conflict"}
