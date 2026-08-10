"""GATE for CORE-03 / CORE-04 — provenance edges must never be hard-deleted, and
a tombstone must always cascade.

Invariant under test (ARCHITECTURE.md:111, memsom/__init__.py:9-11):
    "History is immutable: ... never an in-place mutation or hard delete."
    "revoke = tombstone + cascade to all transitive descendants."

CONTROL TEST (run 2026-07-31 against 9d165b1) — measured results:

    test_no_hard_delete_on_frozen_tables            FAILED  (1 site: stale.py:402)
    test_freshen_preserves_provenance_edge          FAILED  (edge 3->1 gone)
    test_revocation_reaches_freshened_descendant    FAILED  (0 descendants hit)
    test_bridge_sweep_cascades_to_descendants       FAILED  (derived stayed live)
    test_freshen_is_atomic_across_both_phases       FAILED  (edge rewired despite raise)

All five are RED-xfail gates. The first is a static AST gate — it is the one
that also catches the NEXT hard delete somebody adds, which is why it is worth
having even after stale.py is fixed.

RUN:
    pytest gates/test_gate_provenance_immutability.py -v
"""
import ast
import re
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "memsom"

# The two frozen-core tables. Rows in these are history.
FROZEN_TABLES = ("nodes", "edges")
_DELETE = re.compile(r"\bDELETE\s+FROM\s+(%s)\b" % "|".join(FROZEN_TABLES), re.I)
_EXEC = {"execute", "executemany", "executescript"}


def _sql_of(node):
    """Reconstruct a SQL string from Constant / f-string / implicit + concat."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant)
                       and isinstance(v.value, str) else " ? "
                       for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sql_of(node.left) + _sql_of(node.right)
    return ""


def _hard_delete_sites():
    """AST (never grep) — a comment explaining the banned pattern must not fire."""
    hits = []
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in _EXEC and n.args):
                sql = " ".join(_sql_of(n.args[0]).split())
                if _DELETE.search(sql):
                    hits.append(
                        f"{path.relative_to(REPO).as_posix()}:{n.lineno}  {sql[:90]}")
    return hits


@pytest.mark.xfail(strict=True, reason="MS-11: stale.freshen DELETEs FROM edges (stale.py:402)")
def test_no_hard_delete_on_frozen_tables():
    """Static gate: zero DELETE statements against nodes/edges anywhere.

    CONTROL-TESTED: currently RED with exactly one site
    (memsom/integrity/stale.py:402, inside freshen()). Inverting the gate — i.e.
    asserting `hits` is non-empty — goes green, so the gate is discriminating.
    """
    hits = _hard_delete_sites()
    assert hits == [], (
        "hard DELETE on a frozen (history) table:\n  " + "\n  ".join(hits))


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    db = tmp_path / "gate.db"
    monkeypatch.setenv("MEMDAG_DB", str(db))
    monkeypatch.setenv("MEMDAG_HOME", str(tmp_path))
    import memsom
    assert memsom.db_path() == db, "MEMDAG_DB pin did not take effect"
    c = memsom.get_connection()
    yield c
    c.close()


def _freshened(conn):
    import memsom
    from memsom.integrity import stale as memsom_stale
    from memsom.retrieval import rederive as memsom_rederive

    memsom_stale.migrate(conn)
    memsom_rederive.migrate(conn)
    with conn:
        p_old = memsom.insert_node(conn, "The passphrase is correct-horse.",
                                   "user", source_ref="s.md")
        p_new = memsom.insert_node(conn, "The passphrase was rotated.",
                                   "user", source_ref="s2.md")
    derived, _ = memsom.derive_node(
        conn, f"- The passphrase is correct-horse. [mem:{p_old}|user]", [p_old])
    memsom_stale.record_source_supersession(conn, p_old, p_new, "s.md")
    memsom_stale.mark_stale_cascade(conn, p_old, "changed")
    memsom_stale.freshen(conn, derived)
    return p_old, p_new, derived


@pytest.mark.xfail(strict=True, reason="MS-11: freshen hard-deletes the old provenance edge")
def test_freshen_preserves_provenance_edge(conn):
    """The came-from edge is the product. freshen must not erase it."""
    import memsom
    p_old, _p_new, derived = _freshened(conn)
    parents = [r[0] for r in memsom.parents_of(conn, derived)]
    assert p_old in parents, (
        f"freshen deleted the provenance edge {derived}->{p_old}; "
        f"parents are now {parents}")


@pytest.mark.xfail(strict=True, reason="MS-11: the deleted edge makes revoke_cascade miss the descendant")
def test_revocation_reaches_freshened_descendant(conn):
    """Revoking a source must tombstone anything still quoting it."""
    import memsom
    p_old, _p_new, derived = _freshened(conn)
    content = memsom.get_node(conn, derived)["content"]
    assert "correct-horse" in content, "precondition: derived still quotes the source"

    memsom.revoke_cascade(conn, p_old, "source retracted")
    dead = memsom.get_node(conn, derived)["tombstoned"]
    assert dead, (
        f"node {derived} still quotes revoked source {p_old} verbatim but "
        f"survived revoke_cascade")


@pytest.mark.xfail(strict=True, reason="MS-11: rewire and regenerate are separate transactions")
def test_freshen_is_atomic_across_both_phases(conn, tmp_path, monkeypatch):
    """freshen's docstring (stale.py:381) says "explicit, ATOMIC, audited".

    Phase 1 (edge rewire) commits in its own `with conn:`; phase 2
    (rederive.regenerate) runs outside it. Force phase 2 to raise and assert the
    DAG is unchanged. Reproduced here by NOT running rederive.migrate(), which
    makes get_recipe raise `no such table: derivation_recipe`.
    """
    import memsom
    from memsom.integrity import stale as memsom_stale
    memsom_stale.migrate(conn)
    with conn:
        a = memsom.insert_node(conn, "alpha payload text", "user", source_ref="s.md")
        b = memsom.insert_node(conn, "beta payload text", "user", source_ref="s2.md")
    d, _ = memsom.derive_node(conn, "derived from alpha", [a])
    memsom_stale.record_source_supersession(conn, a, b, "s.md")

    before = sorted(r[0] for r in memsom.parents_of(conn, d))
    with pytest.raises(Exception):
        memsom_stale.freshen(conn, d)
    after = sorted(r[0] for r in memsom.parents_of(conn, d))

    assert before == after, (
        f"freshen raised but had already mutated the DAG: parents {before} -> {after}")


@pytest.mark.xfail(strict=True, reason="MS-12: the bridge sweep tombstones the source without cascading")
def test_bridge_sweep_cascades_to_descendants(conn, tmp_path):
    """Deleting a memory file must cascade the tombstone to its derivatives.

    memsom/bridge/bridge_import.py:727 issues a bare
    `UPDATE nodes SET tombstoned = 1 ... WHERE id = ?` instead of calling
    memsom.revoke_cascade, so descendants are left live — and live derived nodes
    are exactly what distill/reflex export into training weights.
    """
    import memsom
    from memsom.bridge import bridge_import as bi
    from memsom.distill import distill as memsom_distill

    bi.migrate(conn)
    memsom_distill.migrate(conn)

    mem = tmp_path / "memdir"
    mem.mkdir()
    (mem / "reference_secret.md").write_text(
        "---\nname: s\ndescription: d\nsection: References\n---\n\n"
        "The operator password is correct-horse-battery-staple.\n", encoding="utf-8")
    (mem / "reference_keep.md").write_text(
        "---\nname: k\ndescription: d\nsection: References\n---\n\n"
        "An unrelated note.\n", encoding="utf-8")
    (mem / "MEMORY.md").write_text("# Memory\n\n## References\n", encoding="utf-8")

    bi.import_memory_dir(conn, mem, dry_run=False)
    src = conn.execute(
        "SELECT id FROM nodes WHERE source_ref='memory:reference_secret'"
    ).fetchone()[0]
    derived, _ = memsom.derive_node(
        conn, "- The operator password is correct-horse-battery-staple. "
              f"[mem:{src}|endorsed]", [src])

    (mem / "reference_secret.md").unlink()
    bi.import_memory_dir(conn, mem, dry_run=False)

    assert conn.execute("SELECT tombstoned FROM nodes WHERE id=?",
                        (src,)).fetchone()[0] == 1, "precondition: source was swept"

    dead = conn.execute("SELECT tombstoned FROM nodes WHERE id=?",
                        (derived,)).fetchone()[0]
    rows = memsom_distill.export_training(conn, min_integrity=1)
    leaking = [r for r in rows if "correct-horse-battery-staple" in r["output"]]

    assert dead, (
        f"bridge reconcile sweep tombstoned {src} without cascading to {derived}")
    assert not leaking, (
        f"{len(leaking)} training-export row(s) still quote the deleted memory")
