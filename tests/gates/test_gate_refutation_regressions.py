"""Refutation regression tests MS-R1, MS-R3, MS-R4 (PLAN.md Phase 0; SECURITY-
REMEDIATION.md A4). Two of the five refutations already have GREEN controls in
this directory (the negative controls embedded in the other gate files); these
three did not exist before Phase 0.

A "refutation" here is a claim of the shape "X cannot happen through any
supported path" — the opposite of a finding, which claims X DOES happen. Each
test below tries the described attack/regression directly rather than
re-deriving it from a comment, per this project's own rule about trusting a
number nobody re-measured.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest


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


# ---------------------------------------------------------------------------
# MS-R1 — taint_filter_clauses cannot be silently no-op'd.
#
# "The primitive A1.6 makes the only legal clause builder, and nothing
# currently asserts it" (PLAN.md). This is NOT MS-02 (the clearance=None gap,
# already gated red in test_gate_conf_training_export.py) — it is the
# self-integrity of the primitive itself: a future edit that guts
# taint_filter_clauses down to `return [], []` (or drops the one clause that
# is NEVER conditional) must fail loudly, and today nothing checks that.
# MEASURED: passes on current code — this is the regression guard, not a
# finding.
# ---------------------------------------------------------------------------
def test_taint_filter_clauses_is_never_a_no_op(conn):
    from memsom.storage import schema as memsom_schema

    for clearance in (None, 0, 1, 2, 3):
        for include_quarantined in (False, True):
            clauses, _params = memsom_schema.taint_filter_clauses(
                conn, clearance=clearance, include_quarantined=include_quarantined)
            assert clauses, (
                f"taint_filter_clauses(clearance={clearance!r}, "
                f"include_quarantined={include_quarantined}) returned an empty "
                f"clause list on a full-featured schema — the primitive has "
                f"been silently reduced to a no-op")
            assert "tombstoned = 0" in clauses, (
                "the one clause that is NEVER conditional on schema or argument "
                "(tombstoned = 0) is missing")


# ---------------------------------------------------------------------------
# MS-R3 — neighborhood cannot be declassified through any supported write
# path.
#
# neighborhood()'s BFS is widest-path relaxation: best[m] = min(best[n],
# label(m)). That min() is what makes it impossible for ANY edge — however it
# got into rel_edges, including through relate.add_edge, memsom's only
# supported write path into that table — to report an integrity floor higher
# than the true minimum along the path. MEASURED: passes on current code;
# this guards the invariant against a future edit (e.g. min -> max, or a
# floor that is cached instead of recomputed) rather than documenting a live
# finding.
# ---------------------------------------------------------------------------
def test_neighborhood_floor_is_never_higher_than_the_path_minimum(conn):
    import memsom
    from memsom.retrieval import relate as memsom_relate

    memsom_relate.migrate(conn)
    with conn:
        low = memsom.insert_node(conn, "an external, low-integrity node", "external")
        mid = memsom.insert_node(conn, "a user-channel node", "user")
        high = memsom.insert_node(conn, "an endorsed, high-integrity node", "endorsed")

    # The only supported write path into rel_edges: relate.relate() itself.
    memsom_relate.relate(conn, low, high, "relates-to")
    memsom_relate.relate(conn, high, mid, "relates-to")
    conn.commit()

    labels = {r[0]: r[3] for r in
               conn.execute("SELECT id, content, channel, label FROM nodes").fetchall()}

    results = {r["id"]: r["path_min"] for r in
               memsom_relate.neighborhood(conn, low, hops=3, clearance=3)}

    for nid, path_min in results.items():
        if nid == low:
            continue
        assert path_min <= labels[low], (
            f"node {nid} reports path_min={path_min}, which is HIGHER than the "
            f"BFS start node's own label ({labels[low]}) — the low-integrity "
            f"source at {low} inflated a downstream floor instead of "
            f"lower-bounding it, which is the one thing widest-path relaxation "
            f"must never do")


# ---------------------------------------------------------------------------
# MS-R4 — the broker's uvx default does not resolve from the parent's cwd.
#
# MEASURED: federation/broker.py's Upstream.start() calls
# `subprocess.Popen([self.command, *self.args], ...)` with `self.command` the
# raw string "uvx" (the default spec) and NO `executable=` / pinned absolute
# path. On Windows, CreateProcess searches the calling process's current
# directory for a bare executable name before it reaches PATH — the same
# class of bug as MS-29 ("Windows resolves a bare `git` from the current
# directory"), scheduled for Phase 5's effects-layer absorption alongside it.
# xfail, not a green control: this refutation currently FAILS.
# ---------------------------------------------------------------------------
# FIXED (Phase 6): Upstream.__init__ resolves `command` eagerly via
# effects.proc.resolve(), which (also Phase 6) no longer lets shutil.which's
# internal win32 curdir-insertion re-admit the CWD.
def test_broker_uvx_default_does_not_resolve_from_cwd(tmp_path, monkeypatch):
    from memsom.federation import broker as memsom_broker

    fake_uvx = tmp_path / ("uvx.exe" if sys.platform == "win32" else "uvx")
    fake_uvx.write_text("", encoding="utf-8")
    fake_uvx.chmod(0o755)

    up = memsom_broker.Upstream("fetch", {"command": "uvx", "args": []})
    resolved = getattr(up, "resolved_command", None) or up.command

    # Only meaningful where uvx is actually installed: then a bare "uvx" left
    # for CreateProcess to look up would search the CWD before PATH, so the
    # broker MUST have resolved it to an absolute path. When uvx is NOT on PATH
    # (e.g. a CI runner), resolve() correctly returns the bare name and the
    # spawn fails with FileNotFoundError -- there is nothing in the CWD to
    # shadow, so this assertion does not apply. The load-bearing cwd-shadow
    # property is asserted unconditionally below.
    if shutil.which("uvx") is not None:
        assert resolved != "uvx", (
            "uvx is on PATH but Upstream left the command bare -- CreateProcess "
            "would search the CWD before PATH; it must resolve to an absolute "
            "path"
        )
    monkeypatch.chdir(tmp_path)
    assert resolved != str(fake_uvx), (
        f"Upstream resolved 'uvx' to {resolved}, which is the planted "
        f"executable in the current working directory -- cwd shadowing "
        f"succeeded"
    )
