"""GATE for CORE-05 / CORE-06 — one taint primitive must feed every read pool,
and every derive path must propagate the confidentiality ceiling.

Invariants under test:
    ARCHITECTURE.md:109  "One taint primitive (taint_filter_clauses) feeds every
                          read pool."
    ARCHITECTURE.md:112  "Integrity min-floor + confidentiality max-ceiling."
    ARCHITECTURE.md:103  the answer node gets "conf=max(used)".

CONTROL TEST (run 2026-07-31 against 9d165b1) — measured results:

    test_live_sources_matches_the_taint_primitive   FAILED  (leaked=[2])
    test_derive_node_propagates_conf_ceiling        FAILED  (child=0, parents=2)
    test_alternate_source_pools_apply_every_dimension FAILED (2 helpers short 3 dims)
    test_enhanced_ask_pool_is_clean                 PASSED  (negative control)

The first three are RED-xfail gates. The fourth PASSES on current code and is
included deliberately as a discrimination control: it proves the test harness
can tell a clean pool from a dirty one, so the three failures above are a
property of the code and not of the gate.

RUN:
    pytest gates/test_gate_one_taint_primitive.py -v
"""
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


def _public_and_secret(conn):
    """One PUBLIC source and one SECRET source, both live, both non-derived."""
    import memsom
    from memsom.integrity import confid as memsom_confid
    memsom_confid.migrate(conn)
    with conn:
        pub = memsom.insert_node(
            conn, "Nebula uses UDP hole punching to connect peers directly.",
            "user", source_ref="notes/nebula.md")
        sec = memsom.insert_node(
            conn, "The CA signing key passphrase is hunter2-topsecret.",
            "user", source_ref="notes/ca.md")
    memsom_confid.classify(conn, sec, "secret")
    return pub, sec


def _primitive_pool(conn, clearance):
    from memsom.storage import schema as memsom_schema
    clauses, params = memsom_schema.taint_filter_clauses(conn, clearance=clearance)
    return {r[0] for r in conn.execute(
        "SELECT id FROM nodes WHERE channel != 'agent-derived' AND "
        + " AND ".join(clauses), params).fetchall()}


@pytest.mark.xfail(strict=True, reason="CORE-04: memsom/__init__.py:145 live_sources filters only tombstoned + channel")
def test_live_sources_matches_the_taint_primitive(conn):
    """memsom.live_sources (memsom/__init__.py:145) is a read pool.

    It is THE pool for the frozen-core `python -m memsom ask` (memsom/__init__.py
    :309), and it filters only `tombstoned = 0 AND channel != 'agent-derived'` —
    no redacted, no archived, no status, no conf_label. It must not return a node
    the shared primitive would have excluded.
    """
    import memsom
    _pub, sec = _public_and_secret(conn)

    allowed = _primitive_pool(conn, clearance=0)   # public clearance
    actual = {r[0] for r in memsom.live_sources(conn)}
    leaked = actual - allowed

    assert not leaked, (
        f"live_sources() returned {sorted(leaked)}, which taint_filter_clauses "
        f"excludes (secret node {sec} is above a public clearance)")


@pytest.mark.xfail(strict=True, reason="MS-10: derive_node never writes conf_label; the frozen-core ask path never calls recompute_conf")
def test_derive_node_propagates_conf_ceiling(conn):
    """Bell-LaPadula high-water: a child must be at least as secret as its
    most-secret parent. The frozen derive_node sets `label=min(parents)` but
    nothing for conf_label, so the child defaults to 0 = PUBLIC.
    """
    import memsom
    _pub, _sec = _public_and_secret(conn)

    text, used = memsom.compose("What is the CA passphrase?",
                                memsom.live_sources(conn))
    nid, _label = memsom.derive_node(conn, text, used)

    child = conn.execute(
        "SELECT conf_label FROM nodes WHERE id=?", (nid,)).fetchone()[0]
    ceiling = conn.execute(
        "SELECT MAX(n.conf_label) FROM nodes n JOIN edges e ON e.parent = n.id"
        " WHERE e.child = ?", (nid,)).fetchone()[0]

    assert "hunter2-topsecret" in text, "precondition: the secret was quoted"
    assert child >= ceiling, (
        f"derive_node stored conf_label={child} on a node quoting SECRET "
        f"content whose parent ceiling is {ceiling} — a declassification")


@pytest.mark.xfail(strict=True, reason="MS-08: alternate source pools re-implement a partial taint filter")
def test_alternate_source_pools_apply_every_dimension(conn):
    """redact.live_unredacted_sources and quarantine.live_unquarantined_sources
    are both documented as source-pool helpers for compose/ask — redact.py:243
    calls its one "the safe source-pool helper for compose/ask" — but each
    hand-rolls a partial filter. Neither has a production caller today, which is
    precisely why they are a trap: the next caller inherits a partial pool.
    """
    from memsom.integrity import redact as memsom_redact
    from memsom.integrity import quarantine as memsom_quarantine
    _pub, sec = _public_and_secret(conn)

    allowed = _primitive_pool(conn, clearance=0)
    offenders = {}
    for name, fn in (("redact.live_unredacted_sources",
                      memsom_redact.live_unredacted_sources),
                     ("quarantine.live_unquarantined_sources",
                      memsom_quarantine.live_unquarantined_sources)):
        leaked = {r[0] for r in fn(conn)} - allowed
        if leaked:
            offenders[name] = sorted(leaked)

    assert not offenders, (
        f"alternate source pools return nodes the primitive excludes: {offenders}")


def test_enhanced_ask_pool_is_clean(conn):
    """NEGATIVE CONTROL — passes on current code.

    interface/cli.py:175 `_build_pool` DOES route through the primitive. If this
    ever fails, the harness itself is wrong and the three failures above cannot
    be trusted.
    """
    from memsom.interface import cli as memsom_cli
    _pub, sec = _public_and_secret(conn)

    pool = {r[0] for r in memsom_cli._build_pool(conn, "public")}
    assert sec not in pool, "the enhanced pool leaked a secret node"
    assert pool == _primitive_pool(conn, clearance=0)
