"""GATE for MS-04 (CRITICAL) -- a re-ingested source's prior version survives
redact + tombstone --hard.

WRITTEN in Phase 1 (SECURITY-REMEDIATION.md §3.2, §2 item 8): no gate existed
for this finding before Phase 1. PoC reference (pentest folder, not in this
tree): `poc/red_source_supersede_lineage.py`.

CONTROL-TESTED reasoning against memsom @ 9d165b1 (pre-fix): `_version_chain`
walked ONLY `derivation_recipe.supersedes`, which by that table's own
docstring covers DERIVED nodes. A re-ingested SOURCE has no recipe at all, so
its prior version was invisible to `erase()`'s lineage walk. Re-ingesting the
same `source_ref` marks the OLD version stale (`mark_stale_cascade`) but NOT
tombstoned -- staleness is deliberately not a liveness dimension -- so once
the operator redacts/tombstones the CURRENT (head) version, the stale-but-
still-"live" old version becomes the only `tombstoned=0` node left for that
`source_ref`, and its plaintext survives untouched.

WHAT A FIX LOOKS LIKE
---------------------
  * `rederive._version_chain` walks `source_supersedes` (integrity/stale.py)
    in addition to `derivation_recipe.supersedes`, so `erase()` -- and
    therefore `tombstone --hard` -- reaches the old version too.
"""

import memsom
from memsom.lifecycle import stale as memsom_stale
from memsom.interface import ingest as memsom_ingest
from memsom.lifecycle import rederive as memsom_rederive

MARKER = "PASSPHRASE-9f3a2b"
OLD_SECRET = f"The production API key is {MARKER}-old, rotate before shipping."
NEW_SAFE = "The production API key was rotated; see the vault for the current value."


def _reingest_scenario(conn):
    """Old (secret) version, then a re-ingest of the SAME source_ref with safe
    content -- the shape every `ingest`/`ingest-dir`/bridge re-sync produces."""
    ref = "vault:prod-api-key"
    old_ids = memsom_ingest.ingest_text(conn, OLD_SECRET, "user", source_ref=ref)
    assert len(old_ids) == 1
    old_id = old_ids[0]
    conn.commit()

    new_ids = memsom_ingest.ingest_text(conn, NEW_SAFE, "user", source_ref=ref)
    assert len(new_ids) == 1
    new_id = new_ids[0]
    conn.commit()

    # Precondition: the re-ingest DID record a source_supersedes link and mark
    # the old version stale, but the old version is still nominally LIVE
    # (tombstoned=0) -- that's the mechanism that makes this dangerous.
    assert memsom_stale.superseding_version(conn, old_id) == new_id, (
        "precondition: supersession was not recorded")
    old_row = conn.execute(
        "SELECT stale, tombstoned FROM nodes WHERE id=?", (old_id,)).fetchone()
    assert old_row == (1, 0), f"precondition: old version stale-but-live, got {old_row}"
    return old_id, new_id, ref


def test_hard_delete_of_the_head_reaches_the_superseded_source(conn):
    """redact + tombstone --hard targets the HEAD (what the operator sees);
    the prior version must not be the only live node left carrying the secret."""
    old_id, new_id, _ref = _reingest_scenario(conn)

    # "redact + tombstone --hard" on the current/head version:
    memsom.revoke_cascade(conn, new_id, "gone")
    conn.commit()
    memsom_rederive.erase(conn, new_id, "hard delete")

    old_row = conn.execute(
        "SELECT tombstoned, redacted, content FROM nodes WHERE id=?",
        (old_id,)).fetchone()
    tombstoned, redacted, content = old_row
    assert redacted == 1 and MARKER not in (content or ""), (
        f"the superseded prior version survives with its secret intact after "
        f"redact+tombstone --hard on the head: tombstoned={tombstoned}, "
        f"redacted={redacted}, content={content!r}")


def test_no_live_node_for_the_source_ref_carries_the_old_plaintext(conn):
    """The literal finding assertion: after the hard delete, no LIVE node for
    this source_ref carries the MARKER."""
    old_id, new_id, ref = _reingest_scenario(conn)
    memsom.revoke_cascade(conn, new_id, "gone")
    conn.commit()
    memsom_rederive.erase(conn, new_id, "hard delete")

    rows = conn.execute(
        "SELECT id, content FROM nodes WHERE source_ref=? AND tombstoned=0",
        (ref,)).fetchall()
    for nid, content in rows:
        assert MARKER not in (content or ""), (
            f"live node [{nid}] for source_ref={ref!r} still carries the secret "
            f"after redact+tombstone --hard")


def test_control_erase_still_reaches_a_derived_chain(conn):
    """GREEN and must stay green: the derivation_recipe.supersedes half of
    _version_chain (the pre-existing, already-working half) is unaffected by
    adding the source_supersedes walk."""
    src = memsom.insert_node(conn, OLD_SECRET, "user")
    conn.commit()
    child, _ = memsom.derive_node(conn, f"- {OLD_SECRET}", [src])
    memsom_rederive.record_recipe(conn, child, "compose", question="q")
    conn.commit()

    chain = memsom_rederive._version_chain(conn, child)
    assert chain == {child}, f"unexpected lineage for a node with no supersedes: {chain}"
