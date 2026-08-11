"""GATE for MS-19 -- dedup + path stamping defeats supersession.

CONTROL-TESTED reasoning against memsom @ 9d165b1: `sync_vault`'s ingest pass
calls `ingest_text`, whose content-hash dedup can hand back a LIVE node id
that already belongs to a DIFFERENT note (the two files' bodies happen to
normalize to the same bytes). `sync_vault` then unconditionally stamped
`obsidian_path` onto every returned id, so the duplicate note's sync silently
overwrote the trusted note's path column. Supersession, the prune pass, and
the reconcile sweep all key on `obsidian_path`, so a later edit/retraction of
the ORIGINAL note resolved against the wrong node -- or nothing at all.

WRITTEN in Phase 4 (SECURITY-REMEDIATION.md Sec3.2): no gate existed for this
finding before Phase 4.

WHAT A FIX LOOKS LIKE
----------------------
`sync_vault` refuses (loudly, via `integrity.ingest.enforce_no_path_steal`) to
re-stamp `obsidian_path` onto a node that already carries a DIFFERENT path --
first path wins, and the collision is visible in the sync log rather than
silently overwriting a trusted note's identity.
"""

import memsom
from memsom.bridge import obsidian as memsom_obsidian
from memsom.integrity import ingest as memsom_ingest


def test_duplicate_note_cannot_steal_a_trusted_notes_path(conn, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    body = "The operator password is correct-horse-battery-staple.\n"

    trusted = vault / "trusted.md"
    trusted.write_text(body, encoding="utf-8")
    memsom_obsidian.sync_vault(conn, vault, default_channel="user")

    trusted_node = memsom_obsidian._live_nodes_for_path(conn, "trusted.md")
    assert trusted_node, "precondition: the trusted note was ingested"
    trusted_id = trusted_node[0]

    # A second note, different filename, byte-identical body -> ingest_text's
    # dedup will hand back the SAME node id trusted.md already owns.
    duplicate = vault / "duplicate.md"
    duplicate.write_text(body, encoding="utf-8")
    memsom_obsidian.sync_vault(conn, vault, default_channel="user")

    row = conn.execute(
        "SELECT obsidian_path FROM nodes WHERE id = ?", (trusted_id,)).fetchone()
    assert row[0] == "trusted.md", (
        f"the duplicate note's sync overwrote trusted.md's identity: "
        f"obsidian_path is now {row[0]!r}")

    # Retracting the ORIGINAL note must still resolve against the ORIGINAL
    # node, not silently no-op because duplicate.md stole its path.
    trusted.unlink()
    memsom_obsidian.sync_vault(conn, vault, default_channel="user")
    dead = memsom.get_node(conn, trusted_id)["tombstoned"]
    assert dead, "deleting trusted.md must revoke the node it actually owns"


def test_enforce_no_path_steal_refuses_a_foreign_path(conn):
    """Unit-level control on the primitive itself."""
    nid = memsom.insert_node(conn, "some content", "user")
    conn.execute("UPDATE nodes SET obsidian_path = 'a.md' WHERE id = ?", (nid,))
    conn.commit()
    # Same path -> no-op, no raise.
    memsom_ingest.enforce_no_path_steal(conn, nid, "obsidian_path", "a.md")
    # Different path -> refused.
    try:
        memsom_ingest.enforce_no_path_steal(conn, nid, "obsidian_path", "b.md")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a foreign path")
