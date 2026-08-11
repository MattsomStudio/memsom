"""GATE for MS-33 -- one refused note must not abort the whole sync_vault batch.

`ingest.py`'s `enforce_source_ref_namespace` raises ValueError for a
source_ref claiming the bridge's reserved `memory:` namespace, and pass 1 of
`sync_vault` had no `except ValueError` around the `ingest_text` call --
unlike the two guards directly above it (oversized / unreadable), which
`say()` and continue. One poisoned note therefore aborted the entire call:
every note sorting after it in the walk never got a look, AND the prune pass
(which runs after pass 1) never ran either, so deleted notes stayed live.

CONTROL-TESTED: on the pre-fix tree this raises ValueError out of sync_vault
entirely; nothing in `summary` is ever returned.
"""

from pathlib import Path

import pytest

import memsom
from memsom.bridge import obsidian as memsom_obsidian


def test_one_refused_note_does_not_abort_the_batch(conn, vault, monkeypatch):
    good_a = vault / "a.md"
    good_a.write_text("---\n---\n\nfirst good note\n", encoding="utf-8")
    poison = vault / "poison.md"
    poison.write_text("---\n---\n\na poisoned note\n", encoding="utf-8")
    good_b = vault / "b.md"
    good_b.write_text("---\n---\n\nsecond good note, sorts after poison\n",
                      encoding="utf-8")

    real_walk = memsom_obsidian._walk_markdown

    def poisoned_walk(vault_arg):
        for rel, ap, mtime_ns, size in real_walk(vault_arg):
            if ap.name == "poison.md":
                # Same file on disk, but a source_ref claiming the bridge's
                # own reserved namespace -- the real, deliberately reachable
                # MS-33 trigger, without needing a literal ':' in a Windows
                # path.
                yield ("memory:poison.md", ap, mtime_ns, size)
            else:
                yield (rel, ap, mtime_ns, size)

    monkeypatch.setattr(memsom_obsidian, "_walk_markdown", poisoned_walk)

    summary = memsom_obsidian.sync_vault(conn, vault, default_channel="user")

    assert summary["refused"] == 1, f"the poisoned note was not counted: {summary}"
    assert summary["ingested"] == 2, (
        f"notes on either side of the poisoned one did not sync: {summary}")


@pytest.fixture()
def vault(tmp_path):
    p = tmp_path / "vault"
    p.mkdir()
    return p
