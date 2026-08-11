#!/usr/bin/env python3
"""memsom -- derivation-DAG memory store, explain/revoke vertical slice.

Every memory is a node; edges mean CAME-FROM (provenance), not relates-to.
Invariants (locked -- see Vault Security/Teachings/2026-06-10-derivation-dag-product-reframe.md):
  1. Integrity labels are assigned by CHANNEL, never by content:
     endorsed(3) > user(2) > agent-derived(1) > external(0).
  2. A derived node's label = min(parent labels) -- Biba low-water-mark, one hop.
  3. History is immutable: changes mint NEW nodes; rows are never edited or deleted.
  4. revoke = tombstone + cascade to all transitive descendants. Rows, edges and
     payloads all survive (redaction is a separate, out-of-scope mode); liveness
     is filtered at READ time via WHERE tombstoned=0.
  5. ask refuses to compose from zero live sources -- no unprovenanced answers.

CLI: seed [--offline] [--reset] -- ask "question" -- explain <id> --
     revoke <id> [--reason ...] [--yes]  (dry-run by default) -- dump
DB:  ~/.memdag/memdag.db (override the file with MEMDAG_DB, or the dir with
     MEMDAG_HOME). Deliberately a user-data dir, NOT beside this module --
     site-packages is wiped on venv upgrade/reinstall. Keep it OUT of any synced
     or backup trees so private memories are not replicated.

PHASE 2 (PLAN.md): this module is now a FACADE. The frozen-core logic that
used to live here (text/compose kernel, the SQLite connection, the DAG store
primitives, the demo CLI) moved to kernel/, storage/, effects/ and
integrity/dag.py; the CLI commands moved into interface/cli.py as the
frozen_* functions. Every public symbol this module exported before the
split is re-exported here unchanged, so `import memsom; memsom.X` keeps
working for memsom-panel and every in-tree caller -- logic fan-in target 0
(scripts/fanin.py memsom/__init__.py).

integrity.dag and interface.cli are re-exported LAZILY (via __getattr__ ->
resolve_facade_attr, PEP 562, importlib underneath) rather than imported at
module level: this package sits at the root of every `import memsom`, so a
static import of integrity or (worse) the top interface layer here would put
that layer on every other layer's transitive closure and break the layers
goal for the whole package, not just this file. kernel/storage/effects stay
eager imports -- they are the untracked/bottom layers, so no caller inherits
a cross-layer edge by reaching them through the facade.
"""

from memsom.kernel.lattice import RANK, NAME

from memsom.kernel.text import (
    STOP, STEM_WIDTH, stems, prose_lines, strip_furniture, snippet,
    candidate_sentences, now_iso, local_date, fmt_node,
)
from memsom.kernel.compose import compose

from memsom.storage.db import db_path, get_connection, SCHEMA
from memsom.storage.db import resolve_facade_attr as __getattr__

from memsom.effects.net import fetch_external, EXT_URL, FALLBACK, HOME
