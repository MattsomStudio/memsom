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
goal for the whole package, not just this file. kernel/storage/effects are
ALSO lazy (review fix MF-3, 2026-09-01): a bare `import memsom` -- e.g. the
panel's `from memsom.childenv import child_env` spawn primitive -- must not
drag in a single feature module, so `__init__.py` imports nothing but stdlib
at module level; every public symbol resolves through `__getattr__` below.
"""

import importlib as _importlib

_FACADE = {
    "RANK": "memsom.kernel.lattice", "NAME": "memsom.kernel.lattice",
    "STOP": "memsom.kernel.text", "STEM_WIDTH": "memsom.kernel.text",
    "stems": "memsom.kernel.text", "prose_lines": "memsom.kernel.text",
    "strip_furniture": "memsom.kernel.text", "snippet": "memsom.kernel.text",
    "candidate_sentences": "memsom.kernel.text", "now_iso": "memsom.kernel.text",
    "local_date": "memsom.kernel.text", "fmt_node": "memsom.kernel.text",
    "compose": "memsom.kernel.compose",
    "db_path": "memsom.storage.db", "get_connection": "memsom.storage.db",
    "SCHEMA": "memsom.storage.db",
    "fetch_external": "memsom.effects.net", "EXT_URL": "memsom.effects.net",
    "FALLBACK": "memsom.effects.net", "HOME": "memsom.effects.net",
}


def __getattr__(name):
    mod = _FACADE.get(name)
    if mod is not None:
        return getattr(_importlib.import_module(mod), name)
    from memsom.storage.db import resolve_facade_attr
    return resolve_facade_attr(name)
