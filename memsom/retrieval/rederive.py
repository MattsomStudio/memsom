"""Content re-derivation layer.

A frozen `agent-derived` node stores the *output* of a producer (compose,
extractive_summary, ...) but nothing about *how* it was made.  This module
records that recipe at mint time so a derived node can later be REGENERATED
from its surviving live parents after a source is revoked or redacted —
minting a fresh node and retiring the stale one (archive, never mutate).

Design contract:
  - Deterministic producers (compose / extractive) replay byte-identically
    from their live parents + the stored recipe, so regeneration is cheap and
    needs no model.  These auto-regenerate (lazy on read, eager on redact).
  - Non-deterministic producers (llm) and legacy nodes ('unknown') are
    FLAGGED, never auto-rolled inside a cascade.  They regenerate only via an
    explicit operator call (`regenerate_llm`, later step) so the cascade stays
    deterministic and reproducible.

Library discipline: no prints, no sys.exit.  `record_recipe` runs inside the
caller's open derive transaction, so it must NOT migrate (DDL would commit and
break atomicity); `migrate(conn)` is run at startup by cli.migrate_all.
"""

import json

import memsom
from memsom.storage import schema as memsom_schema

# Must match extractive_summary's default so a recipe that never stored k
# still replays consistently.
_EXTRACTIVE_DEFAULT_K = 5


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Add the recipe side table idempotently.  Safe to call repeatedly.

    A side table (not columns on `nodes`) keeps the destructive-rebuild target
    pristine and lets the LLM recipe grow inside an opaque JSON blob without
    ever churning the schema.  node_id is INTEGER to match nodes.id
    (INTEGER PRIMARY KEY AUTOINCREMENT) — NOT the federation uuid (TEXT).
    """
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS derivation_recipe (
    node_id     INTEGER PRIMARY KEY REFERENCES nodes(id),
    engine      TEXT    NOT NULL,   -- 'compose'|'extractive'|'corroborate'|'llm'|'unknown'
    recipe_json TEXT,               -- compose: {"question": ...}; llm: +model/prompt/params; else NULL
    supersedes  INTEGER             -- the node_id this one replaced (NULL if original)
  );""")


# ---------------------------------------------------------------------------
# Recipe capture (called in the producer's own transaction, right after mint)
# ---------------------------------------------------------------------------

def record_recipe(conn, node_id, engine, supersedes=None, **recipe):
    """Record how *node_id* was produced.

    *engine* must reflect the branch that ACTUALLY ran: an --llm call that fell
    back to extractive on LlmUnavailable records 'extractive', not 'llm', so it
    regenerates deterministically instead of being parked as non-deterministic.

    *recipe* kwargs are serialised to recipe_json (e.g. question=..., or
    model/prompt/sampling_params for llm).  Empty -> NULL.
    """
    conn.execute(
        "INSERT OR REPLACE INTO derivation_recipe (node_id, engine, recipe_json, supersedes)"
        " VALUES (?,?,?,?)",
        (node_id, engine, json.dumps(recipe) if recipe else None, supersedes))


def get_recipe(conn, node_id):
    """Return {'engine', 'recipe', 'supersedes'} for *node_id*, or None if unrecorded.

    Legacy nodes minted before this layer existed have no row -> None, which
    every caller treats as 'unknown' (flag-stale, never auto-roll).
    """
    row = conn.execute(
        "SELECT engine, recipe_json, supersedes FROM derivation_recipe WHERE node_id = ?",
        (node_id,)).fetchone()
    if row is None:
        return None
    engine, recipe_json, supersedes = row
    return {
        "engine": engine,
        "recipe": json.loads(recipe_json) if recipe_json else {},
        "supersedes": supersedes,
    }


# ---------------------------------------------------------------------------
# Regeneration (deterministic engines only; llm/unknown -> regenerate_llm)
# ---------------------------------------------------------------------------

_DETERMINISTIC = ("compose", "extractive")


def _node_retire_state(conn, node_id):
    """(content, tombstoned, archived) for *node_id*, or None if it doesn't exist.

    archived read defensively (COALESCE / column-existence) so this works before
    compact.migrate has added the column.
    """
    if memsom_schema.column_exists(conn, "nodes", "archived"):
        arch = "COALESCE(archived, 0)"
    else:
        arch = "0"
    return conn.execute(
        f"SELECT content, tombstoned, {arch} FROM nodes WHERE id = ?", (node_id,)).fetchone()


def _live_parents(conn, node_id):
    """Parent rows (id, content, channel, label, source_ref) usable as sources for
    regeneration, in compose trust order (label DESC, id ASC).

    Excludes tombstoned / redacted / archived / quarantined parents — a redacted
    parent has content='' but tombstoned=0, so a tombstone-only filter would
    compose from an empty source. Taint columns are guarded by existence so this
    works regardless of which module migrations have run.
    """
    clauses = ["n.tombstoned = 0"]
    for col in ("redacted", "archived"):
        if memsom_schema.column_exists(conn, "nodes", col):
            clauses.append(f"COALESCE(n.{col}, 0) = 0")
    if memsom_schema.column_exists(conn, "nodes", "status"):
        clauses.append("(n.status IS NULL OR n.status != 'quarantined')")
    where = " AND ".join(clauses)
    return conn.execute(
        "SELECT n.id, n.content, n.channel, n.label, n.source_ref"
        " FROM edges e JOIN nodes n ON n.id = e.parent"
        f" WHERE e.child = ? AND {where}"
        " ORDER BY n.label DESC, n.id ASC", (node_id,)).fetchall()


def _replay(recipe, live_parents):
    """Re-run the recorded deterministic producer over *live_parents*.

    Returns the regenerated text, or None if the engine isn't deterministically
    replayable (missing question, unknown engine).
    """
    engine = recipe["engine"]
    kw = recipe["recipe"]
    if engine == "compose":
        question = kw.get("question")
        if not question:
            return None
        # compose wants (id, content, channel, label, source_ref) in trust order
        # (label DESC, id ASC) — parents_of already returns rows in that order.
        sources = [(p[0], p[1], p[2], p[3], p[4]) for p in live_parents]
        text, _used = memsom.compose(question, sources)
        return text
    if engine == "extractive":
        from memsom.lifecycle import compact as memsom_compact
        rows = sorted(((p[0], p[1]) for p in live_parents), key=lambda r: r[0])  # id asc
        return memsom_compact.extractive_summary(rows, kw.get("k", _EXTRACTIVE_DEFAULT_K))
    return None


def regenerate(conn, node_id):
    """Regenerate a stale deterministic summary from its live parents.

    Mints a fresh node (edges to LIVE parents only), archives the stale one, and
    chains old->new via derivation_recipe.supersedes. Returns the new node id, or
    None when nothing is done: non-deterministic/legacy engine, already retired,
    no live parents left, or content unchanged (this last case is what makes
    regenerate idempotent — re-running a deterministic producer over the same
    live inputs is byte-identical, so a no-op short-circuits).

    MUST be called OUTSIDE an open transaction: it self-heals dependent schema
    and derive_node manages its own write lock.
    """
    recipe = get_recipe(conn, node_id)
    if recipe is None or recipe["engine"] not in _DETERMINISTIC:
        return None  # llm / unknown / corroborate -> explicit regenerate_llm only

    # Ensure the retire (archived) and confidentiality columns exist. compact owns
    # archived; both are idempotent no-ops once migrated.
    from memsom.lifecycle import compact as memsom_compact
    from memsom.integrity import confid as memsom_confid
    memsom_compact.migrate(conn)
    memsom_confid.migrate(conn)

    state = _node_retire_state(conn, node_id)
    if state is None:
        return None
    old_content, tombstoned, archived = state
    if tombstoned or archived:
        return None  # already retired; nothing to do

    live = _live_parents(conn, node_id)
    if not live:
        return None  # all parents gone -> revoke already tombstoned it; retire-no-replace

    new_content = _replay(recipe, live)
    if new_content is None or new_content == old_content:
        return None  # unchanged -> no churn (idempotency for deterministic engines)

    live_ids = [p[0] for p in live]
    new_id, _label = memsom.derive_node(conn, new_content, live_ids)  # commits its own txn
    with conn:  # record recipe + retire old atomically (derive committed just above)
        record_recipe(conn, new_id, recipe["engine"], supersedes=node_id, **recipe["recipe"])
        conn.execute("UPDATE nodes SET archived = 1, archived_at = ? WHERE id = ?",
                     (memsom.now_iso(), node_id))

    # Bell-LaPadula high-water: a summary of SECRET parents must not mint at PUBLIC.
    memsom_confid.recompute_conf(conn, new_id)

    # Keep the retrieval index in lock-step: the archived old node must not surface.
    try:
        from memsom.retrieval import retrieve as memsom_retrieve
        memsom_retrieve.deindex_node(conn, node_id)
    except Exception:  # noqa: BLE001 — retrieval is optional
        pass

    return new_id


# ---------------------------------------------------------------------------
# Erasure (G5): declared-intent scrub across the WHOLE version lineage
# ---------------------------------------------------------------------------

def _version_chain(conn, node_id):
    """Every node id in *node_id*'s version lineage via derivation_recipe.supersedes.

    Walks BOTH directions transitively: predecessors (what this node superseded)
    and successors (what superseded this node). Superseded copies are SIBLINGS of
    their replacement (both derive from the same parents), not descendants — so
    redact_node's edge cascade can't reach them. This is the gap erase() closes.
    """
    seen = set()
    stack = [node_id]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        row = conn.execute(
            "SELECT supersedes FROM derivation_recipe WHERE node_id = ?", (n,)).fetchone()
        if row and row[0] is not None:
            stack.append(row[0])                 # predecessor it replaced
        for r in conn.execute(
                "SELECT node_id FROM derivation_recipe WHERE supersedes = ?", (n,)):
            stack.append(r[0])                   # successor that replaced it
    return seen


def erase(conn, node_id, reason, *, memory_dir=None, vault=None):
    """Erasure: declared-intent scrub of a secret/PII across a node's ENTIRE lineage.

    For *node_id* and every version in its supersedes chain, runs
    redact_node(cascade=True): destroys content of the node + all derived
    descendants (incl. archived copies, which keep their edges), de-indexes from
    retrieval, unlinks the backing file(s), and keeps edges so blame() still walks
    the shape. Idempotent — redact_node is first-write-wins, so overlapping
    cascades are safe.

    Walking the supersedes chain is what makes this the fix for the blame
    edit-history leak: a superseded (edited-over) prior version retains its full
    old content, and only redacting the WHOLE chain removes it from blame.

    *memory_dir* / *vault* are forwarded to redact_node so the on-disk purge
    targets the caller's roots (e.g. a --memory-dir override) instead of only the
    live locations.

    This is a DISTINCT verb, not an inferred reason: normal revoke/regenerate
    preserve archived content; erase is the deliberate destructive path. MUST be
    called outside an open transaction (redact_node manages its own).

    Returns the sorted list of node ids whose content was destroyed.
    """
    from memsom.integrity import redact as memsom_redact
    migrate(conn)                         # derivation_recipe must exist for the chain walk
    erased = set()
    for target in _version_chain(conn, node_id):
        erased.update(memsom_redact.redact_node(
            conn, target, reason, cascade=True, memory_dir=memory_dir, vault=vault))
    return sorted(erased)
