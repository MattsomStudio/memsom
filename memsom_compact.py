"""memsom_compact — CONSOLIDATION ENGINE (compression).

DISTINCT from memsom_quarantine.consolidate (the integrity GATE).
CLI name: compact (NOT consolidate).

Compacts related live episodes into a single semantic (agent-derived) node,
preserving every edge so explain/blame still walk archived parents.

Schema additions (additive only; nodes.status CHECK untouched):
  nodes.archived     INTEGER NOT NULL DEFAULT 0
  nodes.archived_at  TEXT

Public API
----------
migrate(conn)
compact(conn, group_by="similarity", llm=False, min_group=2, k=5,
        sim_threshold=0.5) -> list[int]
list_archived(conn) -> list[dict]
extractive_summary(rows, k=5) -> str

CLI
---
compact  [--group-by similarity|claim] [--llm] [--min-group N]
archived-list
register(subparsers)
main(argv=None)
"""

import argparse
import json
import sqlite3
import sys
import urllib.error
import urllib.request

import memsom
import memsom_schema
import memsom_rederive
import memsom_quarantine
import memsom_llm
import memsom_confid


# ---------------------------------------------------------------------------
# Migration (additive only)
# ---------------------------------------------------------------------------

def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent: add archived + archived_at to nodes. DEFAULT 0 = not archived."""
    memsom_schema.add_column(conn, "nodes", "archived",
                             "INTEGER NOT NULL DEFAULT 0")
    memsom_schema.add_column(conn, "nodes", "archived_at", "TEXT")
    memsom_rederive.migrate(conn)  # record_recipe writes to derivation_recipe at mint


# ---------------------------------------------------------------------------
# Token helper
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set:
    """Return a set of stems for *text*.

    Tries memsom_retrieve.tokenize first (more accurate).
    Falls back to memsom.stems on any import error.
    """
    try:
        import memsom_retrieve
        return set(memsom_retrieve.tokenize(text))
    except Exception:
        return memsom.stems(text)  # import failed or tokenize raised — use the simpler stemmer


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _group_similarity(rows, min_group: int, sim_threshold: float) -> list:
    """Deterministic union-find single-link clustering.

    Parameters
    ----------
    rows : list of (id, content, channel, label)
    min_group : int  — minimum cluster size to keep
    sim_threshold : float — Jaccard threshold to union two nodes

    Returns list of lists of int (member ids), each list sorted ascending,
    only clusters of size >= min_group, ordered by min id.
    """
    if not rows:
        return []

    # Pre-compute token sets
    tok = {r[0]: _tokens(r[1]) for r in rows}
    ids = [r[0] for r in rows]

    # Union-Find
    parent = {nid: nid for nid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # deterministic tie-break: smaller id becomes root
            if px < py:
                parent[py] = px
            else:
                parent[px] = py

    # Single-link: union all pairs i<j with Jaccard >= threshold
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            ta, tb = tok[a], tok[b]
            union_size = len(ta | tb)
            if union_size == 0:
                jaccard = 0.0
            else:
                jaccard = len(ta & tb) / union_size
            if jaccard >= sim_threshold:
                union(a, b)

    # Collect components
    from collections import defaultdict
    components = defaultdict(list)
    for nid in ids:
        components[find(nid)].append(nid)

    # Keep only groups >= min_group; sort members ascending; order by min id
    groups = [sorted(members) for members in components.values()
              if len(members) >= min_group]
    groups.sort(key=lambda g: g[0])
    return groups


def _group_by_claim(conn: sqlite3.Connection) -> list:
    """Group nodes by shared corroboration claim.

    Returns [] gracefully if claim_assertions table does not exist.
    Groups are ordered by claim_id (ascending). Overlap is handled by a
    consumed set in the caller, not here — each group is returned as-is.

    Returns list of lists of int (node ids, ascending), ordered by claim_id.
    """
    if not memsom_schema.table_exists(conn, "claim_assertions"):
        return []

    rows = conn.execute(
        "SELECT ca.claim_id, ca.node_id"
        " FROM claim_assertions ca"
        " JOIN nodes n ON n.id = ca.node_id"
        " WHERE n.tombstoned = 0 AND n.archived = 0"
        "   AND n.channel != 'agent-derived' AND n.status != 'quarantined'"
        " ORDER BY ca.claim_id, ca.node_id"
    ).fetchall()

    if not rows:
        return []

    from collections import defaultdict
    by_claim = defaultdict(list)
    for claim_id, node_id in rows:
        if node_id not in by_claim[claim_id]:
            by_claim[claim_id].append(node_id)

    # Return in claim_id order; members already sorted ascending (ORDER BY)
    ordered_claim_ids = sorted(by_claim.keys())
    return [sorted(by_claim[cid]) for cid in ordered_claim_ids]


# ---------------------------------------------------------------------------
# Extractive summary (deterministic)
# ---------------------------------------------------------------------------

def extractive_summary(rows, k: int = 5) -> str:
    """Pure deterministic extractive summary.

    Parameters
    ----------
    rows : list of (id, content) sorted by id ascending.
    k    : max sentences to include.

    Returns a multi-line string starting with
    "Consolidated from N episodes:".

    Determinism guarantees:
    - No clock, no randomness, no dict-iteration order dependence.
    - Full tie-breakers: (episode_id, position_within_episode).
    - Same input → byte-identical output.
    """
    from collections import Counter

    cands = []
    seen = set()
    for nid, content in rows:
        sents = memsom.candidate_sentences(content)
        if not sents:
            first = next((l.strip() for l in content.splitlines() if l.strip()), "")
            if first:
                sents = [first[:200].rstrip(".:") + "."]
        for pos, s in enumerate(sents):
            key = " ".join(s.lower().split())
            if key in seen:
                continue
            seen.add(key)
            cands.append((nid, pos, s))

    # Stems per episode
    ep_stems = [memsom.stems(c) for _, c in rows]
    freq = Counter(t for st in ep_stems for t in st)
    # Terms shared by >= 2 episodes
    common = {t for t, c in freq.items() if c >= 2}

    # Score: primary = -overlap with common terms (descending), then (episode_id, pos)
    scored = sorted(
        cands,
        key=lambda t: (-len(common & memsom.stems(t[2])), t[0], t[1])
    )
    # Pick top-k, then re-sort into stable doc order (episode_id, pos)
    picked = sorted(scored[:k], key=lambda t: (t[0], t[1]))

    return (
        f"Consolidated from {len(rows)} episodes:\n"
        + "\n".join("- " + s for _, _, s in picked)
    )


# ---------------------------------------------------------------------------
# LLM summary (opt-in, graceful degrade)
# ---------------------------------------------------------------------------

def _llm_summarize(rows, model=None, base_url=None, timeout=60) -> str:
    """Summarize *rows* using Ollama.

    Raises memsom_llm.LlmUnavailable on ANY failure so callers can fall back.
    """
    m, base = memsom_llm.resolve(model, base_url)
    prompt = (
        "Summarize the following notes into at most 5 short factual bullet points."
        " Output ONLY the bullets.\n\n"
        + "\n\n".join(c for _, c in rows)
    )
    payload = json.dumps(memsom_llm._with_keep_alive({
        "model": m,
        "prompt": prompt,
        "stream": False,
    })).encode("utf-8")  # keep_alive stamped only when the env knob is set
    req = urllib.request.Request(
        base, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        answer = json.loads(raw)["response"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            TimeoutError, json.JSONDecodeError, KeyError) as err:
        raise memsom_llm.LlmUnavailable(
            f"ollama unreachable or malformed reply: {err}"
        ) from err

    answer = memsom_llm._THINK_RE.sub("", answer).strip()
    if not answer:
        raise memsom_llm.LlmUnavailable("empty LLM summary")

    return f"Consolidated from {len(rows)} episodes (llm):\n{answer}"


# ---------------------------------------------------------------------------
# Core compact()
# ---------------------------------------------------------------------------

def compact(
    conn: sqlite3.Connection,
    group_by: str = "similarity",
    llm: bool = False,
    min_group: int = 2,
    k: int = 5,
    sim_threshold: float = 0.5,
) -> list:
    """Consolidate related live episodes into semantic nodes.

    Parameters
    ----------
    conn        : open SQLite connection
    group_by    : "similarity" (Jaccard token overlap) or "claim" (corroboration claims)
    llm         : use Ollama for summary; fall back to extractive on failure
    min_group   : minimum episodes per group to trigger compaction
    k           : sentences for extractive summary
    sim_threshold : Jaccard threshold for "similarity" grouping

    Returns list of int (minted semantic node ids).

    Invariants
    ----------
    - Semantic node is derived via memsom.derive_node() — edges to EVERY episode,
      label = min(parents) — no trust laundering.
    - Episodes are ARCHIVED (archived=1), NEVER deleted; tombstoned untouched.
    - memsom_quarantine.consolidate() runs after minting (integrity gate).
    """
    migrate(conn)
    memsom_quarantine.migrate(conn)
    memsom_confid.migrate(conn)   # ensure conf_label exists for high-water recompute

    if group_by not in ("similarity", "claim"):
        raise ValueError(f"unknown group_by: {group_by!r}")

    # Fetch live, non-archived, non-quarantined SOURCE episodes
    candidate_rows = conn.execute(
        "SELECT id, content, channel, label FROM nodes"
        " WHERE tombstoned = 0 AND archived = 0"
        "   AND channel != 'agent-derived' AND status != 'quarantined'"
        " ORDER BY id"
    ).fetchall()

    if not candidate_rows:
        memsom_quarantine.consolidate(conn)
        return []

    content_by_id = {r[0]: r[1] for r in candidate_rows}

    # Build groups
    if group_by == "similarity":
        groups = _group_similarity(candidate_rows, min_group, sim_threshold)
    else:  # "claim"
        raw_groups = _group_by_claim(conn)
        # Filter out nodes not in our candidate set
        groups = [
            [nid for nid in g if nid in content_by_id]
            for g in raw_groups
        ]
        groups = [g for g in groups if len(g) >= min_group]

    minted = []
    consumed = set()
    now = memsom.now_iso()

    for group_ids in groups:
        # Filter already-consumed (overlap from claim grouping)
        group_ids = [i for i in group_ids if i not in consumed]
        if len(group_ids) < min_group:
            continue

        rows = [(i, content_by_id[i]) for i in group_ids]  # id ascending

        if llm:
            try:
                summary = _llm_summarize(rows)
                engine = "llm"
            except memsom_llm.LlmUnavailable:
                summary = extractive_summary(rows, k)
                engine = "extractive"  # fell back: regenerate deterministically, NOT as llm
        else:
            summary = extractive_summary(rows, k)
            engine = "extractive"

        # LOAD-BEARING: edges to EVERY episode, label = min(parents) — no laundering
        nid, _label = memsom.derive_node(conn, summary, group_ids)

        # COMPACT-1: stamp the high-water conf_label AND archive the episodes in ONE
        # transaction, so a crash can't leave the summary archived-but-conf-0 (a
        # SECRET-derived node readable at PUBLIC). Bell-LaPadula high-water: a node
        # summarizing SECRET episodes is SECRET; derive_node leaves conf at the
        # DEFAULT 0, so recompute it to max(live-parent conf) here (inlined rather
        # than memsom_confid.recompute_conf, whose own `with conn:` would commit
        # separately and re-open the window between conf and archive).
        # NB: derive_node (frozen core) commits the node first, so a micro-window at
        # conf=0 remains before this block — the documented LOW residual, corrected
        # by memsom_heal/recompute_conf_all on the next run.
        parent_conf = conn.execute(
            "SELECT COALESCE(MAX(n.conf_label), 0) FROM edges e"
            " JOIN nodes n ON n.id = e.parent"
            " WHERE e.child = ? AND n.tombstoned = 0",
            (nid,),
        ).fetchone()[0]
        with conn:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE nodes SET conf_label = ? WHERE id = ?", (parent_conf, nid)
            )
            # Record how this summary was made, atomically with conf+archive.
            # extractive -> store k so replay matches length; llm -> parked (NULL).
            if engine == "extractive":
                memsom_rederive.record_recipe(conn, nid, engine, k=k)
            else:
                memsom_rederive.record_recipe(conn, nid, engine)
            conn.executemany(
                "UPDATE nodes SET archived = 1, archived_at = ? WHERE id = ?",
                [(now, i) for i in group_ids],
            )
            # COMPACT-2 (index drift): keep the BM25/vector index in lock-step with
            # archival. A just-archived episode is pool-excluded from every read
            # path, but if its postings/docstats survive they still skew the BM25
            # CORPUS STATS (N, avgdl, df/idf) and leave duplicate term mass behind
            # the summary. Purge it from all retrieval structures IN THIS SAME
            # TRANSACTION so the archive + de-index are atomic — a crash can't leave
            # a node archived-but-indexed. Inlined (not memsom_retrieve.deindex_node)
            # because that helper's own `with conn:` would COMMIT here and re-open
            # the very window this atomicity closes. Guarded on table existence:
            # the retrieval schema is optional and may never have been migrated.
            if memsom_schema.table_exists(conn, "postings"):
                conn.executemany("DELETE FROM postings WHERE node_id = ?",
                                 [(i,) for i in group_ids])
                conn.executemany("DELETE FROM docstats WHERE node_id = ?",
                                 [(i,) for i in group_ids])
                if memsom_schema.table_exists(conn, "embeddings"):
                    conn.executemany("DELETE FROM embeddings WHERE node_id = ?",
                                     [(i,) for i in group_ids])
                # BGE side-tables (sparse_vecs/colbert_vecs) follow embeddings in
                # lock-step — same atomic txn, same corpus-hygiene reason. Inlined
                # (not memsom_embed.deindex_bge per id) to stay in this one txn.
                if memsom_schema.table_exists(conn, "sparse_vecs"):
                    conn.executemany("DELETE FROM sparse_vecs WHERE node_id = ?",
                                     [(i,) for i in group_ids])
                if memsom_schema.table_exists(conn, "colbert_vecs"):
                    conn.executemany("DELETE FROM colbert_vecs WHERE node_id = ?",
                                     [(i,) for i in group_ids])

        consumed.update(group_ids)
        minted.append(nid)

    # Integrity gate: external-tainted semantic nodes -> quarantined
    memsom_quarantine.consolidate(conn)

    return minted


# ---------------------------------------------------------------------------
# list_archived
# ---------------------------------------------------------------------------

def list_archived(conn: sqlite3.Connection) -> list:
    """Return list of dicts for all archived nodes, ordered by id.

    Each dict: id, channel, label, archived_at, content.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id, channel, label, archived_at, content"
        " FROM nodes WHERE archived = 1 ORDER BY id"
    ).fetchall()
    keys = ("id", "channel", "label", "archived_at", "content")
    return [dict(zip(keys, r)) for r in rows]


# ---------------------------------------------------------------------------
# CLI handlers (print/sys.exit allowed here)
# ---------------------------------------------------------------------------

def _cmd_compact(args) -> None:
    conn = memsom.get_connection()
    try:
        minted = compact(
            conn,
            group_by=args.group_by,
            llm=args.llm,
            min_group=args.min_group,
        )
        if not minted:
            print("nothing to compact")
        else:
            for nid in minted:
                parents = memsom.parents_of(conn, nid)
                node = memsom.get_node(conn, nid)
                status = conn.execute(
                    "SELECT status FROM nodes WHERE id = ?", (nid,)
                ).fetchone()[0]
                print(
                    f"[{nid}] minted semantic node"
                    f" integrity={memsom.NAME[node['label']]}"
                    f" <- {len(parents)} episodes"
                    f" {[p[0] for p in parents]}"
                )
                if status == "quarantined":
                    print(f"[{nid}] QUARANTINED by consolidation gate")
    finally:
        conn.close()


def _cmd_archived_list(args) -> None:
    conn = memsom.get_connection()
    try:
        rows = list_archived(conn)
        if not rows:
            print("no archived nodes")
        else:
            for r in rows:
                print(
                    f"[{r['id']}] {r['channel']:<13}"
                    f" integrity={memsom.NAME[r['label']]:<13}"
                    f" archived={r['archived_at']}"
                )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# register / main
# ---------------------------------------------------------------------------

def register(subparsers) -> None:
    """Mount compact and archived-list onto *subparsers*."""
    p_c = subparsers.add_parser(
        "compact",
        help="consolidate related episodes into a semantic node (provenance-preserving)",
    )
    p_c.add_argument(
        "--group-by",
        default="similarity",
        choices=["similarity", "claim"],
        dest="group_by",
    )
    p_c.add_argument("--llm", action="store_true",
                     help="use local Ollama LLM for summary (falls back to extractive)")
    p_c.add_argument("--min-group", type=int, default=2, dest="min_group",
                     help="minimum group size to compact (default 2)")
    p_c.set_defaults(func=_cmd_compact)

    p_a = subparsers.add_parser(
        "archived-list",
        help="list archived (compacted) episodes",
    )
    p_a.set_defaults(func=_cmd_archived_list)


def main(argv=None) -> None:
    """Thin CLI wrapper — delegates to register()."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom-compact")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
