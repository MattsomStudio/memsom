#!/usr/bin/env python3
"""memdag_cli — unified CLI surface for the full memdag stack.

Entry point: `python memdag_cli.py <subcommand>`.

This file wires all 16 feature modules into a single argparse CLI.
memdag.py (demo #1) is NOT touched — it stays byte-identical.

Subcommands (38 total):
  Core (delegated to memdag.*):   seed ask explain revoke dump
  CLI-owned enhanced:             ask (enhanced), add, migrate
  Modules via register():
    memdag_recompute  -> recompute
    memdag_redact     -> redact
    memdag_quarantine -> consolidate quarantine promote quarantine-list
    memdag_confid     -> classify conf-recompute
    memdag_federation -> export import
    memdag_blame      -> blame
    memdag_relate     -> relate neighborhood
    memdag_anticipatory -> observe prefetch anticipate-status
    memdag_distill    -> export-training distill-plan
    memdag_heal       -> check rebuild-derived
    memdag_trust      -> elevate meet join elevations
    memdag_llm        -> llm-check
    memdag_profile    -> profile
    memdag_gate       -> check-action gate-log
    memdag_corroborate -> register-root assert-claim corroborate claims-list roots-list
"""

import argparse
import contextlib
import io
import os
import sys

import memdag
import memdag_schema
import memdag_recompute
import memdag_redact
import memdag_quarantine
import memdag_confid
import memdag_federation
import memdag_blame
import memdag_relate
import memdag_anticipatory
import memdag_distill
import memdag_heal
import memdag_trust
import memdag_llm
import memdag_profile
import memdag_gate
import memdag_corroborate
import memdag_ingest
import memdag_retrieve
import memdag_compact
import memdag_reflex
import memdag_chats
import memdag_doctor
import memdag_config
import memdag_obsidian
import memdag_rederive


# ---------------------------------------------------------------------------
# migrate_all: run every module's migrate() against conn
# Exposed so cmd_ask can call it, and also as the 'migrate' subcommand.
# ---------------------------------------------------------------------------

def migrate_all(conn):
    """Run all module migrations idempotently against *conn*."""
    memdag_recompute.migrate(conn)
    memdag_redact.migrate(conn)
    memdag_quarantine.migrate(conn)
    memdag_confid.migrate(conn)
    memdag_federation.migrate(conn)
    memdag_blame.migrate(conn)
    memdag_relate.migrate(conn)
    memdag_anticipatory.migrate(conn)
    memdag_distill.migrate(conn)
    memdag_heal.migrate(conn)
    memdag_trust.migrate(conn)
    memdag_llm.migrate(conn)
    memdag_profile.migrate(conn)
    memdag_gate.migrate(conn)
    memdag_corroborate.migrate(conn)
    memdag_ingest.migrate(conn)
    memdag_retrieve.migrate(conn)
    memdag_compact.migrate(conn)
    memdag_obsidian.migrate(conn)
    memdag_rederive.migrate(conn)
    # Versioned, once-only steps run AFTER all additive per-module migrate()s.
    # Owns operations that must run exactly once in order (e.g. the destructive
    # status-CHECK table rebuild) and is gated by PRAGMA user_version.
    memdag_schema.run_versioned_migrations(conn)


# ---------------------------------------------------------------------------
# Enhanced ask
# ---------------------------------------------------------------------------

def _build_pool(conn, clearance_name):
    """Build the filtered source pool for the enhanced ask.

    Filters applied (all stacked on top of the frozen live_sources behaviour):
      - tombstoned = 0
      - channel != 'agent-derived'
      - status != 'quarantined'   (quarantine module layer)
      - redacted = 0              (redact module layer)
      - conf_label <= clearance   (Bell-LaPadula no-read-up)

    Returns list of rows shaped like memdag.live_sources:
    (id, content, channel, label, source_ref).

    Taint dimensions come from memdag_schema.taint_filter_clauses — the ONE
    shared untainted-pool primitive (same clauses as
    memdag_retrieve._build_retrieve_pool and
    memdag_anticipatory._untainted_clauses, by construction).
    """
    clearance = memdag_confid.parse_conf(clearance_name)
    clauses, params = memdag_schema.taint_filter_clauses(conn, clearance=clearance)
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE channel != 'agent-derived' AND " + " AND ".join(clauses) +
        " ORDER BY label DESC, id ASC",
        params
    ).fetchall()


def _count_excluded(conn, pool_ids, clearance_name):
    """Return a dict of exclusion counts for the summary line.

    Fields: tombstoned, quarantined, redacted, archived, above_clearance.
    These are informational only and derived relative to the total raw
    source count (channel != agent-derived).
    """
    clearance = memdag_confid.parse_conf(clearance_name)
    # CLI-2: the pool filter excludes archived sources too, so the summary must
    # account for them — otherwise, once compact has run, archived sources vanish
    # from excl_total and the audit line under-reports how many were dropped.
    has_archived = memdag_schema.column_exists(conn, "nodes", "archived")
    arch_col = ", archived" if has_archived else ", 0 AS archived"
    all_sources = conn.execute(
        "SELECT id, tombstoned, status, redacted, conf_label" + arch_col +
        " FROM nodes WHERE channel != 'agent-derived'"
    ).fetchall()
    pool_set = set(pool_ids)
    tombstoned = 0
    quarantined = 0
    redacted = 0
    archived = 0
    above_clearance = 0
    for sid, ts, status, red, conf, arch in all_sources:
        if sid in pool_set:
            continue
        if ts:
            tombstoned += 1
        elif status == "quarantined":
            quarantined += 1
        elif red:
            redacted += 1
        elif arch:
            archived += 1
        elif conf > clearance:
            above_clearance += 1
    return {
        "tombstoned": tombstoned,
        "quarantined": quarantined,
        "redacted": redacted,
        "archived": archived,
        "above_clearance": above_clearance,
    }


def cmd_ask(args):
    """Enhanced ask: layers quarantine/clearance/anticipate/llm over core compose."""
    conn = memdag.get_connection()
    try:
        migrate_all(conn)

        clearance = args.clearance  # default 'topsecret' = no filter
        try:
            memdag_confid.parse_conf(clearance)  # CLI-1: validate before any work
        except ValueError as exc:
            print(f"[memdag] invalid --clearance: {exc}", file=sys.stderr)
            sys.exit(1)
        pool = _build_pool(conn, clearance)

        if getattr(args, "graph", False):
            # GraphRAG-lite: retrieval re-ranked by the rel_edges (wikilink) graph.
            # Implies a retrieved pool (graph re-ranking over all-live is a no-op).
            pool = memdag_retrieve.retrieve_graph(
                conn, args.question, k=args.topk, clearance=clearance,
                hops=getattr(args, "hops", 1),
            )
        elif args.retrieve:
            pool = memdag_retrieve.retrieve(conn, args.question, k=args.topk, clearance=clearance)

        # Count total sources for summary
        total_sources = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel != 'agent-derived'"
        ).fetchone()[0]

        if not pool:
            print(
                "[memdag] no live sources - refusing to compose an unprovenanced answer",
                file=sys.stderr,
            )
            sys.exit(1)

        pool_ids = [row[0] for row in pool]
        excl = _count_excluded(conn, pool_ids, clearance)
        excl_total = (excl["tombstoned"] + excl["quarantined"] + excl["redacted"]
                      + excl["archived"] + excl["above_clearance"])

        text = None
        used = []
        nid = None
        label = None
        score = None
        created = True

        if args.anticipate:
            # Phase 2: serve a warm prefetched answer on an exact query match.
            # serve_warm re-validates the cached node against the untainted
            # filter (tombstoned/redacted/quarantined/archived/external-tainted/
            # above-clearance all refuse) — a stale or poisoned cache entry is
            # dropped, never served.
            warm = memdag_anticipatory.serve_warm(conn, args.question,
                                                  clearance=clearance)
            if warm is not None:
                memdag_anticipatory.observe(conn, args.question, warm["node_id"])
                print(warm["content"])
                print(
                    f"\nserved WARM from prefetch cache: node [{warm['node_id']}]"
                    f" | integrity: {memdag.NAME[warm['label']]}"
                    f" | prefetched {warm['prefetched_at']} | hits {warm['hits']}"
                )
                p = memdag_profile.profile(conn, warm["node_id"])
                print(memdag_profile.format_profile(p))
                return

            # surprise_gated_write: semantic dedup — cite existing if
            # low-novelty (BM25-IDF cosine + optional Ollama vectors over the
            # untainted derived corpus), else derive new (label = min(parents)).
            threshold = args.threshold
            try:
                nid, created, score = memdag_anticipatory.surprise_gated_write(
                    conn, args.question, threshold=threshold, sources=pool,
                    clearance=clearance
                )
            except ValueError:
                print(
                    "[memdag] no live source yielded any claim; nothing stored",
                    file=sys.stderr,
                )
                sys.exit(1)
            node = memdag.get_node(conn, nid)
            text = node["content"]
            label = node["label"]
            # Stamp conf for derived nodes
            memdag_confid.recompute_conf(conn, nid)
            print(text)
            if not created:
                print(f"\ncited EXISTING node [{nid}] (novelty {score:.2f} < threshold)")
                p = memdag_profile.profile(conn, nid)
                print(memdag_profile.format_profile(p))
            else:
                # Count used from the content citations
                import re
                used_ids = sorted({int(m) for m in re.findall(r'\[mem:(\d+)\|', text)})
                excl_detail = (
                    f"tombstoned: {excl['tombstoned']}, quarantined: {excl['quarantined']}, "
                    f"redacted: {excl['redacted']}, archived: {excl['archived']}, "
                    f"above-clearance: {excl['above_clearance']}"
                )
                print(
                    f"\nstored as node [{nid}] | integrity: {memdag.NAME[label]}"
                    f" (floor of {len(used_ids)} parents)"
                    f" | sources considered: {total_sources}, used: {len(used_ids)},"
                    f" excluded: {excl_total} ({excl_detail})"
                )
                # Phase 2: episode-recombination warning — flag a parent SET
                # that has never produced a derived node before.
                parent_ids = [r[0] for r in conn.execute(
                    "SELECT parent FROM edges WHERE child = ?", (nid,)
                ).fetchall()]
                is_novel, prior = memdag_anticipatory.novel_recombination(
                    conn, parent_ids, exclude_node=nid, clearance=clearance
                )
                if is_novel:
                    print("new inference: this combination of sources has"
                          " not been seen before.")
                else:
                    seen = ", ".join(f"[{p_}]" for p_ in prior)
                    print(f"combination previously derived in node(s): {seen}")
                p = memdag_profile.profile(conn, nid)
                print(memdag_profile.format_profile(p))
            return

        if args.llm:
            try:
                text, used = memdag_llm.llm_compose(args.question, pool, model=args.model)
                recipe_engine = "llm"
                recipe_kw = {"question": args.question, "model": args.model}
            except memdag_llm.LlmUnavailable as e:
                print(
                    f"[memdag] {e}; falling back to deterministic compose",
                    file=sys.stderr,
                )
                text, used = memdag.compose(args.question, pool)
                recipe_engine = "compose"  # fell back: deterministic, NOT llm
                recipe_kw = {"question": args.question}
            except ValueError:
                print(
                    "[memdag] no live source yielded any claim; nothing stored",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            text, used = memdag.compose(args.question, pool)
            recipe_engine = "compose"
            recipe_kw = {"question": args.question}

        if not text:
            print(
                "[memdag] no live source yielded any claim; nothing stored",
                file=sys.stderr,
            )
            sys.exit(1)

        nid, label = memdag.derive_node(conn, text, used)
        # Record the recipe so this summary can be regenerated from live parents
        # after a source is revoked/redacted. derive_node commits its own txn, so
        # this is a small separate write — same micro-window compact documents.
        with conn:
            memdag_rederive.record_recipe(conn, nid, recipe_engine, **recipe_kw)
        # Stamp confidentiality high-water mark
        memdag_confid.recompute_conf(conn, nid)

        excl_detail = (
            f"tombstoned: {excl['tombstoned']}, quarantined: {excl['quarantined']}, "
            f"redacted: {excl['redacted']}, archived: {excl['archived']}, "
            f"above-clearance: {excl['above_clearance']}"
        )
        print(text)
        print(
            f"\nstored as node [{nid}] | integrity: {memdag.NAME[label]}"
            f" (floor of {len(used)} parents)"
            f" | sources considered: {total_sources}, used: {len(used)},"
            f" excluded: {excl_total} ({excl_detail})"
        )
        p = memdag_profile.profile(conn, nid)
        print(memdag_profile.format_profile(p))

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# add subcommand (inject a source node — used by demo2.ps1)
# ---------------------------------------------------------------------------

def cmd_add(args):
    """Insert a new source node and print its id."""
    conn = memdag.get_connection()
    try:
        migrate_all(conn)
        # Caller-layer trust guards: enforce the optional channel ceiling (F-13)
        # and pin the integrity label to the channel (F-14) so a mismatched label
        # can never be stamped through the `add` path.
        try:
            memdag_ingest.enforce_channel_ceiling(args.channel)
            label = memdag_ingest.authoritative_label(args.channel)
        except ValueError as exc:
            print(f"[memdag] {exc}", file=sys.stderr)
            sys.exit(1)
        with conn:
            nid = memdag.insert_node(conn, args.content, args.channel,
                                     label=label, source_ref=args.ref)
        node = memdag.get_node(conn, nid)
        print(f"[{nid}] {node['channel']:<13} integrity={memdag.NAME[node['label']]:<13}"
              f" {len(node['content']):>6} chars")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# migrate subcommand
# ---------------------------------------------------------------------------

def cmd_migrate(args):
    """Run all module migrations. Safe to call multiple times (idempotent)."""
    conn = memdag.get_connection()
    try:
        migrate_all(conn)
        print("schema up to date")
    finally:
        conn.close()


def cmd_init(args):
    """Create the data dir + DB and run all migrations up front. Idempotent.

    A fresh `get_connection()` only creates the core schema; module tables exist
    only after each module's migrate() runs. `init` runs migrate_all ONCE so a
    friend's first interaction never hits a missing table regardless of which
    tool fires first. Prints the resolved DB path on stdout so the bootstrap can
    wire it into client configs via MEMDAG_DB.
    """
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir or str(memdag.DATA_DIR)))
    os.makedirs(data_dir, exist_ok=True)
    db = os.path.join(data_dir, "memdag.db")
    existed = os.path.exists(db)
    conn = memdag.get_connection(db)
    try:
        migrate_all(conn)
    finally:
        conn.close()
    note = "already present, schema up to date" if existed else "created"
    print(f"[memdag] data dir {data_dir} ({note})", file=sys.stderr)
    print(db)  # stdout: the bare DB path, for the bootstrap to capture


# ---------------------------------------------------------------------------
# Core command bridges (delegate to frozen memdag functions)
# ---------------------------------------------------------------------------

def cmd_seed(args):
    memdag.cmd_seed(args)


def cmd_explain(args):
    memdag.cmd_explain(args)
    # After the frozen explain tree, append profile (Move 1 display — print-only, never gates)
    conn = memdag.get_connection()
    try:
        migrate_all(conn)
        # F-16: the frozen explain renders a redacted node as an empty snippet,
        # indistinguishable from a genuinely empty node. Surface an explicit
        # [REDACTED] marker (with date + reason) via the redact module's describe().
        with contextlib.suppress(ValueError):
            if memdag_redact.is_redacted(conn, args.id):
                for line in memdag_redact.describe(conn, args.id):
                    print(line)
        with contextlib.suppress(ValueError):
            p = memdag_profile.profile(conn, args.id)
            print(memdag_profile.format_profile(p))
    finally:
        conn.close()


def cmd_revoke(args):
    memdag.cmd_revoke(args)


def cmd_dump(args):
    memdag.cmd_dump(args)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    p = argparse.ArgumentParser(prog="memdag")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- Core seed ----
    s_seed = sub.add_parser("seed", help="stamp initial source nodes")
    s_seed.add_argument("--offline", action="store_true")
    s_seed.add_argument("--reset", action="store_true")
    s_seed.set_defaults(func=cmd_seed)

    # ---- Enhanced ask ----
    s_ask = sub.add_parser("ask", help="compose an answer (enhanced: quarantine + clearance + anticipate + llm)")
    s_ask.add_argument("question")
    s_ask.add_argument("--clearance", default="topsecret",
                       help="confidentiality clearance level (default: topsecret = no filter)")
    s_ask.add_argument("--anticipate", action="store_true",
                       help="use surprise-gating: cite existing node if low-novelty")
    s_ask.add_argument("--threshold", type=float, default=0.35,
                       help="novelty threshold for --anticipate (default 0.35)")
    s_ask.add_argument("--llm", action="store_true",
                       help="use local Ollama LLM (opt-in; falls back to deterministic on error)")
    s_ask.add_argument("--model", default=None,
                       help="Ollama model name (overrides MEMDAG_LLM_MODEL)")
    s_ask.add_argument("--retrieve", action="store_true",
                       help="build the source pool via hybrid BM25+vector retrieval instead of all-live")
    s_ask.add_argument("--graph", action="store_true",
                       help="re-rank retrieval using the rel_edges (wikilink) graph (implies retrieval)")
    s_ask.add_argument("--hops", type=int, default=1,
                       help="graph expansion hops for --graph (default 1)")
    s_ask.add_argument("--topk", type=int, default=8,
                       help="max results when using --retrieve/--graph (default 8)")
    s_ask.set_defaults(func=cmd_ask)

    # ---- Core explain ----
    s_explain = sub.add_parser("explain", help="show provenance tree for a node")
    s_explain.add_argument("id", type=int)
    s_explain.set_defaults(func=cmd_explain)

    # ---- Core revoke ----
    s_revoke = sub.add_parser("revoke", help="tombstone a node and cascade to descendants")
    s_revoke.add_argument("id", type=int)
    s_revoke.add_argument("--reason", default="revoked by user")
    s_revoke.add_argument("--yes", action="store_true")
    s_revoke.set_defaults(func=cmd_revoke)

    # ---- Core dump ----
    sub.add_parser("dump", help="dump all nodes and edges").set_defaults(func=cmd_dump)

    # ---- add ----
    s_add = sub.add_parser("add", help="inject a new source node")
    s_add.add_argument("content")
    s_add.add_argument("--channel", required=True,
                       choices=["endorsed", "user", "agent-derived", "external"])
    s_add.add_argument("--ref", default=None, help="source reference / URL")
    s_add.set_defaults(func=cmd_add)

    # ---- migrate ----
    sub.add_parser("migrate", help="run all schema migrations (idempotent)").set_defaults(
        func=cmd_migrate
    )

    # ---- init ----
    s_init = sub.add_parser("init",
                            help="create the data dir + DB and run all migrations (idempotent)")
    s_init.add_argument("--data-dir", default=None,
                        help="where the DB lives (default: ~/.memdag, or $MEMDAG_HOME)")
    s_init.set_defaults(func=cmd_init)

    # ---- module mounts ----
    memdag_recompute.register(sub)
    memdag_redact.register(sub)
    memdag_quarantine.register(sub)
    memdag_confid.register(sub)
    memdag_federation.register(sub)
    memdag_blame.register(sub)
    memdag_relate.register(sub)
    memdag_anticipatory.register(sub)
    memdag_distill.register(sub)
    memdag_heal.register(sub)
    memdag_trust.register(sub)
    memdag_llm.register(sub)
    memdag_profile.register(sub)
    memdag_gate.register(sub)
    memdag_corroborate.register(sub)
    memdag_ingest.register(sub)
    memdag_retrieve.register(sub)
    memdag_compact.register(sub)
    memdag_reflex.register(sub)
    memdag_chats.register(sub)
    memdag_doctor.register(sub)
    memdag_config.register(sub)
    memdag_obsidian.register(sub)

    args = p.parse_args(argv)
    # Propagate a handler's NONZERO return as the process exit code so soft failures
    # (e.g. wire-config exists_differs/malformed) are detectable by callers like
    # bootstrap. A None/0 return falls through to a normal exit 0 — preserving the
    # contract that main() returns normally on success (handlers that sys.exit()
    # themselves still win).
    rc = args.func(args)
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
