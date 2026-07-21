#!/usr/bin/env python3
"""memsom_cli — unified CLI surface for the full memsom stack.

Entry point: `python memsom_cli.py <subcommand>`.

This file wires all 16 feature modules into a single argparse CLI.
memsom.py (demo #1) is NOT touched — it stays byte-identical.

Subcommands (39 total):
  Core (delegated to memsom.*):   seed ask explain revoke dump
  CLI-owned enhanced:             ask (enhanced), add, migrate
  Modules via register():
    memsom_recompute  -> recompute
    memsom_redact     -> redact
    memsom_quarantine -> consolidate quarantine promote quarantine-list
    memsom_confid     -> classify conf-recompute
    memsom_federation -> export import
    memsom_blame      -> blame
    memsom_relate     -> relate neighborhood
    memsom_anticipatory -> observe prefetch anticipate-status
    memsom_distill    -> export-training distill-plan
    memsom_heal       -> check rebuild-derived
    memsom_trust      -> elevate meet join elevations
    memsom_llm        -> llm-check
    memsom_profile    -> profile
    memsom_gate       -> check-action gate-log
    memsom_corroborate -> register-root assert-claim corroborate claims-list roots-list
    memsom_verify_stale -> verify-stale
    memsom_facts      -> fact-set fact-log
"""

import argparse
import contextlib
import io
import os
import sys

import memsom
from memsom.storage import schema as memsom_schema
from memsom.retrieval import recompute as memsom_recompute
from memsom.integrity import redact as memsom_redact
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import confid as memsom_confid
from memsom.federation import federation as memsom_federation
from memsom.interface import blame as memsom_blame
from memsom.retrieval import relate as memsom_relate
from memsom.lifecycle import anticipatory as memsom_anticipatory
from memsom.distill import distill as memsom_distill
from memsom.lifecycle import heal as memsom_heal
from memsom.integrity import trust as memsom_trust
from memsom.distill import llm as memsom_llm
from memsom.interface import profile as memsom_profile
from memsom.integrity import gate as memsom_gate
from memsom.integrity import corroborate as memsom_corroborate
from memsom.interface import ingest as memsom_ingest
from memsom.retrieval import retrieve as memsom_retrieve
from memsom.retrieval import embed as memsom_embed
from memsom.retrieval import code_index as memsom_code_index
from memsom.lifecycle import compact as memsom_compact
from memsom.lifecycle import reflex as memsom_reflex
from memsom.bridge import chats as memsom_chats
from memsom.lifecycle import doctor as memsom_doctor
from memsom.storage import config as memsom_config
from memsom.bridge import obsidian as memsom_obsidian
from memsom.retrieval import rederive as memsom_rederive
from memsom.storage import session as memsom_session
from memsom.integrity import capgate as memsom_capgate
from memsom.federation import broker as memsom_broker
from memsom.bridge import hook as memsom_hook
from memsom.integrity import stale as memsom_stale
from memsom.integrity import verify_stale as memsom_verify_stale
from memsom.bridge import bridge_render as memsom_bridge_render
from memsom.bridge import claude as memsom_claude
from memsom.bridge import wire_claude as memsom_wire_claude
from memsom.interface import audit as memsom_audit
from memsom.interface import dashboard as memsom_dashboard
from memsom.interface import panel as memsom_panel
from memsom.integrity import tombstone as memsom_tombstone
from memsom.integrity import contradict as memsom_contradict
from memsom.bridge import facts as memsom_facts


# ---------------------------------------------------------------------------
# migrate_all: run every module's migrate() against conn
# Exposed so cmd_ask can call it, and also as the 'migrate' subcommand.
# ---------------------------------------------------------------------------

def migrate_all(conn):
    """Run all module migrations idempotently against *conn*."""
    memsom_recompute.migrate(conn)
    memsom_redact.migrate(conn)
    memsom_quarantine.migrate(conn)
    memsom_confid.migrate(conn)
    memsom_federation.migrate(conn)
    memsom_blame.migrate(conn)
    memsom_relate.migrate(conn)
    memsom_anticipatory.migrate(conn)
    memsom_distill.migrate(conn)
    memsom_heal.migrate(conn)
    memsom_trust.migrate(conn)
    memsom_llm.migrate(conn)
    memsom_profile.migrate(conn)
    memsom_gate.migrate(conn)
    memsom_corroborate.migrate(conn)
    memsom_ingest.migrate(conn)
    memsom_retrieve.migrate(conn)
    memsom_embed.migrate(conn)
    memsom_code_index.migrate(conn)
    memsom_compact.migrate(conn)
    memsom_obsidian.migrate(conn)
    memsom_rederive.migrate(conn)
    memsom_stale.migrate(conn)
    # Versioned, once-only steps run AFTER all additive per-module migrate()s.
    # Owns operations that must run exactly once in order (e.g. the destructive
    # status-CHECK table rebuild) and is gated by PRAGMA user_version.
    memsom_schema.run_versioned_migrations(conn)


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

    Returns list of rows shaped like memsom.live_sources:
    (id, content, channel, label, source_ref).

    Taint dimensions come from memsom_schema.taint_filter_clauses — the ONE
    shared untainted-pool primitive (same clauses as
    memsom_retrieve._build_retrieve_pool and
    memsom_anticipatory._untainted_clauses, by construction).
    """
    clearance = memsom_confid.parse_conf(clearance_name)
    clauses, params = memsom_schema.taint_filter_clauses(conn, clearance=clearance)
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
    clearance = memsom_confid.parse_conf(clearance_name)
    # CLI-2: the pool filter excludes archived sources too, so the summary must
    # account for them — otherwise, once compact has run, archived sources vanish
    # from excl_total and the audit line under-reports how many were dropped.
    has_archived = memsom_schema.column_exists(conn, "nodes", "archived")
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
    conn = memsom.get_connection()
    try:
        migrate_all(conn)

        clearance = args.clearance  # default 'topsecret' = no filter
        try:
            memsom_confid.parse_conf(clearance)  # CLI-1: validate before any work
        except ValueError as exc:
            print(f"[memsom] invalid --clearance: {exc}", file=sys.stderr)
            sys.exit(1)

        # --embed-backend overrides MEMDAG_EMBED_BACKEND for this process so the
        # retrieve/reindex paths pick it up. Warn once on an interactive bge
        # single-shot: a per-invocation CLI reloads ~2.2GB cold (10-30s) — bge is
        # meant for batch `reindex` + the long-lived MCP/server, not one-offs.
        if getattr(args, "embed_backend", None):
            os.environ["MEMDAG_EMBED_BACKEND"] = args.embed_backend
            uses_retrieval = getattr(args, "retrieve", False) or getattr(args, "graph", False)
            if (args.embed_backend == "bge-m3" and uses_retrieval
                    and memsom_embed.bge_available()):
                print("[memsom] bge-m3: cold-loading the model (~2.2GB) for a "
                      "single query; prefer a warm reindex/server for repeated use.",
                      file=sys.stderr)

        pool = _build_pool(conn, clearance)

        if getattr(args, "graph", False):
            # GraphRAG-lite: retrieval re-ranked by the rel_edges (wikilink) graph.
            # Implies a retrieved pool (graph re-ranking over all-live is a no-op).
            pool = memsom_retrieve.retrieve_graph(
                conn, args.question, k=args.topk, clearance=clearance,
                hops=getattr(args, "hops", 1),
            )
        elif args.retrieve:
            pool = memsom_retrieve.retrieve(conn, args.question, k=args.topk, clearance=clearance)

        # --prefer-fresh: staleness-aware hybrid serving. SUBSTITUTE each stale
        # source in the pool with its current version (the deterministic supersedes
        # edge rescuing what retrieval surfaced) so the answer resolves clean.
        # Read-time + non-mutating (the store is untouched; that's the `freshen`
        # VERB). Runs BEFORE --fresh-only so the two compose: substitute what we
        # can, then exclude any leftover un-substitutable stale.
        substitutions = []
        if getattr(args, "prefer_fresh", False) and pool:
            pool, substitutions = memsom_stale.substitute_fresh(
                conn, pool, memsom_confid.parse_conf(clearance))

        # --fresh-only: opt-in exclusion of stale sources (disclose-by-default
        # otherwise). Applied uniformly over the finalized pool so it covers the
        # default / --retrieve / --graph paths identically. stale is NOT in
        # taint_filter_clauses by design (see memsom_stale): staleness never
        # excludes unless the operator asks for it here.
        stale_excluded = 0
        if getattr(args, "fresh_only", False) and pool:
            stale_in_pool = set(memsom_stale.stale_annotations(
                conn, [r[0] for r in pool]).keys())
            if stale_in_pool:
                stale_excluded = len(stale_in_pool)
                pool = [r for r in pool if r[0] not in stale_in_pool]

        # Count total sources for summary
        total_sources = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel != 'agent-derived'"
        ).fetchone()[0]

        if not pool:
            print(
                "[memsom] no live sources - refusing to compose an unprovenanced answer",
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
            warm = memsom_anticipatory.serve_warm(conn, args.question,
                                                  clearance=clearance)
            if warm is not None:
                memsom_anticipatory.observe(conn, args.question, warm["node_id"])
                print(warm["content"])
                print(
                    f"\nserved WARM from prefetch cache: node [{warm['node_id']}]"
                    f" | integrity: {memsom.NAME[warm['label']]}"
                    f" | prefetched {warm['prefetched_at']} | hits {warm['hits']}"
                )
                p = memsom_profile.profile(conn, warm["node_id"])
                print(memsom_profile.format_profile(p))
                return

            # surprise_gated_write: semantic dedup — cite existing if
            # low-novelty (BM25-IDF cosine + optional Ollama vectors over the
            # untainted derived corpus), else derive new (label = min(parents)).
            threshold = args.threshold
            try:
                nid, created, score = memsom_anticipatory.surprise_gated_write(
                    conn, args.question, threshold=threshold, sources=pool,
                    clearance=clearance
                )
            except ValueError:
                print(
                    "[memsom] no live source yielded any claim; nothing stored",
                    file=sys.stderr,
                )
                sys.exit(1)
            node = memsom.get_node(conn, nid)
            text = node["content"]
            label = node["label"]
            # Stamp conf for derived nodes
            memsom_confid.recompute_conf(conn, nid)
            print(text)
            if not created:
                print(f"\ncited EXISTING node [{nid}] (novelty {score:.2f} < threshold)")
                p = memsom_profile.profile(conn, nid)
                print(memsom_profile.format_profile(p))
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
                    f"\nstored as node [{nid}] | integrity: {memsom.NAME[label]}"
                    f" (floor of {len(used_ids)} parents)"
                    f" | sources considered: {total_sources}, used: {len(used_ids)},"
                    f" excluded: {excl_total} ({excl_detail})"
                )
                # Phase 2: episode-recombination warning — flag a parent SET
                # that has never produced a derived node before.
                parent_ids = [r[0] for r in conn.execute(
                    "SELECT parent FROM edges WHERE child = ?", (nid,)
                ).fetchall()]
                is_novel, prior = memsom_anticipatory.novel_recombination(
                    conn, parent_ids, exclude_node=nid, clearance=clearance
                )
                if is_novel:
                    print("new inference: this combination of sources has"
                          " not been seen before.")
                else:
                    seen = ", ".join(f"[{p_}]" for p_ in prior)
                    print(f"combination previously derived in node(s): {seen}")
                p = memsom_profile.profile(conn, nid)
                print(memsom_profile.format_profile(p))
            return

        if args.llm:
            try:
                text, used = memsom_llm.llm_compose(args.question, pool, model=args.model)
                recipe_engine = "llm"
                recipe_kw = {"question": args.question, "model": args.model}
            except memsom_llm.LlmUnavailable as e:
                print(
                    f"[memsom] {e}; falling back to deterministic compose",
                    file=sys.stderr,
                )
                text, used = memsom.compose(args.question, pool)
                recipe_engine = "compose"  # fell back: deterministic, NOT llm
                recipe_kw = {"question": args.question}
            except ValueError:
                print(
                    "[memsom] no live source yielded any claim; nothing stored",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            text, used = memsom.compose(args.question, pool)
            recipe_engine = "compose"
            recipe_kw = {"question": args.question}

        if not text:
            print(
                "[memsom] no live source yielded any claim; nothing stored",
                file=sys.stderr,
            )
            sys.exit(1)

        nid, label = memsom.derive_node(conn, text, used)
        # Record the recipe so this summary can be regenerated from live parents
        # after a source is revoked/redacted. derive_node commits its own txn, so
        # this is a small separate write — same micro-window compact documents.
        with conn:
            memsom_rederive.record_recipe(conn, nid, recipe_engine, **recipe_kw)
        # Stamp confidentiality high-water mark
        memsom_confid.recompute_conf(conn, nid)

        # Staleness disclosure: flag exactly the cited sources that have gone
        # stale (precise — `used` is compose's own citation set, not a heuristic).
        stale_used = memsom_stale.stale_annotations(conn, used)

        excl_detail = (
            f"tombstoned: {excl['tombstoned']}, quarantined: {excl['quarantined']}, "
            f"redacted: {excl['redacted']}, archived: {excl['archived']}, "
            f"above-clearance: {excl['above_clearance']}"
        )
        print(text)
        if substitutions:
            print(f"\n[FRESHENED] resolved {len(substitutions)} stale source(s)"
                  " to their current version:")
            for old, new in substitutions:
                print(f"  [{old}] -> [{new}]")
            print("(read-time substitution; the store is unchanged."
                  " run `memsom freshen <id>` to persist.)")
        if stale_used:
            print("\n[STALE] this answer draws on stale source(s):")
            for sid in sorted(stale_used):
                info = stale_used[sid]
                fresh = info.get("fresh_id")
                fresh_note = f" -- fresh version [{fresh}] available" if fresh else ""
                reason = info.get("reason") or "source changed"
                print(f"  [{sid}] {reason} (since {info.get('stale_at')}){fresh_note}")
            print("Pass --fresh-only to exclude, or `memsom freshen <derived_id>` to repoint.")
        print(
            f"\nstored as node [{nid}] | integrity: {memsom.NAME[label]}"
            f" (floor of {len(used)} parents)"
            f" | sources considered: {total_sources}, used: {len(used)},"
            f" excluded: {excl_total} ({excl_detail})"
        )
        if getattr(args, "fresh_only", False) and stale_excluded:
            print(f"({stale_excluded} stale source(s) excluded by --fresh-only)")
        p = memsom_profile.profile(conn, nid)
        print(memsom_profile.format_profile(p))

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# add subcommand (inject a source node — used by demo/demo2.ps1)
# ---------------------------------------------------------------------------

def cmd_add(args):
    """Insert a new source node and print its id."""
    conn = memsom.get_connection()
    try:
        migrate_all(conn)
        # Caller-layer trust guards: enforce the optional channel ceiling (F-13)
        # and pin the integrity label to the channel (F-14) so a mismatched label
        # can never be stamped through the `add` path.
        try:
            memsom_ingest.enforce_channel_ceiling(args.channel)
            label = memsom_ingest.authoritative_label(args.channel)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        with conn:
            nid = memsom.insert_node(conn, args.content, args.channel,
                                     label=label, source_ref=args.ref)
        node = memsom.get_node(conn, nid)
        print(f"[{nid}] {node['channel']:<13} integrity={memsom.NAME[node['label']]:<13}"
              f" {len(node['content']):>6} chars")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# migrate subcommand
# ---------------------------------------------------------------------------

def cmd_migrate(args):
    """Run all module migrations. Safe to call multiple times (idempotent)."""
    conn = memsom.get_connection()
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
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir or str(memsom.DATA_DIR)))
    os.makedirs(data_dir, exist_ok=True)
    db = os.path.join(data_dir, "memdag.db")
    existed = os.path.exists(db)
    conn = memsom.get_connection(db)
    try:
        migrate_all(conn)
    finally:
        conn.close()
    note = "already present, schema up to date" if existed else "created"
    print(f"[memsom] data dir {data_dir} ({note})", file=sys.stderr)
    print(db)  # stdout: the bare DB path, for the bootstrap to capture


# ---------------------------------------------------------------------------
# Core command bridges (delegate to frozen memsom functions)
# ---------------------------------------------------------------------------

def cmd_seed(args):
    memsom.cmd_seed(args)


def cmd_explain(args):
    memsom.cmd_explain(args)
    # After the frozen explain tree, append profile (Move 1 display — print-only, never gates)
    conn = memsom.get_connection()
    try:
        migrate_all(conn)
        # F-16: the frozen explain renders a redacted node as an empty snippet,
        # indistinguishable from a genuinely empty node. Surface an explicit
        # [REDACTED] marker (with date + reason) via the redact module's describe().
        with contextlib.suppress(ValueError):
            if memsom_redact.is_redacted(conn, args.id):
                for line in memsom_redact.describe(conn, args.id):
                    print(line)
        with contextlib.suppress(ValueError):
            p = memsom_profile.profile(conn, args.id)
            print(memsom_profile.format_profile(p))
    finally:
        conn.close()


def cmd_revoke(args):
    memsom.cmd_revoke(args)


def cmd_dump(args):
    memsom.cmd_dump(args)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    p = argparse.ArgumentParser(prog="memsom")
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
    s_ask.add_argument("--embed-backend", choices=["ollama", "bge-m3", "bm25"],
                       default=None,
                       help="embedding backend for this query (overrides "
                            "MEMDAG_EMBED_BACKEND; bge-m3 = FlagEmbedding triple "
                            "fusion, best as a warm reindex/server, ~2.2GB cold load)")
    s_ask.add_argument("--fresh-only", action="store_true",
                       help="exclude stale sources (default: include + flag them)")
    s_ask.add_argument("--prefer-fresh", action="store_true",
                       help="substitute stale sources with their current version (hybrid serving)")
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
    memsom_recompute.register(sub)
    memsom_redact.register(sub)
    memsom_quarantine.register(sub)
    memsom_confid.register(sub)
    memsom_federation.register(sub)
    memsom_blame.register(sub)
    memsom_relate.register(sub)
    memsom_anticipatory.register(sub)
    memsom_distill.register(sub)
    memsom_heal.register(sub)
    memsom_trust.register(sub)
    memsom_llm.register(sub)
    memsom_profile.register(sub)
    memsom_gate.register(sub)
    memsom_corroborate.register(sub)
    memsom_ingest.register(sub)
    memsom_retrieve.register(sub)
    memsom_code_index.register(sub)
    memsom_compact.register(sub)
    memsom_reflex.register(sub)
    memsom_chats.register(sub)
    memsom_doctor.register(sub)
    memsom_config.register(sub)
    memsom_obsidian.register(sub)
    memsom_session.register(sub)
    memsom_capgate.register(sub)
    memsom_broker.register(sub)
    memsom_hook.register(sub)
    memsom_stale.register(sub)
    memsom_verify_stale.register(sub)
    memsom_bridge_render.register(sub)
    memsom_claude.register(sub)
    memsom_wire_claude.register(sub)
    memsom_audit.register(sub)
    memsom_dashboard.register(sub)
    memsom_panel.register(sub)
    memsom_tombstone.register(sub)
    memsom_contradict.register(sub)
    memsom_facts.register(sub)

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
