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
    memdag_anticipatory -> observe prefetch
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
    """
    clearance = memdag_confid.parse_conf(clearance_name)
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE tombstoned = 0"
        "   AND channel != 'agent-derived'"
        "   AND status != 'quarantined'"
        "   AND redacted = 0"
        "   AND conf_label <= ?"
        " ORDER BY label DESC, id ASC",
        (clearance,)
    ).fetchall()


def _count_excluded(conn, pool_ids, clearance_name):
    """Return a dict of exclusion counts for the summary line.

    Fields: tombstoned, quarantined, redacted, above_clearance.
    These are informational only and derived relative to the total raw
    source count (channel != agent-derived).
    """
    clearance = memdag_confid.parse_conf(clearance_name)
    # All source rows (ignoring tombstoned / quarantined / redacted / conf filters)
    all_sources = conn.execute(
        "SELECT id, tombstoned, status, redacted, conf_label FROM nodes"
        " WHERE channel != 'agent-derived'"
    ).fetchall()
    pool_set = set(pool_ids)
    tombstoned = 0
    quarantined = 0
    redacted = 0
    above_clearance = 0
    for sid, ts, status, red, conf in all_sources:
        if sid in pool_set:
            continue
        if ts:
            tombstoned += 1
        elif status == "quarantined":
            quarantined += 1
        elif red:
            redacted += 1
        elif conf > clearance:
            above_clearance += 1
    return {
        "tombstoned": tombstoned,
        "quarantined": quarantined,
        "redacted": redacted,
        "above_clearance": above_clearance,
    }


def cmd_ask(args):
    """Enhanced ask: layers quarantine/clearance/anticipate/llm over core compose."""
    conn = memdag.get_connection()
    try:
        migrate_all(conn)

        clearance = args.clearance  # default 'topsecret' = no filter
        pool = _build_pool(conn, clearance)

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
        excl_total = excl["tombstoned"] + excl["quarantined"] + excl["redacted"] + excl["above_clearance"]

        text = None
        used = []
        nid = None
        label = None
        score = None
        created = True

        if args.anticipate:
            # surprise_gated: cite existing if low-novelty, else derive new
            threshold = args.threshold
            try:
                nid, created, score = memdag_anticipatory.surprise_gated(
                    conn, args.question, threshold=threshold, sources=pool
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
                    f"redacted: {excl['redacted']}, above-clearance: {excl['above_clearance']}"
                )
                print(
                    f"\nstored as node [{nid}] | integrity: {memdag.NAME[label]}"
                    f" (floor of {len(used_ids)} parents)"
                    f" | sources considered: {total_sources}, used: {len(used_ids)},"
                    f" excluded: {excl_total} ({excl_detail})"
                )
                p = memdag_profile.profile(conn, nid)
                print(memdag_profile.format_profile(p))
            return

        if args.llm:
            try:
                text, used = memdag_llm.llm_compose(args.question, pool, model=args.model)
            except memdag_llm.LlmUnavailable as e:
                print(
                    f"[memdag] {e}; falling back to deterministic compose",
                    file=sys.stderr,
                )
                text, used = memdag.compose(args.question, pool)
            except ValueError:
                print(
                    "[memdag] no live source yielded any claim; nothing stored",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            text, used = memdag.compose(args.question, pool)

        if not text:
            print(
                "[memdag] no live source yielded any claim; nothing stored",
                file=sys.stderr,
            )
            sys.exit(1)

        nid, label = memdag.derive_node(conn, text, used)
        # Stamp confidentiality high-water mark
        memdag_confid.recompute_conf(conn, nid)

        excl_detail = (
            f"tombstoned: {excl['tombstoned']}, quarantined: {excl['quarantined']}, "
            f"redacted: {excl['redacted']}, above-clearance: {excl['above_clearance']}"
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
        with conn:
            nid = memdag.insert_node(conn, args.content, args.channel,
                                     source_ref=args.ref)
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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
