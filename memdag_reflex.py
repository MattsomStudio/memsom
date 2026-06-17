"""memdag_reflex — reflex/schema-shaped training export (Phase 3 PoC).

THESIS UNDER TEST (2026-06-03 pivot): "strong base + system prompt + RAG beats
a fine-tuned 8B" for KNOWLEDGE.  This module therefore exports NO knowledge —
it exports RESPONSE SHAPE.  Facts stay in the DAG (RAG retrieves them at ask
time); what gets baked into weights is the house schema: cite-or-refuse,
provenance-tagged evidence bullets, integrity floor, next move.  Complementary
Learning Systems framing: slow weights = schema, fast DAG = episodes.

SECURITY INVARIANT (holds regardless of eval verdict): only UNTAINTED,
CONSOLIDATED memory is eligible.  Eligibility reuses memdag_distill's
belt-and-braces gate semantics — channel='agent-derived', alive, unredacted,
unquarantined, label >= floor, NO live external-channel ancestor — and adds
one more: the node must be a CONSOLIDATION product (every immediate parent
archived by memdag_compact).  Nothing external / quarantined / redacted /
tombstoned can ever reach the JSONL; assert_clean() enforces it at export.

Public API
----------
migrate(conn)
consolidated_ids(conn) -> set[int]
eligible_consolidated(conn, min_integrity=1) -> list[dict]
build_house_answer(nid, content, label, n_parents) -> str
build_refusal_answer() -> str
export_reflex(conn, min_integrity=1) -> list[dict]   # answer + refusal records
assert_clean(records, forbidden) -> None             # raises ValueError on taint

CLI
---
export-reflex <out.jsonl> [--min-integrity NAME]

register(subparsers) mounts this into the unified CLI.
"""

import json
import re
import sys

import memdag
import memdag_compact
import memdag_distill
import memdag_quarantine
import memdag_redact

# Minimal on purpose: for the ADAPTER the schema must live in the WEIGHTS.
# The RAG baseline gets RAG_SYSTEM (full format spec) instead — that asymmetry
# IS the experiment.
REFLEX_SYSTEM = "You are memdag-reflex, Matt's memory-grounded lab assistant."

# Full house-format spec, given only to the prompt-engineered baseline.
RAG_SYSTEM = (
    "You are a memory-grounded lab assistant. Answer ONLY from the retrieved "
    "memory provided in the user message. Respond in EXACTLY this format:\n"
    "Verdict: <one-line answer>\n"
    "Evidence:\n"
    "- <claim taken from retrieved memory> [mem:<id>|<channel>]\n"
    "(one bullet per claim, each citing the [mem:id|channel] tag of the "
    "retrieved block it came from; never cite an id that was not provided)\n"
    "Integrity: <INTEGRITY-NAME> (floor of <N> episodes)\n"
    "Next move: <one concrete action>\n"
    "If the retrieved memory section is '(none)', refuse: state that no live "
    "sources were retrieved and that you will not compose an unprovenanced "
    "answer. Keep the same four-section shape. Do not add any other text."
)

QUESTION_TEMPLATES = (
    "What does memory say about: {topic}?",
    "Give me the current state of: {topic}.",
    "Quick check from the DAG - {topic}?",
)

REFUSAL_TOPICS = (
    "the BGP peering config on the core router",
    "the Kubernetes cluster admission policy",
    "the Active Directory tiering model at work",
    "the iPhone MDM enrollment profile",
    "the smart-fridge firmware version",
    "the Proxmox backup retention schedule",
)

_CITE_RE = re.compile(r"\[mem:(\d+)\|([a-z-]+)\]")


def migrate(conn):
    """Redact + quarantine columns (via distill) plus archived (via compact)."""
    memdag_distill.migrate(conn)
    memdag_compact.migrate(conn)


# ---------------------------------------------------------------------------
# Consolidation detection
# ---------------------------------------------------------------------------

def consolidated_ids(conn):
    """Return ids of agent-derived nodes that are CONSOLIDATION products.

    Signature of memdag_compact.compact(): the minted node's immediate parents
    are ALL archived (archived=1) and there is at least one parent.  Plain
    `ask` derivations never archive their parents, so they are excluded.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT n.id FROM nodes n"
        " WHERE n.channel='agent-derived'"
        "   AND EXISTS (SELECT 1 FROM edges e WHERE e.child = n.id)"
        "   AND NOT EXISTS ("
        "       SELECT 1 FROM edges e JOIN nodes p ON p.id = e.parent"
        "       WHERE e.child = n.id AND p.archived = 0)"
        " ORDER BY n.id"
    ).fetchall()
    return {r[0] for r in rows}


def eligible_consolidated(conn, min_integrity=1):
    """Consolidated nodes that pass the FULL distill gate.

    Mirrors memdag_distill.export_training's belt-and-braces selection
    (same WHERE clause, same ancestor-taint CTE via the shared helper) and
    intersects with consolidated_ids().  Returns list of dicts:
    {id, content, label, n_parents}.
    """
    migrate(conn)
    floor = memdag_distill._parse_min_integrity(min_integrity)
    cids = consolidated_ids(conn)

    rows = conn.execute(
        "SELECT id, content, label FROM nodes"
        " WHERE channel='agent-derived'"
        "   AND tombstoned=0"
        "   AND redacted=0"
        "   AND status != 'quarantined'"
        "   AND label >= ?"
        " ORDER BY id",
        (floor,),
    ).fetchall()

    out = []
    for nid, content, label in rows:
        if nid not in cids:
            continue
        # Reuse the verified ancestor-taint CTE — belt and braces.
        if memdag_distill._has_live_external_ancestor(conn, nid):
            continue
        n_parents = len(memdag.parents_of(conn, nid))
        out.append({"id": nid, "content": content, "label": label,
                    "n_parents": n_parents})
    return out


# ---------------------------------------------------------------------------
# Deterministic shaping helpers (no LLM, no clock)
# ---------------------------------------------------------------------------

def _bullets(content):
    """Extract '- ' bullet lines from a consolidated node's content."""
    out = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("- ") and len(line) > 4:
            out.append(line[2:].strip())
    return out


def topic_of(content, width=60):
    """Deterministic short topic string: first bullet (or first line), squashed."""
    bs = _bullets(content)
    src = bs[0] if bs else next(
        (l.strip() for l in content.splitlines() if l.strip()), "")
    src = " ".join(src.split()).rstrip(".")
    return src[:width].rstrip()


def format_context(items):
    """Render retrieved-memory blocks: [mem:id|channel|INTEGRITY] content."""
    if not items:
        return "(none)"
    blocks = []
    for it in items:
        blocks.append(
            f"[mem:{it['id']}|agent-derived|{memdag.NAME[it['label']]}]"
            f"\n{it['content']}"
        )
    return "\n\n".join(blocks)


def build_house_answer(nid, content, label, n_parents):
    """House-style response, built deterministically from one consolidated node."""
    bs = _bullets(content) or [topic_of(content, 200)]
    verdict = bs[0].rstrip(".") + "."
    evidence = "\n".join(f"- {b.rstrip('.')}. [mem:{nid}|agent-derived]"
                         for b in bs[:4])
    return (
        f"Verdict: {verdict}\n"
        f"Evidence:\n{evidence}\n"
        f"Integrity: {memdag.NAME[label]} (floor of {n_parents} episodes)\n"
        f"Next move: treat as {memdag.NAME[label]} - run `memdag explain {nid}` "
        f"to walk provenance before acting on it."
    )


def build_refusal_answer():
    return (
        "Verdict: cannot answer - no live sources retrieved.\n"
        "Evidence:\n"
        "- (none) - refusing to compose an unprovenanced answer.\n"
        "Integrity: NONE (floor of 0 episodes)\n"
        "Next move: ingest or endorse a source covering this, reindex, then re-ask."
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_reflex(conn, min_integrity=1):
    """Return reflex training records (answer pairs + refusal pairs).

    Each record:
      {
        'conversations': [ {role: system|user|assistant, content: str}, ... ],
        'node_id': int or None,        # None for refusal records
        'kind': 'answer' | 'refusal',
      }
    Deterministic: same DB state -> byte-identical records.
    """
    records = []
    for item in eligible_consolidated(conn, min_integrity):
        ctx = format_context([item])
        answer = build_house_answer(item["id"], item["content"],
                                    item["label"], item["n_parents"])
        topic = topic_of(item["content"])
        for tpl in QUESTION_TEMPLATES:
            q = tpl.format(topic=topic)
            records.append({
                "conversations": [
                    {"role": "system", "content": REFLEX_SYSTEM},
                    {"role": "user",
                     "content": f"{q}\n\nRetrieved memory:\n{ctx}"},
                    {"role": "assistant", "content": answer},
                ],
                "node_id": item["id"],
                "kind": "answer",
            })
    refusal = build_refusal_answer()
    for topic in REFUSAL_TOPICS:
        q = QUESTION_TEMPLATES[0].format(topic=topic)
        records.append({
            "conversations": [
                {"role": "system", "content": REFLEX_SYSTEM},
                {"role": "user",
                 "content": f"{q}\n\nRetrieved memory:\n(none)"},
                {"role": "assistant", "content": refusal},
            ],
            "node_id": None,
            "kind": "refusal",
        })
    return records


def assert_clean(records, forbidden):
    """Raise ValueError if ANY forbidden marker appears anywhere in records.

    This is the export-time enforcement of the security invariant: poisoned /
    external / quarantined / redacted / tombstoned content must never reach
    the training JSONL.
    """
    for i, rec in enumerate(records):
        blob = json.dumps(rec, ensure_ascii=False)
        for marker in forbidden:
            if marker in blob:
                raise ValueError(
                    f"TAINT GATE FAILURE: forbidden marker {marker!r} "
                    f"found in record {i} (node_id={rec.get('node_id')})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _tainted_node_ids(conn):
    """Ids of nodes that must never be the source of a training record."""
    return {
        r[0] for r in conn.execute(
            "SELECT id FROM nodes"
            " WHERE channel = 'external' OR status = 'quarantined'"
            "    OR tombstoned = 1 OR redacted = 1"
        )
    }


def _assert_no_tainted_source(records, tainted_ids):
    """REFLEX-1 backstop: fail if any record's SOURCE node is itself tainted.

    REFLEX-NEW-1: the first attempt matched tainted *content substrings* against
    every record — false-positive-prone (clean extractive summaries legitimately
    share words with tainted nodes), turning a coincidental overlap into a total
    export DoS. Cross-checking the record's node_id against the tainted-id set is
    precise (zero false positives) and catches the real regression: a tainted node
    slipping through export_reflex's selection gate.
    """
    bad = sorted({r["node_id"] for r in records
                  if r.get("node_id") in tainted_ids})
    if bad:
        raise ValueError(
            f"TAINT GATE FAILURE: training record(s) sourced from tainted node(s) {bad}")


def cmd_export_reflex(args):
    conn = memdag.get_connection()
    try:
        records = export_reflex(conn, args.min_integrity)
        _assert_no_tainted_source(records, _tainted_node_ids(conn))  # REFLEX-1
        memdag_distill.write_jsonl(args.out, records)
        n_ans = sum(1 for r in records if r["kind"] == "answer")
        n_ref = len(records) - n_ans
        print(f"wrote {len(records)} records ({n_ans} answer, {n_ref} refusal)"
              f" -> {args.out}")
    finally:
        conn.close()


def register(subparsers):
    p = subparsers.add_parser(
        "export-reflex",
        help="export reflex/schema-shaped chat pairs from untainted consolidated memory",
    )
    p.add_argument("out", metavar="out.jsonl")
    p.add_argument("--min-integrity", default="agent-derived",
                   dest="min_integrity",
                   help="integrity floor: name or int 0-3 (default agent-derived)")
    p.set_defaults(func=cmd_export_reflex)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memdag_reflex")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
