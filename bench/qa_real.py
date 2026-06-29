"""qa_real — end-to-end recall test on a real personal corpus with Claude as the
consumer. The LongMemEval benchmark's utility ceiling was the weak local
synthesizer (qwen2.5:7b); the real system feeds Claude. This harness does the
machine half: sample real sessions, generate a grounded question+gold-answer from
each, then run the REAL recall_bge over the REAL vec_bge.db to retrieve context.
Claude (the actual deployed consumer) answers from that context; scoring is then
done against gold. No synthesizer proxy — the real consumer is in the loop.

Run on the PC:  python qa_real.py --n 20 --out C:\\Users\\you\\qa_real.json
"""
from __future__ import annotations
import argparse, json, sqlite3, sys, urllib.request
from pathlib import Path

EP = Path.home() / ".claude" / "episodic"
sys.path.insert(0, str(EP))
import sqlite_vec                       # noqa: E402
import bge_common                       # noqa: E402

SESS_DB = EP / "sessions.db"
VEC = EP / "vec_bge.db"
GEN_MODEL = "qwen2.5:7b-instruct"
GEN_URL = "http://localhost:11434/api/chat"


def chat(prompt, model=GEN_MODEL, url=GEN_URL, timeout=120):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (json.loads(r.read()).get("message", {}).get("content", "") or "").strip()


def gen_qa(excerpt):
    """Generate one specific factual Q + its short gold answer grounded in the excerpt."""
    p = ("Below is an excerpt from a past conversation. Write ONE specific, factual "
         "question whose answer is stated in the excerpt, and give the short answer. "
         "The question must be answerable ONLY from this excerpt's content (not generic). "
         "Reply EXACTLY as:\nQ: <question>\nA: <short answer>\n\nExcerpt:\n" + excerpt[:3500])
    out = chat(p)
    q = a = ""
    for line in out.splitlines():
        s = line.strip()
        if s.upper().startswith("Q:"):
            q = s[2:].strip()
        elif s.upper().startswith("A:"):
            a = s[2:].strip()
    return q, a


def retrieve_chunks(query, k=6):
    """Top-k chunk texts from the REAL vec_bge.db via dense MATCH (the system's index)."""
    reps = bge_common.embed_safe([query])
    if reps is None:
        return [], []
    qd = reps["dense"][0]
    db = sqlite3.connect(VEC)
    db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False)
    rows = db.execute(
        "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (sqlite_vec.serialize_float32(list(qd)), k)).fetchall()
    ids = [r[0] for r in rows]
    if not ids:
        db.close(); return [], []
    qm = ",".join("?" * len(ids))
    id2 = {i: (sid, txt) for i, sid, txt in
           db.execute(f"SELECT id, session_id, text FROM chunks WHERE id IN ({qm})", ids)}
    db.close()
    texts = [id2[i][1] for i in ids if i in id2]
    sids = [id2[i][0] for i in ids if i in id2]
    return texts, sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    src = sqlite3.connect(SESS_DB)
    # substantive, full sessions; deterministic order; skip tiny/prompts-only
    rows = src.execute(
        "SELECT session_id, transcript FROM sessions "
        "WHERE summary='' AND transcript IS NOT NULL AND length(transcript) > 1500 "
        "ORDER BY session_id LIMIT ?", (args.n * 3,)).fetchall()
    src.close()

    out = []
    for sid, tr in rows:
        if len(out) >= args.n:
            break
        q, gold = gen_qa(tr)
        if not q or not gold or len(gold) > 200:
            continue
        ctx, sids = retrieve_chunks(q, args.k)
        if not ctx:
            continue
        out.append({"source_sid": sid, "question": q, "gold": gold,
                    "retrieved_sids": sids, "source_in_topk": sid in sids,
                    "context": ctx})
        print(f"   [{len(out)}/{args.n}] src_in_topk={sid in sids}  q={q[:60]}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    hit = sum(1 for r in out if r["source_in_topk"])
    print(f"\n[qa_real] generated {len(out)} items | source session retrieved in top-{args.k}: "
          f"{hit}/{len(out)} ({hit/len(out):.2f})" if out else "[qa_real] no items", file=sys.stderr)
    print(f"[qa_real] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
