"""contradict_eval.py — measure the contradiction detector's precision against a
real backup brain, in OBSERVE mode (mutates only a throwaway copy).

Copies the backup DB to a temp file, runs a backfill sweep with the REAL anchored
adjudicator (observe-only), reports the false-positive count + top offenders, then
injects known contradictions and checks recall.

  python bench/contradict_eval.py <backup.db> [anchor] [threshold]

The whole point: FLAGS should be ~0 on a real brain (it has ~no true contradictions),
while the injected pair still gets caught. Tune anchor/threshold from the output.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    if len(sys.argv) > 2:
        os.environ["MEMDAG_CONTRADICT_ANCHOR"] = sys.argv[2]
    if len(sys.argv) > 3:
        os.environ["MEMDAG_CONTRADICT_NLI_THRESHOLD"] = sys.argv[3]

    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "eval.db")
    shutil.copy(src, db)
    os.environ["MEMDAG_DB"] = db
    os.environ["MEMDAG_CONTRADICT_NLI"] = "1"   # opt the semantic tier in

    import memsom
    import memsom_contradict as C
    conn = memsom.get_connection()

    print(f"anchor={C._anchor()} nli_threshold={C._nli_threshold()} "
          f"adjudicator={'ON' if C._default_adjudicator() else 'OFF'}")

    # --- precision: full backfill over the real brain, observe-only ---
    stats = C.sweep(conn, backfill=True, enforce=False)
    print(f"\n[PRECISION] flags={stats['contradictions']} over "
          f"{stats['probed']} probed (mode={stats['mode']})  <-- target ~0")
    rows = C.list_contradictions(conn)
    for r in rows:
        print(f"  old[{r['old_id']}]<-new[{r['new_id']}] {(r['reason'] or '')[:160]}")

    # --- recall: inject a clean contradiction the detector must catch ---
    a = memsom.insert_node(conn, "The active WAF for testsite-eval is Sucuri.",
                           "user", source_ref="eval:a")
    b = memsom.insert_node(conn, "The active WAF for testsite-eval is Cloudflare now.",
                           "user", source_ref="eval:b")
    conn.commit()
    got = C.detect(conn, b, candidates=[(a, "The active WAF for testsite-eval is Sucuri.")],
                   enforce=False)
    print(f"\n[RECALL] injected Sucuri->Cloudflare caught: {bool(got)}  "
          f"(marked={got})  <-- must be True")
    conn.close()
    print(f"\ntemp db: {db} (throwaway)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
