#!/usr/bin/env python3
"""journal_mode_contention -- PLAN.md Sec3 Phase 10: "Journal mode must be
decided here, by measurement, before serve.py ships."

MEASURED (lifecycle/doctor.py, unchanged by this phase): memsom's store runs
in rollback-journal mode (SQLite's default -- no journal_mode PRAGMA is set
anywhere in storage/db.py), already shared live by the panel task, the MCP
server, Stop hooks, and the weekly enforcing sweep. A writer blocks all
readers in that mode, and interface/serve.py (Phase 10) adds a threaded HTTP
server writing the same file -- a remote client's read can now be blocked by
a local `bridge-render`. WAL removes that specific block but adds -wal/-shm
sidecars that interact with syncguard (kernel/syncguard.py, Sec3.4) and any
backup script, and behaves badly on network filesystems.

THIS SCRIPT is the required evidence artifact, not a substitute for it: it
runs a SYNTHETIC N-reader/1-writer contention benchmark against a throwaway
copy of memsom's own schema, because this environment has no real
four-writer production load to measure against. PLAN.md is explicit that
"the deciding measurement is contention under real four-writer load, not a
synthetic benchmark" -- re-run this against a COPY of the live store before
serve.py is actually deployed, and treat that re-run's numbers, not this
synthetic one, as the final word. What this run DOES settle honestly: the
synthetic numbers, labelled as synthetic, recorded once per Sec3's own
instruction ("record the result in _meta/measurements/, treat the outcome as
a Phase 10 deliverable").

USAGE
  python _meta/tools/journal_mode_contention.py --write   # run + record once
  python _meta/tools/journal_mode_contention.py           # run + print only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "measurements" / "journal-mode-decision.json"

sys.path.insert(0, str(REPO))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  content       TEXT    NOT NULL,
  channel       TEXT    NOT NULL,
  label         INTEGER NOT NULL,
  source_ref    TEXT,
  created_at    TEXT    NOT NULL,
  tombstoned    INTEGER NOT NULL DEFAULT 0,
  tombstoned_at TEXT,
  revoke_reason TEXT
);
"""

READERS = 4
DURATION_S = 1.5


def _bench(db_path: Path, journal_mode: str) -> dict:
    setup = sqlite3.connect(db_path)
    setup.execute(f"PRAGMA journal_mode={journal_mode}")
    setup.executescript(_SCHEMA)
    setup.commit()
    setup.close()

    stop = threading.Event()
    reads_done = [0] * READERS
    read_errors = [0] * READERS
    write_errors = [0]
    writes_done = [0]

    def writer():
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA busy_timeout = 5000")
        i = 0
        while not stop.is_set():
            try:
                conn.execute(
                    "INSERT INTO nodes(content, channel, label, created_at) "
                    "VALUES (?,?,?,?)", (f"row{i}", "user", 2, "now"))
                conn.commit()
                writes_done[0] += 1
            except sqlite3.OperationalError:
                write_errors[0] += 1
            i += 1
        conn.close()

    def reader(idx):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA busy_timeout = 5000")
        while not stop.is_set():
            try:
                conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
                reads_done[idx] += 1
            except sqlite3.OperationalError:
                read_errors[idx] += 1
        conn.close()

    threads = [threading.Thread(target=writer)]
    threads += [threading.Thread(target=reader, args=(i,)) for i in range(READERS)]
    for t in threads:
        t.start()
    time.sleep(DURATION_S)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    return {
        "journal_mode": journal_mode,
        "duration_s": DURATION_S,
        "writes": writes_done[0],
        "write_errors": write_errors[0],
        "reads_total": sum(reads_done),
        "reads_per_reader": reads_done,
        "read_errors_total": sum(read_errors),
        "reads_per_second": round(sum(reads_done) / DURATION_S, 1),
    }


def run() -> dict:
    results = {}
    for mode in ("delete", "wal"):  # "delete" = SQLite's rollback-journal default
        with tempfile.TemporaryDirectory(prefix="memsom_journal_bench_") as td:
            db_path = Path(td) / "bench.db"
            results[mode] = _bench(db_path, mode)

    rollback = results["delete"]
    wal = results["wal"]
    wal_faster = wal["reads_per_second"] > rollback["reads_per_second"] * 1.1
    decision = "wal" if wal_faster else "delete (rollback-journal, unchanged)"

    return {
        "kind": "SYNTHETIC benchmark -- see module docstring; not a substitute "
                "for the real four-writer measurement PLAN.md Sec3 requires "
                "before serve.py is deployed",
        "results": results,
        "decision": decision,
        "rationale": (
            f"rollback-journal: {rollback['reads_per_second']}/s reads under 1 "
            f"concurrent writer ({rollback['read_errors_total']} lock errors); "
            f"WAL: {wal['reads_per_second']}/s reads ({wal['read_errors_total']} "
            f"lock errors). " +
            ("WAL cleared the >10% throughput bar under this synthetic load, so "
             "it is the recommendation -- re-verify against the real store "
             "before flipping storage/db.py's journal_mode."
             if wal_faster else
             "WAL did not clear the >10% throughput bar under this synthetic "
             "load; keep the current default (no journal_mode PRAGMA = "
             "rollback-journal) until a real four-writer measurement says "
             "otherwise -- WAL's -wal/-shm sidecars are extra surface for "
             "kernel.syncguard (Sec3.4) and any backup script for a gain that "
             "did not show up here.")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="record the decision + evidence to _meta/measurements/")
    args = ap.parse_args()

    report = run()
    print(json.dumps(report, indent=2))

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"\nwrote {OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
