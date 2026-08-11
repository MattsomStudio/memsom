#!/usr/bin/env python3
"""memsom_doctor — one command that dumps everything needed for a bug report.

WRAPS the server's --selfcheck (it already exercises initialize + tools/list +
a tools/call) and ADDS the environment facts selfcheck doesn't cover: OS/arch,
Python version+path, memsom version, DB location + node count, and an Ollama
reachability probe. Never raises on a dead Ollama or a fresh/locked DB.
"""

import json
import os
import platform
import sys
from pathlib import Path
from urllib.parse import urlsplit

import memsom
from memsom.effects import net as memsom_net
from memsom.effects import proc as memsom_proc


def _version():
    try:
        import importlib.metadata as im
        return im.version("memsom")
    except Exception:
        return "unknown (not installed as a package)"  # metadata lookup failed — best-effort label


def _node_count():
    # Read-only: memsom.get_connection(read_only=True) is the one other
    # connection shape the package needs (Phase 5, effects) -- it carries the
    # same busy_timeout memsom's rollback-journal (not WAL) mode needs, so a
    # concurrent client writer cannot make this raise 'database is locked'.
    db = Path(memsom.db_path())
    if not db.exists():
        return None
    try:
        conn = memsom.get_connection(read_only=True)
        try:
            return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        finally:
            conn.close()
    except Exception:
        return None  # DB missing/locked/corrupt -- node count unknown, not fatal


def _ollama_status():
    embed_url = os.environ.get("MEMDAG_EMBED_URL") or "http://127.0.0.1:11434/api/embeddings"
    parts = urlsplit(embed_url)
    root = f"{parts.scheme}://{parts.netloc}"
    model = os.environ.get("MEMDAG_EMBED_MODEL") or "nomic-embed-text"
    try:
        body = memsom_net.fetch(root + "/api/tags", timeout=3)
        data = json.loads(body)
        names = [m.get("name", "") for m in data.get("models", [])]
        return {"reachable": True, "model": model,
                "model_present": any(model in n for n in names)}
    except Exception as exc:
        return {"reachable": False, "model": model, "error": str(exc)}


def _selfcheck():
    try:
        # encoding pinned: text=True alone decodes with the locale codec (cp1252
        # on Windows), which crashes on the child's UTF-8 output (e.g. a ✓/⚠).
        r = memsom_proc.run([sys.executable, "-m", "memsom.interface.mcp", "--selfcheck"],
                            capture_output=True, text=True, timeout=30,
                            encoding="utf-8", errors="replace")
        return {"returncode": r.returncode, "output": (r.stdout + r.stderr).strip()}
    except Exception as exc:
        return {"returncode": None, "output": f"selfcheck failed to run: {exc}"}


def gather():
    return {
        "memsom_version": _version(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "db_path": str(memsom.db_path()),
        "node_count": _node_count(),
        "ollama": _ollama_status(),
        "selfcheck": _selfcheck(),
    }


def _format(report):
    o = report["ollama"]
    if o["reachable"]:
        ol = f"reachable; {o['model']} {'present' if o['model_present'] else 'NOT pulled'}"
    else:
        ol = f"unreachable ({o.get('error','')})"
    nodes = report["node_count"]
    nodes = "(DB not initialized)" if nodes is None else nodes
    lines = [
        "memsom doctor",
        "=============",
        f"memsom version : {report['memsom_version']}",
        f"OS             : {report['os']} ({report['arch']})",
        f"Python         : {report['python_version']}  [{report['python_executable']}]",
        f"DB path        : {report['db_path']}",
        f"nodes          : {nodes}",
        f"ollama         : {ol}",
        "",
        f"selfcheck (rc={report['selfcheck']['returncode']}):",
        report["selfcheck"]["output"] or "(no output)",
    ]
    return "\n".join(lines)


def register(subparsers):
    p = subparsers.add_parser("doctor",
                              help="print a paste-ready diagnostic report for bug reports")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_doctor)


def cmd_doctor(args):
    report = gather()
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
    else:
        print(_format(report))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="memsom-doctor")
    ap.add_argument("--json", action="store_true")
    cmd_doctor(ap.parse_args())
