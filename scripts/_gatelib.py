"""Shared AST-walk and baseline-JSON helpers for the Phase-0 gate/ratchet scripts.

Every script in this package is a static count over memsom/ plus a baseline
comparison, so the walk (skip docstrings, skip tests/, skip __pycache__) and
the "record vs check" JSON dance are written once here instead of seven times.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "memsom"
BASELINE_FILE = REPO / "_meta" / "measurements" / "baselines.json"


def iter_py_files(root: Path = SRC):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return None


def docstring_node_ids(tree: ast.AST) -> set[int]:
    """id() of every Constant that ast.get_docstring's definition covers."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.add(id(first.value))
    return out


def load_baselines() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return {}


def record(key: str, value, source: str, control: str, target, note: str | None = None) -> None:
    """Write (or overwrite) one ratchet row. Called with no --check: 'record fresh'."""
    doc = load_baselines()
    row = {"value": value, "source": source, "control": control, "target": target}
    if note:
        row["note"] = note
    doc[key] = row
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(key: str, value) -> int:
    """Compare `value` to the recorded baseline. Returns a process exit code.

    A ratchet may only IMPROVE (fall) or hold; a rise is the regression this
    gate exists to catch. Missing baseline is a hard error: `record` must run
    at least once (Phase 0) before `--check` means anything.
    """
    doc = load_baselines()
    if key not in doc:
        print(f"NO BASELINE for {key!r} -- run this script without --check first "
              f"(Phase 0 records it once)")
        return 2
    baseline = doc[key]["value"]
    if isinstance(baseline, (int, float)) and isinstance(value, (int, float)):
        if value > baseline:
            print(f"REGRESSION {key}: {value} > baseline {baseline}")
            return 1
        print(f"OK {key}: {value} (baseline {baseline})")
        return 0
    if value != baseline:
        print(f"REGRESSION {key}: {value!r} != baseline {baseline!r}")
        return 1
    print(f"OK {key}: {value!r}")
    return 0
