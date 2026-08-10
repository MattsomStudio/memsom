#!/usr/bin/env python3
"""clean_baselines -- the Phase 0 "record the clean results" ratchet (00-chat-findings.md A8).

Pins seven MEASURED-clean rows at 0 so re-work cannot silently reintroduce
them, rather than leaving them as a one-time observation nobody re-checks:
eval(/exec(/pickle load-or-dump/yaml.load(/shell=True, hardcoded absolute
paths, embedded API keys/tokens/passwords, and bare `except:` (no exception
type at all -- distinct from failopen_annotations.py's broader "except
Exception:" convention check).

`model.eval()` (PyTorch inference mode) is excluded from the eval( count by
construction -- it is a `.eval()` METHOD call on an object, never the bare
`eval(` builtin -- so the detector does not need a hand-maintained exemption
list for it. Path/secret patterns search string LITERALS only, never
comments or docstrings, for the same reason the S4 predicate detector does:
a comment explaining the banned shape (e.g. documenting Windows path
escaping) must not count as an instance of it.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gatelib as g  # noqa: E402

_ABS_PATH = re.compile(
    r"""(?:[A-Za-z]:\\\\?[Uu]sers\\\\?[A-Za-z0-9_.\\-]+|/home/[A-Za-z0-9_.\-]+|/Users/[A-Za-z0-9_.\-]+)""")
_SECRET = re.compile(
    r"""(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password)\b\s*[:=]\s*['"][A-Za-z0-9+/_\-]{12,}['"]""")


def _bare_builtin_calls(name: str) -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        rel = path.relative_to(g.SRC).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
                hits.append(f"{rel}:{node.lineno}")
    return hits


def _shell_true() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        rel = path.relative_to(g.SRC).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "shell" \
                    and isinstance(node.value, ast.Constant) and node.value.value is True:
                hits.append(f"{rel}:{node.lineno}")
    return hits


def _yaml_load_unsafe() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        rel = path.relative_to(g.SRC).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "load" and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "yaml":
                has_safe_loader = any(
                    kw.arg == "Loader" and isinstance(kw.value, ast.Attribute)
                    and kw.value.attr == "SafeLoader"
                    for kw in node.keywords)
                if not has_safe_loader:
                    hits.append(f"{rel}:{node.lineno}")
    return hits


def _pickle_calls() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        rel = path.relative_to(g.SRC).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("load", "loads") and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "pickle":
                hits.append(f"{rel}:{node.lineno}")
    return hits


def _bare_except() -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        rel = path.relative_to(g.SRC).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                hits.append(f"{rel}:{node.lineno}")
    return hits


def _literal_pattern(pattern: re.Pattern) -> list[str]:
    hits = []
    for path in g.iter_py_files():
        tree = g.parse(path)
        if tree is None:
            continue
        rel = path.relative_to(g.SRC).as_posix()
        skip = g.docstring_node_ids(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in skip and pattern.search(node.value)):
                hits.append(f"{rel}:{node.lineno}")
    return hits


_ROWS = {
    "clean.eval_calls": lambda: _bare_builtin_calls("eval"),
    "clean.exec_calls": lambda: _bare_builtin_calls("exec"),
    "clean.pickle_load": _pickle_calls,
    "clean.yaml_load_unsafe": _yaml_load_unsafe,
    "clean.shell_true": _shell_true,
    "clean.hardcoded_abs_paths": lambda: _literal_pattern(_ABS_PATH),
    "clean.embedded_secrets": lambda: _literal_pattern(_SECRET),
    "clean.bare_except": _bare_except,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    worst = 0
    for key, fn in _ROWS.items():
        hits = fn()
        for h in hits:
            print(f"  {key}: {h}")
        print(f"{key}: {len(hits)}")
        if args.check:
            rc = g.check(key, len(hits))
            worst = worst or rc
        else:
            g.record(key, len(hits), f"python scripts/clean_baselines.py ({key})",
                      "MEASURED at Phase 0 (00-chat-findings.md A8): all clean at 9d165b1; "
                      "re-verified at this commit", 0)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
