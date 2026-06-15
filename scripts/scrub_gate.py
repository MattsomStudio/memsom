#!/usr/bin/env python3
"""scrub_gate — fail the build if any author-identifying data is present in the tree.

This is a dumb, deliberate string match: it is the last line of defence before the
repo is handed to anyone else. It scans tracked text files for tokens that identify
the author or their homelab. It does NOT ban generic open-source product names
(e.g. "nebula", "sqlite") — those carry no private information.

Usage:  python scripts/scrub_gate.py [root]      (default root = repo root)
Exit:   0 = clean, 1 = at least one token found (offending file:line printed).

Importable: LEAK_TOKENS (list) and scan(root) -> list[(path, lineno, token, line)].
"""

import subprocess
import sys
from pathlib import Path

# Case-insensitive substring tokens. Identifying data only.
# NOT included by design: "nebula", "lighthouse", "mesh" (public OSS terms),
# "MattsomStudio" (the public repo handle friends clone from).
LEAK_TOKENS = [
    "you",                        # the author's username / home path component
    "[redacted-subnet]",                  # the author's real overlay subnet
    "[redacted]",                       # the author's host config file
    "[redacted]",                     # author's specific Nebula trust-policy event
    "[redacted]",               # the author's vault note path
    "[redacted]",               # the author's vault tree
    "[redacted]",
    "[redacted]", # the author's verbatim cert policy
]

# Only scan text-like files; skip binaries, the DB, build/scratch dirs, backups.
TEXT_SUFFIXES = {".py", ".txt", ".md", ".toml", ".json", ".yml", ".yaml",
                 ".cfg", ".ini", ".sh", ".ps1", ".ipynb"}
TEXT_NAMES = {"LICENSE", "README"}
SKIP_DIRS = {".git", "__pycache__", "_build", "_weights", "_pkgtest_venv",
             ".pytest_cache", "venv", ".venv", "node_modules"}
SKIP_SUFFIXES = {".db", ".db-journal", ".sqlite", ".pyc", ".bak", ".tar", ".whl"}


def _is_text(path: Path) -> bool:
    if path.suffix in SKIP_SUFFIXES:
        return False
    return path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES


def _tracked_files(root):
    """The files git would actually ship (tracked only). None if not a git repo.

    This is what matters: untracked/gitignored files (planning docs, _build, the
    weights scratch dir) never reach a friend who clones, so the gate must judge
    the shipped set, not the whole working tree."""
    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return [Path(root) / line for line in r.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return None


def scan(root):
    """Return a list of (path, lineno, token, line) for every token hit.

    Scans the git-tracked set when *root* is a repo (what ships); otherwise walks
    the tree (used by the planted-leak test against a temp dir)."""
    root = Path(root)
    self_path = Path(__file__).resolve()
    tracked = _tracked_files(root)
    if tracked is not None:
        candidates = tracked
    else:
        candidates = [p for p in root.rglob("*")
                      if not any(part in SKIP_DIRS for part in p.parts)]
    hits = []
    for path in candidates:
        if not path.is_file() or not _is_text(path):
            continue
        if path.resolve() == self_path:  # the gate naturally contains its own tokens
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for tok in LEAK_TOKENS:
                if tok in low:
                    hits.append((path, lineno, tok, line.strip()))
    return hits


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    hits = scan(root)
    if not hits:
        print("[scrub-gate] clean — no author-identifying tokens found.")
        return 0
    print(f"[scrub-gate] FAIL — {len(hits)} leak token(s) found:", file=sys.stderr)
    for path, lineno, tok, line in hits:
        print(f"  {path}:{lineno}  [{tok}]  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
