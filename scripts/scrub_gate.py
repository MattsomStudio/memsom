#!/usr/bin/env python3
"""scrub_gate — fail the build if any author-identifying data is present in the tree.

This is the last line of defence before the repo is handed to anyone else. It scans
tracked text files for tokens that identify the author or their homelab. It does NOT
ban generic open-source product names (e.g. "nebula", "sqlite") — those carry no
private information.

The tokens themselves are NOT stored here in plaintext — only their SHA-256 digests
(plus each token's length, for windowed matching). That way this file, which ships in
the repo, does not itself publish the author's username / subnet that it exists to
keep out. Detection is still exact, case-insensitive substring matching: for every
known token length we hash each window of that length and compare to the digest set.

  Caveat: short, low-entropy tokens (e.g. a 5-char username) are brute-forceable from
  their digest. Hashing stops casual `grep` and automated secret-scrapers; it is not a
  cryptographic vault. The real guarantee is that the plaintext appears nowhere in the
  shipped tree.

Usage:  python scripts/scrub_gate.py [root]      (default root = repo root)
Exit:   0 = clean, 1 = at least one token found (offending file:line printed).

Importable: scan(root) -> list[(path, lineno, token, line)]
            scan_text(text) -> list[str]   (matched substrings in an arbitrary blob)
"""

import hashlib
import subprocess
import sys
from pathlib import Path

# sha256(token.lower()) -> token length. Identifying data only. The plaintext is
# deliberately absent (see module docstring). NOT included by design: "nebula",
# "lighthouse", "mesh" (public OSS terms), "MattsomStudio" (the public repo handle).
_TOKEN_HASHES = {
    "d8859979d792fa828435ada10fbeb1c231505ddaac61a5703c7b6db160d6f1ee": 5,   # author username / home-path component
    "47f6cd3e1f669265f02ceab24471d9d8dc9b176fb866714060e3415edf18f91c": 11,  # author overlay subnet prefix
    "253c17d442a43ec947c6f49924a172fa959de7b97743004a803a4d29d20ea33b": 6,   # author host-config filename
    "de720e2fb7830331d71e62fcf1288af9c38f8f1c1db3de63d81b3975e6906275": 8,   # author trust-policy event term
    "22572215d9c5133791b2bd3b53e75f9441c9964b939906d3bd0851cd61b97e71": 14,  # author vault note filename
    "7a03b16c38a15b2133513a1b60a1d6fa7ee360ff549fac24b4d86415d8224d90": 14,  # author vault tree (fwd-slash)
    "64f1133ed55f365e7054da407e814a304ef7c11cc5d1d9a14cfe956628499e3f": 14,  # author vault tree (back-slash)
    "ecccfb45f0794e7b2ed874cfef40cfc34c5071a21d5f976c398fb5cecb2358b8": 28,  # author verbatim cert-policy phrase
}

# digests grouped by token length, for windowed matching
_HASHES_BY_LEN = {}
for _h, _L in _TOKEN_HASHES.items():
    _HASHES_BY_LEN.setdefault(_L, set()).add(_h)
_LENGTHS = sorted(_HASHES_BY_LEN)

# Only scan text-like files; skip binaries, the DB, build/scratch dirs, backups.
# (.bat/.cmd matter: a Windows batch file leaked the author's username once and the
# gate missed it because .bat was absent here — never trust a suffix allow-list to be
# complete; this is reviewed against the tree.)
TEXT_SUFFIXES = {".py", ".txt", ".md", ".toml", ".json", ".yml", ".yaml",
                 ".cfg", ".ini", ".sh", ".ps1", ".bat", ".cmd", ".ipynb"}
TEXT_NAMES = {"LICENSE", "README"}
SKIP_DIRS = {".git", "__pycache__", "_build", "_weights", "_pkgtest_venv",
             ".pytest_cache", "venv", ".venv", "node_modules"}
SKIP_SUFFIXES = {".db", ".db-journal", ".sqlite", ".pyc", ".bak", ".tar", ".whl"}


def _hits_in_line(line):
    """Return the matched (lowercased) substrings of *line* that hash to a token."""
    low = line.lower()
    n = len(low)
    found = []
    for L in _LENGTHS:
        if L > n:
            continue
        hset = _HASHES_BY_LEN[L]
        for i in range(n - L + 1):
            w = low[i:i + L]
            if hashlib.sha256(w.encode("utf-8", "replace")).hexdigest() in hset:
                found.append(w)
    return found


def scan_text(text):
    """Return all matched leak substrings in an arbitrary blob (line by line).

    For callers that need to vet a string (seed content, README) without holding the
    plaintext token list themselves."""
    out = []
    for line in text.splitlines():
        out.extend(_hits_in_line(line))
    return out


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
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for tok in _hits_in_line(line):
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
