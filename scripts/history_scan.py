#!/usr/bin/env python3
"""history_scan — block a push if any COMMIT being pushed adds an author-identifying
token, even when the final tree is clean.

This closes the gap the tree-scanning scrub_gate can't see: a leak that is added in
one commit and removed (or amended away) in another still lives in history and ships
to whoever clones or fetches by SHA. The tree gate stays green; the leak is public.
This scanner reads the *diffs* of the commits a push would publish and scans their
ADDED lines with the same hashed denylist scrub_gate uses — one source of truth.

Invoked by the pre-push hook, which pipes git's per-ref stdin lines:

    <local ref> <local oid> <remote ref> <remote oid>

For each ref it computes the set of commits the remote does not yet have and scans
them. New branch (remote oid all-zero) -> everything reachable but not on any remote.
Deletion (local oid all-zero) -> nothing to scan.

Exit: 0 = clean, 1 = at least one leak token found (commit + token printed to stderr).

Standalone (for tests / manual use):
    python scripts/history_scan.py <local_oid> <remote_oid>
"""

import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scrub_gate  # noqa: E402


def _is_zero(oid: str) -> bool:
    """A git null oid is all zeros (40 hex for sha1, 64 for sha256)."""
    return len(oid) > 0 and set(oid) == {"0"}


def _run(args):
    r = subprocess.run(["git"] + args, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def _commits_in_push(local: str, remote: str):
    """The commits this push would add to the remote, oldest-first."""
    if _is_zero(local):
        return []  # branch deletion — no content published
    if _is_zero(remote):
        # New branch on the remote: publish everything reachable from local that
        # isn't already on some other remote ref. On a first-ever push --remotes is
        # empty, so this is the full reachable history (bounded, one-time).
        rev = [local, "--not", "--remotes"]
    else:
        rev = [f"{remote}..{local}"]
    out = _run(["rev-list", "--reverse"] + rev)
    return [c for c in out.split() if c]


def _added_leaks_in_commit(sha: str):
    """(token, file, line) for every leak token added by *sha*, allowlist-aware.

    Mirrors scrub_gate._ALLOW_IN: the author's name is permitted in LICENSE and
    pyproject.toml (deliberate AGPL attribution) and blocked everywhere else, so we
    don't fight the intentional credit while still catching accidental leaks."""
    # unified=0: only changed lines, no context. -M: don't explode renames.
    patch = _run(["show", sha, "--format=", "--unified=0", "-M", "--no-color"])
    hits = []
    current = None  # basename of the file the current hunk writes to
    for line in patch.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = None if target == "/dev/null" else Path(target[2:] if target.startswith("b/") else target).name
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added = line[1:]
            allow = scrub_gate._ALLOW_IN.get(current or "", frozenset())
            for tok in scrub_gate.scan_text(added):
                if allow and hashlib.sha256(
                        tok.encode("utf-8", "replace")).hexdigest() in allow:
                    continue  # intentional attribution in LICENSE / pyproject.toml
                hits.append((tok, current or "?", added.strip()))
    return hits


def scan_push(refs):
    """refs: iterable of (local_oid, remote_oid). Returns list of (sha, tok, file, line)."""
    seen = set()
    findings = []
    for local, remote in refs:
        for sha in _commits_in_push(local, remote):
            if sha in seen:
                continue
            seen.add(sha)
            for tok, f, line in _added_leaks_in_commit(sha):
                findings.append((sha, tok, f, line))
    return findings


def _refs_from_stdin():
    refs = []
    for raw in sys.stdin:
        parts = raw.split()
        if len(parts) == 4:
            _, local_oid, _, remote_oid = parts
            refs.append((local_oid, remote_oid))
    return refs


def main():
    if len(sys.argv) == 3:
        refs = [(sys.argv[1], sys.argv[2])]
    else:
        refs = _refs_from_stdin()
    if not refs:
        return 0
    try:
        findings = scan_push(refs)
    except RuntimeError as e:
        print(f"[history-scan] could not inspect commits: {e}", file=sys.stderr)
        return 1  # fail closed — a scanner that can't scan must not wave the push through
    if not findings:
        print("[history-scan] clean - no author-identifying tokens in pushed commits.")
        return 0
    print(f"[history-scan] BLOCKED - {len(findings)} leak token(s) in commits being pushed:",
          file=sys.stderr)
    for sha, tok, f, line in findings:
        print(f"  {sha[:10]}  {f}  [{tok}]  {line}", file=sys.stderr)
    print("\nThe tree may be clean, but these commits carry the leak in history.",
          file=sys.stderr)
    print("Rewrite the offending commit(s) (amend/rebase) before pushing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
