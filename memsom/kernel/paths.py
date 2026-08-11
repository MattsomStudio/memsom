"""memsom.kernel.paths -- filesystem path discovery, no DB, no upward imports.

Moved out of memsom/bridge/bridge_import.py (Phase 3): default_memory_dir was
the one bridge_import export every other package (distill, interface,
integrity) imported, which is what made bridge a subpackage every other
subpackage had a mutual edge with.
"""

import os
from pathlib import Path


def default_memory_dir():
    """Locate the live Claude memory dir without hard-coding a username.

    Override with $MEMDAG_BRIDGE_MEMORY_DIR; otherwise discover the first
    `~/.claude/projects/*/memory/MEMORY.md` (the project-dir name differs per
    machine, so it is globbed, never hard-coded)."""
    env = os.environ.get("MEMDAG_BRIDGE_MEMORY_DIR")
    if env:
        return Path(env)
    candidates = [m.parent for m in
                  (Path.home() / ".claude" / "projects").glob("*/memory/MEMORY.md")]
    if not candidates:
        # Fail LOUDLY. The old fallback returned ~/.claude/projects itself -- a
        # directory with zero .md files -- which sent the importer's reconcile
        # sweep off to tombstone every live memory node (the mass-wipe path the
        # import guard also blocks). No plausible dir means no import.
        raise FileNotFoundError(
            "no Claude memory dir found (no ~/.claude/projects/*/memory/MEMORY.md); "
            "set MEMDAG_BRIDGE_MEMORY_DIR explicitly")
    # the real brain is the memory dir with the MOST .md files; project-scoped
    # memory dirs (created by running Claude in another cwd) hold ~1 file and must
    # not be mistaken for it just because they sort first.
    return max(candidates, key=lambda d: len(list(d.glob("*.md"))))
