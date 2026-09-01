"""_keyfile — gated OpenAI key loader for the judge subprocess.

Reads the key from disk into os.environ ONLY. The key never appears in argv,
never gets logged, never gets printed. Any status line about the key goes through
redact() first, which masks all but the last 4 chars.

Contract:
  load_openai_key() -> bool
    * reads ~\\.claude\\episodic\\openai_key (single line)
    * if present and non-empty: sets os.environ["OPENAI_API_KEY"], returns True
    * if missing/empty: returns False, leaves env untouched (caller decides
      whether to fall back to the local qwen judge or abort)
"""
from __future__ import annotations

import os
from pathlib import Path

KEY_PATH = Path.home() / ".claude" / "episodic" / "openai_key"


def load_openai_key(path: str | Path = KEY_PATH) -> bool:
    """Load the OpenAI key from disk into os.environ. Returns True on success.

    Never prints, logs, or returns the key itself.
    """
    p = Path(path)
    try:
        if not p.exists():
            return False
        key = p.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not key:
        return False
    os.environ["OPENAI_API_KEY"] = key
    return True


def redact(secret: str | None) -> str:
    """Mask all but the last 4 chars of a secret for a safe status line."""
    if not secret:
        return "<none>"
    s = str(secret)
    if len(s) <= 4:
        return "*" * len(s)
    return "*" * (len(s) - 4) + s[-4:]
