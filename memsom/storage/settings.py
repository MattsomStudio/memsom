"""memsom.storage.settings -- `~/.memdag/memsom.json`, the deployment-mode config
`memsom setup` writes (PLAN.md Sec3.3/3.4).

Separate from memsom.tuning on purpose: tuning.py's registry holds the
runtime knobs (env-sourced, with `<store dir>/tuning.json` as their persisted
override -- `memsom tuning set`). This is the OTHER small, file-backed,
operator-facing state -- deployment mode, the syncguard
acknowledgement, remote client config -- written once by `setup` and read at
connection/serve time. No caching: callers are infrequent (setup, doctor,
get_connection, serve, features) so a fresh read every time is simpler than a
staleness story.
"""

from __future__ import annotations

import json
from pathlib import Path


def settings_path(data_dir) -> Path:
    return Path(data_dir) / "memsom.json"


def load_settings(data_dir) -> dict:
    """Malformed or absent -> {} (never raises; a broken config file must not
    crash every DB open)."""
    p = settings_path(data_dir)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data_dir, settings: dict) -> Path:
    p = settings_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return p
