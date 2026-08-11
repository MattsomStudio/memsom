"""memsom.kernel.syncguard -- detect a file-sync tree ancestor (PLAN.md Sec3.4).

Pure, stdlib-only (rank 0: kernel may not import tuning upward -- see
memsom.tuning's module docstring and scripts/env_ratchet.py's exemption
list, which names this file alongside kernel/paths.py for the same reason).
Testable without a mock: every marker is a plain file/dir existence check or
a path-prefix comparison.

Two callers use `sync_markers()` identically -- `memsom setup` at config time
(interface/setup.py) and `storage.db.get_connection()` at open time -- so a
path that passes one check passes both: there is exactly one truth about
whether a directory sits inside a synced tree.

Detected file-sync clients (PLAN.md Sec3.4):
  Syncthing    .stfolder / .stignore / .stversions
  Dropbox      .dropbox / .dropbox.cache
  OneDrive     $OneDrive / $OneDriveCommercial / $OneDriveConsumer prefix match
  iCloud Drive an ancestor directory literally named "Mobile Documents"
  Google Drive .tmp.driveupload marker, or an ancestor named "DriveFS"
  (custom)     $MEMSOM_EXTRA_SYNC_MARKERS -- comma-separated marker filenames
"""

from __future__ import annotations

import os
from pathlib import Path

_SYNCTHING_MARKERS = (".stfolder", ".stignore", ".stversions")
_DROPBOX_MARKERS = (".dropbox", ".dropbox.cache")
_GDRIVE_MARKERS = (".tmp.driveupload",)
_GDRIVE_DIR_NAMES = ("DriveFS",)
_ICLOUD_DIR_NAMES = ("Mobile Documents",)
_ONEDRIVE_ENV_VARS = ("OneDrive", "OneDriveCommercial", "OneDriveConsumer")
_EXTRA_MARKERS_ENV = "MEMSOM_EXTRA_SYNC_MARKERS"


class SyncGuardError(RuntimeError):
    """Raised when a store path sits inside a detected file-sync tree."""


def _ancestors(path) -> list:
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        pass
    return [p] + list(p.parents)


def _onedrive_roots(environ) -> list:
    roots = []
    for var in _ONEDRIVE_ENV_VARS:
        val = environ.get(var)
        if not val:
            continue
        try:
            roots.append(Path(val).expanduser().resolve())
        except OSError:
            roots.append(Path(val))
    return roots


def _extra_marker_names(environ) -> list:
    raw = (environ.get(_EXTRA_MARKERS_ENV) or "").strip()
    if not raw:
        return []
    return [n.strip() for n in raw.split(",") if n.strip()]


def sync_markers(path, *, environ=None) -> list:
    """Every file-sync marker found in *path* or any ancestor. Empty = clean.

    *environ* defaults to os.environ; a test may pass a plain dict to check
    OneDrive/custom-marker detection without touching the real environment.
    """
    environ = os.environ if environ is None else environ
    found = []
    ancestors = _ancestors(path)
    extra_names = _extra_marker_names(environ)

    for anc in ancestors:
        for marker in _SYNCTHING_MARKERS:
            if (anc / marker).exists():
                found.append(f"Syncthing marker {marker} at {anc}")
        for marker in _DROPBOX_MARKERS:
            if (anc / marker).exists():
                found.append(f"Dropbox marker {marker} at {anc}")
        for marker in _GDRIVE_MARKERS:
            if (anc / marker).exists():
                found.append(f"Google Drive marker {marker} at {anc}")
        if anc.name in _GDRIVE_DIR_NAMES:
            found.append(f"Google Drive mount root ({anc.name}) at {anc}")
        if anc.name in _ICLOUD_DIR_NAMES:
            found.append(f"iCloud Drive ({anc.name}) at {anc}")
        for marker in extra_names:
            if (anc / marker).exists():
                found.append(f"custom marker ({_EXTRA_MARKERS_ENV}) {marker} at {anc}")

    target = ancestors[0]
    for root in _onedrive_roots(environ):
        try:
            target.relative_to(root)
        except ValueError:
            continue
        found.append(f"OneDrive sync root ({root}) contains {target}")

    return found
