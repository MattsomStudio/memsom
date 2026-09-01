"""memsom.interface.dashboard -- deprecated compatibility shim.

The refactor deleted this module (A-9: it held 3 of memsom's 5 stray direct
SQLite connections and 2 subprocess sites, the HTML-dashboard renderer's
`open_file`). It exists again here ONLY so memsom_panel (memsom-agentic-os),
which still imports `memsom.interface.dashboard`, keeps working un-migrated
until it switches to `memsom.interface.telemetry` (PROMOTE-Q11-PANEL.md B3).

Import `memsom.interface.telemetry` instead. This shim is removed one release
after the panel switches (B5 step 7).
"""
import warnings

from .telemetry import build_telemetry, load_weights, default_memory_dir

warnings.warn(
    "memsom.interface.dashboard is deprecated; import memsom.interface.telemetry",
    DeprecationWarning, stacklevel=2)

__all__ = ["build_telemetry", "load_weights", "default_memory_dir"]
