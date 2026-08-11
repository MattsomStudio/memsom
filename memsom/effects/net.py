"""memsom.effects.net -- outbound network calls.

fetch_external moved out of memsom/__init__.py (Phase 2, the core split).
Phase 5 (the effects layer, charter R1) finishes the job: every other
urllib.request call in the package (Ollama's /api/generate and /api/embeddings,
the Qwen code-embedder's /v1/embeddings and /health, ingest_url's plain GET)
routes through fetch() here, so this is the package's only urllib importer.
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent  # memsom/ package root (ships with the wheel)
EXT_URL = "https://raw.githubusercontent.com/sqlite/sqlite/master/README.md"
FALLBACK = HOME / "external_fallback.txt"


class NetworkError(Exception):
    """Wraps any urllib failure from fetch(). `code` is the HTTP status when
    the server itself replied with an error (HTTPError); None for a
    connection-level failure (timeout, DNS, refused, etc.)."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def fetch(url, *, data=None, headers=None, timeout=10, method=None) -> bytes:
    """One urllib.request round trip. Returns the response body on success;
    raises NetworkError on any failure (connection or HTTP error status)."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        raise NetworkError(str(err), code=err.code) from err
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        raise NetworkError(str(err)) from err


def fetch_external(offline):
    if not offline:
        try:
            body = fetch(EXT_URL, headers={"User-Agent": "memsom/0.1"}, timeout=10)
            return body.decode("utf-8", "replace"), f"{EXT_URL} (fetched, stored)"
        except NetworkError as err:
            print(f"[memsom] live fetch failed ({err}); using stored fallback", file=sys.stderr)
    return FALLBACK.read_text(encoding="utf-8", errors="replace"), f"{EXT_URL} (local snapshot)"
