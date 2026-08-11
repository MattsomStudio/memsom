"""memsom.kernel.chunking -- pure text-splitting + hashing for the ingest write path.

Moved out of interface/ingest.py (Phase 4, PLAN.md Sec1.4/Sec1.5): these three
functions touch no DB and no I/O, so they belong at rank 0 alongside the rest
of kernel/. The write path itself (integrity/ingest.py, rank 2) imports them.
"""

import hashlib
import re


def normalize(text: str) -> str:
    """Collapse all whitespace runs to a single space; strip leading/trailing.

    Ensures semantically identical chunks produce the same hash regardless of
    whitespace noise.
    """
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    """Return hex-encoded SHA-256 of the normalized text."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def split_chunks(text: str, chunk_chars: int) -> list:
    """Split *text* into chunks of at most *chunk_chars* characters each.

    Splitting strategy (in order of preference):
      1. Double-newline (paragraph boundary) -- keeps semantic units together.
      2. Single newline.
      3. '. ' (sentence boundary).
      4. Whitespace boundary nearest to chunk_chars (avoids mid-word cuts).
      5. Hard split at chunk_chars (absolute last resort when no whitespace found).

    Guarantees: every yielded chunk is non-empty; no content is dropped
    (all characters from the input appear in some chunk, in order).
    """
    # INGEST-4: chunk_chars <= 0 makes the slice window never shrink `remaining`,
    # spinning forever while holding the connection. Reject it at the boundary.
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be >= 1")
    if len(text) <= chunk_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks = []
    remaining = text

    while len(remaining) > chunk_chars:
        window = remaining[:chunk_chars]

        # 1. Try paragraph break
        pos = window.rfind("\n\n")
        if pos > 0:
            cut = pos + 2
        else:
            # 2. Try single newline
            pos = window.rfind("\n")
            if pos > 0:
                cut = pos + 1
            else:
                # 3. Try sentence boundary
                pos = window.rfind(". ")
                if pos > 0:
                    cut = pos + 2
                else:
                    # 4. Whitespace nearest to chunk_chars (avoid mid-word cut)
                    pos = window.rfind(" ")
                    if pos > 0:
                        cut = pos + 1  # include the space in the consumed slice
                    else:
                        # 5. True hard cut (no whitespace at all in window)
                        cut = chunk_chars

        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:]

    tail = remaining.strip()
    if tail:
        chunks.append(tail)

    return chunks
