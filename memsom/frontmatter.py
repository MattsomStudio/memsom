"""Frontmatter parsing — light, stdlib, and a LEAF.

Lives at the package top level beside ``memsom.paths`` and ``memsom.childenv``
for the same reason they do: it is shared by every layer (bridge, distill,
integrity, retrieval) and depends on nothing, so it must not live INSIDE a
layer. It used to be defined in ``memsom.bridge.bridge_import``, which made
``retrieval.warm`` reach UP into the bridge for two pure string functions
(memsom-layers violation, 2026-08-20). ``bridge_import`` re-exports both names
so its existing importers are unaffected.
"""

from __future__ import annotations

import re


def split_frontmatter(text: str):
    """Return (fm_lines, body, had_fm).

    fm_lines is the raw list of lines between the opening and closing '---'
    fences (exclusive).  If there is no frontmatter, returns ([], text, False).
    """
    if not text.startswith("---"):
        return [], text, False
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], "\n".join(lines[i + 1:]), True
    return [], text, False  # unterminated fence -> treat as no frontmatter


def fm_top_level(fm_lines) -> dict:
    """Parse top-level (non-indented) `key: value` pairs from frontmatter lines.

    Nested blocks (e.g. an indented `metadata:` child) are ignored — we only
    need the flat keys (type, salience, pin, name, description).
    """
    out = {}
    for ln in fm_lines:
        if not ln or ln[0] in " \t#":  # skip indented children + comments
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):\s?(.*)$", ln)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out
