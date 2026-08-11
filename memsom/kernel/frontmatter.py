"""memsom.kernel.frontmatter -- the frontmatter parsers, deduped (Phase 3).

There is exactly one fence-finding implementation: split_frontmatter(text) ->
(fm_lines, body, had_fm). Everything else that needs the frontmatter block is
built on top of it, not a second independent scanner:
  - stamp_fm() edits fm_lines directly (lossless round-trip: comments and
    indentation survive).
  - frontmatter_dict(text) -> (dict, body) calls split_frontmatter() for the
    fence/body split, then interprets those same fm_lines as the richer
    YAML-subset shape (inline + block lists). This is the parser that used
    to be duplicated verbatim between memsom.bridge.obsidian and
    memsom.lifecycle.forget; forget's copy read only flat scalar keys, so
    pointing it at the shared implementation is behaviour-identical there
    (test_memsom_forget.TestParity never touches this function, only
    compute() and its pure-math helpers).

No fixture in the test suite relies on the legacy "..." YAML end-of-document
marker as a frontmatter closer (only "---"), so frontmatter_dict's old
independent support for it is dropped along with its independent scanner.
"""

import re


# --- frontmatter parsing (light, stdlib) -------------------------------------

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


def memory_type(stem: str, fm: dict) -> str:
    """Type from frontmatter `type:` if present, else the filename prefix."""
    t = (fm.get("type") or "").strip()
    if t:
        return t
    return stem.split("_", 1)[0] if "_" in stem else stem


def stamp_fm(text: str, **kv):
    """Return *text* with the given top-level frontmatter keys set (idempotent).

    Existing top-level lines for any key in *kv* are replaced; a key whose value
    is None is dropped.  If there was no frontmatter and nothing is added, the
    text is returned unchanged.
    """
    fm_lines, body, had = split_frontmatter(text)
    keys = set(kv)
    fm_lines = [ln for ln in fm_lines if ln.split(":", 1)[0].strip() not in keys]
    added = False
    for k, v in kv.items():
        if v is not None:
            fm_lines.append(f"{k}: {v}")
            added = True
    if not had and not added:
        return text
    return "---\n" + "\n".join(fm_lines) + "\n---\n" + body


def stamp_section(text: str, section):
    """Thin wrapper: stamp only the `section:` key (kept for the unit tests)."""
    return stamp_fm(text, section=section)



# Frontmatter (YAML subset, stdlib only)
# ---------------------------------------------------------------------------

_FM_LIST_ITEM = re.compile(r"^\s*-\s+(.*)$")
_FM_KV = re.compile(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$")


def _strip_scalar(v: str):
    """Strip surrounding quotes from a YAML scalar; return the bare string."""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def frontmatter_dict(text: str):
    """Parse a leading ``---`` YAML block. Return (frontmatter_dict, body).

    Recognizes ``key: scalar``, ``key: [a, b]`` inline lists, and block lists
    (``key:`` then ``  - item`` lines). Values keep raw strings; lists -> list.
    If there is no valid frontmatter, returns ({}, text).
    Only a tiny, predictable subset — enough for tags/aliases/memsom-* keys.

    Built directly on split_frontmatter(): this function owns no fence-finding
    logic of its own, only the fm_lines -> dict interpretation.
    """
    fm_lines, body, had = split_frontmatter(text)
    if not had:
        return {}, text

    fm = {}
    cur_key = None
    for raw in fm_lines:
        if not raw.strip():
            continue
        m_item = _FM_LIST_ITEM.match(raw)
        if m_item and cur_key is not None:
            # Only an EMPTY scalar ("") preceding '- ' lines becomes a block list.
            # Promoting on any non-list value would silently discard a real scalar
            # (e.g. malformed `tags: foo` then `- bar` must not drop "foo").
            if fm.get(cur_key) == "":
                fm[cur_key] = []
            if isinstance(fm.get(cur_key), list):
                fm[cur_key].append(_strip_scalar(m_item.group(1)))
            continue
        m_kv = _FM_KV.match(raw)
        if m_kv:
            key = m_kv.group(1)
            val = m_kv.group(2).strip()
            cur_key = key
            if val == "":
                fm[key] = ""  # may become a block list on following '- ' lines
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                fm[key] = [_strip_scalar(x) for x in inner.split(",") if x.strip()] if inner else []
            else:
                fm[key] = _strip_scalar(val)

    return fm, body
