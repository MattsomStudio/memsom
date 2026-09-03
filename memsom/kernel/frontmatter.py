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


# --- harness-nested frontmatter (Claude Code's memory stamper) ---------------
#
# Claude Code (measured on 2.1.259, `stampNewMemoryContent`) rewrites every
# memory .md it Writes/Edits that carries no `originSessionId` into
#   name / description / metadata: { node_type: memory, ...every other key...,
#   originSessionId, modified }
# so a flat memsom file comes back with its contract keys one level down, where
# fm_top_level() (deliberately) cannot see them.  These helpers read that shape
# as flat and give writers a flat line list to write back.  fm_top_level itself
# is unchanged: 14 other call sites and the forget parity tests depend on it.

_FM_META_RE = re.compile(r"^metadata\s*:\s*(.*)\Z")           # \Z not $ (F-16)
_FM_CHILD_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s?(.*)\Z")   # exactly one 2-space level
_FM_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:")


def fm_is_nested(fm_lines) -> bool:
    """True when the frontmatter carries a top-level ``metadata:`` line."""
    return any(_FM_META_RE.match(ln) for ln in fm_lines)


def fm_flatten_lines(fm_lines) -> list:
    """Lift the children of a top-level ``metadata:`` block to the top level.

    Each 2-space-indented ``key: value`` child is dedented VERBATIM (quoting
    kept, never re-serialised) in place of the ``metadata:`` line; on a key
    clash the existing top-level line wins and the child is dropped.  Fails
    CLOSED — returns the lines unchanged — on anything that is not exactly one
    level of scalar children (a deeper block, a list item, a comment, a tab, an
    inline ``metadata: {...}`` value), so a shape this parser does not model is
    never half-rewritten.  Idempotent on flat input (no ``metadata:`` line).
    """
    lines = list(fm_lines)
    start = next((i for i, ln in enumerate(lines) if _FM_META_RE.match(ln)), None)
    if start is None:
        return lines
    if _FM_META_RE.match(lines[start]).group(1).strip():
        return lines                       # inline mapping/scalar: not modelled
    end = start + 1
    children = []
    while end < len(lines) and (lines[end][:1] in (" ", "\t")):
        m = _FM_CHILD_RE.match(lines[end])
        if not m or lines[end].startswith("   "):
            return lines                   # deeper / list / comment / tab -> fail closed
        children.append((m.group(1), lines[end][2:]))
        end += 1
    if any(_FM_META_RE.match(ln) for ln in lines[end:]):
        return lines                       # two metadata: blocks: not modelled
    top_keys = set()
    for ln in lines[:start] + lines[end:]:
        m = _FM_KEY_RE.match(ln)
        if m:
            top_keys.add(m.group(1))
    lifted, seen = [], set()
    for key, raw in children:
        if key in top_keys or key in seen:
            continue                       # top-level wins; first child wins
        seen.add(key)
        lifted.append(raw)
    return lines[:start] + lifted + lines[end:]


def fm_flat(fm_lines) -> dict:
    """fm_top_level() over the flattened lines: nested-as-flat for readers."""
    return fm_top_level(fm_flatten_lines(fm_lines))


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

# --- MEMORY.md primary-index / section-map parsing -------------------------
# Moved out of bridge/bridge_import.py (Phase 7 -- these are pure text
# parsers over an already-read MEMORY.md string, no bridge-specific state,
# and distill/digest.py + interface/audit.py both need them without being
# able to reach up into bridge/ (rank 7) from distill's rank 5.

_PRIMARY_RE = re.compile(
    r"^\s*[-*]\s*\[([^\]]+)\]\(([^)]+\.md)\)(?:\s*[—–-]\s*(.+\S))?\s*$")


_HOOK_RE = re.compile(r"\]\(([^)]+\.md)\)\s*[—–-]\s*(.+\S)")


def index_hooks(memory_md_text: str) -> dict:
    """Map each linked filename -> its hand-curated hook (text after the em dash)."""
    out = {}
    for line in memory_md_text.split("\n"):
        m = _HOOK_RE.search(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _strip_render_marker(hook):
    """Drop a render-time staleness marker (' ⚠ ...') from a captured hook.

    The digest appends a ' ⚠' flag to stale lines AT RENDER.  Because the next
    import re-derives each hook by parsing the previous MEMORY.md, an un-stripped
    marker round-trips into the stored hook and compounds one glyph per cycle
    (the 16x-⚠ bug).  Stripping at capture makes the hook idempotent and
    self-healing: a polluted MEMORY.md collapses back to a single flag next render.
    """
    if not hook:
        return hook
    i = hook.find("⚠")
    if i != -1:
        hook = hook[:i].rstrip()
    return hook or None


def parse_primary_index(memory_md_text: str) -> dict:
    """{filename: (title, hook_or_None, section)} for line-leading primary entries.

    The curated title + hook are captured so the digest renders byte-for-byte like
    the hand-maintained index (frontmatter name/description are longer and bloat
    the file past its budget).  Files that appear only as secondary inline links
    (or not at all) are absent -> they get no digest line, matching MEMORY.md.
    """
    out = {}
    section = None
    for line in memory_md_text.split("\n"):
        h = re.match(r"^##\s+(.*\S)\s*$", line)
        if h:
            section = h.group(1).strip()
            continue
        if section is None:
            continue
        m = _PRIMARY_RE.match(line)
        if m:
            out[m.group(2)] = (m.group(1).strip(),
                               _strip_render_marker((m.group(3) or "").strip()),
                               section)
    return out


# --- MEMORY.md section map ----------------------------------------------------

_LINK_IN_LINE = re.compile(r"\]\(([^)]+\.md)\)")


def section_map(memory_md_text: str) -> dict:
    """Map each linked filename -> its `## Section` header in MEMORY.md."""
    out = {}
    current = None
    for line in memory_md_text.split("\n"):
        h = re.match(r"^##\s+(.*\S)\s*$", line)
        if h:
            current = h.group(1).strip()
            continue
        for m in _LINK_IN_LINE.finditer(line):
            out[m.group(1)] = current
    return out


def parse_index_entries(memory_md_text: str):
    """Yield (section, kind, payload) for every index line, in document order.

    kind 'file'    -> payload = linked filename (one yield per link on the line).
    kind 'literal' -> payload = the raw bullet line text (a hand-authored index
                      entry with no file behind it, e.g. the identity lead line or
                      the dated progress-check reminder).
    Section headers, the H1, and blank lines are skipped.
    """
    section = None
    for line in memory_md_text.split("\n"):
        h = re.match(r"^##\s+(.*\S)\s*$", line)
        if h:
            section = h.group(1).strip()
            continue
        if section is None:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        files = _LINK_IN_LINE.findall(line)
        if files:
            for f in files:
                yield (section, "file", f)
        elif stripped[0] in "-*⏰•":  # bullet / ⏰ / •  -> a literal entry
            yield (section, "literal", line.rstrip())
