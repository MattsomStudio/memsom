"""memsom.kernel.text -- pure text helpers for the frozen-core compose pipeline.

Moved out of memsom/__init__.py (Phase 2, the core split). No DB, no I/O, no
imports outside kernel/ -- this is rank 0.
"""

import re
from datetime import datetime, timezone

from memsom.kernel.lattice import NAME

STOP = {"how", "should", "i", "a", "an", "the", "do", "does", "what", "my",
        "to", "is", "it", "of", "for", "with", "in", "on", "and", "or"}

# Shared crude-stem width: prefix length used by stems() here and
# memsom.retrieval.retrieve.tokenize() -- keep them in lock-step or BM25 terms
# and compose()-side stems stop matching each other.
STEM_WIDTH = 6


def now_iso():
    # ISO-8601 TEXT, never datetime objects (3.12 sqlite3 adapter is deprecated)
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_date(iso):
    try:  # stored UTC (canonical); shown local so an evening take doesn't say "tomorrow"
        return datetime.fromisoformat(iso).astimezone().date().isoformat()
    except (TypeError, ValueError):
        return (iso or "")[:10]


def stems(text):
    return {w[:STEM_WIDTH] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def prose_lines(content):
    """Yield content lines that are prose: stateful skip of YAML frontmatter and
    code-fence INTERIORS (prefix checks alone let those leak), plus markdown noise."""
    lines = content.splitlines()
    in_front = bool(lines) and lines[0].strip() == "---"
    in_fence = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if in_front:
            in_front = not (i > 0 and line == "---")
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or raw.startswith("    "):
            continue
        if not line or line.startswith(("#", "|", "---", ">", "**", "![", "[!",
                                        "- [ ]", "- [x]", "- [X]")):
            continue
        yield line


def strip_furniture(line):
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # [text](url) -> text
    return line.replace("**", "").replace("`", "")


def snippet(content, width=70):
    line = strip_furniture(next(prose_lines(content), content))
    return " ".join(line.split())[:width]


def candidate_sentences(content):
    out = []
    for line in prose_lines(content):
        line = re.sub(r"^[-*]\s+", "", strip_furniture(line))
        for sent in re.split(r"(?<=[.?!])\s+", line):  # split on . ? ! (was ". " only)
            sent = sent.strip().rstrip(".:?!")
            if len(sent) >= 30:  # no upper cap: it was dropping long answer sentences
                out.append(sent + ".")
    return out


def fmt_node(node, indent=""):
    line = (f"{indent}[{node['id']}] {node['channel']}"
            f"  integrity={NAME[node['label']]}  {local_date(node['created_at'])}")
    if node["tombstoned"]:
        line += f"  [REVOKED {local_date(node['tombstoned_at'])}: {node['revoke_reason']}]"
    out = [line]
    ref = node["source_ref"] or ("(stated directly)" if node["channel"] == "user" else None)
    if ref:
        out.append(f"{indent}      {ref}")
    out.append(f'{indent}      "{snippet(node["content"])}..."')
    return out
