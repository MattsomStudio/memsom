"""memsom_policy — declarative capability policy for the Gate #3 broker.

Maps a (namespaced) MCP tool name to the minimum session integrity floor
required to forward a call to it, and optionally to the floor that a successful
call's RESULT taints the session down to.

The classification is PURE POLICY LOOKUP — deterministic, declarative, owned by
the user.  No model is ever asked "is this tool dangerous"; that mirrors
memsom's core discipline of labelling by channel, never by content (memsom.py:6).

Policy file (JSON):
  {
    "default": "deny",                 # required floor for unmatched tools
    "rules": [
      {"tool": "fetch.*",        "required": "external", "taints": "external"},
      {"tool": "github.delete_*","required": "endorsed"},
      {"tool": "gmail.send_*",   "required": "user"},
      {"tool": "*.read_*",       "required": "external"}
    ]
  }

Matching is first-rule-wins over fnmatch globs (order matters — most specific
first).  `required`/`default` accept a RANK name, an int 0..3, or the specials
"allow" (always forward) / "deny" (never forward).  Internally a required floor
is an int 0..4 so the gate's uniform `session_floor >= required` comparison
expresses both specials: 0 = always allow, 4 = never satisfiable = always deny.

A tool is CONSEQUENTIAL iff its required floor > external(0): once the session
is tainted to external, every consequential tool is denied.

Pure library: no DB, no prints, no sys.exit.  Malformed policy RAISES — it never
silently degrades to allow.

Public API
----------
  load_policy(path)              -> dict   (normalized; raises on malformed/missing)
  required_floor(policy, tool)   -> int    (0..4)
  taints(policy, tool)           -> int | None
  is_consequential(policy, tool) -> bool
"""

import fnmatch
import json
from pathlib import Path

import memsom

# Sentinels outside the 0..3 RANK lattice for the always-allow / always-deny
# specials, kept comparable under the gate's `floor >= required` test.
ALLOW = 0   # any floor satisfies -> always forward
DENY = 4    # no floor 0..3 satisfies -> always deny


def _parse_floor(value) -> int:
    """Parse *value* into an int rank 0..3 (RANK name, int, or numeric string)."""
    if isinstance(value, bool):
        raise ValueError(f"bad floor: {value!r}")
    if isinstance(value, int):
        if 0 <= value <= 3:
            return value
        raise ValueError(f"bad floor: {value!r}")
    s = str(value).strip()
    if s.isdigit():
        n = int(s)
        if 0 <= n <= 3:
            return n
        raise ValueError(f"bad floor: {value!r}")
    if s.lower() in memsom.RANK:
        return memsom.RANK[s.lower()]
    raise ValueError(f"bad floor: {value!r}")


def _parse_required(value) -> int:
    """Parse a required-floor field: a floor 0..3, or 'allow'/'deny' specials.

    Returns an int 0..4 (ALLOW=0 .. DENY=4).
    """
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "allow":
            return ALLOW
        if low == "deny":
            return DENY
    return _parse_floor(value)


# ---------------------------------------------------------------------------
# Loading + normalization
# ---------------------------------------------------------------------------

def load_policy(path) -> dict:
    """Load and validate a policy file; return a normalized policy dict.

    Normalized shape:
      {"default": int 0..4,
       "rules": [{"tool": str, "required": int 0..4, "taints": int|None}, ...]}

    Raises FileNotFoundError if missing, ValueError if malformed.  Never returns
    a policy that silently permits everything on a parse error.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"policy file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed policy JSON in {p}: {exc}") from exc
    return _normalize(raw, str(p))


def _normalize(raw, where="<policy>") -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: policy must be a JSON object")

    default = _parse_required(raw.get("default", "deny"))

    rules_in = raw.get("rules", [])
    if not isinstance(rules_in, list):
        raise ValueError(f"{where}: 'rules' must be a list")

    rules = []
    for i, r in enumerate(rules_in):
        if not isinstance(r, dict):
            raise ValueError(f"{where}: rule #{i} must be an object")
        tool = r.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError(f"{where}: rule #{i} needs a non-empty string 'tool'")
        if "required" not in r:
            raise ValueError(f"{where}: rule #{i} ({tool!r}) needs 'required'")
        try:
            required = _parse_required(r["required"])
        except ValueError as exc:
            raise ValueError(f"{where}: rule #{i} ({tool!r}) bad required: {exc}") from exc
        taint = None
        if r.get("taints") is not None:
            try:
                taint = _parse_floor(r["taints"])
            except ValueError as exc:
                raise ValueError(f"{where}: rule #{i} ({tool!r}) bad taints: {exc}") from exc
        rules.append({"tool": tool, "required": required, "taints": taint})

    return {"default": default, "rules": rules}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _match(policy, tool):
    """Return the first rule whose glob matches *tool*, or None."""
    for rule in policy.get("rules", []):
        if fnmatch.fnmatchcase(tool, rule["tool"]):
            return rule
    return None


def required_floor(policy, tool) -> int:
    """Minimum session floor (int 0..4) required to forward a call to *tool*."""
    rule = _match(policy, tool)
    if rule is not None:
        return rule["required"]
    return policy.get("default", DENY)


def taints(policy, tool):
    """Floor (int 0..3) a successful *tool* result taints the session to, or None."""
    rule = _match(policy, tool)
    if rule is not None:
        return rule["taints"]
    return None


def is_consequential(policy, tool) -> bool:
    """True if *tool* needs more than external(0) — i.e. an external-tainted
    session would be denied it."""
    return required_floor(policy, tool) > memsom.RANK["external"]
