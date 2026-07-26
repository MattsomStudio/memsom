"""What a run is allowed to touch — hosts and paths, checked per tool call.

Every containment rule this homelab runs on has lived in a human's head and in a
Claude memory file: the home LAN is read-only, the NAS is off limits, real
offense goes on the sandbox network. The runtime knew none of it, and that
mattered less when a run was one agent doing one thing a person had just read.
It matters now: an agent can pick its own next hop and branch in parallel, so
the set of targets a run reaches is no longer something anybody approved in
advance.

**The capability this bounds is narrower than it looks, and naming it precisely
is what keeps this from being theatre.** The user can already reach every one of
these targets from a terminal, and gating a capability somebody already has is
the definition of a pointless wall. What is NEW is *a target chosen by
attacker-influenced text with no human in the loop* — the agent fetched a page,
the page argued, and the next call went somewhere nobody picked. That is the
thing this module exists for.

Three properties, in the order they matter:

* **Default open.** A run that declares no scope behaves exactly as it did
  before this module existed. Scope is opt-in per graph; nothing silently
  tightens under anyone.
* **A seatbelt, not a wall.** `_SEATBELT` is denied even with no declaration —
  but an explicit `hosts` entry that matches the target WAIVES it for that
  target. The dangerous default is safe and the user can unbuckle it on purpose,
  in writing, per target. That distinction is the whole design: a wall gets
  worked around and then resented, a seatbelt gets worn.
* **Honest about `shell`.** `shell` can reach anything a target rule forbids by
  spelling it `curl`. It is in `_TARGETS` as an explicit UNSCOPED entry rather
  than an omission, the run log names it at start, and nothing here pretends
  otherwise. Its bound is the approval gate it already defaults to.
"""

from __future__ import annotations

import ipaddress  # noqa: F401  (tests reach through this module for it)
import urllib.parse
from pathlib import Path

from memsom.providers.net import addrs

#: Denied with no declaration, and *only* with no matching declaration.
#:
#: Loopback is the sharp one and it is not about the model servers being
#: precious: the panel's own HTTP API is on loopback, so an agent that can POST
#: there can start runs, resume paused ones and approve its own gates. There is
#: no legitimate agent reason to reach it, and privilege escalation into your own
#: control plane is not a class of bug worth leaving open for convenience.
#: Link-local is the cloud-metadata SSRF address; same reasoning, no legitimate
#: use.
#:
#: The home LAN is deliberately NOT here. The standing rule permits read-only
#: GETs on it and `http_fetch` IS a read-only GET, so a blanket deny would break
#: permitted use and teach everyone to switch the seatbelt off wholesale — which
#: is how a guardrail becomes theatre. The NAS is not here either: it belongs on
#: the list and its address is the operator's to supply, not this module's to
#: guess.
#:
#: The list itself now lives in `net/addrs.py`, because the CONNECTOR has to
#: enforce it too and two copies of a deny list is how one of them goes stale.
#: This module decides whether a call is in scope; `net` decides what a socket is
#: allowed to reach. Same rules, two questions.
_SEATBELT = addrs.DENIED

#: Marks a tool whose reach a target rule cannot bound. Not an omission — the
#: whole point of writing it down is that `test_every_builtin_tool_type_is_in_
#: the_scope_table` fails when a new tool is added without a decision.
UNSCOPED = "unscoped"

#: builtin tool type -> (argument name, kind) | None (no target) | UNSCOPED.
#:
#: Central rather than a field on each Tool: there are eight builtins and three
#: have targets, so one table is one place to audit rather than eight places to
#: check. It is exhaustive over BUILTIN_TOOLS by test.
_TARGETS = {
    "http_fetch": ("url", "url"),
    "file_read": ("path", "path"),
    # A fixed host (html.duckduckgo.com) the model cannot influence — the query
    # is attacker-reachable, the DESTINATION is not, and destinations are what
    # this module bounds.
    "web_search": None,
    "memory_recall": None,
    "recall": None,
    "state_set": None,
    "state_get": None,
    # Reaches anything by spelling it `curl`. Bounded by its approval gate
    # (`require_approval` defaults True for shell), never by a target rule.
    "shell": UNSCOPED,
}


def unscoped_tools(tool_types) -> list:
    """Which of *tool_types* no scope rule can bound. For the run log."""
    return sorted({t for t in tool_types if _TARGETS.get(t) == UNSCOPED})


#: Address parsing and matching live in `net/addrs.py` — same functions, one
#: implementation, shared with the connector that does the actual enforcing.
#: `as_ip` there is strictly stronger than the version this module used to carry:
#: it also decodes `2130706433`, `0x7f000001`, `0177.0.0.1` and `127.1`, which
#: used to fall through as if they were hostnames.
_as_ip = addrs.as_ip
_entry_matches = addrs.entry_matches
_seatbelt_hit = addrs.seatbelt_hit


def _resolved_ips(host: str) -> list:
    """Every address *host* currently resolves to. Empty on any failure.

    Why resolve at all: without it a string-only host check is defeated by
    pointing `evil.com` at 127.0.0.1, which would make the seatbelt decorative
    against anyone actually trying.

    **This is the advisory half, and it is no longer the load-bearing one.** The
    guarantee now lives at the connector (`net/connect.py`), which vets the
    addresses it is about to dial — so the rebinding window this docstring used
    to apologise for is closed where it matters, by re-checking rather than by
    hoping. What remains here is a fast, early "this call is out of scope" answer
    for the model, and it stays deliberately empty-on-failure: a DNS blip must
    not be reported to the model as a forbidden target.

    It shares the connector's resolver, which buys two things. The obvious one is
    a **bounded** lookup: `socket.getaddrinfo` has no timeout parameter and sits
    outside every run budget — measured on this box 2026-07-25, a single
    `getaddrinfo("anywhere.test")` took **11.4 seconds** and blocked the agent
    thread for all of it. The subtler one is that this check and the connection
    now consult the same cache, so the advisory answer and the enforced one
    rarely disagree.
    """
    try:
        from memsom.providers.net import connect as _net
        return list(_net.shared_resolver().resolve(host, deadline_s=2.0))
    except Exception:
        # Deliberately broad and deliberately silent. Anything at all going
        # wrong here means "no opinion", never "refused" — and never an
        # exception into the tool-dispatch path.
        return []


def _check_host(scope: dict, url: str) -> str:
    hosts = [h for h in (scope.get("hosts") or []) if str(h).strip()]
    split = urllib.parse.urlsplit(url)
    host = split.hostname or ""
    if not host:
        return f"unparseable target {url!r}"
    ips = _resolved_ips(host) if _as_ip(host) is None else []

    declared = any(_entry_matches(str(e), host, ips) for e in hosts)
    if not declared:
        # The seatbelt applies only to what was NOT explicitly named. Naming a
        # target is how you take responsibility for it.
        hit = _seatbelt_hit(host, ips)
        if hit is not None:
            return (f"{host} is in {hit} — refused by default. Name it in the "
                    f"trigger's scope.hosts to allow it deliberately.")
    if hosts and not declared:
        return f"{host} is outside this run's declared scope"
    return ""


def _check_path(scope: dict, raw: str) -> str:
    roots = [r for r in (scope.get("paths") or []) if str(r).strip()]
    if not roots:
        return ""
    try:
        target = Path(raw).resolve()
    except (OSError, ValueError):
        return f"unresolvable path {raw!r}"
    for root in roots:
        try:
            if target.is_relative_to(Path(str(root)).resolve()):
                return ""
        except (OSError, ValueError):
            continue
    return f"{target} is outside this run's declared scope"


def check(scope: dict, tool_type: str, arguments: dict) -> str:
    """Why this call is refused, or ``""`` to allow it.

    A string return rather than an exception because the caller feeds it back to
    the MODEL: an out-of-scope call is a mistake the model can correct on its
    next turn, not a reason to destroy a run that may have banked real work.
    """
    target = _TARGETS.get(tool_type)
    if target is None or target == UNSCOPED:
        return ""
    arg_name, kind = target
    raw = str(arguments.get(arg_name) or "").strip()
    if not raw:
        return ""
    scope = scope or {}
    if kind == "url":
        return _check_host(scope, raw)
    return _check_path(scope, raw)
