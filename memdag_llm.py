"""memdag_llm — opt-in LLM answer path for memdag.

The DEFAULT ask path is 100% deterministic (invariant 7 in memdag.py).
This module ONLY runs when the caller opts in via an explicit --llm flag.
It talks to a local Ollama instance over stdlib urllib and enforces that the
LLM can NEVER add an uncited claim: any violation raises LlmUnavailable, which
callers should catch and fall back to the deterministic memdag.compose().

No DB access anywhere in this module — it operates on source rows passed in.
No DB opened at import time.
"""

import json
import os
import re
import urllib.error
import urllib.request

import memdag
import memdag_schema  # noqa: F401  — imported for uniformity (migrate uses it)

DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3-abliterated:30b-a3b"

# VRAM hygiene: Ollama's default keep_alive (~5min) leaves the model resident
# in VRAM after each call.  On a shared 12GB card that squats memory and can
# evict the operator's daily-driver model.  memdag defaults to "0" = unload
# immediately after every call; set MEMDAG_OLLAMA_KEEP_ALIVE (e.g. "5m") to
# keep the model warm between calls instead.
DEFAULT_KEEP_ALIVE = "0"

# Regex for stripping <think>...</think> blocks (qwen3 thinking residue)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Regex to match claimed citation tags in the LLM response
_CITE_RE = re.compile(r"\[mem:(\d+)\|")

# A "claim line" is any bullet-list line (starts with - or * after optional whitespace)
_CLAIM_LINE_RE = re.compile(r"^\s*[-*]\s+")


class LlmUnavailable(Exception):
    """Raised for both Ollama-unreachable and citation-firewall failures.

    Deliberate simplification: one exception type means callers have exactly
    one fallback path — catch LlmUnavailable -> memdag.compose().
    """


def migrate(conn):
    """No schema changes needed for this module. Stub for uniformity."""
    pass


def resolve(model=None, base_url=None):
    """Resolve model name and base URL with env-var fallback.

    Resolution order (first non-None wins):
      1. Explicit argument
      2. Environment variable (MEMDAG_LLM_MODEL / MEMDAG_LLM_URL)
      3. Module-level defaults

    Returns (model, base_url) as a 2-tuple of strings.
    """
    m = model or os.environ.get("MEMDAG_LLM_MODEL") or DEFAULT_MODEL
    url = base_url or os.environ.get("MEMDAG_LLM_URL") or DEFAULT_URL
    return m, url


def keep_alive():
    """Resolve the Ollama ``keep_alive`` request value (shared helper).

    Every memdag call to the Ollama API (/api/generate, /api/embeddings)
    stamps this into the request body so the model is unloaded from VRAM
    immediately after each call by default (value 0).

    Resolution: MEMDAG_OLLAMA_KEEP_ALIVE env var, else DEFAULT_KEEP_ALIVE
    ("0").  Numeric strings are returned as int (Ollama treats the JSON
    number 0 as unload-now); anything else (e.g. "5m", "1h") is passed
    through verbatim as an Ollama duration string.
    """
    raw = os.environ.get("MEMDAG_OLLAMA_KEEP_ALIVE")
    if raw is None or not raw.strip():
        raw = DEFAULT_KEEP_ALIVE
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def llm_compose(question, sources, model=None, base_url=None, timeout=60):
    """Compose an answer using the local Ollama LLM, with a citation firewall.

    Parameters
    ----------
    question : str
        The question to answer.
    sources : list of (id, content, channel, label, source_ref) tuples
        Live source rows — same format as memdag.live_sources() returns.
    model : str or None
        Ollama model name; falls back via resolve().
    base_url : str or None
        Ollama API URL; falls back via resolve().
    timeout : int
        HTTP timeout in seconds (default 60).

    Returns
    -------
    (text, used) where text is the formatted answer string and used is a
    sorted list of int node IDs that the LLM actually cited.

    Raises
    ------
    ValueError
        When sources is empty / no live source yields any claim.  This is NOT
        LlmUnavailable because there is nothing to fall back to either — the
        deterministic path would also refuse.
    LlmUnavailable
        When Ollama is unreachable, replies malformed, or the LLM output fails
        the citation post-check (uncited claim, invented citation, etc.).
    """
    # Step 1: run the deterministic compose first — it gives us the cited bullets
    # and the set of valid source IDs.
    det_text, det_used = memdag.compose(question, sources)
    if det_text is None:
        raise ValueError("no live source yielded any claim")

    m, base = resolve(model, base_url)

    # Valid citation tags that the LLM is *allowed* to use.
    valid_citations = [f"mem:{sid}|{channel}" for sid, _content, channel, _label, _ref in sources
                       if sid in det_used]

    # Build the deterministic bullets block — exactly the lines from det_text
    # that start with "- " (everything after the "A (composed..." header line).
    det_lines = det_text.splitlines()
    # Skip "Q: ..." and "A (composed...)" header; collect bullet lines.
    bullet_lines = [l for l in det_lines if l.startswith("- ")]
    bullets_block = "\n".join(bullet_lines)

    valid_str = ", ".join(f"[{c}]" for c in valid_citations)
    prompt = (
        "Answer the question using ONLY the source bullets below. "
        "Every bullet in your answer MUST end with its [mem:id|channel] citation "
        "copied VERBATIM from the source bullet it came from. "
        "Do not add any claim that lacks a citation. "
        "Do not invent citations. "
        f"Valid citations: {valid_str}.\n\n"
        f"{bullets_block}\n\n"
        f"Question: {question}"
    )

    payload = json.dumps({"model": m, "prompt": prompt, "stream": False,
                          "keep_alive": keep_alive()}).encode("utf-8")
    req = urllib.request.Request(base, data=payload,
                                 headers={"Content-Type": "application/json"})

    # Step 3: call Ollama
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        data = json.loads(raw)
        answer_raw = data["response"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError,
            json.JSONDecodeError, KeyError) as err:
        raise LlmUnavailable(f"ollama unreachable or malformed reply: {err}") from err

    # Step 4: strip <think>...</think> residue
    answer = _THINK_RE.sub("", answer_raw).strip()

    # Step 5: citation firewall
    cited_strs = _CITE_RE.findall(answer)
    if not cited_strs:
        raise LlmUnavailable(
            "LLM output failed citation check: no [mem:id|channel] citations found in answer"
        )

    cited_ids = {int(s) for s in cited_strs}
    valid_ids = set(det_used)

    # Invented citation check: cited an ID not in the deterministic source set
    invented = cited_ids - valid_ids
    if invented:
        raise LlmUnavailable(
            f"LLM output failed citation check: invented citation(s) {sorted(invented)}"
        )

    # Uncited claim line check: any bullet line that has no [mem: tag
    for line in answer.splitlines():
        if _CLAIM_LINE_RE.match(line) and "[mem:" not in line:
            raise LlmUnavailable(
                f"LLM output failed citation check: uncited claim line: {line!r}"
            )

    # Step 6: return
    text = (
        f"Q: {question}\n"
        f"A (LLM-composed, opt-in, from {len(cited_ids)} live sources):\n"
        f"{answer}"
    )
    return text, sorted(cited_ids)


def ping(base_url=None, timeout=5):
    """Return True if Ollama is reachable (HTTP 200 on /api/tags), False otherwise."""
    _, url = resolve(base_url=base_url)
    # Swap /api/generate for /api/tags
    tags_url = url.replace("/api/generate", "/api/tags")
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def register(subparsers):
    """Register the 'llm-check' subcommand on *subparsers*.

    Only prints reachability; never generates an answer.
    """
    p = subparsers.add_parser("llm-check",
                               help="check whether local Ollama is reachable")
    p.set_defaults(func=_cmd_llm_check)


def _cmd_llm_check(args):
    """CLI handler: print Ollama reachability and exit 1 if not reachable."""
    import sys
    m, url = resolve()
    if ping(base_url=url):
        print(f"ollama reachable at {url} (model: {m})")
    else:
        print(f"ollama NOT reachable at {url}")
        sys.exit(1)
