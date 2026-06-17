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

# VRAM hygiene knob.  By DEFAULT memdag does NOT touch keep_alive — it defers
# to Ollama's own native behaviour (model stays warm ~5min), which is what most
# users want.  On a shared/small-VRAM card you can force the model to unload
# after every call by setting MEMDAG_OLLAMA_KEEP_ALIVE=0 (or any Ollama duration
# string, e.g. "10m", to hold it warm longer).  Unset = let Ollama decide.

# Regex for stripping <think>...</think> blocks (qwen3 thinking residue)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Regex to match claimed citation tags in the LLM response. Captures BOTH the
# source id AND the channel half so the channel can be validated against the real
# source (LLM-2) — the old r"\[mem:(\d+)\|" discarded the channel entirely.
_CITE_RE = re.compile(r"\[mem:(\d+)\|([^\]]+)\]")


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

    Returns None when MEMDAG_OLLAMA_KEEP_ALIVE is unset/blank — callers then
    OMIT keep_alive from the request body so Ollama applies its own native
    default (model stays warm). This keeps the shipped default behaviour normal.

    When the env var is set: a numeric string is returned as int (the JSON
    number 0 tells Ollama to unload immediately after the call); anything else
    (e.g. "10m", "1h") is passed through verbatim as an Ollama duration string.
    """
    raw = os.environ.get("MEMDAG_OLLAMA_KEEP_ALIVE")
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def _with_keep_alive(body):
    """Stamp keep_alive into *body* only when the env knob is set.

    Mutates and returns *body*. Unset knob -> body untouched -> Ollama's
    native keep_alive default applies. One place, used by every call site.
    """
    ka = keep_alive()
    if ka is not None:
        body["keep_alive"] = ka
    return body


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

    payload = json.dumps(_with_keep_alive(
        {"model": m, "prompt": prompt, "stream": False}
    )).encode("utf-8")
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

    # Step 5: citation firewall.
    # Authoritative {source_id: real_channel} map, restricted to the deterministic
    # source set. The LLM may cite ONLY these ids, and ONLY with the correct channel.
    real_channel = {
        sid: channel
        for sid, _content, channel, _label, _ref in sources
        if sid in det_used
    }

    # LLM-2: capture id AND channel; validate both against the real source.
    cited = _CITE_RE.findall(answer)   # list of (id_str, channel_str)
    if not cited:
        raise LlmUnavailable(
            "LLM output failed citation check: no [mem:id|channel] citations found in answer"
        )
    cited_ids = set()
    for id_str, chan in cited:
        sid = int(id_str)
        if sid not in real_channel:
            raise LlmUnavailable(
                f"LLM output failed citation check: invented citation {sid}"
            )
        if chan != real_channel[sid]:
            # Forged integrity provenance: e.g. tagging a real external source as
            # [mem:5|endorsed]. The channel half is content the LLM controls.
            raise LlmUnavailable(
                f"LLM output failed citation check: forged channel for [mem:{sid}] "
                f"(claimed {chan!r}, real {real_channel[sid]!r})"
            )
        cited_ids.add(sid)

    # LLM-1: EVERY non-empty line must END with a VALIDATED citation tag and carry
    # NO text after it. A substring-anywhere test ("[mem:" in line, or even
    # any(tag in line)) let uncited content ride on a validly-cited line — trailing
    # sentences after a real tag, or a fake/garbage "[mem:" token on a line whose
    # only real citation lived elsewhere. Anchoring the tag to end-of-line closes
    # both: the deterministic source bullets are one-claim-per-line ending in their
    # citation, so a well-formed answer line always ends with an allowed tag.
    # SCOPE (honest, documented limit): this guarantees every line carries a valid
    # provenance citation and the node's PARENTS are anchored to det_used (LLM-3),
    # so the DAG label/provenance is correct regardless of the prose. It does NOT
    # verify the rephrased claim text faithfully summarises the cited source — a
    # structural citation firewall cannot, and the moat is provenance, not LLM
    # truthfulness. Semantic faithfulness is out of scope for this gate.
    allowed_tags = {f"[mem:{sid}|{real_channel[sid]}]" for sid in cited_ids}
    for line in answer.splitlines():
        stripped = line.rstrip()
        if stripped.strip() and not any(stripped.endswith(tag) for tag in allowed_tags):
            raise LlmUnavailable(
                f"LLM output failed citation check: line not anchored to a citation: {line!r}"
            )

    # LLM-3: anchor the derived node's provenance to the DETERMINISTIC source set,
    # NOT the LLM-chosen citation subset. Otherwise the model could paraphrase a
    # low-integrity source and cite only a high-integrity one, dropping the low
    # parent so derive_node's label=min(parents) launders the label upward.
    used = sorted(det_used)

    # Step 6: return
    text = (
        f"Q: {question}\n"
        f"A (LLM-composed, opt-in, from {len(used)} live sources):\n"
        f"{answer}"
    )
    return text, used


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
