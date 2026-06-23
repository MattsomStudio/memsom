"""rag adapter — vanilla vector RAG, the 'no defense at all' sanity floor.

nomic embeddings (same Ollama endpoint memdag uses) into an in-memory store,
cosine top-k, deterministic concatenation as the 'answer'. There is NO provenance
and NO integrity label -- so poison competes purely on similarity, and whatever
lands in top-k shapes the answer. This is the reference point: it shows where a
system with zero integrity machinery sits on the poison axis.

Self-contained (urllib + pure-python cosine) so it needs no extra deps.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request

from runner import AskResult, Citation
from adapters.base import MemoryAdapter

_EMBED_URL = os.environ.get("MEMDAG_EMBED_URL") or "http://localhost:11434/api/embeddings"
_EMBED_MODEL = os.environ.get("MEMDAG_EMBED_MODEL") or "nomic-embed-text"


def _embed(text: str, is_query: bool) -> list:
    # nomic's intended task prefixes; matches how memdag embeds, so RAG is a fair
    # retriever rather than a strawman.
    prefix = "search_query: " if is_query else "search_document: "
    body = json.dumps({"model": _EMBED_MODEL, "prompt": prefix + text}).encode("utf-8")
    req = urllib.request.Request(_EMBED_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["embedding"]


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class RagAdapter(MemoryAdapter):
    name = "rag"
    has_provenance = False

    def __init__(self, **_):
        self._mem = []  # list of {id, text, channel, vec}

    def reset(self, item_dir: str) -> None:
        self._mem = []

    def add(self, text: str, channel: str, answer_bearing: bool = False) -> None:
        self._mem.append({"id": len(self._mem) + 1, "text": text,
                          "channel": channel, "vec": _embed(text, is_query=False)})

    def ask(self, question: str, topk: int = 8) -> AskResult:
        if not self._mem:
            return AskResult(composed=False, answer_text="")
        qv = _embed(question, is_query=True)
        ranked = sorted(self._mem, key=lambda m: _cosine(qv, m["vec"]), reverse=True)[:topk]
        cites = [Citation(node_id=m["id"], channel=m["channel"], text=m["text"]) for m in ranked]
        return AskResult(
            composed=bool(cites),
            answer_text=" ".join(m["text"] for m in ranked),
            citations=cites,
            used=len(cites), considered=len(self._mem),
            composed_node_id=None,
            integrity=None,   # no provenance -> laundering reported n/a
        )
