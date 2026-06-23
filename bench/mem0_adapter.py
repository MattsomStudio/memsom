"""mem0 adapter — Mem0 on a LOCAL stack (qwen3 extraction + nomic + in-mem qdrant).

Fairness choices:
- Mem0's value-add is LLM-driven memory MANAGEMENT (extract facts, ADD/UPDATE/
  DELETE), so we let it do that (`infer=True`) -- that's the real Mem0, and also
  its real cost + attack surface.
- For ANSWERING we compose deterministically (concat retrieved memories), exactly
  like memdag/rag, so the comparison isn't muddied by who has the better answer-
  writer. The variable under test is "does the poison memory survive into what's
  retrieved", not prose quality.
- Provenance: Mem0 has none. We stash the origin channel in each memory's metadata
  and echo it back in citations, so the same channel-based citation_asr works.

Per-item isolation: a fresh Memory() with an in-memory qdrant per reset().

Token accounting: best-effort. We wrap the Ollama client's chat() to sum Ollama's
prompt_eval_count + eval_count (Ollama returns real token counts). If the wrap
point moves in a future mem0, llm_tokens stays 0 and latency-per-write still
carries the cost contrast.
"""

from __future__ import annotations

import os

from runner import AskResult, Citation
from adapters.base import MemoryAdapter

# qwen2.5:7b-instruct: a clean NON-thinking instruct model. qwen3 emits
# <think>...</think> before its JSON, which breaks Mem0's fact-extractor (utility
# 0) and would unfairly nerf Mem0. Mem0's defaults assume a clean JSON model
# (gpt-4o-mini-class); this is the fair local equivalent. ~5GB, fits 12GB VRAM.
# Override via --config '{"llm_model": "..."}'.
_DEFAULT_LLM = "qwen2.5:7b-instruct"
_DEFAULT_EMB = "nomic-embed-text"
_OLLAMA = "http://localhost:11434"
_USER = "bench"


class Mem0Adapter(MemoryAdapter):
    name = "mem0"
    has_provenance = False

    def __init__(self, llm_model: str = _DEFAULT_LLM, embed_model: str = _DEFAULT_EMB,
                 ollama_url: str = _OLLAMA, **_):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.ollama_url = ollama_url
        self._m = None
        self._item_dir = None
        self.llm_tokens = 0
        self.write_failures = 0

    def _config(self) -> dict:
        # per-item on-disk qdrant under the item dir -> unique path per item, so
        # sequential Memory() instances in one worker never collide on a locked
        # folder (mem0 defaults qdrant to a shared /tmp/qdrant otherwise).
        qpath = os.path.join(self._item_dir, "qdrant")
        return {
            "llm": {"provider": "ollama", "config": {
                "model": self.llm_model, "ollama_base_url": self.ollama_url,
                "temperature": 0}},
            "embedder": {"provider": "ollama", "config": {
                "model": self.embed_model, "ollama_base_url": self.ollama_url}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "bench", "embedding_model_dims": 768,
                "path": qpath, "on_disk": True}},
        }

    def _wrap_tokens(self):
        """Sum Ollama token counts across Mem0's LLM calls (best-effort)."""
        try:
            client = self._m.llm.client            # ollama.Client
            orig = client.chat
            def counting_chat(*a, **k):
                resp = orig(*a, **k)
                try:
                    self.llm_tokens += int(getattr(resp, "prompt_eval_count", 0) or 0)
                    self.llm_tokens += int(getattr(resp, "eval_count", 0) or 0)
                except Exception:
                    pass
                return resp
            client.chat = counting_chat
        except Exception:
            pass  # wrap point moved; latency-per-write still shows the cost

    def reset(self, item_dir: str) -> None:
        import shutil
        from mem0 import Memory
        self._m = None  # drop old client so it releases its qdrant lock
        shutil.rmtree(item_dir, ignore_errors=True)
        os.makedirs(item_dir, exist_ok=True)
        self._item_dir = item_dir
        self._m = Memory.from_config(self._config())
        self.llm_tokens = 0
        self._wrap_tokens()

    def add(self, text: str, channel: str, answer_bearing: bool = False) -> None:
        # infer=True -> Mem0's real LLM extraction/merge path (its feature + cost).
        # Mem0 (2.0.6) can throw on a malformed extraction shape from a local LLM
        # ('str' has no attribute 'get'); contain it per-write so one bad item
        # doesn't kill the run. Skips are counted + surfaced (no silent loss).
        try:
            self._m.add(text, user_id=_USER, metadata={"channel": channel}, infer=True)
        except Exception as e:  # noqa: BLE001
            self.write_failures += 1
            print(f"[mem0-adapter] write skipped ({type(e).__name__}: {e})")

    def ask(self, question: str, topk: int = 8) -> AskResult:
        res = self._m.search(question, filters={"user_id": _USER}, limit=topk)
        results = res.get("results", res) if isinstance(res, dict) else res
        if not results:
            return AskResult(composed=False, answer_text="")
        cites, texts = [], []
        for i, r in enumerate(results, 1):
            mem = r.get("memory") or r.get("text") or ""
            chan = (r.get("metadata") or {}).get("channel")
            texts.append(mem)
            cites.append(Citation(node_id=i, channel=chan, text=mem))
        return AskResult(
            composed=bool(cites), answer_text=" ".join(texts), citations=cites,
            used=len(cites), considered=len(cites),
            composed_node_id=None, integrity=None,
        )
