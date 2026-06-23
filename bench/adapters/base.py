"""base — the adapter interface every memory system implements.

Adapters return runner.AskResult (not a parallel type) so score.score_item grades
every system with identical logic. AskResult fields the scorer reads:
  composed, answer_text, citations[Citation(node_id, channel, text)],
  integrity (composed label or None), composed_node_id.
"""

from __future__ import annotations

from runner import AskResult


class MemoryAdapter:
    name = "base"
    # systems with deterministic provenance/integrity labels (memdag). False for
    # RAG/Mem0/Zep -> laundering is reported n/a for them, and gate() is a no-op.
    has_provenance = False
    # cumulative generative-LLM tokens consumed by writes (for the tokens/write
    # column). 0 for deterministic/embedding-only systems; Mem0/Zep increment it.
    llm_tokens = 0

    def reset(self, item_dir: str) -> None:
        """Start a fresh, isolated store for one item."""
        raise NotImplementedError

    def add(self, text: str, channel: str, answer_bearing: bool = False) -> None:
        """Ingest one memory. `channel` is the provenance origin
        (user=evidence, external=poison); systems w/o provenance still record it
        so citations can echo it back for uniform scoring."""
        raise NotImplementedError

    def ask(self, question: str, topk: int = 8) -> AskResult:
        """Retrieve + compose an answer."""
        raise NotImplementedError

    def gate(self, node_id: int | None, require: str = "user") -> bool:
        """Action-time integrity gate. Default: allow (no gate). memdag overrides
        with check-action. Returns True if the composed answer clears `require`."""
        return True
