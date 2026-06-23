"""memdag adapter — drives memdag through its CLI in-process.

Reuses the validated call sequence from run_bench_par (init/add/reindex/ask via
memdag_cli.main under stdout capture) and runner.parse_ask. Adds the integrity
GATE via memdag's existing `check-action --require user` -- the action-boundary
gate that is memdag's actual integrity-enforcement mechanism (the pool itself
does not pre-filter by integrity; see plan recon).
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil

from runner import AskResult, parse_ask
from adapters.base import MemoryAdapter


class MemdagAdapter(MemoryAdapter):
    name = "memdag"
    has_provenance = True

    def __init__(self, repo: str, no_embed: bool = False):
        import sys
        sys.path.insert(0, repo)
        import memdag_cli  # noqa: E402
        self._cli = memdag_cli
        self.repo = repo
        self._item_dir = None
        # memdag-bm25 ablation: force the embedder offline so reindex/retrieve
        # degrade to pure keyword/BM25 -- proves the integrity property is
        # invariant to the retrieval backend, and that memdag runs on ZERO models.
        self.no_embed = no_embed
        self.llm_tokens = 0  # deterministic -> never consumes generative LLM tokens

    def _call(self, argv) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                self._cli.main(argv)
            except SystemExit:
                pass
            except Exception as e:  # noqa: BLE001 - contain per-item blowups
                print(f"[memdag-adapter] {type(e).__name__}: {e}")
        return buf.getvalue()

    def reset(self, item_dir: str) -> None:
        shutil.rmtree(item_dir, ignore_errors=True)
        if self.no_embed:
            # dead endpoint -> _call_ollama_embed raises -> BM25-only fallback.
            os.environ["MEMDAG_EMBED_URL"] = "http://127.0.0.1:1/dead"
        else:
            os.environ.pop("MEMDAG_EMBED_URL", None)
        self._call(["init", "--data-dir", item_dir])  # init ignores MEMDAG_DB
        os.environ["MEMDAG_DB"] = os.path.join(item_dir, "memdag.db")
        self._item_dir = item_dir

    def add(self, text: str, channel: str, answer_bearing: bool = False) -> None:
        # `add` injects a source node on the given channel. Integrity floor is
        # computed from the node's CHANNEL, so poison on 'external' drags the
        # composed floor regardless of which ingest primitive created it.
        self._call(["add", text, "--channel", channel])

    def ask(self, question: str, topk: int = 8) -> AskResult:
        self._call(["reindex"])
        out = self._call(["ask", question, "--retrieve", "--topk", str(topk)])
        return parse_ask(out)

    def gate(self, node_id: int | None, require: str = "user") -> bool:
        # No composed answer -> nothing to act on -> denied.
        if node_id is None:
            return False
        out = self._call(["check-action", str(node_id), "--require", require]).lower()
        # robust parse: explicit deny wins; otherwise allow if it says so.
        if "deny" in out or "denied" in out or "block" in out:
            return False
        return "allow" in out or "ok" in out or "pass" in out
