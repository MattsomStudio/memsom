"""memsom adapter — drives memsom through its CLI in-process.

Reuses the validated call sequence from run_bench_par (init/add/reindex/ask via
memsom_cli.main under stdout capture) and runner.parse_ask. Adds the integrity
GATE via memsom's existing `check-action --require user` -- the action-boundary
gate that is memsom's actual integrity-enforcement mechanism (the pool itself
does not pre-filter by integrity; see plan recon).
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil

from runner import AskResult, parse_ask
from adapters.base import MemoryAdapter

_ADD_ID_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)  # cmd_add prints "[<nid>] ..."


class MemsomAdapter(MemoryAdapter):
    name = "memsom"
    has_provenance = True

    def __init__(self, repo: str, no_embed: bool = False, fresh_only: bool = False,
                 prefer_fresh: bool = False, graph: bool = False, hops: int = 2):
        import sys
        sys.path.insert(0, repo)
        import memsom_cli  # noqa: E402
        self._cli = memsom_cli
        self.repo = repo
        self._item_dir = None
        # graph arm: after all adds, a BLIND LLM linker (bench_linker) proposes
        # associative edges among the item's evidence, which are materialised as
        # 'wikilink' rel_edges so `ask --graph` can re-rank over them. graph=False
        # is the retrieve baseline; the two arms differ ONLY by this flag. hops is
        # the graph expansion depth for retrieve_graph.
        self.graph = graph
        self.hops = hops
        self._texts: list[str] = []       # evidence texts, in add() order
        self._node_ids: list[int | None] = []  # parallel node ids from cmd_add
        # fresh_only: ask with --fresh-only so stale sources are EXCLUDED (blunt).
        # prefer_fresh: ask with --prefer-fresh so stale sources are SUBSTITUTED
        # with their current version via the supersedes edge (the hybrid — the
        # arm that should hold fresh_clean under the haystack where exclude fails).
        self.fresh_only = fresh_only
        self.prefer_fresh = prefer_fresh
        # memsom-bm25 ablation: force the embedder offline so reindex/retrieve
        # degrade to pure keyword/BM25 -- proves the integrity property is
        # invariant to the retrieval backend, and that memsom runs on ZERO models.
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
                print(f"[memsom-adapter] {type(e).__name__}: {e}")
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
        self._texts = []
        self._node_ids = []

    def add(self, text: str, channel: str, answer_bearing: bool = False) -> None:
        # `add` injects a source node on the given channel. Integrity floor is
        # computed from the node's CHANNEL, so poison on 'external' drags the
        # composed floor regardless of which ingest primitive created it.
        #
        # NOTE: answer_bearing is intentionally NOT forwarded anywhere near the
        # linker — the graph arm must stay gold-blind. We only remember the text
        # and the node id cmd_add prints, so a later blind pass can relate them.
        out = self._call(["add", text, "--channel", channel])
        m = _ADD_ID_RE.search(out)
        self._texts.append(text)
        self._node_ids.append(int(m.group(1)) if m else None)

    def _inject_blind_edges(self) -> None:
        """Graph arm only: run the blind linker over this item's evidence and
        materialise its proposed pairs as 'wikilink' rel_edges.

        The linker sees ONLY self._texts (no question, no answer_bearing). Edges
        are written straight through memsom_relate against the item DB that
        MEMDAG_DB points at — the same edge kind the real bridge/obsidian paths
        create, so `ask --graph` treats them identically.
        """
        import bench_linker
        import memsom
        import memsom_relate

        pairs = bench_linker.link(self._texts)   # raises if Ollama is down (loud, by design)
        if not pairs:
            return
        conn = memsom.get_connection()            # reads MEMDAG_DB set in reset()
        try:
            memsom_relate.migrate(conn)
            for i, j in pairs:
                a, b = self._node_ids[i], self._node_ids[j]
                if a and b and a != b:
                    memsom_relate.relate(conn, a, b, kind="wikilink")
        finally:
            conn.close()

    def ask(self, question: str, topk: int = 8) -> AskResult:
        self._call(["reindex"])
        if self.graph:
            self._inject_blind_edges()
            argv = ["ask", question, "--graph", "--hops", str(self.hops),
                    "--topk", str(topk)]
        else:
            argv = ["ask", question, "--retrieve", "--topk", str(topk)]
        if self.fresh_only:
            argv.append("--fresh-only")
        if self.prefer_fresh:
            argv.append("--prefer-fresh")
        out = self._call(argv)
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

    # -- staleness harness overrides ------------------------------------------

    def seed(self, text: str, channel: str, source_ref: str | None = None) -> None:
        # ingest-text (not `add`) so content_hash + source_ref are stamped — the
        # signal the auto-detect supersession trigger keys on.
        argv = ["ingest-text", text, "--channel", channel]
        if source_ref:
            argv += ["--ref", source_ref]
        self._call(argv)

    def update(self, source_ref: str, text: str, channel: str) -> None:
        # Re-ingest the SAME source_ref with changed content -> memsom records the
        # supersession and fires the staleness cascade over the derived answer.
        self._call(["ingest-text", text, "--channel", channel, "--ref", source_ref])

    def stale_attribution(self, node_id: int | None) -> bool | None:
        # Ask the shipping CLI whether the prior answer node is now stale — the
        # exact "which answers does the changed source affect?" query.
        if node_id is None:
            return None
        out = self._call(["stale-status", str(node_id)])
        return ": STALE" in out  # CLI prints "node [N]: STALE" vs "node [N]: fresh"
