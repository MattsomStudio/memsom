"""Graph documents — the saved agent canvases.

One JSON file per graph under ``<agents_dir>/graphs/``, atomic tmp+replace
writes (the panel's knob-file discipline), ids fenced by the same regex the
session files use. Concurrency is last-write-wins with a server-side ``rev``
counter — single user, one canvas open at a time; CAS would be ceremony.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from memsom.providers.base import ProviderError, now

_GRAPH_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_NODE_TYPES = {"engine", "agent", "tool", "trigger", "output"}


class GraphStore:
    def __init__(self, graphs_dir) -> None:
        self.dir = Path(graphs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, graph_id: str) -> Path:
        return self.dir / f"{graph_id}.json"

    def list(self) -> list:
        out = []
        for p in sorted(self.dir.glob("*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            schedule = None
            for n in doc.get("nodes", []):
                if isinstance(n, dict) and n.get("type") == "trigger":
                    cfg = n.get("config") or {}
                    if cfg.get("mode") == "schedule":
                        schedule = {"enabled": bool(cfg.get("enabled")),
                                    **(cfg.get("schedule") or {})}
                    break
            out.append({"id": doc.get("id", p.stem),
                        "name": doc.get("name") or p.stem,
                        "rev": doc.get("rev", 0),
                        "updated": doc.get("updated"),
                        "schedule": schedule})
        return out

    def get(self, graph_id: str) -> dict:
        if not _GRAPH_ID_RE.match(graph_id or ""):
            raise ProviderError("invalid graph id")
        p = self._path(graph_id)
        if not p.is_file():
            raise ProviderError(f"unknown graph: {graph_id!r}")
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError(f"unreadable graph {graph_id!r}: {exc}") from exc

    def save(self, graph: dict) -> tuple:
        """Upsert; returns (id, rev). Validates shape, assigns id on first
        save, bumps rev server-side."""
        if not isinstance(graph, dict):
            raise ProviderError("graph must be a JSON object")
        gid = graph.get("id") or uuid.uuid4().hex
        if not _GRAPH_ID_RE.match(str(gid)):
            raise ProviderError("invalid graph id")
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ProviderError("graph needs 'nodes' and 'edges' lists")
        for n in nodes:
            if not isinstance(n, dict) or not n.get("id"):
                raise ProviderError("every node needs an id")
            if n.get("type") not in _NODE_TYPES:
                raise ProviderError(f"unknown node type: {n.get('type')!r}")
        node_ids = {n["id"] for n in nodes}
        for e in edges:
            if not isinstance(e, dict):
                raise ProviderError("edges must be objects")
            if e.get("source") not in node_ids or e.get("target") not in node_ids:
                raise ProviderError("edge references a missing node")

        prev_rev = 0
        p = self._path(gid)
        if p.is_file():
            try:
                prev_rev = json.loads(
                    p.read_text(encoding="utf-8")).get("rev", 0)
            except (OSError, json.JSONDecodeError):
                prev_rev = 0
        doc = {**graph, "id": gid, "rev": prev_rev + 1, "updated": now()}
        doc.setdefault("created", doc["updated"])
        self._atomic_write(p, doc)
        return gid, doc["rev"]

    def delete(self, graph_id: str) -> None:
        if not _GRAPH_ID_RE.match(graph_id or ""):
            raise ProviderError("invalid graph id")
        p = self._path(graph_id)
        if not p.is_file():
            raise ProviderError(f"unknown graph: {graph_id!r}")
        p.unlink()

    @staticmethod
    def _atomic_write(path: Path, doc: dict) -> None:
        tmp = path.with_suffix(f".tmp-{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
