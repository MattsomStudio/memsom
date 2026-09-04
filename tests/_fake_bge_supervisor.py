"""A stdlib stand-in for bge_supervisor.py, for tests only (no torch, no numpy).

Speaks exactly the wire contract memsom.retrieval.bge_client expects:
  GET  /health            -> 200 {"ok": true, ...}   (never "loads" anything)
  POST /embed {"input"}   -> 200 {"dense": [[...]], "sparse": [{...}],
                                  "colbert_b64": [b64 fp16 LE], "colbert_shape": [[n, d]]}
  POST /quit              -> 200 and the server stops
It records the last /embed request body (so a test can assert `idle_ttl` rode
along) and, when run as a script, appends one line to --count-file at startup
(so a test can count how many times memsom's cold start actually spawned it).

Usable two ways:
  * in-process: `srv, thread = serve_in_thread(port, ...)`; `srv.shutdown()`
  * as a spawn_cmd: `python tests/_fake_bge_supervisor.py --port P
    [--count-file F] [--record-file R] [--ttl 30] [--delay 0.0]`
"""
import argparse
import base64
import json
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DENSE = [1.0, 0.0, 0.0, 0.0]
SPARSE = {"7": 0.5, "9": 0.25}
COLBERT = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


def embed_payload(dense=None, sparse=None, colbert=None) -> dict:
    dense = list(DENSE if dense is None else dense)
    sparse = dict(SPARSE if sparse is None else sparse)
    colbert = [list(r) for r in (COLBERT if colbert is None else colbert)]
    flat = [float(x) for row in colbert for x in row]
    b64 = base64.b64encode(struct.pack(f"<{len(flat)}e", *flat)).decode("ascii")
    shape = [len(colbert), len(colbert[0]) if colbert else 0]
    return {"dense": [dense], "sparse": [sparse], "colbert_b64": [b64],
            "colbert_shape": [shape]}


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                state["health_calls"] = state.get("health_calls", 0) + 1
                self._send(200, {"ok": True, "model": "bge-m3", "supervisor": True,
                                 "backend_running": False, "loaded": False,
                                 "backend_exit_after": state.get("idle_ttl")})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.rstrip("/")
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            if path == "/quit":
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if path != "/embed":
                self._send(404, {"error": "not found"})
                return
            try:
                payload = json.loads(raw or b"{}")
            except ValueError:
                self._send(400, {"error": "bad json"})
                return
            state["last_body"] = payload
            state["embed_calls"] = state.get("embed_calls", 0) + 1
            if "idle_ttl" in payload:
                state["idle_ttl"] = payload["idle_ttl"]
            rec = state.get("record_file")
            if rec:
                with open(rec, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            if payload.get("probe"):
                self._send(200, {"idle": True, "loaded": False})
                return
            if state.get("fail_embed"):
                self._send(500, {"error": "forced failure"})
                return
            self._send(200, embed_payload(state.get("dense")))

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve_in_thread(port: int, host: str = "127.0.0.1", **state):
    """Start the fake on a daemon thread; returns (server, state). Stop with
    server.shutdown(); server.server_close()."""
    st = dict(state)
    srv = _Server((host, port), make_handler(st))
    th = threading.Thread(target=srv.serve_forever, name="fake-bge", daemon=True)
    th.start()
    return srv, st


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--count-file")
    ap.add_argument("--record-file")
    ap.add_argument("--ttl", type=float, default=30.0, help="exit after this many seconds")
    ap.add_argument("--delay", type=float, default=0.0, help="sleep before binding")
    args = ap.parse_args(argv)
    if args.delay:
        time.sleep(args.delay)
    if args.count_file:
        with open(args.count_file, "a", encoding="utf-8") as f:
            f.write(f"started {time.time():.3f}\n")
    srv, _st = serve_in_thread(args.port, args.host, record_file=args.record_file)
    deadline = time.monotonic() + args.ttl
    try:
        while time.monotonic() < deadline:
            time.sleep(0.2)
    finally:
        srv.shutdown()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
